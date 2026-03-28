# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import textwrap
from collections import defaultdict
import time
from typing import Any, Callable, Optional, Union
import random
import numpy as np

import torch
import torch.utils.data
import torch.nn.functional as F
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from qwen_vl_utils import process_vision_info

import copy

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using[`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using[`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not[`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of[`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by[`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: GRPOConfig = None,
            script_args=None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
                    None, None),
            peft_config: Optional["PeftConfig"] = None,
            max_pixels: Optional[int] = 12845056,
            min_pixels: Optional[int] = 3136,
            attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id or True:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temporal = script_args.temporal
        self.len_control = script_args.len_control

        try:
            self.force_max_prompt_length = int(os.getenv("FORCE_MAX_PROMPT_LEN", "16384"))
        except Exception:
            self.force_max_prompt_length = 16384

        self.video_logprob_skip_len = getattr(script_args, "video_logprob_skip_len", 16384)

        self.ema_lambda = 0.99
        self.batch_pass_rate_ema = 0.60
        self.batch_statistics = {
            'correct_chain_lengths': [],
            'all_chain_lengths': [],
            'group_pass_rates': [],
        }

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.shuffled_num_generations = self.num_generations // 2
        self.shuffled_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,  # HACK
            num_return_sequences=self.shuffled_num_generations,
            pad_token_id=pad_token_id,
        )

        self.dummy_generation_config = GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            top_p=0.95,
            temperature=1,  # HACK
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )
        self.len_control = script_args.len_control
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Auto-init Weights & Biases on the main process if available and not already started
        if is_wandb_available():
            try:
                if self.accelerator.is_main_process and getattr(wandb, "run", None) is None:
                    run_name = getattr(args, "run_name", None) or os.path.basename(self.args.output_dir.rstrip("/\\"))
                    wandb.init(project=os.environ.get("WANDB_PROJECT", "video-r1"), name=run_name,
                               config=self.args.to_dict())
            except Exception as wandb_init_err:
                print(f"[wandb] init failed: {wandb_init_err}")

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, **kwargs):
        logits = model(input_ids, **kwargs).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def remove_none_from_data(self, data):
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data

    def _log_batch_pass_rate_ema(self, label: str, value: Optional[float]) -> None:
        accelerator = getattr(self, "accelerator", None)
        if accelerator is None or not accelerator.is_main_process:
            return
        step = getattr(getattr(self, "state", None), "global_step", "N/A")
        print(f"📈 [EMA] step={step} | {label}: {value}")  # noqa: T201

    def compute_effort_reward(self, accuracy_rewards, completion_lengths, temporal_reward_value, device):
        B_times_G = len(accuracy_rewards)
        B = B_times_G // self.num_generations
        G = self.num_generations

        accuracy_reshaped = accuracy_rewards.view(B, G)
        group_pass_rates = accuracy_reshaped.mean(dim=1)
        current_batch_avg = group_pass_rates.mean().item()
        self._log_batch_pass_rate_ema("before_update", self.batch_pass_rate_ema)
        if self.batch_pass_rate_ema is None:
            self.batch_pass_rate_ema = current_batch_avg
        else:
            self.batch_pass_rate_ema = (
                    self.ema_lambda * self.batch_pass_rate_ema
                    + (1 - self.ema_lambda) * current_batch_avg
            )
        self._log_batch_pass_rate_ema("after_update", self.batch_pass_rate_ema)

        Pb_t_tensor = torch.full((B_times_G,), self.batch_pass_rate_ema, device=device)
        repeated_group_rates = group_pass_rates.repeat_interleave(G, dim=0)
        delta_diff = repeated_group_rates - Pb_t_tensor

        R_corr = repeated_group_rates
        correct_mask = accuracy_rewards > 0.1
        correct_lengths = completion_lengths[correct_mask]

        T_AVG_FLOOR = 300.0

        if correct_lengths.numel() > 0:
            raw_t_avg = 2.5 * correct_lengths.float().mean().item()
            T_avg_val = max(raw_t_avg, T_AVG_FLOOR)
        else:
            T_avg_val = max(200.0, T_AVG_FLOOR)

        L_pct = torch.quantile(completion_lengths.float(), 0.75)
        denom = max(T_avg_val, 0.5 * L_pct.item())
        z_calib = completion_lengths.float() / (denom + 1e-6)

        P_diff = Pb_t_tensor - 0.70
        k1 = -(3.0 + 0.5 * (R_corr - 0.70).clamp(min=0) + 0.3 * P_diff)
        k2 = -(1.2 + 0.2 * (R_corr - 0.70).clamp(min=0) + 0.1 * P_diff)
        k3 = 1.2 + 0.2 * (0.70 - R_corr) + 0.1 * (0.70 - Pb_t_tensor)
        k4 = 3.0 + 0.2 * (0.70 - R_corr) + 0.3 * (0.70 - Pb_t_tensor)

        mask_extreme_hard = Pb_t_tensor < 0.65
        mask_hard = (Pb_t_tensor >= 0.65) & (Pb_t_tensor < 0.70)
        mask_easy = (Pb_t_tensor >= 0.70) & (Pb_t_tensor < 0.75)
        mask_extreme_easy = Pb_t_tensor >= 0.75

        effort_rewards = torch.zeros_like(accuracy_rewards)
        clip_val = torch.tensor(0.95, device=device)

        if mask_extreme_hard.any():
            base = torch.sigmoid(k4 * z_calib)
            amp = 1.0 + delta_diff.abs()
            effort_rewards[mask_extreme_hard] = 0.3 * torch.minimum(clip_val, base * amp)[mask_extreme_hard]

        if mask_hard.any():
            base = torch.sigmoid(k3 * z_calib)
            amp = 1.0 + 0.5 * delta_diff.abs()
            effort_rewards[mask_hard] = 0.25 * torch.minimum(clip_val, base * amp)[mask_hard]

        if mask_easy.any():
            base = torch.sigmoid(k2 * z_calib)
            amp = 1.0 + 0.2 * delta_diff.clamp(min=0.0)
            effort_rewards[mask_easy] = 0.2 * torch.minimum(clip_val, base * amp)[mask_easy]

        if mask_extreme_easy.any():
            base = torch.sigmoid(k1 * z_calib)
            amp = 1.0 + 0.5 * delta_diff.clamp(min=0.0)
            effort_rewards[mask_extreme_easy] = 0.2 * torch.minimum(clip_val, base * amp)[mask_extreme_easy]

        effort_rewards = effort_rewards * accuracy_rewards

        return effort_rewards, group_pass_rates, z_calib

    # The experimental `compute_effort_reward` helper has been removed to
    # avoid confusion and to keep the trainer consistent with the official
    # length-reward implementation. If you need a custom effort-based reward
    # reintroduce it here with a clear name and ensure it's exercised by the
    # reward aggregation logic.

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial)

        if self.is_world_process_zero():
            step = self.state.global_step
            checkpoint_folder = f"checkpoint-{step}"
            output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

            os.makedirs(output_dir, exist_ok=True)

            custom_state = {
                "batch_pass_rate_ema": self.batch_pass_rate_ema
            }
            save_path = os.path.join(output_dir, "custom_ema_state.pt")
            torch.save(custom_state, save_path)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)

        custom_state_path = os.path.join(resume_from_checkpoint, "custom_ema_state.pt")

        if os.path.exists(custom_state_path):
            try:
                custom_state = torch.load(custom_state_path, map_location="cpu")
                self.batch_pass_rate_ema = custom_state.get("batch_pass_rate_ema", None)
                print(f"✅ [Checkpoint] Successfully loaded batch_pass_rate_ema: {self.batch_pass_rate_ema}")
            except Exception as e:
                print(f"⚠️ [Checkpoint] Failed to load custom state: {e}")
        else:
            print(
                f"ℹ️[Checkpoint] No custom_ema_state.pt found at {resume_from_checkpoint}. EMA difficulty will reset.")

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        if os.getenv("DEBUG_MODE") == "true":
            print(f"\n{'=' * 80}")
            print(f"🔍 DEBUG: Batch info - Rank {self.accelerator.process_index}")
            print(f"   Number of inputs: {len(inputs)}")
            print(f"   Sample data_type: {inputs[0].get('data_type', 'MISSING')}")
            print(f"   Sample problem_id: {inputs[0].get('problem_id', 'MISSING')}")
            print(f"   Sample path: {inputs[0].get('path', 'MISSING')}")
            print(f"{'=' * 80}\n")

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        input_copy = copy.deepcopy(inputs[0]['prompt'])

        input_copy = self.remove_none_from_data(input_copy)

        if inputs[0]['data_type'] == 'image':
            rel_path = inputs[0]['path'].lstrip('./')
            input_copy[0]['content'][0]['image'] = os.getcwd() + "/Video-R1-data/" + rel_path
        elif inputs[0]['data_type'] == 'video':
            rel_path = inputs[0]['path'].lstrip('./')
            video_path = os.getcwd() + "/Video-R1-data/" + rel_path
            input_copy[0]['content'][0]['video'] = video_path
            if not os.path.exists(video_path):
                print(f"⚠️ Video file NOT FOUND: {video_path}")
                print(f"   Original path from JSON: {inputs[0]['path']}")
                print(f"   Current working dir: {os.getcwd()}")
                print(f"   After lstrip: {rel_path}")

        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
            if os.getenv("DEBUG_MODE") == "true":
                print(f"📊 process_vision_info SUCCESS:")
                print(f"   image_inputs: {type(image_inputs)}, len={len(image_inputs) if image_inputs else 0}")
                print(f"   video_inputs: {type(video_inputs)}, len={len(video_inputs) if video_inputs else 0}")
                if video_inputs and len(video_inputs) > 0:
                    print(f"   video_inputs[0].shape: {video_inputs[0].shape}")
                    print(f"   video_inputs[0].numel(): {video_inputs[0].numel()}")
        except Exception as e:
            print(f"⚠️ process_vision_info FAILED with error: {e}")
            print(f"   Traceback: {type(e).__name__}")
            if inputs[0]['data_type'] == 'image':
                input_copy[0]['content'][0][
                    'image'] = os.getcwd() + "/Video-R1-data" + '/Math/Multimath-300k/17ff4c7d14c388134de02381b1fc2824.png'
            elif inputs[0]['data_type'] == 'video':
                input_copy[0]['content'][0][
                    'video'] = os.getcwd() + "/Video-R1-data" + '/LLaVA-Video-178K/liwei_youtube_videos/videos/youtube_video_2024/ytb_7nRmsEw7nsE.mp4'

            image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
            print(f"📊 process_vision_info RETRY SUCCESS after fallback")

        if os.getenv("DEBUG_MODE") == "true":
            print(f"📊 BEFORE processing_class call:")
            print(f"   video_inputs is None?: {video_inputs is None}")
        if video_inputs:
            print(f"   video_inputs len: {len(video_inputs)}")
            if len(video_inputs) > 0:
                print(f"   video_inputs[0] numel: {video_inputs[0].numel()}")

        if inputs[0]['data_type'] == 'video' and video_inputs:
            if len(video_inputs) > 0 and video_inputs[0].numel() == 0:
                print(
                    f"⚠️ Empty video tensor detected for sample {inputs[0].get('problem_id', 'unknown')}, skipping this batch")
            prompts = [x["prompt"] for x in inputs if "prompt" in x]

        prompt_inputs = self.processing_class(
            text=copy.deepcopy(prompts_text),
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )

        if os.getenv("DEBUG_MODE") == "true":
            print(f"📊 processing_class OUTPUT:")
            print(f"   Keys: {list(prompt_inputs.keys())}")
            if 'video_grid_thw' in prompt_inputs:
                vgt = prompt_inputs['video_grid_thw']
                print(f"   video_grid_thw type: {type(vgt)}")
                print(f"   video_grid_thw shape: {vgt.shape}")
                print(f"   video_grid_thw numel: {vgt.numel()}")
                print(f"   video_grid_thw values: {vgt}")
                print(f"   video_grid_thw sum: {vgt.sum()}")
                print(f"   video_grid_thw all zeros?: {(vgt == 0).all()}")
            if 'pixel_values_videos' in prompt_inputs:
                pvv = prompt_inputs['pixel_values_videos']
                print(f"   pixel_values_videos shape: {pvv.shape}")
                print(f"   pixel_values_videos dim: {pvv.dim()}")
            if 'pixel_values' in prompt_inputs:
                pv = prompt_inputs['pixel_values']
                print(f"   pixel_values shape: {pv.shape}")
                print(f"   pixel_values dim: {pv.dim()}")

        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if inputs[0]['data_type'] == 'video':
            if "video_grid_thw" not in prompt_inputs:
                print(
                    f"⚠️ Missing video_grid_thw for sample {inputs[0].get('problem_id', 'unknown')}, path: {inputs[0].get('path', 'N/A')}")
                return torch.zeros((), device=self.accelerator.device, dtype=torch.bfloat16, requires_grad=True)

            vgt = prompt_inputs["video_grid_thw"]
            if vgt.numel() == 0 or len(vgt) == 0:
                print(f"⚠️ Empty video_grid_thw tensor for sample {inputs[0].get('problem_id', 'unknown')}")
                print(f"   Video path: {input_copy[0]['content'][0].get('video', 'N/A')}")
                return torch.zeros((), device=self.accelerator.device, dtype=torch.bfloat16, requires_grad=True)

            if (vgt == 0).all() or vgt.sum() == 0:
                print(f"⚠️ video_grid_thw contains all zeros for sample {inputs[0].get('problem_id', 'unknown')}")
                print(f"   video_grid_thw shape: {vgt.shape}, values: {vgt}")
                print(f"   Video path: {input_copy[0]['content'][0].get('video', 'N/A')}")
                return torch.zeros((), device=self.accelerator.device, dtype=torch.bfloat16, requires_grad=True)

        if self.max_prompt_length is not None:
            if inputs[0].get('data_type') != 'video':
                prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
                prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]
            else:
                if os.getenv("DEBUG_MODE") == "true":
                    print(
                        f"[DEBUG] Skip prompt truncation for video sample to keep all video placeholder tokens. input_ids_len={prompt_inputs['input_ids'].size(1)} max_prompt_length={self.max_prompt_length}")

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None and inputs[0].get('data_type') != 'video':
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
        elif self.max_prompt_length is not None and inputs[0].get('data_type') == 'video' and os.getenv(
                "DEBUG_MODE") == "true":
            print(
                f"[DEBUG] Keeping full prompt (video) after potential truncation stage. prompt_len={prompt_ids.size(1)}")

        try:
            _plen = prompt_ids.size(1)
        except Exception:
            _plen = None
        try:
            import torch.distributed as dist
            local_flag = 1 if (_plen is not None and _plen > self.force_max_prompt_length) else 0
            flag_tensor = torch.tensor([local_flag], device=self.accelerator.device)
            local_len_val = int(_plen) if _plen is not None else 0
            maxlen_tensor = torch.tensor([local_len_val], device=self.accelerator.device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(flag_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(maxlen_tensor, op=dist.ReduceOp.MAX)
                _global_skip = flag_tensor.item() > 0
                _global_max_len = int(maxlen_tensor.item())
            else:
                _global_skip = bool(local_flag)
                _global_max_len = local_len_val
        except Exception:
            _global_skip = bool(_plen is not None and _plen > self.force_max_prompt_length)
            _global_max_len = int(_plen) if _plen is not None else 0
        if _global_skip:
            if self.accelerator.is_main_process:
                print(
                    f"⚠️ Early-skip before shuffle: global_max_len={_global_max_len} > {self.force_max_prompt_length} (local_len={_plen})")
            try:
                if _plen is not None and _plen > self.force_max_prompt_length:
                    self._metrics["skipped_long_prompts"].append(float(_plen))
            except Exception:
                pass
            self.accelerator.wait_for_everyone()
            dummy_loss = torch.tensor(0.0, device=self.accelerator.device, dtype=torch.float32, requires_grad=True)
            return dummy_loss

        if self.temporal and video_inputs:
            indices = torch.randperm(video_inputs[0].size(0))
            shuffled_video_inputs = [video_inputs[0][indices]]
            shuffled_prompt_inputs = self.processing_class(
                text=copy.deepcopy(prompts_text),
                images=image_inputs,
                videos=shuffled_video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            shuffled_prompt_inputs = super()._prepare_inputs(shuffled_prompt_inputs)

            if "video_grid_thw" not in shuffled_prompt_inputs:
                print(
                    f"⚠️ Missing video_grid_thw in shuffled_prompt_inputs for sample {inputs[0].get('problem_id', 'unknown')}")
                return torch.zeros((), device=self.accelerator.device, dtype=torch.bfloat16, requires_grad=True)

            svgt = shuffled_prompt_inputs["video_grid_thw"]
            if svgt.numel() == 0 or len(svgt) == 0:
                print(
                    f"⚠️ Empty video_grid_thw tensor in shuffled_prompt_inputs for sample {inputs[0].get('problem_id', 'unknown')}")
                return torch.zeros((), device=self.accelerator.device, dtype=torch.bfloat16, requires_grad=True)

            if (svgt == 0).all() or svgt.sum() == 0:
                print(
                    f"⚠️ shuffled video_grid_thw contains all zeros for sample {inputs[0].get('problem_id', 'unknown')}")
                print(f"   video_grid_thw shape: {svgt.shape}, values: {svgt}")
                return torch.zeros((), device=self.accelerator.device, dtype=torch.bfloat16, requires_grad=True)

            shuffled_prompt_ids, shuffled_prompt_mask = shuffled_prompt_inputs["input_ids"], shuffled_prompt_inputs[
                "attention_mask"]
            if self.max_prompt_length is not None:
                shuffled_prompt_ids = shuffled_prompt_ids[:, -self.max_prompt_length:]
                shuffled_prompt_mask = shuffled_prompt_mask[:, -self.max_prompt_length:]

        original_prompt_inputs_full = {}
        for key in prompt_inputs.keys():
            if isinstance(prompt_inputs[key], torch.Tensor):
                original_prompt_inputs_full[key] = prompt_inputs[key].clone()
            else:
                original_prompt_inputs_full[key] = prompt_inputs[key]

        try:
            prompt_len_for_skip = prompt_ids.size(1)
        except Exception:
            prompt_len_for_skip = None
        local_skip_generation = bool(
            prompt_len_for_skip is not None and prompt_len_for_skip > self.force_max_prompt_length)
        if local_skip_generation:
            print(
                f"⚠️ Local soft-skip generation on Rank {self.accelerator.process_index}: prompt_len={prompt_len_for_skip} > {self.force_max_prompt_length}")
            try:
                if prompt_len_for_skip is not None and prompt_len_for_skip > self.force_max_prompt_length:
                    self._metrics["skipped_long_prompts"].append(float(prompt_len_for_skip))
            except Exception:
                pass

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            num_generations = self.generation_config.num_return_sequences
            if local_skip_generation:
                temp_generation_config = copy.deepcopy(self.dummy_generation_config)
            else:
                temp_generation_config = copy.deepcopy(self.generation_config)
                temp_generation_config.num_return_sequences = 4

            all_completions = []

            print(f"[Rank {self.accelerator.process_index}] Starting generation loop: {num_generations} iterations")

            for i in range(num_generations // 4):
                print(f"[Rank {self.accelerator.process_index}] Generation {i + 1}/{num_generations} starting...")

                current_prompt_inputs = {}
                for key in original_prompt_inputs_full.keys():
                    if isinstance(original_prompt_inputs_full[key], torch.Tensor):
                        current_prompt_inputs[key] = original_prompt_inputs_full[key].clone()
                    else:
                        current_prompt_inputs[key] = original_prompt_inputs_full[key]

                if 'second_per_grid_ts' in current_prompt_inputs:
                    del current_prompt_inputs['second_per_grid_ts']

                try:
                    completion = unwrapped_model.generate(**current_prompt_inputs,
                                                          generation_config=temp_generation_config)
                    all_completions.append(completion)
                    print(
                        f"[Rank {self.accelerator.process_index}] Generation {i + 1}/{num_generations} completed. Output shape: {completion.shape}")
                except Exception as e:
                    print(f"[Rank {self.accelerator.process_index}] ❌ Generation {i + 1}/{num_generations} FAILED: {e}")
                    raise

            max_length = max(completion.size(1) for completion in all_completions)
            padded_completions = []

            for completion in all_completions:
                if completion.size(1) < max_length:
                    padding = torch.full(
                        (completion.size(0), max_length - completion.size(1)),
                        self.processing_class.tokenizer.pad_token_id,
                        dtype=completion.dtype,
                        device=completion.device,
                    )
                    padded_completion = torch.cat([completion, padding], dim=1)
                else:
                    padded_completion = completion
                padded_completions.append(padded_completion)

            prompt_completion_ids = torch.cat(padded_completions, dim=0)

            try:
                del all_completions
                del padded_completions
            except Exception:
                pass

            torch.cuda.empty_cache()

            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

            if self.temporal and not local_skip_generation:

                if video_inputs:

                    shuffled_prompt_completion_ids = unwrapped_model.generate(**shuffled_prompt_inputs,
                                                                              generation_config=self.shuffled_generation_config)
                    shuffled_prompt_length = shuffled_prompt_ids.size(1)
                    shuffled_prompt_ids = shuffled_prompt_completion_ids[:, :shuffled_prompt_length]
                    shuffled_completion_ids = shuffled_prompt_completion_ids[:, shuffled_prompt_length:]
                    shuffled_prompt_mask = prompt_mask.repeat_interleave(self.shuffled_num_generations, dim=0)

                else:

                    shuffled_prompt_completion_ids = unwrapped_model.generate(**prompt_inputs,
                                                                              generation_config=self.dummy_generation_config)

            torch.cuda.empty_cache()

        print('path:', input_copy[0]['content'][0][inputs[0]['data_type']])
        print('problem_id:', inputs[0]['problem_id'])
        print('prompt_length:', prompt_length)

        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        logits_inputs = {}
        num_completions = len(prompt_completion_ids)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        local_skip_logprob = bool(inputs[0].get('data_type') == 'video' and prompt_length > self.video_logprob_skip_len)
        if local_skip_logprob:
            print(
                f"⚠️ Local-skip logprob on Rank {self.accelerator.process_index}: prompt_len={prompt_length} > {self.video_logprob_skip_len}")

        if inputs[0]['data_type'] == 'image':
            logits_inputs["pixel_values"] = original_prompt_inputs_full["pixel_values"].repeat(num_completions, 1)
            logits_inputs["image_grid_thw"] = original_prompt_inputs_full["image_grid_thw"].repeat(num_completions, 1)
        elif inputs[0]['data_type'] == 'video':
            logits_inputs["pixel_values_videos"] = original_prompt_inputs_full["pixel_values_videos"].repeat(
                num_completions, 1)
            logits_inputs["video_grid_thw"] = original_prompt_inputs_full["video_grid_thw"].repeat(num_completions, 1)

        try:
            if local_skip_logprob:
                per_token_logps = torch.zeros((num_completions, completion_mask.size(1)),
                                              device=self.accelerator.device, dtype=torch.bfloat16, requires_grad=True)
            else:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()
                print(
                    f"[Rank {self.accelerator.process_index}] Logprob(model) start: bs={num_completions}, seq={prompt_completion_ids.size(1)}")
                per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **logits_inputs)
                per_token_logps = per_token_logps[:, prompt_length - 1:]
                print(
                    f"[Rank {self.accelerator.process_index}] 🔍 Model logps: min={per_token_logps.min().item():.6f}, max={per_token_logps.max().item():.6f}, mean={per_token_logps.mean().item():.6f}, requires_grad={per_token_logps.requires_grad}")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = time.time() - t0
                try:
                    mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                    mem_rsrv = torch.cuda.memory_reserved() / (1024 ** 3)
                    print(
                        f"[Rank {self.accelerator.process_index}] Logprob(model) done in {dt:.2f}s | mem {mem_alloc:.2f}G/{mem_rsrv:.2f}G")
                except Exception:
                    print(f"[Rank {self.accelerator.process_index}] Logprob(model) done in {dt:.2f}s")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error computing per_token_logps: {e}. Retrying with zero fallback.")
            per_token_logps = torch.zeros((num_completions, completion_mask.size(1)), device=self.accelerator.device,
                                          dtype=torch.bfloat16, requires_grad=True)

        with torch.inference_mode():
            try:
                if local_skip_logprob:
                    ref_per_token_logps = torch.zeros((num_completions, completion_mask.size(1)),
                                                      device=self.accelerator.device, dtype=torch.bfloat16)
                else:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    t1 = time.time()
                    print(
                        f"[Rank {self.accelerator.process_index}] Logprob(ref) start: bs={num_completions}, seq={prompt_completion_ids.size(1)}")
                    if self.ref_model is not None:
                        ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids,
                                                                        **logits_inputs)
                    else:
                        with self.accelerator.unwrap_model(model).disable_adapter():
                            ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids,
                                                                            **logits_inputs)
                    ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
                    print(
                        f"[Rank {self.accelerator.process_index}] 🔍 Ref logps: min={ref_per_token_logps.min().item():.6f}, max={ref_per_token_logps.max().item():.6f}, mean={ref_per_token_logps.mean().item():.6f}")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    dt1 = time.time() - t1
                    try:
                        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                        mem_rsrv = torch.cuda.memory_reserved() / (1024 ** 3)
                        print(
                            f"[Rank {self.accelerator.process_index}] Logprob(ref) done in {dt1:.2f}s | mem {mem_alloc:.2f}G/{mem_rsrv:.2f}G")
                    except Exception:
                        print(f"[Rank {self.accelerator.process_index}] Logprob(ref) done in {dt1:.2f}s")
            except Exception as e:
                print(f"Error computing ref_per_token_logps: {e}. Using zero fallback.")
                ref_per_token_logps = torch.zeros((num_completions, completion_mask.size(1)),
                                                  device=self.accelerator.device, dtype=torch.bfloat16)

        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1

        shuffled_rewards_per_func = None
        if self.temporal and video_inputs and not local_skip_generation:
            shuffled_completions = self.processing_class.batch_decode(shuffled_completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                shuffled_completions = [[{"role": "assistant", "content": shuffled_completion}] for shuffled_completion
                                        in shuffled_completions]

            shuffled_prompts = [prompt for prompt in prompts for _ in range(self.shuffled_num_generations)]
            shuffled_rewards_per_func = torch.zeros(len(shuffled_prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                    zip(self.reward_funcs, self.reward_processing_classes)
            ):
                shuffled_reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in shuffled_reward_kwargs:
                    for example in inputs:
                        shuffled_reward_kwargs[key].extend([example[key]] * self.shuffled_num_generations)
                shuffled_output_reward_func = reward_func(prompts=shuffled_prompts, completions=shuffled_completions,
                                                          **shuffled_reward_kwargs)
                shuffled_rewards_per_func[:, i] = torch.tensor(shuffled_output_reward_func, dtype=torch.float32,
                                                               device=device)

        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
        ):
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        if self.temporal and video_inputs and not local_skip_generation and shuffled_rewards_per_func is not None:
            temporal_rewards_per_func = rewards_per_func.clone()

            acc_mean = temporal_rewards_per_func[:, 0].mean()
            shuffled_acc_mean = shuffled_rewards_per_func[:, 0].mean()

            if acc_mean >= 0.8 * shuffled_acc_mean:
                mask = temporal_rewards_per_func[:, 0] > 0.1
                temporal_rewards_per_func[mask, 0] = temporal_rewards_per_func[mask, 0] + 0.3
                temporal_rewards = torch.tensor([1.0]).to('cuda')
            else:
                temporal_rewards = torch.tensor([0.0]).to('cuda')
        else:
            temporal_rewards = torch.tensor([0.0]).to('cuda')

        temporal_rewards_list = self.accelerator.gather_for_metrics(temporal_rewards)

        R_acc = rewards_per_func[:, 0] if rewards_per_func.size(1) >= 1 else torch.zeros(rewards_per_func.size(0),
                                                                                         device=device)
        R_format = rewards_per_func[:, 1] if rewards_per_func.size(1) >= 2 else torch.zeros_like(R_acc)

        if self.temporal and video_inputs and not local_skip_generation and shuffled_rewards_per_func is not None:
            if temporal_rewards_per_func is not None and temporal_rewards_per_func.size(1) >= 1:
                R_acc = temporal_rewards_per_func[:, 0]
            if temporal_rewards_per_func is not None and temporal_rewards_per_func.size(1) >= 2:
                R_format = temporal_rewards_per_func[:, 1]

        rewards = R_acc + R_format

        effort_reward_mean = None
        current_pg_mean = None
        z_mean = None
        if self.len_control:
            effort_rewards, group_pass_rates_tensor, z_calib_tensor = self.compute_effort_reward(
                accuracy_rewards=rewards_per_func[:, 0],
                completion_lengths=completion_mask.sum(1),
                temporal_reward_value=temporal_rewards.item(),
                device=device,
            )
            rewards = rewards + effort_rewards
            try:
                effort_reward_mean = self.accelerator.gather_for_metrics(effort_rewards).mean().item()

                gathered_pg = self.accelerator.gather_for_metrics(group_pass_rates_tensor)
                current_pg_mean = gathered_pg.mean().item()
                self._metrics["competence/pg_group_pass_rate"].append(current_pg_mean)

                gathered_z = self.accelerator.gather_for_metrics(z_calib_tensor)
                z_mean = gathered_z.mean().item()
                self._metrics.setdefault("debug/z_calib_mean", []).append(z_mean)
            except Exception:
                pass

            if self.accelerator.is_main_process:
                raw_accuracy = rewards_per_func[:, 0].detach().float()
                raw_lengths = completion_mask.sum(dim=1).detach().float()
                z_vals = z_calib_tensor.detach().float()
                pg_expanded = group_pass_rates_tensor.repeat_interleave(self.num_generations, dim=0).detach().float()

                log_entry = []
                for i in range(len(raw_accuracy)):
                    log_entry.append({
                        "step": int(self.state.global_step),
                        "pg": pg_expanded[i].item(),
                        "acc": raw_accuracy[i].item(),
                        "len": raw_lengths[i].item(),
                        "z": z_vals[i].item(),
                        "problem_id": inputs[i // self.num_generations].get("problem_id", "unknown"),
                    })

                analysis_file = os.path.join(self.args.output_dir, "analysis_logs.jsonl")
                try:
                    os.makedirs(self.args.output_dir, exist_ok=True)
                    with open(analysis_file, "a") as f:
                        for entry in log_entry:
                            f.write(json.dumps(entry) + "\n")
                except Exception as e:
                    print(f"⚠️ Failed to write analysis log: {e}")

        print(rewards)
        print(completion_mask.sum(1))

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        print(
            f"[Rank {self.accelerator.process_index}] per_token_logps stats: min={per_token_logps.min().item():.6f}, max={per_token_logps.max().item():.6f}, mean={per_token_logps.mean().item():.6f}")
        print(
            f"[Rank {self.accelerator.process_index}] ref_per_token_logps stats: min={ref_per_token_logps.min().item():.6f}, max={ref_per_token_logps.max().item():.6f}, mean={ref_per_token_logps.mean().item():.6f}")
        print(
            f"[Rank {self.accelerator.process_index}] advantages stats: min={advantages.min().item():.6f}, max={advantages.max().item():.6f}, mean={advantages.mean().item():.6f}")

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        print(
            f"[Rank {self.accelerator.process_index}] loss value: {loss.item():.6f}, requires_grad={loss.requires_grad}")

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        gathered_rewards = self.accelerator.gather_for_metrics(rewards)

        num_devices = gathered_rewards.size(0) // self.num_generations
        rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
        wrong_devices = (rewards_per_device <= 1).all(dim=1)
        wrong_ratio = wrong_devices.sum().item() / num_devices

        correct_devices = (rewards_per_device >= 2).all(dim=1)
        correct_ratio = correct_devices.sum().item() / num_devices

        self._metrics["all_wrong"].append(wrong_ratio)
        self._metrics["all_correct"].append(correct_ratio)

        if self.batch_pass_rate_ema is not None:
            self._metrics["competence/pass_rate_ema"].append(self.batch_pass_rate_ema)

        if temporal_rewards_list is not None:
            self._metrics["temporal_rewards"].append(temporal_rewards_list.mean().item())
        if effort_reward_mean is not None:
            self._metrics.setdefault("effort_reward", []).append(effort_reward_mean)

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        mean_kl_value = self.accelerator.gather_for_metrics(mean_kl).mean().item()
        self._metrics["kl"].append(mean_kl_value)

        reward_mean = self.accelerator.gather_for_metrics(rewards).mean().item()
        reward_std_mean = self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item()

        if self.accelerator.is_main_process:
            print("=" * 80)
            print("📊[STEP METRICS - Current Step Only]")
            print(f"   completion_length: {completion_length:.4f}")

            for i, reward_func in enumerate(self.reward_funcs):
                if isinstance(reward_func, PreTrainedModel):
                    reward_func_name = reward_func.config._name_or_path.split("/")[-1]
                else:
                    reward_func_name = reward_func.__name__
                print(f"   rewards/{reward_func_name}: {reward_per_func[i].item():.4f}")

            print(f"   all_wrong: {wrong_ratio:.4f}")
            print(f"   all_correct: {correct_ratio:.4f}")
            if current_pg_mean is not None:
                print(f"   Pg (current_batch): {current_pg_mean:.4f}")
            if self.batch_pass_rate_ema is not None:
                print(f"   Pb (historical_ema): {self.batch_pass_rate_ema:.4f}")

            if temporal_rewards_list is not None:
                temporal_reward_mean = temporal_rewards_list.mean().item()
                print(f"   temporal_rewards: {temporal_reward_mean:.4f}")

            print(f"   reward: {reward_mean:.4f}")
            print(f"   reward_std: {reward_std_mean:.4f}")
            print(f"   kl: {mean_kl_value:.6f}")
            print(f"   loss: {loss.item():.6f}")
            print("=" * 80)

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if is_wandb_available() and getattr(wandb, "run", None) is not None and self.accelerator.is_main_process:
            try:
                wandb.log(logs, step=self.state.global_step)
            except Exception as wandb_log_err:
                print(f"[wandb] log failed: {wandb_log_err}")
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
            self,
            model_name: Optional[str] = None,
            dataset_name: Optional[str] = None,
            tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))