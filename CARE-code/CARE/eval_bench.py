import os
import json
import re
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import argparse


BSZ = 64


parser = argparse.ArgumentParser(description="Evaluation benchmark")
parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
parser.add_argument('--file_name', type=str, required=True, help="Name of the file")
args = parser.parse_args()

MODEL_PATH = args.model_path
file_name = args.file_name



llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len = 8192 * 2,
    gpu_memory_utilization=0.8,
    limit_mm_per_prompt={"image": 1, "video": 1},
)


sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    max_tokens=1024,
    stop_token_ids=[],
)


processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer


for dataset_name in ['mvbench','tempcompass','videomme','videommmu','vsibench','mmvu']:

    OUTPUT_PATH = f"./src/r1-v/eval_results/eval_{dataset_name}_{file_name}_greedy_output.json"
    PROMPT_PATH = f"./src/r1-v/Evaluation/eval_{dataset_name}.json"
    data = []
    
    if PROMPT_PATH.endswith('.jsonl'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    elif PROMPT_PATH.endswith('.json'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Input file must be .json or .jsonl")

    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Think step by step and provide your reasoning between <think> and </think> tags. "
        "Then give your final answer between <answer> and </answer> tags."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }


    messages = []
    sample_refs = []
    skipped_missing = 0
    total_samples = 0
    for x in data:
        total_samples += 1
        rel_path = x['path'].lstrip('./')

        full_path = os.path.join(os.getcwd(), 'src', 'r1-v', rel_path)

        tried = [full_path]

        if not os.path.exists(full_path) and full_path.endswith('.mp4'):
            webm_path = full_path[:-4] + '.webm'
            tried.append(webm_path)
            if os.path.exists(webm_path):
                full_path = webm_path


        if not os.path.exists(full_path):
            alt = os.path.join(os.getcwd(), rel_path)
            tried.append(alt)
            if os.path.exists(alt):
                full_path = alt
            else:
                # 额外尝试 alt 的 .webm
                if alt.endswith('.mp4'):
                    alt_webm = alt[:-4] + '.webm'
                    tried.append(alt_webm)
                    if os.path.exists(alt_webm):
                        full_path = alt_webm

        if not os.path.exists(full_path):
            print(f"[WARN] Missing {x.get('data_type','video')} file: {full_path} (skip)")
            skipped_missing += 1
            continue

        x['abs_path'] = full_path
        if x["problem_type"] == 'multiple choice':
            question = x['problem'] + "Options:\n" + "\n".join(x["options"])
        else:
            question = x['problem']
        messages.append([
            {
                'role': 'user',
                'content': [
                    {'type': x['data_type'], x['data_type']: full_path},
                    {'type': 'text', 'text': QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[x['problem_type']]}
                ]
            }
        ])
        sample_refs.append(x)
    
    if skipped_missing > 0:
        print(f"[INFO] Dataset {dataset_name}: skipped {skipped_missing}/{total_samples} missing-file samples.")
        

    final_output = []
    start_idx = 0
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
                final_output = existing.get("results", [])
                start_idx = len(final_output)
                print(f"Resuming from sample index {start_idx}")
        except Exception as e:
            print(f"Error reading existing output file: {e}")


    def extract_think(output_str):
        pattern = r'<think>\s*(.*?)\s*</think>'
        match = re.search(pattern, output_str, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            return None
        
    def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):

        if not torch.is_tensor(pred):
            pred = torch.tensor(pred, dtype=torch.float32)
        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.float32)
        
        epsilon = 1e-8
        rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
        
        thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
        
        conditions = rel_error < (1 - thresholds)  
        mra = conditions.float().mean()  
        return mra.item()


    def reward_fn(sample, model_output, question_type):
        try:
            output_ans = extract_answer(model_output)
            if output_ans == '':
                output_ans = model_output
            gt_ans = extract_answer(sample.get("solution", ""))
            if question_type == "multiple choice":
                return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    return 0.0
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    return 0.0
                return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    return 0.0
                mra = mean_relative_accuracy(out_number, gt_number)
                return mra
            else:
                return 0.0
        except Exception as e:
            return 0.0

    mean_acc = []
    mean_mra = []
    for i in tqdm(range(start_idx, len(messages), BSZ), desc="Processing batches"):
        batch_messages = messages[i:i + BSZ]

        prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        batch_output_text = []
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)
        except Exception as e:
            print(f"[ERROR] process_vision_info failed for batch starting {i}: {e}")
            batch_output_text = ['<answer>error</answer>'] * len(batch_messages)
            image_inputs, video_inputs, video_kwargs = [], [], {}

        if not batch_output_text:
            image_idx = 0
            video_idx = 0
            llm_inputs = []
            
            if image_inputs is None:
                image_inputs = []
            if video_inputs is None:
                video_inputs = []

            for idx, prompt in enumerate(prompts):
                mm_type = batch_messages[idx][0]['content'][0]['type']
                sample_mm_data = {}
                sample_video_kw = {}
                try:
                    if mm_type == 'image' and image_idx < len(image_inputs):
                        sample_mm_data["image"] = image_inputs[image_idx]
                        image_idx += 1
                    elif mm_type == 'video' and video_idx < len(video_inputs):
                        sample_mm_data["video"] = video_inputs[video_idx]
                        for key, value in video_kwargs.items():
                            sample_video_kw[key] = value[video_idx]
                        video_idx += 1
                    else:
                        print(f"[WARN] No mm data available for sample {i+idx} (type={mm_type})")
                except Exception as e:
                    print(f"[WARN] Failed to attach mm data for sample {i+idx}: {e}")
                
                llm_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": sample_mm_data,
                    "mm_processor_kwargs": sample_video_kw,
                })
                
            try:
                outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
                batch_output_text = [out.outputs[0].text for out in outputs]
            except Exception as e:
                print(f"[ERROR] Generation failed for batch starting {i}: {e}")
                batch_output_text = ['<answer>error</answer>'] * len(batch_messages)
            
        original_batch_samples = sample_refs[i:i+len(batch_messages)]
        for j, (sample, model_output) in enumerate(zip(original_batch_samples, batch_output_text), start=i):
            think_chain = extract_think(model_output)
            final_ans = extract_answer(model_output)
            if final_ans == "":
                final_ans = model_output
            sample["output"] = model_output
            sample["prediction"] = final_ans
            q_type = sample.get("problem_type", "")
            sample["reward"] = reward_fn(sample, model_output, q_type)
            sample['correct'] = True if sample["reward"]==1.0 else False
            if sample['problem_type'] != 'regression':
                mean_acc.append(sample["reward"])
            else:
                mean_mra.append(sample["reward"])
            if think_chain:
                sample["process"] = f"<think>{think_chain}</think>"
            final_output.append(sample)
        

        try:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
            print(f"Processed batch {(i - start_idx)//BSZ + 1}, saved {len(final_output)} samples.")
        except Exception as e:
            print(f"Error writing to output file: {e}")

    final_acc={'mean_acc': 0.0, 'mean_mra': 0.0}
    final_acc['mean_acc'] = torch.tensor(mean_acc).mean().item()
    if mean_mra != []:
        final_acc['mean_mra'] = torch.tensor(mean_mra).mean().item()
    
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"results": final_output, "final_acc": [final_acc]}, f, indent=2, ensure_ascii=False)
        print(f"Final accuracy saved to {OUTPUT_PATH}")
    except Exception as e:
        print(f"Error writing final accuracy to output file: {e}")
    
    print(f"Results saved to {OUTPUT_PATH}")
