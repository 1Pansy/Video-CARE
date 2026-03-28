cd src/r1-v
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"

export FORCE_MAX_PROMPT_LEN=16384  
export EMPTY_CACHE_EVERY=10  
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Qwen/Qwen2.5-VL-7B-Instruct

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    src/open_r1/grpo.py \
    --report_to none \
    --output_dir "./log/Qwen2.5-VL-7B-GRPO" \
    --model_name_or_path 'SFT Model Path' \
    --dataset_name "./Video-R1-data/Video-R1-260k.json" \
#  --resume_from_checkpoint "checkpoint Path" \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --temporal true \
    --len_control true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --vision_patch_limit 401408 \
    --video_logprob_skip_len 16384 \
    --num_train_epochs 1 \
    --run_name Video-R1 \
    --save_steps 100 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 8  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  