#!/bin/bash
# run_models.sh

model_paths=(
    "/modelpath"
)

file_names=(
    "CARE"
)

export DECORD_EOF_RETRY_MAX=40960


for i in "${!model_paths[@]}"; do
    model="${model_paths[$i]}"
    file_name="${file_names[$i]}"
    CUDA_VISIBLE_DEVICES=0 python ./src/eval_bench.py --model_path "$model" --file_name "$file_name"
done
