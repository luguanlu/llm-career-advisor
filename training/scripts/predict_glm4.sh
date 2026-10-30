#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
export USE_MODELSCOPE_HUB=1
export HF_ENDPOINT=https://hf-mirror.com

llamafactory-cli train \
    --stage sft \
    --do_predict \
    --model_name_or_path ZhipuAI/glm-4-9b-chat \
    --adapter_name_or_path ./saves/GLM4-9B/lora/tech_advisor_model \
    --eval_dataset tech_advisor_test \
    --dataset_dir ./data \
    --template glm4 \
    --finetuning_type lora \
    --output_dir ./saves/GLM4-9B/lora/tech_advisor_predict \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate True \
    --quantization_bit 4 \
    --trust_remote_code True \
    --bf16