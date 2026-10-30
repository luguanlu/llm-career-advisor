#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
export USE_MODELSCOPE_HUB=1
export HF_ENDPOINT=https://hf-mirror.com

llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path ZhipuAI/glm-4-9b-chat \
    --dataset tech_advisor \
    --dataset_dir ./data \
    --template glm4 \
    --finetuning_type lora \
    --output_dir ./saves/GLM4-9B/lora/tech_advisor_model \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --warmup_ratio 0.1 \
    --save_steps 50 \
    --eval_strategy steps \
    --eval_steps 50 \
    --val_size 0.1 \
    --load_best_model_at_end True \
    --learning_rate 2e-4 \
    --num_train_epochs 3.0 \
    --quantization_bit 4 \
    --trust_remote_code True \
    --bf16