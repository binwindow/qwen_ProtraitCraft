#!/bin/bash

# Qwen-VL Fine-tuning Training Script
# Usage:
#   bash scripts/train.sh --gpu 0 --exp_name my_exp --num_epochs 50 --grad_accum 4
#   bash scripts/train.sh --gpu 0 --exp_name my_exp

set -e

cd "$(dirname "$0")/.."

# Default values
GPU_IDS=""
EXP_NAME="qwen3vl_finetune"
NUM_EPOCHS=50
GRAD_ACCUM_STEPS=8
BATCH_SIZE=1
LR=1e-5
SEED=42
LORA_ENABLE="true"
VAL_NUM=50
VAL_MAX_SAMPLES=200
TRAIN_MAX_SAMPLES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_IDS="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --grad_accum)
            GRAD_ACCUM_STEPS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --no_lora)
            LORA_ENABLE="false"
            shift
            ;;
        --val_num)
            VAL_NUM="$2"
            shift 2
            ;;
        --val_max_samples)
            VAL_MAX_SAMPLES="$2"
            shift 2
            ;;
        --train_max_samples)
            TRAIN_MAX_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    echo "Using GPUs: $GPU_IDS"
fi

MODEL_PATH="/home/lsc/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
DATASET_PATH="/home/lsc/.cache/huggingface/hub/datasets--zijielou--PortraitCraft/snapshots/8d5471d72f6a29d23117f8362c04c45c42fe2acc"
VAL_JSON_PATH="./source/PortraitCraft_dataset/track_1_val.json"
VAL_IMAGES_PATH="./source/PortraitCraft_dataset"

eval "$(conda shell.bash hook)"
conda activate craft

python train.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --exp_name "${EXP_NAME}" \
    --save_dir ./outputs \
    --bf16 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    $(if [ "$LORA_ENABLE" = "true" ]; then echo "--lora_enable"; fi) \
    --learning_rate ${LR} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --val_num ${VAL_NUM} \
    --val_json_path "${VAL_JSON_PATH}" \
    --val_images_path "${VAL_IMAGES_PATH}" \
    --val_max_samples ${VAL_MAX_SAMPLES} \
    $(if [ -n "$TRAIN_MAX_SAMPLES" ]; then echo "--train_max_samples ${TRAIN_MAX_SAMPLES}"; fi) \
    --logging_steps 1 \
    --model_max_length 8192 \
    --max_pixels 50176 \
    --min_pixels 784 \
    --seed ${SEED} \
    --tune_mm_mlp \
    --tune_mm_llm \
    --data_flatten
