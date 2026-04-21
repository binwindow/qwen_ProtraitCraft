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
VAL_MAX_SAMPLES=200
VAL_EVAL_STEPS=5
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
        --grad_accum|--acc)
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
        --val_max_samples)
            VAL_MAX_SAMPLES="$2"
            shift 2
            ;;
        --val_eval_steps)
            VAL_EVAL_STEPS="$2"
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

MODEL_PATH="./source/Qwen3-VL-2B-Instruct"
DATASET_PATH="/home/lsc/.cache/huggingface/hub/datasets--zijielou--PortraitCraft/snapshots/8d5471d72f6a29d23117f8362c04c45c42fe2acc"
OUTPUT_DIR="./outputs/${EXP_NAME}"

mkdir -p $(dirname "$OUTPUT_DIR")

eval "$(conda shell.bash hook)"
conda activate craft

python train.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --bf16 \
    --gradient_checkpointing \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    $(if [ "$LORA_ENABLE" = "true" ]; then echo "--lora_enable"; fi) \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --learning_rate ${LR} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --eval_strategy no \
    --save_strategy no \
    --logging_steps 1 \
    --model_max_length 8192 \
    --max_pixels 50176 \
    --min_pixels 784 \
    --seed ${SEED} \
    --data_flatten \
    --data_packing false \
    $(if [ -n "$TRAIN_MAX_SAMPLES" ]; then echo "--train_max_samples ${TRAIN_MAX_SAMPLES}"; fi) \
    --dataloader_num_workers 4 \
    --val_json_path "${DATASET_PATH}/track_1_val.json" \
    --val_images_path "${DATASET_PATH}" \
    --val_max_samples ${VAL_MAX_SAMPLES} \
    --val_eval_steps ${VAL_EVAL_STEPS}
