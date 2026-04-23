#!/bin/bash

# Qwen-VL Test Script
# Usage:
#   bash scripts/test.sh --gpu 0 --ckpt outputs/my_exp/ckpt/last.ckpt
#   bash scripts/test.sh --gpu 0 --exp_name pretrained_test

set -e

cd "$(dirname "$0")/.."

GPU_IDS=""
EXP_NAME="${EXP_NAME:-qwen3vl_finetune}"
CKPT_PATH="${CKPT_PATH:-}"
MODEL_SELECTION="${MODEL_SELECTION:-qwen3-8b}"  # qwen3-2b, qwen3-4b, qwen3-8b, qwen2.5-7b
MODEL_PATH=""

# HuggingFace dataset path
DATASET_PATH="./source/PortraitCraft_dataset"
IMAGES_PATH="${DATASET_PATH}"

SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
PROMPT_TYPE="${PROMPT_TYPE:-simple}"
DATASET_TYPE="${DATASET_TYPE:-test}"
METRICS_MAX_SAMPLES="${METRICS_MAX_SAMPLES:-}"

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
        --ckpt)
            CKPT_PATH="$2"
            shift 2
            ;;
        --prompt_type)
            PROMPT_TYPE="$2"
            shift 2
            ;;
        --dataset_type)
            DATASET_TYPE="$2"
            shift 2
            ;;
        --model)
            MODEL_SELECTION="$2"
            shift 2
            ;;
        --metrics_max_samples)
            METRICS_MAX_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Resolve model path based on selection
case "$MODEL_SELECTION" in
    qwen3-2b)
        MODEL_PATH="${MODEL_PATH:-./source/Qwen3-VL-2B-Instruct}"
        ;;
    qwen3-4b)
        MODEL_PATH="${MODEL_PATH:-./source/Qwen3-VL-4B-Instruct}"
        ;;
    qwen3-8b)
        MODEL_PATH="${MODEL_PATH:-./source/Qwen3-VL-8B-Instruct}"
        ;;
    qwen2.5-7b)
        MODEL_PATH="${MODEL_PATH:-./source/Qwen2.5-VL-7B-Instruct}"
        ;;
    *)
        echo "Unknown model: $MODEL_SELECTION"
        echo "Available: qwen3-2b, qwen3-4b, qwen3-8b, qwen2.5-7b"
        exit 1
        ;;
esac

# Set GPU
if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    echo "Using GPUs: $GPU_IDS"
fi

eval "$(conda shell.bash hook)"
conda activate craft

if [ -n "$CKPT_PATH" ]; then
    echo "Testing fine-tuned model from: $CKPT_PATH"
    echo "Prompt type: $PROMPT_TYPE"
    echo "Dataset type: $DATASET_TYPE"
    python test.py \
        --ckpt "${CKPT_PATH}" \
        --images_path "${IMAGES_PATH}" \
        --seed ${SEED} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --prompt_type ${PROMPT_TYPE} \
        --dataset_type ${DATASET_TYPE} \
        $([ -n "$METRICS_MAX_SAMPLES" ] && echo "--metrics_max_samples ${METRICS_MAX_SAMPLES}")
else
    echo "Testing pretrained model"
    echo "Prompt type: $PROMPT_TYPE"
    echo "Dataset type: $DATASET_TYPE"
    python test.py \
        --exp_name "${EXP_NAME}" \
        --model_name_or_path "${MODEL_PATH}" \
        --images_path "${IMAGES_PATH}" \
        --seed ${SEED} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --prompt_type ${PROMPT_TYPE} \
        --dataset_type ${DATASET_TYPE} \
        $([ -n "$METRICS_MAX_SAMPLES" ] && echo "--metrics_max_samples ${METRICS_MAX_SAMPLES}")
fi