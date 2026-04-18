# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Qwen-VL (Qwen3-VL-4B-Instruct) fine-tuning project for portrait aesthetics evaluation. The model is trained to assess images against 13 visual criteria (Color Harmony, Sharpness, Depth of Field, etc.) and predict aesthetic scores.

## Environment

All Python scripts run in the **conda environment `craft`**. Activate with:
```bash
conda activate craft
```

When uncertain about implementation details, reference the upstream project:
```
/home/lsc/project/github/PortraitCraft/qwen-vl-finetune
```

## Commands

### Testing
```bash
# Test pretrained model
python test.py --model_name_or_path ./source/Qwen3-VL-4B-Instruct

# Test fine-tuned model from checkpoint
python test.py --ckpt outputs/my_exp/ckpt/last.ckpt

# Or use the shell script
bash scripts/test.sh --gpu 0 --ckpt outputs/my_exp/ckpt/last.ckpt
```

### Key Test Arguments
- `--ckpt` - Load fine-tuned model from checkpoint
- `--prompt_type` - `simple` (level=x) or `enhanced` (detailed descriptions)
- `--dataset_type` - `test` (includes question/options) or `val` (criteria only)
- `--exp_name` - Experiment name for output directory

### Output
- Results saved to `outputs/<exp_name>/test/test_results.json`
- Metrics saved to `outputs/<exp_name>/test/test_metrics.json`
- Metrics computed: **SRCC** (Spearman), **PLCC** (Pearson), **Level Acc**, **QA Acc**

## Architecture

```
src/
├── data/           # Dataset loading, collators, 2D RoPE position encoding
├── train/          # TrainingEngine, optimizer utilities
├── checkpoint/     # CheckpointManager (top-k strategy, always saves last.ckpt)
├── evaluation/     # Metrics (SRCC/PLCC correlation, level accuracy) and Validator
└── utils/          # Seed, experiment setup, scheduling, memory profiling
```

### Data Format
Training/test JSON files contain items with:
- `image_path`: Image file path
- `criteria`: Dict of 13 criteria with `level` (Good/Medium/Poor) and `reason`
- `total_score`: Overall aesthetic score (1-100)
- `question`/`options`/`answer`: Multiple-choice QA (test set only)

### Experiment Directory Structure
```
outputs/<exp_name>/
├── ckpt/          # Checkpoint files (last.ckpt, topk_*.ckpt)
├── log/           # Training logs
├── samples/       # Generated samples
├── plt_fig/       # Visualization figures
└── test/          # Test results and metrics
```

### Checkpoint Manager
- Always saves `last.ckpt` after each step
- Saves top-k checkpoints based on validation metrics (SRCC by default)
- Supports `metric_optimization=max` (for SRCC) or `min` (for loss)

## Key Files

| File | Purpose |
|------|---------|
| `test.py` | Main inference script with prompt building and JSON extraction |
| `src/data/dataset.py` | LazySupervisedDataset, chat template processing, conversation building |
| `src/data/rope2d.py` | 2D RoPE index computation for Qwen2/3 VL models |
| `src/checkpoint/manager.py` | Top-k checkpoint strategy |
| `src/evaluation/metrics.py` | SRCC, PLCC, Level Accuracy computation |

## Dependencies

Key packages: `torch`, `transformers>=4.40.0`, `peft>=0.10.0`, `accelerate>=0.25.0`, `flash-attn>=2.5.0`, `swanlab`, `scipy>=1.11.0`
