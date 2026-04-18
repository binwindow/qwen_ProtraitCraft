# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Qwen-VL (Qwen3-VL-2B-Instruct / Qwen3-VL-4B-Instruct) fine-tuning project for portrait aesthetics evaluation. The model is trained to assess images against 13 visual criteria (Color Harmony, Sharpness, Depth of Field, etc.) and predict aesthetic scores.

## Environment

All Python scripts run in the **conda environment `craft`**. Activate with:
```bash
conda activate craft
```

When uncertain about implementation details, reference the upstream project:
```
/home/lsc/project/github/PortraitCraft/qwen-vl-finetune
```

## Architecture

### Training (transformers.Trainer)

The training code has been migrated to use `transformers.Trainer` for alignment with upstream:

```
train.py                          # Main entry point
src/train/
├── trainer_patch.py             # Flash attention patches, create_optimizer override
├── validation_callback.py        # ValidationAndCheckpointCallback (merged)
└── logging_callback.py           # (merged into validation_callback)
src/evaluation/
├── val_evaluator.py             # ValidationEvaluator for SRCC/PLCC/LevelAcc metrics
└── metrics.py                    # Correlation metrics computation
src/logging/
└── logger.py                    # LoggerManager (3-layer logging)
```

### Data Format
Training/test JSON files contain items with:
- `image_path`: Image file path
- `criteria`: Dict of 13 criteria with `level` (Good/Medium/Poor) and `reason`
- `total_score`: Overall aesthetic score (1-100)
- `question`/`options`/`answer`: Multiple-choice QA (test set only)

## Commands

### Training
```bash
# Train with validation
bash scripts/train.sh --gpu 0 --exp_name my_exp --batch_size 1 --train_max_samples 8 --val_max_samples 10 --val_eval_steps 5
```

**Key Training Arguments**:
- `--gpu` - GPU device ID
- `--exp_name` - Experiment name
- `--batch_size` - Batch size per device
- `--train_max_samples` - Limit training samples (None = all)
- `--val_max_samples` - Limit validation samples
- `--val_eval_steps` - Run validation every N steps
- `--lora_enable` - Enable LoRA training

### Testing
```bash
# Test pretrained model
python test.py --model_name_or_path ./source/Qwen3-VL-2B-Instruct

# Test fine-tuned model from checkpoint
python test.py --ckpt outputs/my_exp/ckpt/checkpoint-50-srcc0.8500
```

**Key Test Arguments**:
- `--ckpt` - Load fine-tuned model from checkpoint
- `--prompt_type` - `simple` (level=x) or `enhanced` (detailed descriptions)
- `--dataset_type` - `test` (includes question/options) or `val` (criteria only)

### Output Directories
```
outputs/<exp_name>/
├── ckpt/                              # Trainer checkpoints with metrics
│   ├── checkpoint-10-srcc0.8500/     # Named: checkpoint-{step}-srcc{value}
│   ├── checkpoint-20-srcc0.8720/
│   └── checkpoint-30-srcc0.8690/     # Top-k (keeps best by srcc)
├── log/
│   ├── train_metrics.jsonl            # Training loss/logs
│   ├── val_metrics.jsonl               # Validation metrics (srcc, plcc, level_acc)
│   └── train.log                      # Python logging
├── samples/
│   └── val_{step}.json               # Validation results per step
├── plt_fig/                          # Reserved for figures
└── test/                             # Test outputs
```

## Key Files

| File | Purpose |
|------|---------|
| `train.py` | Main training entry using transformers.Trainer |
| `src/train/validation_callback.py` | ValidationAndCheckpointCallback - runs validation, saves checkpoints with metrics, manages top-k |
| `src/train/trainer_patch.py` | Flash attention patch, create_optimizer override |
| `src/evaluation/val_evaluator.py` | ValidationEvaluator for inference and metrics |
| `src/evaluation/metrics.py` | SRCC, PLCC, Level Accuracy computation |
| `src/logging/logger.py` | LoggerManager (SwanLab, JSON, Python logging) |
| `test.py` | Main inference script |

## Training Details

### Callbacks
The `ValidationAndCheckpointCallback` merges three functions:
1. **Validation** - Runs inference on validation set at `val_eval_steps` intervals
2. **Checkpointing** - Saves model to `checkpoint-{step}-srcc{value}/` after validation
3. **Top-k Management** - Keeps best k checkpoints by srcc metric

### Validation Flow
```
on_step_end (every eval_steps)
  └─> 1. Run validation (restore attention → inference → re-patch)
      2. Log val_metrics to log/val_metrics.jsonl
      3. Save checkpoint to ckpt/checkpoint-{step}-srcc{value}/
      4. Manage top-k list (remove worst if > k)
```

### Logging
- **train_metrics.jsonl** - Training loss via on_log callback
- **val_metrics.jsonl** - Validation metrics (srcc, plcc, level_acc)
- **train.log** - Python logging format

## Model Path

Use local symlinks in `source/`:
```bash
./source/Qwen3-VL-2B-Instruct  # 2B model
./source/Qwen3-VL-4B-Instruct  # 4B model
```

## Dependencies

Key packages: `torch`, `transformers>=4.40.0`, `peft>=0.10.0`, `flash-attn>=2.5.0`, `swanlab`, `scipy>=1.11.0`
