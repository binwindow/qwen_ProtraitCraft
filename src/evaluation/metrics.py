"""
Evaluation metrics for PortraitCraft test results
"""
import json
import os
from scipy.stats import spearmanr, pearsonr
import numpy as np


def compute_correlation_metrics(gt_data, pred_data):
    """
    Compute SRCC, PLCC, and Level Acc from ground truth and predictions.

    Args:
        gt_data: List of ground truth items
        pred_data: List of prediction items

    Returns:
        dict with srcc, plcc, level_acc, num_samples
    """
    gt_map = {item["image_path"]: item for item in gt_data}
    pred_map = {item["image_path"]: item for item in pred_data}

    scores_gt = []
    scores_pred = []
    level_correct = 0
    level_total = 0
    num_samples = 0

    for img_path, gt_item in gt_map.items():
        if img_path not in pred_map:
            continue

        pred_item = pred_map[img_path]
        num_samples += 1

        gt_score = gt_item.get("total_score", 0)
        pred_score = pred_item.get("total_score", 0)
        if gt_score is not None and pred_score is not None:
            scores_gt.append(gt_score)
            scores_pred.append(pred_score)

        gt_criteria = gt_item.get("criteria", {})
        pred_criteria = pred_item.get("criteria", {})

        for k, gt_v in gt_criteria.items():
            if isinstance(gt_v, dict):
                gt_level = gt_v.get("level", "")
            else:
                gt_level = gt_v

            level_total += 1
            if k in pred_criteria:
                pred_v = pred_criteria[k]
                if isinstance(pred_v, dict):
                    pred_level = pred_v.get("level", "")
                else:
                    pred_level = pred_v

                if gt_level and pred_level and gt_level == pred_level:
                    level_correct += 1

    srcc = spearmanr(scores_gt, scores_pred)[0] if len(scores_gt) > 1 else 0.0
    plcc = pearsonr(scores_gt, scores_pred)[0] if len(scores_pred) > 1 else 0.0
    level_acc = level_correct / level_total if level_total > 0 else 0.0

    return {
        "srcc": float(srcc) if not np.isnan(srcc) else -1.0,
        "plcc": float(plcc) if not np.isnan(plcc) else -1.0,
        "level_acc": level_acc,
        "num_samples": num_samples,
    }


def evaluate_and_save(input_json, pred_json, metrics_path=None, max_samples=None):
    """
    Evaluate prediction results against ground truth.

    Args:
        input_json: Path to ground truth JSON
        pred_json: Path to prediction JSON
        metrics_path: Optional path for metrics output (auto-detected if None)
        max_samples: Optional limit on number of samples to evaluate

    Returns:
        dict with srcc, plcc, level_acc, qa_acc, num_samples, num_total
    """
    with open(input_json, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    if not os.path.exists(pred_json):
        return {
            "srcc": 0.0, "plcc": 0.0,
            "level_acc": 0.0, "qa_acc": 0.0,
            "num_samples": 0, "num_total": len(gt_data)
        }

    with open(pred_json, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    gt_map = {item["image_path"]: item for item in gt_data}
    pred_map = {item["image_path"]: item for item in pred_data}

    scores_gt = []
    scores_pred = []
    level_correct = 0
    level_total = 0
    qa_correct = 0
    qa_total = 0
    num_samples = 0

    num_total = len(gt_data)
    num_samples = 0
    total_criteria_per_image = 13

    for img_path, gt_item in gt_map.items():
        if img_path not in pred_map:
            continue

        pred_item = pred_map[img_path]
        num_samples += 1

        # Limit samples if max_samples is set
        if max_samples is not None and num_samples >= max_samples:
            break

        gt_score = gt_item.get("total_score", 0)
        pred_score = pred_item.get("total_score", 0)
        if gt_score is not None and pred_score is not None:
            scores_gt.append(gt_score)
            scores_pred.append(pred_score)

        gt_criteria = gt_item.get("criteria", {})
        pred_criteria = pred_item.get("criteria", {})

        for k, gt_v in gt_criteria.items():
            if isinstance(gt_v, dict):
                gt_level = gt_v.get("level", "")
            else:
                gt_level = gt_v

            level_total += 1
            if k in pred_criteria:
                pred_v = pred_criteria[k]
                if isinstance(pred_v, dict):
                    pred_level = pred_v.get("level", "")
                else:
                    pred_level = pred_v

                if gt_level and pred_level and gt_level == pred_level:
                    level_correct += 1

        gt_answer = gt_item.get("answer", "")
        pred_answer = pred_item.get("answer", "")
        qa_total += 1
        if gt_answer and pred_answer and gt_answer == pred_answer:
            qa_correct += 1

    srcc = spearmanr(scores_gt, scores_pred)[0] if len(scores_gt) > 1 else 0.0
    plcc = pearsonr(scores_gt, scores_pred)[0] if len(scores_pred) > 1 else 0.0
    level_acc = level_correct / level_total if level_total > 0 else 0.0
    qa_acc = qa_correct / qa_total if qa_total > 0 else 0.0

    metrics = {
        "srcc": srcc if not np.isnan(srcc) else -1.0,
        "plcc": plcc if not np.isnan(plcc) else -1.0,
        "level_acc": level_acc,
        "qa_acc": qa_acc,
        "num_samples": num_samples,
        "num_total": len(gt_data)
    }

    output_dir = os.path.dirname(pred_json)
    if metrics_path is None:
        metrics_path = os.path.join(output_dir, "test_metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    # Append to existing metrics file (as list) or create new
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_list = json.load(f)
        if not isinstance(metrics_list, list):
            metrics_list = [metrics_list]
    else:
        metrics_list = []

    # Add timestamp and max_samples info
    from datetime import datetime
    metrics["timestamp"] = datetime.now().isoformat()
    metrics["max_samples_used"] = max_samples if max_samples else num_samples

    metrics_list.append(metrics)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2)

    return metrics
