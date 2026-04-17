"""
Data collators - moved to dataset.py
This file kept for backwards compatibility
"""
from .dataset import DataCollatorForSupervisedDataset, FlattenedDataCollatorForSupervisedDataset

__all__ = [
    "DataCollatorForSupervisedDataset",
    "FlattenedDataCollatorForSupervisedDataset",
]
