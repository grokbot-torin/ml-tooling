"""Feature encoding helpers."""

from __future__ import annotations

import numpy as np


def one_hot(labels: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """One-hot encode a 1-d integer label array."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if labels.size == 0:
        if num_classes is None:
            raise ValueError("num_classes is required when labels are empty")
        return np.zeros((0, num_classes), dtype=float)
    if np.any(labels < 0):
        raise ValueError("labels must be non-negative")
    if num_classes is None:
        num_classes = int(np.max(labels)) + 1
    if np.any(labels >= num_classes):
        raise ValueError("labels are out of range for num_classes")
    out = np.zeros((labels.shape[0], num_classes), dtype=float)
    out[np.arange(labels.shape[0]), labels.astype(int)] = 1.0
    return out

