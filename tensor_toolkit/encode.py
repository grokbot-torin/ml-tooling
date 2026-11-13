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




class LabelEncoder:
    """Map hashable labels to contiguous integer ids."""

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self._map: dict | None = None

    def fit(self, y) -> "LabelEncoder":
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self._map = {value: idx for idx, value in enumerate(self.classes_)}
        return self

    def transform(self, y) -> np.ndarray:
        if self._map is None:
            raise RuntimeError("LabelEncoder must be fit before transform")
        y = np.asarray(y)
        try:
            return np.array([self._map[value] for value in y], dtype=int)
        except KeyError as exc:
            raise ValueError(f"unseen label: {exc}") from exc

    def inverse_transform(self, ids) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("LabelEncoder must be fit before inverse_transform")
        ids = np.asarray(ids, dtype=int)
        if np.any(ids < 0) or np.any(ids >= len(self.classes_)):
            raise ValueError("ids are out of range")
        return self.classes_[ids]

    def fit_transform(self, y) -> np.ndarray:
        return self.fit(y).transform(y)
