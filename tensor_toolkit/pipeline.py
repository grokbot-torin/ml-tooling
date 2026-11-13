"""Composable array transforms for simple data pipelines."""

from __future__ import annotations

import numpy as np


class Standardizer:
    """Z-score each slice of an array along ``axis``."""

    def __init__(self, axis: int = 0, eps: float = 1e-12) -> None:
        self.axis = axis
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, dtype=float)
        self.mean_ = np.mean(x, axis=self.axis, keepdims=True)
        self.std_ = np.std(x, axis=self.axis, keepdims=True)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer must be fit before transform")
        x = np.asarray(x, dtype=float)
        return (x - self.mean_) / np.maximum(self.std_, self.eps)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer must be fit before inverse_transform")
        x = np.asarray(x, dtype=float)
        return x * np.maximum(self.std_, self.eps) + self.mean_

