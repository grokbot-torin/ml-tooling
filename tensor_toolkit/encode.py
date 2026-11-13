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



def bin_features(
    x: np.ndarray,
    n_bins: int = 5,
    strategy: str = "uniform",
) -> np.ndarray:
    """Discretize numeric features into integer bin ids.

    ``strategy='uniform'`` uses evenly spaced edges between the min
    and max. ``strategy='quantile'`` uses empirical quantiles so
    each bin holds a similar share of the mass. Accepts 1-d or 2-d
    arrays (columns are binned independently).
    """
    x = np.asarray(x, dtype=float)
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    if x.ndim == 1:
        return _bin_1d(x, n_bins, strategy)
    if x.ndim != 2:
        raise ValueError("x must be 1d or 2d")
    columns = [_bin_1d(x[:, i], n_bins, strategy) for i in range(x.shape[1])]
    return np.stack(columns, axis=1)


def _bin_1d(x: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    if edges[0] == edges[-1]:
        return np.zeros(len(x), dtype=int)
    edges = np.unique(edges)
    # digitize interior edges: values fall into 0..len(edges)-2
    return np.digitize(x, edges[1:-1], right=False)
