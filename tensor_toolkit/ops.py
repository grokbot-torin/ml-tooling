"""Core array / tensor helpers."""

from __future__ import annotations

import numpy as np


def batch_l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize slices of ``x`` along ``axis``.

    Parameters
    ----------
    x:
        Input array of any shape.
    axis:
        Axis along which to compute the norm.
    eps:
        Floor added to the norm so zero vectors stay finite.
    """
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / np.maximum(norms, eps)


def sliding_window(x: np.ndarray, window: int, stride: int = 1, axis: int = 0) -> np.ndarray:
    """Return overlapping windows along ``axis``.

    The output shape is the input shape with ``axis`` replaced by
    ``(n_windows, window)``.
    """
    x = np.asarray(x)
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    length = x.shape[axis]
    if window > length:
        raise ValueError(f"window {window} is larger than axis length {length}")
    n_windows = 1 + (length - window) // stride
    indexer = [slice(None)] * x.ndim
    windows = []
    for i in range(n_windows):
        start = i * stride
        indexer[axis] = slice(start, start + window)
        windows.append(x[tuple(indexer)])
    stacked = np.stack(windows, axis=axis)
    # Move the window length next to the window index.
    return np.moveaxis(stacked, axis + 1, axis + 1)
