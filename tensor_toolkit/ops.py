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


def soft_clip(x: np.ndarray, low: float, high: float, sharpness: float = 4.0) -> np.ndarray:
    """Smoothly squash values into ``(low, high)`` with a tanh curve.

    The midpoint of the range is a fixed point. Larger ``sharpness``
    values make the function approach the bounds more quickly.
    """
    if high <= low:
        raise ValueError("high must be greater than low")
    if sharpness <= 0:
        raise ValueError("sharpness must be positive")
    x = np.asarray(x, dtype=float)
    mid = 0.5 * (low + high)
    half = 0.5 * (high - low)
    return mid + half * np.tanh(sharpness * (x - mid) / half)


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Element-wise division that never divides by a near-zero value.

    Denominators with absolute value below ``eps`` are replaced by
    ``eps`` with the original sign preserved (positive if the value
    was exactly zero).
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    replacement = np.where(den >= 0, eps, -eps)
    safe = np.where(np.abs(den) < eps, replacement, den)
    return num / safe


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
    axis: int = -1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Cosine similarity between ``a`` and ``b`` along ``axis``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    num = np.sum(a * b, axis=axis)
    denom = np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis)
    return num / np.maximum(denom, eps)
