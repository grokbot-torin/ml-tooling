"""Lightweight numeric augmentation helpers."""

from __future__ import annotations

import numpy as np


def gaussian_noise(x: np.ndarray, scale: float = 0.01, rng=None) -> np.ndarray:
    """Add i.i.d. Gaussian noise with standard deviation ``scale``."""
    x = np.asarray(x, dtype=float)
    if scale < 0:
        raise ValueError("scale must be non-negative")
    rng = np.random.default_rng(rng)
    return x + rng.normal(0.0, scale, size=x.shape)




def random_scale(
    x: np.ndarray,
    low: float = 0.9,
    high: float = 1.1,
    axis: int | None = None,
    rng=None,
) -> np.ndarray:
    """Multiply ``x`` by a random factor drawn uniformly from ``[low, high]``.

    If ``axis`` is given, a separate factor is drawn for each index
    along that axis and broadcast across the remaining dimensions.
    """
    x = np.asarray(x, dtype=float)
    if high < low:
        raise ValueError("high must be >= low")
    rng = np.random.default_rng(rng)
    if axis is None:
        return x * rng.uniform(low, high)
    shape = [1] * x.ndim
    shape[axis] = x.shape[axis]
    return x * rng.uniform(low, high, size=tuple(shape))
