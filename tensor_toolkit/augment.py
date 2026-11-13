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

