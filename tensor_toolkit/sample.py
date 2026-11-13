"""Sampling and split helpers."""

from __future__ import annotations

import numpy as np


def train_test_split(*arrays, test_size: float = 0.25, shuffle: bool = True, rng=None):
    """Split one or more arrays along the first axis into train and test.

    Returns ``(train_0, test_0, train_1, test_1, ...)`` for the
    provided arrays, all sharing the same shuffled index order.
    """
    if not arrays:
        raise ValueError("at least one array is required")
    length = len(arrays[0])
    if any(len(array) != length for array in arrays):
        raise ValueError("all arrays must have the same length")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    rng = np.random.default_rng(rng)
    idx = np.arange(length)
    if shuffle:
        rng.shuffle(idx)
    n_test = int(np.ceil(length * test_size))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    result = []
    for array in arrays:
        array = np.asarray(array)
        result.append(array[train_idx])
        result.append(array[test_idx])
    return tuple(result)

