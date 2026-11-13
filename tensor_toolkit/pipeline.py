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




class MinMaxScaler:
    """Scale features to a target range, defaulting to ``[0, 1]``."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0), axis: int = 0, eps: float = 1e-12) -> None:
        low, high = feature_range
        if high <= low:
            raise ValueError("feature_range must satisfy high > low")
        self.feature_range = (float(low), float(high))
        self.axis = axis
        self.eps = eps
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "MinMaxScaler":
        x = np.asarray(x, dtype=float)
        self.data_min_ = np.min(x, axis=self.axis, keepdims=True)
        self.data_max_ = np.max(x, axis=self.axis, keepdims=True)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("MinMaxScaler must be fit before transform")
        x = np.asarray(x, dtype=float)
        low, high = self.feature_range
        span = np.maximum(self.data_max_ - self.data_min_, self.eps)
        scaled = (x - self.data_min_) / span
        return scaled * (high - low) + low

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("MinMaxScaler must be fit before inverse_transform")
        x = np.asarray(x, dtype=float)
        low, high = self.feature_range
        span = np.maximum(self.data_max_ - self.data_min_, self.eps)
        return (x - low) / (high - low) * span + self.data_min_



class Pipeline:
    """Apply a sequence of fit/transform steps to an array."""

    def __init__(self, steps: list) -> None:
        if not steps:
            raise ValueError("Pipeline needs at least one step")
        self.steps = list(steps)

    def fit(self, x: np.ndarray) -> "Pipeline":
        x = np.asarray(x)
        for step in self.steps[:-1]:
            x = _fit_transform_step(step, x)
        last = self.steps[-1]
        if hasattr(last, "fit"):
            last.fit(x)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        for step in self.steps:
            x = _transform_step(step, x)
        return x

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        for step in self.steps:
            x = _fit_transform_step(step, x)
        return x


def _fit_transform_step(step, x):
    if hasattr(step, "fit_transform"):
        return step.fit_transform(x)
    if hasattr(step, "fit") and hasattr(step, "transform"):
        return step.fit(x).transform(x)
    if callable(step):
        return step(x)
    raise TypeError(f"unsupported pipeline step: {step!r}")


def _transform_step(step, x):
    if hasattr(step, "transform"):
        return step.transform(x)
    if callable(step):
        return step(x)
    raise TypeError(f"unsupported pipeline step: {step!r}")
