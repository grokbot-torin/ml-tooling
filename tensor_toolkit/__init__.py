"""tensor-toolkit: lightweight NumPy utilities for everyday array work."""

from tensor_toolkit.pipeline import Standardizer, MinMaxScaler, Pipeline
from tensor_toolkit.ops import batch_l2_normalize, sliding_window, soft_clip, safe_divide, cosine_similarity, softmax, log_softmax, masked_mean, topk, pad_to_multiple

__all__ = [
    'batch_l2_normalize',
    'sliding_window',
    'soft_clip',
    'safe_divide',
    'cosine_similarity',
    'softmax',
    'log_softmax',
    'masked_mean',
    'topk',
    'pad_to_multiple',
    'Standardizer',
    'MinMaxScaler',
    'Pipeline',
]
__version__ = "0.1.0"
