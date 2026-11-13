"""tensor-toolkit: lightweight NumPy utilities for everyday array work."""

from tensor_toolkit.ops import batch_l2_normalize, sliding_window, soft_clip, safe_divide, cosine_similarity

__all__ = [
    'batch_l2_normalize',
    'sliding_window',
    'soft_clip',
    'safe_divide',
    'cosine_similarity',
]
__version__ = "0.1.0"
