"""tensor-toolkit: lightweight NumPy utilities for everyday array work."""

from tensor_toolkit.pipeline import Standardizer, MinMaxScaler, Pipeline, winsorize
from tensor_toolkit.augment import gaussian_noise, random_scale, mixup
from tensor_toolkit.encode import one_hot, LabelEncoder, bin_features
from tensor_toolkit.tokenize import whitespace_tokenize, Vocabulary, ngrams, pad_sequences
from tensor_toolkit.sample import train_test_split, weighted_choice
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
    'winsorize',
    'gaussian_noise',
    'random_scale',
    'mixup',
    'one_hot',
    'LabelEncoder',
    'bin_features',
    'whitespace_tokenize',
    'Vocabulary',
    'ngrams',
    'pad_sequences',
    'train_test_split',
    'weighted_choice',
]
__version__ = "0.1.0"
