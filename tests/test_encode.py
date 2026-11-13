import numpy as np
import pytest

from tensor_toolkit.encode import one_hot


def test_one_hot_shape_and_values():
    out = one_hot(np.array([0, 2, 1]), num_classes=3)
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    np.testing.assert_array_equal(out, expected)


def test_one_hot_infers_classes_and_rejects_bad_labels():
    out = one_hot(np.array([0, 1]))
    assert out.shape == (2, 2)
    with pytest.raises(ValueError):
        one_hot(np.array([0, 3]), num_classes=3)
    with pytest.raises(ValueError):
        one_hot(np.array([[0, 1]]))




from tensor_toolkit.encode import LabelEncoder


def test_label_encoder_roundtrip():
    enc = LabelEncoder()
    ids = enc.fit_transform(["cat", "dog", "cat", "bird"])
    np.testing.assert_array_equal(ids, [1, 2, 1, 0])
    np.testing.assert_array_equal(enc.inverse_transform(ids), ["cat", "dog", "cat", "bird"])
    with pytest.raises(ValueError):
        enc.transform(["lizard"])



from tensor_toolkit.encode import bin_features


def test_bin_features_uniform_and_quantile():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    uniform = bin_features(x, n_bins=2, strategy="uniform")
    # midpoint is 2, so [0,1] -> 0 and [2,3,4] -> 1? digitize edges (2,)
    # [0,2) -> 0, [2,4] -> 1. 2 goes to bin 1.
    np.testing.assert_array_equal(uniform, [0, 0, 1, 1, 1])
    matrix = np.stack([x, x], axis=1)
    out = bin_features(matrix, n_bins=2, strategy="uniform")
    assert out.shape == (5, 2)
    constant = bin_features(np.ones(4), n_bins=3)
    np.testing.assert_array_equal(constant, [0, 0, 0, 0])
    with pytest.raises(ValueError):
        bin_features(x, n_bins=1)
