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

