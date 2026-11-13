import numpy as np
import pytest

from tensor_toolkit.ops import batch_l2_normalize, sliding_window


def test_batch_l2_normalize_unit_rows():
    x = np.array([[3.0, 4.0], [0.0, 5.0]])
    out = batch_l2_normalize(x, axis=1)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0)


def test_batch_l2_normalize_zero_vector_stays_finite():
    x = np.zeros((2, 3))
    out = batch_l2_normalize(x)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, 0.0)


def test_sliding_window_shape_and_contents():
    x = np.arange(10)
    windows = sliding_window(x, window=4, stride=3)
    assert windows.shape == (3, 4)
    np.testing.assert_array_equal(windows[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(windows[1], [3, 4, 5, 6])
    np.testing.assert_array_equal(windows[2], [6, 7, 8, 9])


def test_sliding_window_rejects_bad_args():
    with pytest.raises(ValueError):
        sliding_window(np.arange(5), window=0)
    with pytest.raises(ValueError):
        sliding_window(np.arange(5), window=6)
