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



def test_soft_clip_bounds_and_midpoint():
    from tensor_toolkit.ops import soft_clip

    x = np.array([-100.0, 0.0, 100.0])
    out = soft_clip(x, low=-1.0, high=1.0, sharpness=4.0)
    assert -1.0 <= out[0] < -0.9
    assert 0.9 < out[2] <= 1.0
    np.testing.assert_allclose(soft_clip(np.array([0.0]), -1.0, 1.0), 0.0)


def test_soft_clip_rejects_bad_range():
    from tensor_toolkit.ops import soft_clip

    with pytest.raises(ValueError):
        soft_clip(np.array([0.0]), 1.0, 0.0)



def test_safe_divide_regular_and_near_zero():
    from tensor_toolkit.ops import safe_divide

    np.testing.assert_allclose(safe_divide([6.0, 4.0], [3.0, 2.0]), [2.0, 2.0])
    out = safe_divide([1.0, -1.0], [0.0, -1e-20], eps=1e-12)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[0], 1.0 / 1e-12)
    np.testing.assert_allclose(out[1], -1.0 / -1e-12)
