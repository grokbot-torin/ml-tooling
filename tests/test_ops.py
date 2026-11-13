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



def test_cosine_similarity_identical_and_orthogonal():
    from tensor_toolkit.ops import cosine_similarity

    a = np.array([[1.0, 0.0], [1.0, 1.0]])
    b = np.array([[2.0, 0.0], [1.0, -1.0]])
    out = cosine_similarity(a, b, axis=1)
    np.testing.assert_allclose(out[0], 1.0)
    np.testing.assert_allclose(out[1], 0.0, atol=1e-12)



def test_softmax_sums_to_one_and_matches_log():
    from tensor_toolkit.ops import log_softmax, softmax

    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = softmax(x, axis=1)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    np.testing.assert_allclose(np.exp(log_softmax(x, axis=1)), probs)
    # large offsets should not overflow
    huge = softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert np.isfinite(huge).all()
    np.testing.assert_allclose(huge.sum(), 1.0)



def test_masked_mean_ignores_padded_positions():
    from tensor_toolkit.ops import masked_mean

    x = np.array([[1.0, 3.0, 99.0], [2.0, 4.0, 100.0]])
    mask = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    out = masked_mean(x, mask, axis=1)
    np.testing.assert_allclose(out, [2.0, 3.0])
    empty = masked_mean(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    np.testing.assert_allclose(empty, 0.0)



def test_topk_largest_and_smallest():
    from tensor_toolkit.ops import topk

    x = np.array([[3.0, 1.0, 4.0, 2.0], [8.0, 5.0, 7.0, 6.0]])
    vals, idx = topk(x, k=2, axis=1, largest=True)
    np.testing.assert_array_equal(vals, [[4.0, 3.0], [8.0, 7.0]])
    np.testing.assert_array_equal(idx, [[2, 0], [0, 2]])
    vals, idx = topk(x[0], k=2, largest=False)
    np.testing.assert_array_equal(vals, [1.0, 2.0])
    np.testing.assert_array_equal(idx, [1, 3])



def test_pad_to_multiple_extends_axis():
    from tensor_toolkit.ops import pad_to_multiple

    x = np.arange(5)
    out = pad_to_multiple(x, multiple=4, value=-1)
    np.testing.assert_array_equal(out, [0, 1, 2, 3, 4, -1, -1, -1])
    y = np.arange(8).reshape(2, 4)
    same = pad_to_multiple(y, multiple=2, axis=0)
    np.testing.assert_array_equal(same, y)
