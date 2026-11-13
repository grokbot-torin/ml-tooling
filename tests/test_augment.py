import numpy as np
import pytest

from tensor_toolkit.augment import gaussian_noise


def test_gaussian_noise_is_zero_when_scale_is_zero():
    x = np.arange(6.0).reshape(2, 3)
    out = gaussian_noise(x, scale=0.0, rng=0)
    np.testing.assert_array_equal(out, x)


def test_gaussian_noise_is_reproducible_and_centered():
    x = np.zeros((1000, 4))
    a = gaussian_noise(x, scale=0.5, rng=7)
    b = gaussian_noise(x, scale=0.5, rng=7)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a.mean(), 0.0, atol=0.05)
    with pytest.raises(ValueError):
        gaussian_noise(x, scale=-1)




from tensor_toolkit.augment import random_scale


def test_random_scale_global_and_per_axis():
    x = np.ones((4, 3))
    out = random_scale(x, low=2.0, high=2.0, rng=0)
    np.testing.assert_allclose(out, 2.0)
    per = random_scale(x, low=0.5, high=1.5, axis=0, rng=1)
    # each row should be a constant multiple of ones
    row_vals = per[:, 0]
    np.testing.assert_allclose(per, np.repeat(row_vals[:, None], 3, axis=1))
    assert np.all(row_vals >= 0.5) and np.all(row_vals <= 1.5)



from tensor_toolkit.augment import mixup


def test_mixup_is_convex_combination():
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.eye(3)
    x_mix, y_mix, lam = mixup(x, y, alpha=0.4, rng=2)
    assert 0.0 <= lam <= 1.0
    assert x_mix.shape == x.shape
    assert y_mix.shape == y.shape
    np.testing.assert_allclose(y_mix.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        mixup(x, y[:2], alpha=0.4, rng=0)
