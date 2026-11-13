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
