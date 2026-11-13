import numpy as np
import pytest

from tensor_toolkit.pipeline import Standardizer


def test_standardizer_zero_mean_unit_variance_and_inverse():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=2.0, size=(200, 3))
    scaler = Standardizer(axis=0)
    z = scaler.fit_transform(x)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(scaler.inverse_transform(z), x)

def test_standardizer_requires_fit():
    with pytest.raises(RuntimeError):
        Standardizer().transform(np.ones((2, 2)))




from tensor_toolkit.pipeline import MinMaxScaler


def test_minmax_scaler_range_and_inverse():
    x = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
    scaler = MinMaxScaler(feature_range=(0.0, 1.0), axis=0)
    z = scaler.fit_transform(x)
    np.testing.assert_allclose(z.min(axis=0), 0.0)
    np.testing.assert_allclose(z.max(axis=0), 1.0)
    np.testing.assert_allclose(scaler.inverse_transform(z), x)



from tensor_toolkit.pipeline import Pipeline


def test_pipeline_chains_standardizer_and_minmax():
    rng = np.random.default_rng(1)
    x = rng.normal(loc=3.0, scale=4.0, size=(100, 2))
    pipe = Pipeline([Standardizer(axis=0), MinMaxScaler(axis=0)])
    z = pipe.fit_transform(x)
    np.testing.assert_allclose(z.min(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(z.max(axis=0), 1.0, atol=1e-12)
    # transform on the same data should match
    np.testing.assert_allclose(pipe.transform(x), z)

def test_pipeline_rejects_empty_steps():
    with pytest.raises(ValueError):
        Pipeline([])
