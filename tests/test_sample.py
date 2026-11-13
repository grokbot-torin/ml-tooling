import numpy as np
import pytest

from tensor_toolkit.sample import train_test_split


def test_train_test_split_sizes_and_alignment():
    x = np.arange(10)
    y = np.arange(10, 20)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, rng=0)
    assert len(x_test) == 3 and len(x_train) == 7
    np.testing.assert_array_equal(x_train + 10, y_train)
    np.testing.assert_array_equal(x_test + 10, y_test)
    # disjoint and complete
    assert set(x_train).isdisjoint(x_test)
    assert set(x_train).union(x_test) == set(x)

def test_train_test_split_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        train_test_split(np.arange(5), np.arange(4))




from tensor_toolkit.sample import weighted_choice


def test_weighted_choice_respects_zero_weights():
    pop = np.array(["a", "b", "c"])
    out = weighted_choice(pop, weights=[0.0, 1.0, 0.0], size=8, replace=True, rng=3)
    assert set(out.tolist()) == {"b"}
    with pytest.raises(ValueError):
        weighted_choice(pop, weights=[-1.0, 1.0, 0.0])
