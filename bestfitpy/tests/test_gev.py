# Binding-specific behaviour (vectorisation, return types). Oracle-value checks
# live in test_fixtures.py, driven by the shared language-neutral fixtures.
import numpy as np

from bestfitpy import dgev, pgev, qgev


def test_vectorised_over_arrays():
    x = np.array([90.0, 100.0, 110.0])
    out = dgev(x, 100, 10, 0)
    assert out.shape == (3,)
    assert out[1] == dgev(100.0, 100, 10, 0)


def test_scalar_returns_scalar():
    assert np.isscalar(pgev(100.0, 100, 10, 0)) or np.ndim(pgev(100.0, 100, 10, 0)) == 0
    assert np.ndim(qgev(0.5, 100, 10, 0)) == 0
