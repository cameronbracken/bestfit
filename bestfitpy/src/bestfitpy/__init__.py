"""bestfitpy: Bayesian flood-frequency and extreme-value analysis.

Python bindings to a shared C++ port of the USACE-RMC Numerics / RMC.BestFit
libraries. This early version exposes the Generalized Extreme Value distribution.
"""

from __future__ import annotations

import numpy as np

from ._core import GeneralizedExtremeValue, gev_fit as _gev_fit
from .analysis import bulletin17c_analysis, fit_distributions, univariate_analysis

__all__ = [
    "GeneralizedExtremeValue",
    "dgev",
    "pgev",
    "qgev",
    "gev_moments",
    "gev_fit",
    "univariate_analysis",
    "fit_distributions",
    "bulletin17c_analysis",
]


def dgev(x, location=0.0, scale=1.0, shape=0.0):
    """GEV probability density at ``x`` (scalar or array-like)."""
    g = GeneralizedExtremeValue(location, scale, shape)
    return _apply(g.pdf, x)


def pgev(q, location=0.0, scale=1.0, shape=0.0):
    """GEV cumulative distribution at ``q`` (scalar or array-like)."""
    g = GeneralizedExtremeValue(location, scale, shape)
    return _apply(g.cdf, q)


def qgev(p, location=0.0, scale=1.0, shape=0.0):
    """GEV quantile (inverse CDF) at probability ``p`` (scalar or array-like)."""
    g = GeneralizedExtremeValue(location, scale, shape)
    return _apply(g.quantile, p)


def gev_moments(location, scale, shape):
    """Return GEV moments as a dict; undefined moments are ``nan``."""
    g = GeneralizedExtremeValue(location, scale, shape)
    return {
        "mean": g.mean(),
        "median": g.median(),
        "sd": g.standard_deviation(),
        "skewness": g.skewness(),
        "kurtosis": g.kurtosis(),
        "minimum": g.minimum(),
        "maximum": g.maximum(),
    }


def gev_fit(x, method="mle"):
    """Fit a GEV to sample ``x``; returns a dict of location, scale, shape.

    ``method`` is one of ``"mle"``, ``"lmom"``, or ``"mom"``.
    """
    loc, scale, shape = _gev_fit([float(v) for v in np.asarray(x).ravel()], method)
    return {"location": loc, "scale": scale, "shape": shape}


def _apply(fn, values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return fn(float(arr))
    return np.array([fn(float(v)) for v in arr.ravel()]).reshape(arr.shape)
