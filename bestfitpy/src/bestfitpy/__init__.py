"""bestfitpy: Bayesian flood-frequency and extreme-value analysis.

Python bindings to a shared C++ port of the USACE-RMC Numerics / RMC.BestFit
libraries. This early version exposes the Generalized Extreme Value distribution.
"""

from __future__ import annotations

import numpy as np

from ._core import GeneralizedExtremeValue, gev_fit as _gev_fit
from .distributions import Distribution, distribution_names
from .mcmc import mcmc_sample
from .stats import (
    box_cox,
    box_cox_inverse,
    box_cox_lambda,
    latin_hypercube,
    mgbt_test,
    plotting_positions,
    yeo_johnson,
    yeo_johnson_inverse,
    yeo_johnson_lambda,
)
from .analysis import (
    ar_analysis,
    arima_analysis,
    arimax_analysis,
    bivariate_analysis,
    bootstrap_analysis,
    bulletin17c_analysis,
    coincident_frequency_analysis,
    competing_risk_analysis,
    composite_analysis,
    estimation_diagnostics,
    fit_distributions,
    ma_analysis,
    mixture_analysis,
    point_process_analysis,
    posterior_predictive_check,
    prior_predictive_check,
    rating_curve_analysis,
    spatial_gev_analysis,
    univariate_analysis,
)

__all__ = [
    "GeneralizedExtremeValue",
    "Distribution",
    "distribution_names",
    "dgev",
    "pgev",
    "qgev",
    "gev_moments",
    "gev_fit",
    "univariate_analysis",
    "fit_distributions",
    "bulletin17c_analysis",
    "mixture_analysis",
    "competing_risk_analysis",
    "point_process_analysis",
    "ar_analysis",
    "ma_analysis",
    "arima_analysis",
    "arimax_analysis",
    "estimation_diagnostics",
    "composite_analysis",
    "spatial_gev_analysis",
    "bivariate_analysis",
    "coincident_frequency_analysis",
    "rating_curve_analysis",
    "bootstrap_analysis",
    "prior_predictive_check",
    "posterior_predictive_check",
    "mgbt_test",
    "box_cox_lambda",
    "box_cox",
    "box_cox_inverse",
    "yeo_johnson_lambda",
    "yeo_johnson",
    "yeo_johnson_inverse",
    "plotting_positions",
    "latin_hypercube",
    "mcmc_sample",
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
