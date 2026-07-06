"""User-facing analysis wrappers (A10).

Thin Python wrappers over the compiled `_core` analysis bindings, mirroring the R package's
`univariate_analysis` / `fit_distributions` / `bulletin17c_analysis` signatures and semantics.
Because both packages call the identical compiled core with a bit-exact Mersenne Twister, a seeded
call returns identical numbers in either language, so the spec assembly and seed plumbing here
match bestfitr/R/analysis.R exactly.
"""

from __future__ import annotations

import json

import numpy as np

from ._core import (
    analysis_b17c_run as _b17c_run,
    analysis_fit_distributions as _fit_distributions,
    analysis_univariate_run as _univariate_run,
)


def univariate_analysis(
    data,
    distribution: str,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    exceedance_probabilities=None,
    thinning_interval: int = -1,
) -> dict:
    """Bayesian univariate frequency analysis.

    Fit ``distribution`` to ``data`` with a Bayesian MCMC analysis and return the frequency
    (quantile) curve, the posterior mean and credible band, and goodness-of-fit scalars.

    ``sampler`` is one of ``"DEMCz"``, ``"DEMCzs"``, ``"ARWMH"``, ``"NUTS"``. ``thinning_interval``
    of ``-1`` (default) keeps the sampler's own default. The MCMC warmup (burn-in) length is set
    automatically to ``max(50, iterations // 2)`` and is not a user parameter. Returns a dict with
    ``parameters``, ``mode_curve``, ``mean_curve``, ``lower_ci``, ``upper_ci`` (one value per
    exceedance ordinate) and the scalars ``aic``, ``bic``, ``dic``, ``rmse``.
    """
    model_json = json.dumps({"family": distribution, "dataset": "data"})
    ep = [] if exceedance_probabilities is None else [float(v) for v in exceedance_probabilities]
    values = [float(v) for v in np.asarray(data).ravel()]
    return _univariate_run(
        model_json, values, sampler, int(iterations), int(output_length),
        float(credible_level), int(seed), ep, int(thinning_interval),
    )


def fit_distributions(data) -> dict:
    """Fit and rank the 14 candidate distributions by maximum likelihood.

    Returns a dict with equal-length lists ``distribution`` (candidate name), ``aic``, ``bic``,
    ``rmse``, and ``converged`` (bool) -- one entry per candidate. Ranking is left to the caller.
    """
    values = [float(v) for v in np.asarray(data).ravel()]
    return _fit_distributions(values)


def bulletin17c_analysis(
    data,
    uncertainty_method: str = "MultivariateNormal",
    output_length: int = 10000,
    seed: int = 12345,
    confidence_level: float = 0.90,
    exceedance_probabilities=None,
) -> dict:
    """Bulletin 17C (log-Pearson Type III) flood-frequency analysis.

    Fit the model by the generalized method of moments and return the Cohn-style delta-method
    confidence intervals, the fitted parameters, and the sandwich covariance.

    ``uncertainty_method`` is ``"MultivariateNormal"`` (default) or ``"Bootstrap"``; the
    ``"LinkedMultivariateNormal"`` / ``"BiasCorrectedBootstrap"`` methods are deferred and raise.
    Returns a dict with ``exceedance_probabilities``, ``point_estimates``, ``lower_ci``,
    ``upper_ci``, ``confidence_level``, ``beta1``, ``nu``, ``quantile_variance``, ``parameters``,
    and ``covariance`` (a nested p x p list).
    """
    model_json = json.dumps(
        {"type": "bulletin17c", "family": "LogPearsonTypeIII", "dataset": "data"}
    )
    ep = [] if exceedance_probabilities is None else [float(v) for v in exceedance_probabilities]
    values = [float(v) for v in np.asarray(data).ravel()]
    return _b17c_run(
        model_json, values, uncertainty_method, int(output_length), int(seed),
        float(confidence_level), ep,
    )
