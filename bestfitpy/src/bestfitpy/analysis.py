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
    analysis_diagnostics_run as _diagnostics_run,
    analysis_family_run as _family_run,
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


# --- D5: per-family analyses + diagnostics -------------------------------------------------
# The seven per-family analyses share the univariate_analysis result surface; each wrapper below
# assembles the matching model spec and calls the single `_family_run` dispatch. Spec assembly and
# seed plumbing match bestfitr/R/analysis.R exactly, so a seeded call returns identical numbers.


def _family_run_dispatch(
    analysis_type, model, data, sampler, iterations, output_length, credible_level, seed,
    exceedance_probabilities, thinning_interval, training_time_steps, forecasting_time_steps,
) -> dict:
    model_json = json.dumps(model)
    ep = [] if exceedance_probabilities is None else [float(v) for v in exceedance_probabilities]
    values = [float(v) for v in np.asarray(data).ravel()]
    return _family_run(
        analysis_type, model_json, values, sampler, int(iterations), int(output_length),
        float(credible_level), int(seed), ep, int(thinning_interval), int(training_time_steps),
        int(forecasting_time_steps),
    )


def mixture_analysis(
    data,
    families,
    zero_inflated: bool = False,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    exceedance_probabilities=None,
    thinning_interval: int = -1,
) -> dict:
    """Bayesian mixture-model frequency analysis.

    Fit a finite mixture of ``families`` (each a distribution family name) to ``data`` with a
    Bayesian MCMC analysis. Returns the same dict shape as :func:`univariate_analysis`
    (``parameters``, ``mode_curve``, ``mean_curve``, ``lower_ci``, ``upper_ci``, ``aic``, ``bic``,
    ``dic``, ``rmse``).
    """
    model = {
        "type": "mixture",
        "families": [str(f) for f in families],
        "zero_inflated": bool(zero_inflated),
        "dataset": "data",
    }
    return _family_run_dispatch(
        "mixture", model, data, sampler, iterations, output_length, credible_level, seed,
        exceedance_probabilities, thinning_interval, -1, -1,
    )


def competing_risk_analysis(
    data,
    families,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    exceedance_probabilities=None,
    thinning_interval: int = -1,
) -> dict:
    """Bayesian competing-risk frequency analysis.

    Fit a competing-risks model (the observed maximum of several independent parent ``families``)
    to ``data``. Returns the same dict shape as :func:`mixture_analysis`.
    """
    model = {
        "type": "competing_risks",
        "families": [str(f) for f in families],
        "dataset": "data",
    }
    return _family_run_dispatch(
        "competing_risk", model, data, sampler, iterations, output_length, credible_level, seed,
        exceedance_probabilities, thinning_interval, -1, -1,
    )


def point_process_analysis(
    data,
    threshold=None,
    total_years=None,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    exceedance_probabilities=None,
    thinning_interval: int = -1,
) -> dict:
    """Bayesian point-process (peaks-over-threshold) frequency analysis.

    Fit a peaks-over-threshold point-process model to ``data``. ``threshold`` and ``total_years``
    are optional (the model default cascade applies when omitted). Returns the same dict shape as
    :func:`mixture_analysis`.
    """
    model = {"type": "point_process", "dataset": "data"}
    if threshold is not None:
        model["threshold"] = float(threshold)
    if total_years is not None:
        model["total_years"] = float(total_years)
    return _family_run_dispatch(
        "point_process", model, data, sampler, iterations, output_length, credible_level, seed,
        exceedance_probabilities, thinning_interval, -1, -1,
    )


def ar_analysis(
    data,
    order_p: int = 1,
    include_intercept: bool = True,
    training_time_steps=None,
    forecasting_time_steps: int = 0,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    thinning_interval: int = -1,
) -> dict:
    """Bayesian autoregressive AR(p) time-series analysis.

    Fit an AR(``order_p``) model to the observed series ``data`` and return the posterior forecast
    curve + credible band. ``training_time_steps`` defaults to the model's own default
    (``max(30, floor(0.8 * n))``), which is invalid for short series -- set it explicitly (e.g.
    ``15``) when ``n`` is small. ``forecasting_time_steps`` extends the horizon past the observed
    series. Returns the same dict shape as :func:`mixture_analysis`.
    """
    model = {
        "type": "time_series",
        "subtype": "ar",
        "dataset": "data",
        "orders": {"p": int(order_p)},
        "include_intercept": bool(include_intercept),
    }
    tts = -1 if training_time_steps is None else int(training_time_steps)
    return _family_run_dispatch(
        "ar", model, data, sampler, iterations, output_length, credible_level, seed, None,
        thinning_interval, tts, forecasting_time_steps,
    )


def ma_analysis(
    data,
    order_q: int = 1,
    include_intercept: bool = True,
    training_time_steps=None,
    forecasting_time_steps: int = 0,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    thinning_interval: int = -1,
) -> dict:
    """Bayesian moving-average MA(q) time-series analysis. See :func:`ar_analysis`."""
    model = {
        "type": "time_series",
        "subtype": "ma",
        "dataset": "data",
        "orders": {"q": int(order_q)},
        "include_intercept": bool(include_intercept),
    }
    tts = -1 if training_time_steps is None else int(training_time_steps)
    return _family_run_dispatch(
        "ma", model, data, sampler, iterations, output_length, credible_level, seed, None,
        thinning_interval, tts, forecasting_time_steps,
    )


def arima_analysis(
    data,
    order_p: int = 1,
    order_d: int = 0,
    order_q: int = 1,
    include_intercept: bool = True,
    training_time_steps=None,
    forecasting_time_steps: int = 0,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    thinning_interval: int = -1,
) -> dict:
    """Bayesian ARIMA(p,d,q) time-series analysis. See :func:`ar_analysis`."""
    model = {
        "type": "time_series",
        "subtype": "arima",
        "dataset": "data",
        "orders": {"p": int(order_p), "d": int(order_d), "q": int(order_q)},
        "include_intercept": bool(include_intercept),
    }
    tts = -1 if training_time_steps is None else int(training_time_steps)
    return _family_run_dispatch(
        "arima", model, data, sampler, iterations, output_length, credible_level, seed, None,
        thinning_interval, tts, forecasting_time_steps,
    )


def arimax_analysis(
    data,
    order_p: int = 1,
    order_d: int = 0,
    order_q: int = 0,
    order_b: int = 0,
    trend: str = "None",
    include_intercept: bool = True,
    training_time_steps=None,
    forecasting_time_steps: int = 0,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    credible_level: float = 0.90,
    seed: int = 12345,
    thinning_interval: int = -1,
) -> dict:
    """Bayesian ARIMAX(p,d,q) time-series analysis with a deterministic ``trend``.

    Covariate forecasting past the observed range is not exposed; run fit-only
    (``forecasting_time_steps = 0``) with covariates. See :func:`ar_analysis` for the shared knobs.
    """
    model = {
        "type": "time_series",
        "subtype": "arimax",
        "dataset": "data",
        "orders": {"p": int(order_p), "d": int(order_d), "q": int(order_q), "b": int(order_b)},
        "include_intercept": bool(include_intercept),
        "trend": str(trend),
    }
    tts = -1 if training_time_steps is None else int(training_time_steps)
    return _family_run_dispatch(
        "arimax", model, data, sampler, iterations, output_length, credible_level, seed, None,
        thinning_interval, tts, forecasting_time_steps,
    )


def estimation_diagnostics(
    data,
    distribution: str,
    sampler: str = "DEMCz",
    iterations: int = 3000,
    output_length: int = 10000,
    seed: int = 12345,
    thinning_interval: int = -1,
    thin_every: int = 10,
) -> dict:
    """Bayesian estimation diagnostics (leverage / influence / prior influence).

    Fit ``distribution`` to ``data`` with a Bayesian MCMC analysis and compute the three
    diagnostics off that fit. Returns a dict with three sub-dicts: ``leverage`` (``index``,
    ``leverage``, ``fit_influence``, ``variance_influence``, ``value`` lists; ``prior_leverage``;
    ``total_leverage``, ``total_fit_influence``, ``total_variance_influence``), ``influence``
    (``pareto_k``, ``elpd_loo`` lists; ``count``, ``mean_pareto_k``, ``max_pareto_k``,
    ``count_pareto_k_above_05``/``_07``/``_10``, ``proportion_problematic``, ``is_reliable``), and
    ``prior_influence`` (``count``, ``prior_precision_share``, ``total_prior_log_likelihood``,
    ``total_data_log_likelihood``, ``prior_to_data_ratio``, ``is_prior_influential``,
    ``mean_prior_precision_share``).
    """
    model_json = json.dumps({"family": distribution, "dataset": "data"})
    values = [float(v) for v in np.asarray(data).ravel()]
    return _diagnostics_run(
        model_json, values, sampler, int(iterations), int(output_length), int(seed),
        int(thinning_interval), int(thin_every),
    )
