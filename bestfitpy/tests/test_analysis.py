"""Smoke tests for the exported user-facing analysis functions (A10).

These prove the univariate_analysis / fit_distributions / bulletin17c_analysis wrappers are
callable end-to-end and return the documented dict shapes with finite / monotone invariants.
Exact oracle values live in fixtures/ and are checked by test_fixtures.py; these assert shape.
"""

from __future__ import annotations

import math

import pytest

import bestfitpy

SMOKE_PEAKS = [
    12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600,
    19200, 13800, 25600, 10500, 16900, 21300, 14700, 8200, 23800, 15900,
]


def test_univariate_analysis_returns_frequency_curve():
    res = bestfitpy.univariate_analysis(
        SMOKE_PEAKS,
        distribution="Normal",
        sampler="DEMCzs",
        iterations=100,
        output_length=400,
        credible_level=0.90,
        seed=12345,
    )
    assert len(res["parameters"]) == 2
    assert all(math.isfinite(v) for v in res["parameters"])
    assert math.isfinite(res["aic"])
    assert math.isfinite(res["dic"])
    n = len(res["mode_curve"])
    assert n > 0
    assert len(res["mean_curve"]) == n
    assert len(res["lower_ci"]) == n
    assert len(res["upper_ci"]) == n
    # Ascending exceedance ordinates -> descending mode (quantile) curve.
    assert all(b <= a + 1e-6 for a, b in zip(res["mode_curve"], res["mode_curve"][1:]))
    # Each credible band brackets the mode curve.
    assert all(lo <= m + 1e-6 for lo, m in zip(res["lower_ci"], res["mode_curve"]))
    assert all(hi >= m - 1e-6 for hi, m in zip(res["upper_ci"], res["mode_curve"]))


def test_fit_distributions_ranks_14_candidates():
    res = bestfitpy.fit_distributions(SMOKE_PEAKS)
    assert len(res["distribution"]) == 14
    assert len(res["aic"]) == 14
    assert any(res["converged"])
    for i, ok in enumerate(res["converged"]):
        if ok:
            assert math.isfinite(res["aic"][i])
            assert math.isfinite(res["bic"][i])
            assert math.isfinite(res["rmse"][i])


def test_bulletin17c_analysis_returns_cohn_intervals():
    res = bestfitpy.bulletin17c_analysis(
        SMOKE_PEAKS,
        uncertainty_method="MultivariateNormal",
        output_length=200,
        seed=12345,
        confidence_level=0.90,
    )
    assert res["confidence_level"] == 0.90
    n = len(res["exceedance_probabilities"])
    assert n > 0
    assert len(res["point_estimates"]) == n
    assert len(res["lower_ci"]) == n
    assert len(res["upper_ci"]) == n
    assert all(math.isfinite(v) for v in res["point_estimates"])
    assert all(lo <= hi for lo, hi in zip(res["lower_ci"], res["upper_ci"]))
    assert len(res["parameters"]) == 3  # LP3: location / scale / shape


def test_bulletin17c_analysis_rejects_deferred_methods():
    with pytest.raises(Exception):
        bestfitpy.bulletin17c_analysis(
            SMOKE_PEAKS, uncertainty_method="LinkedMultivariateNormal"
        )
