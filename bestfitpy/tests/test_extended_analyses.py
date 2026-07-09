"""Smoke tests for the X11 exported analysis functions (composite / spatial_gev / bivariate /
coincident / rating_curve / bootstrap / prior + posterior predictive). These prove each wrapper is
callable end-to-end and returns the documented dict shape; exact oracle values live in fixtures/
and are checked by test_fixtures.py.
"""

from __future__ import annotations

import math

import bestfitpy

FLOW = [
    12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600,
    19200, 13800, 25600, 10500, 16900, 21300, 14700, 8200, 23800, 15900,
]

BIVAR_X = [-1.2, 0.3, 0.8, -0.5, 1.1, -0.9, 0.2, 1.5, -1.8, 0.6, -0.1, 0.9, -0.7, 1.3, -0.3]
BIVAR_Y = [-0.8, 0.5, 1.0, -0.3, 0.9, -1.1, 0.4, 1.2, -1.5, 0.7, 0.1, 1.1, -0.6, 1.4, -0.2]


def test_composite_analysis_returns_curve():
    res = bestfitpy.composite_analysis(
        FLOW, families=["Normal", "Normal"], composite_type="CompetingRisks",
        sampler="DEMCzs", iterations=100, output_length=200, seed=12345,
        thinning_interval=1, exceedance_probabilities=[0.01, 0.1, 0.5, 0.9, 0.99],
    )
    assert len(res["mode_curve"]) == 5
    assert len(res["lower_ci"]) == 5


def test_spatial_gev_analysis_returns_site_results():
    coords = [[0.0, 0.0], [1.0, 0.5], [0.5, 1.0]]
    at_site = [
        [20, 22, 18], [25, 26, 24], [18, 19, 17], [30, 31, 28], [22, 23, 21],
        [28, 29, 27], [35, 34, 33], [24, 25, 23], [26, 27, 25], [40, 38, 37],
        [21, 22, 20], [27, 28, 26], [33, 32, 31], [23, 24, 22], [29, 30, 28],
    ]
    res = bestfitpy.spatial_gev_analysis(
        coords, at_site, sampler="DEMCzs", iterations=300, output_length=400,
        number_of_chains=4, seed=12345, thinning_interval=1,
    )
    assert res["site_count"] == 3
    assert len(res["site_location_mean"]) == 3
    assert len(res["mode_curve"]) > 0


def test_bivariate_analysis_returns_joint_curve():
    res = bestfitpy.bivariate_analysis(
        "Normal", BIVAR_X, [0.0, 1.0], "Normal", BIVAR_Y, [0.0, 1.0],
        xy_x=[-1.5, -0.5, 0.5, 1.5, 2.0], xy_y=[-1.5, -0.5, 0.5, 1.5, 2.0],
        copula="Normal", sampler="DEMCzs", iterations=300, output_length=400,
        number_of_chains=4, seed=12345, thinning_interval=1,
    )
    assert len(res["mode_curve"]) == 5


def test_coincident_frequency_analysis_returns_aep_curve():
    x = [-2.0, -1.0, 0.0, 1.0, 2.0]
    response = [[xi + yj for yj in x] for xi in x]
    res = bestfitpy.coincident_frequency_analysis(
        "Normal", BIVAR_X, [0.0, 1.0], "Normal", BIVAR_Y, [0.0, 1.0],
        x_values=x, y_values=x, response=response, number_of_bins=5,
        copula="Normal", sampler="DEMCzs", iterations=300, output_length=400,
        number_of_chains=4, seed=12345, thinning_interval=1,
    )
    assert len(res["z_output_values"]) == 5
    assert len(res["mode_curve"]) == 5


def test_rating_curve_analysis_returns_curve():
    res = bestfitpy.rating_curve_analysis(
        stage=[1.0, 1.2, 1.5, 1.8, 2.0, 2.3, 2.6, 2.9, 3.1, 3.4, 3.7, 4.0, 4.3, 4.6, 5.0],
        discharge=[5.0, 7.1, 11.0, 16.2, 20.5, 28.0, 37.1, 47.9, 55.6, 68.0, 82.3, 98.1, 115.4, 134.8, 160.2],
        stage_bins=20, sampler="DEMCzs", iterations=300, output_length=400,
        number_of_chains=4, seed=12345, thinning_interval=1,
    )
    assert len(res["mode_curve"]) == 20


def test_bootstrap_analysis_returns_bands():
    res = bestfitpy.bootstrap_analysis(
        FLOW[:10], distribution="Normal", probabilities=[0.5, 0.9, 0.99],
        estimation_method="MaximumLikelihood", sample_size=30, replications=200, seed=12345,
    )
    assert len(res["mode_curve"]) == 3
    assert len(res["lower_ci"]) == 3
    assert math.isfinite(res["mode_curve"][0])


def test_prior_predictive_check_returns_summary():
    res = bestfitpy.prior_predictive_check(
        FLOW[:10], distribution="Normal", number_of_draws=200, sample_size=30, seed=12345,
    )
    assert res["number_of_valid_draws"] == 200
    assert len(res["summary_mean_quantiles"]) == 5


def test_posterior_predictive_check_returns_pvalues():
    res = bestfitpy.posterior_predictive_check(
        FLOW[:10], distribution="Normal", sampler="DEMCzs", iterations=100,
        output_length=200, seed=12345, thinning_interval=1, number_of_replicates=200,
    )
    assert res["number_of_replicates"] == 200
    assert 0.0 <= res["mean_p_value"] <= 1.0
