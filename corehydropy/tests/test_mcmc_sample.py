"""mcmc_sample() wrapper tests: structural checks plus seed determinism.

Cross-language identity with corehydror is proven by the mcmc_sampler fixtures; here we
check the public wrapper surface only, with no hardcoded oracle values.
"""

import numpy as np
import pytest

import corehydropy as ch

RESULT_KEYS = {
    "parameters", "chains", "acceptance_rates", "map", "map_fitness",
    "posterior_mean", "posterior_sd", "posterior_median",
    "posterior_lower_ci", "posterior_upper_ci", "rhat", "ess",
}


def test_result_is_complete_and_consistent():
    x = ch.Distribution("Normal", [100, 15]).random(100, seed=42)
    fit = ch.mcmc_sample(x, "Normal", sampler="DEMCzs", iterations=1000, seed=12345)
    assert set(fit) == RESULT_KEYS
    assert len(fit["parameters"]) == 2
    assert fit["map"].shape == (2,)
    assert len(fit["chains"]) > 0
    assert fit["chains"][0].shape[1] == 2
    assert abs(fit["posterior_mean"][0] - 100) < 10
    assert abs(fit["posterior_mean"][1] - 15) < 10
    assert np.all(fit["posterior_lower_ci"] < fit["posterior_upper_ci"])
    assert np.all((fit["rhat"] > 0.9) & (fit["rhat"] < 1.2))


def test_seed_determinism():
    x = ch.Distribution("Gumbel", [100, 10]).random(80, seed=9)
    a = ch.mcmc_sample(x, "Gumbel", sampler="RWMH", iterations=500, seed=12345)
    b = ch.mcmc_sample(x, "Gumbel", sampler="RWMH", iterations=500, seed=12345)
    assert np.array_equal(a["posterior_mean"], b["posterior_mean"])
    assert all(np.array_equal(ca, cb) for ca, cb in zip(a["chains"], b["chains"]))
    c = ch.mcmc_sample(x, "Gumbel", sampler="RWMH", iterations=500, seed=999)
    assert not np.array_equal(a["posterior_mean"], c["posterior_mean"])


def test_argument_validation():
    x = ch.Distribution("Normal", [0, 1]).random(30, seed=1)
    with pytest.raises(ValueError, match="unknown sampler"):
        ch.mcmc_sample(x, "Normal", sampler="Gibbs")
    with pytest.raises(Exception):
        ch.mcmc_sample(x, "Normal", initialize="Nope")
