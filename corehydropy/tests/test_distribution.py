"""Public Distribution API wrapper tests.

Oracle values are NOT hardcoded here: numeric expectations are read from the fixture
JSONs (the single source of truth) and routed through the PUBLIC wrappers, so these
tests prove the wrappers hit the same glue the fixture runner validates.
"""

import json
import math
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from corehydropy import Distribution, distribution_names


def _fixtures_dir() -> Path:
    try:
        packaged = files("corehydropy") / "fixtures"
        if packaged.is_dir():
            return Path(str(packaged))
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    return Path(__file__).resolve().parents[2] / "fixtures"


def test_constructor_validation():
    with pytest.raises(ValueError, match="unknown distribution family"):
        Distribution("NotAFamily", [1, 2])
    with pytest.raises(ValueError, match="expects 2 parameters"):
        Distribution("Normal", [1])
    with pytest.raises(TypeError):
        Distribution(3.14, [1, 2])
    d = Distribution("Normal", [100, 15])
    assert d.family == "Normal"
    assert d.params == [100.0, 15.0]


def test_distribution_names():
    nms = distribution_names()
    for expected in ("Normal", "LogNormal", "Gumbel", "GeneralizedExtremeValue", "Weibull"):
        assert expected in nms
    assert len(nms) == len(set(nms))


def test_public_methods_reproduce_fixture_oracles():
    spec = json.loads(
        (_fixtures_dir() / "distributions" / "univariate" / "gumbel.json").read_text()
    )
    for case in spec["cases"]:
        if "params" not in case["construct"]:
            continue
        d = Distribution(spec["target"], case["construct"]["params"])
        for a in case["assertions"]:
            args = a.get("args", [])
            method = a["method"]
            if method == "pdf":
                actual = d.pdf(args[0])
            elif method == "cdf":
                actual = d.cdf(args[0])
            elif method == "quantile":
                actual = d.quantile(args[0])
            elif method == "mean":
                actual = d.moments()["mean"]
            elif method == "sd":
                actual = d.moments()["sd"]
            elif method == "random_value":
                actual = d.random(int(args[0]), seed=int(args[1]))[int(args[2])]
            else:
                continue
            try:
                expected = float(a["expected"])
            except (TypeError, ValueError):
                continue
            if math.isnan(expected):
                continue
            if a["mode"] == "abs":
                assert abs(actual - expected) <= a["tol"]
            elif a["mode"] == "rel":
                assert abs(actual - expected) / abs(expected) <= a["tol"]


def test_vectorization_and_scalar_passthrough():
    d = Distribution("Normal", [0, 1])
    assert d.pdf([-1, 0, 1]).shape == (3,)
    assert isinstance(d.pdf(0.0), float)
    assert d.cdf(0.0) == 0.5
    assert d.quantile(d.cdf(1.3)) == pytest.approx(1.3)
    assert d.log_pdf(0.7) == pytest.approx(math.log(d.pdf(0.7)))
    xs = [0.0, 1.0, -1.0]
    assert d.log_likelihood(xs) == pytest.approx(sum(d.log_pdf(x) for x in xs))


def test_random_is_seed_deterministic():
    d = Distribution("Gumbel", [100, 10])
    a = d.random(10, seed=7)
    assert np.array_equal(a, d.random(10, seed=7))
    assert not np.array_equal(a, d.random(10, seed=8))
    assert d.random(25, seed=1).shape == (25,)


def test_fit_matches_internal_glue():
    from corehydropy import _core

    d = Distribution("Gumbel", [100, 10])
    x = d.random(200, seed=42)
    f = Distribution.fit("Gumbel", x, method="mle")
    assert isinstance(f, Distribution)
    assert f.params == list(_core.dist_fit("Gumbel", [float(v) for v in x], "mle"))
    with pytest.raises(Exception, match="only MethodOfMoments is supported"):
        Distribution.fit("Deterministic", x, method="mle")


def test_properties_and_repr():
    d = Distribution("Normal", [100, 15])
    names = d.parameter_names
    assert len(names["short"]) == 2 and len(names["full"]) == 2
    assert d.is_valid
    assert not Distribution("Normal", [0, -1]).is_valid
    assert "Normal" in repr(d)
    m = d.moments()
    assert list(m) == ["mean", "median", "mode", "sd", "skewness", "kurtosis",
                       "minimum", "maximum"]


def test_linear_moments_errors_without_support():
    with pytest.raises(Exception, match="no L-moments"):
        Distribution("Cauchy", [0, 1]).linear_moments()
