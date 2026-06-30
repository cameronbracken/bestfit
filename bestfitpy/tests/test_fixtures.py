"""Generic, fixture-driven validation for bestfitpy.

Reads the language-neutral oracle fixtures (the single source of truth shared with
the C++ core and the R package) and checks every assertion. No oracle values live
in this file -- only the dispatch from fixture method names to the Python API.
"""
from __future__ import annotations

import json
import math
from importlib.resources import files
from pathlib import Path

import pytest

from bestfitpy import GeneralizedExtremeValue, gev_fit


def _fixtures_dir() -> Path:
    # Prefer the copy shipped inside the installed package; fall back to the repo
    # canonical fixtures/ for in-tree development.
    try:
        packaged = files("bestfitpy") / "fixtures"
        if packaged.is_dir():
            return Path(str(packaged))
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    return Path(__file__).resolve().parents[2] / "fixtures"


def _num(v):
    if isinstance(v, str):
        return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}[v]
    return float(v)


def _build(target, construct, datasets):
    if target != "GeneralizedExtremeValue":
        pytest.skip(f"no Python builder for target {target}")
    if "params" in construct:
        p = [_num(v) for v in construct["params"]]
        return GeneralizedExtremeValue(*p)
    fit = construct["fit"]
    res = gev_fit(datasets[fit["dataset"]], fit["method"])
    return GeneralizedExtremeValue(res["location"], res["scale"], res["shape"])


def _dispatch(g, method, args):
    simple = {
        "mean": g.mean, "median": g.median, "mode": g.mode, "skewness": g.skewness,
        "kurtosis": g.kurtosis, "minimum": g.minimum, "maximum": g.maximum,
        "sd": g.standard_deviation,
    }
    if method in simple:
        return simple[method]()
    if method in ("pdf", "cdf", "quantile"):
        return getattr(g, method)(args[0])
    if method == "parameters_valid":
        return g.parameters_valid
    if method == "param":
        return {"location": g.location, "scale": g.scale, "shape": g.shape}[args[0]]
    if method == "linear_moment":
        return g.linear_moments_from_parameters([g.location, g.scale, g.shape])[int(args[0])]
    if method == "quantile_gradient":
        return g.quantile_gradient(args[0])[int(args[1])]
    if method == "parameter_covariance":
        return g.parameter_covariance(int(args[0]))[int(args[1])][int(args[2])]
    if method == "quantile_variance":
        return g.quantile_variance(args[0], int(args[1]))
    if method == "quantile_se":
        return math.sqrt(g.quantile_variance(args[0], int(args[1])))
    raise KeyError(f"unknown fixture method: {method}")


def _check(actual, a):
    mode, exp = a["mode"], a.get("expected")
    if mode == "bool":
        assert bool(actual) is bool(exp)
    elif mode == "equal":
        e = _num(exp)
        if math.isnan(e):
            assert math.isnan(actual)
        else:
            assert actual == e
    elif mode == "abs":
        assert abs(actual - exp) <= a["tol"]
    elif mode == "rel":
        assert abs(actual - exp) / abs(exp) <= a["tol"]
    else:
        raise KeyError(f"unknown comparison mode: {mode}")


def _load_cases():
    out = []
    for fx in sorted(_fixtures_dir().rglob("*.json")):
        spec = json.loads(fx.read_text())
        for case in spec["cases"]:
            out.append((spec["target"], spec.get("datasets", {}), case))
    return out


CASES = _load_cases()


@pytest.mark.parametrize(
    "target,datasets,case", CASES, ids=[f"{t}:{c['name']}" for t, _, c in CASES]
)
def test_fixture_case(target, datasets, case):
    g = _build(target, case["construct"], datasets)
    for a in case["assertions"]:
        actual = _dispatch(g, a["method"], a.get("args", []))
        _check(actual, a)
