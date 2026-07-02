"""Generic, fixture-driven validation for bestfitpy.

Reads the language-neutral oracle fixtures (the single source of truth shared with
the C++ core and the R package) and checks every assertion. No oracle values live
in this file -- only the dispatch from fixture method names to the Python API. The GEV
slice uses its bespoke object API; every other distribution goes through the polymorphic
``_core.dist_*`` functions (factory + UnivariateDistributionBase).
"""
from __future__ import annotations

import json
import math
from importlib.resources import files
from pathlib import Path

import pytest

from bestfitpy import GeneralizedExtremeValue, gev_fit
from bestfitpy import _core

_MOMENTS = ("mean", "median", "mode", "sd", "skewness", "kurtosis", "minimum", "maximum")


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


# --- GEV slice (bespoke object API) ----------------------------------------------------


def _build_gev(construct, datasets):
    if "params" in construct:
        return GeneralizedExtremeValue(*[_num(v) for v in construct["params"]])
    fit = construct["fit"]
    res = gev_fit(datasets[fit["dataset"]], fit["method"])
    return GeneralizedExtremeValue(res["location"], res["scale"], res["shape"])


def _dispatch_gev(g, method, args):
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


# --- Composite distribution path -------------------------------------------------------
# For TruncatedDistribution (and future Empirical/KernelDensity/Mixture/CompetingRisks),
# the fixture "construct" uses a structured schema rather than flat "params".
# Adding a new composite = one new case in _build_composite + one in _dispatch_composite.

_COMPOSITE_TARGETS = {"TruncatedDistribution", "Empirical", "KernelDensity", "Mixture"}


def _build_composite(target: str, construct: dict, datasets: dict | None = None) -> dict:
    """Parse a composite construct into a dict that _dispatch_composite can use."""
    if target == "TruncatedDistribution":
        base_target = construct["base"]["target"]
        base_params = [_num(v) for v in construct["base"]["params"]]
        lo, hi = float(construct["bounds"][0]), float(construct["bounds"][1])
        return {"base_target": base_target, "base_params": base_params, "lo": lo, "hi": hi}
    if target == "Empirical":
        xv = [float(v) for v in construct["x"]]
        pv = [float(v) for v in construct["p"]]
        pt = construct.get("p_transform", "NormalZ")
        return {"x_vals": xv, "p_vals": pv, "p_transform": pt}
    if target == "KernelDensity":
        ds = datasets or {}
        data_key = construct["data"]
        data_vec = [float(v) for v in ds[data_key]]
        kernel = construct.get("kernel", "Gaussian")
        bandwidth = float(construct["bandwidth"]) if "bandwidth" in construct else -1.0
        bounded = bool(construct.get("bounded_by_data", True))
        return {"data_vec": data_vec, "kernel": kernel, "bandwidth": bandwidth, "bounded": bounded}
    if target == "Mixture":
        comp_targets = [c["target"] for c in construct["components"]]
        comp_params = [[float(v) for v in c["params"]] for c in construct["components"]]
        wts = [float(w) for w in construct["weights"]]
        return {"comp_targets": comp_targets, "comp_params": comp_params, "weights": wts}
    raise KeyError(f"unknown composite target: {target}")


def _dispatch_composite(target: str, cd: dict, method: str, args: list):
    if target == "TruncatedDistribution":
        bt, bp, lo, hi = cd["base_target"], cd["base_params"], cd["lo"], cd["hi"]
        if method in _MOMENTS:
            return _core.trunc_moments(bt, bp, lo, hi)[method]
        if method == "pdf":
            return _core.trunc_pdf(bt, bp, lo, hi, args[0])
        if method == "cdf":
            return _core.trunc_cdf(bt, bp, lo, hi, args[0])
        if method == "quantile":
            return _core.trunc_quantile(bt, bp, lo, hi, args[0])
        if method == "parameters_valid":
            return _core.trunc_valid(bt, bp, lo, hi)
        raise KeyError(f"unknown fixture method for TruncatedDistribution: {method}")
    if target == "Empirical":
        xv, pv, pt = cd["x_vals"], cd["p_vals"], cd["p_transform"]
        if method in _MOMENTS:
            return _core.emp_moments(xv, pv, pt)[method]
        if method == "pdf":
            return _core.emp_pdf(xv, pv, pt, args[0])
        if method == "cdf":
            return _core.emp_cdf(xv, pv, pt, args[0])
        if method == "quantile":
            return _core.emp_quantile(xv, pv, pt, args[0])
        if method == "parameters_valid":
            return _core.emp_valid(xv, pv, pt)
        raise KeyError(f"unknown fixture method for Empirical: {method}")
    if target == "KernelDensity":
        dv, ker, bw, bd = cd["data_vec"], cd["kernel"], cd["bandwidth"], cd["bounded"]
        if method in _MOMENTS:
            return _core.kde_moments(dv, ker, bw, bd)[method]
        if method == "pdf":
            return _core.kde_pdf(dv, ker, bw, bd, args[0])
        if method == "cdf":
            return _core.kde_cdf(dv, ker, bw, bd, args[0])
        if method == "quantile":
            return _core.kde_quantile(dv, ker, bw, bd, args[0])
        if method == "parameters_valid":
            return _core.kde_valid(dv, ker, bw, bd)
        raise KeyError(f"unknown fixture method for KernelDensity: {method}")
    if target == "Mixture":
        ct, cp, wts = cd["comp_targets"], cd["comp_params"], cd["weights"]
        if method in _MOMENTS:
            return _core.mix_moments(ct, cp, wts)[method]
        if method == "pdf":
            return _core.mix_pdf(ct, cp, wts, args[0])
        if method == "cdf":
            return _core.mix_cdf(ct, cp, wts, args[0])
        if method == "quantile":
            return _core.mix_quantile(ct, cp, wts, args[0])
        if method == "parameters_valid":
            return _core.mix_valid(ct, cp, wts)
        raise KeyError(f"unknown fixture method for Mixture: {method}")
    raise KeyError(f"unknown composite target: {target}")


# --- Generic polymorphic path ----------------------------------------------------------


def _build_params(target, construct, datasets):
    if "params" in construct:
        return [_num(v) for v in construct["params"]]
    fit = construct["fit"]
    return list(_core.dist_fit(target, datasets[fit["dataset"]], fit["method"]))


def _dispatch_generic(target, params, method, args):
    if method in _MOMENTS:
        return _core.dist_moments(target, params)[method]
    if method == "pdf":
        return _core.dist_pdf(target, params, args[0])
    if method == "cdf":
        return _core.dist_cdf(target, params, args[0])
    if method == "quantile":
        return _core.dist_quantile(target, params, args[0])
    if method == "parameters_valid":
        return _core.dist_valid(target, params)
    if method == "param":
        return params[int(args[0])]
    if method == "linear_moment":
        return _core.dist_linear_moments(target, params)[int(args[0])]
    raise KeyError(f"unknown fixture method: {method}")


# --- Shared assertion checking ---------------------------------------------------------


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
        # Only validate univariate_distribution fixtures; skip other kinds (e.g. special_function)
        # which are validated in C++ only and are not exposed to the Python package.
        if spec.get("kind") != "univariate_distribution":
            continue
        for case in spec["cases"]:
            out.append((spec["target"], spec.get("datasets", {}), case))
    return out


CASES = _load_cases()


@pytest.mark.parametrize(
    "target,datasets,case", CASES, ids=[f"{t}:{c['name']}" for t, _, c in CASES]
)
def test_fixture_case(target, datasets, case):
    is_gev = target == "GeneralizedExtremeValue"
    is_composite = target in _COMPOSITE_TARGETS
    if is_gev:
        g = _build_gev(case["construct"], datasets)
    elif is_composite:
        cd = _build_composite(target, case["construct"], datasets)
    else:
        params = _build_params(target, case["construct"], datasets)
    for a in case["assertions"]:
        args = a.get("args", [])
        if is_gev:
            actual = _dispatch_gev(g, a["method"], args)
        elif is_composite:
            actual = _dispatch_composite(target, cd, a["method"], args)
        else:
            actual = _dispatch_generic(target, params, a["method"], args)
        _check(actual, a)
