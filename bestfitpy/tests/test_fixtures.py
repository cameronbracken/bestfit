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

_COMPOSITE_TARGETS = {"TruncatedDistribution", "Empirical", "KernelDensity", "Mixture",
                      "CompetingRisks"}


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
    if target == "CompetingRisks":
        comp_targets = [c["target"] for c in construct["components"]]
        comp_params = [[float(v) for v in c["params"]] for c in construct["components"]]
        min_of_rv = bool(construct.get("minimum_of_random_variables", True))
        dependency = construct.get("dependency", "Independent")
        correlation = [[float(v) for v in row] for row in construct.get("correlation", [])]
        return {"comp_targets": comp_targets, "comp_params": comp_params,
                "minimum_of_rv": min_of_rv, "dependency": dependency,
                "correlation": correlation}
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
    if target == "CompetingRisks":
        ct, cp, min_rv = cd["comp_targets"], cd["comp_params"], cd["minimum_of_rv"]
        dep, corr = cd["dependency"], cd["correlation"]
        if method in _MOMENTS:
            return _core.cr_moments(ct, cp, min_rv, dep, corr)[method]
        if method == "pdf":
            return _core.cr_pdf(ct, cp, min_rv, dep, corr, args[0])
        if method == "log_pdf":
            return _core.cr_log_pdf(ct, cp, min_rv, dep, corr, args[0])
        if method == "cdf":
            return _core.cr_cdf(ct, cp, min_rv, dep, corr, args[0])
        if method == "quantile":
            return _core.cr_quantile(ct, cp, min_rv, dep, corr, args[0])
        if method == "parameters_valid":
            return _core.cr_valid(ct, cp, min_rv, dep, corr)
        raise KeyError(f"unknown fixture method for CompetingRisks: {method}")
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


# --- multivariate_distribution path -----------------------------------------------------
# Dirichlet/Multinomial/BivariateEmpirical dispatch to the bespoke _core.dirichlet_val/
# _core.multinomial_val/_core.bve_cdf functions (method + flat numeric args in, double
# out). Extensible: additional multivariate targets add a branch here plus their own
# _core.<name>_val entry point.

def _flatten_mv_args(args: list) -> list[float]:
    """Flattens fixture assertion args to a flat float list.

    Handles every convention seen in the multivariate fixtures: a single nested vector
    argument (e.g. pdf args = [[0.3, 0.4, 0.3]]), flat scalar args (e.g. covariance args =
    [0, 1], log_multivariate_beta args = [1.0, 1.0]), and a nested vector followed by a
    trailing scalar (e.g. MultivariateNormal inverse_cdf args = [[p1, p2], index]).
    """
    out: list[float] = []
    for v in args:
        if isinstance(v, list):
            out.extend(float(x) for x in v)
        else:
            out.append(float(v))
    return out


def _dispatch_multivariate(target: str, construct: dict, method: str, args: list):
    ar = _flatten_mv_args(args)
    if target == "Dirichlet":
        alpha = [float(v) for v in construct["alpha"]]
        return _core.dirichlet_val(method, alpha, ar)
    if target == "Multinomial":
        n = int(construct["n"])
        p = [float(v) for v in construct["p"]]
        return _core.multinomial_val(method, n, p, ar)
    if target == "BivariateEmpirical":
        x1 = [float(v) for v in construct["x1"]]
        x2 = [float(v) for v in construct["x2"]]
        p = [[float(v) for v in row] for row in construct["p"]]
        transforms = [
            construct.get("x1_transform", "None"),
            construct.get("x2_transform", "None"),
            construct.get("p_transform", "None"),
        ]
        return _core.bve_cdf(method, x1, x2, p, transforms, ar)
    if target == "MultivariateNormal":
        mean = [float(v) for v in construct["mean"]]
        cov = [[float(v) for v in row] for row in construct["covariance"]]
        return _core.mvn_val(method, mean, cov, ar)
    if target == "MultivariateStudentT":
        df = float(construct["df"])
        location = [float(v) for v in construct["location"]]
        scale = [[float(v) for v in row] for row in construct["scale"]]
        return _core.mvt_val(method, df, location, scale, ar)
    raise KeyError(f"unknown multivariate target: {target}")


# --- MultivariateNormal seeded batches --------------------------------------------------
# `cdf` (dim>=3), `interval`, and `mvndst` all draw from the seeded MVNUNI stream, so a
# RUN of consecutive same-method assertions in a seeded case must be evaluated on ONE
# persistent instance via the mvn_*_seq bindings in mvd.cpp, not dispatched one call at a
# time (which would silently reset the seed between assertions). _run_mvn_case() below
# groups consecutive assertions of these methods and batches them; everything else (and
# every case without a "seed") falls through to the stateless per-assertion dispatch
# above, unchanged.

_MVN_SEEDED_METHODS = ("cdf", "mvndst", "interval")


def _dispatch_mvn_seeded_seq(construct: dict, method: str, run: list):
    seed = int(construct["seed"])
    mean = [float(v) for v in construct["mean"]]
    cov = [[float(v) for v in row] for row in construct["covariance"]]

    if method == "cdf":
        xs = [[_num(v) for v in a["args"][0]] for a in run]
        return _core.mvn_cdf_seq(mean, cov, seed, xs)
    if method == "interval":
        lowers = [[_num(v) for v in a["args"][0]] for a in run]
        uppers = [[_num(v) for v in a["args"][1]] for a in run]
        return _core.mvn_interval_seq(mean, cov, seed, lowers, uppers)
    if method == "mvndst":
        # args = [n, [lower...], [upper...], [infin...], [correl...], maxpts, abseps, releps]
        n_dim = int(run[0]["args"][0])
        lowers = [[_num(v) for v in a["args"][1]] for a in run]
        uppers = [[_num(v) for v in a["args"][2]] for a in run]
        infins = [[int(v) for v in a["args"][3]] for a in run]
        correls = [[_num(v) for v in a["args"][4]] for a in run]
        maxpts_v = [int(a["args"][5]) for a in run]
        abseps_v = [float(a["args"][6]) for a in run]
        releps_v = [float(a["args"][7]) for a in run]
        return _core.mvn_mvndst_seq(n_dim, seed, lowers, uppers, infins, correls, maxpts_v, abseps_v,
                                     releps_v)
    raise KeyError(f"unknown seeded MultivariateNormal method: {method}")


def _run_mvn_case(construct: dict, assertions: list):
    seeded = "seed" in construct
    i = 0
    n = len(assertions)
    while i < n:
        a = assertions[i]
        method = a["method"]
        if seeded and method in _MVN_SEEDED_METHODS:
            j = i
            while j < n and assertions[j]["method"] == method:
                j += 1
            run = assertions[i:j]
            actuals = _dispatch_mvn_seeded_seq(construct, method, run)
            for actual, assertion in zip(actuals, run):
                _check(actual, assertion)
            i = j
        else:
            args = a.get("args", [])
            actual = _dispatch_multivariate("MultivariateNormal", construct, method, args)
            _check(actual, a)
            i += 1


# --- mcmc_sampler path -------------------------------------------------------------------
# Inherently STATEFUL (unlike multivariate_distribution's/bivariate_copula's per-assertion
# dispatch): one _core.mcmc_run call per case builds the model via the registry, configures
# the sampler from construct["settings"], and samples() ONCE; every assertion in the case
# reads the single returned dict. See fixtures/README.md's mcmc_sampler schema for the full
# method list and tolerance policy.


def _dispatch_mcmc(result: dict, method: str, args: list):
    if method == "posterior_mean":
        return result["posterior_mean"][int(args[0])]
    if method == "posterior_sd":
        return result["posterior_sd"][int(args[0])]
    if method == "posterior_median":
        return result["posterior_median"][int(args[0])]
    if method == "posterior_lower_ci":
        return result["posterior_lower_ci"][int(args[0])]
    if method == "posterior_upper_ci":
        return result["posterior_upper_ci"][int(args[0])]
    if method == "chain_value":
        return result["chains"][int(args[0])][int(args[1])][int(args[2])]
    if method == "chain_fitness":
        return result["chain_fitness"][int(args[0])][int(args[1])]
    if method == "map_value":
        return result["map_values"][int(args[0])]
    if method == "map_fitness":
        return result["map_fitness"]
    if method == "acceptance_rate":
        return result["acceptance_rates"][int(args[0])]
    if method == "mean_log_likelihood":
        return result["mean_log_likelihood"][int(args[0])]
    if method == "rhat":
        return result["rhat"][int(args[0])]
    if method == "ess":
        return result["ess"][int(args[0])]
    raise KeyError(f"unknown mcmc_sampler fixture method: {method}")


def _run_mcmc_case(target: str, construct: dict, assertions: list, datasets: dict):
    model = construct["model"]
    data = [float(v) for v in datasets[model["dataset"]]]
    settings = construct.get("settings", {})
    result = _core.mcmc_run(target, model["name"], model["family"], data, settings)
    for a in assertions:
        args = a.get("args", [])
        actual = _dispatch_mcmc(result, a["method"], args)
        _check(actual, a)


# --- bootstrap path ----------------------------------------------------------------------
# Inherently STATEFUL like mcmc_sampler: one _core.bootstrap_run call per case builds the
# model via the registry, runs() (or run_with_studentized_bootstrap()) ONCE, and computes
# confidence intervals ONCE; every assertion in the case reads the single returned dict. See
# fixtures/README.md's bootstrap schema for the full method list and tolerance policy.


def _dispatch_bootstrap(result: dict, method: str, args: list):
    if method == "statistic_lower_ci":
        return result["statistic_lower_ci"][int(args[0])]
    if method == "statistic_upper_ci":
        return result["statistic_upper_ci"][int(args[0])]
    if method == "parameter_lower_ci":
        return result["parameter_lower_ci"][int(args[0])]
    if method == "parameter_upper_ci":
        return result["parameter_upper_ci"][int(args[0])]
    if method == "population_estimate":
        return result["population_estimate"][int(args[0])]
    if method == "valid_count":
        return result["valid_count"][int(args[0])]
    if method == "replicate_value":
        return result["replicate_values"][int(args[0])][int(args[1])]
    raise KeyError(f"unknown bootstrap fixture method: {method}")


def _run_bootstrap_case(construct: dict, assertions: list, datasets: dict):
    dataset = [float(v) for v in datasets[construct["dataset"]]] if "dataset" in construct else []
    probabilities = [_num(v) for v in construct["probabilities"]]
    result = _core.bootstrap_run(
        construct["model"],
        construct.get("mu", 0.0),
        construct.get("sigma", 0.0),
        construct.get("sample_size", 0),
        probabilities,
        dataset,
        construct["replicates"],
        construct["seed"],
        construct.get("max_retries", 20),
        construct.get("run", "regular"),
        construct["ci_method"],
        construct.get("alpha", 0.1),
    )
    for a in assertions:
        args = a.get("args", [])
        actual = _dispatch_bootstrap(result, a["method"], args)
        _check(actual, a)


# --- model_estimation path -----------------------------------------------------------------
# Inherently STATEFUL like mcmc_sampler/bootstrap: one _core.estimation_run (ML/MAP),
# _core.estimation_bayes_run (BayesianAnalysis, T12), or _core.model_simulate (Simulation,
# M13) call per case builds the model, runs its one stateful call (estimate() or the seeded
# ISimulatable draw), and returns the full result surface; every assertion in the case reads
# that single cached dict. See fixtures/README.md's model_estimation section for the full
# method list and the `bic`/`chain_value` design notes. `bic` is the one exception to the
# "cached dict" contract for ML/MAP: it takes an actual sample size `n` (C#
# `GetBIC(sampleSize)`), read live from the fixture's `args[0]` at dispatch time via
# `_core.estimation_bic`, not precomputed alongside the rest.
#
# M13: `construct.model` is no longer a flat {family, dataset} pair -- it can name any of the
# four Phase 5 model types, a full censored DataFrame, nonstationary trend specs, and explicit
# parameter values. The parsed spec is re-serialized with json.dumps (which round-trips
# doubles exactly) and handed to the SHARED C++ builder (bestfit/models/model_spec.hpp), the
# same code path the C++ runner and the R glue use; only the `dataset` reference is still
# resolved here, like every other fixture kind.


def _dispatch_estimation(
    result: dict,
    method: str,
    args: list,
    target: str,
    model_json: str,
    data: list,
    optimizer: str,
    construct: dict,
):
    # GMM (B11): j_stat/j_stat_pval come from the cached run dict; quantile_variance takes a
    # per-assertion AEP, so -- exactly like `bic`'s per-assertion sample size -- it rebuilds the
    # deterministic fit live via estimation_gmm_qvar. parameter/standard_error/covariance/
    # correlation/simulated_value reuse the shared arms below (the GMM run returns the same keys).
    if method == "j_stat":
        return result["j_stat"]
    if method == "j_stat_pval":
        return result["j_stat_pval"]
    if method == "quantile_variance":
        return _core.estimation_gmm_qvar(
            model_json,
            data,
            construct.get("strategy", "Iterative"),
            construct.get("optimizer", "BFGS"),
            int(construct.get("max_gmm_iterations", -1)),
            float(args[0]),
        )
    if method == "parameter":
        return result["parameters"][int(args[0])]
    if method == "max_log_likelihood":
        return result["max_log_likelihood"]
    if method == "aic":
        return result["aic"]
    if method == "bic":
        return _core.estimation_bic(target, model_json, data, optimizer, int(args[0]))
    if method == "covariance":
        return result["covariance"][int(args[0])][int(args[1])]
    if method == "standard_error":
        return result["standard_errors"][int(args[0])]
    if method == "correlation":
        return result["correlation"][int(args[0])][int(args[1])]
    if method == "dic":
        return result["dic"]
    if method == "waic":
        return result["waic"]
    if method == "looic":
        return result["looic"]
    if method == "posterior_mean":
        return result["posterior_mean"][int(args[0])]
    if method == "chain_value":
        return result["chains"][int(args[0])][int(args[1])][int(args[2])]
    if method == "simulated_value":
        # The seeded ISimulatable draw cached by _core.model_simulate (M13).
        return result["simulated"][int(args[0])]
    # The M14 DataFrame surface (works under any target -- it reads the model, not the
    # estimator): lazily build the frame surface ONCE per case via _core.model_data_frame
    # and memoize it in the case's result dict (the bic lazy-rebuild precedent).
    if method in ("number_of_low_outliers", "low_outlier_threshold", "plotting_position"):
        if "_data_frame" not in result:
            result["_data_frame"] = _core.model_data_frame(model_json, data)
        frame = result["_data_frame"]
        if method == "plotting_position":
            # plotting_position [kind, i]: kind is "exact" | "interval" | "uncertain".
            return frame[f"pp_{args[0]}"][int(args[1])]
        return frame[method]
    raise KeyError(f"unknown model_estimation fixture method: {method}")


def _run_estimation_case(target: str, construct: dict, assertions: list, datasets: dict):
    model = construct["model"]
    # Re-serialize the parsed spec for the shared C++ builder (see the path comment above).
    model_json = json.dumps(model)
    data = [float(v) for v in datasets[model["dataset"]]] if "dataset" in model else []

    if target == "Simulation":
        draws = _core.model_simulate(
            model_json, data, int(construct["sample_size"]), int(construct.get("seed", -1))
        )
        result = {"simulated": draws}
        optimizer = ""
    elif target == "BayesianAnalysis":
        sampler = construct.get("sampler", "DEMCzs")
        settings = construct.get("settings", {})
        result = _core.estimation_bayes_run(model_json, data, sampler, settings)
        optimizer = ""
    elif target == "GeneralizedMethodOfMoments":
        # GMM (B11): a bulletin17c model fit by GMM. One stateful estimate()+post_process, whose
        # full surface is cached here; quantile_variance rebuilds live at dispatch (see above).
        result = _core.estimation_gmm_run(
            model_json,
            data,
            construct.get("strategy", "Iterative"),
            construct.get("optimizer", "BFGS"),
            int(construct.get("max_gmm_iterations", -1)),
            int(construct.get("sample_size", 0)),
            int(construct.get("seed", -1)),
        )
        optimizer = ""
    else:
        optimizer = construct.get("optimizer", "DifferentialEvolution")
        # P3: an optional seeded-draw digest off the FITTED model (sample_size + seed) lets one
        # MLE smoke file cover parameter + max_log_likelihood + a seeded simulated_value.
        result = _core.estimation_run(
            target,
            model_json,
            data,
            optimizer,
            int(construct.get("sample_size", 0)),
            int(construct.get("seed", -1)),
        )

    for a in assertions:
        args = a.get("args", [])
        actual = _dispatch_estimation(
            result, a["method"], args, target, model_json, data, optimizer, construct
        )
        _check(actual, a)


# --- analysis path (Phase 8: user-facing Analyses layer) ------------------------------------
# Stateful like model_estimation: one _core.analysis_* call per case builds + runs the analysis
# and returns the full result surface; every assertion reads that single cached dict. The
# construct fields map 1:1 onto the glue args, so R/Python/C++ build byte-identical analyses.


def _dispatch_analysis(result: dict, method: str, args: list):
    if method == "candidate_count":
        return len(result["aic"])
    if method == "candidate_aic":
        return result["aic"][int(args[0])]
    if method == "candidate_bic":
        return result["bic"][int(args[0])]
    if method == "candidate_rmse":
        return result["rmse"][int(args[0])]
    if method == "candidate_converged":
        return 1.0 if result["converged"][int(args[0])] else 0.0
    if method == "parameter":
        return result["parameters"][int(args[0])]
    if method == "mode_curve":
        return result["mode_curve"][int(args[0])]
    if method == "mean_curve":
        return result["mean_curve"][int(args[0])]
    if method == "lower_ci":
        return result["lower_ci"][int(args[0])]
    if method == "upper_ci":
        return result["upper_ci"][int(args[0])]
    if method == "exceedance_probability":
        return result["exceedance_probabilities"][int(args[0])]
    if method == "point_estimate":
        return result["point_estimates"][int(args[0])]
    if method == "beta1":
        return result["beta1"][int(args[0])]
    if method == "nu":
        return result["nu"][int(args[0])]
    if method == "quantile_variance":
        return result["quantile_variance"][int(args[0])]
    if method in ("aic", "bic", "dic", "rmse", "confidence_level"):
        return result[method]
    raise KeyError(f"unknown analysis fixture method: {method}")


def _run_analysis_case(target: str, construct: dict, assertions: list, datasets: dict):
    ep = [float(v) for v in construct.get("exceedance_probabilities", [])]
    if target == "FittingAnalysis":
        data = [float(v) for v in datasets[construct["dataset"]]]
        result = _core.analysis_fit_distributions(data)
    elif target == "UnivariateAnalysis":
        model = construct["model"]
        model_json = json.dumps(model)
        data = [float(v) for v in datasets[model["dataset"]]]
        result = _core.analysis_univariate_run(
            model_json,
            data,
            construct.get("sampler", "DEMCzs"),
            int(construct.get("iterations", 3000)),
            int(construct.get("output_length", 10000)),
            float(construct.get("credible_level", 0.90)),
            int(construct.get("seed", 12345)),
            ep,
            int(construct.get("thinning_interval", -1)),
        )
    elif target == "Bulletin17CAnalysis":
        model = construct["model"]
        model_json = json.dumps(model)
        data = [float(v) for v in datasets[model["dataset"]]]
        result = _core.analysis_b17c_run(
            model_json,
            data,
            construct.get("uncertainty_method", "MultivariateNormal"),
            int(construct.get("output_length", 10000)),
            int(construct.get("seed", 12345)),
            float(construct.get("confidence_level", 0.90)),
            ep,
        )
    else:
        raise KeyError(f"unknown analysis target: {target}")

    for a in assertions:
        args = a.get("args", [])
        actual = _dispatch_analysis(result, a["method"], args)
        _check(actual, a)


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


# --- bivariate_copula path ---------------------------------------------------------------
# Every copula shares BivariateCopula's uniform theta/get_copula_parameters/pdf/cdf/... API
# (unlike multivariate_distribution's Dirichlet/Multinomial/BivariateEmpirical/..., which
# share no common surface), so this path is fully generic through the factory-driven
# _core.cop_val/_core.cop_fit bindings in copula.cpp -- no per-target branching, mirroring
# copula_factory.hpp's rationale. construct is either {"theta": x} (optionally {"theta": x,
# "df": y} for 2-parameter copulas, and/or {"marginals": {"targets", "params"}} to attach
# marginals directly -- used by the "random_value" sampling oracles) or {"fit": {"x", "y",
# "method", "marginals"?}}; see fixtures/README.md for the full schema.


def _build_copula_params(construct: dict) -> list[float]:
    p = [_num(construct["theta"])]
    if "df" in construct:
        p.append(_num(construct["df"]))
    return p


def _dispatch_copula(
    target: str,
    params: list[float],
    method: str,
    args: list,
    marg_x_target: str = "",
    marg_x_params: list[float] | None = None,
    marg_y_target: str = "",
    marg_y_params: list[float] | None = None,
):
    ar = _flatten_mv_args(args)
    return _core.cop_val(
        target, params, method, ar, marg_x_target, marg_x_params or [], marg_y_target, marg_y_params or []
    )


def _run_copula_case(target: str, construct: dict, assertions: list, datasets: dict):
    if "fit" in construct:
        fit = construct["fit"]
        x = [float(v) for v in datasets[fit["x"]]]
        y = [float(v) for v in datasets[fit["y"]]]
        marginals = fit.get("marginals")
        marg_x = marginals[0] if marginals else ""
        marg_y = marginals[1] if marginals else ""
        result = _core.cop_fit(target, x, y, fit["method"], marg_x, marg_y)
        for a in assertions:
            args = a.get("args", [])
            if a["method"] == "theta":
                actual = result["params"][0]
            elif a["method"] == "df":
                actual = result["params"][1]
            elif a["method"] == "marginal_param":
                which, idx = args[0], int(args[1])
                actual = result["marg_x_params"][idx] if which == "x" else result["marg_y_params"][idx]
            else:
                raise KeyError(f"unsupported post-fit copula fixture method: {a['method']}")
            _check(actual, a)
    else:
        params = _build_copula_params(construct)
        marg = construct.get("marginals")
        marg_x_target = marg["targets"][0] if marg else ""
        marg_y_target = marg["targets"][1] if marg else ""
        marg_x_params = [_num(v) for v in marg["params"][0]] if marg else []
        marg_y_params = [_num(v) for v in marg["params"][1]] if marg else []
        for a in assertions:
            args = a.get("args", [])
            actual = _dispatch_copula(
                target, params, a["method"], args, marg_x_target, marg_x_params, marg_y_target, marg_y_params
            )
            _check(actual, a)


def _load_cases():
    out = []
    for fx in sorted(_fixtures_dir().rglob("*.json")):
        spec = json.loads(fx.read_text())
        # Only validate univariate_distribution / multivariate_distribution /
        # bivariate_copula / mcmc_sampler fixtures; skip other kinds (e.g.
        # special_function) which are validated in C++ only and are not exposed to the
        # Python package.
        kind = spec.get("kind")
        if kind not in (
            "univariate_distribution",
            "multivariate_distribution",
            "bivariate_copula",
            "mcmc_sampler",
            "bootstrap",
            "model_estimation",
            "analysis",
        ):
            continue
        for case in spec["cases"]:
            out.append((kind, spec["target"], spec.get("datasets", {}), case))
    return out


CASES = _load_cases()


@pytest.mark.parametrize(
    "kind,target,datasets,case", CASES, ids=[f"{k}:{t}:{c['name']}" for k, t, _, c in CASES]
)
def test_fixture_case(kind, target, datasets, case):
    if kind == "model_estimation":
        _run_estimation_case(target, case["construct"], case["assertions"], datasets)
        return

    if kind == "analysis":
        _run_analysis_case(target, case["construct"], case["assertions"], datasets)
        return

    if kind == "bootstrap":
        _run_bootstrap_case(case["construct"], case["assertions"], datasets)
        return

    if kind == "mcmc_sampler":
        _run_mcmc_case(target, case["construct"], case["assertions"], datasets)
        return

    if kind == "bivariate_copula":
        _run_copula_case(target, case["construct"], case["assertions"], datasets)
        return

    if kind == "multivariate_distribution":
        if target == "MultivariateNormal":
            _run_mvn_case(case["construct"], case["assertions"])
        else:
            for a in case["assertions"]:
                args = a.get("args", [])
                actual = _dispatch_multivariate(target, case["construct"], a["method"], args)
                _check(actual, a)
        return

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
