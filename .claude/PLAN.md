# Plan: `bestfitr` (R) + `bestfitpy` (Python) from a shared C++ core

> **Current status (kept in sync by hand):** Phase 0, Phase 1, Phase 2, Phase 3, Phase 4,
> Phase 5, Phase 6, Phase 7a, Phase 8, Phase 9a, and Phase 10 are **complete** -- **FULL PARITY
> with the USACE-RMC Numerics / RMC.BestFit C# libraries is reached.**
>
> Phase 1 delivered the full Numerics math/RNG foundation plus all 42 univariate distributions.
> Ported and fixture-validated in C++/R/Python, reproduced against the real Numerics library via
> the dotnet oracle gate: all special functions (Erf, Gamma, Beta, Factorial, Bessel I0/I1), the
> AdaptiveGaussKronrod integrator, Brent root-finder, Nelder-Mead optimizer, Matrix/Vector linear
> algebra, MersenneTwister RNG, Sobol sequence, Statistics (product moments, L-moments),
> GoodnessOfFit, and all 42 univariate distributions including composite types (Mixture,
> TruncatedDistribution, CompetingRisks).
>
> Phase 2 delivered the multivariate distributions and copula layer: Dirichlet, Multinomial,
> BivariateEmpirical, MultivariateNormal (with the ported Genz MVNDST integrator), and
> MultivariateStudentT; all seven bivariate copulas (Clayton, AliMikhailHaq, Frank, Gumbel, Joe,
> Normal, StudentT) sharing one copula-estimation base (tau/MPL/IFM/MLE fits) plus an
> `IMaximumLikelihoodEstimation` mixin; and CompetingRisks' correlated dependency modes
> (`PerfectlyNegative`/`CorrelationMatrix`), un-deferred from Phase 1. Fixture-validated in
> C++/R/Python and reproduced against the real Numerics library by the dotnet oracle gate. Merged
> as PR #4 with CI green.
>
> Phase 3 delivered Sampling/MCMC and Bootstrap. All 8 MCMC samplers (RWMH, ARWMH, DEMCz, DEMCzs,
> HMC, NUTS, Gibbs, SNIS) plus the shared `MCMCSampler` base, model registry, and diagnostics/
> results layer (Gelman-Rubin R-hat, ESS, MAP tracking); the DifferentialEvolution optimizer stack
> (ParameterSet/Optimizer base/DE) the MAP-initialization path depends on; and the regular
> (non-pivotal) Bootstrap workflow (`Run`/`RunDoubleBootstrap`/`RunWithStudentizedBootstrap`,
> Percentile/BiasCorrected/BCa/Normal/BootstrapT confidence intervals). Fixture-validated in
> C++/R/Python and reproduced against the real Numerics library by the dotnet oracle gate; seeded
> MCMC chains and bootstrap replicate streams are proven bit-identical across R and Python via
> `short_exact`-style digest fixtures (`rel: 1e-12`, measured ~1e-15). The covariance-aware
> **pivotal** bootstrap workflow (`RunPivotalBootstrap` and its link-function/EVD/
> MatrixRegularization support) was scoped as the phase's severable final task and is tracked
> separately as a follow-up rather than landing on this branch. Merged as PR #5 with CI green.
>
> Phase 4 delivered BestFit's `Estimation` layer (numbered phasing item 5 -- see the phasing
> list below). The Estimation-support slice: `MatrixRegularization` (MakeSymmetricPositiveDefinite),
> `Stratify`/`StratificationOptions`, `NumericalDiff` (Hessian + pointwise gradients), and the
> `OptimizationMethod` enum. The Models slice: `ModelParameter`/`DataComponent`/`PriorComponent`,
> `IModel`/`ModelBase`, and `UnivariateDistributionModel` (including the Jeffreys 1/scale prior on
> MAP/Bayesian). The estimators: `MaximumLikelihood`, `MaximumAPosteriori`, and `BayesianAnalysis`
> (4 samplers -- DEMCz/DEMCzs/ARWMH/NUTS) with diagnostics (DIC/WAIC/LOOIC via PSIS-LOO, point
> estimates, posterior covariance). A new `model_estimation` fixture kind is wired end-to-end
> across C++/R/Python with bindings. The oracle emitter now subset-compiles the REAL C#
> estimators (`verify_oracles.py`: 3751 reproduced, 0 failed), and a seeded DEMCzs chain digest is
> proven bit-identical across R/Python -- the estimation-layer parity payoff, matching the RNG/MCMC
> parity Phase 3 established. SEVERED/DEFERRED: GMM + `IGMMModel` + BFGS + the Bulletin17C
> coupling (blocked in part by an upstream CS0104 compile bug, see
> `docs/upstream-csharp-issues.md`); the alternate optimizers Powell/MLSL/LocalMethod (gated,
> throw); `LeverageDiagnostics` and the rest of the `Diagnostics` layer (Influence/PriorInfluence
> -- the `Compute*Diagnostics` methods are gated stubs); `BayesianAnalysis`'s async/WPF/XML/
> `GenerateReport` surface. Known limitation carried from Task T7: the Brent/NelderMead optimizer
> adapters always report status `Success` (the standalone C++ optimizers expose no convergence
> state) -- values are correct regardless, and the default DE path reports status faithfully.
> Merged as PR #6 with CI green.
>
> Phase 5 delivered BestFit's flood-frequency Models core (the first slice of phasing item 6 --
> see the phasing list below). `MultipleGrubbsBeckTest` (the Numerics MGBT low-outlier test). The
> DataFrame layer: the censored data types (Exact/Interval/Uncertain/Threshold over a shared
> `Data` base), the four typed series collections, and `DataFrame` itself with `FullTimeSeries`
> threshold expansion, `ProcessThresholdSeries` overlap exclusivity, and MGBT/explicit-threshold
> low outliers; Hirsch-Stedinger plotting positions (`CalculatePlottingPositions`, which required
> a faithful port of .NET's `ArraySortHelper` introsort because `List<T>.Sort` tie order is
> oracle-visible); and `ThresholdDiagnostics` (mean residual life + GPD parameter stability). The
> trend/link function layer: all ten trend models on a shared `TrendModelBase` plus
> `GeneralLinearFunction`. The four models: `UnivariateDistributionModel` extended in place with
> the nonstationary (trend) and censored likelihood surface, `MixtureModel` (EM fitting + zero
> inflation), `CompetingRisksModel`, and the non-seasonal `PointProcessModel`. The
> `model_estimation` fixture kind is extended across all three runners, and M14 re-pinned every
> Phase 5 fixture against real C# oracles through the extended emitter (`verify_oracles.py`: 3837
> reproduced, 0 failed, 11 documented GEV std-err skips); seeded simulation digests reproduce the
> C# Mersenne Twister stream bit-for-bit and identically across R and Python. Port-fidelity items
> worth surfacing: the oracle dumps caught a real divergence -- C# `Mixture.SetParameters(ref)`
> normalizes weights in the optimizer's own arrays, so the Objective/ModelBase likelihood
> signatures became mutable-ref (fixed in M14; the C++ mixture fit now matches the C# to 1e-12) --
> with one documented deviation: `BayesianAnalysis` hands the oracle-locked const-ref MCMC
> samplers a mutable copy, so a mixture-under-MCMC write-back is not ported (no fixture exercises
> it). SEVERED/DEFERRED (documented in headers): the DataFrame hypothesis-test facade and
> summary-statistics/Q-Q surface (Numerics HypothesisTests and the regression facades are
> unported); the seasonal PointProcess data path plus GeneratePOTTimeSeries and CreateBlockSeries
> (they need the unported 2,334-line TimeSeries container); the DataFrame bootstrap/resampling
> surface (JackKnife/Resample/BootstrapDataFrame -- Bulletin17C-only, Phase 6);
> `ExactData.DateTime`; `FittedDistribution`; USGSRawText/CreateFromUSGS; all XML/INPC surfaces;
> and the GeneralizedNormal distribution (on the C# model whitelist but never ported in Phase 1,
> so constructing a model by that type throws). The new NelderMead fit-tolerance tiers on the
> M14 fixtures are untested on Linux/Windows until the CI run. Pending CI run and PR.
>
> Phase 6 delivered BestFit's Bulletin17C GMM track (the second slice of phasing item 6 -- see
> the phasing list below). The Numerics link-function layer: `ILinkFunction`/`LinkController`/
> `LinkFunctionFactory` and the seven standard links (identity/log/logit/probit/
> complementary-log-log/Fisher-z/Yeo-Johnson, the last over a ported `YeoJohnson` transform). The
> six BestFit links (asinh, SES, log-SES, log-asinh, centered, Yeo-Johnson) on
> `BestFitLinkFunctionFactory`. The `ParameterPenalty`/`QuantilePenalty` support types. The
> distribution moment machinery (`ConditionalMoments`, `ParametersFromMoments`,
> `QuantileGradientForMoments`) added additively to the Phase 1 distributions. The `BFGS`/`Powell`/
> `MLSL` optimizers (with `LocalMethod`) beside DifferentialEvolution, un-gating the three Phase 4
> MLE/MAP throws. `GeneralizedMethodOfMoments`, `IGMMModel`, and `Bulletin17CDistribution` with its
> moment heart. Fixture-validated in C++/R/Python and reproduced against the real Numerics/
> RMC.BestFit libraries by the dotnet oracle gate (`verify_oracles.py`: 3871 reproduced, 0 failed,
> 11 documented GEV std-err skips); seeded GMM covariance/standard errors reproduce to ~1e-12 and
> the MLSL seeded stream is bit-identical across R and Python. SEVERED/DEFERRED (documented in
> headers): the GMM Influence/Leverage Diagnostics region (C# `GeneralizedMethodOfMoments.cs` lines
> 1382-2061) ships as documented throwing stubs pending the unported `RMC.BestFit.Diagnostics`
> layer; the DataFrame JackKnife/Resample/BootstrapDataFrame/ShiftDistribution surface
> (`Bulletin17CAnalysis`-only) moves to Phase 7; the Numerics `Functions/` non-link classes stay
> unported. B17C GMM is always just-identified (`NumberOfMomentConditions == NumberOfParameters`),
> so its J-statistic p-value is structurally `NaN` and no over-identified oracle is reachable (see
> `docs/upstream-csharp-issues.md`). Pending CI run and PR.
>
> Phase 7a delivered the four remaining ModelBase model families, fit by the already-ported
> MLE/MAP/Bayesian estimators. TimeSeries -- `AutoRegressive`/`MovingAverage`/`ARIMA`/`ARIMAX`
> (all `ModelBase + ISimulatable`, preserving the AR/MA warm-up and the conditional-vs-all-t
> likelihood divergence). SpatialExtremes -- `SpatialGEV` (the hierarchical Renard GEV) over the
> three correlation models (BasicExponential/PoweredExponential/Spherical) plus
> `CachedMultivariateNormal`, `GaussianCopula`, and `SpatialRegressionErrors`, keeping the
> non-canonical spatial-error log-density decomposition. RatingCurve -- the BaRatin addition-mode
> stage-discharge model (1-3 segments, log10-space Normal residual likelihood, optional Jeffreys
> 1/sigma). And BivariateDistribution -- two `IUnivariateModel` marginals plus a ported Numerics
> copula via `CreateCopula` (CopulaEstimationMethod InferenceFromMargins default / PseudoLikelihood).
> These sit atop the P1/P2 prerequisites: the authorized `IUnivariateModel` resolution on
> `univariate_distribution_model.hpp`, `statistics::maximum`, the ported Numerics `BoxCox` transform
> (`numerics/data/box_cox.hpp`), and the thin `numerics/data/time_series/time_series.hpp` adapter.
> Everything is fixture-validated in C++/R/Python and reproduced against the real Numerics/
> RMC.BestFit libraries by the dotnet oracle gate (`verify_oracles.py`: 3930 reproduced, 0 failed,
> 11 documented GEV std-err skips; ctest 49/49, `test_fixtures` 3941 checks; testthat 3539/0;
> pytest 563). The seeded `GenerateRandomValues`/`GenerateRandomSeries` draws reproduce the C#
> MersenneTwister stream bit-for-bit and are bit-identical across R and Python via the two `*_sim`
> digest fixtures. Documented severances carried in the ported headers: the ARIMAX covariate
> forecast-tail extension (CovariateExtensionMethod BlockBootstrap/KNN) and the heavy 2,334-line
> Numerics TimeSeries container (interpolation / file-I/O / hypothesis tests). Pending CI run and PR.
>
> Phase 8 delivered the user-facing BestFit `Analyses` layer (numbered phasing item 7 -- see the
> phasing list below), the last slice of the port. The three exported analyses -- `univariate_analysis`
> (Bayesian MCMC frequency curve + credible band + goodness-of-fit), `fit_distributions` (the
> 14-candidate MLE ranking surface), and `bulletin17c_analysis` (the LP3 flood-frequency GMM fit) --
> are bound in both packages (`bestfitr/R/analysis.R` + `bestfitr/src/analysis.cpp`,
> `bestfitpy/src/bestfitpy/analysis.py` + `bestfitpy/src/bindings/analysis.cpp`) over the shared C++
> `UnivariateAnalysis`/`FittingAnalysis`/`Bulletin17CAnalysis`, with a new `analysis` fixture kind
> across all three runners. The Numerics output types landed: `ProbabilityOrdinates` (the ordinate
> grid + 25 default exceedance probabilities) and `UncertaintyAnalysisResults` (the mode/mean/CI
> curve container). The DataFrame bootstrap surface (`JackKnife`/`Resample`/`BootstrapDataFrame`/
> `ShiftDistribution`) was un-deferred from Phase 5/6 and added additively to `data_frame.hpp`,
> consumed only by Bulletin17CAnalysis. Three Bulletin17CAnalysis UQ paths shipped: the
> MultivariateNormal default, the parametric `Bootstrap`, and the deterministic Cohn-style
> delta-method confidence interval. The dotnet emitter now subset-compiles the minimal RMC.BestFit
> Analyses closure (via a local CS0104-patched `Bulletin17CAnalysis.cs`) and drives the real C#
> analyses to tighten the smoke fixtures to exact oracles. Everything is fixture-validated in
> C++/R/Python and reproduced against the real Numerics/RMC.BestFit libraries by the dotnet oracle
> gate (`verify_oracles.py`: 3950 reproduced, 0 failed, 11 documented GEV std-err skips; ctest 56/56;
> testthat 3585/0; pytest 574). Deferred to a remaining Phase 9: the per-family analysis
> orchestrators (Composite / Mixture / PointProcess / CompetingRisk / the four TimeSeries /
> SpatialGEV / Bivariate / RatingCurve / CoincidentFrequency, plus Weighted / Batch), the
> LinkedMultivariateNormal path (its ~13 link-builder helpers + InfluenceStatistics) and the
> BiasCorrected / pivot Bootstrap, the Numerics `BootstrapAnalysis` frequentist engine (875 lines,
> unused by this scope), the `Diagnostics/` layer (LeverageDiagnostics / Influence, still throwing
> stubs), and GMM report generation. Carried-forward BUG (A11): seeded population-sampler
> (DEMCz/DEMCzs) runs with `thinning_interval > 1` are NOT oracle-guaranteed C#-vs-C++ until the
> thinned `ChainIteration`/archive-update cadence is bisected and fixed; single-step / thin=1 is
> bit-identical, so every shipped Bayesian fixture (all thin=1) is unaffected. Pending CI run and PR.
>
> Phase 9a delivered the per-family analysis orchestrators and the essential Diagnostics layer (the
> low-risk parity tail of phasing item 7). The univariate-family analyses -- `mixture_analysis`,
> `point_process_analysis`, `competing_risk_analysis` (`analyses/univariate/`) -- and the four
> TimeSeries analyses -- `ar_analysis`, `ma_analysis`, `arima_analysis`, `arimax_analysis`
> (`analyses/time_series/`) -- are mechanical Bayesian clones of the Phase-8 UnivariateAnalysis
> (validate -> `BayesianAnalysis.estimate()` -> UncertaintyAnalysisResults). The Diagnostics layer
> (`diagnostics/`) landed `LeverageDiagnostics` (Cook's-distance + variance-influence decomposition
> at the MAP point via a numerical Hessian), `InfluenceDiagnostics` (the PSIS-LOO Pareto-k wrapper),
> and `PriorInfluenceDiagnostics` (prior-to-data influence off the seeded posterior); wiring them
> un-stubbed the 6 previously-throwing estimator methods across MAP (`compute_leverage_diagnostics`),
> BayesianAnalysis (`compute_influence_`/`compute_prior_influence_`/`compute_leverage_diagnostics`),
> and the GMM quartet (`get_observation_influence`/`get_cooks_distance`/`get_influence_diagnostics`
> x2/`get_leverage_diagnostics`). All eleven are user-callable in both packages, and a diagnostics
> accessor hangs off the estimator results. The dotnet emitter now compiles the REAL Diagnostics
> classes in place (replacing the deleted `DiagnosticsStubs.cs`) and drives all seven per-family
> analyses. Fixture-validated in C++/R/Python and reproduced against the real Numerics/RMC.BestFit
> libraries by the dotnet oracle gate (`verify_oracles.py`: 4003 reproduced, 0 failed, 14 skipped;
> ctest 60/60; testthat 3687/0; pytest 590). Fidelity notes (documented in
> `docs/upstream-csharp-issues.md`): the AR/MA/Mixture seeded-DEMCzs analysis curves diverge C#-vs-C++
> by inherent chaotic short-chain sensitivity (the deterministic DataLogLikelihood matches C# to
> `<= 3 ulp`, Mixture bit-identical, across 238 param vectors; a 100-iter chain on a flat AR/MA
> intercept ridge or the symmetric bimodal Mixture surface flips an accept/reject or DE basin), so
> those three fixtures assert only build-stable structural invariants (`curve_length`; AR
> `mode_curve[0]`) with NO oracle_skip and NO tolerance loosening -- matching the Phase-3 HMC/NUTS
> precedent; CompetingRisk/PointProcess tightened to exact (~1e-10); ARIMA/ARIMAX structural. Three
> `PriorInfluenceDiagnostics` assertions carry `oracle_skip` because the ported `ModelParameter`
> names are empty and collapse the two Normal parameter priors into one component (a deterministic
> name-keyed dedup, not a stream divergence). Remaining Phase 10 (documented, not implemented):
> CompositeAnalysis + the Numerics BootstrapAnalysis engine + WeightedUnivariateAnalysis;
> SpatialGEVAnalysis (+ SpatialGEVSiteResults/CrossValidationResults DTOs); BivariateAnalysis +
> CoincidentFrequencyAnalysis; RatingCurveAnalysis; the Bulletin17C-deferred uncertainty methods
> (LinkedMVN + link-builders + pivot/BiasCorrected bootstrap + InfluenceStatistics); and the two
> predictive checks (PosteriorPredictiveCheck/PriorPredictiveCheck). PERMANENT SKIPS: GMM report
> generation (presentation-only text) and BatchAnalysisRunner + options (GUI batch scheduling).
> Pending CI run and PR.
>
> Phase 10 delivered the final parity tail and closes the port: **FULL PARITY**. The five remaining
> user-facing analysis orchestrators landed -- `RatingCurveAnalysis` (`analyses/rating_curve/`),
> `BivariateAnalysis` + `CoincidentFrequencyAnalysis` (`analyses/bivariate/`), `SpatialGEVAnalysis`
> plus its two DTOs SpatialGEVSiteResults / SpatialGEVCrossValidationResults
> (`analyses/spatial_extremes/`), and `CompositeAnalysis` (`analyses/univariate/`) +
> `WeightedUnivariateAnalysis` (`analyses/support/`) -- each a faithful AnalysisBase clone of the
> Phase-8 template. The Numerics `BootstrapAnalysis` frequentist engine
> (`numerics/distributions/uncertainty_analysis/bootstrap_analysis.hpp`) shipped with all five CI
> methods (Percentile / BiasCorrected / Normal cube-root / Bootstrap-t / BCa) over the new
> `IBootstrappable` mixin (`numerics/distributions/base/i_bootstrappable.hpp`, wired onto Normal).
> The two formerly-throwing Bulletin17C uncertainty dispatch arms were un-gated:
> LinkedMultivariateNormal (the ~13 link-builder helpers + InfluenceStatistics, MVN not MVT, the
> center-shift commented out exactly as C#) and the pivot / BiasCorrected bootstrap. The four
> predictive-check classes (`PosteriorPredictiveCheck` / `PriorPredictiveCheck` /
> `PredictiveCheckResults` / `PredictiveSummary`) landed under `diagnostics/`, completing that layer
> beside the Phase-9a leverage/influence/prior-influence diagnostics. The Phase-1 follow-ups closed
> out: distribution `ParameterNames` on the base + every concrete distribution (+ ModelParameter
> naming) and the mutable UserDefined MCMC seeding hook (`seed_population`/`seed_chain` on
> `mcmc_sampler.hpp` + `bayesian_analysis.hpp`, wired into the MixtureAnalysis EM-seed path). The
> user-facing R/Python API binds every one of these. Everything is fixture-validated in C++/R/Python
> and reproduced against the real Numerics / RMC.BestFit libraries by the dotnet oracle gate
> (`verify_oracles.py`: 4069 reproduced, 0 failed, 11 skipped; ctest 69/69; testthat 3770/0; pytest
> 606). The ONLY remaining permanent skips are presentation-only with no numeric/statistical surface:
> Bulletin17CAnalysis GMM report generation (~607 lines of StringBuilder text) and BatchAnalysisRunner
> + BatchAnalysisResult/Options (the WPF batch scheduler) -- R/Python users supply their own. Honest
> fidelity note (the Phase-9a chaotic-sensitivity precedent, documented in
> `docs/upstream-csharp-issues.md`): five seeded-DEMCzs analysis curves (Bivariate / Coincident-curve
> / Composite / RatingCurve / SpatialGEV) reproduce their posterior MAP C#-vs-C++ only to ~1e-6 by
> short-chain amplification of sub-1e-8 model-density ULP drift (the deterministic copula-MLE /
> log-likelihood / Normal-MLE paths reproduce to 1e-8/1e-9), so those fixtures assert only the
> deterministic structural invariants that reproduce bit-identically across all four runners -- NO
> `oracle_skip` mask, NO loosened tolerance. Pending: the CI run and PR ship step only (driven as a
> separate workflow run, per the standing "WORKFLOW RESUME is UNSAFE" instruction).
>
> CI is green on the full matrix (`sync-check`, `core`, `r-cmd-check`, `python`) on
> Linux/macOS/Windows as of the Phase 4 merge (PR #6); the Phase 5 branch (`phase5-models`) has
> not yet been pushed for CI. The dotnet oracle gate is dev-only (not in CI).
>
> Upstream submodules are present (`upstream/Numerics`, `upstream/RMC-BestFit`, official
> USACE-RMC `main`, shallow, dev-only). Still pending: `PORTING_MANIFEST.toml`,
> `upstream_diff.py`, the auto-scraper for bulk oracle extraction, and the `core/src`
> source-manifest machinery.
> The rest of this document is the originally approved architecture and phasing.

## Context

RMC.BestFit and its dependency Numerics are mature C# (.NET 10) libraries for Bayesian
flood-frequency / extreme-value analysis (~140K LOC, ~3,877 oracle-validated tests). The goal
is to make this capability available as idiomatic, **CRAN- and PyPI-publishable** R and Python
packages that keep BestFit's speed and exact level of validation, while reusing as much of the
existing work as possible.

Key decisions already settled with the user:
- **CRAN + PyPI publishing is a hard requirement.** This rules out a .NET-binding approach for R
  (CRAN builds from source on its own machines and will not install the .NET SDK).
- **Approach: one shared C++ core** (a mechanical-as-possible port of the C# computational
  surface) wrapped by **cpp11** for R and **pybind11** for Python. Write the math once, bind
  twice. C# → C++ is the lowest-friction compiled target that both registries bless.
- **Scope: full parity ("Everything")** — all model families and all of Numerics — phased.
- **Lead with R** (stricter packaging), Python in lockstep behind the same core.

Why C++ and not .NET/Native-AOT: CRAN can't build it. Why not pure-native R/Python on
scipy/stats: no code reuse, highest validation drift, MCMC can't be reproduced. A single
self-contained C++ core preserves the C# algorithms (hence the oracle values hold to 1e-10–1e-12)
and, because both bindings run the *same* compiled code, R and Python MCMC chains are **identical
given the same seed**.

What we deliberately do **not** port: the ~50–65% of BestFit that is WPF / `INotifyPropertyChanged`
/ XML (`ToXElement`/`FromXElement`) / `[Browsable]` boilerplate, and the reflection-bearing
serialization layer (`Enum.TryParse`, custom `JsonConverter`s). These are desktop-app concerns.
`DataFrame`/`TimeSeries` become thin adapters over native `data.frame`/pandas at the binding edge.

## Repository structure

Monorepo with **one canonical C++ core** in `core/`, vendored (copied + committed) into each
package's `src/` by a sync script. CRAN/PyPI sdists must be self-contained, so each package
directory must independently be a valid source tree — submodules and symlinks do not survive
`R CMD build`/sdist export. A CI `sync-check` job re-runs the sync and fails on `git diff`.

```
bestfit/
├── upstream/                   # dev-only git submodules, pinned to a SHA/tag (NOT vendored into packages)
│   ├── Numerics/               #   github.com/USACE-RMC/Numerics @ pinned commit
│   └── RMC.BestFit/            #   the C# source we port from; the diff baseline
├── core/                       # THE canonical C++17 core (all numerical dev happens here)
│   ├── include/bestfit/{numerics/{distributions,math,sampling,data},models,estimation,analyses,diagnostics}/
│   ├── src/                    # matching .cpp translation units; layout MIRRORS the C# tree
│   ├── data/new-joe-kuo-6.21201   # Sobol direction numbers (was an embedded C# resource)
│   ├── tests/                  # doctest/Catch2 suites driven by ../fixtures
│   ├── PORTING_MANIFEST.toml   # provenance map: each .cs file → C++ file(s), status, last-ported SHA+hash
│   └── CMakeLists.txt          # dev-only: builds core static lib + C++ tests
├── fixtures/                   # canonical language-neutral oracle fixtures (JSON)
├── bestfitr/                   # R package (cpp11) — self-contained for CRAN
│   ├── DESCRIPTION             # LinkingTo: cpp11 ; SystemRequirements: C++17
│   ├── src/{bestfit_core/,*.cpp(glue),Makevars,Makevars.win}   # bestfit_core/ is GENERATED+committed
│   ├── R/  inst/fixtures/  inst/extdata/new-joe-kuo-6.21201  tests/testthat/  man/
├── bestfitpy/                  # Python package (scikit-build-core + CMake + pybind11)
│   ├── pyproject.toml  CMakeLists.txt
│   ├── src/{bestfit_core/,bindings/,bestfitpy/}   # bestfit_core/ is a SYMLINK into ../core
│   └── tests/
├── tools/{materialize_core.py,extract_oracles.py,verify_oracles.py,upstream_diff.py}
└── .github/workflows/
```

Each package vendors `core/{include,data}` and `fixtures/` as committed subtree symlinks (the core
stayed header-only, so no source manifest is needed). Git holds one copy; a build dereferences the
symlinks into a self-contained, symlink-free artifact (`R CMD build` for R, `tools/materialize_core.py`
for Python). See `docs/superpowers/specs/2026-07-08-shared-core-symlink-vendoring-design.md`.

## C++ core design

- **Standard: C++17** (`CXX_STD = CXX17` in Makevars, `SystemRequirements: C++17` in DESCRIPTION).
  Nothing in Numerics needs C++20; C++17 is the safe CRAN baseline.
- **Structural mirroring (maintainability investment).** The C++ tree mirrors the C# tree
  file-for-file and class-for-class, keeping method names and method order aligned with the source.
  This is not cosmetic: when upstream ships a change, a C# diff then maps almost line-for-line onto
  the corresponding C++ file, turning re-porting into a localized edit instead of an investigation.
  Each C++ file opens with a provenance header (`// ported from: Numerics/.../GeneralizedExtremeValue.cs
  @ <sha>`).
- **Self-contained — port Numerics' own linear algebra and RNG; do NOT vendor Eigen.** Numerics has
  zero runtime deps and its oracles were validated against *its own* Cholesky/LU/QR/SVD/Eigen and
  `Matrix`/`Vector` (row-major `double[]` wrappers). Reusing the same algorithms maximizes bit-level
  oracle fidelity and keeps the CRAN dependency surface empty. Eigen would risk last-ULP drift and a
  hard `LinkingTo`/vendoring burden.
- **Memory model:** value types for distributions/small math objects (C# `Clone()` → copy ctor);
  factory returns `std::unique_ptr<UnivariateDistribution>`; polymorphic model containers hold
  `std::vector<std::unique_ptr<...>>`. No raw `new`/`delete`; `shared_ptr` only where genuinely
  co-owned.
- **Interfaces:** `IUnivariateDistribution`/`UnivariateDistributionBase` → abstract base with pure
  virtuals (`pdf`/`cdf`/`inverse_cdf`/`set_parameters`/...) plus promoted shared methods
  (`log_likelihood`, `variance`, `generate_random_values`, `central_moments`). Capability mixins
  (`IMaximumLikelihoodEstimation`, `ILinearMomentEstimation`, `IBootstrappable`, ...) → pure-virtual
  interface classes via multiple inheritance; capability checks use `dynamic_cast` (RTTI on, CRAN-safe).
  Enums → `enum class`; the `UnivariateDistributionFactory` `if/else` → `switch` (drop the `XElement`
  overload). C# `ValidateParameters` nullable-exception pattern → `std::optional<std::string>` /
  throw `std::invalid_argument`.
- **RNG (the parity payoff):** port `MersenneTwister` as `std::array<uint32_t,624>` + `mti` with the
  tempering constants verbatim; `next_double() == gen_rand_int32() * (1.0/4294967296.0)` to match C#
  `GenRandReal2()`. Port Sobol; ship `new-joe-kuo-6.21201` as a data file located via
  `system.file()` (R) / `importlib.resources` (Python) and passed into the core loader. Pin all three
  languages with a C++ test asserting the canonical mt19937ar reference stream.

Reference files for the port:
`Numerics/Distributions/Univariate/Base/UnivariateDistributionBase.cs`,
`.../GeneralizedExtremeValue.cs`, `.../Base/UnivariateDistributionFactory.cs`,
`Numerics/Sampling/MersenneTwister.cs`.

## Build systems

- **R (cpp11):** `src/Makevars`/`Makevars.win` with `CXX_STD = CXX17`, `PKG_CPPFLAGS` pointing at
  vendored headers, `OBJECTS` from the generated `core_sources.mk`. **No** `-march=native`, `-O3`,
  LTO, or `-Werror` (let R drive flags — the #1 CRAN portability rule). Glue exposes free functions
  and `external_pointer` handles wrapped in R6/S4 classes.
- **Python (scikit-build-core + CMake + pybind11):** `pyproject.toml` build-backend
  `scikit_build_core.build`; `CMakeLists.txt` sets `CMAKE_CXX_STANDARD 17`, `include()`s
  `core_sources.cmake`, `pybind11_add_module`. Wheels via **cibuildwheel** (manylinux/musllinux,
  macOS universal2, Windows MSVC); Sobol data shipped as package data.

## Test-fixture strategy (validate identically, DRY)

Extract the ~3,877 hardcoded oracles **once** from the `.cs` tests into versioned language-neutral
JSON under `fixtures/`. Three thin generic runners (C++ doctest, R testthat, Python pytest) read the
*same* fixtures, so "validate identically" is structural, not aspirational.

Fixture schema captures per-assertion **comparison mode** + **tolerance** (the C# tests mix absolute
`Assert.AreEqual(exp,act,tol)` and relative `Assert.IsLessThan(0.01,(x-true)/true)`):
each fixture = `{target, kind, source_test, cases:[{name, construct|input, estimate?, seed?,
assertions:[{method,args,expected,tol,mode}]}]}` with modes `abs|rel|exact(NaN/Inf)|bool|vector|matrix`.
RNG/MCMC fixtures add `seed` + expected first-N values / digest to prove identical streams across R/Python.

`tools/extract_oracles.py` scrapes the regular RMC test patterns into **draft** fixtures + a coverage
report; a human curates the residue (interdependent covariance/std-error asserts). Each language has a
~40-entry method-dispatch table mapping fixture `"method"` strings to its API
(`"mean" → Mean / $mean() / .mean()`) — the only per-language test code. `fixtures/` is vendored
into `bestfitr/inst/fixtures/` and the bestfitpy package data as a subtree symlink.

## Upstream synchronization (keeping the port current)

RMC.BestFit/Numerics are updated frequently, so pulling upstream changes must be a guided,
repeatable workflow — not a re-audit. Four mechanisms make this routine:

1. **Pinned upstream submodules** (`upstream/Numerics`, `upstream/RMC.BestFit`) give a local diff
   baseline at a known commit. Dev-only: never vendored into the shipped `core/` or package sdists.
2. **Provenance manifest** (`core/PORTING_MANIFEST.toml`): one entry per upstream `.cs` file recording
   its C++ counterpart(s), a **status** (`ported | adapter | skipped:boilerplate | skipped:test`), and
   the **last-ported upstream SHA + content hash**. The `skipped:boilerplate` status is what lets us
   ignore the high-churn WPF/XML/`INotifyPropertyChanged` files entirely — serialization churn
   generates zero porting noise.
3. **`tools/upstream_diff.py`**: bump the submodule to a new SHA/tag, then this tool walks the manifest
   and, for every non-skipped `.cs` file whose hash changed since its last-ported SHA, emits a
   **porting worklist** — the affected C++ file(s) plus the inline C# diff. New/removed upstream files
   are flagged as manifest gaps. Output is a checklist a human (or an agent) works through.
4. **Fixtures as the change detector / safety net.** Because `extract_oracles.py` reads upstream's own
   `.cs` *test* files, re-running it regenerates `fixtures/`; `git diff fixtures/` shows every changed
   expected value and every new test case. After porting the worklist, the fixtures that still **fail**
   are exactly the behavior not yet ported — the test layer tracks upstream automatically.

**Routine to absorb an upstream release:** (a) bump submodule to the new tag; (b) `upstream_diff.py`
→ worklist; (c) `extract_oracles.py` → regenerate fixtures, review the diff; (d) port the C++ deltas
guided by the worklist, update manifest SHAs/hashes; (e) run all three harnesses — failing fixtures
pinpoint remaining work; (f) CI green → version bump recording "validated against BestFit `<tag>` /
Numerics `<tag>`". A **scheduled CI `upstream-watch` job** runs `upstream_diff.py` against the latest
upstream tag and opens an issue/PR with the worklist + fixture diff, so drift surfaces on its own.

## CI/CD (GitHub Actions)

Gated jobs after a fast **`sync-check`** (re-run sync scripts, `git diff --exit-code`):
1. **core** — build + C++ tests (incl. MT19937 reference stream) on ubuntu/macOS/windows.
2. **r-cmd-check** — {ubuntu, macOS, windows} × {R release, oldrel}, `R CMD check --as-cran`
   WARNING/NOTE-clean; scheduled win-builder/rhub pre-submission.
3. **python-wheels** — cibuildwheel matrix; run pytest inside each built wheel; build sdist.
4. **release** — PyPI via Trusted Publishing (OIDC) on `v*` tags; CRAN tarball produced by CI,
   submission human-gated (`devtools::release()`).
5. **upstream-watch** (scheduled) — `upstream_diff.py` + `extract_oracles.py` against the latest
   upstream tag; opens an issue/PR with the porting worklist + fixture diff when drift is detected.

## Phasing (full parity, dependency-ordered)

**Phase 0 — prove the whole toolchain end-to-end with ONE distribution before any mass porting.**
[DONE] Port the minimum to make **GeneralizedExtremeValue** work fully (moments, L-moments,
MLE via Nelder-Mead, Brent root-finding, Fisher-info matrix inverse, quantile variance): `Tools`
constants, `SpecialFunctions::Gamma`, minimal `Matrix`/`Vector`, `MersenneTwister`,
`RootFinding::Brent`, `Optimization::NelderMead`, `Statistics::{ProductMoments,LinearMoments}`, GEV.
**Exit criterion (met):** the same `generalized_extreme_value.json` passes in C++, R, and Python;
seeded RNG byte-identical between R and Python; all CI jobs green; sync + manifest proven. (Note:
the upstream-sync loop / `PORTING_MANIFEST.toml` / submodules are deferred — see status header.)

**Phases 1–7 — bulk port (tests ported alongside each chunk):**
1. Numerics math + RNG foundation (all `Mathematics`, `Sampling` RNG, `Data/Statistics`).
   [DONE] All special functions (Erf, Gamma, Beta, Factorial, Bessel I0/I1), AdaptiveGaussKronrod
   integrator, Brent root-finder, Nelder-Mead optimizer, Matrix/Vector linear algebra,
   MersenneTwister RNG, Sobol sequence, Statistics (product moments, L-moments), and GoodnessOfFit
   are ported, fixture-validated in all three harnesses, and reproduced against C# by the dotnet
   oracle gate.
2. All 42 univariate distributions (mechanical once the base exists; parallelizable).
   [DONE] All 42 distributions ported, fixture-validated in C++/R/Python, and reproduced against
   C# by the dotnet oracle gate. Includes composite types: Mixture, TruncatedDistribution, and
   CompetingRisks. The polymorphic factory/base/mixin layer is used throughout.
3. Multivariate distributions + copulas.
   [DONE] Dirichlet, Multinomial, BivariateEmpirical, MultivariateNormal (Genz MVNDST), and
   MultivariateStudentT; all seven bivariate copulas (Clayton, AliMikhailHaq, Frank, Gumbel, Joe,
   Normal, StudentT) with a shared copula-estimation base (tau/MPL/IFM/MLE fits) and an
   `IMaximumLikelihoodEstimation` mixin; CompetingRisks' correlated dependency modes un-deferred
   from Phase 1. Fixture-validated in C++/R/Python and reproduced against C# by the dotnet oracle
   gate. Merged as PR #4 with CI green.
4. Sampling/MCMC (RWMH, ARWMH, DEMCz/zs, HMC, NUTS, Gibbs, SNIS) + Bootstrap — fixture digests prove
   identical seeded chains across R/Python.
5. BestFit `Estimation` (MaximumLikelihood, MAP, GMM, BayesianAnalysis, NumericalDiff, OptimizationMethod).
   [DONE] MaximumLikelihood, MaximumAPosteriori, and BayesianAnalysis (DEMCz/DEMCzs/ARWMH/NUTS)
   with diagnostics (DIC/WAIC/LOOIC), NumericalDiff, OptimizationMethod, MatrixRegularization,
   Stratify/StratificationOptions, and the Models slice (ModelParameter/DataComponent/
   PriorComponent, IModel/ModelBase, UnivariateDistributionModel) ported, fixture-validated in
   C++/R/Python, and reproduced against C# by the dotnet oracle gate. GMM + IGMMModel + BFGS +
   the Bulletin17C coupling severed to a follow-up. Merged as PR #6 with CI green.
6. BestFit `Models` (UnivariateDistribution, Bulletin17C, Mixture, CompetingRisks, PointProcess,
   Bivariate, RatingCurve, TimeSeries, SpatialExtremes, Trend/Link functions) — boilerplate skipped,
   DataFrame as adapter.
   [DONE — first slice, the flood-frequency core] MultipleGrubbsBeckTest; the DataFrame layer
   (data types, the four series collections, FullTimeSeries/ProcessThresholdSeries/MGBT low
   outliers, Hirsch-Stedinger plotting positions, ThresholdDiagnostics); the trend/link functions
   including GeneralLinearFunction; and the four models (nonstationary + censored
   UnivariateDistributionModel extended in place, MixtureModel, CompetingRisksModel, non-seasonal
   PointProcessModel) ported, fixture-validated in C++/R/Python, and reproduced against C# by the
   dotnet oracle gate.
   [DONE — second slice, the Bulletin17C GMM track] The Numerics link-function layer
   (ILinkFunction/LinkController/LinkFunctionFactory + the seven standard links incl. Yeo-Johnson)
   and the six BestFit links; ParameterPenalty/QuantilePenalty; the distribution moment machinery
   (ConditionalMoments/ParametersFromMoments/QuantileGradientForMoments); the BFGS/Powell/MLSL
   optimizers un-gating the Phase 4 MLE/MAP throws; GeneralizedMethodOfMoments, IGMMModel, and
   Bulletin17CDistribution ported, fixture-validated in C++/R/Python, and reproduced against C# by
   the dotnet oracle gate (3871 reproduced, 0 failed, 11 GEV std-err skips). Severed to Phase 7:
   the GMM Influence/Leverage Diagnostics region (throwing stubs pending RMC.BestFit.Diagnostics),
   the DataFrame JackKnife/Resample/BootstrapDataFrame/ShiftDistribution surface, and the Numerics
   Functions/ non-link classes.
   [DONE — models slice (Phase 7a)] The four remaining ModelBase model families, fit by the
   already-ported MLE/MAP/Bayesian estimators: TimeSeries (AutoRegressive/MovingAverage/ARIMA/
   ARIMAX), SpatialExtremes (SpatialGEV hierarchical Renard model + the three correlation models +
   CachedMultivariateNormal + GaussianCopula + SpatialRegressionErrors), RatingCurve (BaRatin
   addition-mode stage-discharge), and BivariateDistribution (two IUnivariateModel marginals + a
   ported copula, IFM/Pseudo) -- atop the P1/P2 prerequisites (the authorized IUnivariateModel
   resolution on univariate_distribution_model.hpp, statistics::maximum, numerics/data/box_cox.hpp,
   the thin numerics/data/time_series/time_series.hpp adapter). Fixture-validated in C++/R/Python
   and reproduced against C# by the dotnet oracle gate (3930 reproduced, 0 failed, 11 GEV std-err
   skips; ctest 49/49, test_fixtures 3941 checks; testthat 3539/0; pytest 563); seeded draws are
   bit-identical across R and Python via the two *_sim digest fixtures. Documented severances
   carried in the ported headers: the deferred ARIMAX covariate forecast-tail extension
   (CovariateExtensionMethod BlockBootstrap/KNN) and the heavy 2,334-line Numerics TimeSeries
   container. Pending CI run and PR. The Models phase is now complete; the clear remainder is
   Phase 8 (numbered item 7 below).
7. BestFit `Analyses` + `Diagnostics` — the user-facing API (`univariate_analysis()` etc.).
   [DONE — the user-facing Analyses layer (Phase 8)] The three exported analyses
   `univariate_analysis`/`fit_distributions`/`bulletin17c_analysis` over the shared C++
   `UnivariateAnalysis`/`FittingAnalysis`/`Bulletin17CAnalysis`, bound in both packages
   (`bestfitr/R/analysis.R` + `src/analysis.cpp`, `bestfitpy/.../analysis.py` +
   `src/bindings/analysis.cpp`) with a new `analysis` fixture kind across all three runners; the
   Numerics output types `ProbabilityOrdinates` (ordinate grid + 25 default exceedance
   probabilities) and `UncertaintyAnalysisResults`; the DataFrame bootstrap surface
   (JackKnife/Resample/BootstrapDataFrame/ShiftDistribution + `Random.NextIntegers`, consumed ONLY
   by Bulletin17CAnalysis, un-deferred from Phase 5/6 and added additively to `data_frame.hpp`); the
   three Bulletin17CAnalysis UQ paths shipped (MultivariateNormal default + parametric Bootstrap +
   the deterministic Cohn-style delta-method CI); and the CS0104 YeoJohnsonLink emitter patch that
   unlocks the minimal Analyses closure. Fixture-validated in C++/R/Python and reproduced against C#
   by the dotnet oracle gate (3950 reproduced, 0 failed, 11 GEV std-err skips; ctest 56/56; testthat
   3585/0; pytest 574). Pending CI run and PR.
   [DONE — the per-family analyses + Diagnostics layer (Phase 9a)] The remaining univariate-family
   analysis orchestrators (`mixture_analysis` / `point_process_analysis` / `competing_risk_analysis`
   under `analyses/univariate/`) and the four TimeSeries analyses (`ar_analysis` / `ma_analysis` /
   `arima_analysis` / `arimax_analysis` under `analyses/time_series/`) -- mechanical Bayesian clones
   of the Phase-8 UnivariateAnalysis (validate -> `BayesianAnalysis.estimate()` ->
   UncertaintyAnalysisResults). The essential Diagnostics layer (`diagnostics/`): `LeverageDiagnostics`
   (Cook's-distance + variance-influence decomposition at the MAP point via a numerical Hessian),
   `InfluenceDiagnostics` (PSIS-LOO Pareto-k wrapper), and `PriorInfluenceDiagnostics` (prior-to-data
   influence off the seeded posterior) -- wiring them un-stubbed the 6 previously-throwing estimator
   methods (MAP `compute_leverage_diagnostics`; BayesianAnalysis `compute_influence_` /
   `compute_prior_influence_` / `compute_leverage_diagnostics`; and the GMM quartet
   `get_observation_influence` / `get_cooks_distance` / `get_influence_diagnostics` x2 /
   `get_leverage_diagnostics`). All seven analyses plus a diagnostics accessor are user-callable in
   both packages, and the emitter now compiles the REAL Diagnostics classes in place (replacing the
   deleted `DiagnosticsStubs.cs`) and drives all seven analyses. Fixture-validated in C++/R/Python and
   reproduced against C# by the dotnet oracle gate (4003 reproduced, 0 failed, 14 skipped; ctest
   60/60; testthat 3687/0; pytest 590). Fidelity notes (docs/upstream-csharp-issues.md): the
   AR/MA/Mixture seeded-DEMCzs analysis curves diverge C#-vs-C++ by chaotic short-chain sensitivity
   (deterministic densities match C# to <= 3 ulp, Mixture bit-identical, across 238 param vectors),
   so those three fixtures assert only structural invariants (curve_length; AR mode_curve[0]) with NO
   oracle_skip and NO tolerance loosening (Phase-3 HMC/NUTS precedent); CompetingRisk/PointProcess
   tightened to exact; ARIMA/ARIMAX structural. Three PriorInfluenceDiagnostics assertions carry
   oracle_skip (the ported empty ModelParameter names collapse the two Normal parameter priors -- a
   deterministic dedup, not a stream divergence). Pending CI run and PR.
   Deferred to Phase 10: CompositeAnalysis + the Numerics `BootstrapAnalysis` frequentist engine +
   WeightedUnivariateAnalysis; SpatialGEVAnalysis (+ SpatialGEVSiteResults/CrossValidationResults
   DTOs); BivariateAnalysis + CoincidentFrequencyAnalysis; RatingCurveAnalysis; the
   LinkedMultivariateNormal path (its ~13 link-builder helpers + InfluenceStatistics) and the
   BiasCorrected / pivot Bootstrap; and the two predictive checks (PosteriorPredictiveCheck /
   PriorPredictiveCheck). PERMANENT SKIPS: GMM report generation (presentation-only text) and
   BatchAnalysisRunner + options (GUI batch scheduling).
   Carried-forward BUG (A11): seeded DEMCz/DEMCzs runs with `thinning_interval > 1` are NOT
   oracle-guaranteed C#-vs-C++ until bisected (thin=1 is bit-identical; every shipped Bayesian
   fixture uses thin=1).
   [DONE — FULL PARITY (Phase 10)] The final parity tail: the five remaining analysis orchestrators
   (RatingCurveAnalysis; BivariateAnalysis + CoincidentFrequencyAnalysis; SpatialGEVAnalysis + its
   two DTOs; CompositeAnalysis + WeightedUnivariateAnalysis); the Numerics BootstrapAnalysis engine
   (+ the IBootstrappable mixin on Normal); the two un-gated Bulletin17C uncertainty arms
   (LinkedMultivariateNormal + the pivot/BiasCorrected bootstrap); the four predictive checks
   (PosteriorPredictiveCheck/PriorPredictiveCheck/PredictiveCheckResults/PredictiveSummary,
   completing the Diagnostics layer); and the Phase-1 follow-ups (distribution ParameterNames + the
   mutable UserDefined MCMC seeding hook wired into MixtureAnalysis's EM seed). All bound in R and
   Python, fixture-validated in C++/R/Python, and reproduced against the real Numerics/RMC.BestFit
   libraries by the dotnet oracle gate (4069 reproduced, 0 failed, 11 skipped; ctest 69/69; testthat
   3770/0; pytest 606). Only permanent skips remain: GMM report generation and BatchAnalysisRunner +
   options (both presentation-only). With this, every distribution, the multivariate/copula layer,
   MCMC + bootstrap, all estimators (MLE/MAP/Bayesian/GMM/BootstrapAnalysis + the B17C LinkedMVN +
   pivot uncertainty), all model families, the complete Analyses layer, the complete Diagnostics
   layer, and the user-facing R/Python API are ported and validated. Pending: the CI run and PR ship
   step (a separate workflow run).

Each phase merges only when its fixtures pass in all three harnesses and all CI jobs are green.

## Documentation

- **R:** roxygen2 → `man/`, pkgdown site, vignettes mirroring the `examples/*.md` walkthroughs.
- **Python:** numpydoc docstrings, Sphinx or mkdocs-material site, runnable example notebooks.
- Shared: a porting/architecture doc and a "validated against C# BestFit vX" provenance note.

## Verification

- **C++:** `cmake --build core && ctest` — doctest suite green incl. RNG reference-stream test.
- **R:** `R CMD build bestfitr && R CMD check --as-cran bestfitr` clean on all 3 platforms;
  `testthat` reads `inst/fixtures` and passes.
- **Python:** `cibuildwheel` builds wheels on all 3 platforms; `pytest` (reading packaged fixtures)
  passes inside each wheel; `pip install` from sdist succeeds.
- **Cross-language identity:** a harness script confirms (a) every fixture passes identically in
  C++/R/Python to its stated tolerances, and (b) a seeded MCMC chain is byte-identical between R and
  Python. Spot-check selected outputs against the original C# tests to confirm no oracle drift.
- **Phase-0 gate** is the first real proof; no bulk porting begins until it is green.
