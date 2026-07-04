# Plan: `bestfitr` (R) + `bestfitpy` (Python) from a shared C++ core

> **Current status (kept in sync by hand):** Phase 0, Phase 1, Phase 2, Phase 3, and Phase 4 are
> **complete**.
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
> C++/R/Python and reproduced against the real Numerics library by the dotnet oracle gate. Pending
> CI run and PR.
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
> separately as a follow-up rather than landing on this branch. Pending CI run and PR.
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
> Pending CI run and PR.
>
> CI is green on the full matrix (`sync-check`, `core`, `r-cmd-check`, `python`) on
> Linux/macOS/Windows as of the Phase 2 merge (PR #4); the Phase 3 and Phase 4 branches have not
> yet been pushed for CI. The dotnet oracle gate is dev-only (not in CI).
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
│   ├── src/{bestfit_core/,bindings/,bestfitpy/}   # bestfit_core/ is GENERATED+committed
│   └── tests/
├── tools/{sync_core.py,sync_fixtures.py,extract_oracles.py,upstream_diff.py}
└── .github/workflows/
```

`tools/sync_core.py` copies `core/{include,src,data}` into both packages **and** regenerates one
source manifest in two forms — `core_sources.mk` (R `Makevars` `OBJECTS`) and `core_sources.cmake`
(CMake `include()`). Adding a core file is picked up by both builds after one sync. Avoid build
wildcards (CRAN dislikes them) — the manifest is explicit.

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
(`"mean" → Mean / $mean() / .mean()`) — the only per-language test code. `sync_fixtures.py` copies
`fixtures/` into `bestfitr/inst/fixtures/` and bestfitpy package data.

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
   gate. Pending CI run and PR.
4. Sampling/MCMC (RWMH, ARWMH, DEMCz/zs, HMC, NUTS, Gibbs, SNIS) + Bootstrap — fixture digests prove
   identical seeded chains across R/Python.
5. BestFit `Estimation` (MaximumLikelihood, MAP, GMM, BayesianAnalysis, NumericalDiff, OptimizationMethod).
   [DONE] MaximumLikelihood, MaximumAPosteriori, and BayesianAnalysis (DEMCz/DEMCzs/ARWMH/NUTS)
   with diagnostics (DIC/WAIC/LOOIC), NumericalDiff, OptimizationMethod, MatrixRegularization,
   Stratify/StratificationOptions, and the Models slice (ModelParameter/DataComponent/
   PriorComponent, IModel/ModelBase, UnivariateDistributionModel) ported, fixture-validated in
   C++/R/Python, and reproduced against C# by the dotnet oracle gate. GMM + IGMMModel + BFGS +
   the Bulletin17C coupling severed to a follow-up. Pending CI run and PR.
6. BestFit `Models` (UnivariateDistribution, Bulletin17C, Mixture, CompetingRisks, PointProcess,
   Bivariate, RatingCurve, TimeSeries, SpatialExtremes, Trend/Link functions) — boilerplate skipped,
   DataFrame as adapter.
7. BestFit `Analyses` + `Diagnostics` — the user-facing API (`univariate_analysis()` etc.).

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
