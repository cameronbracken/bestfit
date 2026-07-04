# CLAUDE.md — bestfit

Context for Claude Code working in the `bestfit` repo. See `PLAN.md` (same dir) for the
full approved architecture and phasing.

## What this is

`bestfit` provides **R (`bestfitr`) and Python (`bestfitpy`) packages** for stochastic
hydrology / flood-frequency / extreme-value analysis, built on a **single shared C++17 core**
that is a faithful port of the USACE-RMC C# libraries **Numerics** and **RMC.BestFit**.

Write the math once in C++, bind it twice (cpp11 for R, pybind11 for Python). Because both
packages run the same compiled code with a bit-exact Mersenne Twister, seeded results are
identical across R and Python. Hard requirement: **publishable to CRAN and PyPI**.

Upstream C# sources are vendored as **dev-only git submodules** at `upstream/Numerics` and
`upstream/RMC-BestFit` (shallow, tracking the official USACE-RMC `main` branches; pinned via
gitlink). They are the diff baseline for the upstream-sync workflow and are NOT referenced by
either package (CRAN/PyPI sdists and CI are unaffected; `actions/checkout` uses `submodules:
false`). See `upstream/CLAUDE.md` for the per-library architecture notes (consolidated there and
tracked by this repo). `dotnet` **is now installed**, so oracle values are curated from the C#
test files AND verified reproducible against the real Numerics library (see below). (The user's
own forks live elsewhere on disk under different names.)

## Layout & the vendoring invariant

- `core/` — **canonical** C++17 core (`include/`, `src/`, `tests/`, `CMakeLists.txt`). All
  numerical development happens here.
- `fixtures/` — **canonical** language-neutral oracle fixtures (JSON). Single source of truth
  for expected values; see `fixtures/README.md` for the schema.
- `bestfitr/`, `bestfitpy/` — the packages. Each contains a **generated, committed copy** of the
  core at `src/bestfit_core/` and (R) `inst/fixtures` / (Py) `src/bestfitpy/fixtures`.
- `tools/sync_core.py`, `tools/sync_fixtures.py` — copy canonical → vendored. Run after ANY
  change to `core/` or `fixtures/`. CI runs `--check` and fails on drift. **If you edit a core
  header, re-run `python3 tools/sync_core.py` or the packages build stale code.**
- `tools/oracle_emitter/` (C#) + `tools/verify_oracles.py` — the dotnet oracle gate: replays every
  fixture against the real Numerics library and fails on any value that doesn't reproduce to
  tolerance. **Dev-only** (needs `dotnet` + the submodule); not wired into CI.
- An auto-scraper to harvest the C# test literals *en masse* is still planned for the bulk port;
  for now fixtures are curated by hand and confirmed by the dotnet gate.

The univariate distribution layer lives under `core/include/bestfit/numerics/distributions/`:
`base/` holds `UnivariateDistributionBase`, the type enum, the factory, and the `IEstimation` /
`ILinearMomentEstimation` capability mixins; all 42 distributions derive from the base.
`distributions/multivariate/` holds Dirichlet, Multinomial, BivariateEmpirical,
MultivariateNormal (with the ported Genz MVNDST integrator), and MultivariateStudentT, plus the
`MultivariateDistributionBase` and factory. `distributions/copulas/` holds all seven bivariate
copulas (Clayton, AliMikhailHaq, Frank, Gumbel, Joe, Normal, StudentT), the shared
`BivariateCopula`/`ArchimedeanCopula` base classes, `BivariateCopulaEstimation` (tau/MPL/IFM/MLE
fits), the `IMaximumLikelihoodEstimation` mixin, and the copula factory.

`core/include/bestfit/numerics/sampling/mcmc/` holds the MCMC subsystem: `mcmc_sampler.hpp` (the
shared base -- seeding cascade, chain initialization incl. the MAP/DE/Hessian path, the serial
`sample()` driver), all 8 concrete samplers (`rwmh.hpp`, `arwmh.hpp`, `demcz.hpp`, `demczs.hpp`,
`hmc.hpp`, `nuts.hpp`, `gibbs.hpp`, `snis.hpp`), `model_registry.hpp` (the bestfit-addition model
registry the fixtures build against), and the diagnostics/results support headers (Gelman-Rubin
R-hat, ESS, `MCMCResults`/`MCMCDiagnostics`). `core/include/bestfit/numerics/sampling/bootstrap/`
holds the regular (non-pivotal) `Bootstrap<TData>` port -- `Run`/`RunDoubleBootstrap`/
`RunWithStudentizedBootstrap`, the five `GetConfidenceIntervals` methods, and its own model
registry (the covariance-aware pivotal workflow is a documented omission, tracked as a separate
severable follow-up -- see the file header). `core/include/bestfit/numerics/math/optimization/`
holds `ParameterSet`, `Optimizer` (the shared base every optimizer -- currently
DifferentialEvolution -- builds on), and
`DifferentialEvolution` itself, the global optimizer the MCMC MAP-initialization path depends on.

`core/include/bestfit/models/` holds the RMC.BestFit Models port (Phase 5). `models/data_frame/`
is the input-data container layer: `data_types/` (the `Data` base plus ExactData, IntervalData,
UncertainData, ThresholdData), `data_collections/` (the generic `DataSeries` plus the four typed
series), `data_frame.hpp` (`DataFrame` -- FullTimeSeries threshold expansion,
ProcessThresholdSeries, MGBT and explicit-threshold low outliers) with `data_frame_plotting.hpp`
(the Hirsch-Stedinger plotting positions, including a faithful port of .NET's ArraySortHelper
introsort because `List<T>.Sort` tie order is oracle-visible), and `threshold_diagnostics.hpp`
(mean residual life + GPD parameter stability). `models/trend_functions/` holds the ten trend
models plus `general_linear_function.hpp` on the shared `support/` base
(ITrendModel/TrendModelBase and the type enum). `models/univariate_distribution/` holds the four
models: `univariate_distribution_model.hpp` (with its nonstationary companion
`univariate_distribution_model_trends.hpp` and `base/univariate_distribution_model_base.hpp`),
`mixture_model.hpp`, `competing_risks_model.hpp`, and `point_process_model.hpp`.
`models/support/` carries the Phase 4 model-support types (ModelParameter, DataComponent,
PriorComponent, ModelBase, QuantilePrior and its interfaces) plus `subscript_formatter.hpp` and
`validation_result.hpp`; `models/json_lite.hpp` + `models/model_spec.hpp` are the shared fixture
spec builder all three runners and the oracle emitter drive. The MultipleGrubbsBeckTest
low-outlier test lives at `numerics/data/multiple_grubbs_beck_test.hpp`.

## Build & test commands

```bash
# C++ core
cmake -S core -B core/build && cmake --build core/build && ctest --test-dir core/build
# R  (regenerate registration only after editing bestfitr/src/*.cpp)
Rscript -e 'cpp11::cpp_register("bestfitr")'; R CMD INSTALL bestfitr
Rscript -e 'testthat::test_local("bestfitr")'
# Python (dev venv at ~/venv/bestfitpy)
~/venv/bestfitpy/bin/python -m pip install --force-reinstall --no-deps ./bestfitpy
~/venv/bestfitpy/bin/python -m pytest bestfitpy/tests -q
# sync guards (must pass before commit)
python3 tools/sync_core.py --check && python3 tools/sync_fixtures.py --check
# oracle reproduction gate (dev-only; needs dotnet + the upstream submodule)
python3 tools/verify_oracles.py
```

Toolchain present: clang++, cmake, R 4.6.1 (`/opt/homebrew/bin`), python3.14 + `~/venv/bestfitpy`,
roxygen2/jsonlite/testthat/cpp11 installed, and **dotnet 10** (for the oracle gate; not in CI).
After any core change that alters a class layout, rebuild R clean (`R CMD INSTALL --preclean
bestfitr`) — stale `bestfitr/src/*.o` from a prior ABI can otherwise return garbage / abort R.

## Validation model (DRY)

Oracle values live ONLY in `fixtures/*.json`. Three thin generic runners load the same JSON and
apply every assertion: C++ `core/tests/test_fixtures.cpp` (nlohmann/json, vendored test-only under
`core/tests/third_party/`), R `bestfitr/tests/testthat/test-fixtures.R` (jsonlite), Python
`bestfitpy/tests/test_fixtures.py` (stdlib). The runners are **polymorphic**: non-GEV targets are
built through the factory and dispatched on `UnivariateDistributionBase` (+ capability casts) via
the `bf_dist_*` (R) / `_core.dist_*` (Py) glue; GEV keeps a bespoke path for its standard-error
methods. **Adding a distribution = new fixture file + a couple of dispatch entries per runner** —
no new per-distribution glue. Don't hardcode oracle values in test files. The dotnet gate
(`verify_oracles.py`) is the fourth, dev-only check that the fixtures still match the C# source.

## Conventions & gotchas

- **Structural mirroring:** C++ mirrors the C# file/class/method layout so upstream diffs map
  almost line-for-line. Each ported file carries a `// ported from: <path> @ <sha>` header.
- **Portability (learned from CI):** never use `M_PI` (absent under strict `-std=c++17` on Linux
  and on MSVC) — use `bestfit::numerics::kPi`. Don't name a namespace alias `gamma` (clashes with
  glibc's libm `gamma()`) or `stat` (clashes with the MSVC/POSIX CRT `stat` symbol). Pass
  `-Wall/-Wextra` only to non-MSVC compilers in CMake.
- **Self-contained core:** no external C++ deps (port Numerics' own linear algebra / RNG). Keeps
  the CRAN dependency surface empty and preserves oracle fidelity. Don't add Eigen to the core.
- **CRAN:** `bestfitr` uses `License: file LICENSE` (R can't standardize the `0BSD` token).
  Makevars: `CXX_STD = CXX17`, no `-O3/-march/-Werror`. cpp11 internal functions (`bf_gev_*`,
  `bf_dist_*`) are unexported — tests reach them via `asNamespace("bestfitr")`. After editing any
  `bestfitr/src/*.cpp`, re-run `cpp11::cpp_register("bestfitr")`.
- **Mutation:** the global "never mutate" rule is relaxed for these binding/model objects (they
  mirror the C# stateful API), matching the upstream design.

## Git & CI

- Commits are GPG-signed automatically (key `F4C82FB462F850C1`, "Cam Bracken (GitHub)"),
  verified on GitHub. Identity: `Cam Bracken <cameron.bracken@pm.me>`. Push only when asked.
- `.github/workflows/ci.yml`: `sync-check` gate → `core` (3 OS) + `r-cmd-check` (3 OS) +
  `python` (3 OS × {3.10, 3.12}). Use `gh run watch <id> --exit-status` to follow a run.
- Never commit: OS junk, IDE settings, secrets, the dotnet build output, other AI-tool files
  (`.gitignore` covers these). **Exception (deliberate):** this repo *does* track three curated
  context files — `.claude/CLAUDE.md`, `.claude/PLAN.md`, and `upstream/CLAUDE.md` — because the
  C++/R/Python-from-C# port is a complex process whose plan and porting guidance should travel with
  the repo. The `.gitignore` un-ignores exactly those three; everything else in `.claude/` (settings,
  scratch plans) stays ignored.

## Status

Phase 0, Phase 1, Phase 2, Phase 3, Phase 4, and Phase 5 are **complete**; Phases 1-4 are merged
(latest: PR #6) with CI green on the full matrix. Phase 1 delivered the full
Numerics math/RNG foundation plus all 42 univariate distributions; CI is green on 3 platforms for
that merge. Phase 2 delivered the multivariate distributions and copula layer -- Dirichlet, Multinomial,
BivariateEmpirical, MultivariateNormal (Genz MVNDST), MultivariateStudentT; all seven bivariate
copulas with shared estimation (tau/MPL/IFM/MLE) and an `IMaximumLikelihoodEstimation` mixin; and
CompetingRisks' correlated dependency modes un-deferred from Phase 1. Phase 3 delivered
Sampling/MCMC -- all 8 samplers (RWMH, ARWMH, DEMCz, DEMCzs, HMC, NUTS, Gibbs, SNIS) on the shared
`MCMCSampler` base plus diagnostics/results (Gelman-Rubin R-hat, ESS) and the DifferentialEvolution
optimizer stack the MAP-initialization path needs -- and the regular (non-pivotal) Bootstrap
workflow (Percentile/BiasCorrected/BCa/Normal/BootstrapT); the covariance-aware pivotal bootstrap
was scoped as the phase's severable final task and is tracked separately rather than landing on
this branch. Phase 4 delivered BestFit's `Estimation` layer -- MaximumLikelihood,
MaximumAPosteriori, and BayesianAnalysis (DEMCz/DEMCzs/ARWMH/NUTS, with DIC/WAIC/LOOIC
diagnostics) on top of the Models slice (ModelParameter/DataComponent/PriorComponent, IModel/
ModelBase, UnivariateDistributionModel including the Jeffreys 1/scale prior) and the Estimation
support layer (MatrixRegularization, Stratify/StratificationOptions, NumericalDiff,
OptimizationMethod); GMM + IGMMModel + BFGS + the Bulletin17C coupling, the alternate optimizers
(Powell/MLSL/LocalMethod), and the Diagnostics/LeverageDiagnostics layer are severed to follow-ups.
Phase 5 delivered BestFit's flood-frequency Models core, the first slice of the Models phase --
MultipleGrubbsBeckTest; the DataFrame layer (censored data types, four series collections,
FullTimeSeries/ProcessThresholdSeries/MGBT low outliers, Hirsch-Stedinger plotting positions via a
faithful .NET ArraySortHelper introsort port, ThresholdDiagnostics); the trend/link functions
including GeneralLinearFunction; and the four models (nonstationary + censored
UnivariateDistributionModel extended in place, MixtureModel with EM + zero inflation,
CompetingRisksModel, non-seasonal PointProcessModel), with the `model_estimation` fixture kind
extended across all three runners. M14 re-pinned every Phase 5 fixture against real C# oracles and
fixed a real port divergence (the C# `Mixture.SetParameters(ref)` weight-normalization write-back
into the optimizer's arrays; the documented residual deviation is that BayesianAnalysis hands the
const-ref MCMC samplers a mutable copy). Severed from Phase 5: the DataFrame
hypothesis-test/summary-statistics facades, the seasonal PointProcess path +
GeneratePOTTimeSeries + CreateBlockSeries (need the unported TimeSeries container), the DataFrame
bootstrap/resampling surface (Phase 6), ExactData.DateTime, FittedDistribution, USGSRawText, and
all XML/INPC. Everything ported through Phase 5 is fixture-validated in C++/R/Python and
reproduced against the real Numerics/RMC.BestFit libraries by the dotnet oracle gate (3837
reproduced, 0 failed, 11 documented GEV std-err skips); seeded MCMC chains, bootstrap replicate
streams, a DEMCzs posterior chain digest, and the Phase 5 model simulation digests are all proven
bit-identical across R and Python via `short_exact`-style digest fixtures. Pending: CI run and PR
for the Phase 5 branch. See `PLAN.md`.
