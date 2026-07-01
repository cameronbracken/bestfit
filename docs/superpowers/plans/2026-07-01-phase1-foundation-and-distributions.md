# Phase 1 — Numerics Foundation + All Univariate Distributions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Phase 1 of the `bestfit` port — port the Numerics special-function/RNG/statistics foundation the distributions need, then port the remaining ~38 univariate distributions, each fixture-validated bit-identically in C++/R/Python and reproduced against the real C# Numerics library.

**Architecture:** All numerical code is a method-for-method mirror-port of the vendored C# sources in `upstream/Numerics/` into the canonical header-only C++17 core under `core/`. The validation pipeline (C++ runner, R cpp11 glue, Python pybind11 glue, and the dotnet oracle emitter) is already fully generic: it dispatches by the fixture `target` string through `UnivariateDistributionFactory`, and the `UnivariateDistributionType` enum already lists every distribution. Therefore a **standard** distribution (flat `std::vector<double>` parameters) needs exactly three edits — a new header, a factory entry, a fixture JSON — and nothing in any binding, runner, or emitter. A small set of composite/data-driven distributions break the flat-parameter contract and get a bespoke path (like GEV).

**Tech Stack:** C++17 (header-only core, no external deps), cpp11 (R), pybind11 + scikit-build-core (Python), nlohmann/json (C++ test-only), jsonlite (R), stdlib json (Python), .NET 10 (dev-only oracle gate), CMake + ctest, testthat, pytest.

## Global Constraints

- **C++ standard:** C++17 only. `CXX_STD = CXX17`. No C++20.
- **Portability:** never use `M_PI` — use `bestfit::numerics::kPi`. Never name a namespace alias `gamma` (clashes with glibc `gamma()`). Pass `-Wall/-Wextra` only to non-MSVC compilers. No `-O3`, `-march=native`, LTO, or `-Werror` in `Makevars`.
- **Self-contained core:** no external C++ deps. Port Numerics' own algorithms. Do NOT add Eigen, Boost, or any library to the core.
- **Structural mirroring:** each C++ file mirrors the C# file/class/method layout and order. Every ported file opens with `// ported from: <upstream-path> @ <sha>`.
- **Vendoring invariant:** after ANY change under `core/` run `python3 tools/sync_core.py`; after ANY change under `fixtures/` run `python3 tools/sync_fixtures.py`. CI `sync-check` fails on drift.
- **No hardcoded oracle values in test files.** Oracle values live ONLY in `fixtures/*.json`.
- **Mutation:** the global "never mutate" rule is relaxed for distribution/model objects (they mirror the C# stateful API).
- **ABI safety:** after a core change that alters a class layout, rebuild R clean with `R CMD INSTALL --preclean bestfitr` or stale `.o` files return garbage / abort R.
- **cpp11:** after editing any `bestfitr/src/*.cpp`, re-run `Rscript -e 'cpp11::cpp_register("bestfitr")'`.
- **Commits:** GPG-signed automatically. Identity `Cam Bracken <cameron.bracken@pm.me>`. Conventional-commit messages. Push only when asked.
- **Git identity for this tree is personal**, never the PNNL identity.

---

## How to use this plan (mirror-port convention)

This is a **mechanical mirror-port**, so the literal numerical spec for each distribution is its upstream C# file — the executor transcribes it method-for-method into C++ rather than inventing math. Duplicating ~15k lines of upstream C# into this plan would be error-prone and violate DRY, so per-item task cards do **not** re-transcribe the C# body. Instead each card gives everything that is *not* a mechanical transcription and everything an executor needs to not go wrong:

- the exact **upstream C# source path** to port from (the literal spec),
- the exact **C++ target path**,
- the **constructor parameters** and their order,
- the **special-function dependencies** (which gate the task on a foundation task),
- whether it implements `IEstimation` / `ILinearMomentEstimation`,
- the **exact factory edit** (fully literal),
- the **fixture cases/methods** to include and their oracle source,
- whether it is **standard** (3-edit path) or **bespoke** (needs a schema/binding extension),
- the exact **verification commands**.

Two fully-worked exemplars are given verbatim — **Task F0 (Gumbel)** for a standard distribution and **Task A2 (Erf)** for a special-function module. Every other card says "follow the Task F0 recipe" (or A2) and lists only its distribution-specific facts. Read the exemplar before working any card.

**Definition of done for every distribution/foundation task:** `cmake --build core/build && ctest --test-dir core/build` green; `python3 tools/sync_core.py && python3 tools/sync_fixtures.py` clean; R fixtures pass; Python fixtures pass; `python3 tools/verify_oracles.py` green (the C# reproduces every fixture value). Then commit.

---

## Milestone map & dependency order

```
A. Foundation: special-function fixture infra + Erf, Gamma-ext, Beta, Factorial, Bessel      (prereqs)
B. Wave 1 — trivial closed-form distributions (need nothing new; can start immediately)
C. Wave 2 — Normal-derived (need Erf: A2)
D. Wave 3 — Gamma family (need Gamma-ext: A3)
E. Wave 4 — Gamma-skew family (need A3; harder)
F. Wave 5 — Beta family (need Beta+Factorial: A4, A5)
G. Wave 6 — exotic: VonMises (need Bessel: A6 + a numerical integrator: A7)
H. Wave 7 — composite / data-driven (BESPOKE: schema + binding extensions)
I. Remaining Phase-1 foundation breadth — Sobol RNG + goodness-of-fit stats
J. Wrap-up — docs, status refresh, CI green, PR
```

Waves B–G are independent of each other except through their foundation gate; within a wave, distributions are independent and parallelizable. Wave B needs no foundation and can run first or in parallel with Milestone A. Milestone H is the risky tail and may be split into its own follow-on plan (see H0).

**Distributions covered (38 remaining):** Bernoulli, Beta, Binomial, Cauchy, ChiSquared, CompetingRisks, Deterministic, Empirical, GammaDistribution, GeneralizedBeta, GeneralizedLogistic, GeneralizedNormal, GeneralizedPareto, Geometric, Gumbel, InverseChiSquared, InverseGamma, KappaFour, KernelDensity, LnNormal, Logistic, LogNormal, LogPearsonTypeIII, Mixture, NoncentralT, Pareto, PearsonTypeIII, Pert, PertPercentile, PertPercentileZ, Poisson, Rayleigh, StudentT, Triangular, TruncatedNormal, UniformDiscrete, VonMises, Weibull. (Already ported: Normal, Uniform, Exponential, GeneralizedExtremeValue.)

---

## Milestone A — Foundation: special functions

The core currently has (in `math/special/gamma.hpp`) only `function` (Γ), `lanczos`, `stirling`, `digamma`, plus `detail::polynomial_rev`. The distributions collectively need the regularized incomplete gamma/beta and their inverses, log-gamma, trigamma, error-function inverse, factorial/binomial-coefficient, and Bessel I0/I1. These are the highest-risk numerical code (continued fractions), so validate them directly with oracle fixtures.

### Task A1: Special-function fixture kind + runners

**Files:**
- Modify: `fixtures/README.md` (document the new `special_function` kind)
- Create: `fixtures/special_functions/erf.json` (skeleton written here, filled by A2)
- Modify: `core/tests/test_fixtures.cpp` (add a `special_function` branch)
- Modify: `tools/oracle_emitter/Program.cs` (add a `special_function` branch)
- Test: reuses `core/tests/test_fixtures.cpp`

**Interfaces:**
- Produces: a fixture schema `{ "kind": "special_function", "target": "<Module.method>", "cases": [ { "name", "args": [..], "assertions": [ { "method": "value", "expected", "mode", "tol" } ] } ] }`. The runner maps `target` (e.g. `"Erf.function"`, `"Gamma.lower_incomplete"`) to a C++ free-function pointer and calls it with `args`.
- Consumes: nothing.

Special functions are **not** exposed to R/Python (they are internal core math, YAGNI for the public API). They are validated in C++ + the dotnet gate only. Ensure the R and Python fixture runners **skip** any file whose `kind != "univariate_distribution"` (they must not choke on the new kind).

- [ ] **Step 1: Write the failing test** — add to `core/tests/test_fixtures.cpp` a `run_special_function(const json& spec)` that resolves `spec["target"]` through a `std::map<std::string, std::function<double(const std::vector<double>&)>>` dispatch table and checks each case's assertions with the existing `check_value`. In `main()`, add `else if (spec.value("kind","") == "special_function") run_special_function(spec);`. Seed the dispatch table with one entry `{"Erf.function", [](auto& a){ return bestfit::numerics::math::special::erf::function(a[0]); }}` (function added in A2 — expect a compile error now).

- [ ] **Step 2: Run to verify it fails** — `cmake -S core -B core/build && cmake --build core/build` → FAIL (no `erf::function`).

- [ ] **Step 3: Guard the R/Python runners** — in `bestfitr/tests/testthat/test-fixtures.R` and `bestfitpy/tests/test_fixtures.py`, skip specs where `kind != "univariate_distribution"` before building. (Distributions still validate; special-function specs are ignored.)

- [ ] **Step 4: Extend the dotnet emitter** — in `tools/oracle_emitter/Program.cs`, add a branch for `kind == "special_function"` that resolves `target` (`"Erf.function"` → `Numerics.Mathematics.SpecialFunctions.Erf.Function`, etc.) and compares to `expected`. Mirror the C++ dispatch names.

- [ ] **Step 5: Commit** — `git commit -m "feat(core): add special_function fixture kind + runners"` (A2 will add the first real module and fixture; commit the infra with a minimal placeholder erf entry compiling once A2 lands, or fold A1+A2 into one commit if preferred).

### Task A2: Erf module (WORKED EXEMPLAR for special-function tasks)

**Upstream:** `upstream/Numerics/Numerics/Mathematics/Special Functions/Erf.cs`
**Files:**
- Create: `core/include/bestfit/numerics/math/special/erf.hpp`
- Create: `fixtures/special_functions/erf.json`
- Modify: `core/tests/test_fixtures.cpp` (dispatch entries)
- Test: `core/tests/test_fixtures.cpp`

**Interfaces:**
- Produces (namespace `bestfit::numerics::math::special::erf`):
  `double function(double x);` `double erfc(double x);` `double inverse_erf(double y);` `double inverse_erfc(double y);`
- Consumes: `bestfit::numerics::kSqrt2` etc. from `tools.hpp`.

Rationale for the module even though `std::erf` exists: `inverse_erf`/`inverse_erfc` are not in the standard library and are needed by TruncatedNormal and others; `function`/`erfc` are ported for bit-fidelity where a distribution's oracle traces to `Erf.cs`. Normal already passes using `std::erf`, so keep Normal as-is unless a fixture drifts.

- [ ] **Step 1: Write the failing fixture-driven test** — create `fixtures/special_functions/erf.json`:

```json
{
  "kind": "special_function",
  "target": "Erf.function",
  "source": "Numerics/Test_Numerics/Mathematics/Special Functions/Test_SpecialFunctions.cs",
  "cases": [
    { "name": "erf_zero",  "args": [0.0], "assertions": [ { "method": "value", "expected": 0.0, "mode": "equal" } ] },
    { "name": "erf_one",   "args": [1.0], "assertions": [ { "method": "value", "expected": 0.8427007929497149, "mode": "abs", "tol": 1e-12 } ] },
    { "name": "erf_neg",   "args": [-1.0], "assertions": [ { "method": "value", "expected": -0.8427007929497149, "mode": "abs", "tol": 1e-12 } ] }
  ]
}
```

  Add sibling fixture files or extra `target`s for `Erf.erfc`, `Erf.inverse_erf`, `Erf.inverse_erfc`, using the oracle values from `Test_SpecialFunctions.cs` (confirmed later by the dotnet gate). Add the four dispatch entries to `test_fixtures.cpp`'s special-function table.

- [ ] **Step 2: Run to verify it fails** — `cmake --build core/build && ctest --test-dir core/build` → FAIL (no `erf.hpp`).

- [ ] **Step 3: Port the module** — create `core/include/bestfit/numerics/math/special/erf.hpp`, mirroring `Erf.cs` method-for-method (rational/continued-fraction approximations verbatim; coefficient tables copied exactly). Provenance header required. Use `kPi`, never `M_PI`.

- [ ] **Step 4: Run to verify it passes** — `cmake --build core/build && ctest --test-dir core/build` → PASS.

- [ ] **Step 5: Sync + oracle gate** — `python3 tools/sync_core.py && python3 tools/sync_fixtures.py`; `python3 tools/verify_oracles.py` → the C# `Erf.*` reproduces every fixture value.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat(core): port Erf special functions with oracle fixtures"`.

### Task A3: Gamma extensions

**Upstream:** `upstream/Numerics/Numerics/Mathematics/Special Functions/Gamma.cs`
**Files:** Modify `core/include/bestfit/numerics/math/special/gamma.hpp`; create `fixtures/special_functions/gamma.json`; add dispatch entries.
**Produces (namespace `...math::special` reusing the existing gamma namespace):**
`double log_gamma(double z);` `double trigamma(double x);` `double lower_incomplete(double a, double x);` (regularized P) `double upper_incomplete(double a, double x);` (regularized Q) `double inverse_lower_incomplete(double a, double p);` `double inverse_upper_incomplete(double a, double q);`
**Recipe:** follow Task A2. Oracle from `Test_Gamma.cs` (has Test_Digamma, Test_Trigamma, incomplete-gamma methods with hardcoded arrays). Do not rename the existing `function`/`digamma`. **Gate:** unblocks Milestones D & E.

### Task A4: Beta module

**Upstream:** `upstream/Numerics/Numerics/Mathematics/Special Functions/Beta.cs`
**Files:** Create `core/include/bestfit/numerics/math/special/beta.hpp`; create `fixtures/special_functions/beta.json`; add dispatch entries.
**Produces (namespace `...math::special::beta`):**
`double function(double a, double b);` `double incomplete(double a, double b, double x);` (regularized Iₓ) `double incomplete_inverse(double a, double b, double y);`
Port the internal continued-fraction helpers (`incbcf`, `incbd`, `power_series`) as `detail::` functions. **Recipe:** Task A2. Oracle from `Test_Beta.cs`. **Gate:** unblocks Milestone F.

### Task A5: Factorial module

**Upstream:** `upstream/Numerics/Numerics/Mathematics/Special Functions/Factorial.cs`
**Files:** Create `core/include/bestfit/numerics/math/special/factorial.hpp`; create `fixtures/special_functions/factorial.json`; add dispatch entries.
**Produces (namespace `...math::special::factorial`):**
`double function(int n);` `double log_factorial(int n);` `double binomial_coefficient(int n, int k);`
Skip the combinatorics enumeration helpers (`FindCombinations`/`AllCombinations`) — not needed by any distribution (YAGNI). **Recipe:** Task A2. Oracle from `Test_SpecialFunctions.cs`. **Gate:** unblocks Binomial, Poisson.

### Task A6: Bessel I0/I1

**Upstream:** `upstream/Numerics/Numerics/Mathematics/Special Functions/Bessel.cs`
**Files:** Create `core/include/bestfit/numerics/math/special/bessel.hpp`; create `fixtures/special_functions/bessel.json`; add dispatch entries.
**Produces (namespace `...math::special::bessel`):** `double i0(double x);` `double i1(double x);`
Port ONLY the modified Bessel functions of the first kind order 0 and 1 (all VonMises needs). Skip J/Y/K/In unless a later consumer needs them (YAGNI). **Recipe:** Task A2. Oracle from `Test_Bessel.cs`. **Gate:** unblocks VonMises.

### Task A7: Adaptive numerical integrator (only if a distribution needs it)

**Upstream:** `upstream/Numerics/Numerics/Mathematics/Integration/AdaptiveSimpsonsRule.cs` (or `AdaptiveGaussKronrod.cs` if the target distribution's C# uses it).
**Files:** Create `core/include/bestfit/numerics/math/integration/adaptive_simpson.hpp`.
**Produces (namespace `...math::integration`):** `double integrate(const std::function<double(double)>& f, double a, double b, double tol = 1e-10, int max_depth = 50);`
**Do this task ONLY when reached by VonMises (G) or GeneralizedNormal (C) and the C# source confirms numerical integration is used for its CDF.** Otherwise skip (YAGNI). Validate via the distribution whose CDF consumes it (no separate integrator fixture required, but a small C++ unit test asserting ∫₀¹ x dx = 0.5 and a Gaussian integral is cheap insurance).

---

## Milestone B — Wave 1: trivial closed-form distributions

No foundation gate. Each is a **standard** distribution: follow the Task F0 (Gumbel) recipe exactly. All have closed-form PDF/CDF/InverseCDF. Batchable/parallelizable.

### Task F0: Gumbel (WORKED EXEMPLAR for standard distributions)

**Upstream:** `upstream/Numerics/Numerics/Distributions/Univariate/Gumbel.cs`
**Files:**
- Create: `core/include/bestfit/numerics/distributions/gumbel.hpp`
- Modify: `core/include/bestfit/numerics/distributions/base/univariate_distribution_factory.hpp`
- Create: `fixtures/distributions/univariate/gumbel.json`
- Test: `core/tests/test_fixtures.cpp` (unchanged — generic path)

**Interfaces:**
- Produces: `class Gumbel : public UnivariateDistributionBase, public IEstimation, public ILinearMomentEstimation`. Params `{ξ location, α scale}`, order `[xi, alpha]`. Type `UnivariateDistributionType::Gumbel`, name `"Gumbel"`.
- Consumes: `data::product_moments`, `data::linear_moments`, `NelderMead`, `kPi`, `kEuler`.

- [ ] **Step 1: Write the failing fixture** — create `fixtures/distributions/univariate/gumbel.json`. Standard-parameter cases below use analytically-exact values (safe to hardcode); add dataset-fit cases (mom/lmom/mle) using oracle values from `Test_Gumbel.cs`, confirmed later by the dotnet gate.

```json
{
  "target": "Gumbel",
  "kind": "univariate_distribution",
  "source": "Numerics/Test_Numerics/Distributions/Univariate/Test_Gumbel.cs",
  "cases": [
    {
      "name": "standard_pdf_cdf_moments",
      "construct": { "params": [0.0, 1.0] },
      "assertions": [
        { "method": "pdf",      "args": [0.0], "expected": 0.36787944117144233, "mode": "abs", "tol": 1e-12 },
        { "method": "cdf",      "args": [0.0], "expected": 0.36787944117144233, "mode": "abs", "tol": 1e-12 },
        { "method": "quantile", "args": [0.36787944117144233], "expected": 0.0, "mode": "abs", "tol": 1e-9 },
        { "method": "mean",     "args": [], "expected": 0.5772156649015329, "mode": "abs", "tol": 1e-12 },
        { "method": "median",   "args": [], "expected": 0.36651292058166435, "mode": "abs", "tol": 1e-12 },
        { "method": "mode",     "args": [], "expected": 0.0, "mode": "equal" },
        { "method": "sd",       "args": [], "expected": 1.2825498301618641, "mode": "abs", "tol": 1e-12 },
        { "method": "parameters_valid", "args": [], "expected": true, "mode": "bool" }
      ]
    }
  ]
}
```

- [ ] **Step 2: Run to verify it fails** — `cmake --build core/build && ctest --test-dir core/build` → FAIL: `create_distribution` throws "unknown distribution name: Gumbel".

- [ ] **Step 3: Port the header** — create `gumbel.hpp` mirroring `Gumbel.cs` method-for-method and mirroring the *structure* of `exponential.hpp` (constructors, `type()`, `number_of_parameters()==2`, `get_parameters()=={xi_,alpha_}`, `set_parameters(location,scale)` with `validate`, moments/support, closed-form `pdf`/`cdf`/`inverse_cdf`, `clone`, `estimate` dispatching MoM/L-moments/MLE, `parameters_from_linear_moments`/`linear_moments_from_parameters`, `get_parameter_constraints`+`mle`). Provenance header. Closed forms: `pdf = (1/α)·exp(-(z+exp(-z)))`, `cdf = exp(-exp(-z))`, `inverse_cdf = ξ - α·ln(-ln p)`, with `z=(x-ξ)/α`; verify every moment/L-moment formula against `Gumbel.cs`.

- [ ] **Step 4: Register in factory** — edit `univariate_distribution_factory.hpp`:

```cpp
// add with the other includes:
#include "bestfit/numerics/distributions/gumbel.hpp"

// add to the create_distribution(UnivariateDistributionType) switch:
        case UnivariateDistributionType::Gumbel:
            return std::make_unique<Gumbel>();

// add to the create_distribution(const std::string&) chain:
    if (name == "Gumbel") return create_distribution(UnivariateDistributionType::Gumbel);
```

- [ ] **Step 5: Run to verify it passes** — `cmake --build core/build && ctest --test-dir core/build` → PASS.

- [ ] **Step 6: Sync, build bindings, cross-language + oracle** —
```bash
python3 tools/sync_core.py && python3 tools/sync_fixtures.py
Rscript -e 'cpp11::cpp_register("bestfitr")'; R CMD INSTALL --preclean bestfitr
Rscript -e 'testthat::test_local("bestfitr")'
~/venv/bestfitpy/bin/python -m pip install --force-reinstall --no-deps ./bestfitpy
~/venv/bestfitpy/bin/python -m pytest bestfitpy/tests -q
python3 tools/verify_oracles.py
python3 tools/sync_core.py --check && python3 tools/sync_fixtures.py --check
```
All green; R and Python agree; the C# reproduces every Gumbel fixture value.

- [ ] **Step 7: Commit** — `git add -A && git commit -m "feat(dist): port Gumbel distribution"`.

The remaining Wave-1 distributions follow Task F0 exactly. Per card: source path, params (in factory-vector order), estimation interfaces, and any wrinkle.

### Task F1: Logistic
Upstream `.../Logistic.cs` → `distributions/logistic.hpp`. Params `[xi location, alpha scale]`. `IEstimation` (MoM, MLE), `ILinearMomentEstimation`. Closed form. Factory: `Logistic`.

### Task F2: Pareto
`.../Pareto.cs` → `pareto.hpp`. Params `[Xm scale, alpha shape]` (confirm order in C#). `IEstimation` + `ILinearMomentEstimation`. Support `[Xm, ∞)`. Factory: `Pareto`.

### Task F3: Rayleigh
`.../Rayleigh.cs` → `rayleigh.hpp`. Param `[sigma scale]` (1 param). `IEstimation` (MoM). Closed form. Factory: `Rayleigh`.

### Task F4: Cauchy
`.../Cauchy.cs` → `cauchy.hpp`. Params `[x0 location, gamma scale]`. Mean/variance undefined → return `kNaN`/`kInf` per C# (mirror exactly, including which moments are NaN). No estimation if C# has none. Closed form with `atan`. Factory: `Cauchy`.

### Task F5: Triangular
`.../Triangular.cs` → `triangular.hpp`. Params `[min, mode, max]` (confirm order). `IEstimation` (MoM, MLE). Piecewise closed form; watch the `x==mode` boundary. Factory: `Triangular`.

### Task F6: Deterministic
`.../Deterministic.cs` → `deterministic.hpp`. Param `[value]`. Degenerate point mass: `pdf` = ∞ at value else 0 (mirror C# exactly), `cdf` step, `inverse_cdf` = value. `mean=median=mode=value`, `sd=0`. `IEstimation` (MoM). Factory: `Deterministic`.

### Task F7: UniformDiscrete
`.../UniformDiscrete.cs` → `uniform_discrete.hpp`. Params `[min, max]` (integer support, stored as double). `IEstimation` (MoM). Factory: `UniformDiscrete`. Enum name is `UniformDiscrete`.

### Task F8: Geometric
`.../Geometric.cs` → `geometric.hpp`. Param `[probability]`. Discrete. `IEstimation` (MoM). Factory: `Geometric`.

### Task F9: Bernoulli
`.../Bernoulli.cs` → `bernoulli.hpp`. Param `[probability]`. Discrete two-point. Factory: `Bernoulli`.

### Task F10: GeneralizedPareto
`.../GeneralizedPareto.cs` → `generalized_pareto.hpp`. Params `[xi location, alpha scale, kappa shape]`. `IEstimation` + `ILinearMomentEstimation`. Closed form (κ→0 limit branch — mirror the C# guard). Factory: `GeneralizedPareto`.

### Task F11: GeneralizedLogistic
`.../GeneralizedLogistic.cs` → `generalized_logistic.hpp`. Params `[xi location, alpha scale, kappa shape]`. `IEstimation` + `ILinearMomentEstimation`. Closed form with κ→0 branch. Factory: `GeneralizedLogistic`.

---

## Milestone C — Wave 2: Normal-derived

Gate: Task A2 (Erf). Standard path (Task F0 recipe). Each internally uses Normal's math / `erf`.

### Task C1: LogNormal
`.../LogNormal.cs` → `log_normal.hpp`. Params `[mu, sigma]` (of the log). `IEstimation` + `ILinearMomentEstimation`. `inverse_cdf` = `exp(Normal.inverse_cdf(p))`-style. Factory: `LogNormal`.

### Task C2: LnNormal
`.../LnNormal.cs` → `ln_normal.hpp`. Params per C# (real-space mean/sd variant of log-normal — read carefully; LnNormal and LogNormal differ in parameterization). `IEstimation` + `ILinearMomentEstimation`. Factory: `LnNormal`.

### Task C3: TruncatedNormal
`.../TruncatedNormal.cs` → `truncated_normal.hpp`. Params `[mu, sigma, min, max]`. Uses Normal CDF/PDF + `erf`/`inverse_erf`. `IEstimation` (MoM). Standard (flat 4-param vector), so still 3-edit path. Factory: `TruncatedNormal`.

### Task C4: GeneralizedNormal
`.../GeneralizedNormal.cs` → `generalized_normal.hpp`. Params `[xi location, alpha scale, kappa shape]`. `IEstimation` + `ILinearMomentEstimation`. **If** its CDF uses numerical integration in C#, do Task A7 first; otherwise closed form via `erf`. Factory: `GeneralizedNormal`.

---

## Milestone D — Wave 3: Gamma family

Gate: Task A3 (Gamma extensions). Standard path.

### Task D1: GammaDistribution
`.../GammaDistribution.cs` → `gamma_distribution.hpp`. Params `[theta scale, kappa shape]` (confirm order). Uses `log_gamma`, `lower_incomplete`, `inverse_lower_incomplete`, `trigamma` (MLE). `IEstimation` + `ILinearMomentEstimation`. Factory type `GammaDistribution`, name `"GammaDistribution"`. **File name note:** avoid a bare `gamma.hpp` in `distributions/` colliding conceptually with the special-functions gamma — use `gamma_distribution.hpp`.

### Task D2: ChiSquared
`.../ChiSquared.cs` → `chi_squared.hpp`. Param `[degrees_of_freedom]`. Special case of Gamma; uses regularized incomplete gamma. `IEstimation` (MoM, MLE) + `ILinearMomentEstimation` if present. Factory: `ChiSquared`.

### Task D3: InverseGamma
`.../InverseGamma.cs` → `inverse_gamma.hpp`. Params `[beta scale, alpha shape]`. Uses incomplete gamma. Estimation only if C# has it. Factory: `InverseGamma`.

### Task D4: InverseChiSquared
`.../InverseChiSquared.cs` → `inverse_chi_squared.hpp`. Params `[df, scale]` per C#. Uses incomplete gamma. Factory: `InverseChiSquared`.

### Task D5: Weibull
`.../Weibull.cs` → `weibull.hpp`. Params `[lambda scale, kappa shape]` (confirm order). CDF closed form; moments use `Gamma::function`/`log_gamma`. `IEstimation` + `ILinearMomentEstimation`. Factory: `Weibull`.

### Task D6: Poisson
`.../Poisson.cs` → `poisson.hpp`. Param `[lambda]`. Discrete; CDF via `upper_incomplete`/`lower_incomplete`; uses `factorial`. `IEstimation` (MoM, MLE). Gate also on A5 (Factorial). Factory: `Poisson`.

---

## Milestone E — Wave 4: Gamma-skew family (harder)

Gate: Task A3. Standard path but numerically delicate (skew reparameterization, root-finding in inverse-CDF). Give each its own task and extra fixture coverage across the skew sign.

### Task E1: PearsonTypeIII
`.../PearsonTypeIII.cs` → `pearson_type_iii.hpp`. Params `[mu mean, sigma sd, gamma skew]`. Reparameterizes to Gamma; handle γ=0 (→ Normal) and γ<0 (reflected) branches exactly as C#. Uses incomplete gamma + inverse. `IEstimation` + `ILinearMomentEstimation`. Factory: `PearsonTypeIII`. Fixture: include positive-skew, near-zero-skew, negative-skew cases.

### Task E2: LogPearsonTypeIII
`.../LogPearsonTypeIII.cs` → `log_pearson_type_iii.hpp`. Params `[mu, sigma, gamma]` (of log10 or ln — confirm which; flood-frequency standard is log10). Wraps PearsonTypeIII in log space. `IEstimation` + `ILinearMomentEstimation`. Factory: `LogPearsonTypeIII`. Gate on E1.

### Task E3: KappaFour
`.../KappaFour.cs` → `kappa_four.hpp`. Params `[xi location, alpha scale, kappa, h]` (4 params). InverseCDF has closed form with limit branches (h→0, κ→0); PDF/CDF closed-ish. `IEstimation` + `ILinearMomentEstimation` (L-moment solve may need Brent/NelderMead). Factory: `KappaFour`. Standard (flat 4-param).

---

## Milestone F — Wave 5: Beta family

Gate: Task A4 (Beta), Task A5 (Factorial for Binomial). Standard path.

### Task F20: BetaDistribution
`.../BetaDistribution.cs` → `beta_distribution.hpp`. Params `[alpha, beta]`. Uses `beta::function`, `beta::incomplete`, `beta::incomplete_inverse`. `IEstimation` (MoM, MLE) + `ILinearMomentEstimation` if present. Factory type `Beta`, name `"Beta"`, file `beta_distribution.hpp` (avoid colliding with special-functions `beta.hpp`).

### Task F21: GeneralizedBeta
`.../GeneralizedBeta.cs` → `generalized_beta.hpp`. Params `[alpha, beta, min, max]` (4-param). Rescaled Beta. Factory: `GeneralizedBeta`.

### Task F22: Binomial
`.../Binomial.cs` → `binomial.hpp`. Params `[probability, n_trials]`. Discrete; CDF via regularized incomplete beta; uses `binomial_coefficient`. `IEstimation` (MoM, MLE). Factory: `Binomial`.

### Task F23: Pert
`.../Pert.cs` → `pert.hpp`. Params `[min, mode, max]`. Rescaled Beta with derived α,β. `IEstimation` (MoM, MLE). Factory: `Pert`.

### Task F24: StudentT
`.../StudentT.cs` → `student_t.hpp`. Params `[mu location, sigma scale, nu df]` (confirm; base StudentT may be 1-param df). CDF via regularized incomplete beta; inverse via `incomplete_inverse`. Estimation only if C# has it. Factory: `StudentT`. HARD.

### Task F25: NoncentralT
`.../NoncentralT.cs` → `noncentral_t.hpp`. Params `[nu df, lambda noncentrality]`. Series/numerical PDF & CDF using incomplete beta. Factory: `NoncentralT`. HARD — allow extra fixture cases; may need A7 integrator (check C#).

---

## Milestone G — Wave 6: exotic

### Task G1: VonMises
Gate: Task A6 (Bessel I0/I1) and likely Task A7 (integrator for CDF). `.../VonMises.cs` → `von_mises.hpp`. Params `[mu mean-direction, kappa concentration]`. PDF uses `bessel::i0`; CDF via numerical integration or series (mirror C#). `IEstimation` (MoM, MLE). Factory: `VonMises`. HARD.

---

## Milestone H — Wave 7: composite / data-driven (BESPOKE)

These break the flat-`vector<double>` contract: they are constructed from **sub-distributions** or **data arrays**, so the factory-by-name + `set_parameters(vector<double>)` path and the generic fixture `construct: {params|fit}` schema do not fit. Each needs a GEV-style bespoke path **and** a fixture-schema extension. This is the risky tail.

### Task H0: Decide scope & design the schema extension (CHECKPOINT)

**This is a design/decision task — do it before H1+ and get approval.**
- [ ] Read `EmpiricalDistribution.cs`, `KernelDensity.cs`, `TruncatedDistribution.cs`, `Mixture.cs`, `CompetingRisks.cs`, `PertPercentile.cs`, `PertPercentileZ.cs`.
- [ ] Design a fixture `construct` extension, e.g. `"construct": { "components": [ {"target":"Normal","params":[0,1]}, ... ], "weights": [...] }` for Mixture/CompetingRisks; `{ "base": {"target":"Normal","params":[0,1]}, "bounds":[a,b] }` for TruncatedDistribution; `{ "data": [...], "bandwidth": h }` for KernelDensity; `{ "x": [...], "p": [...] }` for Empirical; percentile-triples for PertPercentile*.
- [ ] Decide the C++/R/Python construction glue: a small bespoke `build_<name>()` in each runner + emitter (mirror the GEV bespoke pattern) OR a generic "component builder" the runners share. Recommend the latter (one shared component-builder) to keep DRY.
- [ ] **Present this design to the user for approval; it may become its own sub-plan.** PertPercentile/PertPercentileZ are flat-param solvers (percentile triples → Pert) and may actually fit the standard path — confirm and, if so, move them to Milestone F.

### Tasks H1–H7 (after H0 approval)
Each: port the C# distribution to `distributions/<name>.hpp`; add the bespoke construction path in `test_fixtures.cpp`, `bestfitr/src/dist.cpp` (or a new `composite.cpp`), `bestfitpy/src/bindings/dist.cpp`, and `tools/oracle_emitter/Program.cs`; add the extended-schema fixture; register in factory where a default-constructible form exists. Verify with the full DoD command block.
- **H1 EmpiricalDistribution** (`Empirical` enum) — data-driven interpolation.
- **H2 KernelDensity** — data + bandwidth + kernel; interpolated CDF/inverse.
- **H3 TruncatedDistribution** — wraps any base distribution + bounds.
- **H4 Mixture** — weights + component distributions.
- **H5 CompetingRisks** — component distributions (min-composite).
- **H6 PertPercentile** / **H7 PertPercentileZ** — solve to Pert (may be standard; see H0).

---

## Milestone I — Remaining Phase-1 foundation breadth

The numbered Phase-1 scope also lists Sobol + RNG breadth and goodness-of-fit stats. Port the pieces that have a near-term consumer; explicitly defer the rest (YAGNI) with a note.

### Task I1: Sobol sequence
**Upstream:** `.../Sampling/SobolSequence.cs` + resource `new-joe-kuo-6.21201`.
- Create `core/include/bestfit/numerics/sampling/sobol.hpp`; place the direction-number file at `core/data/new-joe-kuo-6.21201`; teach `tools/sync_core.py` to also copy `core/data/` (verify it already does — PLAN says it copies `core/{include,src,data}`) into `bestfitr/inst/extdata/` and `bestfitpy` package data; load via a path passed from `system.file()` (R) / `importlib.resources` (Python).
- Add a C++ reference-stream test (first-N points) as the oracle; add a `sampling` fixture or a dedicated `core/tests/test_sobol.cpp`.
- Pin R/Python identity with a seeded test (mirror the existing `test_mersenne_twister.cpp` pattern).

### Task I2: Goodness-of-fit statistics
**Upstream:** `.../Data/Statistics/GoodnessOfFit.cs`.
- Add Kolmogorov-Smirnov, Anderson-Darling, and chi-squared test statistics to `core/include/bestfit/numerics/data/` (new `goodness_of_fit.hpp`).
- Fixture-validate against `Test_GoodnessOfFit.cs` (locate under `Test_Numerics/Data/Statistics/`).
- Only surface to R/Python if a consumer needs it now; otherwise C++ + oracle only.

### Task I3: Foundation deferral note (documentation only)
- [ ] In `.claude/PLAN.md`, record which Phase-1 foundation items are **deferred (YAGNI, no current consumer)**: general integration beyond A7, numerical differentiation, and the fuller linear algebra (Cholesky/LU/QR/SVD/Eigen). They land when Phase 3/4/5 consumers arrive. This keeps the CRAN dependency surface and build time minimal.

---

## Milestone J — Wrap-up

### Task J1: Provenance headers & SHA backfill
- [ ] Replace every `@ <pending-sha>` / `@ <sha>` provenance header (existing and new) with the pinned `upstream/Numerics` commit SHA (`git -C upstream/Numerics rev-parse HEAD`). One `sed`-guided pass; verify no header is left with a placeholder.

### Task J2: Docs
- [ ] R: ensure any new *public* R API has roxygen2 (`Rscript -e 'roxygen2::roxygenise("bestfitr")'`); most distributions are reached generically so this is minimal. Python: numpydoc docstrings on any new public surface.
- [ ] Update `README`/vignette distribution coverage list if one exists.

### Task J3: Status refresh
- [ ] Update `.claude/PLAN.md` and `.claude/CLAUDE.md` status blocks: Phase 1 complete — foundation special functions + all 42 univariate distributions ported, fixture-validated in C++/R/Python, reproduced against C#, CI green.

### Task J4: Full CI + PR
- [ ] `python3 tools/sync_core.py --check && python3 tools/sync_fixtures.py --check` clean.
- [ ] Full local matrix: `ctest --test-dir core/build`, `R CMD check --as-cran bestfitr` (WARNING/NOTE-clean), `pytest bestfitpy/tests`.
- [ ] `python3 tools/verify_oracles.py` green across all fixtures.
- [ ] Branch (e.g. `phase1-foundation-and-distributions`), push when the user asks, open PR; follow with `gh run watch <id> --exit-status` until the full matrix (sync-check, core×3, r-cmd-check×3, python×6) is green.

---

## Self-review notes

- **Spec coverage:** Phase-1 scope = "Numerics math + RNG foundation" (Milestones A, I) + "all 42 univariate distributions" (B–H). All 38 remaining distributions have a task; all special functions the distribution survey flagged (incomplete gamma/beta + inverses, log-gamma, trigamma, erf-inverse, factorial/binomial, Bessel I0/I1) have a foundation task. RNG breadth (Sobol) and goodness-of-fit are I1/I2; unused foundation is explicitly deferred (I3).
- **Type consistency:** factory registration uses the exact enum spellings from `univariate_distribution_type.hpp` (`GammaDistribution`, `Beta`, `Empirical`, `UniformDiscrete`, etc.); file names avoid the `gamma.hpp`/`beta.hpp` special-function collisions by using `gamma_distribution.hpp`/`beta_distribution.hpp`.
- **Bespoke vs standard:** the flat-param 3-edit path (verified against runner/glue/emitter) covers all of B–G; H is correctly isolated as needing schema + binding extensions and a design checkpoint (H0).
- **Open decisions for the user:** (1) confirm Milestone H composite scope belongs in this plan vs a follow-on; (2) confirm whether PertPercentile* are flat-param (→ move to F); (3) confirm foundation deferral list (I3).
```
