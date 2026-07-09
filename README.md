# bestfit

R (`bestfitr`) and Python (`bestfitpy`) packages for stochastic hydrology, including
Bayesian flood-frequency and extreme-value analysis. Both packages are built on a
**shared C++17 core** that is a faithful port of the USACE-RMC
[Numerics](https://github.com/USACE-RMC/Numerics) and
[RMC.BestFit](https://github.com/USACE-RMC/RMC-BestFit) C# libraries.

The C++ core is wrapped by [cpp11](https://cpp11.r-lib.org/) for R and
[pybind11](https://pybind11.readthedocs.io/) for Python. The core and both packages are
validated by tests derived from the original RMC libraries, ensuring the ported code
produces the same results as the originals — and, because the Mersenne Twister RNG is
ported bit-exact, identical output across R and Python given the same seed.

## Status

Early development. **Phase 0** (prove the full toolchain end-to-end on one distribution —
the Generalized Extreme Value) is **complete**, with CI green on Linux, macOS, and Windows:

- [x] Monorepo scaffold + C++ build (CMake/ctest)
- [x] Mersenne Twister ported and pinned to the canonical mt19937ar reference stream
- [x] Gamma special functions (Cephes `function`, Lanczos, digamma)
- [x] GEV distribution core: moments, PDF/CDF/InverseCDF, log-likelihood
- [x] GEV estimation: method of moments, L-moments, MLE (Nelder-Mead), quantile standard error
- [x] Language-neutral oracle fixtures + generic runners (C++ / R / Python validate from one JSON)
- [x] `bestfitr` (cpp11) and `bestfitpy` (pybind11 + scikit-build-core) bindings for the GEV slice
- [x] CI: C++ ctest, `R CMD check --as-cran`, Python build + pytest, drift guards — all 3 platforms

Verified: an MLE fit agrees to 15 significant figures between R and Python (same compiled core).

**Phase 1** is underway. The first increment landed the polymorphic distribution layer and the
first three pilot distributions:

- [x] `UnivariateDistributionBase` + the `UnivariateDistributionType` enum and factory, plus the
      `IEstimation` / `ILinearMomentEstimation` capability mixins (`dynamic_cast`-dispatched)
- [x] **Normal**, **Uniform**, and **Exponential** distributions (GEV refactored onto the base)
- [x] Polymorphic, factory-driven runners and bindings in all three languages, so a new
      distribution needs only a fixture file plus a couple of dispatch entries — no new glue
- [x] Hybrid oracle workflow: fixtures are curated from the C# unit tests **and** confirmed
      reproducible against the real Numerics library (`tools/verify_oracles.py`, needs `dotnet`)

Next in Phase 1: the rest of Numerics' math/RNG foundation, then the remaining ~38 univariate
distributions, all driven by the same fixture pipeline. See the implementation plan.

## Layout

| Path | Purpose |
|------|---------|
| `core/` | canonical C++17 core (headers, sources, tests) — single source of truth |
| `fixtures/` | language-neutral oracle fixtures (JSON) — single source of truth for validation |
| `bestfitr/` | R package (cpp11); vendors the core via subtree symlinks under `src/bestfit_core/` |
| `bestfitpy/` | Python package (scikit-build-core + pybind11); vendors the core via subtree symlinks |
| `tools/` | `materialize_core.py` (dereference symlinks for a self-contained build), `verify_oracles.py`, `extract_oracles.py` |

The vendored `bestfit_core/` and the shipped fixtures are subtree symlinks into the canonical
`core/` and `fixtures/`, so git holds one copy and there is no drift to guard against.

## Vendoring: one source of truth, self-contained builds

Two design constraints pull in opposite directions, and the layout reconciles them:

- **Development wants one source of truth.** Every line of C++ math lives in `core/`, and every
  expected value lives in `fixtures/`. You edit each in exactly one place.
- **CRAN and PyPI build each package from a self-contained tarball, on their own machines.** The
  shipped tarball/sdist must physically contain the core; it cannot reference a sibling `core/`.

The resolution is **symlink vendoring**: each package's vendored location is a committed *symlink*
into `core/{include,data}` or `fixtures/` (e.g. `bestfitr/src/bestfit_core/include ->
../../../core/include`). Git holds one copy, and the symlinks resolve for local development and for
a full clone. A build then dereferences them into a self-contained, symlink-free artifact:

- **R:** `R CMD build` dereferences symlinks into the tarball automatically.
- **Python:** `tools/materialize_core.py` dereferences them into real files (run in a throwaway
  checkout, e.g. `make build-py`); scikit-build-core then packs the real files into the sdist.

CI runs `materialize_core.py` in the R and Python jobs (this also covers Windows, where a checkout
may materialize a committed symlink as a text stub). The **upstream C# libraries**
(`upstream/Numerics`, `upstream/RMC-BestFit`) remain dev-only git submodules, not shipped in the
packages.

| Command | Effect |
|---------|--------|
| (none needed) | the core + fixtures are subtree symlinks; editing a core header is live through the link |
| `python3 tools/materialize_core.py` | dereference the symlinks into real files (a symlink-free tree for a build; rewrites the tree) |
| `make build-r` / `make build-py` | build a self-contained tarball / sdist (dereferences symlinks; Python via a throwaway worktree) |

CI runs the `--check` form as a gate (the `sync-check` job): commit a `core/` or `fixtures/` change
without re-syncing and the build goes red. **Rule: after any edit under `core/` or `fixtures/`, run
the matching sync script before committing** — otherwise the packages compile yesterday's code.

### Oracles and the "bulk oracle" extraction

An **oracle** is a known-correct expected value a test checks against. The USACE-RMC C# test
suites contain roughly 3,877 of them — each `Assert.AreEqual(expected, actual, tol)` is one oracle
(a mean, a quantile, a PDF value) that was itself validated against R, SciPy, or published tables.

Rather than hand-transcribe oracle values one distribution at a time, the **bulk oracle**
extraction harvests them *en masse* into language-neutral `fixtures/*.json`. All three test
runners — C++ (`core/tests/test_fixtures.cpp`), R (`bestfitr/tests/testthat/test-fixtures.R`),
Python (`bestfitpy/tests/test_fixtures.py`) — then validate against that one shared JSON set.
Write the assertions once; validate identically in three languages. This is why oracle values are
never hardcoded in the test files themselves.

The oracles are produced by a **hybrid workflow**: the expected values are curated from the C#
unit tests (the literals that were themselves validated against R / SciPy / published tables),
**and** every fixture is then confirmed reproducible against the real Numerics library by
`tools/verify_oracles.py`. That gate builds the small C# `tools/oracle_emitter` project, replays
each fixture's construct + assertions through the actual C# distributions, and fails on any value
that does not reproduce to its stated tolerance. It is **dev-only** — it needs `dotnet` and the
`upstream/Numerics` submodule, neither of which is required to build or ship the packages — so the
fixtures are pinned to the C# source they were ported from without adding any package dependency.

## Build & test

C++ core:

```bash
cmake -S core -B core/build && cmake --build core/build && ctest --test-dir core/build
```

R package:

```bash
Rscript -e 'cpp11::cpp_register("bestfitr")'   # only after editing src/*.cpp
R CMD INSTALL bestfitr
Rscript -e 'testthat::test_local("bestfitr")'
```

Python package:

```bash
pip install ./bestfitpy
pytest bestfitpy/tests
```

## License

The ported C++ library and both packages are licensed under the 0BSD license, matching the
original USACE-RMC libraries.
