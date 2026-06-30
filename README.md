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

Next: **Phase 1** — the dependency-ordered bulk port (the rest of Numerics' math/RNG, then the
42 univariate distributions), driven by the same fixture pipeline. See the implementation plan.

## Layout

| Path | Purpose |
|------|---------|
| `core/` | canonical C++17 core (headers, sources, tests) — single source of truth |
| `fixtures/` | language-neutral oracle fixtures (JSON) — single source of truth for validation |
| `bestfitr/` | R package (cpp11); vendored copy of the core in `src/bestfit_core/` |
| `bestfitpy/` | Python package (scikit-build-core + pybind11); vendored copy of the core |
| `tools/` | `sync_core.py`, `sync_fixtures.py` (vendoring + CI drift guards), `extract_oracles.py` |

The vendored `bestfit_core/` copies and the shipped fixtures are generated from the canonical
`core/` and `fixtures/`; CI fails if they drift (`tools/sync_*.py --check`).

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
