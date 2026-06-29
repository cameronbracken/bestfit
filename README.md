# bestfit

R (`bestfitr`) and Python (`bestfitpy`) packages for Bayesian flood-frequency and
extreme-value analysis, built on a **shared C++17 core** that is a faithful port of the
USACE-RMC [Numerics](https://github.com/USACE-RMC/Numerics) and
[RMC.BestFit](https://github.com/USACE-RMC/RMC-BestFit) C# libraries.

The C++ core is wrapped by [cpp11](https://cpp11.r-lib.org/) for R and
[pybind11](https://pybind11.readthedocs.io/) for Python. Writing the math once and binding
it twice means both packages run identical code — and, because the Mersenne Twister RNG is
ported bit-exact, seeded MCMC chains are identical across R and Python.

## Status

Early development. **Phase 0** (prove the full toolchain end-to-end on one distribution)
is in progress:

- [x] Monorepo scaffold + C++ build (CMake/ctest)
- [x] Mersenne Twister ported and pinned to the canonical mt19937ar reference stream
- [x] Gamma special functions (Cephes `function`, Lanczos, digamma)
- [x] GEV distribution core: moments, PDF/CDF/InverseCDF, log-likelihood — validated
      against the upstream C# oracle values
- [ ] GEV estimation (L-moments, MLE) + standard error — needs Brent, Statistics, Nelder-Mead
- [ ] Language-neutral oracle fixtures + generic test runners (C++/R/Python)
- [ ] R (`bestfitr`) and Python (`bestfitpy`) bindings for the GEV slice
- [ ] CI: C++ tests, `R CMD check --as-cran`, cibuildwheel

See the implementation plan for the full architecture and phasing.

## Layout

| Path | Purpose |
|------|---------|
| `core/` | canonical C++17 core (headers, sources, tests) — single source of truth |
| `fixtures/` | language-neutral oracle fixtures (JSON) extracted from the C# tests |
| `bestfitr/` | R package (cpp11) — vendored copy of the core in `src/` |
| `bestfitpy/` | Python package (scikit-build-core + pybind11) — vendored copy of the core |
| `tools/` | core/fixture sync + upstream-diff + oracle-extraction scripts |

## Building the C++ core tests

```bash
cmake -S core -B core/build && cmake --build core/build && ctest --test-dir core/build
```

## License

0BSD, matching the upstream USACE-RMC libraries.
