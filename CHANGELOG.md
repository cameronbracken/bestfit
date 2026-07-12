# Changelog

All notable changes to corehydro (the shared C++ core, the `corehydror` R package, and
the `corehydropy` Python package) are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). The three components are versioned together.

## [Unreleased]

## [0.1.0] - 2026-07-11

First tagged release. Everything below is new.

### Added

- **Shared C++17 core** with full parity to the USACE-RMC Numerics and RMC-BestFit C#
  libraries (the probability, estimation, and analysis layers): all 42 univariate
  distributions, multivariate distributions and seven bivariate copulas, eight MCMC
  samplers (RWMH, ARWMH, DEMCz, DEMCzs, HMC, NUTS, Gibbs, SNIS) with Gelman-Rubin and
  ESS diagnostics, bootstrap uncertainty, the MLE/MAP/Bayesian/GMM estimators, all
  RMC-BestFit model families (flood frequency, Bulletin 17C, mixtures, competing risks,
  point process, time series, spatial GEV, rating curves, bivariate/copula), the full
  Analyses layer, and the Diagnostics layer (leverage, PSIS-LOO influence, prior
  influence, predictive checks). See the porting status page for the exact scope.
- **R package `corehydror`** and **Python package `corehydropy`**, thin bindings over
  the same compiled core. A bit-exact Mersenne Twister port means seeded results are
  identical across R, Python, and the upstream C# libraries.
- **Public distribution API**: construct any of 38 families by name; density, CDF,
  quantile, moments, L-moments, log-likelihood, seeded random generation, and fitting
  by MLE, L-moments, or product moments.
- **Analysis functions** (19 in each package): `univariate_analysis`,
  `fit_distributions`, `bulletin17c_analysis`, mixture/competing-risk/point-process/
  composite/spatial-GEV analyses, AR/MA/ARIMA/ARIMAX time-series analyses, bivariate
  and coincident-frequency analyses, rating curves, bootstrap uncertainty, predictive
  checks, and estimation diagnostics.
- **`mcmc_sample()`**: direct MCMC over any distribution family (7 samplers) with
  chains, acceptance rates, MAP, posterior summaries, R-hat, and ESS.
- **Statistics utilities**: Multiple Grubbs-Beck low-outlier test, Box-Cox and
  Yeo-Johnson transforms, plotting positions, Latin hypercube sampling.
- **Validation**: a language-neutral oracle fixture suite consumed by C++, R, and
  Python runners, plus a dev-only dotnet gate that replays every fixture against the
  real C# libraries (4,100+ values reproduced, 0 failures).
- **Documentation site** with worked examples in both languages ported from the
  official Numerics-Python-Examples repository (11 example pairs, each ending in an
  executable reproduction check against the upstream C# outputs), a Python API
  reference (quartodoc), an R API reference (pkgdown), and a porting status page.
- **Developer tooling**: pixi environment with tasks mirroring the Makefile targets,
  GitHub Actions CI (3 OS matrix) and Pages deployment.

### Changed

- Renamed from `bestfit` (packages `bestfitr`/`bestfitpy`) to **corehydro**
  (`corehydror`/`corehydropy`), reflecting the goal of carrying code from both
  USACE-RMC and HEC libraries in one package family.

[Unreleased]: https://github.com/cameronbracken/corehydro/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cameronbracken/corehydro/releases/tag/v0.1.0
