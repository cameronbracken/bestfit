# Oracle fixtures

Language-neutral validation data — the **single source of truth** for the expected
values that the C++ core, the R package (`bestfitr`), and the Python package
(`bestfitpy`) all check against. Each language has a thin generic *runner* that loads
these JSON files and applies every assertion; no oracle values are duplicated in any
test file. This is what makes "validate identically" structural rather than aspirational.

Oracle values trace to the upstream C# unit tests (and through them to published
references such as Rao & Hamed, *Flood Frequency Analysis*). They are curated from those
tests and then **verified reproducible against the real Numerics library** by
`tools/verify_oracles.py` (which runs the C# `tools/oracle_emitter` project). That gate is
dev-only — it needs `dotnet` and the `upstream/Numerics` submodule — and confirms every
expected value still reproduces to its stated tolerance. A scripted extractor to harvest
the C# test literals *en masse* is planned for the bulk distribution port.

`tools/sync_fixtures.py` copies this directory into each package
(`bestfitr/inst/fixtures`, `bestfitpy/src/bestfitpy/fixtures`) so each ships its own
verbatim copy; a CI `--check` run fails on drift.

## Schema

### `univariate_distribution`

```jsonc
{
  "target":  "GeneralizedExtremeValue",     // which model/component
  "kind":    "univariate_distribution",     // runner family
  "source":  "Numerics/.../Test_*.cs",      // provenance
  "datasets": { "name": [ ...numbers... ] }, // samples referenced by fit cases
  "cases": [
    {
      "name": "default_moments",
      "construct": { "params": [100, 10, 0] },              // OR
      "construct": { "fit": { "dataset": "white_river", "method": "mle" } },
      "assertions": [
        { "method": "mean", "args": [], "expected": 105.77, "mode": "abs", "tol": 1e-6 }
      ]
    }
  ]
}
```

#### Composite `univariate_distribution` targets

Five `univariate_distribution` targets are composites over other distributions and use a
structured `construct` instead of flat `"params"`/`"fit"`. Each of the four runners (C++
`build_composite` in `core/tests/test_fixtures.cpp`, R `build_composite_data`/
`dispatch_composite` in `bestfitr/tests/testthat/test-fixtures.R`, Python `_build_composite`/
`_dispatch_composite` in `bestfitpy/tests/test_fixtures.py`, and the emitter's `BuildComposite`
in `tools/oracle_emitter/Program.cs`) implements the same schema:

- `TruncatedDistribution`: `{"base": {"target": ..., "params": [...]}, "bounds": [lo, hi]}`.
- `Empirical`: `{"x": [...], "p": [...], "p_transform": "None" | "NormalZ"}` (`p_transform`
  optional, default `"NormalZ"`).
- `KernelDensity`: `{"data": "<dataset name>", "kernel": "Gaussian" | "Epanechnikov" |
  "Triangular" | "Uniform", "bandwidth": <double>, "bounded_by_data": <bool>}` (`kernel`,
  `bandwidth`, `bounded_by_data` all optional).
- `Mixture`: `{"components": [{"target": ..., "params": [...]}, ...], "weights": [...]}`.
  `components` entries may also use `{"fit": {"dataset": ..., "method": ...}}` instead of
  `"params"` (recursive `build_component`).
- `CompetingRisks`: `{"components": [{"target": ..., "params": [...]}, ...],
  "minimum_of_random_variables": <bool>, "dependency": "Independent" | "PerfectlyPositive" |
  "PerfectlyNegative" | "CorrelationMatrix", "correlation": [[...], ...]}`.
  - `minimum_of_random_variables` (default `true`): `true` = min-of-components (series system);
    `false` = max-of-components (parallel system).
  - `dependency` (default `"Independent"`): the four `Probability.DependencyType` modes. Only
    `Independent`/`PerfectlyPositive` are closed-form; `PerfectlyNegative`/`CorrelationMatrix`
    route through a Product-of-Conditional-Marginals (HPCM) engine that lazily builds a
    `MultivariateNormal` purely to hold/validate a covariance matrix (see
    `core/include/bestfit/numerics/data/probability.hpp`'s header comment and
    `docs/upstream-csharp-issues.md` for why this path is fully deterministic, unlike
    `MultivariateNormal.CDF()`'s own seeded Genz-Bretz integrator for dimension >= 3).
  - `correlation` (optional, default `[]`): a square matrix, one row per component. Only
    consulted when `dependency == "CorrelationMatrix"` (ignored, and may be omitted, for the
    other three modes -- `PerfectlyNegative` synthesizes its own correlation internally).

### `multivariate_distribution`

```jsonc
{
  "target":  "Dirichlet",                   // "Dirichlet" | "Multinomial" | "BivariateEmpirical"
                                             // | "MultivariateNormal" | "MultivariateStudentT"
  "kind":    "multivariate_distribution",
  "source":  "Numerics/.../Test_Dirichlet.cs",
  "cases": [
    {
      "name": "pdf_uniform",
      "construct": { "alpha": [1.0, 1.0, 1.0] },
      "assertions": [
        { "method": "pdf", "args": [[0.333333333333333, 0.333333333333333, 0.333333333333333]],
          "expected": 2.0, "mode": "abs", "tol": 1e-6 }
      ]
    }
  ]
}
```

Per-case `construct` is target-specific:
- `Dirichlet`: `{"alpha": [..]}` (a symmetric Dir(K, alpha) case is just alpha repeated K times).
- `Multinomial`: `{"n": 10, "p": [..]}`.
- `BivariateEmpirical`: `{"x1": [..], "x2": [..], "p": [[..], [..]]}` (`p` is a 2D array, row `i`
  = `x1[i]`, column `j` = `x2[j]`) plus optional `"x1_transform"`/`"x2_transform"`/`"p_transform"`
  strings (`"None"`, `"Logarithmic"`, `"NormalZ"`; default `"None"`).
- `MultivariateNormal`: `{"mean": [..], "covariance": [[..]]}` plus an optional `"seed": <int>` --
  see Statefulness below.
- `MultivariateStudentT`: `{"df": <double>, "location": [..], "scale": [[..]]}` (`scale` is the
  scale matrix Sigma, not the covariance matrix -- the covariance is `df/(df-2)*scale` for
  `df > 2`; 1- and 2-arg C# constructor cases pass the identity/zero defaults explicitly rather
  than needing a different `construct` shape, keeping the schema uniform). Always stateless (its
  CDF is a deterministic K=200 stratified chi-squared mixture -- see Statefulness below).

Assertions use the same `{method, args, expected, mode, tol}` shape as `univariate_distribution`
(modes `abs`/`rel`/`equal`/`bool`). `args` may contain a nested array for a single vector-valued
argument (e.g. `"pdf"` args `[[0.3, 0.4, 0.3]]` is one 3-vector, not three scalars). An assertion
may also carry an optional `"source"` string overriding the file-level `"source"`/`"reference"`
provenance for that one value -- used by MultivariateNormal's and MultivariateStudentT's curated
(not-lifted-verbatim-from-a-C#-test-literal) cases, whose expected values were generated by
running the ported logic's inputs through `tools/oracle_emitter --dump` and confirming them
against the real Numerics library rather than copied from a C# test file; see `"source": "curated
via oracle_emitter --dump"` throughout `fixtures/distributions/multivariate/multivariate_normal.json`
and `multivariate_student_t.json`. MultivariateStudentT's curated cases exist for two reasons: (a)
the upstream `Test_CDF_2D` literals are loose Monte-Carlo references (tol 0.01) -- curation adds a
tight 1e-8 companion assertion for the same inputs so silent drift becomes a loud failure, and (b)
some C# assertions (`Test_CDF_1D`, `Test_PDF_3D_AtMode`) compare two runtime-computed values (a
dynamically-constructed univariate `StudentT.CDF`, or a `LogGamma` expression) rather than a
hardcoded literal, so there is no literal to transcribe. Methods:

- `pdf`, `log_pdf`, `cdf` (vector arg: `args: [[x1, x2, ...]]`) -- dispatch to the distribution's
  own PDF/LogPDF/CDF.
- `mean [i]`, `median [i]`, `variance [i]`, `mode [i]`, `sd [i]` -- vector-returning members,
  asserted element-wise via the trailing index arg (0-based).
- `covariance [i, j]` -- pairwise covariance.
- `dimension`, `parameters_valid` (`mode: "bool"`) -- generic to every target.
- `log_multivariate_beta [alpha...]` (Dirichlet only; static, flat `args` = the alpha vector --
  independent of the case's own `construct.alpha`).
- `cdf_xy [x1, x2]` (BivariateEmpirical only; the scalar-pair CDF overload, equivalent to
  `cdf([x1, x2])` but exercises the two-argument entry point directly).
- `mahalanobis [x...]` (MultivariateNormal, MultivariateStudentT) -- Mahalanobis/squared
  Mahalanobis distance of the point `x` from the distribution's mean/location.
- `inverse_cdf [[p...], i]` (MultivariateNormal, MultivariateStudentT) -- inverse CDF at the
  probability vector `p`, asserted element-wise via the trailing index arg (0-based); stateless
  (fresh instance per call). For MultivariateStudentT, `p` has length `Dimension + 1` -- the extra
  trailing probability drives the chi-squared mixing variable.
- `degrees_of_freedom` (MultivariateStudentT only).
- `interval [[lower...], [upper...]]` (MultivariateNormal only) -- probability mass in the
  hyper-rectangle `[lower, upper]`; draws from the seeded MVNUNI stream for dim >= 3 (see
  Statefulness below).
- `mvndst [n, [lower...], [upper...], [infin...], [correl...], maxpts, abseps, releps]`
  (MultivariateNormal only) -- direct entry point to the ported Genz MVNDST routine, exercising the
  flattened-covariance / `INFIN`-code Fortran-style interface rather than the mean/covariance
  wrapper; also draws from the seeded MVNUNI stream (see Statefulness below).
- `random_value [sample_size, seed, row, col]` (all four multivariate targets: Dirichlet,
  Multinomial, MultivariateNormal, MultivariateStudentT) -- element `[row, col]` (0-based) of the
  matrix returned by `generate_random_values(sample_size, seed)`; locks the first-N draws of the
  ported Mersenne Twister stream against the real C# `GenerateRandomValues`.
- `lhs_value [sample_size, seed, row, col]` (MultivariateNormal, MultivariateStudentT only -- the
  only two multivariate targets with a `LatinHypercubeRandomValues` method in the C# source;
  Dirichlet/Multinomial have no such method and carry no `lhs_value` cases) -- same shape,
  dispatching to `latin_hypercube_random_values(sample_size, seed)`.

**Statefulness:** Dirichlet/Multinomial/BivariateEmpirical are stateless per case (every assertion
is independent, dispatched against a fresh instance). MultivariateNormal is stateless UNLESS its
case's `construct` carries a `"seed"`: `cdf` (dim >= 3), `interval`, and `mvndst` all draw from the
seeded MVNUNI stream, so a run of consecutive same-method assertions in a seeded case must be
evaluated against ONE persistent instance, in listed order, not dispatched one call at a time
(which would silently reset the seed between assertions) -- each fixture runner groups consecutive
same-method assertions in a seeded case and batches them (see `_run_mvn_case` in
`bestfitpy/tests/test_fixtures.py`, `run_mvn_case` in `bestfitr/tests/testthat/test-fixtures.R`).
Assertions for other methods (or in a case with no `"seed"`) fall through to the stateless
per-assertion dispatch, unchanged. MultivariateStudentT is ALWAYS stateless -- its CDF (dim >= 2)
is a deterministic K=200 stratified chi-squared(v) mixture over MultivariateNormal's CDF, not a
seeded quasi-Monte-Carlo integrator, so no seeded batching is needed even though its `cdf`
internally constructs and evaluates an MVN instance. `random_value`/`lhs_value` are ALWAYS
stateless for every target that has them, REGARDLESS of a `"seed"` in `construct` -- this is a
property of the C# source itself, not a port simplification: `GenerateRandomValues`/
`LatinHypercubeRandomValues` each seed their OWN fresh MersenneTwister/LatinHypercube draw from
their `seed` **argument** on every call (unlike the MVNUNI stream `cdf`/`interval`/`mvndst` share
across a persistent instance), so evaluating each assertion against a fresh instance reproduces
correctly with no batching -- see `seeded_sampling` in
`fixtures/distributions/multivariate/multivariate_normal.json` for an example case that mixes
`random_value`/`lhs_value` assertions alongside a case with no `"seed"` key at all.

### `bivariate_copula`

```jsonc
{
  "target":  "Clayton",                     // the CopulaType enum name: "AliMikhailHaq" |
                                             // "Clayton" | "Frank" | "Gumbel" | "Joe" |
                                             // "Normal" | "StudentT"
  "kind":    "bivariate_copula",
  "source":  "Numerics/.../Test_ClaytonCopula.cs",
  "datasets": { "data1": [ ...100 numbers... ], "data1_pp": [ ...plotting positions... ] },
  "cases": [
    {
      "name": "pdf_theta10",
      "construct": { "theta": 10 },                            // OR
      "construct": { "fit": { "x": "data1", "y": "data2", "method": "tau",
                              "marginals": ["Normal", "Normal"] } },
      "assertions": [
        { "method": "pdf", "args": [0.2, 0.8], "expected": 1.3113e-05, "mode": "abs", "tol": 1e-6 }
      ]
    }
  ]
}
```

Every copula shares `BivariateCopula`'s uniform parameter API (`theta`/`get_copula_parameters`/
`pdf`/`cdf`/...), so -- unlike `multivariate_distribution`, whose targets share no common surface
-- all four runners dispatch through one fully generic path keyed by the factory
(`copula_factory.hpp`, a bestfit addition with no upstream counterpart, documented in its header);
adding a new copula target is a new header + a factory case + a fixture file, with **zero** runner
changes (the one exception, the `"tau"` fit method, is explained below).

`construct` is one of:

- `{"theta": <double>}` -- direct construction. 2-parameter copulas (StudentT, a later task) add a
  second key, `{"theta": <double>, "df": <double>}`; the runners map this to
  `set_copula_parameters([theta, df])`, matching `GetCopulaParameters`'s declared order.
- `{"theta": <double>, "marginals": {"targets": [<x-type>, <y-type>], "params": [[<x-params...>],
  [<y-params...>]]}}` -- an ADD-ON to the direct-construction key above (not a substitute for it)
  that attaches FIXED marginal distributions via the C# `Copula(theta, marginX, marginY)` ctor
  overload, e.g. `{"theta": 5, "marginals": {"targets": ["Normal", "Normal"], "params": [[100.0,
  15.0], [80.0, 20.0]]}}` sets `MarginalDistributionX = new Normal(100.0, 15.0)`. Distinct from the
  `"fit"` construct's own `"marginals"` key below, which FITS marginal parameters from data rather
  than fixing them outright; used by the `random_value` sampling oracles that need marginals set to
  exercise the back-transform branch of `GenerateRandomValues` (see `seeded_sampling_with_marginals`
  in `fixtures/distributions/copulas/clayton_copula.json`).
- `{"fit": {"x": "<dataset>", "y": "<dataset>", "method": "tau"|"mpl"|"ifm"|"mle", "marginals":
  ["Normal", "Normal"]?}}` -- fits the copula (and, for `"ifm"`/`"mle"`, its marginals) from sample
  data. `"marginals"` is a 2-element array of univariate distribution type names (the same strings
  `univariate_distribution_factory.hpp` accepts); omit it for `"tau"`/`"mpl"`, which have no
  parametric marginals.
  - `"tau"`: the Kendall's-tau method-of-moments fit (e.g. `ClaytonCopula::set_theta_from_tau`).
    This is **not** part of the shared copula API in the C# source either -- `SetThetaFromTau` is a
    member of each concrete Archimedean class, not `IBivariateCopula`/`IArchimedeanCopula` -- so
    every runner resolves it with a small per-target dispatch (one `if (target == "Clayton") ...`
    branch each; see `set_theta_from_tau_dispatch` in `core/tests/test_fixtures.cpp` and its R/Python/
    emitter counterparts). Tau-capable copulas: Clayton, AliMikhailHaq (AMH), Gumbel. NOTE: an
    earlier draft of this doc also listed Joe here, but `JoeCopula.cs` has no `SetThetaFromTau`
    method (confirmed by grep across the whole "Bivariate Copulas" directory and by
    `Test_JoeCopula.cs` having no `Test_MOM_Fit`) -- Joe has no "tau" fixture case and no dispatch
    branch (Task 8, see `.superpowers/sdd/task-8-report.md`).
  - `"mpl"` (maximum pseudo likelihood): `"x"`/`"y"` must already be the **plotting positions** of
    the data (rank/(n+1) via `Statistics.RanksInPlace`), not the raw sample -- mirroring the C# test
    flow (`Test_MPL_Fit`), which computes plotting positions itself before calling
    `BivariateCopulaEstimation.Estimate`. To keep every runner thin (no rank/tie logic duplicated
    four times), fixtures precompute the plotting-position arrays once and store them as their own
    named datasets (e.g. `data1_pp`/`data2_pp` in `clayton_copula.json`), and `"mpl"` cases point `x`/
    `y` directly at those.
  - `"ifm"` (inference from margins): the runner pre-fits `marginals` by maximum likelihood against
    the RAW `x`/`y` sample data BEFORE calling `bivariate_copula_estimation.hpp`'s `estimate(...,
    InferenceFromMargins)` -- mirroring `Test_IFM_Fit`'s `((IEstimation)...).Estimate(...)` calls
    that precede `BivariateCopulaEstimation.Estimate`.
  - `"mle"` (full likelihood): the runner constructs `marginals` UNFITTED (default-parameterized) and
    lets the joint NelderMead optimizer inside `estimate(..., FullLikelihood)` fit the copula and
    both marginals simultaneously -- mirroring `Test_MLE_Fit`, which just assigns
    `copula.MarginalDistributionX = new Normal();` before calling
    `BivariateCopulaEstimation.Estimate`.

Assertions use the same `{method, args, expected, mode, tol}` shape as the other kinds. Methods:

- `pdf`, `log_pdf`, `cdf` (`args: [u, v]`), `inverse_cdf` (`args: [u, v, index]`, 0-based) -- the
  reduced-variate copula functions.
- `upper_tail_dependence`, `lower_tail_dependence`, `theta`, `df` (2-parameter copulas only;
  `get_copula_parameters()[1]`) -- no args.
- `or_exceedance`, `and_exceedance` (`args: [u, v]`) -- the OR/AND joint exceedance probabilities.
- `marginal_param` (`args: ["x"|"y", index]`) -- a fitted marginal's parameter, 0-based; only valid
  after a `"fit"` construct with `"marginals"` set.
- `parameters_valid` (`mode: "bool"`).
- `random_value [sample_size, seed, row, col]` -- element `[row, col]` (0-based; column 0 is `u`,
  column 1 is `v`, or their marginal-back-transformed values when `construct.marginals` is set) of
  the matrix returned by `generate_random_values(sample_size, seed)`; stateless (`GenerateRandomValues`
  seeds its own fresh `LatinHypercube` draw from the `seed` argument on every call, mirroring
  `multivariate_distribution`'s `random_value`/`lhs_value` -- see that kind's Statefulness note
  above). Every copula has this method (there is no copula-specific `lhs_value` -- copulas only ever
  draw via Latin Hypercube internally, so there's no separate LHS entry point to lock).

**Fit tolerance:** every `"fit"` assertion uses `mode: "abs"`/`"rel"` with a tolerance, NEVER
`"equal"`. `"mpl"`/`"ifm"`/`"mle"` walk BrentSearch/NelderMead, which can diverge from the C#
optimizer by a libm-ULP branch on transcendental objectives (`std::sin` vs `Math.Sin` and similar) --
see the Task 3 ledger note in `.superpowers/sdd/progress.md`. `clayton_copula.json` uses the same
tolerances as the upstream C# test literals (`1e-4` for theta on `"tau"`/`"mpl"`/`"ifm"`, `1e-3` for
theta on `"mle"`, `1e-2` for MLE's marginal parameters).

**Seeded sampling coverage:** every copula target carries a `seeded_sampling` case exercising
`random_value` (reduced-variate `(u, v)` space, no marginals). Clayton additionally carries a
`seeded_sampling_with_marginals` case using the `"marginals"` direct-attach construct documented
above, the one target chosen to cover the back-transform branch (all seven copulas share the same
`generate_random_values` implementation in `bivariate_copula.hpp`, so one marginals-attached
companion case is sufficient to lock that branch; it does not need repeating per target).

### `special_function`

For internal C++ math utilities (not exposed to R/Python). The R and Python fixture runners
skip files with this kind; only the C++ runner and the dotnet oracle gate process them.

```jsonc
{
  "target":  "Erf.function",               // "<Module>.<method>" dispatch key
  "kind":    "special_function",
  "source":  "Numerics/.../Erf.cs",        // provenance
  "cases": [
    {
      "name": "erf_one",
      "args": [1.0],                        // positional arguments to the function
      "assertions": [
        { "method": "value", "expected": 0.8427007929497149, "mode": "abs", "tol": 1e-12 }
      ]
    }
  ]
}
```

The dispatch key maps to a free function: `"Erf.function"` → `bestfit::numerics::math::special::erf::function(x)`.
Each case has a flat `args` array (not nested in `construct`); the single assertion method is always `"value"`.

Most files dispatch every case through that one file-level `target`. A file may instead group
several related dispatch keys and let individual cases override it with their own `"target"`; a
case without one falls back to the file-level `target`, so single-target files are unaffected. See
`fixtures/special_functions/cholesky.json`, whose cases select among `Cholesky.determinant`,
`Cholesky.log_determinant`, `Cholesky.inverse_element`, `Cholesky.solve_element`, and
`Cholesky.is_positive_definite`.

The `Cholesky.*` targets take a matrix flattened to row-major doubles in `args`, with `n` inferred
from the argument count: `n = sqrt(len(args))` for `determinant`/`log_determinant`/
`is_positive_definite`; `inverse_element` appends two trailing `(i, j)` indices
(`n = sqrt(len(args) - 2)`); `solve_element` appends an `n`-length right-hand-side vector followed
by a trailing solution index `i` (`n` solves `n^2 + n + 1 = len(args)`).

The `Correlation.*` targets (`fixtures/statistics/correlation.json`) take two equal-length samples
flattened into `args` as `[x_1..x_n, y_1..y_n]`, split at the midpoint (`n = len(args) / 2`).

**Comparison modes:** `abs` (|actual−expected| ≤ tol), `rel` (|actual−expected|/|expected| ≤ tol),
`equal` (exact; `expected` may be the strings `"inf"`, `"-inf"`, `"nan"`), `bool`.

**Special values:** in `params` and `equal`-mode `expected`, the strings `"nan"`,
`"inf"`, `"-inf"` denote the corresponding floating-point values (JSON has no literals
for them).

Each runner maps fixture `method` names to its own API (e.g. `"sd"` →
`standard_deviation()`), so a new distribution usually needs only a new fixture file
plus a few dispatch entries per language.
