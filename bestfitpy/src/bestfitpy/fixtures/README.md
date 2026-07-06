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
running the case's inputs through `tools/oracle_emitter --dump` against the REAL C# Numerics
library (not computed by the ported C++/R/Python logic) and confirming the printed actuals
reproduce via `verify_oracles.py`, rather than copied from a C# test file; see `"source": "curated
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

### `mcmc_sampler`

```jsonc
{
  "target":  "RWMH",                        // the concrete sampler type: RWMH/ARWMH/Gibbs/SNIS/
                                             // DEMCz/DEMCzs today (P3.5/P3.6/P3.7); HMC/NUTS in
                                             // later tasks
  "kind":    "mcmc_sampler",
  "source":  "Numerics/Sampling/MCMC/RWMH.cs ; Test_Numerics/Sampling/MCMC/Test_RWMH.cs",
  "datasets": { "tippecanoe": [ ...48 numbers... ] },
  "cases": [
    {
      "name": "normal_rstan",
      "construct": {
        "model":    { "name": "uniform_constraints", "family": "Normal", "dataset": "tippecanoe" },
        "settings": { "initialize": "MAP", "proposal_sigma": "zeros" }
      },
      "assertions": [
        { "method": "posterior_mean", "args": [0], "expected": 12663.69, "mode": "rel", "tol": 0.05 },
        { "method": "chain_value", "args": [0, 0, 0], "expected": 13287.230350045900, "mode": "rel", "tol": 1e-5 }
      ]
    }
  ]
}
```

One sampler run per case: build the model via the model registry (below), construct the sampler,
apply non-default settings, call `sample()` **once**, then post-process an `MCMCResults` (`alpha =
0.1`, i.e. 90% credible intervals). Every assertion in the case reads that single cached run --
this kind is inherently **stateful** (unlike `multivariate_distribution`'s/`bivariate_copula`'s
per-assertion dispatch): each of the four runners makes exactly one glue call per case (C++ caches
`(sampler, results)` locally; R's `bf_mcmc_run_`/Python's `_core.mcmc_run` each return one big
struct/dict that every assertion for the case reads from, with no "seq machinery" -- there is
nothing to batch, since there is only ever one run).

**Model registry** (`core/include/bestfit/numerics/sampling/mcmc/model_registry.hpp`): a **bestfit
addition** with no upstream C# counterpart -- the C# equivalent is inline model-construction code
that lives only inside the individual `Test_*.cs` files, not in the library. `construct.model` is
`{"name": <registry key>, "family": <univariate distribution type name>, "dataset": <dataset
name>}`. The registry is **closed**; today it has two entries:

- `"uniform_constraints"`: uninformative Uniform priors bounded by `family`'s own
  `IMaximumLikelihoodEstimation.GetParameterConstraints(data)` lower/upper arrays, and a
  log-likelihood closure `params -> new <family>(params).LogLikelihood(data)` -- transcribed
  verbatim from `Test_RWMH_NormalDist_RStan` and family-generic from the start (any
  `IMaximumLikelihoodEstimation`-capable `family`, not just `"Normal"`; ARWMH's rstan cases use
  `"Logistic"`/`"Gumbel"`/`"Weibull"` too). The C# oracle emitter's own `BuildMcmcModel` was
  originally hard-coded to `"Normal"` only (P3.5); P3.6 generalized it via
  `UnivariateDistributionFactory.CreateDistribution(Enum.Parse<UnivariateDistributionType>(family))`
  + a `IMaximumLikelihoodEstimation` cast, matching the C++ registry's approach, so `--dump`
  curation works for every family this schema references.
- `"normal_conjugate_gibbs"`: a conjugate Normal(mean)-InverseGamma(variance) model, transcribed
  VERBATIM from `Test_Gibbs_NormalDist_RStan` (`family` must be `"Normal"`; the conjugate math is
  Normal-InverseGamma-specific, not family-generic). `McmcModel` gains a third member,
  `proposal` (a `Gibbs::Proposal` closure; empty/falsy for every other registry entry). The
  proposal closure captures the SAME `shared_ptr<Normal>`/`shared_ptr<InverseGamma>` prior
  objects also held in `priors` -- see the header comment for why `shared_ptr` (not
  `unique_ptr`) backs `McmcModel::priors`: `Gibbs::Proposal` is a `std::function`, which requires
  its captured state to be copy-constructible, something a `unique_ptr` capture cannot satisfy.
  `SetParameters`/`set_parameters` inside the closure mutates the shared prior instance in place
  every call, exactly mirroring the C# closure's `muPrior.SetParameters(...)`/
  `sigmaPrior.SetParameters(...)` reference-type mutation -- see
  `docs/upstream-csharp-issues.md` for a note on the closure's `mu0 / 2` term (transcribed
  verbatim; unobservable in this fixture's `mu0 = 0` test data, but not the textbook
  conjugate-Normal-update formula).

`construct.settings` holds only **non-default** sampler settings (every key optional):
`initialize` (`"MAP"` | `"Randomize"` | `"UserDefined"` -- a plain field, does NOT trigger
`reset()`), `prng_seed`, `initial_iterations`, `warmup_iterations`, `iterations`,
`number_of_chains`, `thinning_interval` (all integers, applied via the `reset()`-triggering
setters, in that order -- each setter's C# property (`PRNGSeed`/`InitialIterations`/
`WarmupIterations`/`Iterations`/`NumberOfChains`/`ThinningInterval`) calls `Reset()` on assignment),
`output_length` (int; also a plain field like `initialize` -- `OutputLength`'s C# property has no
`Reset()` call, so setting it does not re-seed or re-initialize anything), `proposal_sigma`
(RWMH-specific; a *sentinel string*, not a literal matrix, since every current case only needs a
zero or identity matrix): `"zeros"` is the literal `Matrix(D)` the C# `Test_RWMH_NormalDist_RStan`
test constructs; `"identity"` is the `D x D` identity matrix and has **no upstream literal
counterpart** -- see the divergence note below. Omitting `proposal_sigma` also defaults to zeros.
And `scale`/`beta` (ARWMH-specific doubles; override the ctor-computed defaults `Scale = 2.38^2 /
D` and `Beta = 0.05`) -- none of the ARWMH rstan cases below override them (the C# tests don't
either), but the `normal_short_exact` case does, to exercise both the JSON-settings wiring itself
and (with `beta: 0.0` forcing the identity-covariance-vs-adaptive-covariance branch to be decided
purely by the `100 * NumberOfParameters` sample-count threshold, never by the Beta-test draw) the
adaptive-covariance (`RunningCovarianceMatrix`-scaled) proposal branch, which the rstan cases only
reach implicitly after their first `100 * D` samples.

`jump`/`jump_threshold`/`noise` (DEMCz- and DEMCzs-specific doubles; override the ctor-computed
`Jump = 2.38 / sqrt(2*D)`, `JumpThreshold = 0.1`, and `Noise = 1e-12`) and `snooker_threshold`
(DEMCzs-only; overrides `SnookerThreshold = 0.1`) -- none of `demcz.json`/`demczs.json`'s cases
below override any of these; every case relies on the ctor defaults, matching
`Test_DEMCz.cs`/`Test_DEMCzs.cs` (neither ever touches these properties).

Gibbs and SNIS force several settings in their own constructors (mirroring the C# ctors, which
write the PROTECTED backing fields directly and call `Reset()` once, rather than going through
the public reset()-triggering setters -- see `gibbs.hpp`/`snis.hpp`'s file headers): Gibbs forces
`NumberOfChains = 1`, `WarmupIterations = 1`, `ThinningInterval = 1`, `InitialIterations = 1`,
`Iterations = 100000`, `OutputLength = 10000`; SNIS forces the same except `WarmupIterations = 0`.
These are ordinary defaults, not permanently enforced settings -- `construct.settings` can still
override them afterward via the normal setters (both fixtures' `*_short_exact` cases do, trimming
`iterations`/`output_length` down to 100 for a fast digest -- SNIS's own `ValidateSettings`
requires `Iterations >= OutputLength >= 100`, which 100/100 satisfies).

**Divergence note -- `"zeros"` + `Randomize` throws (verified against the real C# library):** a
`Randomize`-initialized RWMH sampler with an all-zero `proposal_sigma` throws on its very first
`ChainIteration`, in BOTH the C++ port and the real C# library: `RWMH.ChainIteration` calls
`mvn[index].SetParameters(state.Values, ProposalSigma.Array)` on every iteration, and
`MultivariateNormal.SetParameters` constructs a `CholeskyDecomposition` of the covariance
unconditionally, which throws for any non-positive-definite (including all-zero/singular) input
(`CholeskyDecomposition`'s own `sum <= 0.0` diagonal-pivot guard). This is harmless for a `MAP`-
initialized sampler ONLY because `RWMH.InitializeCustomSettings` overwrites `ProposalSigma` with
the MAP/Fisher-information-derived covariance BEFORE the first `ChainIteration` call -- but for
`Randomize` initialization (no MAP/Fisher covariance to substitute), `proposal_sigma` stays
whatever `construct.settings` set it to for the entire run, so an all-zero matrix is never usable.
Confirmed with a standalone console app against the real built `upstream/Numerics` library (see
`.superpowers/sdd/task-p3-5-report.md`); logged in `docs/upstream-csharp-issues.md` as a
CONSISTENCY/API note (not a bug -- an all-zero proposal covariance is not a meaningful RWMH
configuration under either language). `"identity"` is this fixture set's substitute for any
`Randomize`-initialized case; it is the simplest non-degenerate covariance, not a tuned one --
these fixtures aim for exact cross-language reproducibility, not statistical realism, so a
near-stationary chain (very small steps relative to the parameter scale) is fine.

Assertion methods (0-based indices throughout):

- `posterior_mean|posterior_sd|posterior_median|posterior_lower_ci|posterior_upper_ci [p]` --
  `MCMCResults.ParameterResults[p].SummaryStatistics.{Mean,StandardDeviation,Median,LowerCI,UpperCI}`.
- `chain_value [chain, draw, p]` -- `MarkovChains[chain][draw].Values[p]` (the raw, un-thinned-
  again per-iteration chain state -- `MarkovChains` already reflects `ThinningInterval`). **SNIS
  exception:** SNIS's `MarkovChains[0]` is not time-ordered at all -- `Sample()` records every
  draw into it independently (no accept/reject, no chain dynamics), then SORTS the ENTIRE list by
  `Fitness` ascending as part of computing the resampling CDF (see `Sample()`'s own docstring on
  this "ascending order of posterior weights" comment being misleading -- it actually sorts by
  `Fitness`, not `Weight`; logged as a finding in `docs/upstream-csharp-issues.md`), so `draw` here
  indexes into that POST-SORT array, not draw order. `Output` (the resampled posterior list
  `posterior_mean`/etc. above read via `MCMCResults`) is a SEPARATE, further-resampled collection
  with no directly exposed per-index accessor in this schema; the digest/rstan cases in
  `snis.json` therefore assert on `MarkovChains[0]` (documented explicitly in-fixture).
- `chain_fitness [chain, draw]` -- `MarkovChains[chain][draw].Fitness`.
- `map_value [p]` -- `MCMCResults.MAP.Values[p]`.
- `map_fitness []` -- `MCMCResults.MAP.Fitness`. See the MAP-fitness-sign-quirk note in
  `docs/upstream-csharp-issues.md`: after a successful MAP initialization, this value is
  `DifferentialEvolution`'s *scaled* fitness (`-logLH`), not the sampler's own unscaled
  log-likelihood convention every chain state otherwise uses.
- `acceptance_rate [chain]` -- `AcceptanceRates[chain]`.
- `mean_log_likelihood [i]` -- `MeanLogLikelihood[i]`.
- `rhat [p]`, `ess [p]` -- `ParameterResults[p].SummaryStatistics.{Rhat,ESS}`.

**Tolerance policy for the two RWMH cases:**

- `normal_rstan` (`Initialize = MAP`, transcribed from `Test_RWMH_NormalDist_RStan`): the 10 rstan
  literals (mean/sd/5%/50%/95% for mu and sigma) at `mode: "rel", tol: 0.05`, EXACTLY as the C#
  test asserts. The curated `chain_value`/`chain_fitness` companions (first 5 draws x 4 chains x 2
  params, via `--dump`) use `mode: "rel", tol: 1e-5` -- **not** `1e-12`. This is a deliberate,
  empirically-justified deviation: a `MAP`-initialized chain's proposal covariance is derived
  through `DifferentialEvolution` (a population search with floating-point `<=` comparison
  branches) -> numerical Hessian -> `Matrix.Inverse()` -> `Cholesky`-backed `MultivariateNormal`, a
  much longer chain of floating-point operations than a plain random-walk step. Measured directly
  against the real C# library, the resulting first-5-draws-per-chain agree to ~1e-8 to 3e-8
  relative (`map_value` to ~8e-9 relative), not ~1e-15 -- consistent with the Task 3 finding that
  BrentSearch-style iterative selection can diverge from C# by a libm-ULP branch on transcendental
  objectives (see the Task 3 progress-ledger note), here amplified through ~30-generation
  Differential Evolution rather than a root-search. `map_value` still uses `tol: 1e-4` and
  `acceptance_rate` `mode: "abs", tol: 5e-3`, comfortably covering the measured ~1e-8/exact-match
  reality with margin.
- `normal_short_exact` (`Initialize = Randomize`, `proposal_sigma = "identity"`, trimmed
  `iterations`/`warmup_iterations`/`thinning_interval` -- the smallest legal `ValidateSettings`
  config): NO `DifferentialEvolution`/Hessian/Fisher-information machinery is on this path at all,
  so this is the TRUE cross-language chain-identity digest and ULP tripwire the brief intends:
  curated `chain_value`/`chain_fitness` (first 5 + last draw per chain, both params) at
  `mode: "rel", tol: 1e-12` -- measured agreement against the real C# library is ~1e-15 relative,
  three orders of magnitude inside this tolerance. `map_value` and `mean_log_likelihood`/`rhat`/
  `ess` use `tol: 1e-9`, and `acceptance_rate` (an exact ratio of small integers, since both
  languages consume the identical PRNG stream and therefore make identical accept/reject decisions)
  uses `mode: "equal"`.

**Tolerance policy for the four ARWMH rstan cases + `normal_short_exact`** (`arwmh.json`): unlike
RWMH's `normal_rstan`, NONE of `Test_ARWMH.cs`'s four rstan tests set `Initialize = MAP` -- they
all use the default `Initialize = Randomize`, so there is no `DifferentialEvolution`/Hessian/
Fisher-information machinery anywhere on these paths (per the P3.5 TOLERANCE POLICY REFINEMENT:
Randomize-init chain companions get the TRUE `rel: 1e-12` digest, not `1e-5`). Each of
`normal_rstan`/`logistic_rstan`/`gumbel_rstan` asserts its 10 rstan literals at `mode: "rel", tol:
0.05` (mean/sd/5%/50%/95% for both parameters) EXACTLY as the C# test asserts, plus curated
`chain_value`/`chain_fitness` companions (first 5 draws x 4 chains x 2 params) at `mode: "rel",
tol: 1e-12` and `acceptance_rate` at `mode: "equal"` (both languages consume the identical PRNG
stream end to end, so this is an exact ratio of small integers, not merely close). `weibull_rstan`
is identical except `settings.iterations = 20000` (`Test_ARWMH_WeibullDist_RStan`'s own comment:
"Requires much more sample to converge for this distribution") and `mode: "rel", tol: 0.10` on the
10 rstan literals, EXACTLY matching the C# test's own `0.1 *` tolerance multiplier (do not use
`0.05` here). `normal_short_exact` trims `iterations`/`warmup_iterations`/`thinning_interval` to
100/10/1 and additionally overrides `scale: 1.0, beta: 0.0` (see the settings section above) --
dense `chain_value`/`chain_fitness` digests (first 5 + last draw per chain, both params) at `mode:
"rel", tol: 1e-12`, `map_value`/`mean_log_likelihood`/`rhat`/`ess` at `tol: 1e-9`, and
`acceptance_rate` at `mode: "equal"`.

**Tolerance policy for the two Gibbs cases** (`gibbs.json`): `normal_rstan` (default settings,
transcribed from `Test_Gibbs_NormalDist_RStan`) asserts its 10 rstan literals at `mode: "rel",
tol: 0.05`, plus curated `chain_value`/`chain_fitness` digests at draws 0-4 (INCLUDING draw >= 1,
the mutated-prior path -- see the model-registry section above) at `mode: "rel", tol: 1e-12`
(Gibbs has no `DifferentialEvolution`/MAP path at all -- every draw is a plain conjugate-posterior
sample, so the whole run is bit-reproducible). `acceptance_rate` is asserted at `mode: "equal",
expected: 0.0`: Gibbs's `ChainIteration` never touches `AcceptCount`/`SampleCount` (see
`gibbs.hpp`'s file header -- "a Gibbs draw from the full conditional is always accepted", so there
is no accept/reject step to count at all; this is an intentional invariant, not an unexercised
default). `normal_short_exact` overrides `settings.iterations = 100, output_length = 100` (Gibbs's
ctor forces `Iterations = 100000`, but nothing prevents a caller overriding it afterward via the
ordinary setters -- see the settings section above) with the same dense digest + `mode: "equal"`
acceptance-rate pattern.

**Tolerance policy for the two SNIS cases** (`snis.json`): `normal_rstan` (`prng_seed: 45678,
initialize: "MAP"`, transcribed from `Test_SNIS_NormalDist_RStan`) asserts its 10 rstan literals
at `mode: "rel", tol: 0.05`, plus (per the P3.5 TOLERANCE POLICY REFINEMENT for MAP-init cases)
`chain_value`/`chain_fitness` companions at `mode: "rel", tol: 1e-5` and `map_value`/`map_fitness`
at `tol: 1e-4`. **Draw-index hazard (new, this task):** the companion draws are deliberately the
TOP five indices of the 100000-length `MarkovChains[0]` (`99995`-`99999`), not the first five. A
first attempt at this fixture used the natural-looking `[0, 1, 2, 3, 4]` and every `chain_value`
assertion in that range FAILED to reproduce against the C++ port (while `chain_fitness` passed) --
diagnosed as a genuine cross-language sort-stability divergence, not a transcription bug: this
model's wide, uninformative Uniform priors make MANY draws' log-likelihood underflow to exactly
`-Infinity` (`3` of `100000` in the rstan case, `12` of `100` in `normal_short_exact`), and those
tied `-Infinity` entries cluster at the BOTTOM of the fitness-ascending sort. `List<T>.Sort` (C#,
an unstable introspective sort) and `std::stable_sort` (this port -- see `snis.hpp`'s own
SORT-COMPARATOR file-header note) are both free to place EQUAL elements in different relative
order, so which specific tied `-Infinity` draw ends up at index 0 vs. index 1 vs. ... genuinely
differs between the two languages, even though the SET of values at those indices (all
`-Infinity`) is identical. `chain_fitness` assertions at those low indices still pass (`-Infinity
== -Infinity` regardless of WHICH draw produced it), but a `chain_value` assertion pinned to a
specific low index is not a safe cross-language digest. The untied, strictly-monotonic-fitness
tail near the top of the sort (`chain_value` differs measurably between adjacent high indices --
see the raw `--dump` output) has no such hazard; logged as a new finding in
`docs/upstream-csharp-issues.md`. `normal_short_exact` (`Initialize = Randomize` -- the default;
`settings.iterations = 100, output_length = 100`, the smallest legal `ValidateSettings` config
per SNIS's own override) is naive Monte Carlo with no `DifferentialEvolution`/MAP machinery, so
its `chain_value`/`chain_fitness` companions (top 5 of 100, indices `95`-`99`, for the identical
tie-hazard reason above) use the TRUE `mode: "rel", tol: 1e-12` digest tolerance.

**Tolerance policy for the DEMCz/DEMCzs cases** (`demcz.json`/`demczs.json`): both files carry
`normal_rstan`/`logistic_rstan`/`gumbel_rstan`/`weibull_rstan` (all `Initialize = Randomize`, the
default -- neither `Test_DEMCz.cs` nor `Test_DEMCzs.cs` ever sets `Initialize`, so there is no
`DifferentialEvolution`/Hessian machinery on any of these paths) plus a `normal_short_exact` case
(`number_of_chains: 3`, the smallest legal `ValidateSettings` config for both samplers -- `< 3`
throws `"There must be at least 3 chains."`). Each rstan case asserts its 10 rstan literals at
`mode: "rel", tol: 0.05`, plus curated `chain_value`/`chain_fitness` companions (first 5 draws x 4
chains x 2 params) at the TRUE `mode: "rel", tol: 1e-12` digest tolerance -- confirmed via a
standalone throwaway harness against the real C# library to reproduce to ~1e-14 relative or
tighter at these EARLY draws, for both samplers.

**Population-sampler divergence finding (new, this task, NOT an upstream C# bug -- an inherent
property of the DE-MC algorithm family, equally present in both languages):** unlike RWMH/ARWMH/
Gibbs/SNIS, DEMCz and DEMCzs are `IsPopulationSampler = true` -- every chain's proposal draws from
the SAME, ever-growing, cross-chain `PopulationMatrix` (`R1`/`R2` index into it; `population_matrix_.
push_back(...)` runs once per chain per outer `sample()` iteration, in BOTH the `Iterations` phase
AND the trailing `OutputLength`-driven "output phase" -- see `mcmc_sampler.hpp`'s `sample()`). This
is a fundamentally different coupling from RWMH/ARWMH's per-chain-independent proposals (each of
their chains reads only its OWN history), and it makes DEMCz/DEMCzs's long-run aggregate statistics
measurably LESS cross-language-reproducible than the early-draw digest: a single accept/reject
decision that differs by a libm-ULP between languages (occasionally inevitable over enough draws,
same root cause as the RWMH MAP-path/BrentSearch findings already logged) immediately corrupts the
SHARED pool every OTHER chain proposes from next, compounding across O(10^4-10^5) total
`ChainIteration` calls per rstan case. Diagnosed directly (a throwaway harness plus a targeted
`--dump` sweep over draws 0/100/200/.../3400 of `normal_rstan`'s chain 0): the per-draw relative
divergence from the real C# library is ~1e-15 at draw 0, grows past ~1e-8 by draw ~300, and reaches
several percent by draw ~500 -- textbook chaotic amplification of sub-ULP noise, not a transcription
defect (the SHORT digest window, draws 0-4, sits entirely before this amplification becomes
significant). Two `MCMCResults` quantities are built from data spanning this divergent tail and are
NOT bit-reproducible as a result: `AcceptanceRates` (accumulates over the ENTIRE run, output phase
included) and everything derived from `Output` (`posterior_mean`/`sd`/`median`/`lower_ci`/
`upper_ci`, and `ESS` -- `effective_sample_size(sampler.output())`). By contrast, `Rhat`
(`gelman_rubin(sampler.markov_chains(), ...)`), `chain_value`/`chain_fitness`
(`MarkovChains` directly), `MeanLogLikelihood` (accumulated only while `i <= Iterations`, i.e.
entirely within the `MarkovChains`-recorded phase), and `MAP`/`MAP.Fitness` (tracks the best
fitness seen -- both languages' searches land on comparably-good optima even when their specific
trajectories diverge) all stay near-bit-exact throughout, confirmed by the same sweep. Accordingly:
`acceptance_rate` on every DEMCz/DEMCzs rstan case (and DEMCzs's `normal_short_exact`, whose
`OutputLength`-driven output phase is `ceil(10000/3) = 3334` iterations -- MUCH longer than the
250-iteration `MarkovChains` window the digest itself covers) uses `mode: "rel", tol: 0.05`, not
`"equal"`; `chain_value`/`chain_fitness`/`map_value`/`map_fitness`/`mean_log_likelihood`/`rhat`
keep the tight digest tolerances (`1e-12`/`1e-9`, exactly the RWMH/ARWMH/Gibbs precedent). DEMCz's
`normal_short_exact` (no snooker branch, chains=3, only 250 total `ChainIteration` calls) is the
one exception that stays bit-exact end to end: its `acceptance_rate` keeps `mode: "equal"`, and
only `ess` needs loosening, from `1e-9` to `1e-5` (measured ~1.4e-6 relative -- `ESS`'s Geyer-style
autocorrelation-lag truncation is itself a discontinuous function of its input series, so even the
sub-ULP `MarkovChains` noise already present can occasionally nudge a truncation-lag decision by
one step and shift `ESS` measurably more than the `MarkovChains` values themselves moved).
DEMCzs's `normal_short_exact` additionally exercises the snooker branch's `Vector::project`/
`Vector::distance` arithmetic (extra dot-product/norm/sqrt operations per snooker draw, atop the
already-present likelihood-evaluation noise), which needs `chain_value` loosened from `1e-12` to
`1e-11` (measured max ~2.9e-12 across an EXHAUSTIVE sweep of all 3 chains x 250 draws x 2 params --
confirming there is no genuine trajectory divergence anywhere in the `MarkovChains` window, only
slightly larger sub-ULP noise than DEMCz's simpler branch) and `ess` loosened to `0.25` (measured
~16% relative -- the same Geyer-truncation discontinuity as DEMCz above, amplified further because
`output()`'s `3334`-iteration divergent tail feeds `effective_sample_size` directly here, unlike
`Rhat`, which stays exact because it reads only `MarkovChains`). One
degenerate `chain_value` entry (`DEMCzs`/`normal_rstan`, chain 3, draw 0-4, `mu` parameter) is
curated as EXACTLY `0.0` -- that chain's Latin-Hypercube initial draw landed on the `mu` prior's
literal lower bound and never escaped it (every subsequent proposal for that chain was rejected,
freezing it there for the life of the run, in BOTH languages identically) -- `mode: "equal"` is
used there instead of `"rel"` (a relative-tolerance check against an EXACT-zero expected value is
undefined, `0/0`, not a looser check).

**Tolerance policy for the two HMC cases** (`hmc.json`, P3.8; `chain_value`/`chain_fitness`/
`map_value` below all read from the general `mcmc_sampler` assertion-method list above --
`MarkovChains[chain][draw].Values[p]`/`.Fitness` and `MCMCResults.MAP.Values[p]`): `normal_rstan`
(`step_size: 2.5, steps: 10`, transcribed from `Test_HMC_NormalDist_RStan`, `Initialize =
Randomize` -- the default) asserts its 10 rstan literals at `mode: "rel", tol: 0.05`, plus a
curated `chain_fitness` companion (first 3 draws per chain, not 5) at `mode: "rel", tol: 1e-9` --
**not** the RWMH/ARWMH/Gibbs/DEMCz(s) family's `1e-12`: HMC's default gradient is a
finite-difference approximation (`NumericalDerivative.Gradient`/`differentiation::gradient()`,
P3.3), and every leapfrog step evaluates it -- each evaluation is itself several extra
floating-point operations (adaptive step-halving, central/one-sided differencing) beyond a plain
likelihood call, so a `step_size`/`steps` jitter draw, a momentum draw, or an accept/reject
comparison lands closer to a decision boundary sooner than the gradient-free samplers'
digests. `chain_value` had the same companion at the same tolerance through CI run 28657005803;
see the cross-platform trip note below for why it was dropped. `normal_short_exact` (`Initialize
= Randomize`, `step_size: 2.5, steps: 5`, trimmed `iterations`/`warmup_iterations`/
`thinning_interval` -- the smallest legal `ValidateSettings` config) uses the same
`chain_value`/`chain_fitness` `tol: 1e-9`/first-3-draws digest, plus
`map_value`/`map_fitness`/`mean_log_likelihood`/`rhat` at `tol: 1e-9`, `ess` at `tol: 0.25` (see
below), and `acceptance_rate` at `mode: "equal"` (both languages consume the identical PRNG
stream through this short a run, so accept/reject decisions still land identically end to end). A
third, C++-only smoke test (`test_mcmc_extra.cpp`, transcribed from
`Test_HMC_NonFiniteGradient_DoesNotCrash`) exercises `ChainIteration`'s `catch (const
std::domain_error&)` -- the non-finite-gradient guard -- with narrow priors and an aggressive
`step_size`/`steps`; it asserts only "does not throw" + "produced output", no numeric literal
(same rationale as the SNIS invalid-weight C++-only cases above).

**Cross-platform CI trip (new, this task, diagnose-and-shorten per the divergence playbook -- NOT
an upstream C# bug, and NOT a widened tolerance):** CI run 28657005803 (PR #5) was green on macOS
(the curation platform) and failed exactly 3 of 3741 fixture checks on Ubuntu (3x
`HMC/normal_rstan/chain_value`) and Windows (2x `HMC/normal_rstan/chain_value` + 1x
`HMC/normal_short_exact/ess`); the `r-cmd-check`/`python` job failures on those platforms were the
same fixture files propagating through the other two runners, not independent bugs. Root cause:
HMC's bound-aware finite-difference gradient divides a Normal log-likelihood difference by
`h~1e-5`; an ~1-ULP libm difference between glibc/MSVC and Apple's `libsystem_m` in that
likelihood evaluation amplifies to ~1e-11 relative gradient noise, which leapfrog integrates
through `normal_rstan`'s MAP/DE-initialized path and can flip an early accept/reject decision --
the same finite-difference-amplification mechanism the tolerance-policy paragraph above already
documents for `chain_value`/`chain_fitness` generally, just observed in practice on the other two
platforms rather than macOS. Per the playbook ("a CI platform trip means diagnose, record in the
ledger, and shorten/drop the probe -- posterior summaries remain the always-on gate"), the fix
drops rather than loosens: `normal_rstan`'s `chain_value` assertions (24 entries: 4 chains x 3
draws x 2 params) are removed outright -- `chain_fitness` at the identical indices passed on every
platform in that run and is kept unchanged, as are the 10 rstan posterior literals. macOS and the
dotnet oracle gate (`verify_oracles.py`) still reproduce the dropped `chain_value` literals against
the real C# library, so they remain valid, just no longer asserted cross-platform.
`normal_short_exact`'s single `ess` trip is handled via the DEMCzs precedent instead of dropping:
`ess`'s Geyer-style autocorrelation-lag truncation is a discontinuous function of its input series
(same class of finding as the DEMCz/DEMCzs `ess` loosenings above), so the same sub-ULP noise can
flip a truncation-lag decision by one step and move `ess` measurably more than the underlying chain
values moved. `ess` is loosened from `tol: 1e-9` to `tol: 0.25`, matching DEMCzs's
`normal_short_exact` `ess` treatment exactly (same discontinuity, same fixture shape); `rhat`, which
reads only `MarkovChains` with no output-phase truncation step, stayed exact on every platform in
that run and is untouched. See `.superpowers/sdd/progress.md`'s Phase 3 section for the dated
ledger entry.

**Tolerance policy for the two NUTS cases** (`nuts.json`, P3.8): `normal_rstan`/`logistic_rstan`/
`gumbel_rstan` (all default settings -- `Test_NUTS.cs` never overrides `stepSize`/`maxTreeDepth`,
and `Initialize = Randomize` is the default) each assert their 10 rstan literals at `mode: "rel",
tol: 0.05`, exactly transcribing `Test_NUTS_NormalDist_RStan`/`Test_NUTS_LogisticDist_RStan`/
`Test_NUTS_GumbelDist_RStan`. `normal_short_exact` (`Initialize = Randomize`, `iterations: 100,
warmup_iterations: 10, thinning_interval: 1, output_length: 100` -- the smallest legal
`ValidateSettings` config, `iterations`/`output_length` both floored at 100) asserts curated
`chain_value`/`chain_fitness` companions for the first 3 draws x 4 chains x 2 params at the same
HMC-precedent `mode: "rel", tol: 1e-9` (measured worst case ~9.1e-10 relative against the real C#
library -- comfortably inside), plus `map_value`/`map_fitness`/`mean_log_likelihood` at `tol:
1e-9` (measured worst case ~8.6e-10) and `acceptance_rate` at `mode: "equal", expected: 1` for
every chain -- NOT a measured coincidence: `ChainIteration` increments `AcceptCount` UNCONDITIONALLY
every call (see `nuts.hpp`'s file header), so NUTS's "acceptance rate" is always exactly 1.0 by
construction in both languages, unlike every Metropolis-family sampler above. One `chain_value`
entry (chain 3, param 0 (`mu`), draws 0 and 1) is curated as EXACTLY `0.0`: that chain's proposal
is bit-identical across draws 0 and 1 in BOTH languages, consistent with `ChainIteration`'s
tree-build-divergence guard (`BuildTree`'s depth-0 base case returning `InvalidTreeState` when the
trial Hamiltonian is non-finite or diverges by more than `MAX_DELTA_H`) leaving the input state
unmodified twice in a row; `mode: "equal"` is used there instead of `"rel"` for the same `0/0`
reason as the DEMCzs precedent above.
**NUTS `rhat`/`ess` divergence finding (new, this task, NOT an upstream C# bug -- an inherent
property of the recursive-tree-doubling algorithm, equally present in both languages):** unlike
every other sampler ported so far, a single NUTS `ChainIteration` can itself draw MANY random
numbers (a direction draw plus a multinomial subtree-acceptance draw at every tree depth, up to
`MaxTreeDepth`), not just one -- so a single ULP-level floating-point difference between languages
(inevitable over enough comparisons, same root cause as the DEMCz/DEMCzs and RWMH MAP-path
findings already logged) has many more opportunities per iteration to flip an accept/reject or
U-turn decision than any prior sampler. Measured directly against the real C# library on
`normal_short_exact`'s mandatory-minimum ~100-iteration window: `chain_value`/`chain_fitness`
(first 3 draws) and `map_value`/`mean_log_likelihood` all stay within ~1e-9 relative (the intended
digest tolerance), but `Rhat` (`gelman_rubin(sampler.markov_chains(), ...)`, spanning all 100
recorded draws) and `ESS` (`effective_sample_size(sampler.output())`, spanning the trailing
25-draw output-phase window) diverge measurably further -- ~7.9e-6/3.5e-6 relative for `Rhat`'s two
parameters and ~3.3e-5/2.8e-8 for `ESS`'s -- three to five orders of magnitude looser than the
1e-9 digest but still five orders of magnitude tighter than the 0.05 rstan tolerance, confirming
this is sub-ULP chaotic amplification over the many extra tree-building comparisons, not a
transcription defect (the same conclusion the DEMCz/DEMCzs population-sampler finding reached for
a different mechanism). `rhat`/`ess` are therefore asserted at `mode: "rel", tol: 1e-4` on
`normal_short_exact` -- roughly 3x-13x margin over the worst measured value, not loosened further.

**NEVER loosen a tolerance below what's documented above.** If a curated value fails to reproduce,
the streams have desynced somewhere -- diagnose the draw path (`--dump` intermediates, compare
against a standalone throwaway harness against the real C# library) and fix the transcription slip;
do not paper over it with a looser tolerance.

### `bootstrap`

```jsonc
{
  "target":  "Bootstrap",
  "kind":    "bootstrap",
  "source":  "Numerics/Sampling/Bootstrap/Bootstrap.cs ; Test_Numerics/Sampling/Test_Bootstrap.cs",
  "datasets": { "bca_sample": [ ...100 numbers... ] },
  "cases": [
    {
      "name": "percentile",
      "construct": {
        "model": "normal_quantiles",
        "mu": 3.122599, "sigma": 0.5573654, "sample_size": 100,
        "probabilities": [0.999, 0.99, 0.95, 0.9, 0.5, 0.1, 0.05, 0.01],
        "replicates": 10000, "seed": 12345,
        "run": "regular", "ci_method": "Percentile", "alpha": 0.1
      },
      "assertions": [
        { "method": "statistic_lower_ci", "args": [0], "expected": 4.621151758616323, "mode": "rel", "tol": 1e-9 }
      ]
    }
  ]
}
```

One bootstrap run per case (mirrors `mcmc_sampler`'s single-stateful-glue-call contract): build
the model via the model registry (below), configure `replicates`/`seed`/`max_retries`, call `Run()`
or `RunWithStudentizedBootstrap()` **once** per `construct.run`, then call
`GetConfidenceIntervals(ci_method, alpha)` **once**. Every assertion in the case reads that single
cached `(Bootstrap, BootstrapResults)` pair -- each of the four runners makes exactly one glue call
per case (C++ caches the pair locally; R's `bf_bootstrap_run_`/Python's `_core.bootstrap_run` each
return one big list/dict every assertion for the case reads from).

**Scope note:** this kind (and the underlying `Bootstrap<TData>` port) covers only the REGULAR
(non-pivotal) workflow -- `Run()`, `RunDoubleBootstrap()` (ported, not fixture-wired -- see below),
`RunWithStudentizedBootstrap()`, and `GetConfidenceIntervals`'s Percentile/BiasCorrected/BCa/Normal/
BootstrapT methods. The covariance-aware PIVOTAL bootstrap (`RunPivotalBootstrap`,
`TransformPivotalBootstrap`, `GetRawPivotalConfidenceIntervals`, and every `Pivotal*` construct
field) is out of scope for this task -- ported in a follow-up task, which will extend this schema
with a `"run": "pivotal"` value and the additional `PivotalLinkFactory`/`PivotalInvalidDrawPolicy`/
etc. settings. `RunDoubleBootstrap` is ported in full in `bootstrap.hpp` (per the task brief) but
has no fixture case: `Test_DoubleBootstrap` itself asserts only bracket inequalities ("CI brackets
the true quantile"), not a fixed numeric literal, so it is not useful oracle-fixture material under
this repo's curated-literal convention.

**Model registry** (`core/include/bestfit/numerics/sampling/bootstrap/model_registry.hpp`): a
**bestfit addition** with no upstream C# counterpart -- the C# equivalent is inline construction
code living only inside `Test_Bootstrap.cs`'s private `CreateNormalBootstrap()` helper and
`Test_BCaCI()`, not in the library. The registry is **closed**; today it has one entry:

- `"normal_quantiles"`: parametric bootstrap of a `Normal(mu, sigma)` distribution fitted by
  Method of Moments, evaluated at a fixed set of quantile probabilities. `ResampleFunction` draws
  `resample_size` values from `Normal(ps[0], ps[1])` via `GenerateRandomValues(resampleSize,
  rng.Next())` -- `resample_size` is a FIXED closure-captured value (`construct.sample_size`, or
  `dataset`'s length when a dataset is supplied), not derived from the delegate's own `data`
  argument, matching both `CreateNormalBootstrap` and `Test_BCaCI` ignoring their own `data`
  parameter. `FitFunction` is a Method-of-Moments `Normal` fit (throws `"Invalid parameters."` when
  the fit is invalid, mirroring `if (!d.ParametersValid) throw ...`). `StatisticFunction` is
  `Normal(ps[0], ps[1]).InverseCDF(p)` for each `p` in `construct.probabilities`.

  `construct.dataset`, when present, names an entry in the file-level `datasets` map supplying a
  REAL sample (`Test_BCaCI`'s literal 100-value array, curated as `bca_sample` above): `mu`/`sigma`/
  `sample_size` are then IGNORED and instead `mu`/`sigma` are fit from the dataset via Method of
  Moments (mirrors `Test_BCaCI`'s own `((IEstimation)dist).Estimate(sampleData, ...)` call before
  constructing `Bootstrap`), `resample_size` becomes the dataset's length, and
  `JackknifeFunction`/`SampleSizeFunction` are wired to leave-one-out over the dataset -- required
  by the `BCa` CI method (`ComputeAccelerationConstants` needs both). When `construct.dataset` is
  absent, the original data is unused (matches `CreateNormalBootstrap`'s `new
  Bootstrap<double[]>(null, parms)`) and `JackknifeFunction`/`SampleSizeFunction` are left unset --
  `BCa` is not requestable on that configuration, mirroring the C# test's own
  `Test_BCa_RequiresJackknifeFunction`. This `dataset`-driven divergence from a flat `{model, mu,
  sigma, sample_size, probabilities}` schema is a deliberate, documented extension needed to give
  the `BCa` case (which the brief explicitly scopes in, tolerance and all) a real sample to
  jackknife over -- the alternative (omitting `BCa` from the fixture entirely) would leave the
  acceleration-constant code path, and its HAZARD tolerance policy below, unverified.

`construct` fields (every key besides `model`/`probabilities`/`replicates`/`seed`/`ci_method`
optional): `mu`/`sigma` (doubles, ignored when `dataset` is set), `sample_size` (int, ignored when
`dataset` is set), `probabilities` (array of doubles, the quantile levels `StatisticFunction`
evaluates), `dataset` (string, a `datasets` key), `replicates`/`seed` (ints, override `Bootstrap`'s
`Replicates`/`PRNGSeed` -- both required by every case in practice, since the 10,000-default
`Replicates` makes an omitted value expensive but not incorrect), `max_retries` (int, default 20,
overrides `MaxRetries`), `run` (`"regular"` | `"studentized"` -- selects `Run()` vs.
`RunWithStudentizedBootstrap()`), `ci_method` (`"Percentile"` | `"BiasCorrected"` | `"BCa"` |
`"Normal"` | `"BootstrapT"`, matching the C# `BootstrapCIMethod` enum names verbatim), `alpha`
(double, default 0.1, the two-sided CI alpha).

Assertion methods (0-based indices throughout):

- `statistic_lower_ci|statistic_upper_ci [i]` -- `BootstrapResults.StatisticResults[i].{LowerCI,
  UpperCI}` (`i` indexes `construct.probabilities`).
- `parameter_lower_ci|parameter_upper_ci [p]` -- `BootstrapResults.ParameterResults[p].{LowerCI,
  UpperCI}` (`p` indexes the fitted `Normal` parameters, `0` = mu, `1` = sigma). Always Percentile
  CIs regardless of `ci_method` -- `GetConfidenceIntervals` computes `ParameterResults` via
  `ComputePercentileCI` unconditionally, matching the C# source.
- `population_estimate [p]` -- `BootstrapResults.ParameterResults[p].PopulationEstimate` (echoes
  `_originalParameters.Values[p]` unchanged -- curated directly from the literal `_mu`/`_sigma` test
  constants, not `--dump`, since this value is a pure input echo with no floating-point computation
  involved).
- `valid_count [i]` -- `BootstrapResults.StatisticResults[i].ValidCount`, `mode: "equal"` (an exact
  integer count -- the "normal_quantiles" model's Method-of-Moments fit never invalidates on a
  parametric Normal resample, so every non-BCa case's `valid_count` equals `Replicates` exactly).
- `replicate_value [idx, p]` -- `Bootstrap.BootstrapParameterSets[idx].Values[p]`, the exact seeded
  replicate's fitted parameter value. `mode: "rel", tol: 1e-12` -- Method-of-Moments is a
  closed-form (non-iterative) fit applied to a `MersenneTwister`-seeded resample, so this is a TRUE
  cross-language bit-identity digest, the same role `chain_value` plays for `mcmc_sampler` fixtures.

**Tolerance policy:** every CI-bound assertion (`statistic_lower_ci`/`statistic_upper_ci`/
`parameter_lower_ci`/`parameter_upper_ci`) on the `percentile`/`normal_ci`/`bias_corrected`/
`bootstrap_t` cases uses `mode: "rel", tol: 1e-9` -- these CI methods are entirely deterministic
functions of the `MersenneTwister`-seeded replicate stream (percentile extraction, or a closed-form
Normal-quantile transform of the sorted replicates), so they reproduce the real C# library to
1e-9 or tighter (measured well under 1e-9 for every curated value). `population_estimate` uses
`mode: "equal"` (pure input echo) and `valid_count` uses `mode: "equal"` (exact integer count).
`replicate_value` uses `mode: "rel", tol: 1e-12` (see above).

**BCa HAZARD (per the task brief):** `ComputeAccelerationConstants` (`compute_acceleration_
constants` in this port) uses C#'s `Tools.ParallelAdd` inside its own `Parallel.For` over the
jackknife samples -- an order-DEPENDENT floating-point reduction that is not, in general,
guaranteed bit-reproducible run-to-run in the real C# library (thread-scheduling-dependent
summation order). This port replaces it with a plain SERIAL sum in jackknife-index order (see
`bootstrap.hpp`'s file header) -- deterministic within this port, but not a bit-for-bit
reproduction of C#'s reduction order. The `bca` case's `statistic_lower_ci`/`statistic_upper_ci`
therefore use a LOOSE `mode: "rel", tol: 1e-6` (three orders of magnitude looser than the other
CI methods' `1e-9`), sized per the brief's instruction to measure the real C# run-to-run wobble:
the oracle emitter's `--dump` output was diffed across four independent runs (same fixture, same
process invocation, run sequentially) and was found to be BIT-IDENTICAL every time on the
development machine (a low-core-count environment where .NET's `Parallel.For` over only 100
jackknife samples per statistic apparently schedules deterministically) -- i.e. the measured
wobble was exactly `0`, well inside `1e-6`. `1e-6` is kept as the fixture's tolerance regardless of
that zero-wobble measurement, rather than tightening to `1e-9`/`1e-12`: the reduction is
INHERENTLY order-dependent by construction (a different core count, thread pool configuration, or
.NET version could legitimately produce a different summation order and a different last-few-bits
result), so a tight tolerance here would be fragile to environmental factors this repo's CI matrix
does not control for, not a reflection of any measured instability. This is a deliberate,
documented margin -- not evidence the C++ port's serial-sum choice is itself imprecise (the C++
port's own `bca` case values, cross-checked directly against the same C# `--dump` output, agree to
below `1e-9`, well inside the `1e-6` fixture tolerance).

**NEVER loosen a tolerance below what's documented above.** If a curated value fails to reproduce,
the streams have desynced somewhere -- diagnose the draw path (`--dump` intermediates, compare
against a standalone throwaway harness against the real C# library) and fix the transcription slip;
do not paper over it with a looser tolerance.

### `model_estimation`

```jsonc
{
  "target":  "MaximumLikelihood",           // "MaximumLikelihood" | "MaximumAPosteriori" |
                                            // "BayesianAnalysis" | "Simulation" |
                                            // "GeneralizedMethodOfMoments" (B11)
  "kind":    "model_estimation",
  "source":  "RMC-BestFit/src/RMC.BestFit/Estimation/MaximumLikelihood.cs ; core/tests/test_maximum_likelihood.cpp",
  "datasets": { "annual_peaks": [12500, 15300, 9870, 21000, 18400, 11200, 26800, 14100, 19500, 11600] },
  "cases": [
    {
      "name": "normal_nelder_mead",
      "construct": {
        "model":     { "family": "Normal", "dataset": "annual_peaks" },
        "optimizer": "NelderMead"
      },
      "assertions": [
        { "method": "parameter", "args": [0], "expected": 16026.055013737448, "mode": "rel", "tol": 1e-9 }
      ]
    }
  ]
}
```

One `estimate()` run per case (mirrors `mcmc_sampler`'s/`bootstrap`'s single-stateful-glue-call
contract): build the model described by `construct.model` (see the schema below -- `family`
goes through the same univariate distribution factory every other kind uses
(`create_distribution(name)`), `dataset` names an entry in the file-level `datasets` map), then
construct the estimator named by the file-level `target`, call `estimate()` **once**, then dispatch
every assertion in the case against the cached `(model, estimator)` pair (or, for
`BayesianAnalysis`, the cached `(model, analysis)` pair -- see below). Unlike `mcmc_sampler`'s
and `bootstrap`'s model registries (each a small closed name -> construction-recipe map with no
upstream C# library counterpart), `model_estimation`'s "registry" is the shared spec builder
described next: every family the factory supports, plus each model's own default-parameter
machinery (`set_default_parameters`, itself driven by
`IMaximumLikelihoodEstimation::get_parameter_constraints`), is already enough to build a model --
there is no separate closed-name mapping to maintain.

**The `construct.model` schema (M13).** All three harnesses hand the (re-serialized) spec to ONE
shared C++ builder, `bestfit::models::spec::build_model_from_json` in
`core/include/bestfit/models/model_spec.hpp` (a bestfit addition; it parses with the dependency-free
mini reader `models/json_lite.hpp`). The C++ runner serializes with nlohmann `dump()`, R with
`jsonlite::toJSON(auto_unbox = TRUE, digits = I(17))`, Python with `json.dumps()` -- all three
round-trip doubles exactly, so the three harnesses construct byte-identical models. Only the
`model.dataset` reference is resolved by the calling harness (from the file-level `datasets` map,
like every other fixture kind) and passed alongside the spec as a flat vector.

```jsonc
"model": {
  "type": "univariate_distribution",  // default when absent (the Phase 4 behavior) |
                                      // "mixture" | "competing_risks" | "point_process" |
                                      // "bulletin17c" (B11; the GMM/Simulation model type)
  // -- data (exactly ONE of): -------------------------------------------------------------
  "dataset": "annual_peaks",          // a `datasets` key -> ExactSeries with sequential
                                      // 0-based indexes (the Phase 4 vector-ctor path,
                                      // byte-for-byte)
  "data_frame": {                     // a full censored DataFrame, values inline:
    "exact":     [ { "index": 1990, "value": 12500, "is_low_outlier": false } ],
    "interval":  [ { "index": 1985, "lower": 12000, "value": 15000, "upper": 18000 } ],
    "threshold": [ { "start_index": 1900, "end_index": 1979, "value": 22000, "number_above": 1 } ],
    "uncertain": [ { "index": 1988, "distribution": { "family": "Normal", "parameters": [17000, 1500] } } ],
    "low_outlier_threshold": 10000,   // optional; the left-censoring bound for flagged exact points
    "mgbt_low_outliers": true         // optional (M14): run the PUBLIC SetLowOutliersFromMGBT()
                                      // path at the frame boundary instead -- flags the MGBT-detected
                                      // low outliers and sets low_outlier_threshold itself. By fixture
                                      // convention mutually exclusive with explicit `is_low_outlier`
                                      // flags / `low_outlier_threshold` (MGBT clears both first).
  },
  // -- per-type fields: -------------------------------------------------------------------
  "family": "Normal",                 // univariate_distribution: a factory type name
  "trends": [                         // univariate_distribution only, optional (M9 layer):
    { "parameter": 0,                 //   0-based distribution parameter index
      "type": "Linear",               //   a TrendModelType name: Constant | Cubic | Exponential |
                                      //   Linear | Logistic | Power | Quadratic | Reciprocal |
                                      //   Sinusoidal | StepFunction | GeneralLinear (which falls
                                      //   through to ConstantTrend, mirroring the C# SetTrendModel
                                      //   construction chain)
      "start_index": 0,               //   optional: overrides the trend's time anchor
      "values": [15000.0, 120.0] }    //   optional: the trend's own parameter values
  ],
  "families": ["Normal", "Normal"],   // mixture / competing_risks: component factory names
  "zero_inflated": false,             // mixture only, optional (default false)
  "use_defaults": true,               // point_process only, optional knobs, applied in this
  "threshold": 900.0,                 //   order after the DataFrame cascade (an explicit
  "total_years": 20.0,                //   threshold/total_years is never clobbered by defaults)
  // bulletin17c (B11): `family` is one of the six IsSupportedDistributionType families
  //   (Exponential | Gamma | LogNormal | LogPearsonTypeIII (default) | Normal | PearsonTypeIII);
  //   data via `dataset` or `data_frame` (censored series + low_outlier_threshold /
  //   mgbt_low_outliers all supported). Unlike the other types it is NOT a ModelBase (it is an
  //   IGMMModel + ISimulatable + IUnivariateModel), so the shared builder exposes it through a
  //   dedicated build_bulletin17c_from_json entry point returning the concrete type.
  // -- Phase 7a families (P3): --------------------------------------------------------------
  // time_series (AR/MA/ARIMA/ARIMAX): the series data comes from `dataset` (or an inline `data`
  //   array), wrapped into the P2 TimeSeries adapter (`time_interval` default OneDay,
  //   `start_index` default 0 -- the index is a sequence position / join key, never calendar
  //   arithmetic).
  "type": "time_series", "subtype": "arima",  // "ar" | "ma" | "arima" | "arimax"
  "orders": { "p": 1, "d": 0, "q": 0, "b": 0 },  // subtype-dependent (ar->p; ma->q; arima->p,d,q;
                                      //   arimax->p,d,q,b)
  "include_intercept": true,          // optional (default true)
  "transform": "None",                // optional TransformType: None | Logarithmic | BoxCox |
                                      //   YeoJohnson (transform_type.hpp)
  "trend": "Linear",                  // arimax only: None | Linear | Quadratic | Cubic
  "include_seasonality": false,       // arimax only, optional
  "covariates": [[1,2,3]],            // arimax only, optional: a list of covariate value arrays
  // spatial_gev: SpatialGEV(at_site_data [obs][sites], coordinates [sites][2], and three
  //   intercept-only level-2 GeneralLinearFunction trends). The gating flags are optional.
  "type": "spatial_gev",
  "coordinates": [[0.0, 0.0], [1.0, 0.5]],       // [sites][2]
  "at_site_data": [[20, 22], [25, 26]],          // [observations][sites]
  "use_copula_dependence": false,     // optional gating flags (all default false / log-links true):
  "use_location_errors": false, "use_scale_errors": false, "use_shape_errors": false,
  "use_log_link_for_location": true, "use_log_link_for_scale": true,
  // rating_curve: RatingCurve(stage TimeSeries, discharge TimeSeries, segments). Both arrays are
  //   wrapped with a shared start index so the date-inner-join aligns them 1:1 (the model
  //   enforces MinimumAlignedObservations = 10).
  "type": "rating_curve", "segments": 1,         // 1..3
  "stage": [1.0, 1.2, 1.5], "discharge": [5.0, 7.1, 11.0],
  // bivariate: a copula-coupled BivariateDistribution. The marginals are pre-fit IUnivariateModel
  //   models (each an inline `family` + `data` array + pinned `parameter_values`), held FIXED
  //   during the copula fit; only the copula dependency parameter is estimated.
  "type": "bivariate",
  "copula": "Normal",                 // a CopulaType name: AliMikhailHaq | Clayton | Frank |
                                      //   Gumbel | Joe | Normal | StudentT
  "estimation_method": "InferenceFromMargins",   // optional: InferenceFromMargins (default) |
                                      //   PseudoLikelihood | FullLikelihood
  "marginal_x": { "family": "Normal", "data": [ ... ], "parameter_values": [ ... ] },
  "marginal_y": { "family": "Normal", "data": [ ... ], "parameter_values": [ ... ] },
  "parameter_values": [16000.0, 5000.0]  // optional, ALL types: applied last via ONE
                                      // set_parameter_values call (the sync-safe setter)
}
```

**Phase 7a families (P3).** The four new `model.type` values above wire the remaining `ModelBase`
model families into the shared spec builder on the SAME polymorphic dispatch the Phase 4-6 models
use -- no bespoke per-model glue. Five of them (`ar`/`ma`/`arima`/`arimax`, `spatial_gev`,
`rating_curve`) are `ISimulatable<std::vector<double>>`, so their seeded draws ride the existing
`simulated_value` arm unchanged. `bivariate` is `ISimulatable<Matrix2D>` (an n-row x 2-col matrix,
col 0 = X, col 1 = Y), so its seeded draw is **flattened ROW-MAJOR** (`simulated_value [i]` reads
element `i = row*2 + col`: `[0]` = row 0 X, `[1]` = row 0 Y, `[2]` = row 1 X, ...). This flatten is
mirrored identically in all three harnesses (C++ `simulate_flat`, R `simulate_flat`, Python
`simulate_flat`).

`MaximumLikelihood`/`MaximumAPosteriori` cases MAY carry an optional `sample_size` (+ `seed`): after
the fit, the estimator's best parameters are pinned back into the model and one seeded
`generate_random_values` draw is cached, so a single MLE case can assert `parameter` +
`max_log_likelihood` + a seeded `simulated_value` together (the same seeded-draw arm the
`GeneralizedMethodOfMoments` target already uses off a fitted model). Without `sample_size` the
arm is inert (every pre-P3 fixture is unaffected). The single-parameter bivariate copula fit skips
the `covariance`/`standard_error`/`correlation` surface (the covariance stack needs >= 2
parameters); no fixture asserts those for a 1-param model.

Notes on the trend spec: attaching any trend first sets `is_nonstationary(true)`;
`set_trend_model` then supplies the C#-mirrored data-driven defaults per trend; the optional
`start_index` overrides the anchor afterwards; and explicit trend `values` (plus the model-level
`parameter_values`) are applied through ONE final `set_parameter_values` call after every trend is
attached (the model header mandates that setter -- poking `parameters()` elements directly would
desync the trend-owned copies). Uncertain items build their measurement-error distribution through
the same factory (`family` + `parameters`, via `set_parameters`). Threshold items carry
`number_above` only -- `number_below` is a derived value that `process_threshold_series` computes
at the model boundary, exactly like the C#. `point_process` exposes the non-seasonal surface only
(the seasonal path is a project-wide deferral) and takes no `family`: its mark distribution is the
model's own single-GEV competing-risks default. `competing_risks` is NOT an `IUnivariateModel`
(the C# omits it) but derives from `ModelBase` like the rest -- which is all the estimators
accept, so the harness caches simply hold a `ModelBase` pointer (C++: the `EstimationCase.model`
member; the `Simulation` target adds a `std::monostate` alternative to the estimator variant).

`construct` fields for `MaximumLikelihood`/`MaximumAPosteriori`: `model` (the schema above),
`optimizer` (`"NelderMead"` | `"DifferentialEvolution"` | `"Brent"` | `"BFGS"` | `"Powell"` |
`"MultilevelSingleLinkage"` (alias `"MLSL"`); defaults to `"DifferentialEvolution"` when omitted,
matching both estimator classes' C# default -- the last three were un-gated in B7 and exposed to
the fixture parser in B11). `construct` fields for `BayesianAnalysis` (Task T12): `model`
(same shape), `sampler` (`"DEMCz"` | `"DEMCzs"` | `"ARWMH"` | `"NUTS"`; defaults to `"DEMCzs"`,
matching the C# default), and an optional `settings` object with any of `seed`, `iterations`,
`warmup_iterations`, `number_of_chains`, `thinning_interval`, `initial_iterations`,
`output_length` -- each applied only if present (all four harnesses turn OFF
`UseSimulationDefaults`/`UseAdvancedSimulationDefaults` first so the explicit values below aren't
clobbered by the ctor's own defaulting, then apply exactly the settings supplied).

Assertion methods (0-based indices throughout except `bic`'s `n`, an actual sample size):

**`MaximumLikelihood`/`MaximumAPosteriori` (Task T11 + T12):**
- `parameter [p]` -- `BestParameterSet.Values[p]`.
- `max_log_likelihood []` -- `MaximumLogLikelihood` (data log-likelihood at the optimum for
  `MaximumLikelihood`; log-POSTERIOR, data + prior, at the optimum for `MaximumAPosteriori`).
- `aic []`, `bic [n]` -- `GetAIC()`/`GetBIC(n)`.
- `covariance [i,j]` -- `GetCovarianceMatrix()[i,j]`.
- `standard_error [p]` -- `GetStandardErrors()[p]`.
- `correlation [i,j]` -- `GetCorrelationMatrix()[i,j]` (T12).

These seven are shared verbatim by `MaximumLikelihood` and `MaximumAPosteriori` (identical method
names/signatures on both classes), dispatched through one `std::variant<unique_ptr<
MaximumLikelihood>, unique_ptr<MaximumAPosteriori>, unique_ptr<BayesianAnalysis>>` + a generic
lambda rather than duplicating the switch per class -- see `dispatch_estimation` in
`core/tests/test_fixtures.cpp`.

`bic`'s `n` is an explicit, arbitrary sample size, matching C# `GetBIC(sampleSize)` -- it is
**not** necessarily the fitted dataset's own length, and all three harnesses read it live from
each assertion's `args[0]` at dispatch time rather than caching a precomputed value alongside
`parameter`/`max_log_likelihood`/`aic`/`covariance`/`standard_error`/`correlation`. C++'s
`dispatch_estimation` calls `est->get_bic(a[0].get<int>())` directly; R's `bf_estimation_bic_` and
Python's `estimation_bic` each rebuild the model/estimator and call `get_bic(n)` live (safe because
`estimate()` is deterministic -- NelderMead/Brent have no randomness, and
DifferentialEvolution's default `prng_seed` is fixed -- so the rebuilt fit exactly reproduces the
one `estimation_run`/`bf_estimation_run_` already computed). A future fixture asserting `bic`
with `n != len(dataset)` must pass identically across all three harnesses.

**`BayesianAnalysis` (Task T12):**
- `dic []`, `waic []`, `looic []` -- `Dic()`/`Waic()`/`Looic()`.
- `posterior_mean [p]` -- `Results.PosteriorMean.Values[p]`.
- `chain_value [chain, iter, param]` -- `Sampler.MarkovChains[chain][iter].Values[param]`, the
  same seeded chain digest semantics `mcmc_sampler`'s own `chain_value` uses (see that section
  above) -- BayesianAnalysis internally builds and samples one of the 4 ported MCMC samplers, so
  the digest is bit-identical to a standalone `mcmc_sampler` fixture run with the same seed/model
  (confirmed: `bayes_normal.json`'s first-iteration chain values are the SAME literals as
  `sampling/mcmc/demczs.json`'s `normal_short_exact` case).

These four (plus `chain_value`) are `BayesianAnalysis`-only surface with no ML/MAP equivalent,
dispatched in a separate branch of the same `std::visit` (see `dispatch_estimation`) rather than
joining the ML/MAP method list, since the two surfaces are disjoint. R/Python each expose
`BayesianAnalysis` through a SEPARATE registered function (`bf_estimation_bayes_run_` / R,
`estimation_bayes_run` / Python) rather than folding it into `bf_estimation_run_`/`estimation_run`,
because its construct shape (`sampler` + numeric knobs) doesn't fit the ML/MAP
`{target, model_json, dataset, optimizer}` signature.

**`Simulation` (M13):** a fourth file-level `target` with NO estimator: the case builds the model
(any of the four types above), calls the ISimulatable surface
`generate_random_values(construct.sample_size, construct.seed)` **once** (`seed` optional,
defaulting to `-1` = clock-seeded -- seeded fixtures must always set it), caches the draw vector,
and dispatches assertions against it:

- `simulated_value [i]` -- the i-th draw of the cached seeded sample, following the Phase 3/4
  `short_exact`/`chain_value` digest precedent: the point is cross-language bit-identity of the
  seeded Mersenne Twister stream, so the same literals pass in C++, R, and Python.

R/Python expose it through a third registered function (`bf_model_simulate_` / R,
`model_simulate` / Python); C++'s `EstimationCase` holds `std::monostate` in place of an estimator
and the cached `simulated` vector. Simulation draws come from the model's CURRENT parameters
(there is no fit), so seeded cases should pin them explicitly via `parameter_values` (or trend
`values`) -- otherwise the draw reflects whatever the model's construction defaults left behind
(EM defaults for `mixture` from `dataset`, factory defaults for the component distributions of
`competing_risks`/`point_process`). (The `Simulation` target builds a ModelBase; the
`bulletin17c` model type is NOT a ModelBase, so its seeded draw rides the GMM case below instead
of a standalone `Simulation` case.)

**`GeneralizedMethodOfMoments` (B11):** a file-level `target` that fits a `bulletin17c` model by
GMM. `construct` fields: `model` (a `type: "bulletin17c"` spec), `strategy` (`"OneStep"` |
`"TwoStep"` | `"Iterative"`; default `"Iterative"`, the C# GMM default), `optimizer` (the same
shared parser as ML/MAP; the GMM ctor's own default is `"BFGS"`), an optional
`max_gmm_iterations` (int), and -- to ride the seeded-draw digest on this case -- an optional
`sample_size` (+ `seed`). One `estimate()` + `post_process(sandwich=true, jstat=true)` runs per
case, caching the full surface. Assertion methods:
- `parameter [p]` -- `BestParameterSet().Values[p]`.
- `standard_error [p]` -- `GetStandardErrors()[p]`.
- `covariance [i,j]` / `correlation [i,j]` -- `GetCovarianceMatrix()[i,j]` /
  `GetCorrelationMatrix()[i,j]` (same names/semantics as ML/MAP, so the dispatcher reuses those
  arms).
- `j_stat []` -- `Jstat()` (Hansen's J; ~0 for a just-identified fit where `g(theta-hat) ~ 0`).
- `j_stat_pval []` -- `JstatPval()` (NaN when the degrees of freedom `q - p == 0`, i.e. a
  just-identified fit cannot test its own specification).
- `quantile_variance [aep]` -- the B17C delta-method `Var(Q_p) = g' Sigma g` evaluated at the
  fitted parameters and the fitted covariance. `args[0]` is the annual EXCEEDANCE probability
  (AEP); the C# `QuantileVariance` takes a NON-exceedance probability, so the harness passes
  `1 - aep`. Like `bic [n]`, the AEP is per-assertion, so R/Python rebuild the deterministic fit
  live (`bf_estimation_gmm_qvar_` / `estimation_gmm_qvar`).
- `simulated_value [i]` -- the shared seeded-draw arm: after the fit, the estimator's best
  parameters are pinned into the B17C parent and a seeded `generate_random_values(sample_size,
  seed)` stream is drawn (the same `short_exact` digest semantics as `Simulation`).

The GMM surface is disjoint enough from ML/MAP (adds `j_stat`/`j_stat_pval`/`quantile_variance`,
drops `max_log_likelihood`/`aic`/`bic`) that it joins the estimator `std::variant` on its own
`std::visit` arm, and R/Python expose it through a SEPARATE registered function
(`bf_estimation_gmm_run_` / `estimation_gmm_run`) -- the same split pattern `BayesianAnalysis`
uses, since its construct shape (`strategy` + `max_gmm_iterations`, and a `bulletin17c` model)
doesn't fit the ML/MAP signature.

**DataFrame surface (M14):** three methods reachable from the model's DataFrame under ANY
file-level `target` (they read the model, not the estimator), corroborating the M1/M5
plotting-position and MGBT ctest oracles through the PUBLIC model path:

- `plotting_position [kind, i]` -- item `i`'s plotting position in the named series (`kind` is
  `"exact"` | `"interval"` | `"uncertain"`, indices in spec order) after ONE
  `calculate_plotting_positions()` pass (idempotent -- a pure function of the collections plus the
  frame's plotting parameter). The threshold series is deliberately NOT exposed: the C# assigns
  its positions to a sorted CLONE, so the original items never carry one.
- `number_of_low_outliers []`, `low_outlier_threshold []` -- the frame's current state, as set by
  the spec's `mgbt_low_outliers` MGBT trigger or the explicit `low_outlier_threshold`.

C++/the emitter dispatch these straight off the cached model's frame; R
(`bf_model_data_frame_`) and Python (`model_data_frame`) lazily build the frame surface once per
case through the shared spec builder and memoize it (a rebuild is byte-identical -- the frame
surface is a pure function of the construct, never of the fit -- the `bic` lazy-rebuild
precedent).

**Port-fidelity finding (Task T12, real-C#-oracle-driven): the Jeffreys 1/scale prior.** The T11
port's `UnivariateDistributionModel` inherited `ModelBase`'s generic `prior_log_likelihood` (sum of
per-parameter Uniform priors only), which made `MaximumAPosteriori`'s posterior mode coincide with
the plain MLE under the model's default (flat Uniform) parameter priors. Dumping exact oracle
values from the REAL C# `MaximumAPosteriori`/`BayesianAnalysis` surfaced a real divergence: the C#
`UnivariateDistribution.Prior_LogLikelihood` (and every other C# univariate model) ALSO applies a
Jeffreys `1/scale` prior on the scale parameter whenever `UseJeffreysRuleForScale` is set (`true`
by default in `UnivariateDistributionModelBase`), which measurably pulls the posterior-mode scale
below the pure MLE (~4.7% lower for `map_normal.json`'s Normal/NelderMead case). This was a genuine
port gap, not a fixture-tolerance problem: `UnivariateDistributionModel::prior_log_likelihood` (and
its `pointwise_prior_log_likelihood` counterpart) were extended in T12 to apply the same `-log(scale)`
term (scale parameter index: 0 for Gamma/Weibull, 1 for every other family, matching the C# source),
gated by a new `use_jeffreys_rule_for_scale()`/`set_use_jeffreys_rule_for_scale()` toggle (defaults
`true`). Quantile priors (`EnableQuantilePriors`, C# default `false`) are NOT applied -- that surface
remains out of scope per this model's existing T6 header comment. With this fix, `map_normal.json`
and `bayes_normal.json` reproduce the real C# MAP/Bayesian posteriors exactly (see their `source`
fields for the measured tolerances).

**Tolerance policy (Task T12, exact emitter-dumped oracles):**
- `mle_normal_smoke.json` (renamed target retained -- still a fine name once tightened):
  `parameter`/`max_log_likelihood`/`aic`/`bic` at `mode: "rel", tol: 1e-9` (NelderMead is a
  deterministic simplex -- no RNG, no parallel reduction -- so these reproduce tightly across the
  4 independently-built cores); diagonal `covariance`/`standard_error`/`correlation` at `tol: 1e-5`
  (the numerical-Hessian finite-difference step is one order more rounding-sensitive); the
  off-diagonal `covariance`/`correlation` entries are theoretically zero for a Normal MLE (mu and
  sigma are orthogonal) and are checked with an ABSOLUTE tolerance instead of relative (relative
  against a near-zero expected value is meaningless).
- `map_normal.json`: same tolerance tiers as above, applied to the MAP (posterior-mode) values,
  which now differ measurably from the MLE because of the Jeffreys-prior fix above.
- `bayes_normal.json`: `chain_value` at `tol: 1e-11` (DEMCzs is pure-RNG, no gradients -- the
  Phase-3 digest precedent); `posterior_mean` at `tol: 1e-9`; `waic`/`looic` at `tol: 1e-8`; `dic`
  at `tol: 1e-6` (DIC's C# `Parallel.For` reduction is not bit-reproducible even C#-to-C#, measured
  ~1e-13 absolute run-to-run -- the same reduction-order divergence class as the bootstrap BCa
  fixtures above, NOT faked). This case deliberately sets a small `output_length` (400, vs. the
  `iterations`-driven default) so the `Output`-derived quantities (`posterior_mean`/`dic`/`waic`/
  `looic`) stay inside the population-sampler's pre-amplification window (see the DEMCz/DEMCzs
  population-sampler divergence finding earlier in this file) -- confirmed empirically: at this
  `output_length` the aggregates reproduce to ~1e-11..1e-14, far tighter than the long-run rstan
  cases' `tol: 0.05`, because the divergent tail never gets long enough to amplify sub-ULP noise
  into a measurable difference.

**Port-fidelity finding (Task M14, real-C#-oracle-driven): the Mixture `SetParameters(ref)`
write-back.** Dumping exact oracles for the M13 mixture smoke cases surfaced a real trajectory
divergence: the C# `MixtureModel.DataLogLikelihood`/`PriorLogLikelihood` call
`Mixture.SetParameters(ref parameters)`, which normalizes the weight entries IN THE CALLER'S
ARRAY -- and since C# objectives are `Func<double[], double>` over reference-type arrays, that
write-back lands in the optimizer's (and NumericalDiff's) own working vectors, re-projecting every
evaluated point onto the normalized-weights manifold and steering the whole search path. The M13
port normalized a private copy instead, converging to a scale-equivalent optimum with different
raw weight coordinates (w = [1.0, 0.333] vs the C# [0.75, 0.25]; same likelihood). M14 ported the
mutation semantics: the `Optimizer`/`NelderMead` `Objective` and `NumericalDiff`'s scalar function
type became `std::function<double(std::vector<double>&)>` (non-const; const-taking lambdas still
convert, so no other caller changed behavior), `ModelBase::log_likelihood`/`data_log_likelihood`/
`prior_log_likelihood` take `std::vector<double>&` (the pointwise variants keep const -- the C#
pointwise methods normalize a private `parmsCopy`), and `MixtureModel` now writes back. With the
fix the C++ mixture fit reproduces the real C# values to rel 1e-12. ONE documented deviation: the
MCMC sampler's `LogLikelihood` type stays const-ref (Phase 3, oracle-locked), so
`BayesianAnalysis` hands the model a mutable COPY -- a mixture's write-back into the sampler's
proposal/chain arrays is not ported (no fixture runs a mixture through BayesianAnalysis; every
other model is non-mutating, so the copy is behaviorally identical for all wired paths).

**M14 exact fixtures (naming + tolerance policy):** M14 re-pinned every M13 `*_smoke.json` against
the real RMC.BestFit via the emitter dump path and dropped the suffix:
`mle_censored.json` (all four data-array types + the M14 DataFrame surface incl. an MGBT case),
`map_censored.json` + `bayes_censored.json` (the same censored frame through MaximumAPosteriori
and seeded DEMCzs BayesianAnalysis), `mle_nonstationary.json`, `mixture.json`,
`competing_risks.json` (re-derived dataset -- the M13 smoke data pinned the Weibull shape at its
100 bound; the new 25-point max(Gumbel, Weibull) series fits fully interior, shape ~13.0),
`point_process.json`, and `simulation.json`. Every value is either emitter-dumped from the real
C# or byte-verified against it by `verify_oracles.py` (each file's `source` says which).
Tolerances follow the T12 policy: `rel 1e-9` for the deterministic NelderMead point values
(`1e-8` for the flatter GEV point-process surface), `rel 1e-11` for seeded `simulated_value`/
`chain_value` digests, `1e-9`/`1e-8`/`1e-6` for posterior_mean/waic+looic/dic (bayes_normal
precedent), and `rel 5e-3` for the censored/competing-risks covariance/SE/correlation -- a
DOCUMENTED looser tier than the Phase 4 `1e-5` Hessian band: for O(1e4)-scale parameters
NumericalDiff clamps the finite-difference step to `MaxStep = 1e-2`, putting the Hessian second
difference (~`H*h^2` ~ 1e-10) within ~3e-4 of the `|logL|` ~ 1e2 double-rounding floor (measured
C++-vs-C# divergence 2e-4..1.1e-3 -- pure cancellation noise; the underlying fits match to 1e-12
or better). Mixture covariance/standard errors remain deliberately NOT asserted: the REAL C#
covariance at that optimum is INDEFINITE (negative diagonals; `GetStandardErrors` clamps to 0)
with off-diagonals at finite-difference noise scale (~1e-10) -- the M10 near-singular-Hessian
finding, now confirmed against the real library. The frozen T12 files (`mle_normal_smoke.json`,
`map_normal.json`, `bayes_normal.json`) are byte-identical to their Phase 4 state.

**NEVER loosen a tolerance below what's documented above.** If a curated value fails to reproduce,
the streams have desynced somewhere -- diagnose the draw path (`--dump` intermediates, compare
against a standalone throwaway harness against the real C# library) and fix the transcription slip;
do not paper over it with a looser tolerance.

### `analysis`

The Phase 8 user-facing Analyses layer (A10): `UnivariateAnalysis` (Bayesian frequency curve),
`FittingAnalysis` (multi-distribution GoF ranking), and `Bulletin17CAnalysis` (B17C flood-frequency
+ Cohn-style CIs). Stateful like `model_estimation`: one build+run per case caches a flat result
surface, then every assertion dispatches against it. The `construct` fields map 1:1 onto the R/Python
glue arguments (`bf_analysis_*_` / `_core.analysis_*`), so all three harnesses build byte-identical
analyses from the same spec.

```jsonc
{
  "target":  "UnivariateAnalysis",           // "UnivariateAnalysis" | "FittingAnalysis" |
                                             // "Bulletin17CAnalysis"
  "kind":    "analysis",
  "source":  "RMC-BestFit/.../UnivariateAnalysis.cs @ fc28c0c",
  "datasets": { "peaks": [ ... ] },
  "cases": [
    {
      "name": "normal_demczs_short",
      "construct": {
        // FittingAnalysis: only `dataset` (a datasets key -> exact-only frame).
        "dataset": "peaks",
        // UnivariateAnalysis / Bulletin17CAnalysis: a `model` spec built by the SHARED spec
        // builder (bestfit/models/model_spec.hpp) -- the same schema model_estimation uses.
        "model": { "family": "Normal", "dataset": "peaks" },        // univariate_distribution
        // "model": { "type": "bulletin17c", "family": "LogPearsonTypeIII", "dataset": "peaks" },
        // -- UnivariateAnalysis MCMC knobs (all optional; forwarded to the held BayesianAnalysis):
        "sampler": "DEMCzs",                 // DEMCz | DEMCzs | ARWMH | NUTS
        "iterations": 100,                   // post-warmup iterations (warmup = max(50, it/2))
        "output_length": 400,                // posterior samples used for the credible band
        "credible_level": 0.90,              // credible-interval width
        "seed": 12345,                       // sampler PRNG seed
        "thinning_interval": 1,              // (A11) explicit thinning; absent -> sampler default
                                             //   (20 for a 2-param DEMCzs). The default thin=20
                                             //   path diverges C#-vs-C++ (docs/upstream-csharp-
                                             //   issues.md); pin 1 for an exact reproducible oracle.
        // "number_of_chains", "initial_iterations": also honored (integers; absent -> default)
        // -- Bulletin17CAnalysis knobs (all optional):
        "uncertainty_method": "MultivariateNormal",  // MultivariateNormal (default) | Bootstrap
                                             //   (LinkedMultivariateNormal / BiasCorrectedBootstrap
                                             //   are deferred to Phase 9 and rejected)
        "confidence_level": 0.90,            // Cohn-CI confidence level (== credible width)
        // -- shared, optional:
        "exceedance_probabilities": [0.01, 0.1, 0.5, 0.9, 0.99]  // strictly-increasing grid;
                                             //   absent -> the 25 standard default ordinates
      },
      "assertions": [
        // UnivariateAnalysis: parameter [i], mode_curve [i], mean_curve [i], lower_ci [i],
        //   upper_ci [i] (indexed by the exceedance grid), and scalars aic / bic / dic / rmse.
        { "method": "parameter", "args": [0], "expected": 16775.69498994981, "mode": "rel", "tol": 1e-9 },
        // FittingAnalysis: candidate_count (exact 14), candidate_aic / candidate_bic /
        //   candidate_rmse / candidate_converged [candidate_index].
        // Bulletin17CAnalysis: exceedance_probability [i], point_estimate [i] (log10 space),
        //   lower_ci [i], upper_ci [i] (discharge space), beta1 [i], nu [i],
        //   quantile_variance [i], confidence_level, parameter [i] (fitted GMM params).
      ]
    }
  ]
}
```

**Oracle strategy (A11 EXACT).** A11 unlocked the dotnet emitter's Analyses closure (a local
patched `Bulletin17CAnalysis.cs` clears the CS0104 `YeoJohnsonLink` ambiguity; see
`docs/upstream-csharp-issues.md`) and tightened all three smoke files to EXACT oracles verified
against the real C# analyses. Deterministic point values (GMM point estimate + parameters, mode
curve = InverseCDF, candidate aic/bic) use `rel 1e-9`; GoF/quadrature-derived quantities (candidate
rmse, Cohn CI bounds, the credible band) use `rel 1e-8`; DIC keeps `rel 1e-6` (parallel-reduction
noise). `candidate_count == 14`, `confidence_level`, and the exceedance-probability round-trip stay
exact structural invariants (`abs 1e-12..1e-15`, `equal`). The UnivariateAnalysis case PINS
`thinning_interval = 1`: the sampler-default thin=20 exposes a real C#-vs-C++ divergence in the
thinned population-sampler stream (documented in `docs/upstream-csharp-issues.md` as a tracked
follow-up); thin=1 is the `bayes_normal`-proven bit-identical path.

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

The `Correlation.*` targets (`fixtures/special_functions/correlation.json`) take two equal-length samples
flattened into `args` as `[x_1..x_n, y_1..y_n]`, split at the midpoint (`n = len(args) / 2`).

The `LU.*` targets (`fixtures/special_functions/lu_decomposition.json`) reuse the exact Cholesky
matrix-args convention above (same flattened-row-major-plus-trailing-indices shapes for
`determinant`/`inverse_element`/`solve_element`), but -- unlike Cholesky, which requires a
symmetric positive-definite matrix -- accept any square matrix, so its cases are deliberately
non-symmetric to exercise LU's row-pivoting path.

`Statistics.percentile` (`fixtures/special_functions/percentile.json`) takes
`args = [data_1..data_n, k, data_is_sorted (0.0/1.0)]`, with `n = len(args) - 2`.

The `Extensions.*`/`Mt.*` targets (`fixtures/special_functions/extension_methods.json`) exercise
seeded-draw entry points with no upstream C# test literal to scrape, so every case is curated via
`oracle_emitter --dump` (see Statefulness note below): `Extensions.next_doubles_grid` takes
`args = [n, dim, seed, row, col]` (element `[row, col]` of `MersenneTwister(seed).NextDoubles(n,
dim)`); `Extensions.next_integers_at` takes `args = [n, seed, i]`; `Mt.next_range` takes `args =
[seed, min, max, i]` (the `i`-th, 0-based, call to `MersenneTwister(seed).Next(min, max)`).

The `RunningCovariance.*` targets (`fixtures/special_functions/running_covariance.json`) take
`args = [size, num_pushes, data_flat(num_pushes*size), trailing index/indices]` -- one row index
for `mean_element`, `(i, j)` for `covariance_element`/`sample_covariance_element`/
`sample_correlation_element`/`population_covariance_element`/`population_correlation_element`. The
`RunningStatistics.*` targets (`fixtures/special_functions/running_statistics.json`) take `args` =
the flat sample (the constructor pushes every element in order); each target reads one property off
the result. The `RunningStatistics.combined_*` targets exercise `RunningStatistics::combine()`/
`operator+` (which the plain per-property targets never touch, since they only ever build a single
instance via the list constructor): `args = [n1, sample1(n1 values), sample2(remaining values)]`, a
"split-index" convention distinct from `Correlation.*`'s equal-length two-halves split -- the
upstream `Test_Combine`/`Test_Add` literals split their 69-value sample into UNEQUAL 48/21
sub-samples, so a fixed midpoint doesn't apply. Each target builds `RunningStatistics(sample1) +
RunningStatistics(sample2)` and reads one property off the merged result.

The `Fourier.*` targets (`fixtures/special_functions/fourier.json`) take: `fft_at`/`real_fft_at`
`args = [data..., inverse (0/1), index]` (n = len(args) - 2; runs `FFT`/`RealFFT` in place on a
copy of `data`, returns `data[index]`); `correlation_at` `args = [data1..., data2..., index]`
(equal-length `data1`/`data2` concatenated, n = (len(args) - 1) / 2); `autocorrelation_at`
`args = [series..., lag_max, lag]` (n = len(args) - 2; `lag_max == -1` triggers the default
auto-lag `floor(min(10*log10(n), n-1))`, returns the acf value at row `lag`). The FFT/RealFFT and
200-point-series autocorrelation cases are scraped verbatim from `Test_FastFourierTransform.cs`;
`correlation_at` and the explicit-`lag_max` autocorrelation cases have no upstream literal and are
curated via `oracle_emitter --dump`.

The `NumericalDerivative.*` targets (`fixtures/special_functions/numerical_derivative.json`)
dispatch through a **closed registry of two named functions**, implemented identically in
`core/tests/test_fixtures.cpp` (`numerical_derivative_quadratic`/`numerical_derivative_normal_loglik`)
and `tools/oracle_emitter/Program.cs` (`NumericalDerivativeQuadratic`/`NumericalDerivativeNormalLoglik`)
so the emitter runs the REAL C# `NumericalDerivative` against the same functions the C++ runner
does: `"quadratic"` is `f(x) = sum_i (x_i - i)^2` (0-based `i`); `"normal_loglik"` is
`Normal(mu=x0, sigma=x1).LogLikelihood(sample)` on an embedded 5-point sample `{9,10,11,12,13}` --
the exact 2-parameter log-likelihood shape MCMC's default HMC/NUTS gradient differentiates. Target
names encode the function: `gradient_element_quadratic`, `hessian_element_normal_loglik`, etc.
`args` for `gradient_element_<fn>` = `[p, theta(p values), lower(p values), upper(p values),
index]`; for `hessian_element_<fn>` = `[p, theta(p values), lower(p values), upper(p values), i,
j]`. `lower`/`upper` always carry a full `p`-length bound array, using the `"-inf"`/`"inf"` string
literals (see Special values below) for unbounded dimensions rather than omitting the array --
`AvailableLeft`/`AvailableRight` treat an explicit `+-infinity` bound identically to a null one, so
this is a behavior-preserving flattening of the C#'s nullable-array API onto a flat numeric `args`
convention. `rel_step`/`abs_step`/`max_backtrack` always use the library defaults, matching every
real call site (`HMC.cs`'s `Gradient` call and `Optimizer.cs`'s `Hessian` call both omit them).
Every case is curated via `oracle_emitter --dump` (no upstream `Test_NumericalDerivative.cs`
exists); the `_near_bound` cases put a bound within one default step of `theta`, forcing the
one-sided (forward/backward, or one-sided-mixed-partial for Hessian) branch instead of central; the
`_pinned` gradient case sets `lower == upper == theta` (zero room on either side), exercising the
"every backtrack attempt fails -> derivative set to 0" fallback.

The `Histogram.*` targets (`fixtures/special_functions/histogram.json`) take
`args = [explicit_bins, data...]` for the whole-histogram scalar targets (`number_of_bins`,
`bin_width`, `lower_bound`, `upper_bound`, `data_count`, `mean`, `median`, `mode`,
`standard_deviation`; `explicit_bins == 0` selects the Rice-Rule ctor, `explicit_bins > 0` selects
the explicit-bin-count ctor); the `bin_lower_bound_at`/`bin_upper_bound_at`/`bin_frequency_at`/
`get_bin_index_of` element-lookup targets append one trailing probe value (a 0-based bin index, or
an x value to look up) after `data`. The 69-value sample and its Rice-rule/explicit-20-bin
lower/upper/frequency arrays are scraped from `Test_Histogram.cs`; `mean`/`median`/`mode`/
`standard_deviation` were independently recomputed in Python from that test's own closed-form
formulas (1e-6 tolerance, matching); `get_bin_index_of` cases port `Test_Indexing`'s invariant
(`GetBinIndexOf(bin[i].Midpoint) == i`) with the midpoint recomputed from the identical
bin-construction arithmetic.

The `PlottingPositions.*` targets (`fixtures/special_functions/plotting_positions.json`) take
`weibull_at` `args = [N, i]` (0-based `i`, analytic `PP[i] = (i+1)/(N+1)`) and `function_at`
`args = [N, alpha, i]` (analytic `PP[i] = (i+1-alpha)/(N+1-2*alpha)`); both scraped from
`Test_PlottingPositions.cs` (`N=30`, `alpha=0.1`).

The `Search.*` targets (`fixtures/special_functions/search.json`) take
`args = [values..., x, start]` (n = len(args) - 2), the default `SortOrder.Ascending` only -- the
only order any real caller uses (SNIS's `Search.Sequential` call and Histogram's
`Search.Bisection` call both omit the `order` argument). No upstream `Test_Search.cs` exists, so
every case is curated via `oracle_emitter --dump`; probes cover both boundary sentinels
(below/above the range, exact first/last-element match) and interior lookups, including one with a
non-zero `start`.

The `MCMCDiagnostics.minimum_sample_size` target (`fixtures/special_functions/mcmc_diagnostics.json`)
takes `args = [quantile, tolerance, probability]` (the Raftery-Lewis normal-approximation
minimum-sample-size heuristic, rounded to the nearest 100). No upstream `Test_MCMCDiagnostics.cs`
case exercises this method (that test file covers only `GelmanRubin`, transcribed as C++-only
regression tests in `core/tests/test_mcmc_extra.cpp` since its assertions are inequalities over
seeded synthetic chains rather than fixture-shaped literals), so every case here is curated via
`oracle_emitter --dump`.

**Comparison modes:** `abs` (|actual−expected| ≤ tol), `rel` (|actual−expected|/|expected| ≤ tol),
`equal` (exact; `expected` may be the strings `"inf"`, `"-inf"`, `"nan"`), `bool`.

**Special values:** in `params` and `equal`-mode `expected`, the strings `"nan"`,
`"inf"`, `"-inf"` denote the corresponding floating-point values (JSON has no literals
for them).

Each runner maps fixture `method` names to its own API (e.g. `"sd"` →
`standard_deviation()`), so a new distribution usually needs only a new fixture file
plus a few dispatch entries per language.
