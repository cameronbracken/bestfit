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
  "target":  "MaximumLikelihood",           // "MaximumLikelihood" | "MaximumAPosteriori" today
                                             // (Task T11); "BayesianAnalysis" is schema-documented
                                             // but not yet constructible -- see below
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
        { "method": "parameter", "args": [0], "expected": 16026.055013737448, "mode": "rel", "tol": 1e-3 }
      ]
    }
  ]
}
```

One `estimate()` run per case (mirrors `mcmc_sampler`'s/`bootstrap`'s single-stateful-glue-call
contract): build a `UnivariateDistributionModel` from `construct.model.{family, dataset}` --
`family` goes through the same univariate distribution factory every other kind uses
(`create_distribution(name)`), `dataset` names an entry in the file-level `datasets` map -- then
construct the estimator named by the file-level `target`, call `estimate()` **once**, then dispatch
every assertion in the case against the cached `(model, estimator)` pair. Unlike `mcmc_sampler`'s
and `bootstrap`'s model registries (each a small closed name -> construction-recipe map with no
upstream C# library counterpart), `model_estimation`'s "registry" is the one-line factory call
above: every family the factory supports, plus `UnivariateDistributionModel::set_default_parameters`
(itself driven by `IMaximumLikelihoodEstimation::get_parameter_constraints`), is already enough to
build a model -- there is no separate closed-name mapping to maintain.

`construct` fields: `model.family` (a univariate distribution type name), `model.dataset` (a
`datasets` key), `optimizer` (`"NelderMead"` | `"DifferentialEvolution"` | `"Brent"`, applies to
`MaximumLikelihood`/`MaximumAPosteriori`; defaults to `"DifferentialEvolution"` when omitted,
matching both estimator classes' C# default). `BayesianAnalysis`'s documented construct fields
(`sampler`, `seed`, `iterations`, ...) are schema-reserved for Task T12 -- seeing them in a fixture
today would be premature, since `target: "BayesianAnalysis"` is not yet constructible (see below).

Assertion methods (0-based indices throughout except `bic`'s `n`, an actual sample size):

**Wired (Task T11):**
- `parameter [p]` -- `BestParameterSet.Values[p]`.
- `max_log_likelihood []` -- `MaximumLogLikelihood` (data log-likelihood at the optimum for
  `MaximumLikelihood`; log-POSTERIOR, data + prior, at the optimum for `MaximumAPosteriori`).
- `aic []`, `bic [n]` -- `GetAIC()`/`GetBIC(n)`.
- `covariance [i,j]` -- `GetCovarianceMatrix()[i,j]`.
- `standard_error [p]` -- `GetStandardErrors()[p]`.

These six are shared verbatim by `MaximumLikelihood` and `MaximumAPosteriori` (identical method
names/signatures on both classes), dispatched through one `std::variant<unique_ptr<
MaximumLikelihood>, unique_ptr<MaximumAPosteriori>>` + a generic lambda rather than duplicating the
switch per class -- see `dispatch_estimation` in `core/tests/test_fixtures.cpp`.

`bic`'s `n` is an explicit, arbitrary sample size, matching C# `GetBIC(sampleSize)` -- it is
**not** necessarily the fitted dataset's own length, and all three harnesses read it live from
each assertion's `args[0]` at dispatch time rather than caching a precomputed value alongside
`parameter`/`max_log_likelihood`/`aic`/`covariance`/`standard_error`. C++'s `dispatch_estimation`
calls `est->get_bic(a[0].get<int>())` directly; R's `bf_estimation_bic_` and Python's
`estimation_bic` each rebuild the model/estimator and call `get_bic(n)` live (safe because
`estimate()` is deterministic -- NelderMead/Brent have no randomness, and
DifferentialEvolution's default `prng_seed` is fixed -- so the rebuilt fit exactly reproduces the
one `estimation_run`/`bf_estimation_run_` already computed). A future fixture asserting `bic`
with `n != len(dataset)` must pass identically across all three harnesses.

**Left for Task T12 (structure only -- no live path yet):**
- `correlation [i,j]` -- `GetCorrelationMatrix()[i,j]` (exists on both wired classes today, but
  deliberately not dispatched yet -- adding it is a one-line `if` arm).
- `dic []`, `waic []`, `looic []`, `posterior_mean [p]` -- `BayesianAnalysis`-only surface
  (`Dic()`/`Waic()`/`Looic()`/`PointEstimate().Values[p]`); each needs a `BayesianAnalysis`
  construction arm in `build_and_run_estimation` (today a clear "deferred to Task T12" throw) in
  addition to its own dispatch `if` arm, since `MaximumLikelihood`/`MaximumAPosteriori` have no
  equivalent.

**Tolerance policy for the T11 smoke fixture** (`estimation/mle_normal_smoke.json`): this is a
PLUMBING PROOF, not the T12 emitter-dumped oracle -- every assertion anchors to the SAME value
`core/tests/test_maximum_likelihood.cpp`'s `test_loose_oracle_anchor` already spike-proved against
the real C# library (`mu ~ 16026.055`, `sigma ~ 5058.828`, `maximum_log_likelihood ~ -99.4793`),
plus `aic`/`bic` computed directly from that anchor and `covariance`/`standard_error` values read
straight off this port's own (deterministic, seed-free) NelderMead run. `parameter`/
`max_log_likelihood`/`aic`/`bic` use `mode: "rel", tol: 1e-3` (matching the ctest's own tolerance
for the shared anchor); `covariance`/`standard_error` use `mode: "rel", tol: 1e-2` (one order of
magnitude looser again -- the numerical Hessian behind them is a finite-difference computation one
step further removed from the optimizer's own convergence tolerance, so it is more sensitive to
platform/compiler rounding across the three harnesses' independently-built copies of the same
core). Task T12 replaces/augments this fixture with exact emitter-dumped oracles and a tighter,
measured tolerance policy, the same way T12-era work tightened the MCMC/bootstrap fixtures above.

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
