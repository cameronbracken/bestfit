# Upstream sync T0: oracle failure census

Date: 2026-07-19. Submodules bumped to Numerics `v2.1.4` (`2a0357a`) and RMC-BestFit `v2.0.0`
(`c2e6192`); emitter repaired; gate run against the **unmodified** `fixtures/*.json` corpus (no
fixture edited in this task). This is the empirical enumeration of every oracle the new C#
moved, to be consulted by every later porting task (T1-T22) instead of re-discovering the same
list. See `docs/superpowers/specs/2026-07-19-upstream-sync-numerics-2.1.4-bestfit-2.0.0-design.md`
for the classified risk lists this census is cross-checked against.

## Emitter repair

The stale CS0104 workaround (`tools/oracle_emitter/patched/Bulletin17CAnalysis.cs` + its two
`OracleEmitter.csproj` compile-remove/include lines) is gone: upstream deleted
`RMC.BestFit/Models/LinkFunctions/YeoJohnsonLink.cs` (BestFit `v2.0.0`), which removed the
ambiguity with `Numerics.Functions.YeoJohnsonLink` the patch worked around. The real
`Bulletin17CAnalysis.cs` now compiles in place.

Two genuinely new compile breaks surfaced from the v2 restructure (neither predicted by name in
the brief, both are exactly the class of break the brief anticipated generically -- "new using
targets from the v2 restructure"):

1. **DataFrame series/data types moved namespace.** `ExactSeries`/`ExactData`/`IntervalSeries`/
   `IntervalData`/`ThresholdSeries`/`ThresholdData`/`UncertainSeries`/`UncertainData` moved from
   the bare `namespace RMC.BestFit` to `namespace RMC.BestFit.Models` (confirmed via
   `git diff fc28c0c c2e6192 -- src/RMC.BestFit/Models/DataFrame/` in the submodule: every
   `namespace RMC.BestFit` in that subtree became `namespace RMC.BestFit.Models`). The emitter's
   `using RMC.BestFit;` no longer resolved the unqualified series/data constructors in
   `BuildSpecDataFrame`. Fix: replace `using RMC.BestFit;` with `using RMC.BestFit.Models;`
   (`tools/oracle_emitter/Program.cs`); the existing `BestFitModels = RMC.BestFit.Models` alias is
   kept for the call sites already qualified with it.
2. **New `Transform` ambiguity.** Adding the plain `using RMC.BestFit.Models;` above collides
   `RMC.BestFit.Models.Transform` (the TimeSeries None/Logarithmic/BoxCox/YeoJohnson transform
   enum, already present pre-bump) against `Numerics.Data.Transform` (None/Logarithmic/NormalZ,
   used for `EmpiricalDistribution.ProbabilityTransform` and `BivariateEmpirical` margins) --
   effectively a second CS0104, this time self-inflicted by fix #1 rather than upstream. Fixed
   minimally by fully qualifying the 7 unqualified `Transform.*` call sites in
   `BuildDistribution`/`BuildDistribution` (BivariateEmpirical branch) as `Numerics.Data.Transform`
   rather than adding another alias; the (already-qualified) `BestFitModels.Transform.*` call
   sites for TimeSeries fixtures were untouched.

No other compile breaks. `AnalysisProgress` (the new file the brief flagged to verify) is indeed
already covered by the existing `Analyses/Support/**` glob -- confirmed no action needed. Build
is clean (`dotnet build tools/oracle_emitter -c Release`: 0 errors, the same 2 pre-existing
nullable warnings at the same call site, now at a shifted line number). No genuinely new CS error
needing a local-copy patch (the brief's fallback plan) was needed.

## Gate summary

```
oracle verification: 4099 reproduced, 10 failed, 11 skipped (GEV std-err + oracle-exempt)
```

Total fixture-assertion count is unchanged from the pre-sync baseline (4099+10 = 4109 reproduced
pre-bump, 11 skipped both times) -- confirming no fixture gained or lost an assertion; exactly 10
previously-reproducing values flipped to failing under the new C#. Exit code 1 (expected). Full
raw output archived at `/tmp/oracle-census-raw.txt` (not committed; ephemeral).

Full failure list (10 lines, all 10 shown -- nothing elided):

```
FAIL MaximumLikelihood/normal_censored_data_frame/plotting_position: expected 0.6956162117452439 got 0.69892473118279574
FAIL MaximumLikelihood/normal_censored_data_frame/plotting_position: expected 0.005376344086021506 got 0.014336917562724039
FAIL MaximumLikelihood/normal_censored_data_frame/plotting_position: expected 0.543424317617866 got 0.54838709677419351
FAIL MaximumLikelihood/normal_censored_data_frame/plotting_position: expected 0.391232423490488 got 0.39784946236559138
FAIL RWMH/normal_rstan/map_fitness: expected 473.55840961774777 got -473.55840961774777
FAIL GeneralizedBeta/alpha0p42_beta1p57_unit/mode: expected 57.999999999999957 got 0
FAIL StudentT/student_t_shifted_mu2p5_sigma0p5_nu4/pdf: expected 0.0516476521260042 got 0.10329530425200877
FAIL StudentT/student_t_large_df_sigma2_normal_approx/pdf: expected 0.3989422804014327 got 0.19947114020071632
FAIL StudentT/student_t_large_df_sigma2_normal_approx/pdf: expected 0.3520653267642995 got 0.17603266338214973
FAIL PearsonTypeIII/lmom_forward_negative_skew/linear_moment: expected 0.10127700839455282 got -0.10127700839455282
```

## Failures grouped by subsystem

### 1. StudentT PDF -- 1/sigma Jacobian (Numerics risk #1)
- `fixtures/distributions/univariate/student_t.json`: `student_t_shifted_mu2p5_sigma0p5_nu4/pdf`
  (sigma=0.5, new value = old x2 = 1/0.5) and `student_t_large_df_sigma2_normal_approx/pdf` (sigma=2,
  new value = old / 2, both assertions in that case) -- exactly the two sigma!=1 PDF cases in the
  corpus (verified: the other two StudentT PDF cases, `student_t_standard_nu4_pdf_cdf` and
  `student_t_large_df_normal_approx`, both use sigma=1 and correctly did not flip). Matches
  `docs/upstream-csharp-issues.md` "BUG -- StudentT PDF omits the 1/sigma Jacobian" exactly; upstream
  fixed our documented finding.

### 2. PearsonTypeIII/LogPearsonTypeIII -- signed L-skewness (Numerics risk #2)
- `fixtures/distributions/univariate/pearson_type_iii.json`: `lmom_forward_negative_skew/linear_moment`
  sign-flips (old positive T3, new `T3 *= Sign(gamma)` makes it negative for this negative-skew
  case). Matches `docs/upstream-csharp-issues.md` "VERIFY (arguable BUG) -- PearsonTypeIII /
  LogPearsonTypeIII L-skewness sign for negative skew" -- upstream resolved our "arguable bug" in
  the signed direction. No LogPearsonTypeIII fixture case flipped (none exercises a comparable
  negative-skew L-moment path in the current corpus -- a coverage gap, not a surprise).

### 3. MCMCSampler -- MAP fitness scale (Numerics risk #3)
- `fixtures/sampling/mcmc/rwmh.json`: `normal_rstan/map_fitness` sign-flips (was pinned to the
  buggy positive `-DE.BestParameterSet.Fitness` value; upstream now stores the log-likelihood-scale
  value, i.e. the C++ side already had it right and C# comes into line). Matches
  `docs/upstream-csharp-issues.md` "CONSISTENCY -- MCMCSampler.MAP.Fitness is on a different scale"
  verbatim -- this doc's own text says the fixture "asserts `map_fitness` at its (buggy, positive)
  value specifically to lock this behavior in", so this flip was anticipated by name in that entry.

### 4. GeneralizedBeta -- Mode boundary fix (Numerics risk #4)
- `fixtures/distributions/univariate/generalized_beta.json`: `alpha0p42_beta1p57_unit/mode`
  (J-shape, alpha<1<beta) goes from the out-of-support `~58.0` to the boundary `0`. Matches
  `docs/upstream-csharp-issues.md` "BUG -- Beta / GeneralizedBeta Mode can fall outside the
  support" exactly (that section cites this same oracle value, `~58.0`). The corresponding
  plain-`Beta` fixture (`beta_distribution.json` `alpha0p42_beta1p57`) is already pinned to the
  boundary value `0.0` and did NOT flip -- consistent with Beta's own Mode formula already being
  correct pre-bump; only GeneralizedBeta's shift/scale variant carried the bug.

### 5. BestFit DataFrame plotting positions -- Hirsch-Stedinger rewrite (BestFit risk #1)
- `fixtures/estimation/mle_censored.json`: all 4 `plotting_position` assertions in
  `normal_censored_data_frame` (a frame with exact + interval + one threshold window + uncertain
  data) changed value under the new peakFQ-faithful ARRANGE2/PPLOT2/PLPOS port. The sibling case
  in the same file, `normal_mgbt_low_outliers`, has 2 `plotting_position` assertions and did
  **not** flip -- it's an all-exact-data frame with no threshold window/interval/uncertain mix, so
  the old sequential Kj/Kl scheme and the new per-index-threshold-classification scheme coincide
  for a frame with nothing to classify against. This is the expected shape of the risk (the spec:
  "multi-threshold and tied-value frames change oracles") -- not a surprise.

## Cross-check against the spec's risk lists

**Predicted-and-observed (5 of 27 ranked risk items produced a census failure):** Numerics #1
(StudentT), #2 (PT3/LP3 signed T3), #3 (MCMC MAP fitness), #4 (Beta/GenBeta Mode); BestFit #1
(DataFrame plotting positions). All 10 raw failure lines are accounted for by these five items --
**zero surprises**: no failure in the census falls outside the spec's predicted risk list.

**Predicted risk areas with NO census failure** (the brief's other required half of the
cross-check). For each, the fixture corpus was checked directly (not just inferred from the
absence of a FAIL line) to distinguish "genuinely unaffected" from "changed but untested by the
current corpus":

- Numerics #5 Mixture zero-inflation renormalization -- `fixtures/estimation/mixture.json` has
  zero-inflation cases; none flipped. Needs the T6 implementer to check by hand whether the
  current cases' weights already sit on the post-renormalization simplex (no visible change) or
  whether the emitter path just doesn't reach the changed setter for these configs.
- Numerics #6 ArchimedeanCopula `ParametersValid` sentinel (Clayton/Gumbel/Joe false->true) --
  **genuine gap**: none of `clayton_copula.json`/`gumbel_copula.json`/`joe_copula.json` asserts
  `parameters_valid` at all (confirmed by direct query). Exactly the case the spec called out:
  "copula `parameters_valid` flips appear only if a fixture asserts them." T8 needs a new fixture
  case to pin this at all.
- Numerics #7 EmpiricalDistribution ascending-X / duplicate-X legality -- `empirical_distribution.json`'s
  `palisades_atRisk` case already contains duplicate X values (10800, 11100, 11700, ... each
  appear twice) constructed via explicit pre-computed `p`, bypassing the internal validation path
  that changed; no case constructs from raw ascending/duplicate X through the validated
  constructor. Gap, not a contradiction.
- Numerics #8 MultivariateNormal COVSRT singular-covariance branch -- `multivariate_normal.json`'s
  "extreme"/"perfect" cases (`curated_bivariate_extreme_correlation*`, `r_mvtnorm_3d_perfect_*`)
  top out at |rho|=0.999/0.99 -- still positive-definite, per the spec's own "PD-covariance cases
  unchanged" carve-out. No case in the corpus has a truly singular (|rho|=1 or rank-deficient)
  covariance. Gap.
- Numerics #9 GammaDistribution `PartialKp` near-zero-skew branch -- the one B17C GMM fixture
  (`gmm_bulletin17c_smoke.json` `lp3_exact_iterative_bfgs`) doesn't land at |skew| < 1e-4. Gap.
- Numerics #10 GeneralizedLogistic near-zero kappa truncated series -- no `generalized_logistic.json`
  case is pinned at a kappa magnitude inside the new near-zero band (distinct from the existing
  exact-kappa==0 divergence case, which the spec says already agrees). Gap.
- Numerics #11 BrentSearch `Bracket` k-factor -- not directly fixture-asserted anywhere (Brent is
  an internal solver step, not a public oracle surface); would only show up indirectly through a
  distribution/model fit that depends on bracket width. No indirect flip observed.
- Numerics #12 grab-bag (Search descending-order, Bilinear log10 guards, Histogram auto-adapt,
  Probability HPCM underflow guard, Tools.Log10 NaN guard, BivariateEmpirical cache invalidation,
  CompetingRisks setter/PerfectlyNegative property, MultivariateStudentT finite validation) --
  checked `special_functions/search.json` directly: all 12 cases are ascending-order
  (below/interior/above-range), none descending -- gap, matches the spec's own risk note ("we
  mirror the old dead branch") implying no current coverage. `competing_risks.json` has several
  `perfectly_negative` cases (min/max rule, 2- and 3-normal) but the changed behavior
  (`CorrelationMatrix` no longer zeroed) is a public-property-only change with no effect on the
  pinned CDF/PDF oracle values, so no flip is expected there regardless of coverage.
- Numerics #13 BoxCox/YeoJohnson `FitLambda` hardening (`CanFitLambda` pre-checks, Brent objective
  clamping, candidate rejection at non-finite/|lambda|>5, YeoJohnson lambda==2 tolerance band) --
  `fixtures/data/statistics_utilities.json` (`kind: data_utility`) has exactly the fixture surface
  for this: `box_cox_lambda_fit`/`box_cox_transform_half`/`box_cox_transform_neg` over the
  `positive_sample` dataset (30 points, range ~76-6096, no degenerate/constant values) and
  `yeo_johnson_lambda_fit`/`yeo_johnson_transform_half`/`yeo_johnson_transform_17` over
  `mixed_sign` (30 points, range ~-4.5 to 3.1). None flipped. Checked directly: neither dataset is
  degenerate (so `CanFitLambda`'s new NaN short-circuit can't trigger), the two direct-`Transform`
  cases (`_half` at lambda=0.5, `_neg`/`_17` at lambda=-0.3/1.7) never approach the new lambda==2
  tolerance band, and no case's args or data are positioned to hit a non-finite Brent objective or
  the new |lambda|>5 rejection. **Genuine gap**, not "unaffected" -- every guard this item adds is
  a rejection/clamp path for a degenerate or extreme input, and every current case is a
  well-behaved, ordinary sample. T2 needs new cases (a constant/near-constant sample for
  `CanFitLambda`, and a lambda near 2 for the YeoJohnson tolerance band) to observe this item at
  all.
- Numerics #14 distribution validation wave (assign-before-validate setter ordering +
  NaN/Inf rejection: Gumbel/Logistic/InverseChiSquared/Binomial/Deterministic/NoncentralT/
  ChiSquared/KernelDensity) -- checked all 8 named fixtures directly. Six already carry an
  explicit invalid-parameter case asserting `parameters_valid: false`: `gumbel.json`
  (`invalid_nan_location`, `invalid_inf_location`), `logistic.json` (same two names),
  `inverse_chi_squared.json` (`invalid_params` at dof=0,sigma=0), `binomial.json`
  (`binomial_invalid_params` at p=-1), `noncentral_t.json` (`nct_invalid_params_nu_too_small`,
  `nct_invalid_params_lambda_inf`), `chi_squared.json` (`invalid_params_dof0`) -- none flipped.
  `deterministic.json` and `kernel_density.json` have **no** invalid-parameter case at all (both
  only construct valid params). Two distinct findings, not one:
  1. For the six with coverage, the surface-level "does `parameters_valid` reject NaN/Inf"
     semantic already returned `false` pre-bump and still does -- plausibly already-correct
     (a live re-check of current field values, independent of the setter-ordering bug), not a
     coverage gap in the usual sense.
  2. The actual **assign-before-validate ordering** bug this item describes only manifests on a
     two-step sequence (construct valid -> `SetParameters` with bad values -> read back the
     *previous* state to see whether it leaked/corrupted before the throw). Every fixture case
     in this corpus is single-shot construct-then-assert; none does a construct-then-reset
     sequence. So this half of item #14 is **untestable by the current harness shape for any of
     the 8 distributions**, not merely uncovered by these particular fixtures -- a structural gap
     for whichever task (T3) owns this item, not a one-line fixture addition.
  `deterministic.json`/`kernel_density.json` additionally lack even the surface-level NaN/Inf
  case -- an ordinary coverage gap on top of the structural one.
- Numerics #15 `UnivariateDistributionFactory` switch completeness / throws-on-unknown /
  `KernelDensity` case presence (distinct from the `TryCreateDistribution` new-API item noted
  below) -- confirmed via `git diff a2c4dbf 2a0357a --
  Numerics/Distributions/Univariate/Base/UnivariateDistributionFactory.cs` in the submodule: the
  old if/else chain had no branch at all for `Deterministic`, `Empirical`, `KernelDensity`,
  `VonMises`, `CompetingRisks`, `Mixture`, or `UserDefined`, and silently fell through to
  `return new Deterministic()` for any of them (or for a genuinely undefined enum value); the new
  switch has an explicit case for every one of those (including `KernelDensity` at line 74-75 and
  `VonMises` at line 112-113), throws `NotSupportedException` for the three composite/user types,
  and `ArgumentOutOfRangeException` for anything undefined. This is a real, substantial behavior
  change -- and yet produced zero census failures, for a specific, checkable reason:
  `tools/oracle_emitter/Program.cs` `Build()` (and its `BuildComponent()` sibling) special-cases
  exactly the types the old factory silently mishandled. `KernelDensity`/`Empirical` are built via
  the bespoke `BuildComposite()` path (never call `UnivariateDistributionFactory.CreateDistribution`
  at all), and `VonMises` is explicitly bypassed with `target == "VonMises" ? new VonMises() :
  UnivariateDistributionFactory.CreateDistribution(type)` -- with the comment "VonMises is in the
  C# enum but not in the upstream factory yet -- construct directly," i.e. a workaround for
  precisely the bug `docs/upstream-csharp-issues.md` documents ("BUG -- UnivariateDistributionFactory
  has no case for VonMises (falls through to Deterministic)"). Every OTHER plain distribution
  (Gumbel, Normal, StudentT, etc.) already had an explicit if/else branch pre-bump, so the
  switch-refactor changes nothing observable for them. **Net finding: the fix is real and landed,
  but corehydro's own pre-existing compensating workaround for the VonMises half of it hides it
  from the census, and the emitter's composite-path shape hides the KernelDensity/Empirical/
  throws-on-unknown halves structurally (they never call the flat factory in the first place).**
  T7 should both verify the C++ factory already has (or add) the `KernelDensity` case per this
  repo's own item-15 note, and decide whether to retire the VonMises `Build()`/`BuildComponent()`
  workaround now that upstream's factory handles it -- retiring it is the only way to get this fix
  under census coverage, since no fixture asserts factory-throw behavior directly.
- BestFit #2 Bulletin17CAnalysis bootstrap UQ rework -- `bulletin17c_analysis_smoke.json` has a
  `lp3_bias_corrected_bootstrap` case; it did not flip. Per the design, this rework only becomes
  visible when a replicate is discarded/retried/warm-started or the Mahalanobis gate rejects
  something -- plausible this seeded smoke case's replicates all succeed cleanly on the first try
  under both old and new semantics. Needs closer inspection in T20 rather than assumed-clean.
- BestFit #3 GeneralizedMethodOfMoments acceptance-on-non-failure + sticky NM fallback --
  `gmm_bulletin17c_smoke.json`'s one case converges normally in both directions apparently; no
  fallback triggered. Gap/plausible-inert, needs T13 to construct a case that actually hits a BFGS
  failure to exercise the new fallback.
- BestFit #4 Bulletin17CDistribution `CloneWithDataFrame` / ROS trigger change -- no dedicated
  clone-path fixture exists yet. Gap.
- BestFit #5 BootstrapDiagnostics discard bookkeeping (`AttemptedReplicates`/`RetainedReplicates`/
  `TransformFailures`/GMM `OptimizationStatus` counters/`OptimizerFallbacks`;
  `ValidReplicates`/`FailureRate`/`AverageRetries` redefinition) -- **genuine gap, the cleanest of
  the four omitted items**: a direct search of the whole `fixtures/` tree and of
  `tools/oracle_emitter/Program.cs` for any of `AttemptedReplicates`, `RetainedReplicates`,
  `TransformFailures`, `OptimizerFallbacks`, `BootstrapDiagnostics`, `ValidReplicates`,
  `FailureRate`, or `AverageRetries` returns zero hits in either place. No fixture (not even the
  three `bulletin17c_analysis_smoke.json` UQ cases, `lp3_multivariate_normal`/
  `lp3_linked_multivariate_normal`/`lp3_bias_corrected_bootstrap`) asserts any bootstrap-diagnostic
  field at all -- only `confidence_level`/`exceedance_probability`/`point_estimate`/`lower_ci`/
  `upper_ci`/`parameter`. The emitter has no code path that would even read a
  `BootstrapDiagnostics` object today. T19 needs new fixture cases (and, since nothing dumps this
  DTO yet, likely new emitter/dispatch glue) from scratch to observe this item at all -- there is
  no re-pin here, only net-new coverage.
- BestFit #6 ThresholdData source-vs-effective count split -- `mle_censored.json`'s
  `normal_censored_data_frame` has a threshold series but only `parameter`/`plotting_position`
  assertions are checked, not a repeated-`ProcessThresholdSeries`-idempotency probe. The
  plotting-position flip in that same case (group 5 above) may already be entangled with this
  item since both changes land in the same DataFrame code path -- worth the T12 implementer
  double-checking count-split behavior isn't itself contributing to the plotting delta versus the
  Hirsch-Stedinger rewrite alone.
- BestFit #7 Jeffreys guards (single-parameter skip, bounds-checked scale index) -- present in
  MLE/MAP censored/normal fixtures indirectly (Jeffreys priors are used in several `map_*.json`
  cases) but none flipped; none of the current MAP cases appears to hit a single-parameter model
  (the crash case) or an out-of-bounds scale index. Gap.
- BestFit #8 RatingCurve default priors (exponent -5..5 -> -10..10, `a` lower bound 0.5 -> 0) --
  `rating_curve_smoke.json`'s one case (`single_segment_nelder_mead`) converges inside the old
  bounds already, so the widened bounds don't move the fit. Gap/plausible-inert.
- BestFit #9 TimeSeries BoxCox/YeoJohnson `FitLambda` failure handling -- none of the
  `time_series_{ar,ma,arima,arimax}_smoke.json` cases hits a degenerate/failing transform fit.
  Gap.
- BestFit #10 BivariateDistribution PseudoLikelihood auto-run `CalculatePlottingPositions` -- both
  `bivariate_smoke.json` cases (`normal_copula_ifm_nelder_mead`, `studentt_copula_ifm_nelder_mead`)
  use IFM, not PseudoLikelihood, so the changed code path (which the design says "fixes our audit
  finding that the path could never estimate") isn't exercised by any current fixture at all --
  matches `docs/upstream-csharp-issues.md` "CONSISTENCY -- BivariateDistribution model-level
  PseudoLikelihood MLE returns Estimate()==false" (i.e. the old path never worked, so naturally no
  passing fixture exists yet to have flipped). T15 needs a new PseudoLikelihood fixture case from
  scratch, not a re-pin.
- BestFit #11 NumericalDiff Hessian/Jacobian edge handling -- no diagnostics fixture isolates an
  edge-of-domain Hessian step. Gap.
- BestFit #12 BestFitLinkFunctionFactory YeoJohnsonLink routing to the Numerics link -- no
  fixture currently exercises the BestFit YeoJohnsonLink factory branch directly (it's reached
  only via B17C link selection, none of the current link fixtures selects YeoJohnson). Gap; this
  item is also structurally forced (T17 must delete `models/link_functions/yeo_johnson_link.hpp`
  regardless of oracle coverage, since the upstream class disappeared).

**New API items** (Numerics `TryCreateDistribution`/`TrySetParameters`/`Marginal`/`Conditional`/
`RunningStatistics.Clone`, `BivariateCopula.CloneMarginal`) are additive and fixture-silent by
construction -- no census signal expected or found, consistent with the spec listing them
separately from "behavior changes."

**Net read:** every risk item that DID move an existing oracle is now enumerated and none were
missed by the spec. The much longer list of risk items that did NOT move anything is, on
inspection, overwhelmingly a fixture-coverage gap rather than evidence the described C# change is
a no-op -- the design document's own execution plan already anticipates this ("each behavior
change gains at least one new fixture case pinned from upstream's new tests"), so later tasks
should not treat "no census failure" as "nothing to port" for any of the items listed above.

## Sanity check: C++/R/Python untouched

```
ctest --test-dir core/build -R fixtures --output-on-failure
100% tests passed, 0 tests failed out of 1 (test_fixtures, 59.08 sec)
```

Confirms the C++ core still matches the OLD (unmodified) fixtures bit-for-bit; only the C# side
moved under the submodule bump, exactly as expected before any porting begins.
