# Upstream sync: Numerics v2.1.4 + RMC-BestFit v2.0.0

Date: 2026-07-19. Status: approved design. This is also the first live run of the recurring
upstream-release update process; the process itself is a deliverable (see "Process capture").

## Context

corehydro is pinned to Numerics `a2c4dbf` (3 commits past v2.1.1) and RMC-BestFit `fc28c0c`
(v2.0-beta.5). Upstream shipped Numerics v2.1.4 (`2a0357a`, spanning v2.1.2/2.1.3/2.1.4) and the
official RMC-BestFit v2.0.0 (`c2e6192`). Both tags equal their repo's current `main`. Goal:
restore 1:1 feature parity with both releases and re-pin every oracle to the new C# behavior.

The defining property of this sync: both releases incorporate fixes RMC made in response to
corehydro's own port audit (`docs/upstream-csharp-issues.md`). Numerics `93d374e` added (and
`a8006fc` later removed) a `NUMERICS_PORT_REVIEW_SUMMARY.md` that answers our findings item by
item; `33dc1af`/`651035e` implement them. BestFit `b43943c` "Fix BestFit port review findings"
fixes five more of our documented findings. Consequently the ground truth flips in two
directions:

1. Where the C++ mirrors an old C# bug "faithfully (oracle-verified)", upstream fixed the bug.
   The C++ adopts the fix and the affected oracles re-pin.
2. Where the C++ carries a documented intentional divergence, upstream adopted our behavior.
   The divergence notes retire; where upstream's exact formulation differs from ours (e.g. a
   new near-zero series branch), we converge to upstream exactly.

## Scope: Numerics a2c4dbf -> v2.1.4 (library diff: 48 files, +2243/-806)

### Behavior changes to port (fixture-relevant, ranked by risk)

1. StudentT: PDF gains the 1/sigma Jacobian (every sigma != 1 PDF oracle changes by exactly
   1/sigma); InverseCDF tail refactor (ulp drift, tail retains mu/sigma, MaxValue saturation
   guard removed); finite-dof validation; setter-ordering fix.
2. PearsonTypeIII + LogPearsonTypeIII: `T3 *= Sign(gamma)` (signed L-skewness; our
   negative-skew L-moment fixture pins the old positive value and breaks), T3==0 and gamma==0
   branches in both L-moment directions, alpha>=100 Stirling branch upstreamed, L1 simplified
   to `mu` (ulp drift).
3. MCMCSampler: MAP initialization stores fitness on the log-likelihood scale
   (`-DE.BestParameterSet.Fitness`); MAP/mode oracles for MAP-initialized Bayesian fixtures
   are exposed. NUTS: step-size heuristic and leapfrog init call the configured
   `GradientFunction` (streams unchanged under the default numerical gradient; verify).
4. Beta + GeneralizedBeta: Mode boundary fix (U-shape -> 0.5/midpoint, J-shapes -> boundary;
   old code returned values outside the support). PERT degenerate midpoint preserved.
5. Mixture: IsZeroInflated/ZeroWeight setters renormalize component weights to `1-ZeroWeight`;
   EM/MLE fit rescales weights onto the configured simplex with equal-weight fallback;
   `_parametersValid` maintenance. Zero-inflated fixtures change.
6. ArchimedeanCopula: ValidateParameter sentinel bug fixed (Clayton/Gumbel/Joe
   `ParametersValid` flips false -> true). All copulas: NaN/Inf theta rejection; `Clone`
   deep-copies marginals via new `BivariateCopula.CloneMarginal`.
7. EmpiricalDistribution: ascending-X validation enforced (`ValidateData` on construction +
   lazy throw from CDF/InverseCDF); duplicate X now legal (`strictX=false`); exception-type
   changes.
8. MultivariateNormal: COVSRT singular-covariance permutation branch repaired (loop bounds,
   packed indices, swap offsets); MVNDST over rank-deficient/perfectly-correlated matrices
   changes; PD-covariance cases unchanged. MVNDNT void + INFORM=0.
9. GammaDistribution: `PartialKp` near-zero-skew branch returns the Cornish-Fisher derivative
   `(z^2-1)/6` (affects B17C GMM gradients at |skew| < 1e-4).
10. GeneralizedLogistic: exact kappa==0 L-moment limits (agrees with our existing divergence)
    plus a `|kappa| <= NearZero` truncated-series branch we do NOT have; adopt upstream's
    series exactly.
11. BrentSearch: `Bracket` applies the geometric expansion factor k (was unused; we mirror
    with `(void)k`), validates inputs, guards non-finite objectives, bounds iterations.
12. Search: descending-order Bisection/SequentialSearch/Hunt fixed (we mirror the old dead
    branch). Bilinear: guarded `Tools.Log10` everywhere. Histogram: out-of-range `AddData`
    auto-adapts endpoint bins. Probability: HPCM `cdf < 1e-300` underflow guard enforced.
    Tools: `Log10` NaN guard change. BivariateEmpirical: `SetParameters` invalidates the
    cached bilinear interpolator; finite validation. CompetingRisks: Dependency setter
    invalidates cached MVN; PerfectlyNegative no longer zeroes the public CorrelationMatrix;
    SetParameters length/validity handling. MultivariateStudentT: finite validation.
13. BoxCox + YeoJohnson: `FitLambda` gains `CanFitLambda` pre-checks (degenerate sample ->
    NaN), non-finite-objective clamping inside Brent, candidate rejection (non-finite,
    |lambda| > 5, non-finite LL); `LogLikelihood` -> -Inf on non-finite intermediates;
    YeoJohnson lambda==2 branch now tolerance-based (`abs(lambda-2) < 1e-8`, real numeric
    change in that band). YeoJohnsonLink: throws on NaN FitLambda.
14. Validation wave (many distributions): assign-before-validate setter ordering
    (Gumbel/Logistic/InverseChiSquared/Binomial/Deterministic/NoncentralT), NaN/Inf rejection
    (ChiSquared/KernelDensity and others). NoncentralT: AS 243 de-goto refactor (math
    identical; watch ulp). TruncatedDistribution: validation centralized (we already slice
    parameters correctly; adopt the validation/exception semantics).
15. UnivariateDistributionFactory: switch-based, complete cases, throws on unknown (largely
    pre-aligned; verify KernelDensity case exists in C++).

### New API to port (parity)

- `UnivariateDistributionFactory.TryCreateDistribution(type, out dist)`.
- `MultivariateNormal.TrySetParameters` / `TrySetCovariance` / `IsDensityValid` (invalid
  state: PDF=0, LogPDF=-Inf), `Marginal(indices)`, `Conditional(indices, values)`.
- `RunningStatistics.Clone()`; `Combine` with an empty operand returns a clone (C++ value
  semantics largely pre-aligned; add the API name).
- `BivariateCopula.CloneMarginal` (protected helper backing the deep-copy clones).

### Not ported (documented)

- `TimeSeriesDownload.cs` (+754/-237, network downloads) and `Series.cs` container changes:
  severed TimeSeries surface.
- `Tools.ParallelAdd` hardening: corehydro uses serial reductions by design.
- XML-doc/BOM churn (`5367945` enforced XML-doc warnings): no code effect.

## Scope: RMC-BestFit fc28c0c -> v2.0.0 (library diff: 64 files, +4750/-1807)

Library paths under `src/RMC.BestFit/` are unchanged by the repo restructure (no renames; our
`ported from:` headers resolve as-is). Pervasive BOM/mojibake comment churn is noise.

### Behavior changes to port (ranked by risk)

1. DataFrame plotting positions REWRITTEN (`79d608b` + `2727a1e` + `ab211a3` + `42b106b`):
   peakFQ-faithful ARRANGE2/PPLOT2/PLPOS port (each observation classified against the
   threshold covering its OWN index; old sequential Kj/Kl scheme gone); strict validation now
   throws (non-finite values, overlapping threshold windows, counts exceeding duration,
   infeasible NumberBelow); `EnsureDistinctPlottingPositions` spreads duplicate positions
   deterministically (tie center preserved; value desc/index/ordinal comparator). The legacy
   introsort tie permutation is explicitly preserved upstream, so our ArraySortHelper introsort
   port stays load-bearing, but `calculate_plotting_positions` needs a substantial re-port.
   Empirical-distribution construction collapses repeated X into right-continuous CDF steps
   and returns NaN/null on degenerate input (feeds `GetNonparametricMoments*` and B17C
   defaults via ROS).
2. Bulletin17CAnalysis bootstrap UQ rework (`1b424e3`, `71b7d4b`, `0dc8594`, `7efa9d0`; net
   state only, `c155732` is a superseded staging snapshot): failed replicates DISCARDED (no
   parent-parameter substitution); replicates warm-start via
   `Bulletin17CDistribution.CloneWithDataFrame` when low outliers/censored/threshold data
   present; acceptance gate keyed to the new GMM non-failure-termination semantics;
   maxRetries 5 -> 10; Mahalanobis threshold ChiSq(0.9999) -> adaptive `1 - 1/(5B)`; singular
   parent covariance regularized via `MatrixRegularization.MakeSymmetricPositiveDefinite`;
   pivot path compacts to accepted replicates, drops z-limit rejections (zLimit=6) and failed
   transforms; `CreatePivotYeoJohnsonLink` (finite/roundtrip validation, lambda-rail
   rejection at |lambda| >= 4.999, IdentityLink fallback); new public
   `UncertaintyDiagnosticMessage`; abort-with-warning when < 2 realizations survive or > half
   discarded; master PRNG seed block keyed to original replicate index.
3. GeneralizedMethodOfMoments (`5e1877f`, `b43943c`): estimates accepted on any NON-FAILURE
   optimizer termination with finite best point; sticky Nelder-Mead fallback after one BFGS
   failure (`OptimizerFallbackCount`); internal `TryGetCovariance`;
   `ConvergedWithinTolerance` off-by-one fixed (our audit finding).
4. Bulletin17CDistribution: `CloneWithDataFrame` (binds a resampled frame without rebuilding
   parameters; preserves parent initial values and penalties; regional-skew prior no longer
   dropped on clones); `SetDefaultParameters` ROS override now triggers only for low outliers
   or threshold series (not uncertain/interval) — GMM starting values change for those fits.
5. BootstrapDiagnostics: discard bookkeeping (`AttemptedReplicates`, `RetainedReplicates`,
   `TransformFailures`, GMM `OptimizationStatus` counters, `OptimizerFallbacks`;
   `ValidReplicates`/`FailureRate`/`AverageRetries` redefined over attempts).
6. ThresholdData: source-vs-effective count split (`SourceNumberAbove`, `SetProcessedCounts`,
   Validate/Clone updates) making `ProcessThresholdSeries` idempotent (our audit finding).
7. Jeffreys guards (`1abe795`, `b43943c`): `TryGetJeffreysScaleParameter` overloads on
   UnivariateDistributionModelBase (Gamma/Weibull scale index 0, others 1, bounds-checked);
   single-parameter components skipped; `-Inf` short-circuit; MixtureModel.Clone propagates
   IsZeroInflated/ZeroWeight (our audit finding); CompetingRisksModel same pattern;
   UnivariateDistribution single-parameter Jeffreys crash fixed (our audit finding).
8. RatingCurve default priors: exponent bounds -5..5 -> -10..10; a lower bound 0.5 -> 0 with
   Uniform(0,5), `IsPositive` removed; sigma upper-bound helpers extracted.
9. TimeSeries models (AR/MA/ARIMA/ARIMAX): BoxCox/YeoJohnson `FitLambda` failure handling
   (lambda=0 + emptied training series + validation message instead of a crash).
10. BivariateDistribution: PseudoLikelihood validates pseudo observations on (0,1) and
    auto-runs `CalculatePlottingPositions()` (cached against `PlottingPositionVersion`) —
    fixes our audit finding that the path could never estimate.
11. NumericalDiff: Hessian diagonal step search bounded (64 attempts) with cached stencil
    values; Jacobian one-sided fallback when a side is non-finite. Interior values identical;
    edges change.
12. BestFitLinkFunctionFactory: YeoJohnsonLink case constructs the NUMERICS link (BestFit's
    duplicate class DELETED upstream, which also removes the CS0104 ambiguity). We delete
    `models/link_functions/yeo_johnson_link.hpp` and route the factory accordingly;
    parity-check the Numerics link's DLink against the deleted BestFit one.

### Not ported (documented as skips)

- `Analyses/Support/AnalysisProgress.cs` (new): GUI progress plumbing (phase percents,
  SafeProgressReporter, AsyncLocal parallelism budget). Order-independent seeded loops mean no
  numeric effect.
- The 15 analysis orchestrators' progress/report/persistence churn, including the entire
  BayesianAnalysis +694 diff (report text + Task.Run threading; sampler numerics untouched).
- BatchAnalysisRunner threading changes (permanent WPF skip).
- The new `src/RMC.BestFit.App` / `.UI` / `.Api` projects (WPF GUI, UI wrapper layer, REST/MCP
  server): out of scope for the stats-library port. Noted in docs as a visible upstream
  addition R/Python users get natively through corehydro's own APIs.
- XML persistence additions (PlottingPositionVersion stamps in XML, TrainingTimeSteps restore,
  `FromXElement` recomputation): corehydro severs XML.

## Execution design

Approach: bump-first, oracle-driven, on branch `upstream-sync-2026-07` in this worktree, one
PR at the end.

- T0 bumps both submodule gitlinks to the release tags and repairs the oracle emitter: remove
  the stale CS0104 `patched/Bulletin17CAnalysis.cs` (compile the real file), fix any compile
  breaks from the new C# surface. Then run `tools/verify_oracles.py` against the not-yet-touched fixtures: the failure
  list is the empirical enumeration of every fixture whose oracle the new C# moved.
  Cross-check it against this spec's risk lists; investigate surprises in either direction.
- Port in dependency order, Numerics before BestFit (B17C/GMM/DataFrame consume the
  distribution, L-moment, and optimizer changes): small-utilities wave -> Box-Cox/YJ ->
  validation wave -> StudentT/NoncentralT -> PT3/LP3/Gamma/GenLogistic L-moment wave ->
  Beta/GenBeta/Mixture -> Empirical/Truncated/factory -> copulas -> MVN -> MCMC fixes ->
  BestFit: DataFrame/ThresholdData plotting rewrite -> GMM/NumericalDiff ->
  Jeffreys/Bivariate/RatingCurve/TimeSeries -> YJ link removal -> B17C distribution ->
  B17C analysis bootstrap + BootstrapDiagnostics (last: consumes GMM, DataFrame, and the
  B17C distribution changes).
- Subagent-driven development: fresh implementer + reviewer per task, TDD (re-pin or add the
  fixture first, then port until C++/R/Python all agree), ledger, commit per task.
- Fixture policy: oracles live only in `fixtures/*.json`; changed values re-pinned via the
  emitter against the new C#; each behavior change gains at least one new fixture case pinned
  from upstream's new tests (Test_ParameterValidity, Test_MCMCInitialization, Test_Search,
  MVN singular-CDF cases, plotting-position/bootstrap-diagnostics/GMM tests, signed-T3 cases,
  StudentT normalization). The Gibbs registry model adopts upstream's corrected conjugate
  formulas (`Test_Gibbs` rework) and its fixture re-pins.
- Known chaotic-sensitivity precedent stands: deterministic surfaces pin exact; seeded
  short-chain MCMC curves assert structural invariants only where already documented. B17C
  bootstrap streams are seeded and deterministic: re-pin exact.
- ABI: distribution/MVN class layouts change, so R rebuilds use `R CMD INSTALL --preclean`.
- Bindings: new core API is bound in R/Python only where that class surface is already
  exported (e.g. distribution verbs); otherwise core + fixtures only. Every new enum arm or
  binding knob must be exercised by a fixture (Phase 10 lesson).
- Docs pass at the end: `docs/upstream-csharp-issues.md` entries marked RESOLVED-upstream at
  v2.1.4/v2.0.0 with commit refs; `upstream/CLAUDE.md` and `.claude/CLAUDE.md` status updated;
  package versions bump 0.1.0 -> 0.2.0 recording "validated against Numerics v2.1.4 /
  RMC.BestFit v2.0.0"; `_pkgdown.yml` + `site/_quarto.yml` updated if any new export lands.

## Definition of done

ctest, testthat, and pytest fully green; `verify_oracles.py` 0 failed (documented skips only);
cross-language seeded digests bit-identical; CI green on the full matrix; PR open. Plus the
process deliverables below.

## Process capture (this run tests the update process)

- Write `docs/upstream-sync.md`: the repeatable release-absorption process as actually
  executed (fetch tags -> classify diffs -> bump + emitter repair -> oracle-gate failure list
  -> port in dependency order -> re-pin -> verify -> ship), with the classification prompts
  and the pitfalls hit.
- Save lessons-learned memory files as they occur, not just at the end.
- Record what automation would have earned its keep (candidate requirements for a future
  `upstream_diff.py`), without building it now.

## Risks

- The Hirsch-Stedinger rewrite is the largest single re-port; multi-threshold and tied-value
  frames change oracles, and new validation throws where values were returned. Mitigate with
  upstream's PlottingPositionTests as the fixture source.
- The B17C bootstrap rework couples to GMM acceptance semantics; port GMM first and re-pin
  B17C last among BestFit tasks.
- Emitter compile risk after the bump (new C# language surface, InternalsVisibleTo, csproj
  changes): contained in T0 before any porting begins.
- NUTS gradient change: verify the default-gradient equivalence before assuming seeded-stream
  stability.
