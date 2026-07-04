# Upstream C# issues found during the Phase 1 port

Running log of potential **bugs, edge-case gaps, and consistency issues in the upstream
USACE-RMC C# libraries** (`Numerics` @ `a2c4dbf`, `RMC.BestFit` @ `fc28c0c`) that surfaced while
porting the univariate distribution layer to the C++ core.

The port's governing rule is bit-for-bit fidelity with the C# (so the oracle values hold), so in
almost every case below the C++ **faithfully mirrors the C# behaviour** — including its bugs — and
the divergence, where we made one, is documented at the call site. This document is the backlog for
a later pass to (a) confirm each finding against upstream intent and (b) potentially submit fixes to
the C# repositories. Nothing here blocks the C++/R/Python packages.

Severity: **BUG** = produces a wrong/undefined result a user could hit; **ROBUSTNESS** = works for
tested inputs but fragile at an edge; **CONSISTENCY/API** = surprising but arguably intentional;
**COSMETIC** = dead code / comments.

Each entry: what, where, evidence, how the port handled it, suggested fix.

---

## BUG — StudentT PDF omits the 1/σ Jacobian (density does not integrate to 1 for σ≠1)

- **Where:** `Numerics/Distributions/Univariate/StudentT.cs`, `PDF(x)`.
- **What:** For the location-scale Student-t with parameters `[μ, σ, ν]`, `PDF` computes the
  standard t density at `Z=(x−μ)/σ` and returns it **without** the `1/σ` scaling factor. The result
  is not a proper probability density when `σ≠1` (it does not integrate to 1).
- **Evidence (oracle):** `StudentT(2.5, 0.5, 4).PDF(1.4)` returns `0.0516476521260042`; a correctly
  scaled density is `0.1032953...` (exactly 2× = 1/σ with σ=0.5).
- **Port handling:** mirrored faithfully (oracle-verified); the C++ header notes it.
- **Suggested C# fix:** multiply the returned density by `1.0 / Sigma`. NOTE this changes oracle
  values for every σ≠1 case — coordinate with the test literals.

## BUG — Beta / GeneralizedBeta Mode can fall outside the support

- **Where:** `Numerics/Distributions/Univariate/BetaDistribution.cs` and `GeneralizedBeta.cs`, `Mode`.
- **What:** The mode uses `(α−1)/(α+β−2)` (rescaled to `[min,max]` for GeneralizedBeta). The
  guard only special-cases `α≤1 && β≤1` (→ midpoint). When exactly one shape is `<1` and
  `α+β<2`, the denominator `α+β−2` is a small negative number and the formula returns a value far
  outside the support.
- **Evidence (oracle):** `GeneralizedBeta(α=0.42, β=1.57, min=0, max=1).Mode` returns `≈ 58.0`
  (support is `[0,1]`). Same math applies to `BetaDistribution` on `[0,1]`.
- **Port handling:** mirrored faithfully (oracle-verified).
- **Suggested C# fix:** for a U- or J-shaped Beta (`α<1` xor `β<1`) the density has no interior mode;
  return the maximising boundary (`min` or `max`) or `NaN`, not the extrapolated formula. Clamp the
  formula result to `[min,max]` at minimum.

## BUG — GeneralizedLogistic L-moment methods divide by zero at κ=0

- **Where:** `Numerics/Distributions/Univariate/GeneralizedLogistic.cs`,
  `LinearMomentsFromParameters` (`1/κ − π/sin(κπ)`) and `ParametersFromLinearMoments`
  (`sin(κπ)/(κπ)`).
- **What:** Neither guards `κ→0`; both evaluate `0/0` at `κ=0`, yielding `NaN`/`Inf`. κ=0 is the
  ordinary Logistic (a valid, common case).
- **Port handling:** **intentional divergence** — the C++ returns the correct L'Hôpital limits
  (`L1=ξ, L2=α, T3=0, T4=1/6`), documented in-header; not oracle-verifiable (C# returns NaN there).
- **Suggested C# fix:** add a `|κ| < NearZero` branch returning those limits.

## BUG — LogPearsonTypeIII.LinearMomentsFromParameters overflows for small skew (large α)

- **Where:** `Numerics/Distributions/Univariate/LogPearsonTypeIII.cs`, `LinearMomentsFromParameters`
  (the `L2 = ... * Gamma.Function(α+0.5)/Gamma.Function(α) ...` line).
- **What:** For small skew, `α = 4/γ²` becomes large; `Gamma.Function(α)` overflows to `+Inf`
  around `α≳171`, so the ratio is `Inf/Inf = NaN`. The **inverse** method
  `ParametersFromLinearMoments` already has an `α≥100` Stirling-approximation branch for exactly this
  ratio; the forward method does not, so the two directions are inconsistent.
- **Port handling:** **intentional divergence** — the C++ adds the matching `α≥100` Stirling branch
  to the forward method (finite, correct), documented in-header; not oracle-verifiable (C# NaN).
- **Suggested C# fix:** apply the same Stirling ratio (`Γ(α+0.5)/Γ(α) ≈ √α·(1 − 1/(8α) + …)`) in
  `LinearMomentsFromParameters` as already done in `ParametersFromLinearMoments`.

## BUG — UnivariateDistributionFactory has no case for VonMises (falls through to Deterministic)

- **Where:** `Numerics/Distributions/Univariate/Base/UnivariateDistributionFactory.cs`,
  `CreateDistribution(UnivariateDistributionType)`.
- **What:** `UnivariateDistributionType.VonMises` has no `case`, so the factory returns a
  `Deterministic` distribution instead — silently wrong for any code that constructs VonMises by type
  (e.g. serialization round-trips, generic UIs).
- **Evidence:** the dotnet oracle emitter had to bypass the factory and `new VonMises()` directly.
- **Port handling:** the C++ factory includes VonMises; the emitter uses a direct-construction bypass.
- **Suggested C# fix:** add the `VonMises` case. **Also audit the factory against the full
  `UnivariateDistributionType` enum** for any other missing entries.

## BUG (pattern) — SetParameters validates before assigning fields (invalid scale reported valid)

- **Where:** `Numerics/Distributions/Univariate/Gumbel.cs`, `SetParameters` / `ValidateParameters`
  ordering (and potentially other distributions using the same field-assignment order).
- **What:** `SetParameters` assigns the location, then validates using the scale field that has **not
  yet been updated** (reads the stale/previous `_alpha`). An invalid incoming scale (`0`, `NaN`,
  negative) can therefore leave `ParametersValid == true`. Location-invalidity is detected correctly;
  scale-invalidity is not, in the affected ordering.
- **Evidence:** scale-invalid `parameters_valid` cases could not be reproduced through the C# oracle
  for Gumbel — the C# reports them valid.
- **Port handling:** the C++ validates the incoming arguments directly (correct), so scale-invalid
  cases are caught; those specific bool cases are simply not oracle-checked.
- **Suggested C# fix:** assign all fields before calling `ValidateParameters`, or validate the passed
  arguments rather than the fields. **Audit every `SetParameters` for this ordering.**

## VERIFY (arguable BUG) — PearsonTypeIII / LogPearsonTypeIII L-skewness sign for negative skew

- **Where:** `Numerics/Distributions/Univariate/PearsonTypeIII.cs` (and `LogPearsonTypeIII.cs`),
  `LinearMomentsFromParameters`.
- **What:** `T3` (L-skewness) is computed from `α = 4/γ²`, which is always positive, and the
  rational approximation yields a **positive** T3 regardless of the sign of the skew `γ`. For a
  negative-skew distribution the L-skewness should be negative. The sign of `γ` appears to be dropped.
- **Port handling:** we initially added a sign flip, then **reverted it to match C# exactly** (an
  early implementer "correction" that broke round-trip consistency with the faithfully-ported
  `ParametersFromLinearMoments`). Kept bug-for-bug; a negative-skew L-moment fixture reproduces the
  C# (positive) value.
- **Suggested action:** confirm with the upstream authors whether T3 is meant to carry the sign of
  γ (likely yes → multiply by `Sign(γ)`), and if so fix both the forward method and ensure the
  inverse method stays consistent.

## CONSISTENCY — StudentT.InverseCDF extreme-tail overflow ignores location/scale

- **Where:** `Numerics/Distributions/Univariate/StudentT.cs`, `InverseCDF` overflow guard.
- **What:** on the extreme-tail overflow path it returns a bare `rflg * double.MaxValue`, without the
  `μ + σ·t` transform applied on the normal return path. For non-default `μ`/`σ` the extreme quantile
  is not on the same scale as the rest of the function.
- **Port handling:** mirrored faithfully.
- **Suggested C# fix:** return `μ + σ · rflg · double.MaxValue` (or `±Infinity`) for consistency.

## CONSISTENCY — CentralMoments(1000) resolves to the int-steps (trapezoidal) overload

- **Where:** e.g. `TruncatedDistribution.cs`, `Mixture.cs` calling `CentralMoments(1000)`; overloads
  in `UnivariateDistributionBase.cs` (`CentralMoments(int steps=300)` vs `CentralMoments(double
  tolerance=1e-8)`).
- **What:** passing the integer literal `1000` binds to the **fixed-step trapezoidal** overload, not
  the adaptive-tolerance one. This may be intentional, but the two overloads with very different
  argument meanings (step count vs tolerance) are an easy foot-gun.
- **Port handling:** the C++ uses the adaptive AdaptiveGaussKronrod integrator; reproduces the C#
  values to fixture tolerance.
- **Suggested action:** verify the intent; consider renaming one overload (e.g.
  `CentralMomentsBySteps` / `CentralMomentsByTolerance`) to remove the ambiguity.

## CONSISTENCY — BivariateEmpirical.SetParameters does not invalidate the cached Bilinear

- **Where:** `Numerics/Distributions/Multivariate/BivariateEmpirical.cs`, `SetParameters` /
  `CDF(double, double)`.
- **What:** `CDF` lazily builds the `bilinear` field only `if (bilinear == null)`. Calling
  `SetParameters` a second time (new grid) after a `CDF` call has already run does not reset
  `bilinear` to null, so subsequent `CDF` calls keep interpolating against the OLD grid.
- **Evidence:** read from source; not exercised by the ported fixture (constructed once, `CDF`
  called several times against the same grid, per `Test_BivariateEmpirical.Test_BivariateEmp`).
- **Port handling:** mirrored faithfully (`bilinear_` is likewise never reset in `set_parameters()`).
- **Suggested C# fix:** set `bilinear = null;` at the end of `SetParameters`.

## CONSISTENCY — Linear vs. Bilinear use different (clamped vs. unclamped) log10 for the Logarithmic transform

- **Where:** `Numerics/Data/Interpolation/Linear.cs` (`Tools.Log10`, clamps values `< 1E-16` to
  `1E-16`) vs. `Numerics/Data/Interpolation/Bilinear.cs` (`Math.Log10` directly, no clamp).
- **What:** the two interpolation classes apply the Logarithmic transform inconsistently: Linear
  is guarded against `log(0)`/`log(negative)` producing `-Inf`/`NaN`; Bilinear is not, despite
  Bilinear internally reusing Linear instances for its search machinery.
- **Port handling:** mirrored faithfully (`Linear::base_interpolate`/`extrapolate` use
  `bestfit::numerics::clamped_log10`; `Bilinear::interpolate` uses plain `std::log10`), documented
  at both call sites.
- **Suggested C# fix:** have `Bilinear` call `Tools.Log10` for consistency, unless the lack of
  clamping there is intentional (e.g. grids are assumed always positive).

## ROBUSTNESS — NoncentralT moments use AdaptiveGaussKronrod (heavy) with no analytic fallback

- **Where:** `Numerics/Distributions/Univariate/NoncentralT.cs`, `Skewness`/`Kurtosis` via
  `CentralMoments`.
- **What:** not a bug, but the moments are pure numerical integration; for large `|λ|` with small
  `ν` the tails are heavy and integration is delicate.
- **Port handling:** the C++ uses a composite Gauss-Legendre quadrature (documented as accurate only
  near-symmetric until it is switched to the now-ported AGK). Only a limitation on the C++ side.
- **Suggested action:** none for C#; noted for context.

## BUG (risk) — MultivariateNormal.COVSRT "permute limits" loop condition is inverted

- **Where:** `Numerics/Distributions/Multivariate/MultivariateNormal.cs`, `COVSRT`, the
  `for (int j = i - 1; j < 0; j--)` loop inside the `CVDIAG <= 0` branch (permute limits/rows when
  a covariance diagonal entry is degenerate).
- **What:** counting DOWN from `j = i - 1` with a `j < 0` continuation condition means the loop body
  only ever runs when `i == 0` (so `j` starts at `-1`, which already satisfies `j < 0`); for every
  `i >= 1` the condition fails immediately and the loop never executes. When it does run (`i == 0`,
  `j == -1`), it immediately indexes `COV[II + j]` with `II == 0`, i.e. `COV[-1]` — in C# this throws
  `IndexOutOfRangeException`. The condition looks like a Fortran `DO j = i-1, 0, -1` mistranslated
  (should be `j >= 0`).
- **Evidence:** static analysis of the loop bounds; not hit by any existing unit test (requires a
  degenerate/near-zero effective covariance diagonal at the very first COVSRT-sorted pivot).
- **Port handling:** the C++ (`core/include/.../multivariate_normal.hpp`, `covsrt`) transcribes the
  loop verbatim but adds a minimal bounds guard immediately before the first `COV[II + j]` access:
  if `II + j < 0` it throws `std::out_of_range` instead of indexing. `std::vector::operator[]`
  performs no bounds check, so the verbatim `i==0` access would otherwise be undefined behavior (a
  heap-corrupting out-of-bounds write) rather than the catchable `IndexOutOfRangeException` C#
  raises there. The guard reproduces C#'s *observable* behavior (a thrown exception on this path)
  without restructuring the loop or any other `covsrt` logic.
- **Suggested C# fix:** change the loop condition to `j >= 0` (or reverse the iteration order to
  match the apparent Fortran intent); add a regression test with a rank-deficient covariance matrix
  that forces the first sorted pivot to be numerically zero.

## COSMETIC — MultivariateNormal.MVNDNT return value is always 0

- **Where:** `Numerics/Distributions/Multivariate/MultivariateNormal.cs`, `MVNDNT`.
- **What:** the local `result` is initialized to `0` and never reassigned anywhere in the method
  body, so `MVNDNT` always returns `0.0`. Its only caller, `MVNDST`, casts this return straight into
  `INFORM` (`INFORM = (int)MVNDNT(...)`), so `INFORM` is always (re)initialized to `0` regardless of
  what `COVSRT`/`MVNLMS`/`BVNMVN` computed; only the `N-INFIS >= 2` branch (via `DKBVRC`) can set it
  to anything else afterward.
- **Port handling:** mirrored faithfully (`mvndnt` always returns `0.0`), documented in-header.
  Harmless in practice — `INFORM` ends up correct for the only case that matters (multi-dimensional
  integration) — but the return value itself is dead code.
- **Suggested C# fix:** either remove the unused return value (change `MVNDNT` to `void`) or wire it
  up if some future INFORM semantics were intended for the `N-INFIS` 0/1 branches.

## COSMETIC — dead variables / heritage artifacts

- `NoncentralT.cs`: a `TT` variable is assigned then never used after the sign flip (a FORTRAN
  translation artifact). Harmless.
- Several distributions declare a member-field initializer (e.g. a scale of `0.0`) that the
  constructor immediately overwrites — harmless but misleading.
- `Numerics/Data/Interpolation/Support/Interpolater.cs`: `deltaStart = Math.Min(1,
  (int)Math.Pow(Count, 0.25))` always evaluates to `1` for any `Count >= 2` (`Math.Pow(2, 0.25)`
  already truncates to `>= 1`, and `Math.Min` caps at `1`), so the "correlated" hunt-vs-bisection
  search heuristic's tolerance is effectively a hardcoded `1`, not scaled with the table size as
  the formula suggests. Ported verbatim (see `core/include/bestfit/numerics/data/interpolation/interpolater.hpp`).

## BUG — ArchimedeanCopula.ValidateParameter never returns null, so ParametersValid is always false

- **Where:** `Numerics/Distributions/Bivariate Copulas/Base/ArchimedeanCopula.cs`,
  `ValidateParameter(double parameter, bool throwException)`.
- **What:** The base `BivariateCopula.Theta` setter is `_parametersValid = ValidateParameter(value,
  false) is null;` (see `BivariateCopula.cs`) -- the C# convention used correctly by
  `NormalCopula`/`StudentTCopula`, whose `ValidateParameter` `return null;` when the parameter is in
  range. `ArchimedeanCopula.ValidateParameter`'s final branch instead does
  `return new ArgumentOutOfRangeException(nameof(Theta), "Parameter is valid");` -- a non-null
  exception object, even though the message says the parameter IS valid. Because `is null` is
  therefore always false, `ParametersValid` is unconditionally `false` for any Archimedean copula
  that does NOT override `ValidateParameter` itself. **UPDATE (Task 8):** this affects Clayton,
  Gumbel, and Joe, but NOT AliMikhailHaq (AMH) or Frank -- `AMHCopula.cs` and `FrankCopula.cs` each
  have their own `ValidateParameter` override that is textually identical to
  `ArchimedeanCopula`'s except the final branch correctly `return null;`, so `AMHCopula`/
  `FrankCopula` instances get a correctly-working `ParametersValid`. (An earlier version of this
  entry said the bug affected "every Archimedean-derived copula (Clayton, AliMikhailHaq, Frank,
  Gumbel, Joe)" -- that blanket claim was wrong for AMH/Frank; corrected here after reading all
  five concrete `.cs` files directly.) This does not affect `PDF`/`CDF`/`InverseCDF` or any fit for
  any copula -- the "valid" branch never throws -- only the `ParametersValid` getter is wrong, and
  only for Clayton/Gumbel/Joe.
- **Evidence (reproduced against the real C# library):** `new ClaytonCopula(2.0).ParametersValid`
  returns `false` even though `theta_minimum = -1`, `theta_maximum = +inf`, and `2.0` is well
  within range; `ValidateParameter(2.0, false).Message` is `"Parameter is valid (Parameter
  'Theta')"` -- a non-null object. `new NormalCopula(0.5).ParametersValid` correctly returns `true`
  for the equivalent in-range case, confirming the divergence is specific to
  `ArchimedeanCopula.ValidateParameter`, not a design choice shared by all copulas. `AMHCopula.cs`/
  `FrankCopula.cs` source inspection (Task 8) confirms their own overrides return `null` in range,
  so `new AMHCopula(0.5).ParametersValid`/`new FrankCopula(5.0).ParametersValid` are expected `true`
  (not independently re-verified against the built library for this entry, since PDF/CDF/fit
  fidelity was the Task 8 verification priority, but the source override is unambiguous).
- **Port handling:** mirrored faithfully. `archimedean_copula.hpp`'s `validate_parameter` still
  returns a non-nullopt "Parameter is valid" message in the final branch (affecting
  `clayton_copula.hpp`/`gumbel_copula.hpp`/`joe_copula.hpp`, which do not override it), but
  `amh_copula.hpp`/`frank_copula.hpp` (Task 8) each add their own correct override (`return
  std::nullopt;` in range), matching their C# counterparts. Documented in-header at
  `bivariate_copula.hpp`, `archimedean_copula.hpp`, and each of the five concrete copula headers.
  No fixture asserts `parameters_valid` on any Archimedean copula since the value is not
  independently informative once the bug (and which copulas it does/doesn't affect) is known.
- **Suggested C# fix:** change `ArchimedeanCopula.ValidateParameter`'s final branch to `return
  null;`, matching `NormalCopula`/`StudentTCopula`/`AMHCopula`/`FrankCopula`, and delete the
  now-redundant overrides on `AMHCopula`/`FrankCopula`. This would flip `ParametersValid` from
  `false` to `true` for every existing in-range Clayton/Gumbel/Joe instance -- audit any downstream
  code (UI validity indicators, `RMC.BestFit` copula fitting) that currently branches on the
  (always-false, for those three types) value before shipping the fix.

---

## CONSISTENCY/API — JoeCopula has no SetThetaFromTau, unlike its Archimedean siblings

- **Where:** `Numerics/Distributions/Bivariate Copulas/JoeCopula.cs`.
- **What:** `ClaytonCopula`, `AMHCopula`, and `GumbelCopula` each implement a `SetThetaFromTau`
  method-of-moments fit (Kendall's tau -> theta, closed-form for Clayton/Gumbel, Brent-solved for
  AMH). `JoeCopula` has no such method -- confirmed by `grep -n SetThetaFromTau` across the entire
  `Numerics/Distributions/Bivariate Copulas/` directory (three hits: Clayton, AMH, Gumbel; zero for
  Joe) and by `Test_JoeCopula.cs` having every other concrete copula's `Test_MOM_Fit` test method
  but not its own. This is not a wrong-output bug (nothing crashes or returns a bad value) --
  it is a missing feature relative to sibling classes that otherwise share an (almost) identical
  API surface, and there is no algorithmic reason Joe's tau could not be Brent-solved the same way
  AMH's is (Joe's generator, like AMH's, has no closed-form tau inversion, but that has not stopped
  the other three).
- **Evidence:** direct inspection of all five `Bivariate Copulas/*.cs` files (Task 8); this is also
  why the Phase 2 plan text and an earlier draft of `fixtures/README.md` incorrectly listed Joe as
  tau-capable (both were apparently written from the class's general shape/expected symmetry with
  Clayton/AMH/Gumbel rather than the actual source) -- corrected in both places during Task 8.
- **Port handling:** `joe_copula.hpp` (Task 8) does NOT add a `set_theta_from_tau` method, matching
  the C# source exactly; `joe_copula.json` has no `"tau"` fixture case, and the three
  `set_theta_from_tau_dispatch` glue functions (`core/tests/test_fixtures.cpp`,
  `bestfitr/src/copula.cpp`, `bestfitpy/src/bindings/copula.cpp`) plus the oracle emitter's
  `SetThetaFromTauDispatch` have no `"Joe"` branch (each has a NOTE comment explaining the
  omission).
- **Suggested C# fix:** add `JoeCopula.SetThetaFromTau`, e.g. `Theta = Brent.Solve(t => { ... } -
  tau, 1d, 100d)` mirroring `GumbelCopula`'s pattern but for Joe's tau relationship, for API parity
  with Clayton/AMH/Gumbel. Not urgent -- MPL/IFM/MLE fits already work for Joe via the shared
  `BivariateCopulaEstimation` path.

---

## CONSISTENCY/API — CompetingRisks' correlated CDF never touches MultivariateNormal.CDF()

- **Where:** `Numerics/Distributions/Univariate/CompetingRisks.cs`, `CDF(double)` /
  `CumulativeIncidenceFunctions`, calling `Numerics/Data/Statistics/Probability.cs`.
- **What:** for the `PerfectlyNegative`/`CorrelationMatrix` dependency modes, `CDF` builds a
  `MultivariateNormal` (`CreateMultivariateNormal()`) and calls `Probability.UnionPCM(cdf,
  _mvn.Covariance)` / `Probability.JointProbability(cdf, ind, _mvn.Covariance)`. The second call's
  3rd argument is a `double[,]` (`_mvn.Covariance`), which only matches the overload
  `JointProbability(IList<double>, int[], double[,]? correlationMatrix = null, DependencyType
  dependency = DependencyType.CorrelationMatrix)` — there is no `JointProbability(IList<double>,
  int[], MultivariateNormal)` overload. With a non-null `correlationMatrix` and the default
  (`CorrelationMatrix`) dependency, that overload unconditionally dispatches to
  `JointProbabilityHPCM` ("Haden Smith's modification of Pandey's Product of Conditional
  Marginals"), never to `JointProbabilityMVN`. `UnionPCM` reaches the same 3-arg overload
  internally for every inclusion-exclusion term. The upshot: the `MultivariateNormal` instance
  `CompetingRisks` constructs is used ONLY to hold/validate a mu/sigma covariance matrix — its
  `.CDF()` (the seeded Genz-Bretz MVNDST quasi-Monte-Carlo integrator for dimension >= 3) is never
  invoked anywhere in `CompetingRisks.cs`. This is surprising given the class name and the
  presence of a full `MultivariateNormal` instance, but is unambiguous from static C# overload
  resolution (confirmed by direct inspection, not runtime reflection, since C# overload binding is
  determined entirely by argument types at compile time).
- **Consequence for the port:** the correlated CDF/PDF paths this task un-defers are fully
  deterministic (no RNG) for any number of components — only `MultivariateNormal.BivariateCDF`
  (Drezner/Genz closed-form bivariate normal CDF) is used. This differs from the original task
  brief's assumption that CompetingRisks reaches "the MVN-backed joint path" (`JointProbabilityMVN`)
  and would need to worry about the C# `MultivariateNormal._MVNUNI` clock-seeded default (the
  concern Task 6's carry-forward note flagged for MultivariateStudentT/MultivariateNormal's own
  `dimension >= 3` `CDF()`, a genuinely different code path). Governed here by "the actual C#
  source over any brief or plan text" (this repo's standing rule): `core/include/bestfit/numerics/
  data/probability.hpp` ports `JointProbabilityHPCM`/`UnionPCM`, not `JointProbabilityMVN`/
  `UnionMVN`, which remain unported (no reachable caller).
- **Port handling:** mirrored faithfully; documented at length in `probability.hpp`'s header
  comment and `competing_risks.hpp`'s CDF comment.
- **Suggested action:** none required (not a bug — HPCM is a legitimate, if approximate,
  alternative to direct MVN-CDF integration) — flagged here purely so a future reader tracing
  "why does CompetingRisks build a MultivariateNormal but never call its CDF" doesn't need to
  re-derive the overload-resolution chain from scratch.

## ROBUSTNESS — JointProbabilityHPCM's `cdf < 1e-300` underflow guard is commented out in cycle 1

- **Where:** `Numerics/Data/Statistics/Probability.cs`, `JointProbabilityHPCM`.
- **What:** the "First cycle" block computes `cdf = Normal.StandardCDF(z1);` and then has a
  commented-out line `//if (cdf < 1e-300) cdf = 1e-300;` immediately before `A = pdf / cdf;` (a
  potential division by a near-zero `cdf`). The "Remaining cycles" loop (only reached when the
  number of correlated events `n >= 3`) recomputes the analogous `cdf` and keeps the identical
  guard ACTIVE (not commented out) right before the same `A = pdf / cdf;` division. This asymmetry
  — a guard present in one loop and disabled (via comment) in a structurally identical one three
  lines apart — looks like an accidental omission (e.g. a guard added later to fix a NaN/Infinity
  seen only in the multi-cycle case, never back-ported to cycle 1) rather than an intentional
  design choice.
- **Evidence:** direct inspection of `Probability.cs`; not hit by any CompetingRisks fixture
  (`R[0,0] = Normal.StandardZ(probabilities[0])` is never so extreme that
  `Normal.StandardCDF(R[0,0])` underflows below `1e-300` for any component/x combination the
  fixtures exercise — the two closest CompetingRisks fixture cases keep `z1` well within a few
  standard deviations of the median).
- **Port handling:** mirrored faithfully (`joint_probability_hpcm` in `probability.hpp` leaves the
  guard commented out in the analogous "First cycle" block, applies it in "Remaining cycles"),
  documented in-header at both the file comment and the function itself.
- **Suggested C# fix:** either add the matching `if (cdf < 1e-300) cdf = 1e-300;` guard to the
  first cycle (for consistency and to avoid a potential `A = pdf / 0` -> `Infinity`/`NaN` if `z1`
  is extreme enough), or confirm the omission is deliberate (e.g. cycle 1's `cdf` is provably
  bounded away from zero by some invariant not obvious from the code) and document why.

## CONSISTENCY — CompetingRisks.CreateMultivariateNormal() zeroes the public CorrelationMatrix as a side effect (PerfectlyNegative only)

- **Where:** `Numerics/Distributions/Univariate/CompetingRisks.cs`, `CreateMultivariateNormal()`.
- **What:** in the `PerfectlyNegative` branch, the method does `CorrelationMatrix = new double[D,
  D];` (assigning a FRESH all-zero `D x D` matrix through the public property setter) and then
  fills a SEPARATE local `sigma` array with the actual synthetic rho matrix
  (`rho = -1/(D-1) + sqrt(ε)`) that gets passed to `new MultivariateNormal(mu, sigma)`. The public
  `CorrelationMatrix` getter therefore reads back all zeros after any CDF/PDF call in
  `PerfectlyNegative` mode — not the rho matrix the MVN's `.Covariance` actually holds. This looks
  like a leftover from refactoring (the zero matrix was probably meant to be a scratch buffer, not
  assigned to the public property) rather than intentional API design.
- **Evidence:** direct inspection; not exercised by any fixture assertion (no fixture reads
  `CorrelationMatrix` back after a `PerfectlyNegative` CDF/PDF call — the fixtures only check
  CDF/PDF/moments values, which are unaffected since the MVN itself uses the correct local
  `sigma`).
- **Port handling:** mirrored faithfully (`create_multivariate_normal()` in `competing_risks.hpp`
  likewise overwrites the mutable `correlation_matrix_` field with zeros in the `PerfectlyNegative`
  branch), documented in-header.
- **Suggested C# fix:** use a local scratch array (e.g. `var scratchCorr = new double[D, D];`)
  instead of assigning through the public `CorrelationMatrix` property, so
  `CorrelationMatrix` retains whatever the caller last set (or `null`) rather than being
  silently zeroed by a `PerfectlyNegative`-mode CDF/PDF evaluation.

## BUG — Histogram.AddData's out-of-range "auto-adapt" branches are unreachable dead code

- **Where:** `Numerics/Data/Statistics/Histogram.cs`, `AddData(double data)`.
- **What:** the XML doc comment promises "If the data value falls outside the range of the
  histogram, the start or end bin will automatically adapt," and the method body has branches
  (`data <= LowerBound` / `data >= UpperBound`) that look like they implement this by widening
  the first/last bin. But the method calls `GetBinIndexOf(data)` **unconditionally**, before
  either branch is checked — and `GetBinIndexOf` itself throws `ArgumentException` for any value
  strictly outside `[_bins.First().LowerBound, _bins.Last().UpperBound]` (which track the same
  bounds as the histogram's own `LowerBound`/`UpperBound`). So a point that would need the
  histogram to "auto-adapt" throws instead, before the adapting branch is ever reached. The two
  branches are only reachable at an **exact boundary match** (`data == LowerBound` or
  `data == UpperBound`), where the "expansion" is a no-op (each sets a bound to the value it
  already equals).
- **Evidence:** direct inspection of the method body's statement order (`SortBins(); int index =
  GetBinIndexOf(data); if (data <= LowerBound) {...}`); confirmed by tracing `GetBinIndexOf`'s own
  guard (`if (value < _bins.First().LowerBound || value > _bins.Last().UpperBound) throw ...`).
  Not exercised by any Test_Histogram.cs test (none call `AddData` a second time with an
  out-of-range point after construction).
- **Port handling:** mirrored faithfully — `histogram.hpp`'s `add_data(double)` calls
  `get_bin_index_of()` first and lets it throw, exactly like the C#; documented in the file
  header and at the call site.
- **Suggested C# fix:** reorder the method to check `data <= LowerBound` / `data >= UpperBound`
  **before** calling `GetBinIndexOf`, so the intended auto-adapt behavior is reachable; or, if the
  auto-adapt behavior was never actually intended (a histogram's bins are usually meant to be
  fixed once constructed), update the doc comment and let `AddData` throw for genuinely
  out-of-range points instead of silently documenting a promise it can't keep.

## BUG — Search.Bisection always returns `start` in descending order (dead-branch comparator)

- **Where:** `Numerics/Data/Interpolation/Support/Search.cs`, `Bisection(double x, IList<double>
  values, int start, SortOrder order)` (all three overloads share the same loop body).
- **What:** the bisection loop's branch condition is `x >= values[xm] && order ==
  SortOrder.Ascending` — a logical AND against the order flag, not the `(x >= values[xm]) ==
  ascending` equality test the algorithm needs to work in both directions (compare
  `Interpolater.cs`'s own `BisectionSearch`, which correctly uses the `==`-style test, or
  `Search.Hunt`'s analogous loop in the same file, which also gets it right via a boolean `ASCND`
  compared with `==`). For `order == SortOrder.Descending`, `order == SortOrder.Ascending` is
  always `false`, so the whole condition is always `false` regardless of `x` — the loop only ever
  shrinks `xhi`, `xlo` never advances past `start`, and `Bisection` returns `start` unconditionally
  instead of the correct bracketing index.
- **Evidence:** direct inspection of the loop body; reproduced independently in a standalone
  Python re-implementation of the exact algorithm during the P3.3 port (a 5-element descending
  array bisected for a midrange value returns `start` regardless of where the value actually
  falls, while `Search.Sequential` on the same inputs returns the correct index).
- **Port handling:** mirrored faithfully (verbatim, not "fixed") — `search.hpp`'s `bisection()`
  keeps the same `&&`-against-`SortOrder::Ascending` condition; documented at length in the file
  header, including a warning that it's dead code for every current caller (Histogram and SNIS
  both only ever call with the default `Ascending` order) but a live bug if a future caller passes
  `Descending`.
- **Suggested C# fix:** change the condition to `(x >= values[xm]) == (order ==
  SortOrder.Ascending)`, matching `Interpolater.BisectionSearch`'s already-correct phrasing of the
  same test.

---

## CONSISTENCY — MCMCSampler.MAP.Fitness is on a different scale than every other chain-state fitness after a successful MAP initialization

- **Where:** `Numerics/Sampling/MCMC/Base/MCMCSampler.cs`, `InitializeChains()`'s `MAP` branch
  (`MAP = DE.BestParameterSet.Clone();`) versus `Sample()`'s output-phase MAP tracking
  (`if (_chainStates[j].Fitness > MAP.Fitness) MAP = _chainStates[j].Clone();`).
- **What:** `DifferentialEvolution.Maximize()` sets `_functionScale = -1` and every fitness it
  records (including `BestParameterSet.Fitness`) is `-1 * LogLikelihoodFunction(x)` -- the
  *scaled* (negated) objective, not the sampler's own unscaled log-likelihood convention. When
  `Initialize == MAP` succeeds, `MAP = DE.BestParameterSet.Clone();` copies that *scaled*
  fitness directly into `MCMCSampler.MAP.Fitness`. For any typical negative log-likelihood
  (the overwhelmingly common case) this makes `MAP.Fitness` a large *positive* number, while
  every `_chainStates[j].Fitness` the output phase compares it against is the sampler's normal
  *unscaled* (negative) log-likelihood. The comparison `_chainStates[j].Fitness > MAP.Fitness`
  is thereby nearly always false, so `MAP` is effectively frozen at the DE estimate for the
  rest of `Sample()` -- the output-phase MAP-tracking loop is live code but practically
  never fires after a successful MAP initialization.
- **Evidence (reproduced against the real C# library):** the `normal_rstan` MCMC fixture case
  (`fixtures/sampling/mcmc/rwmh.json`, `Initialize = MAP`) shows `MCMCResults.MAP.Fitness ==
  473.558...` (positive) while every per-draw chain fitness recorded in the same run is in the
  `[-478, -473]` range (negative) -- confirming the comparison never re-triggers once MAP is
  seeded from `DifferentialEvolution`. A `Randomize`-initialized run (no `DifferentialEvolution`
  in the picture, `MAP` starts at `double.NegativeInfinity` and accumulates normally) does NOT
  exhibit this: its `MAP.Fitness` (`normal_short_exact` case) is a normal, properly-tracked
  negative log-likelihood value.
- **Port handling:** mirrored faithfully -- `mcmc_sampler.hpp`'s `sample()` has the identical
  `chain_states_[j].fitness > map_.fitness` comparison, and `initialize_chains()`'s MAP branch
  copies `de.best_parameter_set()` (also on DE's scaled-fitness convention) into `map_`
  unmodified. Both are documented in-header at the call site; the `normal_rstan` fixture case
  asserts `map_fitness` at its (buggy, positive) value specifically to lock this behavior in,
  not to celebrate it.
- **Suggested C# fix:** either re-scale `DE.BestParameterSet.Fitness` back to the unscaled
  log-likelihood convention when copying it into `MAP` (`MAP = new
  ParameterSet(DE.BestParameterSet.Values, LogLikelihoodFunction(DE.BestParameterSet.Values));`
  recomputes it cleanly), or track a separate scaled/unscaled flag on `ParameterSet` so the
  output-phase comparison in `Sample()` can normalize before comparing. Either fix changes the
  observable `MAP.Fitness` value for every successful-MAP-init run -- coordinate with any
  downstream consumer (`RMC.BestFit`'s Bayesian analysis) that reads it before shipping.

## CONSISTENCY/API — an all-zero RWMH proposal covariance is only safe under `Initialize = MAP`

- **Where:** `Numerics/Sampling/MCMC/RWMH.cs`, `ChainIteration` (`mvn[index].SetParameters(
  state.Values, ProposalSigma.Array)`), via `Numerics/Distributions/Multivariate/
  MultivariateNormal.cs`'s `SetParameters` -> `CholeskyDecomposition` ctor.
- **What:** `RWMH.ChainIteration` calls `SetParameters` with the CURRENT `ProposalSigma` on
  every single iteration. `MultivariateNormal.SetParameters` constructs a
  `CholeskyDecomposition` of that covariance unconditionally, and `CholeskyDecomposition`
  throws for any non-positive-definite input (an all-zero matrix's first diagonal pivot is
  exactly `0`, which fails the decomposition's `sum <= 0` guard). `Test_RWMH_NormalDist_RStan`
  constructs `new RWMH(priors, logLH, new Matrix(2))` -- a literal all-zero 2x2 proposal
  covariance -- and this is harmless ONLY because the test also sets `Initialize = MAP`, and a
  successful MAP initialization's `InitializeCustomSettings()` unconditionally overwrites
  `ProposalSigma` with the Fisher-information-derived covariance BEFORE the first
  `ChainIteration` call. Nothing in the public API prevents constructing (or leaving)
  `ProposalSigma` as all-zero under `Initialize = Randomize` or `UserDefined`, where no such
  override ever happens -- that configuration throws on the very first `ChainIteration`.
- **Evidence (reproduced against the real C# library):** a standalone console app (built
  against `upstream/Numerics/Numerics/Numerics.csproj`) constructing `new RWMH(priors, logLH,
  new Matrix(2))` with `Initialize = Randomize` and calling `Sample()` throws `AggregateException
  ("... Cholesky Decomposition failed. The input matrix is not positive-definite. ...")` (wrapped
  by the `Parallel.For` in `Sample()`) on the very first iteration. The identical construction
  with `Initialize = MAP` (the actual `Test_RWMH_NormalDist_RStan` configuration) succeeds.
  Substituting `Matrix.Identity(2)` for the proposal covariance under `Initialize = Randomize`
  succeeds and reproduces bit-close (~1e-15 relative) between the C++ port and the real C#
  library across all sampled draws.
- **Port handling:** mirrored faithfully -- `MultivariateNormal::set_parameters` throws
  identically for a non-positive-definite covariance (`cholesky_decomposition.hpp`'s `sum <=
  0.0` guard, unchanged from the Phase 2 port). This is not treated as a bug to fix; an
  all-zero proposal covariance is not a meaningful sampler configuration under either language.
  It IS a fixture-authoring hazard worth flagging: the `normal_short_exact` MCMC fixture case
  (`Initialize = Randomize`) uses a `proposal_sigma: "identity"` sentinel instead of the
  upstream test's literal `"zeros"` for exactly this reason -- see the divergence note in
  `fixtures/README.md`'s `mcmc_sampler` section.
- **Suggested action:** none required upstream (working as designed) -- flagged purely so a
  future MCMC-sampler port (ARWMH/DEMCz/DEMCzs/HMC/NUTS/Gibbs/SNIS) doesn't rediscover this the
  hard way when authoring a `Randomize`-initialized fixture case with a degenerate proposal.

## COSMETIC — MCMCSampler.InitializeChains computes an unused midpoint vector in its MAP branch

- **Where:** `Numerics/Sampling/MCMC/Base/MCMCSampler.cs`, `InitializeChains()`'s `MAP` branch:
  `var inititals = lowerBounds.Add(upperBounds).Divide(2d);` (note the typo: `inititals`, not
  `initials`).
- **What:** this local variable is computed (an elementwise midpoint of the prior bounds) but
  never referenced again anywhere in the method -- confirmed by inspection of the full method
  body. `DifferentialEvolution`'s own population initialization (inside `DE.Maximize()`) draws
  from its own `LatinHypercube`-seeded population, not from this vector. Dead code with no
  observable effect (the `Vector.Add`/`Divide` extension calls have no side effects and cannot
  throw for the fixed-length arrays involved here).
- **Port handling:** omitted -- `mcmc_sampler.hpp`'s `initialize_chains()` has a NOTE comment at
  the equivalent location explaining the omission (this port has no `Vector::divide`, since no
  other ported call site needs one; adding it solely to reproduce dead code was judged not
  worthwhile).
- **Suggested C# fix:** delete the unused `inititals` line (and fix the typo if a future
  version does end up needing this midpoint vector for something).

## CONSISTENCY — SNIS sorts its resampling list by `Fitness`, not the `Weight` the surrounding comment/CDF describe; tied `-Infinity` fitness makes the sort order itself unstable across runtimes

- **Where:** `Numerics/Sampling/MCMC/SNIS.cs`, `Sample()`:
  `MarkovChains[0].Sort((x, y) => x.Fitness.CompareTo(y.Fitness));` and the CDF-construction loop
  immediately below it.
- **What:** two related issues at the same call site.
  1. **Sort key mismatch.** The line directly above the sort reads `// Sort list in ascending
     order of posterior weights`, and the very next lines build a CDF by accumulating
     `Math.Max(0.0, MarkovChains[0][i].Weight)` -- i.e. the algorithm's intent, and its
     correctness, depend on the list being sorted by `Weight` (the just-computed normalized
     posterior weight). The comparator actually sorts by `Fitness` (the raw, un-normalized
     log-likelihood/importance weight computed earlier in `Sample()`). For **naive Monte Carlo**
     (no importance distribution supplied), `Weight` and `Fitness` are numerically identical at
     the point of the sort (`weight = logLH` with no `mvn.LogPDF` correction), so this is
     unobservable. **With** an importance distribution (`Weight = Fitness -
     mvn.LogPDF(parameters)`), the two orderings genuinely differ -- the CDF is still
     mathematically valid either way (both `Sort` and the CDF loop iterate the SAME sorted list,
     so `Search.Sequential`'s binary-search precondition -- an ascending CDF -- still holds
     regardless of which key produced the ordering), but the specific `Output[0][i]` a given
     `rndOut[i]` plotting position resolves to differs from what sorting by `Weight` would
     produce.
  2. **Sort-tie instability.** `List<T>.Sort` is .NET's unstable introspective sort. Any model
     with a non-trivial fraction of `-Infinity`-fitness draws (common for a naive/wide-prior SNIS
     configuration, since `LogLikelihood` easily underflows for implausible parameter draws) has
     MANY tied elements at the bottom of the sort. An unstable sort is free to place those tied
     elements in ANY relative order -- which specific `-Infinity` draw lands at output index 0 vs.
     1 vs. ... is not determined by the algorithm's contract, only by the sort implementation's
     internal pivot/partition choices.
- **Evidence (reproduced against the real C# library):** the `fixtures/sampling/mcmc/snis.json`
  fixture's first authoring attempt anchored `chain_value` digest assertions to
  `MarkovChains[0]` indices `[0, 1, 2, 3, 4]` (the natural "first few" choice, matching every
  other MCMC fixture's convention). Every one of those `chain_value` assertions FAILED to
  reproduce against this port's `std::stable_sort`-based C++ (`ctest`'s `test_fixtures`), while
  the corresponding `chain_fitness` assertions (`-Infinity == -Infinity`, order-insensitive by
  construction) PASSED. The `normal_short_exact` case (100 naive-Monte-Carlo draws, wide Uniform
  priors from `Normal().GetParameterConstraints`) has 12 of 100 draws at exactly `-Infinity`
  fitness; the `normal_rstan` case (100000 draws, `Initialize = MAP` concentrating the importance
  distribution near the posterior mode) has only 3 of 100000. Re-anchoring the digest to the
  UNTIED, strictly-monotonic-fitness tail (the top 5 indices of each sorted list) reproduces
  cleanly (`normal_rstan` chain companions to ~1e-8 relative via the MAP/DE/Hessian path, per the
  usual P3.5 tolerance policy; `normal_short_exact`'s naive-Monte-Carlo companions to ~1e-15
  relative).
- **Port handling:** both aspects mirrored faithfully, not fixed -- `snis.hpp`'s `sample()` sorts
  by `.fitness` (not `.weight`), exactly matching the C# comparator's actual (not commented)
  behavior, using `std::stable_sort` (this port's usual convention for reproducing an unstable C#
  `List<T>.Sort`'s comparator -- see `mcmc_sampler.hpp`'s own `stable_sort` note for the same
  precedent in `InitializeChains()`). `stable_sort` does NOT make the two languages' outputs
  agree on tied-element ordering (a stable sort's tie-breaking is "preserve original order",
  which is only meaningful if BOTH languages process elements in the same original order AND use
  a stable sort -- true for C++ here, false for C#'s `List<T>.Sort`). `fixtures/README.md`'s
  SNIS tolerance-policy section documents the resulting draw-index hazard for future fixture
  authors.
- **Suggested C# fix:** for (1), either fix the comment to describe what the code does (sort by
  `Fitness`) or change the comparator to `x.Weight.CompareTo(y.Weight)` to match the comment and
  the CDF loop's own variable name -- these are NOT equivalent when an importance distribution is
  supplied, so this is a real behavioral choice, not just a comment fix, and should be resolved
  with the library's intent for how the resampled `Output` list should be ordered/weighted. For
  (2), switch to a stable sort (`OrderBy(x => x.Fitness).ToList()` or an explicit stable
  merge-sort) if bit-reproducible resampling across runs/platforms is a design goal; otherwise
  document that `Output`'s specific draw-to-plotting-position mapping is order-nondeterministic
  whenever tied fitness values occur.

## CONSISTENCY — Gibbs's conjugate Normal-posterior-mean formula has a `mu0 / 2` term instead of the textbook `mu0 / sigma0^2`

- **Where:** `Numerics/Sampling/MCMC/Test_Gibbs.cs`, `Test_Gibbs_NormalDist_RStan`'s local
  `proposal` closure: `double mun = (n * mu + mu0 / 2) / (n + 1 / (sigma0 * sigma0));` (the test's
  own inline proposal function -- there is no shared library implementation of this conjugate
  update; `Gibbs.cs` itself only calls whatever `Proposal` delegate the caller supplies).
- **What:** the standard closed-form posterior mean for a Normal likelihood with known variance
  `sigma^2` and a Normal(`mu0`, `sigma0^2`) conjugate prior on the mean is `mu_n = (n * xbar /
  sigma^2 + mu0 / sigma0^2) / (n / sigma^2 + 1 / sigma0^2)`. The test's formula instead computes
  `mun = (n * mu + mu0 / 2) / (n + 1 / sigma0^2)` -- it (a) omits the `/ sigma^2` scaling on the
  `n * xbar` numerator term and the `n` denominator term entirely (the companion `sigma2` line
  just above it DOES correctly compute the analogous posterior-variance formula, `sigma2 =
  x[1]^2 / (n + x[1]^2 / sigma0^2)`, so the omission is specific to the mean formula, not a
  wholesale simplification), and (b) uses `mu0 / 2` where the textbook formula has `mu0 /
  sigma0^2`. Because the test's own data uses `mu0 = 0`, term (b) vanishes identically regardless
  of which coefficient multiplies it, and because `sigma0 = 5e5` is enormous, `1 / sigma0^2 ~
  4e-12` is negligible next to `n = 48` in the denominator either way -- so the test's rstan
  comparison (which only checks the RESULTING posterior summary statistics to 5% tolerance, not
  the formula's algebraic form) cannot distinguish this transcription from the textbook one; both
  degenerate to `mun ~ mu` (the sample mean) in this near-noninformative-prior limit. This is
  therefore unverified/unverifiable from the test alone whether it's a genuine library bug or an
  intentional simplification for this specific (`mu0 = 0`) worked example -- flagged as
  CONSISTENCY rather than BUG for that reason.
- **Port handling:** transcribed verbatim into `model_registry.hpp`'s `"normal_conjugate_gibbs"`
  proposal closure (`double mun = (n * mu + mu0 / 2.0) / (n + 1.0 / (sigma0 * sigma0));`) -- this
  is the oracle-governing rule (C# source, including this specific test's inline formula, governs
  over what a textbook derivation "should" say), and the `gibbs.json` fixture's curated
  `chain_value` digests (draws 0-4, including the mutated-prior-path draws >= 1) reproduce this
  exact formula bit-for-bit against the real C# library at `rel: 1e-12`.
- **Suggested action:** if a future caller of this conjugate-Gibbs pattern uses a non-zero
  `mu0`, re-derive/re-verify the formula against the textbook conjugate-Normal update before
  reusing this test's `proposal` closure as a template -- the `mu0 / 2` coefficient will NOT
  degenerate away in that case and would materially bias the posterior mean draws.

## CONSISTENCY — NUTS's step-size heuristic bypasses a caller-supplied custom `GradientFunction`

- **Where:** `Numerics/Sampling/MCMC/NUTS.cs`, `LeapfrogInPlace` (called only by
  `FindReasonableEpsilon`/`TrySingleStepLogAcceptance`, i.e. the Hoffman & Gelman 2014 Algorithm 4
  step-size-search heuristic run once at chain initialization and again after every mass-matrix
  adaptation-window update).
- **What:** every ACTUAL trajectory step in `BuildTree` goes through `Leapfrog`, which correctly
  calls `GradientFunction(...)` -- the (possibly caller-supplied) gradient delegate stored on the
  instance. `LeapfrogInPlace`, used only by the step-size-search heuristic, instead calls
  `NumericalDerivative.Gradient(...)` DIRECTLY, hardcoding a finite-difference gradient regardless
  of what `gradientFunction` the constructor was given. A caller who supplies an exact analytic
  `gradientFunction` (to avoid finite-difference cost/noise entirely) still gets a
  finite-difference-based initial step size and every post-adaptation-window re-tuned step size --
  only the trajectory itself uses their analytic gradient. This does not affect correctness of the
  sampled trajectory (the step size is just a tuning heuristic, not part of the target
  distribution), but it is a surprising, easy-to-miss asymmetry between two call sites that both
  claim to leapfrog-integrate "the" gradient.
- **Evidence:** direct code reading; both call sites are reproduced verbatim in this port's
  `nuts.hpp` (`leapfrog_in_place()` calls `diff::gradient(...)` directly, `leapfrog()` calls
  `gradient_function_(...)`) -- see that file's header comment. Unexercised by any fixture (every
  `nuts.json`/`hmc.json` case uses the default finite-difference gradient, so the two code paths
  are numerically identical in every case actually tested here).
- **Port handling:** mirrored faithfully -- both C++ call sites reproduce the C# asymmetry exactly.
- **Suggested C# fix:** route `LeapfrogInPlace` through `GradientFunction` too, so a custom gradient
  is honored everywhere the class claims to use "the" gradient function.

## CONSISTENCY — `NextDoubles(length, dimension)` draws each column from its own fresh sub-`MersenneTwister`, not the caller's stream

- **Where:** `Numerics/Utilities/ExtensionMethods.cs`, `NextDoubles(this Random random, int
  length, int dimension)`.
- **What:** the 1-D overload (`NextDoubles(this Random random, int length)`) draws `length`
  values straight off the caller's own stream, as expected. The 2-D overload does something
  different: for each of the `dimension` columns it draws exactly ONE value off the caller's
  stream (`random.Next()`) to seed a brand-new `MersenneTwister` (or plain `Random`, if the
  caller wasn't itself a `MersenneTwister`), then fills that entire column by advancing the
  FRESH sub-generator's own stream `length` times. The caller's stream is therefore consumed at
  a rate of exactly `dimension` draws total (one per column, to seed the sub-generators), not
  `length * dimension` -- every actual random double returned comes from one of the `dimension`
  independent sub-streams, never from the parent stream directly. This is a real behavioral
  choice (not obviously a mistake -- it decorrelates columns even when the parent stream has a
  short period or column-wise correlation), but it is easy to miss reading only the method
  signature: a caller expecting "draw `length * dimension` numbers off my stream in row-major
  order" (the naive reading `NextDouble()` in a nested loop would produce) gets a materially
  different, though still uniform, output.
- **Evidence:** direct code reading (`ExtensionMethods.cs` lines ~144-157); `Test_NextDoubles2D`
  (`Test_Numerics/Utilities/Test_ExtensionMethods.cs`) only range-checks the output (`[0, 1)`
  for every cell), so it does not itself distinguish this from the naive single-stream reading --
  the sub-stream-per-column behavior was confirmed by tracing a seeded `MersenneTwister(12345)`
  through both this method and a column-by-column reconstruction using `new
  MersenneTwister(random.Next())` per column, which reproduce identically.
- **Port handling:** transcribed exactly -- `extension_methods.hpp`'s `next_doubles(rng, n, dim)`
  overload constructs one `bestfit::numerics::sampling::MersenneTwister sub(random.next())` per
  column and fills that column from `sub`, in dimension order (see the header's file comment).
  This pattern is load-bearing for `SNIS::sample()`, which calls
  `_masterPRNG.NextDoubles(Iterations, NumberOfParameters)` once up front; `fixtures/special_
  functions/extension_methods.json`'s `next_doubles_grid` cases lock the exact per-cell values
  this produces from a known seed, independently of any MCMC sampler fixture.
- **Suggested C# fix:** none required (working as designed) -- flagged purely as a
  non-obvious-from-the-signature quirk for anyone reusing this overload outside the ported
  call sites (SNIS is the only current consumer within this port's scope).

## ROBUSTNESS — `Bootstrap.ComputeAccelerationConstants`'s `Tools.ParallelAdd` reduction is not bit-reproducible run-to-run

- **Where:** `Numerics/Sampling/Bootstrap/Bootstrap.cs`, `ComputeAccelerationConstants` (its
  `Parallel.For(0, N, idx => { ... Tools.ParallelAdd(ref I2[i], diff * diff); Tools.ParallelAdd
  (ref I3[i], diff * diff * diff); ... })` loop), backed by `Numerics/Utilities/Tools.cs`'s
  `ParallelAdd` (a CAS retry loop over `Interlocked.CompareExchange`).
- **What:** `ParallelAdd` is a correct lock-free accumulator (no lost updates), but it does NOT
  fix the ORDER in which concurrent jackknife-sample contributions land in `I2[i]`/`I3[i]` --
  that order depends on the .NET thread pool's scheduling of the `Parallel.For` partitions, which
  is not guaranteed deterministic across runs, machines, or core counts. Floating-point addition
  is not associative, so a different accumulation order can (in general) produce a different
  last-few-bits sum, even though every run adds the exact same set of addends. The resulting BCa
  acceleration constant, and therefore the BCa confidence interval bounds, inherit this
  run-to-run variability.
- **Evidence (reproduced against the real C# library):** the oracle emitter's `--dump` output for
  the `bca` bootstrap fixture case was captured across four independent runs of the SAME process
  invocation, sequentially, on the development machine, and diffed byte-for-byte -- all four
  runs were BIT-IDENTICAL (a low-core-count environment apparently schedules this small,
  100-jackknife-sample `Parallel.For` deterministically in practice), i.e. the measured wobble
  was exactly `0` on this machine, though the reduction remains order-dependent BY CONSTRUCTION
  and a different core count/thread-pool configuration/.NET version could legitimately produce a
  different summation order and a different last-few-bits result.
- **Port handling:** this port replaces the `Parallel.For` + `Tools.ParallelAdd` pair with a
  plain serial accumulation in jackknife-index order (see `bootstrap.hpp`'s file header BCa
  HAZARD note and `compute_acceleration_constants`'s own comment) -- deterministic within the
  C++ port, but not a bit-for-bit reproduction of C#'s reduction order. The `bca` fixture case's
  CI-bound assertions therefore use a LOOSE `mode: "rel", tol: 1e-6` (three orders of magnitude
  looser than every other CI method's `1e-9`), sized to the reduction's inherent
  order-dependence rather than to any measured instability (which was zero on this machine) --
  see `fixtures/README.md`'s `bootstrap` schema section for the full tolerance rationale.
- **Suggested C# fix:** none required upstream for correctness (the CAS loop is race-free); if
  bit-reproducible BCa intervals across runs/machines becomes a design goal, replace the
  `Parallel.For`/`ParallelAdd` pair with a deterministic-order reduction (e.g. `Parallel.For`
  into per-partition local accumulators, combined in a fixed final pass) or a plain serial loop.

---

## BUG — CS0104 ambiguous `YeoJohnsonLink` blocks the entire `RMC.BestFit` assembly from compiling

- **Where:** `Analyses/Univariate/Bulletin17CAnalysis.cs` (~lines 2132, 2144).
- **What:** both `RMC.BestFit.Models.LinkFunctions.YeoJohnsonLink` and `Numerics.Functions.
  YeoJohnsonLink` are `using`-imported in this file; two unqualified `new YeoJohnsonLink(...)`
  constructor calls are therefore ambiguous (CS0104), which fails compilation of the entire
  `RMC.BestFit` assembly, not just this file.
- **Evidence:** `tools/oracle_emitter` (Task T12) had to work around this to compile any real C#
  estimator code at all.
- **Port handling:** the oracle emitter subset-compiles only `Estimation/**` + `Models/**`
  (excluding `Analyses/**` -- the two GMM/Bulletin17C files -- and GMM) against the clean
  Numerics build, so the ambiguous file is never compiled. This is a build workaround, not a port
  of the file; `Bulletin17CAnalysis`/GMM remain unported (severed, tracked separately).
- **Suggested C# fix:** fully-qualify one of the two `new YeoJohnsonLink(...)` calls (e.g. `new
  Numerics.Functions.YeoJohnsonLink(...)` or `new RMC.BestFit.Models.LinkFunctions.
  YeoJohnsonLink(...)`), or remove one of the two conflicting `using` directives from the file.

## ROBUSTNESS — DIC / WAIC / PSIS-LOO parallel-reduction non-reproducibility (extends the BCa `Tools.ParallelAdd` finding)

- **Where:** `RMC.BestFit`'s `BayesianAnalysis.ComputeDIC` (and the population sampler's pooled
  `Output` accumulation), following the same `Parallel.For`/`Tools.ParallelAdd` pattern as the
  `Bootstrap.ComputeAccelerationConstants` finding above.
- **What:** like the BCa acceleration constant, DIC's parallel reduction over posterior draws is
  not bit-reproducible C#-to-C# run-to-run (~1e-13 relative), for the same reason:
  `ParallelAdd`'s CAS-retry loop is race-free but not order-fixed, and floating-point addition is
  not associative.
- **Evidence:** measured during Task T12's oracle verification of the `bayes_normal` fixture; the
  fixture's DIC tolerance (`rel: 1e-6`) is sized to this reduction-order noise, not to any
  C++-vs-C# divergence.
- **Port handling:** this port computes DIC/WAIC/LOOIC with a plain serial reduction
  (deterministic within C++, consistent with the same choice made for BCa's acceleration
  constant), documented at the relevant `bayesian_analysis.hpp` diagnostics code and in
  `fixtures/README.md`'s tolerance-policy notes.
- **Suggested C# fix:** none required for correctness; if bit-reproducible DIC/WAIC across runs
  is a design goal, replace the `Parallel.For`/`ParallelAdd` reduction with a deterministic-order
  accumulation, matching the suggested fix for the BCa finding above.

## BUG (latent, untested) — UnivariateDistribution's Jeffreys-scale prior indexes `GetParameters[1]`, which throws for single-parameter families

- **Where:** `Numerics/Distributions/Univariate/Base/UnivariateDistribution.cs`,
  `Prior_LogLikelihood` (~1822-1843), under `UseJeffreysRuleForScale`.
- **What:** the scale-parameter lookup is `GetParameters[1]` for every family except Gamma/Weibull
  (which use index 0). For a genuine one-parameter family -- Poisson, Bernoulli, Geometric,
  Deterministic -- there is no `GetParameters[1]`; indexing it would throw
  `IndexOutOfRangeException`. This path is only reached if a MAP or Bayesian estimation is
  actually run against a one-parameter model with the (default-true) `UseJeffreysRuleForScale`
  flag set -- untested upstream (no `Test_UnivariateDistribution.cs` case constructs MAP/Bayesian
  against a 1-parameter family) and flood-frequency-irrelevant in practice (GEV/LP3/etc. all have
  two or more parameters).
- **Evidence:** static inspection during Tasks T12/T13 while porting the Jeffreys 1/scale prior
  (`UnivariateDistributionModel::prior_log_likelihood`, `core/include/bestfit/models/
  univariate_distribution_model.hpp`); not independently reproduced against the real C# library
  (no fixture exercises a 1-parameter family under MAP/Bayesian).
- **Port handling:** **intentional divergence** -- the C++ port's `scale_parameter_index()`
  returns 1 for these families same as C#, but the caller guards `scale_index < p.size()` and
  silently skips the Jeffreys term instead of indexing out of range (see the code comment at that
  guard). No crash, no oracle case exists either way.
- **Suggested C# fix:** guard `GetParameters[1]` (e.g. `if (GetParameters.Length > 1)`) and skip
  the Jeffreys scale term for one-parameter families, matching the C++ port's behavior; add a
  regression test constructing MAP/Bayesian against Poisson/Bernoulli/Geometric/Deterministic
  with `UseJeffreysRuleForScale = true`.

## BUG — MixtureModel.Clone() strips the cloned Mixture's zero-inflation while the cloned model still reports IsZeroInflated

- **Where:** `RMC.BestFit/Models/UnivariateDistribution/MixtureModel.cs`, `Clone()` (~line 1276),
  interacting with the `Mixture` property setter (~line 242), the `IsZeroInflated` property
  setter (~line 285), and the `(DataFrame, Mixture)` constructor (~line 53).
- **What:** `Clone()` builds `new MixtureModel(DataFrame, Mixture!) { _isZeroInflated =
  IsZeroInflated, ... }`. The constructor runs `Mixture = (Mixture)distribution.Clone();` through
  the `Mixture` property setter. Numerics' `Mixture.Clone()` itself correctly copies
  `IsZeroInflated`/`ZeroWeight` (Mixture.cs ~line 1019), but at that moment the fresh model's
  `_isZeroInflated` field still holds its default `false`, so the setter immediately overwrites
  the cloned distribution with `IsZeroInflated = false; ZeroWeight = 0.0;`. The object
  initializer then writes `_isZeroInflated = IsZeroInflated` DIRECTLY to the private field --
  bypassing the `IsZeroInflated` property setter, the only code that would re-sync the
  distribution. End state when cloning a zero-inflated model: the cloned MODEL reports
  `IsZeroInflated == true` while its underlying `Mixture` has `IsZeroInflated == false,
  ZeroWeight == 0` -- the clone's likelihood/PDF/CDF surface silently loses the zero-inflated
  mass while claiming to have it.
- **Evidence:** static inspection of the three members during Task M10 (the setter/field-write
  ordering is unambiguous from the source); no upstream `MixtureModelTests` method clones a
  zero-inflated model and asserts the distribution's state, so no C# test observes it.
- **Port handling:** **intentional divergence** -- `mixture_model.hpp`'s `clone()` re-syncs the
  cloned mixture's zero-inflation state from the original after the field writes, so the clone
  ends in the same effective state as the original. Documented in the file header and pinned by
  the extra checks in `test_clone_preserves_is_zero_inflated`.
- **Suggested C# fix:** re-apply the zero-inflation to the cloned `Mixture` after the initializer
  (or assign through the public `IsZeroInflated` property rather than the `_isZeroInflated`
  field, minding its side effects); add a regression test that clones a zero-inflated model and
  asserts the cloned distribution's `IsZeroInflated`/`ZeroWeight`.

## ROBUSTNESS — DataFrame.ProcessThresholdSeries is destructive and not idempotent when explicit points exactly cover a threshold window

- **Where:** `RMC.BestFit/Models/DataFrame/DataFrame.cs`, `ProcessThresholdSeries()` (~line 618).
- **What:** the method reads the STORED `thresholdData.NumberAbove`, computes `nBelow = Duration
  - nAbove - (explicit interval/uncertain/exact points inside the window)`, then writes both
  counts back: `NumberAbove = nBelow == 0 ? 0 : nAbove; NumberBelow = Math.Max(0, nBelow);`.
  Because the recomputation consumes its own previous output, the zeroing branch is destructive:
  when the explicit points exactly account for `Duration - NumberAbove` (first run: `nBelow ==
  0`, so `NumberAbove` is zeroed and `NumberBelow` set to 0), a SECOND run starts from `nAbove =
  0` and computes `nBelow = Duration - overlaps` -- flipping `NumberBelow` from 0 to the original
  `NumberAbove`, i.e. years the first pass classified as above-threshold are re-counted as
  censored-below years. Upstream this method re-runs constantly: every series
  `CollectionChanged`/item `PropertyChanged` event triggers it, and `CalculatePlottingPositions()`
  calls it unconditionally as its first step (~lines 1142-1144) -- so in the exact-coverage edge
  case any later mutation or a repeated plotting-positions call silently corrupts the threshold
  likelihood counts.
- **Evidence:** static inspection of the read-modify-write during Task M4 (report concern #1);
  no upstream test hits the exact-coverage-then-rerun sequence, and no ported fixture exercises
  the edge either.
- **Port handling:** mirrored faithfully -- `data_frame.hpp`'s `process_threshold_series()` is
  the same read-modify-write, and `calculate_plotting_positions()` calls it first exactly like
  the C#. The C++ replaces the INPC auto-trigger with the documented explicit "call once after
  mutations" cadence (see the file-header invalidation strategy), which matches the C#
  once-per-mutation event cadence.
- **Suggested C# fix:** make the pass idempotent by recomputing from immutable inputs -- retain
  the originally supplied `NumberAbove` (e.g. a private `_originalNumberAbove` set on
  construction/assignment) and derive both published counts from it on every pass; add a
  regression test that processes a fully covered threshold window twice and asserts stable
  counts.

---

## How to work this list later

1. Reproduce each finding directly against the pinned upstream (`dotnet test` a targeted case, or a
   tiny console snippet), confirming the C# behaviour.
2. For each confirmed bug, decide: patch upstream (PR to USACE-RMC) vs. keep the intentional C++
   divergence documented. Any upstream fix that changes an oracle value must be paired with updated
   test literals and a re-run of `tools/verify_oracles.py`.
3. The two intentional C++ divergences already in place (GeneralizedLogistic κ→0, LogPearsonTypeIII
   large-α) become non-divergences once/if the C# is fixed — revisit their in-header notes then.
