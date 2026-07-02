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

## How to work this list later

1. Reproduce each finding directly against the pinned upstream (`dotnet test` a targeted case, or a
   tiny console snippet), confirming the C# behaviour.
2. For each confirmed bug, decide: patch upstream (PR to USACE-RMC) vs. keep the intentional C++
   divergence documented. Any upstream fix that changes an oracle value must be paired with updated
   test literals and a re-run of `tools/verify_oracles.py`.
3. The two intentional C++ divergences already in place (GeneralizedLogistic κ→0, LogPearsonTypeIII
   large-α) become non-divergences once/if the C# is fixed — revisit their in-header notes then.
