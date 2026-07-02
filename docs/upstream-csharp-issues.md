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

---

## How to work this list later

1. Reproduce each finding directly against the pinned upstream (`dotnet test` a targeted case, or a
   tiny console snippet), confirming the C# behaviour.
2. For each confirmed bug, decide: patch upstream (PR to USACE-RMC) vs. keep the intentional C++
   divergence documented. Any upstream fix that changes an oracle value must be paired with updated
   test literals and a re-run of `tools/verify_oracles.py`.
3. The two intentional C++ divergences already in place (GeneralizedLogistic κ→0, LogPearsonTypeIII
   large-α) become non-divergences once/if the C# is fixed — revisit their in-header notes then.
