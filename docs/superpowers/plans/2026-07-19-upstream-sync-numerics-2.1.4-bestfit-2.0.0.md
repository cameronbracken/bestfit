# Upstream Sync Implementation Plan: Numerics v2.1.4 + RMC-BestFit v2.0.0

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore 1:1 parity with Numerics v2.1.4 (`2a0357a`) and RMC-BestFit v2.0.0
(`c2e6192`), re-pinning every affected oracle to the new C# behavior.

**Architecture:** Bump-first, oracle-driven. Task 0 bumps both submodules and repairs the
oracle emitter, then runs the gate against unmodified fixtures to produce the empirical
failure census. Numerics tasks (1-11) port in dependency order before BestFit tasks (12-20)
that consume them. Every task is TDD against fixtures: re-pin or add the fixture case first
(fail), port the C++ (pass in all runners), verify with the dotnet gate, commit.

**Tech Stack:** C++17 header-only core, cpp11 (R), pybind11 (Python), dotnet 10 oracle
emitter, JSON fixtures.

**Spec:** `docs/superpowers/specs/2026-07-19-upstream-sync-numerics-2.1.4-bestfit-2.0.0-design.md`
(read it before starting any task; its scope tables are the authority on what changed).

## Global Constraints

- Work in this worktree on branch `upstream-sync-2026-07`. Commit per task; do NOT push.
- Commit messages: `<type>: <description>` (feat/fix/refactor/docs/test/chore). No
  Co-Authored-By trailers. GPG signing is automatic.
- Structural mirroring: C++ mirrors the C# file/class/method layout. The C# diff is
  normative; when in doubt, transcribe. Read the diff with
  `git -C upstream/Numerics diff a2c4dbf..v2.1.4 -- <path>` or
  `git -C upstream/RMC-BestFit diff fc28c0c..v2.0.0 -- <path>`.
- Every touched C++ file updates its provenance header to the new SHA:
  `// ported from: <csharp path> @ 2a0357a` (Numerics) or `@ c2e6192` (BestFit).
- Oracle values live ONLY in `fixtures/*.json`. Never hardcode expected values in test
  files. New cases may be authored with placeholder `expected` values and harvested via
  `python3 tools/verify_oracles.py --dump` where the emitter supports the target; otherwise
  curate from the cited C# test and confirm with a normal gate run.
- No `M_PI` (use `corehydro::numerics::kPi`); no namespace alias named `gamma` or `stat`;
  no Eigen or any external C++ dependency.
- After any class-layout change, R rebuilds MUST use `R CMD INSTALL --preclean corehydror`.
- Only re-run `Rscript -e 'cpp11::cpp_register("corehydror")'` if `corehydror/src/*.cpp`
  changed (header-only core changes don't need it).
- Build/test commands (from the worktree root):
  - C++: `cmake --build core/build -j8 && ctest --test-dir core/build --output-on-failure`
  - Fixture runner only: `ctest --test-dir core/build -R fixtures --output-on-failure`
  - R: `R CMD INSTALL --preclean corehydror && Rscript -e 'testthat::test_local("corehydror")'`
  - Python: `pixi run python -m pip install --force-reinstall --no-deps ./corehydropy && pixi run python -m pytest corehydropy/tests -q`
  - Oracle gate: `python3 tools/verify_oracles.py` (add `--fixtures fixtures/<subdir>` to
    target a subset while iterating)
- The R and Python suites are slow. Within a task iterate on C++ + the targeted oracle
  gate; run the full R + Python suites once before the task's final commit.
- Fidelity policy: mirror the NEW C# exactly, including any remaining quirks. Where the C++
  carried a documented intentional divergence that upstream has now adopted, converge to
  upstream's exact formulation and retire the divergence note in the file header.
- Deterministic surfaces pin exact oracles. Seeded MCMC short-chain curves keep their
  existing structural-invariant assertions (do not loosen tolerances, do not add
  oracle_skip). B17C bootstrap replicate streams are seeded and deterministic: pin exact.

---

### Task 0: Submodule bump, emitter repair, oracle failure census

**Files:**
- Modify: `upstream/Numerics` (gitlink -> `2a0357a`, tag v2.1.4)
- Modify: `upstream/RMC-BestFit` (gitlink -> `c2e6192`, tag v2.0.0)
- Delete: `tools/oracle_emitter/patched/Bulletin17CAnalysis.cs`
- Modify: `tools/oracle_emitter/OracleEmitter.csproj` (drop the CS0104 patch lines)
- Create: `docs/superpowers/plans/2026-07-19-upstream-sync-oracle-census.md`

**Interfaces:**
- Produces: submodules at the release tags (all later tasks diff against them); a working
  emitter build; the census file every later task consults for its expected-failure list.

- [ ] **Step 1: Bump both submodules to the release tags**

```bash
git -C upstream/Numerics checkout v2.1.4
git -C upstream/RMC-BestFit checkout v2.0.0
git add upstream/Numerics upstream/RMC-BestFit
git commit -m "chore: bump upstream submodules to Numerics v2.1.4 and RMC-BestFit v2.0.0"
```

- [ ] **Step 2: Remove the stale CS0104 patch**

Upstream deleted `src/RMC.BestFit/Models/LinkFunctions/YeoJohnsonLink.cs` (commit `68b07a8`),
which removes the ambiguity the patch worked around. In `OracleEmitter.csproj` delete these
two lines (near line 100) and the explanatory comment block above them:

```xml
<Compile Remove="$(BF)/Analyses/Univariate/Bulletin17CAnalysis.cs" />
<Compile Include="patched/Bulletin17CAnalysis.cs" />
```

Then `rm tools/oracle_emitter/patched/Bulletin17CAnalysis.cs` (and the now-empty `patched/`
directory).

- [ ] **Step 3: Build the emitter; fix compile breaks**

```bash
dotnet build tools/oracle_emitter -c Release
```

Expected breaks to handle (fix minimally, keep the subset-compile shape):
- `Bulletin17CAnalysis.cs` now references `AnalysisProgress` (new file
  `src/RMC.BestFit/Analyses/Support/AnalysisProgress.cs`): already covered by the
  `Analyses/Support/**` glob; verify.
- Any new `using` targets from the v2 restructure. The emitter compiles BestFit sources
  directly against the local Numerics ProjectReference, so the csproj's new
  `RMC.Numerics` PackageReference does NOT apply; keep the ProjectReference.
- If a genuinely new CS error appears, record it in the census file and patch the same way
  the old CS0104 was handled (local copy + Compile Remove), documenting why.

- [ ] **Step 4: Run the gate against unmodified fixtures; record the census**

```bash
python3 tools/verify_oracles.py 2>&1 | tee /tmp/oracle-census-raw.txt
```

Expected: MANY failures (this is the point). Exit code 1.

Write `docs/superpowers/plans/2026-07-19-upstream-sync-oracle-census.md` containing: the
summary counts (reproduced / failed / skipped), the failing fixture files grouped by
subsystem, and a short cross-check against the spec's risk lists — flag any failure the
spec did NOT predict and any predicted-risk area with NO failure (both are findings, not
errors; e.g. copula `parameters_valid` flips appear only if a fixture asserts them).

- [ ] **Step 5: Sanity-check that the C++/R/Python suites are untouched**

```bash
ctest --test-dir core/build -R fixtures --output-on-failure
```

Expected: PASS (the C++ still matches the OLD fixtures; only the C# moved).

- [ ] **Step 6: Commit**

```bash
git add -A tools/oracle_emitter docs/superpowers/plans/2026-07-19-upstream-sync-oracle-census.md
git commit -m "chore: repair oracle emitter for v2.0.0 (CS0104 patch obsolete); record pre-port oracle census"
```

---

### Task 1: Numerics small utilities (Tools, Search, Bilinear, Histogram, Probability, RunningStatistics)

**Files:**
- Modify: `core/include/corehydro/numerics/tools.hpp`
- Modify: `core/include/corehydro/numerics/data/interpolation/search.hpp`
- Modify: `core/include/corehydro/numerics/data/interpolation/bilinear.hpp`
- Modify: `core/include/corehydro/numerics/data/histogram.hpp`
- Modify: `core/include/corehydro/numerics/data/probability.hpp`
- Modify: `core/include/corehydro/numerics/data/running_statistics.hpp`
- Modify: fixtures under `fixtures/data/` (new cases; find the exact files by target via
  `grep -rl '"target"' fixtures/data/`)

**C# refs (all `git -C upstream/Numerics diff a2c4dbf..v2.1.4 -- <path>`):**
- `Numerics/Utilities/Tools.cs` (33dc1af, 9b4af52): `Log10` guard becomes `x >= 0d`
  (Log10(NaN) no longer throws; NaN propagates). Ignore the `ParallelAdd` hunks (not ported;
  serial reductions by design — note this in the file header).
- `Numerics/Data/Interpolation/Support/Search.cs` (33dc1af): descending-order
  Bisection/SequentialSearch/Hunt split into ascending/descending loops with
  `(x >= v[m]) == ascending` comparators. The C++ mirrors the OLD dead-branch bug — replace
  it with the fixed logic and delete the divergence note.
- `Numerics/Data/Interpolation/Bilinear.cs` (33dc1af): every `Math.Log10` -> guarded
  `Tools.Log10` (C++: the ported `tools::log10_guarded` equivalent — check the existing
  name in `tools.hpp` and use it consistently).
- `Numerics/Data/Statistics/Histogram.cs` (33dc1af): `AddData` out-of-range values stretch
  the endpoint bins and update Lower/UpperBound; in-range indexing moved after bounds
  checks. C++ currently throws via `get_bin_index_of` first.
- `Numerics/Data/Statistics/Probability.cs` (33dc1af): the commented-out `cdf < 1e-300`
  guard in `JointProbabilityHPCM` cycle 1 is now live as `minimumCdf`.
- `Numerics/Data/Statistics/RunningStatistics.cs` (41f78b4): new `Clone()`;
  `Combine(a, b)` with an empty operand returns a clone (verify the C++ value-type port
  already gives independent copies; add `clone()` for API parity).

**Interfaces:**
- Produces: fixed `search` used by interpolation/empirical paths; guarded bilinear log10.

- [ ] **Step 1: Add failing fixture cases.** For each behavior change add cases to the
  matching `fixtures/data/*.json` file, curated from the new C# tests
  (`Test_Numerics/Data/Interpolation/Test_Search.cs` — new file, descending bisection/hunt;
  `Test_Numerics/Utilities/Test_Tools.cs` — Log10 NaN; `Test_Histogram.cs` — AddData
  adapts endpoint bins; `Test_Probability.cs` — HPCM extreme probabilities stay finite;
  `Test_Bilinear.cs` — log-transform floor matches linear; `Test_RunningStatistics.cs` —
  Clone independence). Read `fixtures/README.md` for the assertion schema of each kind; if a
  needed method is not yet dispatched by the runners, add the dispatch entry to ALL THREE
  runners (`core/tests/test_fixtures.cpp`, `corehydror/tests/testthat/test-fixtures.R`,
  `corehydropy/tests/test_fixtures.py`) AND the emitter driver in
  `tools/oracle_emitter/Program.cs`.
- [ ] **Step 2: Run the C++ fixture runner; confirm the new cases FAIL** (old behavior).
  `ctest --test-dir core/build -R fixtures --output-on-failure`
- [ ] **Step 3: Port the C++ changes** hunk-by-hunk from the C# diffs above. Update each
  provenance header to `@ 2a0357a`. Retire the Search divergence note.
- [ ] **Step 4: Rebuild + run C++ fixtures to green.**
  `cmake --build core/build -j8 && ctest --test-dir core/build -R fixtures --output-on-failure`
- [ ] **Step 5: Verify with the dotnet gate** (targeted):
  `python3 tools/verify_oracles.py --fixtures fixtures/data` — the new cases must
  reproduce; any pre-existing failures in OTHER files remain until their task.
- [ ] **Step 6: Full C++ suite + R + Python suites.** Expected: all green (these files
  have no class-layout change, plain `R CMD INSTALL` is fine, but `--preclean` is always
  acceptable).
- [ ] **Step 7: Commit** `feat: sync small numerics utilities to v2.1.4 (search descending fix, guarded log10, histogram adapt, HPCM guard, RunningStatistics.Clone)`

---

### Task 2: Box-Cox / Yeo-Johnson fit hardening + YeoJohnsonLink

**Files:**
- Modify: `core/include/corehydro/numerics/data/box_cox.hpp`
- Modify: `core/include/corehydro/numerics/data/yeo_johnson.hpp`
- Modify: `core/include/corehydro/numerics/functions/yeo_johnson_link.hpp`
- Modify: `fixtures/data/statistics_utilities.json` (box_cox/yeo_johnson cases)

**C# refs:** `Numerics/Data/Statistics/BoxCox.cs`, `YeoJohnson.cs`,
`Numerics/Functions/Link Functions/YeoJohnsonLink.cs` (all 5a4d677).

Semantics to mirror exactly:
- `FitLambda`: new `CanFitLambda` pre-check (null / < 2 points / non-finite / non-positive
  for BoxCox / constant sample -> NaN); inside the Brent objective non-finite log-likelihood
  clamps to `-double.MaxValue`; the found candidate is rejected (NaN returned) if
  non-finite, `|lambda| > 5`, or `LogLikelihood(candidate)` is non-finite.
- `LogLikelihood`: returns -Inf on any non-finite intermediate.
- YeoJohnson only: `Transform`/`InverseTransform` lambda==2 branch becomes
  `Math.Abs(lambda - 2.0) < 1e-8` (tolerance band takes the log branch).
- YeoJohnsonLink `SetParametersFromValues`: throws ArgumentException when FitLambda
  returns NaN.

**Interfaces:**
- Produces: hardened `box_cox::fit_lambda` / `yeo_johnson::fit_lambda` NaN semantics,
  consumed by Task 16 (time-series transform failure handling) and Task 20 (pivot YJ link).

- [ ] **Step 1: Add failing fixture cases** to `fixtures/data/statistics_utilities.json`
  from `Test_Numerics/Data/Statistics/Test_YeoJohnson.cs` (new file: valid -> finite,
  invalid -> NaN, LogLikelihood invalid -> -Inf) and `Test_BoxCox.cs` (+38: invalid-sample
  NaN cases). Include a lambda-within-1e-8-of-2 transform case.
- [ ] **Step 2: Confirm they FAIL** (C++ currently computes/throws where C# now yields NaN).
- [ ] **Step 3: Port** the three files; update headers to `@ 2a0357a`.
- [ ] **Step 4: C++ fixtures green; targeted gate green** (`--fixtures fixtures/data`).
- [ ] **Step 5: Full suites (C++/R/Python) green.** Note the docs example finding
  ("Box-Cox/Yeo-Johnson fitted lambdas agree across R/Python only to ~1e-8") still holds —
  clean-sample lambdas do not change in this task.
- [ ] **Step 6: Commit** `feat: sync BoxCox/YeoJohnson fit hardening to v2.1.4`

---

### Task 3: Distribution validation wave

**Files (all under `core/include/corehydro/numerics/distributions/`):**
- Modify: `gumbel.hpp`, `logistic.hpp`, `inverse_chi_squared.hpp`, `binomial.hpp`,
  `deterministic.hpp`, `chi_squared.hpp`, `kernel_density.hpp`, `noncentral_t.hpp`,
  `truncated_distribution.hpp`, `multivariate/multivariate_student_t.hpp`
- Modify: fixtures under `fixtures/distributions/` (parameter-validity cases)

**C# refs:** the corresponding `Numerics/Distributions/Univariate/*.cs` (313d7ba) and
`Multivariate/MultivariateStudentT.cs` (313d7ba); `TruncatedDistribution.cs` also 33dc1af.

Semantics:
- Setter ordering: assign fields FIRST, then set `_parametersValid` from validation (the C#
  now matches what the C++ already does in most files — VERIFY each file rather than
  assuming; where the C++ mirrored the old stale-field consult, fix it).
- NaN/Inf rejection added to `ValidateParameters` (ChiSquared dof, KernelDensity bandwidth,
  MVT dof/location/scale, NoncentralT dof).
- NoncentralT: the AS 243 de-goto refactor is behavior-preserving — do NOT restructure the
  C++ to match cosmetically; only mirror the validation changes and confirm oracle values
  unchanged. Note the refactor in the provenance header comment.
- TruncatedDistribution: adopt `ValidateParametersCore` (validate base params on a clone,
  finite min < max, nonzero mass, Fmin/Fmax computed once), constructor validates instead
  of raw CDF calls, count mismatches throw. The C++ already slices parameters correctly
  (upstream fixed their `Subset` bug to match) — retire that divergence note.

- [ ] **Step 1: Add failing `parameters_valid` fixture cases** per distribution (non-finite
  inputs -> invalid; recovery after valid SetParameters), curated from the NEW
  `Test_Numerics/Distributions/Test_ParameterValidity.cs`. Follow the existing
  `parameters_valid`-style assertion pattern (grep `parameters_valid` under `fixtures/` for
  the method name the runners dispatch).
- [ ] **Step 2: Confirm FAIL where behavior differs; note pre-aligned files** (some cases
  will pass immediately — that is a finding for the census margin, keep the case).
- [ ] **Step 3: Port the deltas; update headers to `@ 2a0357a`.**
- [ ] **Step 4: C++ fixtures green; targeted gate `--fixtures fixtures/distributions` —
  new cases reproduce; expected remaining failures only in files owned by Tasks 4-9.**
- [ ] **Step 5: Full suites green.**
- [ ] **Step 6: Commit** `feat: sync distribution parameter-validation wave to v2.1.4`

---

### Task 4: StudentT

**Files:**
- Modify: `core/include/corehydro/numerics/distributions/student_t.hpp`
- Modify: `fixtures/distributions/student_t.json` (re-pin PDF values; add normalization case)

**C# ref:** `Numerics/Distributions/Univariate/StudentT.cs` (313d7ba, 33dc1af).

Semantics:
- `PDF(x)` multiplies by `1.0 / Sigma` (the audited Jacobian fix). EVERY existing PDF
  oracle with sigma != 1 changes by exactly that factor.
- `InverseCDF` extreme-tail: `double.MaxValue` saturation guard removed; quantile
  refactored to `Math.Sqrt(df) * Math.Sqrt(1-z) / Math.Sqrt(z)` (algebraic ulp drift);
  tail retains `Mu + Sigma * ...` location-scale.
- Finite-dof validation, setter-ordering (from Task 3's pattern; this file does both).

- [ ] **Step 1: Re-pin** `student_t.json` PDF expecteds (multiply sigma!=1 cases by 1/sigma
  by hand, then confirm via gate) and add the normalization + location-scale identity cases
  from `Test_StudentT.cs` (+44) and the extreme-tail InverseCDF mu/sigma retention case.
- [ ] **Step 2: C++ fixture runner FAILS on the re-pinned values.**
- [ ] **Step 3: Port; update header to `@ 2a0357a`; delete the StudentT bug note
  cross-reference** (the `docs/upstream-csharp-issues.md` doc pass happens in Task 22 —
  only the in-header note is touched here).
- [ ] **Step 4: C++ green; targeted gate green for `student_t.json`.**
- [ ] **Step 5: Full suites green.**
- [ ] **Step 6: Commit** `feat!: StudentT PDF gains the 1/sigma Jacobian (upstream v2.1.4 fix); re-pin oracles`

---

### Task 5: L-moment wave (PearsonTypeIII, LogPearsonTypeIII, Gamma PartialKp, GeneralizedLogistic)

**Files:**
- Modify: `core/include/corehydro/numerics/distributions/pearson_type_iii.hpp`
- Modify: `core/include/corehydro/numerics/distributions/log_pearson_type_iii.hpp`
- Modify: `core/include/corehydro/numerics/distributions/gamma_distribution.hpp`
- Modify: `core/include/corehydro/numerics/distributions/generalized_logistic.hpp`
- Modify: `fixtures/distributions/{pearson_type_iii,log_pearson_type_iii,gamma_distribution,generalized_logistic}.json`

**C# refs:** `PearsonTypeIII.cs`, `LogPearsonTypeIII.cs` (33dc1af, 5a4d677),
`GammaDistribution.cs` (33dc1af), `GeneralizedLogistic.cs` (33dc1af).

Semantics:
- PT3 + LP3, BOTH L-moment directions: `T3 *= Math.Sign(gamma)` (SIGNED L-skewness — the
  C++ deliberately reverted its own sign fix for bug-compatibility; now re-apply, matching
  upstream's exact placement); new `T3 == 0` branch in ParametersFromLinearMoments
  (`[L1, L2*sqrt(pi), 0]`) and `gamma == 0` branch in LinearMomentsFromParameters
  (`[mu, sigma/sqrt(pi), 0, 0.12260172]`); the `alpha >= 100` Stirling branch now upstream
  in BOTH directions (C++ already carries it as a divergence — converge exactly, retire
  note); `L1` simplified to exactly `mu` (accept ulp drift; oracles re-pin).
- Gamma: `PartialKp` near-zero-skew (`|skew| < 1e-4`) returns `(z*z - 1.0) / 6.0`.
- GenLogistic: exact `kappa == 0` limits (C++ divergence upstreamed — converge, retire
  note) PLUS the new `|kappa| <= NearZero` truncated-series branch — transcribe upstream's
  series coefficients exactly (do not derive your own).

- [ ] **Step 1: Re-pin + add cases.** Re-pin any negative-skew L-moment oracles (the sign
  flip); add `Test_LinearMoments_SignedSmallAndZeroSkew` cases (PT3 +50, LP3 +47), the
  Gamma `PartialKp` zero-skew limit/continuity cases (+33), and GenLogistic zero/near-zero
  kappa cases (+45).
- [ ] **Step 2: FAIL confirmed on changed values.**
- [ ] **Step 3: Port all four files; headers to `@ 2a0357a`; retire the three divergence
  notes (LP3 Stirling, GenLogistic kappa=0, PT3/LP3 T3 sign).**
- [ ] **Step 4: C++ green; targeted gate green for the four files.**
- [ ] **Step 5: Full suites green.** Watch the B17C fixtures: they may STILL fail the gate
  at this point (their C# oracles moved for other reasons owned by Tasks 12-20) — confirm
  their C++ results are unchanged by THIS task (run
  `ctest --test-dir core/build -R fixtures` before/after if in doubt).
- [ ] **Step 6: Commit** `feat!: signed T3 + zero-skew branches for PT3/LP3, Gamma PartialKp limit, GenLogistic near-zero series (v2.1.4); re-pin oracles`

---

### Task 6: Beta/GeneralizedBeta Mode + Mixture zero-inflation

**Files:**
- Modify: `core/include/corehydro/numerics/distributions/beta_distribution.hpp`
- Modify: `core/include/corehydro/numerics/distributions/generalized_beta.hpp`
- Modify: `core/include/corehydro/numerics/distributions/mixture.hpp`
- Modify: `fixtures/distributions/{beta_distribution,generalized_beta,mixture}.json` (+ any
  `*_sim` digest fixture touching zero-inflated mixtures — find with
  `grep -rl zero_inflat fixtures/`)

**C# refs:** `BetaDistribution.cs`, `GeneralizedBeta.cs` (33dc1af), `Mixture.cs`
(313d7ba, 7f8c652).

Semantics:
- Mode: U-shape (`a < 1 && b < 1`) -> 0.5 (midpoint for GeneralizedBeta); boundary
  comparisons widen to `<=`/`>=` returning the boundary; no more out-of-support values.
  PERT's degenerate midpoint behavior is preserved upstream — do not change it.
- Mixture: `IsZeroInflated`/`ZeroWeight` setters immediately renormalize finite nonnegative
  component weights to sum to `1 - ZeroWeight`; EM/MLE fit rescales `mleWeights` onto the
  configured simplex with equal-weight fallback when the sum is 0; `SetParameters`/
  `SetDistributions` maintain `_parametersValid`.

- [ ] **Step 1: Re-pin Mode oracles** for U/J-shaped Beta/GeneralizedBeta (including the
  documented `GeneralizedBeta(0.42, 1.57, 0, 1).Mode ~= 58` case which becomes a boundary
  value); add zero-inflated Mixture weight-renormalization cases from `Test_Mixture.cs`
  (+50). Re-pin any zero-inflated `*_sim` digests via the emitter.
- [ ] **Step 2: FAIL confirmed.**
- [ ] **Step 3: Port; headers to `@ 2a0357a`.**
- [ ] **Step 4: C++ green; targeted gate green.**
- [ ] **Step 5: Full suites green (R needs `--preclean` only if member layout changed —
  Mixture gains no members, but verify).**
- [ ] **Step 6: Commit** `feat!: Beta-family Mode boundary fix + Mixture zero-inflation renormalization (v2.1.4); re-pin oracles`

---

### Task 7: EmpiricalDistribution + UnivariateDistributionFactory

**Files:**
- Modify: `core/include/corehydro/numerics/distributions/empirical_distribution.hpp`
- Modify: `core/include/corehydro/numerics/distributions/base/univariate_distribution_factory.hpp`
- Modify: `fixtures/distributions/empirical_distribution.json` (or the file containing the
  `Empirical` composite target cases) and the factory's fixture if one exists

**C# refs:** `EmpiricalDistribution.cs` (313d7ba, 5a4d677, adbc873),
`Base/UnivariateDistributionFactory.cs` (33dc1af, 5c7b1a1).

Semantics:
- Empirical: new `ValidateData` (>= 2 points, equal lengths, ascending X, finite
  probabilities in [0,1]) evaluated on construction/SetParameters, lazily THROWN from
  CDF/InverseCDF when invalid; `OrderedPairedData` built with `strictX = false` so
  duplicate (nondecreasing) X become VALID; the OPD constructor throw type changes to
  ArgumentOutOfRange (mirror whatever exception type the C++ uses consistently for that
  case — match the C++ error-handling convention, note the C# type in a comment).
- Factory: verify the C++ switch already covers Deterministic/Empirical/KernelDensity/
  VonMises and throws on CompetingRisks/Mixture/UserDefined/undefined (the C# now matches
  our behavior; add KernelDensity if missing). Add `try_create_distribution(type)` ->
  `std::unique_ptr<UnivariateDistributionBase>` (nullptr on unsupported — the C# `bool
  TryCreateDistribution(type, out dist)` shape adapted to the factory's existing C++ idiom;
  look at how the factory returns and match it).

- [ ] **Step 1: Add cases:** duplicate-X valid; decreasing-X throws on use; non-finite
  probability throws; descending probability-order supported (`Test_EmpiricalDistribution.cs`
  +135); factory completeness + TryCreate contract (`Test_UnivariateDistributionFactory.cs`,
  new). If exceptions are asserted, follow the fixture schema's error-assertion pattern
  (grep `"throws"` under fixtures/ — if absent, assert via `parameters_valid`-style checks
  instead and note the C# exception in the fixture's `source` comment).
- [ ] **Step 2: FAIL confirmed.**
- [ ] **Step 3: Port; headers to `@ 2a0357a`.**
- [ ] **Step 4: C++ green; targeted gate green.**
- [ ] **Step 5: Full suites green.**
- [ ] **Step 6: Commit** `feat: EmpiricalDistribution validation + duplicate ordinates, factory TryCreate (v2.1.4)`

---

### Task 8: Copulas (sentinel fix, finite validation, deep-copy clones)

**Files (under `core/include/corehydro/numerics/distributions/copulas/`):**
- Modify: `base/archimedean_copula.hpp`, `base/bivariate_copula.hpp`, `amh_copula.hpp`,
  `clayton_copula.hpp`, `frank_copula.hpp`, `gumbel_copula.hpp`, `joe_copula.hpp`,
  `normal_copula.hpp`, `student_t_copula.hpp`
- Modify: copula fixtures under `fixtures/distributions/` (grep `-l copula fixtures/`)

**C# refs:** the `Bivariate Copulas/` files (313d7ba, 6a8a8d0).

Semantics:
- ArchimedeanCopula `ValidateParameter`: return null on valid (the "Parameter is valid"
  sentinel left `ParametersValid` permanently false for Clayton/Gumbel/Joe — the C++
  mirrors that bug; fix it; `parameters_valid` flips false -> true).
- All copulas: NaN/Inf theta (and rho for StudentT) rejected first.
- `Clone` deep-copies marginals via new protected static `BivariateCopula.CloneMarginal`
  (clone the marginal when it is a `UnivariateDistributionBase`).

- [ ] **Step 1: Re-pin/add cases:** `parameters_valid` true-after-valid-set for
  Clayton/Gumbel/Joe (re-pin if pinned false); non-finite theta rejection per family; clone
  deep-copy independence (from the per-family `Test_ParametersValid`/`Test_Clone`
  additions, ~+40 each).
- [ ] **Step 2: FAIL confirmed on the sentinel flip.**
- [ ] **Step 3: Port; headers to `@ 2a0357a`.**
- [ ] **Step 4: C++ green; targeted gate green.**
- [ ] **Step 5: Full suites green (copula estimation fixtures must not move — tau/MPL/IFM/
  MLE math is untouched; if any moves, STOP and investigate before committing).**
- [ ] **Step 6: Commit** `fix: Archimedean ParametersValid sentinel + copula finite validation + deep-copy clones (v2.1.4)`

---

### Task 9: MultivariateNormal + BivariateEmpirical + CompetingRisks

**Files:**
- Modify: `core/include/corehydro/numerics/distributions/multivariate/multivariate_normal.hpp`
- Modify: `core/include/corehydro/numerics/distributions/multivariate/bivariate_empirical.hpp`
- Modify: `core/include/corehydro/numerics/distributions/competing_risks.hpp`
- Modify: `fixtures/distributions/multivariate_normal.json` (or equivalent target file),
  `bivariate_empirical`, `competing_risks` fixtures

**C# refs:** `MultivariateNormal.cs` (313d7ba, 33dc1af, 651035e, b8e7c31),
`BivariateEmpirical.cs` (313d7ba, 33dc1af), `CompetingRisks.cs` (313d7ba, 33dc1af).

Semantics:
- MVN COVSRT repair (651035e): the singular-covariance permutation branch loop bounds
  (`j >= 0`, `l <= j`, `l <= i-1`), packed-index `l(l+1)/2`, swap offset, and `IJ`
  decrement fixes — the C++ transcribed the broken loops verbatim plus a bounds guard;
  replace with the FIXED loops exactly (transcribe from the new C#, not from the old C++ +
  patches). MVNDNT: void return + explicit `INFORM = 0` (mirror shape).
- MVN new API (b8e7c31): `TrySetParameters`, `TrySetCovariance`, `IsDensityValid`
  (invalid -> PDF 0, LogPDF -Inf), `Marginal(indices)`, `Conditional(observedIndices,
  values)` (Schur complement). Mirror C# names in the C++ naming convention
  (`try_set_parameters`, `try_set_covariance`, `is_density_valid`, `marginal`,
  `conditional`).
- MVN validation: finite mean/covariance checks (313d7ba).
- BivariateEmpirical: `SetParameters` nulls the cached bilinear interpolator (stale-cache
  audit fix — C++ mirrors the stale cache; fix); finite validation of x1/x2/p.
- CompetingRisks: Dependency setter invalidates the cached MVN; PerfectlyNegative branch no
  longer zeroes the public CorrelationMatrix; SetParameters throws on wrong flattened
  length; ValidateParameters honors throwException for empty distributions.

- [ ] **Step 1: Add cases** from `Test_MultivariateNormal.cs` (+347): MVNDST status codes
  (invalid dimension -> 2, insufficient budget -> 1), all-unbounded exact, bivariate
  collapse, near-singular / perfectly-(anti)correlated / permuted rank-deficient CDF
  collapses, Marginal/Conditional closed forms, TrySetCovariance invalid state; from
  `Test_BivariateEmpirical.cs` (+24): SetParameters invalidates the interpolator; from
  `Test_CompetingRisks.cs` (+38): dependency change invalidates MVN without mutating the
  correlation matrix. Use `--dump` to harvest (multivariate targets support it).
- [ ] **Step 2: FAIL confirmed on the COVSRT-repair cases and stale-cache cases.**
- [ ] **Step 3: Port; headers to `@ 2a0357a`; retire the MVN COVSRT bounds-guard
  divergence note.**
- [ ] **Step 4: C++ green; targeted gate green.**
- [ ] **Step 5: Full suites; MVN is a class-layout change -> `R CMD INSTALL --preclean`.**
- [ ] **Step 6: Commit** `feat!: MVN COVSRT repair + Try/Marginal/Conditional API, stale-cache fixes (v2.1.4); re-pin oracles`

---

### Task 10: MCMC fixes (MAP fitness scale, NUTS gradient, Gibbs registry model)

**Files:**
- Modify: `core/include/corehydro/numerics/sampling/mcmc/base/mcmc_sampler.hpp`
- Modify: `core/include/corehydro/numerics/sampling/mcmc/nuts.hpp`
- Modify: `core/include/corehydro/numerics/sampling/mcmc/model_registry.hpp`
- Modify: affected fixtures under `fixtures/sampling/` and `fixtures/estimation/` (census
  file lists exactly which)

**C# refs:** `Sampling/MCMC/Base/MCMCSampler.cs` (313d7ba, 33dc1af), `Sampling/MCMC/NUTS.cs`
(33dc1af), `Test_Numerics/Sampling/MCMC/Test_Gibbs.cs` (the conjugate reference model
rework — corehydro's `normal_conjugate_gibbs` registry model transcribed the OLD test's
`mu0/2` conditional-mean formula; adopt the new `ConditionalMeanParameters`).

Semantics:
- MAP initialization stores `ParameterSet(values, -DE.BestParameterSet.Fitness)` — fitness
  on the log-likelihood scale (the C++ mirrors the old negated-objective scale; fix). Every
  MAP-initialized Bayesian fixture's mode/MAP values may re-pin; seeded chain STREAMS are
  unchanged (RNG-driven).
- NUTS: both call sites (step-size heuristic + leapfrog init) call the configured
  `GradientFunction` instead of the hardcoded numerical gradient. BEFORE porting, verify in
  the C# that the default `GradientFunction` equals the old inline call (it does — the
  default is `NumericalDerivative.Gradient`); therefore default-gradient seeded streams are
  bit-stable. Confirm by running the existing NUTS fixtures unchanged after the port.
- Gibbs registry model: update `normal_conjugate_gibbs` conditional-mean formula to the
  corrected conjugate math from the reworked `Test_Gibbs.cs`; its seeded fixture re-pins.

- [ ] **Step 1: Re-pin from the census:** the Gibbs fixture digest and any MAP/mode
  assertions the census flagged under sampling/estimation. Harvest new values via the
  emitter (`--dump` where supported; otherwise run the gate and transcribe the reported
  actual values, then re-verify).
- [ ] **Step 2: FAIL confirmed on re-pinned values; NUTS stream fixtures still PASS
  (pre-port) — they must ALSO pass post-port unchanged.**
- [ ] **Step 3: Port the three files; headers to `@ 2a0357a`; retire the MAP-fitness and
  NUTS divergence notes.**
- [ ] **Step 4: C++ green; targeted gate green for sampling fixtures.**
- [ ] **Step 5: Full suites; the seeded R-vs-Python digest fixtures must stay
  bit-identical.**
- [ ] **Step 6: Commit** `fix: MAP fitness on log-likelihood scale, NUTS uses configured gradient, corrected conjugate Gibbs model (v2.1.4); re-pin oracles`

---

### Task 11: BrentSearch bracket

**Files:**
- Modify: `core/include/corehydro/numerics/math/optimization/brent_search.hpp`
- Modify: the optimizer fixture file that exercises Brent (find via
  `grep -rl -i brent fixtures/`)

**C# ref:** `Mathematics/Optimization/Local/BrentSearch.cs` (651035e).

Semantics: `Bracket` applies the geometric expansion factor `k` (the C++ carries
`(void)k;` mirroring the unused parameter — now use it as upstream does), validates
`s`/`k`, guards NaN objective values and non-finite coordinates via status updates, bounds
iterations by MaxIterations, leaves bounds unchanged on failure. Upstream's evaluation-count
regression: a distant minimum brackets in ~16 evaluations, not 10,002.

Note: `BoxCox`/`YeoJohnson.FitLambda` do NOT call `Bracket` (fixed [-5,5] bounds) — Task 2
fixtures are unaffected; do not re-pin them here.

- [ ] **Step 1: Add cases** from the rebuilt `Test_BrentSearch.cs` (+197): geometric
  expansion left/right, custom k, invalid-input rejection, plateau/monotone termination,
  and the evaluation-count regression if the fixture schema can express it (if not, add a
  ctest-level assertion in the existing Brent C++ unit test and keep values in the fixture).
- [ ] **Step 2: FAIL confirmed.**
- [ ] **Step 3: Port; header to `@ 2a0357a`; retire the `(void)k` divergence note.**
- [ ] **Step 4: C++ green; targeted gate green.**
- [ ] **Step 5: Full suites green (verify no MLSL/DE/B17C fixture moved — they use Brent
  only through paths that don't call Bracket; if one moves, STOP and investigate).**
- [ ] **Step 6: Commit** `fix: BrentSearch.Bracket applies geometric expansion (v2.1.4)`

---

### Task 12: DataFrame plotting rewrite + ThresholdData + nonparametric dedupe

The largest single task. Budget accordingly; if the implementer finds it exceeding one
session, split at the marked seam and commit the first half.

**Files:**
- Modify: `core/include/corehydro/models/data_frame/data_frame.hpp`
- Modify: `core/include/corehydro/models/data_frame/data_frame_plotting.hpp`
- Modify: `core/include/corehydro/models/data_frame/data_types/threshold_data.hpp`
- Modify: plotting/DataFrame fixtures (census lists them; also
  `fixtures/data/statistics_utilities.json` `plotting_positions` cases if affected)

**C# refs:** `Models/DataFrame/DataFrame.cs` (79d608b, 2727a1e, ab211a3, 42b106b, b43943c,
473fb70), `Models/DataFrame/DataTypes/ThresholdData.cs` (b43943c era). Port the NET
v2.0.0 state; intermediate commits are superseded.

Semantics (see spec for the full list):
- Hirsch-Stedinger rewrite: peakFQ ARRANGE2/PPLOT2/PLPOS structure; each explicit
  observation classified against the threshold covering its OWN index; strict validation
  THROWS (non-finite values, overlapping threshold windows, counts exceeding duration,
  infeasible NumberBelow). The legacy tie permutation is preserved via the value-only
  `List<T>.Sort` comparator — our ArraySortHelper introsort port REMAINS the sort;
  transcribe the new above/below split ("global minimum finite threshold").
- `EnsureDistinctPlottingPositions`: deterministic duplicate spreading (tie center
  preserved; value desc / index / ordinal comparator; positions spaced within midpoint
  boundaries).
- Empirical construction: repeated X collapse into right-continuous CDF steps; degenerate
  input -> NaN/null result (affects `GetNonparametricMoments*`).
- `ProcessThresholdSeries` idempotent via ThresholdData source/effective count split:
  `NumberAbove` setter records `_sourceNumberAbove`; `SetProcessedCounts` mutates effective
  counts; `Validate` checks source; `Clone` preserves both.
- SKIP: `PlottingPositionVersion` XML stamps, `FromXElement` recomputation,
  `RecalculatePlottingPositionsAfterEdit` GUI tolerance path — XML/GUI severance; note in
  header. PORT `PlottingPositionVersion` itself ONLY if Task 15's BivariateDistribution
  cache needs it (it does — expose a monotonically increasing version counter bumped by
  `CalculatePlottingPositions`; that is the non-GUI core of the feature).

**Interfaces:**
- Produces: `threshold_data.set_processed_counts(...)`, `source_number_above()` (consumed
  by Tasks 18-19); `data_frame.plotting_position_version()` (consumed by Task 15);
  rewritten `calculate_plotting_positions()` (consumed by Tasks 15, 18-20 and the public
  `plotting_positions` API).
- Split seam if needed: (a) ThresholdData + ProcessThresholdSeries idempotency + empirical
  dedupe; (b) the H-S rewrite + EnsureDistinctPlottingPositions.

- [ ] **Step 1: Re-pin/add cases** from `src/RMC.BestFit.Tests` PlottingPositionTests,
  NonparametricEmpiricalTests, ThresholdDataTests (fc28c0c..v2.0.0 added them): multi-
  threshold frames, tied values, duplicate positions, idempotent re-processing, degenerate
  empirical input. Census tells you which existing fixtures re-pin.
- [ ] **Step 2: FAIL confirmed.**
- [ ] **Step 3: Port; headers to `@ c2e6192`; document the XML/GUI skips in headers.**
- [ ] **Step 4: C++ green; targeted gate green (`--fixtures fixtures/analyses` +
  wherever the plotting fixtures live per census).**
- [ ] **Step 5: Full suites; `--preclean` (ThresholdData/DataFrame layouts change).**
- [ ] **Step 6: Commit** `feat!: peakFQ-faithful Hirsch-Stedinger rewrite, deterministic tie spreading, idempotent threshold processing (BestFit v2.0.0); re-pin oracles`

---

### Task 13: GeneralizedMethodOfMoments + NumericalDiff

**Files:**
- Modify: `core/include/corehydro/estimation/generalized_method_of_moments.hpp`
- Modify: `core/include/corehydro/estimation/numerical_diff.hpp`
- Modify: GMM/B17C estimation fixtures per census

**C# refs:** `Estimation/GeneralizedMethodOfMoments.cs` (5e1877f, b43943c),
`Estimation/NumericalDiff.cs` (0d6821d, 7efa9d0).

Semantics:
- GMM: accept estimates on any NON-FAILURE optimizer termination with a finite best point
  (max-iterations / max-fevals no longer fail); sticky Nelder-Mead after one BFGS failure
  (`OptimizerFallbackCount` tracked); internal `TryGetCovariance` (finite +
  positive-diagonal gate); `ConvergedWithinTolerance` off-by-one fixed (tracked flag;
  final-pass convergence counts; one/two-step reports false).
- NumericalDiff: Hessian diagonal step search bounded at 64 attempts, returns cached
  forward/backward stencil values without re-evaluation, tracks last finite step, throws
  when no finite central stencil exists; Jacobian one-sided fallback when one side is
  non-finite. Interior smooth values are IDENTICAL — existing interior fixtures must not
  move.

**Interfaces:**
- Produces: `gmm.optimizer_fallback_count()`, `gmm.converged_within_tolerance()`,
  `gmm::try_get_covariance(...)` (or member equivalent — mirror the C# placement),
  consumed by Tasks 19-20.

- [ ] **Step 1: Re-pin/add cases** from the expanded GMM tests and NumericalDiffTests in
  `src/RMC.BestFit.Tests`; census lists which existing GMM/B17C fixtures re-pin here
  (starting-value-sensitive ones may wait for Task 18 — check the census notes per file
  and only re-pin what THIS task's semantics move).
- [ ] **Step 2: FAIL confirmed.**
- [ ] **Step 3: Port; headers to `@ c2e6192`; retire the ConvergedWithinTolerance
  audit-finding note.**
- [ ] **Step 4: C++ green; targeted gate for the touched fixtures.**
- [ ] **Step 5: Full suites; `--preclean` (GMM layout changes).**
- [ ] **Step 6: Commit** `feat!: GMM non-failure acceptance + sticky NM fallback, NumericalDiff edge hardening (BestFit v2.0.0); re-pin oracles`

---

### Task 14: Jeffreys guards + MixtureModel.Clone

**Files:**
- Modify: `core/include/corehydro/models/univariate_distribution/base/univariate_distribution_model_base.hpp`
- Modify: `core/include/corehydro/models/univariate_distribution/univariate_distribution_model.hpp`
- Modify: `core/include/corehydro/models/univariate_distribution/mixture_model.hpp`
- Modify: `core/include/corehydro/models/univariate_distribution/competing_risks_model.hpp`
- Modify: model-estimation fixtures per census

**C# refs:** `Base/UnivariateDistributionModelBase.cs`, `UnivariateDistribution.cs`,
`MixtureModel.cs`, `CompetingRisksModel.cs` (1abe795, b43943c).

Semantics:
- Two protected static `TryGetJeffreysScaleParameter` overloads on the base (Gamma/Weibull
  scale at index 0, others index 1, bounds-checked).
- UnivariateDistribution: Jeffreys 1/scale prior guarded — single-parameter families SKIP
  the Jeffreys term (no more `GetParameters[1]` crash).
- Mixture/CompetingRisks models: `continue` past single-parameter components; immediate
  `-Inf` return for non-positive scale (avoids Inf-Inf NaN).
- MixtureModel.Clone: propagate `IsZeroInflated`/`ZeroWeight` onto the cloned inner
  Mixture — this RETIRES the documented C++ intentional divergence; the C++ header note
  changes from divergence to plain port.

- [ ] **Step 1: Add cases:** single-parameter-family Jeffreys fit (previously would
  throw/NaN — new fixture from the Mixture/CompetingRisks Jeffreys tests in
  `src/RMC.BestFit.Tests`); zero-inflated clone propagation (couples with Task 6's Mixture
  semantics).
- [ ] **Step 2: FAIL confirmed.**
- [ ] **Step 3: Port; headers to `@ c2e6192`.**
- [ ] **Step 4: C++ green; targeted gate green.**
- [ ] **Step 5: Full suites green.**
- [ ] **Step 6: Commit** `fix: Jeffreys scale-parameter guards + MixtureModel.Clone zero-inflation propagation (BestFit v2.0.0)`

---

### Task 15: BivariateDistribution + RatingCurve default priors

**Files:**
- Modify: `core/include/corehydro/models/bivariate_distribution/bivariate_distribution.hpp`
- Modify: `core/include/corehydro/models/rating_curve/rating_curve.hpp`
- Modify: bivariate/rating-curve fixtures per census

**C# refs:** `Models/BivariateDistribution/BivariateDistribution.cs` (0d6821d, b43943c),
`Models/RatingCurve/RatingCurve.cs` (0d6821d).

Semantics:
- BivariateDistribution PseudoLikelihood: validate pseudo observations on (0,1); when
  invalid, auto-run `CalculatePlottingPositions()` and re-validate, caching against
  Task 12's `plotting_position_version()`. Skip the GUI-only
  `GetSampleDataAlignmentCounts`. The previously-failing PseudoLikelihood path now
  estimates — the audit finding retires.
- RatingCurve defaults: exponent Uniform priors and bounds -5..5 -> -10..10; the `a` lower
  bound 0.5 -> 0 with Uniform(0, 5) and `IsPositive` removed; sigma upper-bound derivation
  via the extracted helpers (`GetDefaultPriorStageScale`, `GetDefaultPriorSigmaUpperBound`)
  — transcribe the helpers; `h3Max` recomputed equivalently.

- [ ] **Step 1: Re-pin** rating-curve fixtures that build default priors (census) and add
  a PseudoLikelihood-now-estimates case from BivariateDistributionTests.
- [ ] **Step 2: FAIL confirmed.**
- [ ] **Step 3: Port; headers to `@ c2e6192`.**
- [ ] **Step 4: C++ green; targeted gate green.**
- [ ] **Step 5: Full suites green. The seeded RatingCurve analysis fixture asserts
  structural invariants only (Phase-10 precedent) — verify it still passes WITHOUT
  loosening anything.**
- [ ] **Step 6: Commit** `feat!: BivariateDistribution pseudo-obs auto-plotting + RatingCurve default prior widening (BestFit v2.0.0); re-pin oracles`

---

### Task 16: TimeSeries transform-failure handling

**Files:**
- Modify: `core/include/corehydro/models/time_series/auto_regressive.hpp`
- Modify: `core/include/corehydro/models/time_series/moving_average.hpp`
- Modify: `core/include/corehydro/models/time_series/arima.hpp`
- Modify: `core/include/corehydro/models/time_series/arimax.hpp`
- Modify: time-series model fixtures per census

**C# refs:** `Models/TimeSeries/{AutoRegressive,MovingAverage,ARIMA,ARIMAX}.cs`
(f140c4d + 0d6821d).

Semantics: BoxCox/YeoJohnson `FitLambda` wrapped in try/catch + non-finite check — on
failure set lambda 0, empty the training series, record the validation message surfaced
via `Validate` (mirror the C# message text exactly; it is oracle-visible only through
Validate results, which the fixtures check structurally). SKIP the XML
TrainingTimeSteps/UseDefaultTrainingSteps persistence and the GUI training-steps reset
(document in headers). Uses Task 2's hardened `fit_lambda` NaN semantics.

- [ ] **Step 1: Add a transform-lambda-failure case** per model (from the new per-model
  transform-failure tests in `src/RMC.BestFit.Tests`): degenerate series -> model validates
  false with the failure message, no crash.
- [ ] **Step 2: FAIL confirmed (C++ currently crashes or propagates NaN).**
- [ ] **Step 3: Port; headers to `@ c2e6192`.**
- [ ] **Step 4: C++ green; targeted gate green.**
- [ ] **Step 5: Full suites green; the AR/MA/ARIMA/ARIMAX seeded digests must be
  unchanged (transform failure paths don't touch the happy-path stream).**
- [ ] **Step 6: Commit** `feat: time-series transform lambda-failure handling (BestFit v2.0.0)`

---

### Task 17: BestFit YeoJohnsonLink removal

**Files:**
- Delete: `core/include/corehydro/models/link_functions/yeo_johnson_link.hpp`
- Modify: `core/include/corehydro/models/link_functions/best_fit_link_function_factory.hpp`
- Modify: any `#include` of the deleted header (find:
  `grep -rl 'link_functions/yeo_johnson_link' core/ corehydror/src corehydropy/src`)
- Modify: link-function fixtures if any pin the BestFit YJ link (census)

**C# ref:** `Models/LinkFunctions/BestFitLinkFunctionFactory.cs` + the DELETED
`Models/LinkFunctions/YeoJohnsonLink.cs` (68b07a8).

Semantics: the factory's YeoJohnsonLink case constructs the NUMERICS
`numerics/functions/yeo_johnson_link.hpp` link. Before deleting, parity-check `DLink`
between the two implementations (the Numerics one validates lambda in [-5, 5] and is
immutable) — if any behavior differs on inputs our fixtures exercise, the fixtures re-pin
to the Numerics behavior (that is what upstream now runs). Skip the legacy-XML
IdentityLink fallback (XML severance; note in the factory header).

- [ ] **Step 1: Census check + parity diff of the two link implementations; adjust/add
  fixture cases pinning the Numerics-link behavior through the factory.**
- [ ] **Step 2: FAIL confirmed if behavior differs (may already pass — record either way).**
- [ ] **Step 3: Delete the header, reroute the factory, fix includes; factory header to
  `@ c2e6192`.**
- [ ] **Step 4: C++ green; targeted gate green.**
- [ ] **Step 5: Full suites; class-layout deletion -> `R CMD INSTALL --preclean`.**
- [ ] **Step 6: Commit** `feat!: remove BestFit YeoJohnsonLink, route factory to the Numerics link (BestFit v2.0.0)`

---

### Task 18: Bulletin17CDistribution

**Files:**
- Modify: `core/include/corehydro/models/univariate_distribution/bulletin17c_distribution.hpp`
- Modify: (possibly) `core/include/corehydro/models/univariate_distribution/bulletin17c_moment_machinery.hpp`
- Modify: B17C fixtures per census (starting-value-sensitive GMM cases)

**C# ref:** `Models/UnivariateDistribution/Bulletin17CDistribution.cs` (0dc8594, c420d48).

Semantics:
- `CloneWithDataFrame(frame)`: binds a resampled DataFrame WITHOUT rebuilding parameters
  from the boot data — preserves parent initial values and parameter penalties (the C#
  routes through its XElement round-trip; the C++ implements the equivalent state-copy
  directly since XML is severed — copy the same fields the C# round-trip preserves, and
  document that equivalence in the header).
- `SetDefaultParameters`: the ROS nonparametric override triggers ONLY for low outliers or
  threshold series — no longer for uncertain/interval series (GMM starting values change
  for those fits).

**Interfaces:**
- Consumes: Task 12's `set_processed_counts`/plotting rewrite; Task 13's GMM semantics.
- Produces: `clone_with_data_frame(const DataFrame&)` consumed by Tasks 19-20.

- [ ] **Step 1: Re-pin** interval/uncertain-data B17C GMM fixtures (census) and add a
  CloneWithDataFrame state-preservation case from Bulletin17CDistributionTests.
- [ ] **Step 2: FAIL confirmed.**
- [ ] **Step 3: Port; header to `@ c2e6192`.**
- [ ] **Step 4: C++ green; targeted gate green.**
- [ ] **Step 5: Full suites; `--preclean`.**
- [ ] **Step 6: Commit** `feat!: B17C CloneWithDataFrame + narrowed ROS default trigger (BestFit v2.0.0); re-pin oracles`

---

### Task 19: BootstrapDiagnostics + Bulletin17CAnalysis bootstrap (discard semantics)

**Files:**
- Modify: `core/include/corehydro/analyses/support/bootstrap_diagnostics.hpp`
- Modify: `core/include/corehydro/analyses/univariate/bulletin17c_analysis.hpp`
- Modify: B17C analysis fixtures per census

**C# refs:** `Analyses/Support/BootstrapDiagnostics.cs`,
`Analyses/Univariate/Bulletin17CAnalysis.cs` — the parametric-bootstrap arm only
(1b424e3, 71b7d4b, 0dc8594, 7efa9d0 net state; `c155732` is superseded staging).

Semantics (bootstrap arm):
- Failed replicates DISCARDED (never substitute parent parameters).
- Replicates warm-start via Task 18's `clone_with_data_frame` when low outliers / censored
  / threshold data present (mirror the C# condition exactly).
- Acceptance gate: reject only hard optimizer failures and non-finite estimates (Task 13's
  non-failure-termination semantics).
- `maxRetries` 5 -> 10; Mahalanobis threshold ChiSq(p) at `1 - 1/(5B)` (adaptive, was
  0.9999); singular parent covariance regularized via
  `MatrixRegularization.MakeSymmetricPositiveDefinite` instead of aborting.
- Master PRNG: one `NextIntegers(B)` block keyed to the ORIGINAL replicate index (seeded
  stream layout — transcribe exactly; this is what keeps the replicate streams
  deterministic and oracle-exact).
- Abort-with-warning when < 2 realizations survive or > half discarded; new public
  `UncertaintyDiagnosticMessage` (port as a plain string accessor).
- BootstrapDiagnostics: `AttemptedReplicates`, `RetainedReplicates`, `TransformFailures`,
  GMM `OptimizationStatus` counters, `OptimizerFallbacks`; `ValidReplicates`/`FailureRate`/
  `AverageRetries` redefined over attempts. Skip the XML persistence (severance note).

**Interfaces:**
- Consumes: Task 13 (`optimizer_fallback_count`, `try_get_covariance`), Task 18
  (`clone_with_data_frame`).
- Produces: the reworked bootstrap replicate collector reused by Task 20's pivot path.

- [ ] **Step 1: Re-pin** the B17C bootstrap-UQ fixtures (census; seeded streams are
  deterministic -> exact) and add BootstrapDiagnostics field assertions from
  BootstrapDiagnosticsTests.
- [ ] **Step 2: FAIL confirmed.**
- [ ] **Step 3: Port; headers to `@ c2e6192`.**
- [ ] **Step 4: C++ green; targeted gate green (`--fixtures fixtures/analyses`).**
- [ ] **Step 5: Full suites; `--preclean`.**
- [ ] **Step 6: Commit** `feat!: B17C bootstrap discard semantics + diagnostics bookkeeping (BestFit v2.0.0); re-pin oracles`

---

### Task 20: Bulletin17CAnalysis pivot path + CreatePivotYeoJohnsonLink

**Files:**
- Modify: `core/include/corehydro/analyses/univariate/bulletin17c_analysis.hpp`
- Modify: pivot/BiasCorrected B17C fixtures per census

**C# ref:** `Analyses/Univariate/Bulletin17CAnalysis.cs` pivot/link-builder region
(7efa9d0 net state).

Semantics:
- Pivot path compacts to ACCEPTED replicates before link fitting; drops z-limit rejections
  (`zLimit = 6`, new constant) and failed transforms.
- `CreatePivotYeoJohnsonLink`: validates finiteness/roundtrip, rejects lambda railed at
  `|lambda| >= 4.999`, falls back to IdentityLink. Uses the Numerics YJ link (Task 17).
- The LinkedMultivariateNormal path's link-builder helpers pick up any adjacent diffs in
  the same region — port whatever the v2.0.0 diff shows for the helpers, nothing more.

**Interfaces:**
- Consumes: Task 19's replicate collector; Task 17's Numerics-link routing; Task 2's
  hardened YJ FitLambda NaN semantics.

- [ ] **Step 1: Re-pin** the pivot/BiasCorrected fixtures (census).
- [ ] **Step 2: FAIL confirmed.**
- [ ] **Step 3: Port; header stays `@ c2e6192`.**
- [ ] **Step 4: C++ green; targeted gate green.**
- [ ] **Step 5: Full suites green.**
- [ ] **Step 6: Commit** `feat!: B17C pivot-path compaction + guarded pivot YJ link (BestFit v2.0.0); re-pin oracles`

---

### Task 21: Parity closeout — new-API fixture completeness, bindings check, full verification

**Files:**
- Modify: (as needed) the three fixture runners + emitter drivers for any new-API method
  not yet dispatched; `corehydror/_pkgdown.yml` + `site/_quarto.yml` ONLY if a new export
  landed (default: none)

- [ ] **Step 1: Sweep the spec's "New API to port" list; confirm each item has a fixture
  case exercised by all four runners** (C++/R/Python + emitter). Add any missing dispatch
  entries. (Phase-10 lesson: unexercised arms get no oracle.)
- [ ] **Step 2: Bindings decision check:** confirm no NEW user-facing R/Python export is
  needed (the new API rides existing class surfaces internally; the public `distribution()`
  / `Distribution` verbs and analysis functions are unchanged). If the implementer believes
  an export IS warranted, STOP and surface it for a user decision rather than adding it.
- [ ] **Step 3: Full verification, in order:**
  - `cmake --build core/build -j8 && ctest --test-dir core/build --output-on-failure`
    (expect: all suites, including non-fixture ctests, green)
  - `python3 tools/verify_oracles.py` (expect: 0 failed; reproduced count RISES vs the
    4109 pre-sync baseline; skips only the documented GEV std-err set — record the new
    counts)
  - `R CMD INSTALL --preclean corehydror && Rscript -e 'testthat::test_local("corehydror")'`
    (expect: 0 failures)
  - `pixi run python -m pip install --force-reinstall --no-deps ./corehydropy && pixi run python -m pytest corehydropy/tests -q`
    (expect: 0 failures)
  - Cross-language digest check: the seeded `*_sim` / chain-digest fixtures pass in BOTH R
    and Python (bit-identical by construction if both suites are green).
- [ ] **Step 4: Update the census file** with the final counts (before/after table).
- [ ] **Step 5: Commit** `test: parity closeout for the v2.1.4/v2.0.0 sync; final oracle census`

---

### Task 22: Docs, versions, process capture

**Files:**
- Modify: `docs/upstream-csharp-issues.md`
- Modify: `upstream/CLAUDE.md`, `.claude/CLAUDE.md` (status sections)
- Modify: `corehydror/DESCRIPTION` (Version: 0.2.0), `corehydropy/pyproject.toml`
  (version = "0.2.0"); `corehydror/NEWS.md` / any changelog if present (check)
- Create: `docs/upstream-sync.md`

- [ ] **Step 1: `docs/upstream-csharp-issues.md` resolution pass.** For every finding fixed
  upstream, append a line to the entry: `RESOLVED upstream in Numerics v2.1.4 (33dc1af)` /
  `RESOLVED upstream in RMC.BestFit v2.0.0 (b43943c)` etc., and state that the C++ now
  ports the fixed behavior. Do NOT delete the historical entries. Update the header pins
  (`a2c4dbf` -> `2a0357a`, `fc28c0c` -> `c2e6192`).
- [ ] **Step 2: Write `docs/upstream-sync.md`** — the repeatable release-absorption process
  as actually executed: fetch tags -> classify diffs (record the two classification-agent
  prompts) -> bump + emitter repair -> oracle census -> port in dependency order (Numerics
  before BestFit) -> per-task re-pin -> closeout verification -> ship. Include the pitfalls
  actually hit this run (pull them from the SDD ledger and commit history) and a
  "candidate automation" section listing what a future `upstream_diff.py` would need.
- [ ] **Step 3: Update the status sections** of `.claude/CLAUDE.md` and `upstream/CLAUDE.md`:
  new validated-against versions, new oracle counts, the retired divergences, the new
  skips (AnalysisProgress, BestFit App/UI/Api projects).
- [ ] **Step 4: Bump versions to 0.2.0** in both packages, noting "validated against
  Numerics v2.1.4 / RMC.BestFit v2.0.0" in DESCRIPTION/NEWS and pyproject metadata as each
  file's convention allows.
- [ ] **Step 5: Rebuild + spot-run both package suites once more (version metadata only —
  expect green).**
- [ ] **Step 6: Commit** `docs: upstream-sync process doc, issue-log resolution pass, v0.2.0 version bump`

---

## Ship step (driven by the orchestrator, NOT a subagent task)

Per the standing "workflow resume is unsafe once committed" lesson, CI + PR runs as a
fresh step after all tasks land: push `upstream-sync-2026-07`, `gh run watch` the CI matrix
to green, open the PR (body summarizes: what upstream changed, what re-pinned, retired
divergences, new skips, oracle before/after counts), then write the lessons-learned
memories. Do not push before Task 22 is committed.

## Self-review notes

- Spec coverage: every spec scope item maps to a task (Numerics 1-11, BestFit 12-20,
  closeout 21, docs/process 22; T0 census). The spec's "Not ported" lists are encoded as
  in-header severance notes in Tasks 1, 12, 16, 17, 19.
- Census-dependency: Tasks 4-20 say "per census" for re-pin targets because the exact
  fixture list is an OUTPUT of Task 0 — that is deliberate, not a placeholder; the census
  file is committed and versioned.
- Type consistency: `clone_with_data_frame` (18 -> 19/20), `set_processed_counts` /
  `source_number_above` (12 -> 18/19), `plotting_position_version` (12 -> 15),
  `optimizer_fallback_count` / `try_get_covariance` / `converged_within_tolerance`
  (13 -> 19/20), `try_set_parameters` / `try_set_covariance` / `is_density_valid` /
  `marginal` / `conditional` (9), `try_create_distribution` (7).
