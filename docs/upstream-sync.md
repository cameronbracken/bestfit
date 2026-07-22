# Absorbing an upstream RMC release

corehydro is a line-for-line C++17 port of two C# libraries, USACE-RMC `Numerics` and
`RMC.BestFit`, vendored as dev-only submodules under `upstream/`. When RMC ships a release, the
port has to absorb it: adopt the new behavior, re-pin every oracle the new C# moved, and retire
any divergence upstream has adopted.

This document is the process, written from the July 2026 run that took Numerics `a2c4dbf`
(v2.1.1 plus 3 commits) to `2a0357a` (v2.1.4) and RMC-BestFit `fc28c0c` (v2.0-beta.5) to
`c2e6192` (v2.0.0). That run took 24 porting and closeout tasks plus a final docs pass, and moved
the oracle gate from 4109 reproduced values to 4497. It is written as instructions, with the
incidents that produced each rule attached, so a future run can follow it start to finish and skip
the mistakes.

The run's own artifacts are the worked example: the design doc at
`docs/superpowers/specs/2026-07-19-upstream-sync-numerics-2.1.4-bestfit-2.0.0-design.md`, the
task plan beside it, and the failure census at
`docs/superpowers/plans/2026-07-19-upstream-sync-oracle-census.md`.

## Prerequisites

- `dotnet` and the two submodules checked out. The oracle gate is dev-only and is not in CI, so a
  sync cannot be done from a CI-shaped environment.
- A worktree and a branch. This is a long-running, many-commit change; do it on a branch in a
  worktree, one PR at the end.
- The full local toolchain: `cmake`, R with the package's dev library, and the pixi Python env.
  Every task runs C++, R, and Python before it commits.

## Step 1: fetch the tags and size the ranges

```bash
git -C upstream/Numerics fetch --tags origin
git -C upstream/RMC-BestFit fetch --tags origin
git -C upstream/Numerics log --oneline <old>..<newtag>
git -C upstream/RMC-BestFit log --oneline <old>..<newtag>
git -C upstream/Numerics diff --stat <old>..<newtag> -- Numerics/
git -C upstream/RMC-BestFit diff --stat <old>..<newtag> -- src/RMC.BestFit/
```

Restrict the stat to the library subtree. The July 2026 BestFit range touched 871 files, but only
64 of them were under `src/RMC.BestFit/`; the rest were the new App, UI, and Api projects, their
tests, docs, and examples. Sizing the library subtree separately is what turns an intimidating
diff into a scoped one.

Read the release's own commit log at this point, but only to find the interesting commits. Do not
trust it for semantics; see the first hard-won rule below.

## Step 2: classify the diff

The classification is the real worklist. Run one agent per repository against the library subtree
diff, in parallel, and have each produce a ranked table. The two prompts below are reconstructed
from what the July 2026 classification produced (the originals were not archived, which is itself
a lesson: save the prompts). They are written to be pasted with the placeholders filled in.

**Numerics classification prompt:**

> Read the full diff of the vendored USACE-RMC Numerics submodule between `<old>` and `<newtag>`,
> restricted to the `Numerics/` library subtree:
> `git -C upstream/Numerics diff <old>..<newtag> -- Numerics/`.
> Ignore the `Test_Numerics/` tree for classification, but DO read it afterwards to find new or
> changed unit tests, because those are the oracle sources for anything you classify as a behavior
> change.
>
> corehydro is a faithful C++17 port of this library. Every numeric behavior change invalidates
> pinned oracle values in `fixtures/*.json`, so I need to know exactly which changes are numeric.
>
> Classify every changed file into exactly one of:
> 1. BEHAVIOR CHANGE TO PORT. The output of some public method changes for some input. State the
>    class, the method, the precise change, and which inputs are affected. Say whether the change
>    is a bug fix, a new guard or validation, or a refactor with ulp-level drift.
> 2. NEW API TO PORT. New public members with no behavior change to existing ones.
> 3. COSMETIC. XML-doc comments, byte-order marks, whitespace, namespace moves, dead-code
>    removal with no reachable effect. Say what makes you confident it is cosmetic.
> 4. OUT OF SCOPE. Areas this port does not cover. Name the reason.
>
> Rank the BEHAVIOR CHANGE items by risk to existing oracles, highest first: a change that moves
> a value every caller sees ranks above one that only fires on a degenerate input.
>
> Cross-reference `docs/upstream-csharp-issues.md`. That file is corehydro's log of bugs found in
> this C# library, and upstream sometimes fixes them. For every entry in that file, say whether
> this release fixes it. Where it does, note that the C++ currently mirrors the OLD behavior
> deliberately, so the fix will flip an oracle.
>
> Read the SHIPPED source at `<newtag>` for anything you are unsure about. Commit messages
> describe intent, not what shipped, and later commits in the range can revert earlier ones.
> Quote the file and line you relied on.

**RMC.BestFit classification prompt:** the same, with these substitutions and additions:

> ... restricted to `src/RMC.BestFit/`:
> `git -C upstream/RMC-BestFit diff <old>..<newtag> -- src/RMC.BestFit/`.
> Ignore `src/RMC.BestFit.Tests/` for classification but read it afterwards for oracle sources.
> Treat the `src/RMC.BestFit.App`, `.UI`, and `.Api` projects as out of scope without reading them
> in detail; note only that they exist.
>
> BestFit source files are not valid UTF-8 in the working tree. Read them with
> `git show <newtag>:<path>`, not with a text search over the checkout, or you will get mojibake
> and miss matches.
>
> This library layers on Numerics, so flag any change whose behavior depends on a Numerics change
> in the same release; those have to be ported second.
>
> Pay particular attention to the analysis orchestrators. Their diffs are dominated by GUI progress
> reporting, report text, and XML persistence, none of which this port carries. Separate the
> numeric changes from that churn explicitly rather than classifying a whole file at once.

Write the merged output into a design doc with two scope sections, each carrying a ranked "behavior
changes to port" list, a "new API" list, and an explicit "not ported (documented)" list. The
not-ported list is not decoration: each item becomes a severance note in a C++ header, and the
next sync's provenance sweep depends on those notes existing.

## Step 3: bump the submodules and repair the oracle emitter

```bash
git -C upstream/Numerics checkout <newtag>
git -C upstream/RMC-BestFit checkout <newtag>
git add upstream/Numerics upstream/RMC-BestFit
git commit -m "chore: bump upstream submodules to <versions>"
dotnet build tools/oracle_emitter -c Release
```

The emitter subset-compiles real upstream sources, so a release restructure breaks it. Fix the
breaks minimally and record each one, because they are evidence about what moved. The July 2026
run hit three:

1. The stale CS0104 workaround became obsolete. Upstream deleted its duplicate
   `YeoJohnsonLink.cs`, so `tools/oracle_emitter/patched/Bulletin17CAnalysis.cs` and the two
   csproj lines that swapped it in were deleted and the real file compiles in place.
2. The DataFrame series and data types moved from `namespace RMC.BestFit` to
   `namespace RMC.BestFit.Models`, so the emitter's `using RMC.BestFit;` stopped resolving the
   unqualified constructors.
3. Fixing (2) created a fresh ambiguity between `RMC.BestFit.Models.Transform` and
   `Numerics.Data.Transform`, resolved by fully qualifying seven call sites rather than adding
   another alias.

None of these were predicted by name. All three were the anticipated class of break. Contain them
here, before any porting starts.

## Step 4: run the gate against untouched fixtures for the failure census

```bash
python3 tools/verify_oracles.py 2>&1 | tee /tmp/oracle-census-raw.txt
```

Every failure is a fixture whose oracle the new C# moved. Write the census to a committed file:
summary counts, the complete failure list with old and new values, the failures grouped by
subsystem, and a two-way cross-check against the classification. Flag both a failure the
classification did not predict and a predicted risk area with no failure. Both are findings.

Also confirm the C++ suite still passes at this point. It should: the port still matches the old
fixtures, and only the C# moved.

July 2026 census: 4099 reproduced, 10 failed, 11 skipped, against a pre-bump baseline of 4109
reproduced and 11 skipped. All 10 failures were predicted. Read the next section before drawing
any comfort from that number.

## Step 5: plan the tasks, and port in dependency order

Numerics before BestFit, always. BestFit's Bulletin 17C, GMM, and DataFrame layers consume the
distribution, L-moment, and optimizer changes, so porting them first means porting against a
moving floor. Within Numerics, work outward from the leaves: small utilities, then transforms,
then the validation wave, then individual distributions, then the multivariate and copula layer,
then MCMC. Within BestFit, the analysis-level bootstrap work goes last because it consumes
everything else.

One task per coherent slice, each with its own brief, its own fresh implementer, and its own fresh
reviewer. Keep a ledger file with one line per task recording the commit, the fix rounds, the gate
counts, and anything the task carried forward for a later task to own. The July 2026 ledger is
`.superpowers/sdd/progress.md`; several of the rules below exist because a carried note in that
ledger was picked up three tasks later.

## Step 6: the per-task loop

1. Read the SHIPPED C# for the slice. Not the brief, not the commit message.
2. Move the fixture first. Re-pin the changed oracle, or author the new case, and watch it fail.
   New values are harvested from the real library with `python3 tools/verify_oracles.py --dump`
   where the emitter supports the target, or curated from the cited C# test otherwise.
3. Port the C++ until all four runners agree: ctest, testthat, pytest, and the oracle gate.
4. Update the provenance header of every file touched to the new SHA.
5. Run the targeted gate while iterating; run the full R and Python suites once before committing.
6. Commit, then review with a fresh agent, then apply the review's fixes as a follow-up commit.
7. Record the outcome in the ledger, including anything the task could not close.

After any change to a class layout, rebuild R with `R CMD INSTALL --preclean corehydror`. This is
not optional. A field moving between a public and a private section counts as a layout change even
when the field count and types are unchanged; that exact case bit the Mixture task.

## Step 7: closeout

A dedicated closeout task, after the last porting task and before the docs task:

- Sweep the ledger for every carried-forward note and adjudicate each one. Port or document, no
  third option. The July 2026 closeout found three TimeSeries deltas carried forward from an
  earlier task, adjudicated all three as real library surface, and ported all three.
- Sweep the new API surface: every new enum arm, knob, or method must be exercised by a fixture in
  all four runners, or by a C++-only ctest where the fixture schema cannot express it
  (stateful mutation sequences, lambda delegates, pointer-identity checks, string returns). Any
  arm that ends up exercised by nothing gets documented as deliberately unused.
- Sweep for never-named upstream files: list the changed C# files that no task ever cited, and
  confirm each is a documented severance or an unported area.
- Run everything: ctest, the full gate, testthat, pytest. Record the numbers.
- Confirm the submodules are clean at the release tags.

## Step 8: docs, versions, ship

- Reconcile `docs/upstream-csharp-issues.md` in ONE pass at the end, not per task. Doing it per
  task produced an inconsistent file in July 2026: six tasks updated their entries inline and four
  deferred, and the deferred ones had to be reconstructed later. For every entry, check the shipped
  source at the new tag and append a Status bullet. Update the header pins. Keep the historical
  entries; the point of the file is the trail.
- Run the provenance sweep (below).
- Update the status sections of `.claude/CLAUDE.md` and `upstream/CLAUDE.md`: validated-against
  versions, final oracle counts, retired divergences, new severances.
- Bump the package versions in `corehydror/DESCRIPTION` and `corehydropy/pyproject.toml`, and
  record which upstream versions the release was validated against. Check whether any new export
  needs adding to `corehydror/_pkgdown.yml` and `site/_quarto.yml`; a sync usually adds no new
  export, but verify rather than assume.
- Push, watch CI to green on the full matrix, open the PR. Run the ship step as a fresh session
  after the last task has committed.

## The provenance sweep

Every ported file carries `// ported from: <csharp path> @ <sha>`. Tasks re-pin the files they
touch, which leaves every untouched file on the old SHA. That is a problem, because on the NEXT
sync a stale pin is indistinguishable from a file nobody has reviewed.

At closeout, re-pin every header whose upstream counterpart is byte-identical across the range.
The safe set is computable:

```bash
git -C upstream/Numerics    diff --name-only <old>..<newtag> > /tmp/num_changed.txt
git -C upstream/RMC-BestFit diff --name-only <old>..<newtag> > /tmp/bf_changed.txt
```

A file NOT in those lists did not change upstream, so re-pinning its header is provably safe. A
file that IS in the list and still carries the old pin is a finding: either a task missed a delta,
or the delta is under a documented severance. Adjudicate every one of them and write the verdict
down. Do not re-pin them silently. Leaving them on the old pin is the correct outcome for a
severed delta, because it makes the next sweep re-surface the severance.

Note that upstream paths can contain spaces. Split those file lists on newlines.

## Hard-won rules

### Port from the shipped source at the tag, never from a commit message

Upstream commit `1b424e3` announced that failed Bulletin 17C bootstrap replicates would be
discarded rather than substituted with the parent parameter vector. The design doc for this sync
propagated that description. It was wrong: `7efa9d0`, the last commit before the v2.0.0 tag,
reverted it for the parametric arm. Shipped v2.0.0 keeps the strict non-success gate and the
parent-vector fallback. Discard semantics shipped only in the pivot arm. Porting the announced
behavior would have produced a port that reproduces nothing.

It happened twice. The task after that one found the source contradicting even the corrected
brief, on two more points: there is no compaction before link fitting, and the z-limit is a clip
rather than a rejection. Source won both times, and the second point is why that arm's seeded
oracle is pinnable at all.

Read the file at the tag. When the brief and the source disagree, the source is right and the
brief gets corrected.

### The oracle-gate census is a lower bound, not the worklist

Ten fixtures failed at the bump, against roughly 27 classified risk areas. That is not evidence
that only ten things changed. The fixtures were curated from the OLD C# test suite, and upstream's
fixes target exactly the gaps those tests missed. A bug nobody tested is a bug nobody pinned, so
fixing it moves no oracle.

The census tells you which oracles to re-pin. The diff classification tells you what to port. Work
the classification; use the census as a cross-check in both directions, including "predicted risk
area with no census failure", which usually means missing coverage rather than an unaffected area.

### Prove chaotic sensitivity by measuring conditioning, never by asserting it

Short seeded MCMC chains and short bootstrap loops can amplify a sub-ulp difference into a visibly
different answer. That is a real phenomenon and it is also the most convenient excuse available for
a port bug. The difference has to be measured:

1. Perturb one input at the known cross-language agreement level, around 1e-13, and measure the
   displacement in the output.
2. Confirm a smaller perturbation, around 1e-15, is bit-identical.
3. Confirm the unperturbed deterministic path reproduces tightly.
4. Drive the REAL C# at the failing configuration before blaming the port.

Step 4 is the one people skip. In this run a reviewer raised a HIGH finding that the ported BFGS
had a defect, because the C++ stalled where C# was assumed to converge. Driving the real C# at the
same configuration showed it stalls identically: `BFGS.cs` declares a TOLX stagnation guard at
line 101 and never uses it, so there is no stagnation exit in either language. The reviewer
withdrew the finding, then independently established the inherence: a 1e-13 input perturbation
produced a 2.4e-5 displacement, an amplification of 2.4e8, and flipped the stall, while 1e-15 was
bit-identical and the unresampled parent fit agreed to 3e-14. The actual mechanism turned out to be
an absolute pass-to-pass tolerance straddle in the GMM iterative loop, 1.23e-8 against a 1e-8
threshold on one side and 3.3e-11 on the other, which is a documented accepted divergence with
reproducible numbers rather than a defect.

### A new arm or knob needs a fixture that exercises it

An enum arm with no fixture case gets no oracle, and a binding that nothing calls is a binding
nobody has checked. Every new arm needs a case in all four runners, or an explicit note saying why
the schema cannot express it and where the C++-only coverage lives.

The stronger version of this rule: check whether a new case would still pass against the OLD code.
Several new cases in this run did not discriminate until they were reworked, which means they were
testing that the code runs, not that it is right. Prove the discrimination by reverting the
implementation and watching the new case fail.

### Never loosen a tolerance or add an oracle skip to make a divergence go away

Deterministic surfaces pin exact. A divergence is either a port bug to fix or a documented,
measured, reproducible fidelity note. It is never a tolerance edit.

The one legitimate case for authoring a loose tolerance is a stochastic quantity, and
even then the reason has to be numeric rather than convenient. The example from this run is a
lattice-rule Monte Carlo value in the multivariate normal CDF, where the compiler optimization
level shifts the low bits, so the fixture carries a build-portable tolerance and says why.

Where a seeded short-chain curve cannot be pinned, assert the deterministic structural invariants
that do reproduce bit-identically across all four runners, with no skip mask and no loosened
tolerance on anything else.

### Provenance pins are load-bearing

A header still reading `@ <old-sha>` after a sync is indistinguishable, on the next sync, from a
file nobody reviewed. That is precisely how two TimeSeries deltas went un-owned in this run: they
sat in files whose pins had never moved, so nothing flagged them until a reviewer swept the
carried-forward notes at closeout.

Re-pin on touch, sweep the byte-identical remainder at closeout, and adjudicate the rest in
writing.

## Working practices

### Subagent crash recovery

Long tasks lose their agent, either to a dropped connection or to machine load. The pattern that
worked: resume the SAME agent at most twice, then stop resuming and hand a FRESH agent a
verify-and-land brief. Work in progress survives in the worktree, so the finisher's job is to
verify what is on disk, complete what is missing, and commit. Three tasks in this run needed it,
including one that survived two stalls and a handoff.

Do not let a crashed agent's partial claim about its own state enter the ledger unverified. The
finisher re-runs the suites.

### BestFit source files are not UTF-8

The working-tree copies contain mojibake. A `grep` over the checkout silently misses matches, which
in one task produced a wrong conclusion about which members a class has. Read BestFit sources with
`git show <tag>:<path>` and diff with `git diff`, not with a text search over the working tree.

### Workflow resume is unsafe once a task has committed

Resuming a multi-task workflow from a mid-run snapshot after commits have landed produces
duplicated or conflicting work. Run the ship step, and any resumed phase, as a fresh session.

## Candidate automation

Nothing here was automated in July 2026. These are the pieces that would have earned their keep,
in the order they hurt, as requirements for a future `tools/upstream_diff.py`:

1. **Provenance manifest and sweep.** Parse every `// ported from: <path> @ <sha>` in `core/`,
   resolve the cited path against the submodule at both pins, and emit three lists: safe to
   re-pin because the upstream file is byte-identical, changed upstream and therefore needing
   adjudication, and unresolvable because the cited path does not exist at the tag. Apply the safe
   set with `--apply`. This is a hundred lines and it replaces the most error-prone manual step. It
   would also have caught a citation in this repo that pointed at a filename which has never
   existed upstream.
2. **Cosmetic-versus-code diff classifier.** For each changed C# file, strip byte-order marks,
   comment-only lines, and pure whitespace, then report whether anything remains. In this run 27 of
   the 45 changed upstream files that kept an old pin were pure comment, BOM, XML-doc, or
   namespace churn, and separating them by hand cost real time. Namespace-only moves should get
   their own bucket, because they break the emitter but change no behavior.
3. **Never-named-file report.** Cross the changed-file list against every C# path cited anywhere in
   `core/`, and list the changed files nothing cites. That is the missed-delta detector, and it is
   two set operations.
4. **Census differ.** Run the gate before and after the bump and emit the failure list already
   grouped by fixture file and subsystem, with old and new values side by side, plus the two-way
   cross-check against a machine-readable version of the classification's risk list.
5. **New-upstream-test harvester.** List the test methods added or changed in the release's test
   tree, mapped to the library class each one exercises. Those methods are the oracle sources for
   the behavior changes, and finding them by hand is pure search.
6. **Conditioning probe.** A small harness that takes a fixture case and a perturbation size and
   reports the output displacement, so the chaotic-sensitivity rule above can be executed in one
   command instead of assembled by hand each time.

What should NOT be automated: the classification itself, and the decision about what a change
means. Both times the process nearly went wrong in this run, the failure was a plausible-sounding
description of a change being trusted over the shipped code. An automated summary is one more
plausible-sounding description.
