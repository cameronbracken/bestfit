# Shared-core organization: symlink vendoring, dereferenced at build

Date: 2026-07-08
Status: approved (design), pending implementation plan

## Problem

The canonical C++17 core lives in `core/` and the language-neutral oracle fixtures live in
`fixtures/`. Today `tools/sync_core.py` and `tools/sync_fixtures.py` _copy_ both into each package
and those copies are _committed_ to git:

- `bestfitr/src/bestfit_core/` (275 tracked files) and `bestfitr/inst/fixtures/`
- `bestfitpy/src/bestfit_core/` and `bestfitpy/src/bestfitpy/fixtures/`
- roughly 560 header/data files plus 286 fixture files, duplicated

A CI `sync-check` job runs the sync scripts with `--check` and fails on drift, so the copies never
diverge. The mechanism is safe but the same source exists three times in version control: every
core change is a 3x diff, pull requests are noisy, and the repository carries redundant content.

The goal is that _git holds exactly one copy_ of the core and the fixtures, with no loss of the
project's hard requirement.

## Constraints

- Publishing to _both CRAN and PyPI_ is non-negotiable. CRAN and PyPI source packages must be
  self-contained: they cannot reference a sibling `core/` directory, so the shipped tarball / sdist
  must physically contain the core source. This means vendoring cannot be removed, only relocated in
  time (materialized at build instead of committed).
- CRAN does not accept a package that links against a separately distributed `libbestfit`, and its
  policy discourages symlinks inside a package. The core is header-only C++17 with no external
  dependencies, so "consuming the core" means including its headers, not linking a shared object.
- The R package is checked on Linux, macOS, and Windows; the Python package is built on the same
  three platforms across two Python versions. Any solution must hold on all of them.

## Decision

Represent the vendored core and fixtures as _committed symbolic links_ into `core/` and `fixtures/`,
and _dereference them into real files at build time_ so every shipped artifact is symlink-free and
self-contained.

The key enabling fact: `R CMD build` dereferences symlinks when it assembles the source tarball. So
the rule "always `R CMD build` first, then install or check the resulting tarball" produces a
symlink-free, self-contained `.tar.gz` and sidesteps both the Windows checkout issue and CRAN's
no-symlinks rule in a single step. Python has no equivalent auto-dereference, so its build path
gets an explicit materialize step.

### Alternatives considered

- _Committed copies (status quo)._ Rejected: this is the duplication being removed.
- _Gitignore the vendored dirs, generate copies at build (Approach A)._ Viable and simplest, but it
  breaks `remotes::install_github(subdir = "bestfitr")` against the raw tree (the subdir tarball
  would not contain `core/`), forcing installs through CRAN / PyPI / release tarballs only. Symlinks
  preserve `install_github` because the whole repo (including `core/`) is downloaded and the link
  resolves.
- _Separate library in its own repo, packages as thin bindings._ Rejected for now. CRAN still forces
  the R package to vendor the source, so a repo split does not remove the copy; it moves it across a
  repo boundary and adds cross-repo release choreography and loss of atomic core+bindings+fixtures
  commits. Revisit only if genuine external consumers of the core appear. The design keeps `core/`
  extraction-ready so that future move is cheap (see section 5).

## Design

### 1. Git representation

`core/` and `fixtures/` are the sole copies in version control. Each package's vendored location
symlinks only the _specific subtrees the sync scripts copy today_ (`core/include`, `core/data`, and
the fixtures), not the whole `core/` directory. Linking all of `core/` would expose `core/tests`,
`core/build`, and `core/CMakeLists.txt`, which `R CMD build` would then dereference into the tarball
(bloat and a likely CRAN NOTE). So `bestfit_core/` stays a real directory holding subtree symlinks:

- `bestfitr/src/bestfit_core/include` -> `../../../core/include`
- `bestfitr/src/bestfit_core/data` -> `../../../core/data`
- `bestfitr/inst/fixtures` -> `../../fixtures`
- `bestfitpy/src/bestfit_core/include` -> `../../../../core/include`
- `bestfitpy/src/bestfit_core/data` -> `../../../../core/data`
- `bestfitpy/src/bestfitpy/fixtures` -> the corresponding `fixtures/` relative path

The existing copied files are removed with `git rm -r` and replaced by these symlinks, which are
tracked (that is what keeps `install_github` resolving). No `.gitignore` entry for vendored content
is needed. The exact relative depths and any include-path adjustments (Makevars, CMake) are verified
in the implementation plan against the current `tools/sync_core.py` destinations; the principle is
that only the copied subtrees are linked, so the dereferenced tarball contains only needed source.

### 2. Golden rule: never build from the raw symlinked tree

Every install, check, or publish operates on a symlink-free artifact, never on the raw working tree.

- _R:_ `R CMD build bestfitr` dereferences the symlinks into the tarball. `R CMD check --as-cran`
  and any install then run against that self-contained `.tar.gz`. This is the first step of any R
  CI or release job.
- _Python:_ `pip install ./bestfitpy` compiles a wheel, and on Linux/macOS the symlink resolves at
  build time, so a quick local wheel build needs nothing extra. But the _sdist_ must install in an
  environment with no monorepo present, and Windows checkouts may not materialize symlinks, so the
  Python build path runs a materialize step first (`tools/materialize_core.py`, see section 6) that
  dereferences the symlinks into real files in place. The plan confirms whether scikit-build-core's
  sdist can force-include the dereferenced target directly, in which case the explicit step is only
  needed for Windows / editable-dev on Windows.

### 3. CI

- Remove the `sync-check` job entirely. Nothing can drift when the symlink _is_ the link.
- `core` job: unchanged. `ctest` builds `core/` directly.
- `r-cmd-check` job: build-first. The `r-lib/actions` check step already runs `R CMD build` before
  `R CMD check`, so the tarball is dereferenced for free; confirm and make it explicit.
- `python` job: run `tools/materialize_core.py` as step 1, then build and test.

Net effect: one fewer job and no drift gate; the build-first discipline replaces the drift check.

### 4. Developer loop

Editing `core/` is live: R and Python pick up the change on the next build through the symlink
(Linux/macOS). A root `Makefile` (or `justfile`) provides the single entry point so no one has to
remember the golden rule by hand:

- `make test-core` - configure + build + ctest
- `make test-r` - `R CMD build` + install the tarball + testthat
- `make test-py` - materialize + build/install + pytest
- `make build-r` / `make build-py` - the release tarball / sdist + wheel
- `make materialize` - dereference symlinks into real files (for a symlink-free tree)
- `make oracles` - `python3 tools/verify_oracles.py`

### 5. Core extraction-readiness (light)

Honoring "maybe split later, not now", make a future extraction cheap:

- Add a version stamp: `core/include/bestfit/version.hpp` with a `BESTFIT_CORE_VERSION` macro (or a
  `core/VERSION` file), so bindings can record which core they built against and a future release
  has a version to publish.
- Make `core/CMakeLists.txt` expose a consumable `bestfit::core` INTERFACE target that sets the
  include directory, so a C++ consumer or a future split can `add_subdirectory(core)` or use CMake
  `FetchContent` with no restructuring.
- Add a short `core/README.md` documenting that the core is a standalone header-only library and how
  to consume it (add_subdirectory / vendor via the materialize script).

These are additive and do not change any numerical behavior.

### 6. `sync_core.py` becomes `materialize_core.py`

The sync scripts are repurposed, not deleted. Instead of "copy `core/` into the committed vendored
dirs", the new `tools/materialize_core.py` dereferences the symlinks into real files in place (a
symlink-free tree for a build). `sync_fixtures.py` folds in the same way. The scripts keep their
role as the mechanism that produces a self-contained tree; only the destination representation
changes from committed copies to on-demand materialization.

## Acceptance criteria

1. `git ls-files` shows only symlinks at the vendored locations, no copied core or fixture content.
2. `R CMD check --as-cran` is clean on Linux, macOS, and Windows, built from a self-contained
   tarball that contains no symlinks.
3. A Python sdist installs in a clean environment with no monorepo present.
4. The existing `ctest`, `testthat`, `pytest`, and `verify_oracles` suites all stay green, and the
   cross-language bit-identical guarantee (seeded results identical across R and Python) still holds.
5. `remotes::install_github(subdir = "bestfitr")` and the Python equivalent from the repo still
   resolve (the committed symlinks resolve against the full downloaded repo).

## Migration outline (detailed in the plan)

1. Create the symlinks; `git rm -r` the copied dirs; commit.
2. Adjust include paths / Makevars / CMake if the symlink layout requires it.
3. Add `tools/materialize_core.py` (repurpose `sync_core.py` + `sync_fixtures.py`).
4. Update CI: drop `sync-check`; add the materialize step to the Python job; confirm build-first for R.
5. Add the root Makefile / justfile.
6. Add the core version stamp, consumable CMake target, and `core/README.md`.
7. Verify all acceptance criteria on all three platforms.
8. Update `.claude/CLAUDE.md` and the script docstrings to describe the symlink model.

## Out of scope

- Actual CRAN / PyPI submission (R CMD check --as-cran polish, cibuildwheel matrix, doc sites,
  vignettes, submission mechanics). That is a separate publishing project.
- Splitting the core into its own repository. Deferred; this design only makes it cheap later.
