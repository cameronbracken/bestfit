# Shared-core symlink vendoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the committed vendored copies of the C++ core and oracle fixtures with committed subtree symlinks into `core/` and `fixtures/`, so git holds one copy while every shipped tarball/sdist stays self-contained.

**Architecture:** Each package's vendored location becomes a committed symlink into `core/include`, `core/data`, or `fixtures/`. Builds never run from the raw symlinked tree: `R CMD build` dereferences symlinks into the R tarball automatically, and the Python build runs a `materialize` step (dereference symlinks into real files) in a throwaway `git worktree` first. CI drops the drift-check job; a root Makefile is the dev/CI entry point. `core/` gets a light version stamp + consumable CMake target so a future repo split is cheap.

**Tech Stack:** POSIX symlinks, Python 3 (materialize script), R (`R CMD build`/`check`), scikit-build-core + pybind11 (Python), CMake, GitHub Actions.

## Global Constraints

- CRAN and PyPI publishing are both non-negotiable; every shipped tarball/sdist must be self-contained and symlink-free. (spec: Constraints)
- Only the copied subtrees are symlinked (`core/include`, `core/data`, `fixtures/`), never the whole `core/` dir, so `core/tests`/`core/build`/`CMakeLists.txt` never land in a tarball. (spec: Design 1)
- `bestfitr/src/Makevars` and `Makevars.win` set `PKG_CPPFLAGS = -I./bestfit_core/include`; the include path must keep resolving. (verified)
- No `M_PI`/behavior changes to `core/`; the extraction-readiness additions are purely additive. (spec: Design 5)
- NO Co-Authored-By / AI-attribution trailer on any commit. Commits are GPG-signed automatically; identity `Cam Bracken <cameron.bracken@pm.me>`; push only when asked.
- The existing `ctest`, `testthat`, `pytest`, and `verify_oracles` suites must stay green; the cross-language bit-identical guarantee must hold. (spec: Acceptance)
- The eight vendored destinations that must become symlinks (from `tools/sync_core.py` + `tools/sync_fixtures.py`):
  1. `bestfitr/src/bestfit_core/include` -> `../../../core/include`
  2. `bestfitr/src/bestfit_core/data` -> `../../../core/data`
  3. `bestfitr/inst/extdata` -> `../../core/data`
  4. `bestfitr/inst/fixtures` -> `../../fixtures`
  5. `bestfitpy/src/bestfit_core/include` -> `../../../core/include`
  6. `bestfitpy/src/bestfit_core/data` -> `../../../core/data`
  7. `bestfitpy/src/bestfitpy/data` -> `../../../core/data`
  8. `bestfitpy/src/bestfitpy/fixtures` -> `../../../fixtures`

---

## File map

- Create `tools/materialize_core.py` - dereference: copy `core/{include,data}` + `fixtures/` into all 8 vendored destinations, replacing symlinks with real dirs. Supersedes `tools/sync_core.py` + `tools/sync_fixtures.py`.
- Delete `tools/sync_core.py`, `tools/sync_fixtures.py` (Task 7).
- Replace 8 committed directories with symlinks (Task 2).
- Create `bestfitr/.Rbuildignore` - exclude local build cruft from the R tarball (Task 3).
- Modify `.github/workflows/ci.yml` - drop `sync-check`, add materialize to the python job (Task 5).
- Create `Makefile` (repo root) - dev/CI entry point (Task 5).
- Create `core/include/bestfit/version.hpp`, modify `core/CMakeLists.txt`, create `core/README.md` (Task 6).
- Modify `.claude/CLAUDE.md` and any lingering `sync_core`/`sync_fixtures` references (Task 7).

---

### Task 1: `materialize_core.py` (symlink-aware dereference tool)

**Files:**
- Create: `tools/materialize_core.py`
- Test: run it against the repo (manual verification steps below)

**Interfaces:**
- Produces: `python3 tools/materialize_core.py [--check]` - copies `core/include`, `core/data`, and `fixtures/` into the 8 destinations listed in Global Constraints, replacing a symlink destination with a real directory copy. `--check` reports staleness without writing (used only transitionally; removed conceptually once copies are symlinks). Exit 0 on success.

- [ ] **Step 1: Write the failing verification**

There is no committed unit test; the deliverable is a script whose behavior is verified by running it. Write this check script inline and run it after implementation:

```bash
# after implementation, this must print OK
python3 tools/materialize_core.py && \
diff -r core/include bestfitr/src/bestfit_core/include >/dev/null && \
diff -r fixtures bestfitpy/src/bestfitpy/fixtures >/dev/null && echo OK
```

- [ ] **Step 2: Run it to confirm the tool does not exist yet**

Run: `python3 tools/materialize_core.py`
Expected: FAIL with `No such file or directory` / `can't open file`.

- [ ] **Step 3: Implement `tools/materialize_core.py`**

```python
#!/usr/bin/env python3
"""Materialize the shared core + fixtures into each package as REAL files.

The single sources of truth are ``core/`` and ``fixtures/``. In version control each
package's vendored location is a symlink into them. For a self-contained build
(Python sdist/wheel, Windows, any offline consumer) the symlinks must be dereferenced
into real files -- that is what this script does, in place, replacing a symlink
destination with a real directory copy.

R does not need this: ``R CMD build`` dereferences symlinks into the tarball itself.
Run this in a throwaway checkout / CI runner; it rewrites the working tree.

Usage:  python3 tools/materialize_core.py [--check]
  --check : exit non-zero if any destination is stale or still a symlink (informational).
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "core"
FIXTURES = ROOT / "fixtures"

# (source dir, destination dir) pairs. Mirrors the 8 vendored destinations.
PAIRS: list[tuple[Path, Path]] = [
    (CORE / "include", ROOT / "bestfitr" / "src" / "bestfit_core" / "include"),
    (CORE / "data", ROOT / "bestfitr" / "src" / "bestfit_core" / "data"),
    (CORE / "data", ROOT / "bestfitr" / "inst" / "extdata"),
    (FIXTURES, ROOT / "bestfitr" / "inst" / "fixtures"),
    (CORE / "include", ROOT / "bestfitpy" / "src" / "bestfit_core" / "include"),
    (CORE / "data", ROOT / "bestfitpy" / "src" / "bestfit_core" / "data"),
    (CORE / "data", ROOT / "bestfitpy" / "src" / "bestfitpy" / "data"),
    (FIXTURES, ROOT / "bestfitpy" / "src" / "bestfitpy" / "fixtures"),
]


def dirs_equal(a: Path, b: Path) -> bool:
    if b.is_symlink() or not b.exists():
        return False
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.diff_files or cmp.funny_files:
        return False
    return all(dirs_equal(a / sub, b / sub) for sub in cmp.common_dirs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report staleness / remaining symlinks; make no changes")
    args = ap.parse_args()

    stale = False
    for src, dest in PAIRS:
        # A subtree with no files (e.g. an empty core/data) is skipped: shipping an
        # empty dir triggers a spurious R CMD check NOTE.
        if not src.exists() or not any(p.is_file() for p in src.rglob("*")):
            continue
        if args.check:
            if dest.is_symlink() or not dirs_equal(src, dest):
                print(f"NOT MATERIALIZED: {dest}")
                stale = True
            continue
        if dest.is_symlink():
            dest.unlink()
        elif dest.is_file():
            # Windows: a committed symlink can be checked out as a small text stub
            # (a regular file holding the link target). Treat it like a symlink.
            dest.unlink()
        elif dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest)
        print(f"materialized {src} -> {dest}")

    if args.check and stale:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the verification**

Run: `python3 tools/materialize_core.py && diff -r core/include bestfitr/src/bestfit_core/include >/dev/null && diff -r fixtures bestfitpy/src/bestfitpy/fixtures >/dev/null && echo OK`
Expected: prints 8 `materialized ...` lines then `OK`. (At this point the destinations are still the committed real copies; materialize overwrites them identically, so `git status` shows no change.)

- [ ] **Step 5: Commit**

```bash
git add tools/materialize_core.py
git commit -m "build: add materialize_core.py (symlink-aware dereference of core + fixtures)"
```

---

### Task 2: Replace committed copies with subtree symlinks

**Files:**
- Delete (from git): the 8 committed destination directories' contents.
- Create: the 8 symlinks (Global Constraints list).

**Interfaces:**
- Consumes: `tools/materialize_core.py` (to regenerate real files for builds).
- Produces: the vendored locations are now symlinks; `git ls-files` shows no copied core/fixture content.

- [ ] **Step 1: Write the failing verification**

```bash
# after the change this must print OK
test -L bestfitr/src/bestfit_core/include && \
test -L bestfitpy/src/bestfitpy/fixtures && \
[ "$(git ls-files bestfitr/src/bestfit_core/include | wc -l | tr -d ' ')" = "1" ] && echo OK
```

- [ ] **Step 2: Run it to confirm copies are still committed**

Run the check from Step 1.
Expected: FAIL (the paths are real dirs with many tracked files, not symlinks).

- [ ] **Step 3: Remove the committed copies and create the symlinks**

```bash
# 1. untrack + remove the committed copy content
git rm -r --quiet \
  bestfitr/src/bestfit_core/include bestfitr/src/bestfit_core/data \
  bestfitr/inst/extdata bestfitr/inst/fixtures \
  bestfitpy/src/bestfit_core/include bestfitpy/src/bestfit_core/data \
  bestfitpy/src/bestfitpy/data bestfitpy/src/bestfitpy/fixtures

# 2. create the 8 subtree symlinks (target paths are relative to each symlink's dir)
ln -s ../../../core/include bestfitr/src/bestfit_core/include
ln -s ../../../core/data    bestfitr/src/bestfit_core/data
ln -s ../../core/data       bestfitr/inst/extdata
ln -s ../../fixtures        bestfitr/inst/fixtures
ln -s ../../../core/include bestfitpy/src/bestfit_core/include
ln -s ../../../core/data    bestfitpy/src/bestfit_core/data
ln -s ../../../core/data    bestfitpy/src/bestfitpy/data
ln -s ../../../fixtures     bestfitpy/src/bestfitpy/fixtures

git add bestfitr/src/bestfit_core/include bestfitr/src/bestfit_core/data \
  bestfitr/inst/extdata bestfitr/inst/fixtures \
  bestfitpy/src/bestfit_core/include bestfitpy/src/bestfit_core/data \
  bestfitpy/src/bestfitpy/data bestfitpy/src/bestfitpy/fixtures
```

- [ ] **Step 4: Verify symlinks resolve and the core still builds**

```bash
# every symlink resolves to the expected real dir
for l in bestfitr/src/bestfit_core/include bestfitr/src/bestfit_core/data \
         bestfitr/inst/extdata bestfitr/inst/fixtures \
         bestfitpy/src/bestfit_core/include bestfitpy/src/bestfit_core/data \
         bestfitpy/src/bestfitpy/data bestfitpy/src/bestfitpy/fixtures; do
  test -e "$l/" || test -f "$(readlink -f "$l")" || { echo "BROKEN: $l"; exit 1; }
done
# a header is reachable through the R include symlink
test -f bestfitr/src/bestfit_core/include/bestfit/version.hpp -o \
     -f bestfitr/src/bestfit_core/include/bestfit/numerics/tools.hpp && echo LINKS_OK
# the canonical core still builds + tests (unaffected: it builds from core/ directly)
cmake --build core/build && ctest --test-dir core/build 2>&1 | grep -E 'tests passed'
```
Expected: `LINKS_OK` and `100% tests passed`. Also run the Step-1 check: prints `OK`.

- [ ] **Step 5: Commit**

```bash
git commit -m "build: vendor the core + fixtures as subtree symlinks, not committed copies

Removes the ~560 committed core files + 286 fixture copies. Each package's
vendored location is now a symlink into core/{include,data} or fixtures/.
Builds dereference them (R CMD build for R; materialize_core.py for Python)."
```

---

### Task 3: R build path - `.Rbuildignore` + self-contained dereferenced tarball

**Files:**
- Create: `bestfitr/.Rbuildignore`
- Verify: `R CMD build bestfitr` produces a symlink-free, self-contained tarball that `R CMD check` accepts.

**Interfaces:**
- Consumes: the symlinks from Task 2.
- Produces: a validated R release path (build-then-check).

- [ ] **Step 1: Write the failing verification**

```bash
# after this task, building the tarball and inspecting it must print OK
R CMD build bestfitr >/tmp/rbuild.log 2>&1 && \
TB=$(ls -t bestfitr_*.tar.gz | head -1) && \
tar tzf "$TB" | grep -q 'bestfitr/src/bestfit_core/include/bestfit/' && \
! tar tvzf "$TB" | grep -q '^l' && echo OK
```
(`^l` in `tar tvzf` output marks a symlink entry; there must be none.)

- [ ] **Step 2: Run it to see the current failure mode**

Run the Step-1 block.
Expected: it may FAIL because (a) local `src/*.o`/`*.so` build cruft bloats or breaks the tarball, or (b) `R CMD build` is not yet confirmed to dereference. This step establishes the baseline before adding `.Rbuildignore`.

- [ ] **Step 3: Add `bestfitr/.Rbuildignore`**

```
^.*\.o$
^.*\.so$
^src/bestfitr\.so$
^\.\.Rcheck$
^.*\.Rproj$
^\.Rproj\.user$
```

- [ ] **Step 4: Build the tarball and verify it is self-contained + symlink-free**

```bash
R CMD build bestfitr
TB=$(ls -t bestfitr_*.tar.gz | head -1)
# (a) the core headers are present as real files inside the tarball
tar tzf "$TB" | grep -q 'bestfitr/src/bestfit_core/include/bestfit/'
# (b) no symlink entries survived
tar tvzf "$TB" | grep '^l' && echo "FAIL: tarball has symlinks" || echo "no symlinks"
# (c) the tarball installs and checks
R CMD check "$TB" 2>&1 | tail -20
```
Expected: header path present; `no symlinks`; `R CMD check` ends with `Status: OK` (or only pre-existing NOTEs unrelated to this change). If `R CMD build` does NOT dereference on this platform, STOP and report - the design assumption failed and needs revisiting.

- [ ] **Step 5: Commit**

```bash
git add bestfitr/.Rbuildignore
git commit -m "build(r): .Rbuildignore for a clean self-contained tarball; verify R CMD build dereferences symlinks"
```

---

### Task 4: Python build path - materialize-first, self-contained sdist

**Files:**
- Verify/modify: `bestfitpy/pyproject.toml` (scikit-build-core `sdist.include` if needed).
- Verify: a materialized sdist installs in a clean environment with no monorepo present.

**Interfaces:**
- Consumes: the symlinks (Task 2), `tools/materialize_core.py` (Task 1).
- Produces: a validated Python release path.

- [ ] **Step 1: Write the failing verification (clean-env sdist install)**

```bash
# after this task the following must print IMPORT_OK, run from a throwaway worktree so
# materialize never clobbers the dev tree's symlinks.
git worktree add -q /tmp/bf-sdist HEAD
( cd /tmp/bf-sdist && python3 tools/materialize_core.py >/dev/null &&
  python3 -m build --sdist bestfitpy -o /tmp/bf-dist >/tmp/bf-sdist.log 2>&1 )
python3 -m venv /tmp/bf-venv && /tmp/bf-venv/bin/pip -q install numpy pytest &&
/tmp/bf-venv/bin/pip -q install /tmp/bf-dist/bestfitpy-*.tar.gz &&
/tmp/bf-venv/bin/python -c "import bestfitpy; print('IMPORT_OK')"
git worktree remove --force /tmp/bf-sdist
```

- [ ] **Step 2: Run it to see whether the sdist is self-contained**

Run the Step-1 block.
Expected: it may FAIL at the `pip install` of the sdist if scikit-build-core did not include the materialized `bestfit_core` in the sdist (missing headers at compile time). This tells you whether Step 3 is needed.

- [ ] **Step 3: If the sdist is missing the core, force-include it**

Only if Step 2 failed. Add to `bestfitpy/pyproject.toml` under `[tool.scikit-build]`:

```toml
sdist.include = [
  "src/bestfit_core/include/**",
  "src/bestfit_core/data/**",
  "src/bestfitpy/data/**",
  "src/bestfitpy/fixtures/**",
]
```
(These paths are real directories after `materialize_core.py` runs, so scikit-build-core packs them into the sdist.)

- [ ] **Step 4: Re-run the clean-env verification**

Run the Step-1 block again.
Expected: `IMPORT_OK`. Then confirm the in-tree editable/dev build still works via the symlink on this platform:

```bash
python3 -m pip install --force-reinstall --no-deps ./bestfitpy && \
~/venv/bestfitpy/bin/python -m pytest bestfitpy/tests -q 2>&1 | tail -3
```
Expected: pytest passes (the symlink resolves at compile time on Linux/macOS).

- [ ] **Step 5: Commit**

```bash
git add bestfitpy/pyproject.toml
git commit -m "build(py): self-contained sdist via materialize + scikit-build-core sdist.include"
```
(If Step 3 was not needed, commit only a docs note in the report that the sdist was already self-contained and skip the pyproject change.)

---

### Task 5: Root Makefile + CI update

**Files:**
- Create: `Makefile` (repo root)
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `materialize_core.py`, the symlinks.
- Produces: `make {test-core,test-r,test-py,build-r,build-py,materialize,oracles}`; a CI without the `sync-check` job.

- [ ] **Step 1: Write the failing verification**

```bash
# after this task:
make test-core 2>&1 | grep -E 'tests passed' && \
! grep -q 'sync-check' .github/workflows/ci.yml && echo OK
```

- [ ] **Step 2: Run it to confirm no Makefile + sync-check still present**

Run the Step-1 block.
Expected: FAIL (`make: *** No rule` and `sync-check` still in the yaml).

- [ ] **Step 3: Create the root `Makefile`**

```makefile
# Dev / CI entry points. core/ + fixtures/ are the single source of truth; each
# package vendors them via subtree symlinks (see docs/superpowers/specs). Builds
# dereference the symlinks: R CMD build does it automatically, Python via
# tools/materialize_core.py in a throwaway worktree.
.PHONY: test-core test-r test-py build-r build-py materialize oracles

test-core:
	cmake -S core -B core/build && cmake --build core/build && ctest --test-dir core/build

test-r:
	R CMD build bestfitr
	R CMD INSTALL --preclean $$(ls -t bestfitr_*.tar.gz | head -1)
	Rscript -e 'testthat::test_local("bestfitr")'

test-py:
	python3 -m pip install --force-reinstall --no-deps ./bestfitpy
	python3 -m pytest bestfitpy/tests -q

build-r:
	R CMD build bestfitr

build-py:
	git worktree add -q /tmp/bf-buildpy HEAD
	cd /tmp/bf-buildpy && python3 tools/materialize_core.py && python3 -m build bestfitpy -o $(CURDIR)/dist
	git worktree remove --force /tmp/bf-buildpy

materialize:
	python3 tools/materialize_core.py

oracles:
	python3 tools/verify_oracles.py
```

- [ ] **Step 4: Update `.github/workflows/ci.yml`**

Remove the `sync-check` job entirely and the `needs: sync-check` line from every job that had it (`core`, `r-cmd-check`, `python`).

Add a materialize step to BOTH the `python` job and the `r-cmd-check` job, on all platforms. Rationale: on Windows runners `actions/checkout` may materialize a committed symlink as a text stub rather than a real link, which would break the built tarball. Materializing real files from `core/`+`fixtures/` first makes every OS deterministic. (The `core` job needs no materialize: it builds `core/` directly.)

In the `python` job, place it immediately before the existing `Build and install` step (`python -m pip install ./bestfitpy`):

```yaml
      - name: Materialize shared core (dereference symlinks)
        run: python3 tools/materialize_core.py
```

In the `r-cmd-check` job, add it immediately after the `actions/checkout@v4` step, overriding the job's `working-directory: bestfitr` default so it runs from the repo root (Python 3 is preinstalled on all GitHub runners):

```yaml
      - name: Materialize shared core (dereference symlinks)
        working-directory: ${{ github.workspace }}
        run: python3 tools/materialize_core.py
```

The `r-lib/actions/check-r-package` step then runs `R CMD build` on a tree of real files, producing a self-contained, symlink-free tarball on every OS.

- [ ] **Step 5: Verify + commit**

```bash
make test-core 2>&1 | grep -E 'tests passed'
python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('YAML_OK')"
git add Makefile .github/workflows/ci.yml
git commit -m "build: root Makefile entry points; drop CI sync-check, materialize in the python job"
```
Expected: `100% tests passed`, `YAML_OK`.

---

### Task 6: Core extraction-readiness (version stamp + consumable CMake + README)

**Files:**
- Create: `core/include/bestfit/version.hpp`
- Modify: `core/CMakeLists.txt`
- Create: `core/README.md`

**Interfaces:**
- Produces: `BESTFIT_CORE_VERSION` macro; a `bestfit::core` INTERFACE CMake target consumable via `add_subdirectory(core)` / FetchContent.

- [ ] **Step 1: Write the failing verification (a scratch consumer compiles against `bestfit::core`)**

```bash
# after this task the scratch consumer must print CONSUMER_OK. The repo path is
# passed as a CMake cache variable (no in-file substitution).
mkdir -p /tmp/bfc && cat > /tmp/bfc/CMakeLists.txt <<'EOF'
cmake_minimum_required(VERSION 3.15)
project(bfc CXX)
add_subdirectory(${BESTFIT_REPO}/core core_build)
add_executable(bfc main.cpp)
target_link_libraries(bfc PRIVATE bestfit::core)
EOF
cat > /tmp/bfc/main.cpp <<'EOF'
#include "bestfit/version.hpp"
#include <cstdio>
int main() { std::printf("CONSUMER_OK %s\n", BESTFIT_CORE_VERSION); }
EOF
cmake -S /tmp/bfc -B /tmp/bfc/build -DBESTFIT_REPO="$(pwd)" >/dev/null 2>&1 && \
  cmake --build /tmp/bfc/build >/dev/null 2>&1 && /tmp/bfc/build/bfc
```

- [ ] **Step 2: Run it to confirm no version header + no target**

Run the Step-1 block.
Expected: FAIL (`bestfit/version.hpp` not found and/or `bestfit::core` target unknown).

- [ ] **Step 3: Add the version header**

Create `core/include/bestfit/version.hpp`:

```cpp
// ported from: (bestfit-native) -- shared-core version stamp
#ifndef BESTFIT_VERSION_HPP
#define BESTFIT_VERSION_HPP

// Single source of truth for the C++ core version. Bindings may record this to
// report which core they were built against; a future standalone release uses it.
#define BESTFIT_CORE_VERSION_MAJOR 0
#define BESTFIT_CORE_VERSION_MINOR 1
#define BESTFIT_CORE_VERSION_PATCH 0
#define BESTFIT_CORE_VERSION "0.1.0"

#endif  // BESTFIT_VERSION_HPP
```

- [ ] **Step 4: Expose a consumable `bestfit::core` INTERFACE target**

Add to `core/CMakeLists.txt` (append; do not disturb the existing test wiring):

```cmake
# Header-only consumable target so external consumers / a future repo split can
# `add_subdirectory(core)` or FetchContent and link `bestfit::core`.
add_library(bestfit_core INTERFACE)
add_library(bestfit::core ALIAS bestfit_core)
target_include_directories(bestfit_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(bestfit_core INTERFACE cxx_std_17)
```

- [ ] **Step 5: Add `core/README.md`**

```markdown
# bestfit core

The canonical, header-only C++17 implementation shared by the `bestfitr` (R) and
`bestfitpy` (Python) packages. No external dependencies.

## Consuming the core

CMake, in-tree or via FetchContent:

    add_subdirectory(core)          # or FetchContent
    target_link_libraries(app PRIVATE bestfit::core)

The R and Python packages vendor the headers as subtree symlinks into
`core/include` and `core/data`; a build dereferences them (R CMD build for R,
`tools/materialize_core.py` for Python). See
`docs/superpowers/specs/2026-07-08-shared-core-symlink-vendoring-design.md`.
```

- [ ] **Step 6: Verify + commit**

```bash
# scratch consumer compiles + links the header-only core
cmake -S /tmp/bfc -B /tmp/bfc/build && cmake --build /tmp/bfc/build && /tmp/bfc/build/bfc
# the core's own tests still build (the new target is additive)
cmake -S core -B core/build && cmake --build core/build && ctest --test-dir core/build 2>&1 | grep 'tests passed'
git add core/include/bestfit/version.hpp core/CMakeLists.txt core/README.md
git commit -m "feat(core): version stamp + consumable bestfit::core CMake target + README (extraction-ready)"
```
Expected: `CONSUMER_OK 0.1.0`, `100% tests passed`.

---

### Task 7: Retire the old sync scripts + docs

**Files:**
- Delete: `tools/sync_core.py`, `tools/sync_fixtures.py`
- Modify: `.claude/CLAUDE.md` (the sync commands + vendoring-invariant section), and any other lingering references.

**Interfaces:**
- Consumes: everything above.
- Produces: a repo whose docs describe the symlink model and whose only vendoring tool is `materialize_core.py`.

- [ ] **Step 1: Find every reference to the old scripts**

```bash
grep -rn 'sync_core\|sync_fixtures' --include='*.md' --include='*.yml' --include='*.py' \
  --include='*.R' --include='Makefile' . | grep -v '/upstream/' | grep -v 'materialize_core'
```
Expected: a list of hits in `.claude/CLAUDE.md`, possibly `docs/`, and the `sync-check` remnants (already gone from ci.yml in Task 5). Note them.

- [ ] **Step 2: Delete the superseded scripts**

```bash
git rm tools/sync_core.py tools/sync_fixtures.py
```

- [ ] **Step 3: Update `.claude/CLAUDE.md`**

In the "Layout & the vendoring invariant" and "Build & test commands" sections, replace the `sync_core.py`/`sync_fixtures.py` description with the symlink model: the vendored dirs are subtree symlinks into `core/`+`fixtures/`; builds dereference them (`R CMD build` for R; `python3 tools/materialize_core.py` for Python, run via `make build-py` in a throwaway worktree); the CI `sync-check` gate is gone. Replace the two `sync ... --check` lines in the "sync guards" block with:

```
# builds are self-contained via symlink dereference; regenerate real files with:
python3 tools/materialize_core.py    # (CI/release only; rewrites the working tree)
```

- [ ] **Step 4: Verify no dangling references + full suite still green**

```bash
grep -rn 'sync_core\|sync_fixtures' --include='*.md' --include='*.yml' . | grep -v '/upstream/' && echo "STILL REFERENCED" || echo "clean"
make test-core 2>&1 | grep 'tests passed'
```
Expected: `clean` and `100% tests passed`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs: describe the symlink vendoring model; retire sync_core/sync_fixtures"
```

---

## Final verification (run before opening the PR)

Confirm every acceptance criterion from the spec:

```bash
# 1. git holds only symlinks at vendored locations
git ls-files bestfitr/src/bestfit_core bestfitpy/src/bestfit_core \
  bestfitr/inst/fixtures bestfitpy/src/bestfitpy/fixtures | wc -l   # expect small (the symlinks), not hundreds
# 2. R tarball is self-contained + symlink-free (Task 3 block)
# 3. Python sdist installs clean (Task 4 block)
# 4. suites green
cmake --build core/build && ctest --test-dir core/build 2>&1 | grep 'tests passed'
make test-r ; make test-py
python3 tools/verify_oracles.py 2>&1 | tail -1
# 5. install_github-style resolution: a fresh clone has resolving symlinks
git clone --quiet . /tmp/bf-clone && test -f /tmp/bf-clone/bestfitr/src/bestfit_core/include/bestfit/version.hpp && echo CLONE_OK ; rm -rf /tmp/bf-clone
```
