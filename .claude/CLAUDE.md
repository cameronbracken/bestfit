# CLAUDE.md — bestfit

Context for Claude Code working in the `bestfit` repo. See `PLAN.md` (same dir) for the
full approved architecture and phasing.

## What this is

`bestfit` provides **R (`bestfitr`) and Python (`bestfitpy`) packages** for stochastic
hydrology / flood-frequency / extreme-value analysis, built on a **single shared C++17 core**
that is a faithful port of the USACE-RMC C# libraries **Numerics** and **RMC.BestFit**.

Write the math once in C++, bind it twice (cpp11 for R, pybind11 for Python). Because both
packages run the same compiled code with a bit-exact Mersenne Twister, seeded results are
identical across R and Python. Hard requirement: **publishable to CRAN and PyPI**.

Upstream C# sources are vendored as **dev-only git submodules** at `upstream/Numerics` and
`upstream/RMC-BestFit` (shallow, tracking the official USACE-RMC `main` branches; pinned via
gitlink). They are the diff baseline for the upstream-sync workflow and are NOT referenced by
either package (CRAN/PyPI sdists and CI are unaffected; `actions/checkout` uses `submodules:
false`). See `upstream/CLAUDE.md` for the per-library architecture notes (consolidated there and
tracked by this repo). `dotnet` **is now installed**, so oracle values are curated from the C#
test files AND verified reproducible against the real Numerics library (see below). (The user's
own forks live elsewhere on disk under different names.)

## Layout & the vendoring invariant

- `core/` — **canonical** C++17 core (`include/`, `src/`, `tests/`, `CMakeLists.txt`). All
  numerical development happens here.
- `fixtures/` — **canonical** language-neutral oracle fixtures (JSON). Single source of truth
  for expected values; see `fixtures/README.md` for the schema.
- `bestfitr/`, `bestfitpy/` — the packages. Each contains a **generated, committed copy** of the
  core at `src/bestfit_core/` and (R) `inst/fixtures` / (Py) `src/bestfitpy/fixtures`.
- `tools/sync_core.py`, `tools/sync_fixtures.py` — copy canonical → vendored. Run after ANY
  change to `core/` or `fixtures/`. CI runs `--check` and fails on drift. **If you edit a core
  header, re-run `python3 tools/sync_core.py` or the packages build stale code.**
- `tools/oracle_emitter/` (C#) + `tools/verify_oracles.py` — the dotnet oracle gate: replays every
  fixture against the real Numerics library and fails on any value that doesn't reproduce to
  tolerance. **Dev-only** (needs `dotnet` + the submodule); not wired into CI.
- An auto-scraper to harvest the C# test literals *en masse* is still planned for the bulk port;
  for now fixtures are curated by hand and confirmed by the dotnet gate.

The univariate distribution layer lives under `core/include/bestfit/numerics/distributions/`:
`base/` holds `UnivariateDistributionBase`, the type enum, the factory, and the `IEstimation` /
`ILinearMomentEstimation` capability mixins; each distribution (`normal.hpp`, `uniform.hpp`,
`exponential.hpp`, `generalized_extreme_value.hpp`) derives from the base.

## Build & test commands

```bash
# C++ core
cmake -S core -B core/build && cmake --build core/build && ctest --test-dir core/build
# R  (regenerate registration only after editing bestfitr/src/*.cpp)
Rscript -e 'cpp11::cpp_register("bestfitr")'; R CMD INSTALL bestfitr
Rscript -e 'testthat::test_local("bestfitr")'
# Python (dev venv at ~/venv/bestfitpy)
~/venv/bestfitpy/bin/python -m pip install --force-reinstall --no-deps ./bestfitpy
~/venv/bestfitpy/bin/python -m pytest bestfitpy/tests -q
# sync guards (must pass before commit)
python3 tools/sync_core.py --check && python3 tools/sync_fixtures.py --check
# oracle reproduction gate (dev-only; needs dotnet + the upstream submodule)
python3 tools/verify_oracles.py
```

Toolchain present: clang++, cmake, R 4.6.1 (`/opt/homebrew/bin`), python3.14 + `~/venv/bestfitpy`,
roxygen2/jsonlite/testthat/cpp11 installed, and **dotnet 10** (for the oracle gate; not in CI).
After any core change that alters a class layout, rebuild R clean (`R CMD INSTALL --preclean
bestfitr`) — stale `bestfitr/src/*.o` from a prior ABI can otherwise return garbage / abort R.

## Validation model (DRY)

Oracle values live ONLY in `fixtures/*.json`. Three thin generic runners load the same JSON and
apply every assertion: C++ `core/tests/test_fixtures.cpp` (nlohmann/json, vendored test-only under
`core/tests/third_party/`), R `bestfitr/tests/testthat/test-fixtures.R` (jsonlite), Python
`bestfitpy/tests/test_fixtures.py` (stdlib). The runners are **polymorphic**: non-GEV targets are
built through the factory and dispatched on `UnivariateDistributionBase` (+ capability casts) via
the `bf_dist_*` (R) / `_core.dist_*` (Py) glue; GEV keeps a bespoke path for its standard-error
methods. **Adding a distribution = new fixture file + a couple of dispatch entries per runner** —
no new per-distribution glue. Don't hardcode oracle values in test files. The dotnet gate
(`verify_oracles.py`) is the fourth, dev-only check that the fixtures still match the C# source.

## Conventions & gotchas

- **Structural mirroring:** C++ mirrors the C# file/class/method layout so upstream diffs map
  almost line-for-line. Each ported file carries a `// ported from: <path> @ <sha>` header.
- **Portability (learned from CI):** never use `M_PI` (absent under strict `-std=c++17` on Linux
  and on MSVC) — use `bestfit::numerics::kPi`. Don't name a namespace alias `gamma` (clashes with
  glibc's libm `gamma()`). Pass `-Wall/-Wextra` only to non-MSVC compilers in CMake.
- **Self-contained core:** no external C++ deps (port Numerics' own linear algebra / RNG). Keeps
  the CRAN dependency surface empty and preserves oracle fidelity. Don't add Eigen to the core.
- **CRAN:** `bestfitr` uses `License: file LICENSE` (R can't standardize the `0BSD` token).
  Makevars: `CXX_STD = CXX17`, no `-O3/-march/-Werror`. cpp11 internal functions (`bf_gev_*`,
  `bf_dist_*`) are unexported — tests reach them via `asNamespace("bestfitr")`. After editing any
  `bestfitr/src/*.cpp`, re-run `cpp11::cpp_register("bestfitr")`.
- **Mutation:** the global "never mutate" rule is relaxed for these binding/model objects (they
  mirror the C# stateful API), matching the upstream design.

## Git & CI

- Commits are GPG-signed automatically (key `F4C82FB462F850C1`, "Cam Bracken (GitHub)"),
  verified on GitHub. Identity: `Cam Bracken <cameron.bracken@pm.me>`. Push only when asked.
- `.github/workflows/ci.yml`: `sync-check` gate → `core` (3 OS) + `r-cmd-check` (3 OS) +
  `python` (3 OS × {3.10, 3.12}). Use `gh run watch <id> --exit-status` to follow a run.
- Never commit: OS junk, IDE settings, secrets, the dotnet build output, other AI-tool files
  (`.gitignore` covers these). **Exception (deliberate):** this repo *does* track three curated
  context files — `.claude/CLAUDE.md`, `.claude/PLAN.md`, and `upstream/CLAUDE.md` — because the
  C++/R/Python-from-C# port is a complex process whose plan and porting guidance should travel with
  the repo. The `.gitignore` un-ignores exactly those three; everything else in `.claude/` (settings,
  scratch plans) stays ignored.

## Status

Phase 0 complete. **Phase 1 pilot done and merged** (PR #1): the polymorphic distribution layer
(`UnivariateDistributionBase`, factory, `IEstimation`/`ILinearMomentEstimation` mixins) plus
**Normal / Uniform / Exponential**, with GEV refactored onto the base. All harnesses + the dotnet
oracle gate green; R and Python bit-identical; CI green on 3 platforms. Next: the rest of the
math/RNG foundation and the remaining ~38 univariate distributions, fixture-driven. See `PLAN.md`.
