# corehydro core

The canonical, header-only C++17 implementation shared by the `corehydror` (R) and
`corehydropy` (Python) packages. No external dependencies. It is a faithful port of the
USACE-RMC `Numerics` and `RMC.BestFit` C# libraries, validated against them by the
dotnet oracle gate.

## Consuming the core

CMake, in-tree or via `FetchContent`:

    add_subdirectory(core)          # or FetchContent_Declare + MakeAvailable
    target_link_libraries(app PRIVATE corehydro::core)

`corehydro::core` is a header-only INTERFACE target that puts `core/include` on the
include path and requires C++17. The version is in `include/corehydro/version.hpp`
(`BESTFIT_CORE_VERSION`).

## How the language packages use it

The R and Python packages vendor the headers as subtree symlinks into `core/include`
and `core/data` (and the fixtures into `fixtures/`). A build dereferences them into a
self-contained, symlink-free artifact:

- R: `R CMD build` dereferences symlinks into the tarball automatically.
- Python: `tools/materialize_core.py` dereferences them into real files (run in a
  throwaway checkout; see `make build-py`).

See `docs/superpowers/specs/2026-07-08-shared-core-symlink-vendoring-design.md`.
