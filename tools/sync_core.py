#!/usr/bin/env python3
"""Vendor the canonical C++ core into each language package.

CRAN/PyPI source packages must be self-contained, so the single source of truth in
``core/`` is COPIED (and committed) into each package's ``src/bestfit_core/``. A CI
check re-runs this and fails on any diff, keeping the vendored copies in lock-step.

For now the core is header-only, so only ``core/include`` (and ``core/data``) are
vendored. When ``.cpp`` translation units are added, this script will also copy
``core/src`` and regenerate the per-build source manifests.

Usage:  python3 tools/sync_core.py [--check]
  --check : exit non-zero if any vendored copy is stale (for CI), making no changes.
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "core"
# Each package gets a vendored copy of these core subtrees under src/bestfit_core/.
PACKAGES = [ROOT / "bestfitr" / "src" / "bestfit_core",
            ROOT / "bestfitpy" / "src" / "bestfit_core"]
SUBTREES = ["include", "data"]

# core/data/ also needs to land in the runtime data directories of each package so
# callers can locate the direction-number file at run time:
#   R:      system.file("extdata", "new-joe-kuo-6.21201", package = "bestfitr")
#   Python: importlib.resources.files("bestfitpy") / "data" / "new-joe-kuo-6.21201"
DATA_RUNTIME_DESTS = [
    ROOT / "bestfitr" / "inst" / "extdata",
    ROOT / "bestfitpy" / "src" / "bestfitpy" / "data",
]


def dirs_equal(a: Path, b: Path) -> bool:
    if not b.exists():
        return False
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.diff_files or cmp.funny_files:
        return False
    return all(dirs_equal(a / sub, b / sub) for sub in cmp.common_dirs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report staleness; make no changes")
    args = ap.parse_args()

    stale = False
    for dest_root in PACKAGES:
        for sub in SUBTREES:
            src = CORE / sub
            # Skip a subtree that does not exist or contains no files (e.g. an empty
            # data/ before the Sobol direction-number file is added). Shipping an empty
            # directory triggers a spurious R CMD check NOTE.
            if not src.exists() or not any(p.is_file() for p in src.rglob("*")):
                continue
            dest = dest_root / sub
            if args.check:
                if not dirs_equal(src, dest):
                    print(f"STALE: {dest} differs from {src}")
                    stale = True
            else:
                if dest.exists():
                    shutil.rmtree(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dest)
                print(f"synced {src} -> {dest}")

    # --- Runtime data copies (core/data/ -> R inst/extdata/ and Python package data) ---
    data_src = CORE / "data"
    if data_src.exists() and any(p.is_file() for p in data_src.rglob("*")):
        for dest in DATA_RUNTIME_DESTS:
            if args.check:
                if not dirs_equal(data_src, dest):
                    print(f"STALE: {dest} differs from {data_src}")
                    stale = True
            else:
                if dest.exists():
                    shutil.rmtree(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(data_src, dest)
                print(f"synced {data_src} -> {dest}")

    if args.check and stale:
        print("Vendored core is stale; run: python3 tools/sync_core.py")
        return 1
    if args.check:
        print("Vendored core is up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
