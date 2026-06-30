#!/usr/bin/env python3
"""Ship the canonical oracle fixtures into each language package.

``fixtures/`` is the single source of truth. The R package reads them via
``system.file("fixtures", ...)`` and the Python package via
``importlib.resources``, so each package ships its own verbatim copy. A CI check
re-runs this and fails on drift.

Usage:  python3 tools/sync_fixtures.py [--check]
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "fixtures"
DESTS = [ROOT / "bestfitr" / "inst" / "fixtures",
         ROOT / "bestfitpy" / "src" / "bestfitpy" / "fixtures"]


def dirs_equal(a: Path, b: Path) -> bool:
    if not b.exists():
        return False
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.diff_files or cmp.funny_files:
        return False
    return all(dirs_equal(a / sub, b / sub) for sub in cmp.common_dirs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    stale = False
    for dest in DESTS:
        if args.check:
            if not dirs_equal(SRC, dest):
                print(f"STALE: {dest} differs from {SRC}")
                stale = True
        else:
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(SRC, dest)
            print(f"synced {SRC} -> {dest}")

    if args.check and stale:
        print("Fixtures are stale; run: python3 tools/sync_fixtures.py")
        return 1
    if args.check:
        print("Fixtures are up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
