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
