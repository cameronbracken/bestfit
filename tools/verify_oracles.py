#!/usr/bin/env python3
"""Verify the oracle fixtures reproduce against the upstream Numerics C# library.

This is the "run dotnet" half of the hybrid oracle workflow. The fixture expected values
are scraped/curated from the C# unit tests; this gate rebuilds each case against the real
``Numerics`` library (via ``tools/oracle_emitter``) and checks every value reproduces to its
stated tolerance. It is a DEV-ONLY check: it needs ``dotnet`` and the ``upstream/Numerics``
submodule, neither of which is required to build or ship the packages.

Usage:  python3 tools/verify_oracles.py [--fixtures DIR]
  exits 0 if every supported assertion reproduces, 1 on any mismatch, 2 if the toolchain
  is unavailable (no dotnet / submodule not checked out).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMITTER = ROOT / "tools" / "oracle_emitter"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", default=str(ROOT / "fixtures"), help="fixtures directory")
    args = ap.parse_args()

    if shutil.which("dotnet") is None:
        print("dotnet not found on PATH; skipping oracle reproduction check.", file=sys.stderr)
        return 2
    if not (ROOT / "upstream" / "Numerics" / "Numerics" / "Numerics.csproj").exists():
        print("upstream/Numerics submodule not checked out; run: git submodule update --init",
              file=sys.stderr)
        return 2

    cmd = ["dotnet", "run", "--project", str(EMITTER), "-c", "Release", "--",
           str(Path(args.fixtures).resolve())]
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
