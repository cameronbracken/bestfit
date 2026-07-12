"""Identity tests for the Sobol quasi-random sequence.

Pins the Python binding to the canonical first-10 points for dimension 2.
Oracle values come from upstream/Numerics/Test_Numerics/Sampling/Test_SobolSequence.cs
(Test_Sobol), verified against R's randtoolbox::sobol().
"""
from __future__ import annotations

from importlib.resources import files

import pytest

from corehydropy import _core


def _data_path() -> str:
    """Return the path to the installed direction-numbers file."""
    return str(files("corehydropy") / "data" / "new-joe-kuo-6.21201")


# Oracle: first 10 points for dimension 2 (exact binary fractions).
_EXPECTED_DIM2 = [
    [0.5000, 0.5000],
    [0.7500, 0.2500],
    [0.2500, 0.7500],
    [0.3750, 0.3750],
    [0.8750, 0.8750],
    [0.6250, 0.1250],
    [0.1250, 0.6250],
    [0.1875, 0.3125],
    [0.6875, 0.8125],
    [0.9375, 0.0625],
]


def test_sobol_dim2_first10():
    """First 10 points for dimension 2 match the C# oracle (exact equality)."""
    path = _data_path()
    pts = _core.sobol_generate(2, 10, path)
    for i, (actual, expected) in enumerate(zip(pts, _EXPECTED_DIM2)):
        for j in range(2):
            assert actual[j] == expected[j], (
                f"point[{i}][{j}]: got {actual[j]}, expected {expected[j]}"
            )


def test_sobol_dim1_first_point():
    """Dimension 1 first point is 0.5 (unit initialization, no file needed)."""
    pts = _core.sobol_generate(1, 1, "")
    assert pts[0][0] == 0.5


def test_sobol_skip_to_first():
    """skip_to(1) returns the same value as the first sequential point."""
    path = _data_path()
    first_seq = _core.sobol_generate(2, 1, path)[0]
    first_skip = _core.sobol_skip_to(2, 1, path)
    for j in range(2):
        assert first_skip[j] == first_seq[j]


def test_sobol_skip_to_fifth():
    """skip_to(5) matches the 5th sequential point (0-indexed: pts[4])."""
    path = _data_path()
    fifth_seq = _core.sobol_generate(2, 5, path)[4]
    fifth_skip = _core.sobol_skip_to(2, 5, path)
    for j in range(2):
        assert fifth_skip[j] == fifth_seq[j]


def test_sobol_identity_r_python():
    """Python sequence matches the R-validated oracle values bit-for-bit."""
    path = _data_path()
    pts = _core.sobol_generate(2, 10, path)
    for i, expected_row in enumerate(_EXPECTED_DIM2):
        for j, expected_val in enumerate(expected_row):
            assert pts[i][j] == expected_val, (
                f"identity mismatch at point[{i}][{j}]"
            )
