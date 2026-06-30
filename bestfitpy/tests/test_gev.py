# GEV bindings validated against the same upstream C# oracle values used by the
# C++ core and the R package (Rao & Hamed, "Flood Frequency Analysis", 2000).
import math

import numpy as np
import pytest

from bestfitpy import GeneralizedExtremeValue, dgev, gev_fit, gev_moments, pgev, qgev

WHITE_RIVER = [
    23200, 2950, 10300, 23200, 4540, 9960, 10800, 26900, 23300, 20400, 8480, 3150,
    9380, 32400, 20800, 11100, 7270, 9600, 14600, 14300, 22500, 14700, 12700, 9740,
    3050, 8830, 12000, 30400, 27000, 15200, 8040, 11700, 20300, 22700, 30400, 9180,
    4870, 14700, 12800, 13700, 7960, 9830, 12500, 10700, 13200, 14700, 14300, 4050,
    14600, 14400, 19200, 7160, 12100, 8650, 10600, 24500, 14400, 6300, 9560, 15800,
    14300, 28700,
]


def test_density_distribution_quantile():
    assert dgev(0, 100, 10, 0) == 0
    assert dgev(0, 100, 10, 1) == pytest.approx(1.67017007902456e-06, abs=1e-10)
    assert pgev(100, 100, 10, 0) == pytest.approx(0.367879, abs=1e-4)
    assert pgev(200, 100, 10, 0) == pytest.approx(0.9999546, abs=1e-7)
    assert qgev(0.5, 100, 10, 0) == pytest.approx(103.66512, abs=1e-5)
    assert qgev(0.0, 100, 10, 0) == -math.inf
    assert qgev(1.0, 100, 10, 0) == math.inf


def test_vectorised():
    x = np.array([90.0, 100.0, 110.0])
    out = dgev(x, 100, 10, 0)
    assert out.shape == (3,)
    assert out[1] == pytest.approx(dgev(100.0, 100, 10, 0))


def test_moments():
    m = gev_moments(100, 10, 0)
    assert m["mean"] == pytest.approx(100 + 10 * 0.5772156649015328606)
    assert m["sd"] == pytest.approx(12.825498, abs=1e-5)
    assert m["skewness"] == pytest.approx(1.1396)
    assert m["minimum"] == -math.inf
    assert math.isnan(gev_moments(100, 10, 10)["mean"])


def test_lmom_fit():
    s = [1953, 1939, 1677, 1692, 2051, 2371, 2022, 1521, 1448, 1825, 1363,
         1760, 1672, 1603, 1244, 1521, 1783, 1560, 1357, 1673, 1625, 1425,
         1688, 1577, 1736, 1640, 1584, 1293, 1277, 1742, 1491]
    fit = gev_fit(s, method="lmom")
    assert fit["location"] == pytest.approx(1543.933, abs=0.001)
    assert fit["scale"] == pytest.approx(218.1148, abs=0.001)
    assert fit["shape"] == pytest.approx(0.1068473, abs=0.001)


def test_mom_and_mle_fit():
    mom = gev_fit(WHITE_RIVER, method="mom")
    assert mom["location"] == pytest.approx(11012, rel=0.01)
    assert mom["scale"] == pytest.approx(6209.4, rel=0.01)
    assert mom["shape"] == pytest.approx(0.0736, rel=0.01)

    mle = gev_fit(WHITE_RIVER, method="mle")
    assert mle["location"] == pytest.approx(10849, rel=0.01)
    assert mle["scale"] == pytest.approx(5745.6, rel=0.01)


def test_quantile_standard_error():
    g = GeneralizedExtremeValue(10849, 5745.6, 0.005)
    se = math.sqrt(g.quantile_variance(0.99, len(WHITE_RIVER)))
    assert se == pytest.approx(5142, rel=0.01)
