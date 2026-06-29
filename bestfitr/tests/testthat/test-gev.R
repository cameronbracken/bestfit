# GEV bindings validated against the same upstream C# oracle values used by the
# C++ core tests (Rao & Hamed, "Flood Frequency Analysis", CRC Press 2000).

test_that("density, distribution, and quantile match oracles", {
  # default GEV(100, 10, 0)
  expect_equal(dgev(0, 100, 10, 0), 0)
  expect_equal(dgev(0, 100, 10, 1), 1.67017007902456e-06, tolerance = 1e-10)
  expect_equal(pgev(100, 100, 10, 0), 0.367879, tolerance = 1e-4)
  expect_equal(pgev(200, 100, 10, 0), 0.9999546, tolerance = 1e-7)
  expect_equal(qgev(0.5, 100, 10, 0), 103.66512, tolerance = 1e-5)
  expect_equal(qgev(0, 100, 10, 0), -Inf)
  expect_equal(qgev(1, 100, 10, 0), Inf)
})

test_that("density, distribution, quantile are vectorised", {
  x <- c(90, 100, 110)
  expect_equal(dgev(x, 100, 10, 0), vapply(x, function(v) dgev(v, 100, 10, 0), numeric(1)))
  expect_length(pgev(x, 100, 10, 0), 3L)
})

test_that("moments match oracles", {
  m <- gev_moments(100, 10, 0)
  expect_equal(unname(m["mean"]), 100 + 10 * 0.5772156649015328606)
  expect_equal(unname(m["median"]), 103.66512, tolerance = 1e-4)
  expect_equal(unname(m["sd"]), 12.825498, tolerance = 1e-5)
  expect_equal(unname(m["skewness"]), 1.1396)
  expect_equal(unname(m["minimum"]), -Inf)
  # undefined moments return NaN
  expect_true(is.nan(gev_moments(100, 10, 10)["mean"]))
})

# White River near Nora, IN (Rao & Hamed Table 7.1.2)
white_river <- c(
  23200, 2950, 10300, 23200, 4540, 9960, 10800, 26900, 23300, 20400, 8480, 3150,
  9380, 32400, 20800, 11100, 7270, 9600, 14600, 14300, 22500, 14700, 12700, 9740,
  3050, 8830, 12000, 30400, 27000, 15200, 8040, 11700, 20300, 22700, 30400, 9180,
  4870, 14700, 12800, 13700, 7960, 9830, 12500, 10700, 13200, 14700, 14300, 4050,
  14600, 14400, 19200, 7160, 12100, 8650, 10600, 24500, 14400, 6300, 9560, 15800,
  14300, 28700)

test_that("L-moment fit matches oracle", {
  s <- c(1953, 1939, 1677, 1692, 2051, 2371, 2022, 1521, 1448, 1825, 1363,
         1760, 1672, 1603, 1244, 1521, 1783, 1560, 1357, 1673, 1625, 1425,
         1688, 1577, 1736, 1640, 1584, 1293, 1277, 1742, 1491)
  fit <- gev_fit(s, method = "lmom")
  expect_equal(unname(fit["location"]), 1543.933, tolerance = 0.001)
  expect_equal(unname(fit["scale"]), 218.1148, tolerance = 0.001)
  expect_equal(unname(fit["shape"]), 0.1068473, tolerance = 0.001)
})

test_that("MOM and MLE fits match oracles (1% relative)", {
  mom <- gev_fit(white_river, method = "mom")
  expect_equal(unname(mom["location"]), 11012, tolerance = 0.01)
  expect_equal(unname(mom["scale"]), 6209.4, tolerance = 0.01)
  expect_equal(unname(mom["shape"]), 0.0736, tolerance = 0.01)

  mle <- gev_fit(white_river, method = "mle")
  expect_equal(unname(mle["location"]), 10849, tolerance = 0.01)
  expect_equal(unname(mle["scale"]), 5745.6, tolerance = 0.01)
})

test_that("quantile standard error matches oracle (Example 7.1.3)", {
  se <- gev_quantile_se(0.99, 10849, 5745.6, 0.005, sample_size = 62)
  expect_equal(se, 5142, tolerance = 0.01)
})
