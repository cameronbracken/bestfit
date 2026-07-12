# Stats utility wrapper tests. Numeric expectations come from the data_utility fixture
# (fixtures/data/statistics_utilities.json) routed through the PUBLIC wrappers; the rest
# are structural (round trips, shapes, argument validation) with no oracle content.

test_that("public stats wrappers reproduce the data_utility fixture oracles", {
  skip_if_not_installed("jsonlite")
  path <- file.path(system.file("fixtures", package = "corehydror"),
    "data", "statistics_utilities.json")
  spec <- jsonlite::read_json(path, simplifyVector = FALSE)
  datasets <- lapply(spec$datasets, \(d) as.double(unlist(d)))
  for (case in spec$cases) {
    args <- if (is.null(case$args)) numeric() else as.double(unlist(case$args))
    data <- if (is.null(case$dataset)) numeric() else datasets[[case$dataset]]
    actual <- switch(case[["function"]],
      MGBT = as.double(mgbt_test(data)),
      BoxCoxLambda = box_cox_lambda(data),
      BoxCoxTransform = box_cox(data, args[[1]])[[as.integer(args[[2]]) + 1L]],
      YeoJohnsonLambda = yeo_johnson_lambda(data),
      YeoJohnsonTransform = yeo_johnson(data, args[[1]])[[as.integer(args[[2]]) + 1L]],
      PlottingPosition = plotting_positions(as.integer(args[[1]]),
        alpha = args[[2]])[[as.integer(args[[3]]) + 1L]],
      LHSRandom = latin_hypercube(as.integer(args[[1]]), as.integer(args[[2]]),
        seed = as.integer(args[[3]]))[as.integer(args[[4]]) + 1L, as.integer(args[[5]]) + 1L],
      LHSMedian = latin_hypercube(as.integer(args[[1]]), as.integer(args[[2]]),
        seed = as.integer(args[[3]]), median = TRUE)[
        as.integer(args[[4]]) + 1L, as.integer(args[[5]]) + 1L]
    )
    a <- case$assertions[[1]]
    expected <- as.numeric(a$expected)
    if (identical(a$mode, "equal")) {
      expect_equal(actual, expected)
    } else if (identical(a$mode, "abs")) {
      expect_lte(abs(actual - expected), a$tol)
    } else {
      expect_lte(abs(actual - expected) / abs(expected), a$tol)
    }
  }
})

test_that("box_cox and yeo_johnson round-trip", {
  x <- dist_random(distribution("LogNormal", c(3, 0.5)), 50, seed = 3)
  for (lambda in c(-0.5, 0, 0.7)) {
    expect_equal(box_cox_inverse(box_cox(x, lambda), lambda), x)
  }
  y <- dist_random(distribution("Normal", c(0, 2)), 50, seed = 3)
  for (lambda in c(0.3, 1, 1.8)) {
    expect_equal(yeo_johnson_inverse(yeo_johnson(y, lambda), lambda), y)
  }
})

test_that("plotting_positions validates and orders", {
  pp <- plotting_positions(10)
  expect_length(pp, 10L)
  expect_true(all(diff(pp) > 0))
  expect_true(all(pp > 0 & pp < 1))
  expect_identical(plotting_positions(10, method = "hazen"),
    plotting_positions(10, alpha = 0.5))
  expect_error(plotting_positions(10, method = "nope"))
})

test_that("latin_hypercube is stratified, seeded, and shaped", {
  m <- latin_hypercube(20, 3, seed = 5)
  expect_identical(dim(m), c(20L, 3L))
  expect_true(all(m > 0 & m < 1))
  # LHS property: exactly one sample per 1/n bin in every dimension.
  for (j in 1:3) expect_identical(sort(findInterval(m[, j], seq(0, 1, by = 1 / 20))), 1:20)
  expect_identical(latin_hypercube(20, 3, seed = 5), m)
})

test_that("mgbt_test flags planted low outliers", {
  x <- dist_random(distribution("LogNormal", c(3, 0.5)), 40, seed = 7)
  expect_identical(mgbt_test(c(x, 1, 2)), 2L)
  expect_identical(mgbt_test(x), 0L)
})
