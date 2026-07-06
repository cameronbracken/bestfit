# Smoke tests for the exported user-facing analysis functions (A10). These prove the
# univariate_analysis / fit_distributions / bulletin17c_analysis wrappers are callable
# end-to-end and return the documented list shapes with finite / monotone invariants. Exact
# oracle values live in fixtures/ and are checked by test-fixtures.R; these assert structure.

# A small annual-peak record reused across the univariate + B17C smoke calls.
smoke_peaks <- function() {
  c(
    12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600,
    19200, 13800, 25600, 10500, 16900, 21300, 14700, 8200, 23800, 15900
  )
}

test_that("univariate_analysis returns a Bayesian frequency curve", {
  res <- univariate_analysis(
    smoke_peaks(),
    distribution = "Normal", sampler = "DEMCzs",
    iterations = 100, output_length = 400, credible_level = 0.90, seed = 12345
  )
  expect_length(res$parameters, 2)
  expect_true(all(is.finite(res$parameters)))
  expect_true(is.finite(res$aic))
  expect_true(is.finite(res$dic))
  n <- length(res$mode_curve)
  expect_gt(n, 0)
  expect_equal(length(res$mean_curve), n)
  expect_equal(length(res$lower_ci), n)
  expect_equal(length(res$upper_ci), n)
  # The default exceedance ordinates ascend, so the mode (quantile) curve descends.
  expect_true(all(diff(res$mode_curve) <= 1e-6))
  # Each credible band brackets the mode curve.
  expect_true(all(res$lower_ci <= res$mode_curve + 1e-6))
  expect_true(all(res$upper_ci >= res$mode_curve - 1e-6))
})

test_that("fit_distributions ranks the 14 ported candidates", {
  res <- fit_distributions(smoke_peaks())
  expect_length(res$distribution, 14)
  expect_length(res$aic, 14)
  expect_true(any(res$converged))
  ok <- res$converged
  expect_true(all(is.finite(res$aic[ok])))
  expect_true(all(is.finite(res$bic[ok])))
  expect_true(all(is.finite(res$rmse[ok])))
})

test_that("bulletin17c_analysis returns Cohn-style confidence intervals", {
  res <- bulletin17c_analysis(
    smoke_peaks(),
    uncertainty_method = "MultivariateNormal",
    output_length = 200, seed = 12345, confidence_level = 0.90
  )
  expect_equal(res$confidence_level, 0.90)
  n <- length(res$exceedance_probabilities)
  expect_gt(n, 0)
  expect_equal(length(res$point_estimates), n)
  expect_equal(length(res$lower_ci), n)
  expect_equal(length(res$upper_ci), n)
  expect_true(all(is.finite(res$point_estimates)))
  # Cohn CI brackets are ordered at every level (preserved through monotonicity enforcement).
  expect_true(all(res$lower_ci <= res$upper_ci))
  expect_length(res$parameters, 3) # LP3: location / scale / shape
})

test_that("bulletin17c_analysis rejects deferred uncertainty methods", {
  expect_error(
    bulletin17c_analysis(smoke_peaks(), uncertainty_method = "LinkedMultivariateNormal")
  )
})
