# Smoke tests for the X11 exported analysis functions (composite / spatial_gev / bivariate /
# coincident / rating_curve / bootstrap / prior + posterior predictive). These prove each wrapper
# is callable end-to-end and returns the documented list shape; exact oracle values live in
# fixtures/ and are checked by test-fixtures.R.

x11_flow <- function() {
  c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600,
    19200, 13800, 25600, 10500, 16900, 21300, 14700, 8200, 23800, 15900)
}

test_that("composite_analysis returns a composite frequency curve", {
  res <- composite_analysis(
    x11_flow(), families = c("Normal", "Normal"), composite_type = "CompetingRisks",
    sampler = "DEMCzs", iterations = 100, output_length = 200, seed = 12345,
    thinning_interval = 1, exceedance_probabilities = c(0.01, 0.1, 0.5, 0.9, 0.99)
  )
  expect_equal(length(res$mode_curve), 5)
  expect_equal(length(res$lower_ci), 5)
})

test_that("spatial_gev_analysis returns per-site + regional results", {
  coords <- matrix(c(0, 0, 1, 0.5, 0.5, 1), ncol = 2, byrow = TRUE)
  at_site <- matrix(c(
    20, 22, 18, 25, 26, 24, 18, 19, 17, 30, 31, 28, 22, 23, 21,
    28, 29, 27, 35, 34, 33, 24, 25, 23, 26, 27, 25, 40, 38, 37,
    21, 22, 20, 27, 28, 26, 33, 32, 31, 23, 24, 22, 29, 30, 28
  ), ncol = 3, byrow = TRUE)
  res <- spatial_gev_analysis(
    coords, at_site, sampler = "DEMCzs", iterations = 300, output_length = 400,
    number_of_chains = 4, seed = 12345, thinning_interval = 1
  )
  expect_equal(res$site_count, 3)
  expect_length(res$site_location_mean, 3)
  expect_gt(length(res$mode_curve), 0)
})

x11_bivar_x <- function() c(-1.2, 0.3, 0.8, -0.5, 1.1, -0.9, 0.2, 1.5, -1.8, 0.6, -0.1, 0.9, -0.7, 1.3, -0.3)
x11_bivar_y <- function() c(-0.8, 0.5, 1.0, -0.3, 0.9, -1.1, 0.4, 1.2, -1.5, 0.7, 0.1, 1.1, -0.6, 1.4, -0.2)

test_that("bivariate_analysis returns a joint-exceedance curve", {
  res <- bivariate_analysis(
    "Normal", x11_bivar_x(), c(0, 1), "Normal", x11_bivar_y(), c(0, 1),
    xy_x = c(-1.5, -0.5, 0.5, 1.5, 2.0), xy_y = c(-1.5, -0.5, 0.5, 1.5, 2.0),
    copula = "Normal", sampler = "DEMCzs", iterations = 300, output_length = 400,
    number_of_chains = 4, seed = 12345, thinning_interval = 1
  )
  expect_equal(length(res$mode_curve), 5)
})

test_that("coincident_frequency_analysis returns an AEP curve", {
  x <- c(-2, -1, 0, 1, 2)
  response <- outer(x, x, "+")
  res <- coincident_frequency_analysis(
    "Normal", x11_bivar_x(), c(0, 1), "Normal", x11_bivar_y(), c(0, 1),
    x_values = x, y_values = x, response = response, number_of_bins = 5,
    copula = "Normal", sampler = "DEMCzs", iterations = 300, output_length = 400,
    number_of_chains = 4, seed = 12345, thinning_interval = 1
  )
  expect_equal(length(res$z_output_values), 5)
  expect_equal(length(res$mode_curve), 5)
})

test_that("rating_curve_analysis returns a predicted-discharge curve", {
  res <- rating_curve_analysis(
    stage = c(1.0, 1.2, 1.5, 1.8, 2.0, 2.3, 2.6, 2.9, 3.1, 3.4, 3.7, 4.0, 4.3, 4.6, 5.0),
    discharge = c(5.0, 7.1, 11.0, 16.2, 20.5, 28.0, 37.1, 47.9, 55.6, 68.0, 82.3, 98.1, 115.4, 134.8, 160.2),
    stage_bins = 20, sampler = "DEMCzs", iterations = 300, output_length = 400,
    number_of_chains = 4, seed = 12345, thinning_interval = 1
  )
  expect_equal(length(res$mode_curve), 20)
})

test_that("bootstrap_analysis returns percentile confidence bands", {
  res <- bootstrap_analysis(
    x11_flow()[1:10], distribution = "Normal", probabilities = c(0.5, 0.9, 0.99),
    estimation_method = "MaximumLikelihood", sample_size = 30, replications = 200, seed = 12345
  )
  expect_equal(length(res$mode_curve), 3)
  expect_equal(length(res$lower_ci), 3)
  expect_true(is.finite(res$mode_curve[[1]]))
})

test_that("prior_predictive_check returns a summary", {
  res <- prior_predictive_check(
    x11_flow()[1:10], distribution = "Normal", number_of_draws = 200, sample_size = 30, seed = 12345
  )
  expect_equal(res$number_of_valid_draws, 200)
  expect_length(res$summary_mean_quantiles, 5)
})

test_that("posterior_predictive_check returns common p-values", {
  res <- posterior_predictive_check(
    x11_flow()[1:10], distribution = "Normal", sampler = "DEMCzs", iterations = 100,
    output_length = 200, seed = 12345, thinning_interval = 1, number_of_replicates = 200
  )
  expect_equal(res$number_of_replicates, 200)
  expect_true(res$mean_p_value >= 0 && res$mean_p_value <= 1)
})
