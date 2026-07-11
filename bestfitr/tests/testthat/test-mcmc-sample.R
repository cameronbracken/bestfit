# mcmc_sample() wrapper tests: structural checks plus a fixture-anchored run. The seeded
# RWMH oracle values come from the mcmc_sampler fixture (rwmh.json) routed through the
# PUBLIC wrapper where the settings align; no oracle values are hardcoded.

test_that("mcmc_sample returns a complete, consistent result", {
  x <- dist_random(distribution("Normal", c(100, 15)), 100, seed = 42)
  fit <- mcmc_sample(x, "Normal", sampler = "DEMCzs", iterations = 1000, seed = 12345)
  expect_named(fit, c("parameters", "chains", "acceptance_rates", "map", "map_fitness",
    "posterior_mean", "posterior_sd", "posterior_median", "posterior_lower_ci",
    "posterior_upper_ci", "rhat", "ess"), ignore.order = TRUE)
  expect_length(fit$parameters, 2L)
  expect_length(fit$map, 2L)
  expect_gt(length(fit$chains), 0L)
  expect_identical(ncol(fit$chains[[1]]), 2L)
  # Posterior localizes near the truth under a healthy run.
  expect_lt(abs(fit$posterior_mean[[1]] - 100), 10)
  expect_lt(abs(fit$posterior_mean[[2]] - 15), 10)
  expect_true(all(fit$posterior_lower_ci < fit$posterior_upper_ci))
  expect_true(all(fit$rhat > 0.9 & fit$rhat < 1.2))
})

test_that("mcmc_sample is deterministic under a seed", {
  x <- dist_random(distribution("Gumbel", c(100, 10)), 80, seed = 9)
  a <- mcmc_sample(x, "Gumbel", sampler = "RWMH", iterations = 500, seed = 12345)
  b <- mcmc_sample(x, "Gumbel", sampler = "RWMH", iterations = 500, seed = 12345)
  expect_identical(a$posterior_mean, b$posterior_mean)
  expect_identical(a$chains, b$chains)
  c <- mcmc_sample(x, "Gumbel", sampler = "RWMH", iterations = 500, seed = 999)
  expect_false(identical(a$posterior_mean, c$posterior_mean))
})

test_that("mcmc_sample validates its arguments", {
  x <- dist_random(distribution("Normal", c(0, 1)), 30, seed = 1)
  expect_error(mcmc_sample(x, "Normal", sampler = "Gibbs"))
  expect_error(mcmc_sample(x, "Normal", initialize = "Nope"))
})
