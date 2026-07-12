# Public distribution API wrapper tests. Oracle values are NOT hardcoded here: numeric
# expectations are read from the fixture JSONs (the single source of truth) and routed
# through the PUBLIC wrappers, so these tests prove the wrappers hit the same glue the
# fixture runner validates.

fixture_path <- function(...) {
  file.path(system.file("fixtures", package = "bestfitr"), ...)
}

test_that("distribution() validates its inputs", {
  expect_error(distribution("NotAFamily", c(1, 2)), "unknown distribution family")
  expect_error(distribution("Normal", 1), "expects 2 parameters")
  expect_error(distribution(c("Normal", "Gumbel"), c(1, 2)), "single distribution name")
  d <- distribution("Normal", c(100, 15))
  expect_s3_class(d, "bestfit_dist")
  expect_identical(d$family, "Normal")
})

test_that("distribution_names() lists the factory families", {
  nms <- distribution_names()
  expect_true(all(c("Normal", "LogNormal", "Gumbel", "GeneralizedExtremeValue", "Weibull")
    %in% nms))
  expect_identical(anyDuplicated(nms), 0L)
})

test_that("public verbs reproduce the fixture oracle values", {
  skip_if_not_installed("jsonlite")
  spec <- jsonlite::read_json(fixture_path("distributions", "univariate", "gumbel.json"),
    simplifyVector = FALSE)
  for (case in spec$cases) {
    if (is.null(case$construct$params)) next
    d <- distribution(spec$target, as.double(unlist(case$construct$params)))
    for (a in case$assertions) {
      x <- if (is.null(a$args)) NULL else as.double(unlist(a$args))
      actual <- switch(a$method,
        pdf = dist_pdf(d, x[[1]]),
        cdf = dist_cdf(d, x[[1]]),
        quantile = dist_quantile(d, x[[1]]),
        mean = unname(dist_moments(d)[["mean"]]),
        sd = unname(dist_moments(d)[["sd"]]),
        random_value = dist_random(d, as.integer(x[[1]]), seed = as.integer(x[[2]]))[[
          as.integer(x[[3]]) + 1L]],
        NULL
      )
      if (is.null(actual)) next
      expected <- suppressWarnings(as.numeric(a$expected))
      if (is.na(expected)) next
      if (identical(a$mode, "abs")) {
        expect_lte(abs(actual - expected), a$tol)
      } else if (identical(a$mode, "rel")) {
        expect_lte(abs(actual - expected) / abs(expected), a$tol)
      }
    }
  }
})

test_that("verbs are vectorized with scalar passthrough", {
  d <- distribution("Normal", c(0, 1))
  expect_length(dist_pdf(d, c(-1, 0, 1)), 3L)
  expect_length(dist_pdf(d, 0), 1L)
  expect_identical(dist_cdf(d, 0), 0.5)
  expect_equal(dist_quantile(d, dist_cdf(d, 1.3)), 1.3)
  expect_equal(dist_log_pdf(d, 0.7), log(dist_pdf(d, 0.7)))
  expect_equal(
    dist_log_likelihood(d, c(0, 1, -1)),
    sum(dist_log_pdf(d, c(0, 1, -1)))
  )
})

test_that("dist_random is deterministic under a seed", {
  d <- distribution("Gumbel", c(100, 10))
  expect_identical(dist_random(d, 10, seed = 7), dist_random(d, 10, seed = 7))
  expect_false(identical(dist_random(d, 10, seed = 7), dist_random(d, 10, seed = 8)))
  expect_length(dist_random(d, 25, seed = 1), 25L)
})

test_that("dist_fit returns a fitted bestfit_dist matching the internal glue", {
  d <- distribution("Gumbel", c(100, 10))
  x <- dist_random(d, 200, seed = 42)
  f <- dist_fit("Gumbel", x, method = "mle")
  expect_s3_class(f, "bestfit_dist")
  ns <- asNamespace("bestfitr")
  expect_identical(f$params, as.double(ns$bf_dist_fit_("Gumbel", x, "mle")))
  expect_error(dist_fit("Deterministic", x, method = "mle"), "only MethodOfMoments")
})

test_that("dist_params names parameters and print method works", {
  d <- distribution("Normal", c(100, 15))
  p <- dist_params(d)
  expect_identical(unname(p), c(100, 15))
  expect_length(names(p), 2L)
  expect_output(print(d), "bestfit_dist")
  expect_output(print(distribution("Normal", c(0, -1))), "not valid")
})

test_that("dist_lmoments errors for families without L-moment support", {
  expect_error(dist_lmoments(distribution("Cauchy", c(0, 1))), "no L-moments")
})
