# Generic, fixture-driven validation for bestfitr.
#
# Reads the language-neutral oracle fixtures (the single source of truth shared with
# the C++ core and the Python package) and checks every assertion. No oracle values
# live here -- only the dispatch from fixture method names to the package's API. The GEV
# slice uses its bespoke bf_gev_* glue; every other distribution goes through the
# polymorphic bf_dist_* glue (factory + UnivariateDistributionBase).

parse_num <- function(v) {
  if (is.character(v)) {
    return(switch(v, "nan" = NaN, "inf" = Inf, "-inf" = -Inf,
                  stop(sprintf("unexpected string value: %s", v))))
  }
  as.numeric(v)
}

build_params <- function(target, construct, datasets) {
  ns <- asNamespace("bestfitr")
  if (!is.null(construct$params)) {
    return(vapply(construct$params, parse_num, numeric(1)))
  }
  fit <- construct$fit
  data <- as.numeric(unlist(datasets[[fit$dataset]]))
  if (target == "GeneralizedExtremeValue") {
    f <- gev_fit(data, fit$method)
    return(c(f[["location"]], f[["scale"]], f[["shape"]]))
  }
  ns$bf_dist_fit_(target, data, fit$method)
}

dispatch_gev <- function(p, method, args) {
  loc <- p[1]; scale <- p[2]; shape <- p[3]
  ns <- asNamespace("bestfitr")  # internal cpp11 functions are not exported
  moment_names <- c("mean", "median", "mode", "sd", "skewness", "kurtosis",
                    "minimum", "maximum")
  if (method %in% moment_names) {
    return(unname(ns$bf_gev_moments_(loc, scale, shape)[[method]]))
  }
  switch(method,
    pdf = ns$bf_gev_pdf_(as.double(args[[1]]), loc, scale, shape),
    cdf = ns$bf_gev_cdf_(as.double(args[[1]]), loc, scale, shape),
    quantile = ns$bf_gev_quantile_(as.double(args[[1]]), loc, scale, shape),
    parameters_valid = ns$bf_gev_valid_(loc, scale, shape),
    param = c(location = loc, scale = scale, shape = shape)[[args[[1]]]],
    linear_moment = ns$bf_gev_linear_moments_(loc, scale, shape)[[as.integer(args[[1]]) + 1L]],
    quantile_gradient = ns$bf_gev_quantile_gradient_(as.double(args[[1]]), loc, scale, shape)[[
      as.integer(args[[2]]) + 1L]],
    parameter_covariance = ns$bf_gev_parameter_covariance_(loc, scale, shape,
      as.integer(args[[1]]))[[as.integer(args[[2]]) * 3L + as.integer(args[[3]]) + 1L]],
    quantile_variance = ns$bf_gev_quantile_variance_(as.double(args[[1]]), loc, scale, shape,
      as.integer(args[[2]])),
    quantile_se = sqrt(ns$bf_gev_quantile_variance_(as.double(args[[1]]), loc, scale, shape,
      as.integer(args[[2]]))),
    stop(sprintf("unknown fixture method: %s", method))
  )
}

dispatch_generic <- function(target, p, method, args) {
  ns <- asNamespace("bestfitr")
  moment_names <- c("mean", "median", "mode", "sd", "skewness", "kurtosis",
                    "minimum", "maximum")
  if (method %in% moment_names) {
    return(unname(ns$bf_dist_moments_(target, p)[[method]]))
  }
  switch(method,
    pdf = ns$bf_dist_pdf_(target, p, as.double(args[[1]])),
    cdf = ns$bf_dist_cdf_(target, p, as.double(args[[1]])),
    quantile = ns$bf_dist_quantile_(target, p, as.double(args[[1]])),
    parameters_valid = ns$bf_dist_valid_(target, p),
    param = p[[as.integer(args[[1]]) + 1L]],
    linear_moment = ns$bf_dist_linear_moments_(target, p)[[as.integer(args[[1]]) + 1L]],
    stop(sprintf("unknown fixture method: %s", method))
  )
}

check_assertion <- function(actual, a) {
  mode <- a$mode
  if (mode == "bool") {
    expect_identical(as.logical(actual), a$expected)
  } else if (mode == "equal") {
    e <- parse_num(a$expected)
    if (is.nan(e)) expect_true(is.nan(actual)) else expect_equal(actual, e)
  } else if (mode == "abs") {
    expect_lte(abs(actual - a$expected), a$tol)
  } else if (mode == "rel") {
    expect_lte(abs(actual - a$expected) / abs(a$expected), a$tol)
  } else {
    stop(sprintf("unknown comparison mode: %s", mode))
  }
}

test_that("oracle fixtures validate", {
  skip_if_not_installed("jsonlite")
  fdir <- system.file("fixtures", package = "bestfitr")
  files <- list.files(fdir, pattern = "\\.json$", recursive = TRUE, full.names = TRUE)
  expect_gt(length(files), 0)
  for (f in files) {
    spec <- jsonlite::read_json(f, simplifyVector = FALSE)
    target <- spec$target
    datasets <- spec$datasets
    for (case in spec$cases) {
      p <- build_params(target, case$construct, datasets)
      for (a in case$assertions) {
        args <- if (is.null(a$args)) list() else a$args
        actual <- if (target == "GeneralizedExtremeValue") {
          dispatch_gev(p, a$method, args)
        } else {
          dispatch_generic(target, p, a$method, args)
        }
        check_assertion(actual, a)
      }
    }
  }
})
