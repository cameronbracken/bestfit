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

# --- Composite distribution path -------------------------------------------------------
# For TruncatedDistribution (and future Empirical/KernelDensity/Mixture/CompetingRisks),
# the "construct" field uses a structured schema instead of flat "params".
# build_composite_data() parses the construct into a list consumed by dispatch_composite().
# Adding a new composite = one new case in build_composite_data + one in dispatch_composite.

kCompositeTargets <- c("TruncatedDistribution", "Empirical", "KernelDensity", "Mixture",
                       "CompetingRisks")

build_composite_data <- function(target, construct, datasets = list()) {
  if (target == "TruncatedDistribution") {
    base_target <- construct$base$target
    base_params <- vapply(construct$base$params, parse_num, numeric(1))
    lo <- as.double(construct$bounds[[1]])
    hi <- as.double(construct$bounds[[2]])
    return(list(base_target = base_target, base_params = base_params, lo = lo, hi = hi))
  }
  if (target == "Empirical") {
    xv <- as.double(unlist(construct$x))
    pv <- as.double(unlist(construct$p))
    pt <- if (!is.null(construct$p_transform)) construct$p_transform else "NormalZ"
    return(list(x_vals = xv, p_vals = pv, p_transform = pt))
  }
  if (target == "KernelDensity") {
    data_key  <- construct$data
    data_vec  <- as.double(unlist(datasets[[data_key]]))
    kernel    <- if (!is.null(construct$kernel)) construct$kernel else "Gaussian"
    bandwidth <- if (!is.null(construct$bandwidth)) as.double(construct$bandwidth) else -1.0
    bounded   <- if (!is.null(construct$bounded_by_data)) as.logical(construct$bounded_by_data) else TRUE
    return(list(data_vec = data_vec, kernel = kernel, bandwidth = bandwidth, bounded = bounded))
  }
  if (target == "Mixture") {
    comp_targets <- vapply(construct$components, function(c) c$target, character(1))
    comp_params  <- lapply(construct$components, function(c) vapply(c$params, parse_num, numeric(1)))
    wts          <- as.double(unlist(construct$weights))
    return(list(comp_targets = comp_targets, comp_params = comp_params, weights = wts))
  }
  if (target == "CompetingRisks") {
    comp_targets <- vapply(construct$components, function(c) c$target, character(1))
    comp_params  <- lapply(construct$components, function(c) vapply(c$params, parse_num, numeric(1)))
    min_of_rv    <- if (!is.null(construct$minimum_of_random_variables))
                      as.logical(construct$minimum_of_random_variables) else TRUE
    return(list(comp_targets = comp_targets, comp_params = comp_params,
                minimum_of_rv = min_of_rv))
  }
  stop(sprintf("unknown composite target: %s", target))
}

dispatch_composite <- function(target, cd, method, args) {
  ns <- asNamespace("bestfitr")
  moment_names <- c("mean", "median", "mode", "sd", "skewness", "kurtosis",
                    "minimum", "maximum")
  if (target == "TruncatedDistribution") {
    if (method %in% moment_names) {
      return(unname(ns$bf_trunc_moments_(cd$base_target, cd$base_params, cd$lo, cd$hi)[[method]]))
    }
    return(switch(method,
      pdf      = ns$bf_trunc_pdf_(cd$base_target, cd$base_params, cd$lo, cd$hi,
                                  as.double(args[[1]])),
      cdf      = ns$bf_trunc_cdf_(cd$base_target, cd$base_params, cd$lo, cd$hi,
                                  as.double(args[[1]])),
      quantile = ns$bf_trunc_quantile_(cd$base_target, cd$base_params, cd$lo, cd$hi,
                                       as.double(args[[1]])),
      parameters_valid = ns$bf_trunc_valid_(cd$base_target, cd$base_params, cd$lo, cd$hi),
      stop(sprintf("unknown fixture method for TruncatedDistribution: %s", method))
    ))
  }
  if (target == "Empirical") {
    xv <- cd$x_vals; pv <- cd$p_vals; pt <- cd$p_transform
    if (method %in% moment_names) {
      return(unname(ns$bf_emp_moments_(xv, pv, pt)[[method]]))
    }
    return(switch(method,
      pdf      = ns$bf_emp_pdf_(xv, pv, pt, as.double(args[[1]])),
      cdf      = ns$bf_emp_cdf_(xv, pv, pt, as.double(args[[1]])),
      quantile = ns$bf_emp_quantile_(xv, pv, pt, as.double(args[[1]])),
      parameters_valid = ns$bf_emp_valid_(xv, pv, pt),
      stop(sprintf("unknown fixture method for Empirical: %s", method))
    ))
  }
  if (target == "KernelDensity") {
    dv <- cd$data_vec; ker <- cd$kernel; bw <- cd$bandwidth; bd <- cd$bounded
    if (method %in% moment_names) {
      return(unname(ns$bf_kde_moments_(dv, ker, bw, bd)[[method]]))
    }
    return(switch(method,
      pdf              = ns$bf_kde_pdf_(dv, ker, bw, bd, as.double(args[[1]])),
      cdf              = ns$bf_kde_cdf_(dv, ker, bw, bd, as.double(args[[1]])),
      quantile         = ns$bf_kde_quantile_(dv, ker, bw, bd, as.double(args[[1]])),
      parameters_valid = ns$bf_kde_valid_(dv, ker, bw, bd),
      stop(sprintf("unknown fixture method for KernelDensity: %s", method))
    ))
  }
  if (target == "Mixture") {
    ct <- cd$comp_targets; cp <- cd$comp_params; wts <- cd$weights
    if (method %in% moment_names) {
      return(unname(ns$bf_mix_moments_(ct, cp, wts)[[method]]))
    }
    return(switch(method,
      pdf              = ns$bf_mix_pdf_(ct, cp, wts, as.double(args[[1]])),
      cdf              = ns$bf_mix_cdf_(ct, cp, wts, as.double(args[[1]])),
      quantile         = ns$bf_mix_quantile_(ct, cp, wts, as.double(args[[1]])),
      parameters_valid = ns$bf_mix_valid_(ct, cp, wts),
      stop(sprintf("unknown fixture method for Mixture: %s", method))
    ))
  }
  if (target == "CompetingRisks") {
    ct <- cd$comp_targets; cp <- cd$comp_params; min_rv <- cd$minimum_of_rv
    if (method %in% moment_names) {
      return(unname(ns$bf_cr_moments_(ct, cp, min_rv)[[method]]))
    }
    return(switch(method,
      pdf              = ns$bf_cr_pdf_(ct, cp, min_rv, as.double(args[[1]])),
      cdf              = ns$bf_cr_cdf_(ct, cp, min_rv, as.double(args[[1]])),
      quantile         = ns$bf_cr_quantile_(ct, cp, min_rv, as.double(args[[1]])),
      parameters_valid = ns$bf_cr_valid_(ct, cp, min_rv),
      stop(sprintf("unknown fixture method for CompetingRisks: %s", method))
    ))
  }
  stop(sprintf("unknown composite target: %s", target))
}

test_that("oracle fixtures validate", {
  skip_if_not_installed("jsonlite")
  fdir <- system.file("fixtures", package = "bestfitr")
  files <- list.files(fdir, pattern = "\\.json$", recursive = TRUE, full.names = TRUE)
  expect_gt(length(files), 0)
  for (f in files) {
    spec <- jsonlite::read_json(f, simplifyVector = FALSE)
    # Only validate univariate_distribution fixtures; skip other kinds (e.g. special_function)
    # which are validated in C++ only and are not exposed to the R package.
    if (!identical(spec$kind, "univariate_distribution")) next
    target <- spec$target
    datasets <- spec$datasets
    is_composite <- target %in% kCompositeTargets
    for (case in spec$cases) {
      if (is_composite) {
        cd <- build_composite_data(target, case$construct, datasets)
        for (a in case$assertions) {
          args <- if (is.null(a$args)) list() else a$args
          actual <- dispatch_composite(target, cd, a$method, args)
          check_assertion(actual, a)
        }
      } else {
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
  }
})
