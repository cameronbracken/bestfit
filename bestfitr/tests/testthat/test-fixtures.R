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
    dependency   <- if (!is.null(construct$dependency)) construct$dependency else "Independent"
    correlation  <- if (!is.null(construct$correlation))
                      lapply(construct$correlation, function(r) as.double(unlist(r))) else list()
    return(list(comp_targets = comp_targets, comp_params = comp_params,
                minimum_of_rv = min_of_rv, dependency = dependency, correlation = correlation))
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
    dep <- cd$dependency; corr <- cd$correlation
    if (method %in% moment_names) {
      return(unname(ns$bf_cr_moments_(ct, cp, min_rv, dep, corr)[[method]]))
    }
    return(switch(method,
      pdf              = ns$bf_cr_pdf_(ct, cp, min_rv, dep, corr, as.double(args[[1]])),
      log_pdf          = ns$bf_cr_log_pdf_(ct, cp, min_rv, dep, corr, as.double(args[[1]])),
      cdf              = ns$bf_cr_cdf_(ct, cp, min_rv, dep, corr, as.double(args[[1]])),
      quantile         = ns$bf_cr_quantile_(ct, cp, min_rv, dep, corr, as.double(args[[1]])),
      parameters_valid = ns$bf_cr_valid_(ct, cp, min_rv, dep, corr),
      stop(sprintf("unknown fixture method for CompetingRisks: %s", method))
    ))
  }
  stop(sprintf("unknown composite target: %s", target))
}

# --- multivariate_distribution path -----------------------------------------------------
# Dirichlet/Multinomial/BivariateEmpirical dispatch to the bespoke bf_dirichlet_val_/
# bf_multinomial_val_/bf_bve_cdf_ glue in mvd.cpp (method + flat numeric args in, double
# out). Extensible: additional multivariate targets add a branch here plus their own
# bf_<name>_val_ entry point.

# Flattens fixture assertion args to a numeric vector. Handles both conventions: a single
# nested vector argument (e.g. pdf args = [[0.3, 0.4, 0.3]]) and flat scalar args (e.g.
# covariance args = [0, 1], log_multivariate_beta args = [1.0, 1.0]).
flatten_mv_args <- function(args) {
  if (length(args) == 1 && is.list(args[[1]])) {
    return(as.double(unlist(args[[1]])))
  }
  as.double(unlist(args))
}

dispatch_multivariate <- function(target, construct, method, args) {
  ns <- asNamespace("bestfitr")
  ar <- flatten_mv_args(args)
  if (target == "Dirichlet") {
    alpha <- vapply(construct$alpha, parse_num, numeric(1))
    return(ns$bf_dirichlet_val_(method, alpha, ar))
  }
  if (target == "Multinomial") {
    n <- as.integer(construct$n)
    p <- vapply(construct$p, parse_num, numeric(1))
    return(ns$bf_multinomial_val_(method, n, p, ar))
  }
  if (target == "BivariateEmpirical") {
    x1 <- vapply(construct$x1, parse_num, numeric(1))
    x2 <- vapply(construct$x2, parse_num, numeric(1))
    p_flat <- as.double(unlist(lapply(construct$p, function(row) vapply(row, parse_num, numeric(1)))))
    transforms <- c(
      if (!is.null(construct$x1_transform)) construct$x1_transform else "None",
      if (!is.null(construct$x2_transform)) construct$x2_transform else "None",
      if (!is.null(construct$p_transform)) construct$p_transform else "None"
    )
    return(ns$bf_bve_cdf_(method, x1, x2, p_flat, length(x1), transforms, ar))
  }
  if (target == "MultivariateNormal") {
    mean <- vapply(construct$mean, parse_num, numeric(1))
    cov_flat <- as.double(unlist(lapply(construct$covariance, function(row) vapply(row, parse_num, numeric(1)))))
    return(ns$bf_mvn_val_(method, mean, cov_flat, ar))
  }
  if (target == "MultivariateStudentT") {
    df <- parse_num(construct$df)
    location <- vapply(construct$location, parse_num, numeric(1))
    scale_flat <- as.double(unlist(lapply(construct$scale, function(row) vapply(row, parse_num, numeric(1)))))
    return(ns$bf_mvt_val_(method, df, location, scale_flat, ar))
  }
  stop(sprintf("unknown multivariate target: %s", target))
}

# --- MultivariateNormal seeded batches --------------------------------------------------
# `cdf` (dim>=3), `interval`, and `mvndst` all draw from the seeded MVNUNI stream, so a
# RUN of consecutive same-method assertions in a seeded case must be evaluated on ONE
# persistent instance via the bf_mvn_*_seq_ glue in mvd.cpp, not dispatched one call at a
# time (which would silently reset the seed between assertions). run_mvn_case() below
# groups consecutive assertions of these methods and batches them; everything else (and
# every case without a "seed") falls through to the stateless per-assertion dispatch
# above, unchanged.

kMvnSeededMethods <- c("cdf", "mvndst", "interval")

flatten_num_list <- function(x) as.double(unlist(lapply(x, parse_num)))

dispatch_mvn_seeded_seq <- function(construct, method, run) {
  ns <- asNamespace("bestfitr")
  seed <- as.integer(construct$seed)
  mean <- vapply(construct$mean, parse_num, numeric(1))
  cov_flat <- as.double(unlist(lapply(construct$covariance, function(row) vapply(row, parse_num, numeric(1)))))

  if (method == "cdf") {
    xs_flat <- unlist(lapply(run, function(a) flatten_num_list(a$args[[1]])))
    return(ns$bf_mvn_cdf_seq_(mean, cov_flat, seed, xs_flat, length(run)))
  }
  if (method == "interval") {
    lowers_flat <- unlist(lapply(run, function(a) flatten_num_list(a$args[[1]])))
    uppers_flat <- unlist(lapply(run, function(a) flatten_num_list(a$args[[2]])))
    return(ns$bf_mvn_interval_seq_(mean, cov_flat, seed, lowers_flat, uppers_flat, length(run)))
  }
  if (method == "mvndst") {
    # args = [n, [lower...], [upper...], [infin...], [correl...], maxpts, abseps, releps]
    n_dim <- as.integer(run[[1]]$args[[1]])
    lower_flat <- unlist(lapply(run, function(a) flatten_num_list(a$args[[2]])))
    upper_flat <- unlist(lapply(run, function(a) flatten_num_list(a$args[[3]])))
    infin_flat <- as.integer(unlist(lapply(run, function(a) unlist(a$args[[4]]))))
    correl_flat <- unlist(lapply(run, function(a) flatten_num_list(a$args[[5]])))
    maxpts_v <- as.integer(vapply(run, function(a) a$args[[6]], numeric(1)))
    abseps_v <- as.double(vapply(run, function(a) a$args[[7]], numeric(1)))
    releps_v <- as.double(vapply(run, function(a) a$args[[8]], numeric(1)))
    return(ns$bf_mvn_mvndst_seq_(n_dim, seed, lower_flat, upper_flat, infin_flat, correl_flat,
                                  maxpts_v, abseps_v, releps_v, length(run)))
  }
  stop(sprintf("unknown seeded MultivariateNormal method: %s", method))
}

run_mvn_case <- function(construct, assertions) {
  seeded <- !is.null(construct$seed)
  i <- 1
  n <- length(assertions)
  while (i <= n) {
    a <- assertions[[i]]
    method <- a$method
    if (seeded && method %in% kMvnSeededMethods) {
      j <- i
      while (j <= n && identical(assertions[[j]]$method, method)) j <- j + 1
      run <- assertions[i:(j - 1)]
      actuals <- dispatch_mvn_seeded_seq(construct, method, run)
      for (idx in seq_along(run)) check_assertion(actuals[idx], run[[idx]])
      i <- j
    } else {
      args <- if (is.null(a$args)) list() else a$args
      actual <- dispatch_multivariate("MultivariateNormal", construct, method, args)
      check_assertion(actual, a)
      i <- i + 1
    }
  }
}

# --- bivariate_copula path ---------------------------------------------------------------
# Every copula shares BivariateCopula's uniform theta/get_copula_parameters/pdf/cdf/... API
# (unlike multivariate_distribution's Dirichlet/Multinomial/BivariateEmpirical/..., which
# share no common surface), so this path is fully generic through the factory-driven
# bf_cop_val_/bf_cop_fit_ glue in copula.cpp -- no per-target branching, mirroring
# copula_factory.hpp's rationale. construct is either {"theta": x} (optionally {"theta":
# x, "df": y} for 2-parameter copulas, and/or {"marginals": {"targets", "params"}} to
# attach marginals directly -- used by the "random_value" sampling oracles) or {"fit": {"x",
# "y", "method", "marginals"?}}; see fixtures/README.md for the full schema.

build_copula_params <- function(construct) {
  p <- parse_num(construct$theta)
  if (!is.null(construct$df)) p <- c(p, parse_num(construct$df))
  p
}

dispatch_copula <- function(target, params, method, args, marg_x_target = "", marg_x_params = numeric(0),
                             marg_y_target = "", marg_y_params = numeric(0)) {
  ns <- asNamespace("bestfitr")
  ar <- flatten_mv_args(if (length(args) == 0) list() else args)
  ns$bf_cop_val_(target, params, method, ar, marg_x_target, marg_x_params, marg_y_target, marg_y_params)
}

run_copula_case <- function(target, construct, assertions, datasets) {
  ns <- asNamespace("bestfitr")
  if (!is.null(construct$fit)) {
    fit <- construct$fit
    x <- as.double(unlist(datasets[[fit$x]]))
    y <- as.double(unlist(datasets[[fit$y]]))
    marg_x <- if (!is.null(fit$marginals)) fit$marginals[[1]] else ""
    marg_y <- if (!is.null(fit$marginals)) fit$marginals[[2]] else ""
    result <- ns$bf_cop_fit_(target, x, y, fit$method, marg_x, marg_y)
    for (a in assertions) {
      args <- if (is.null(a$args)) list() else a$args
      actual <- switch(a$method,
        theta = result$params[[1]],
        df = result$params[[2]],
        marginal_param = {
          which <- args[[1]]
          idx <- as.integer(args[[2]]) + 1L
          if (which == "x") result$marg_x_params[[idx]] else result$marg_y_params[[idx]]
        },
        stop(sprintf("unsupported post-fit copula fixture method: %s", a$method))
      )
      check_assertion(actual, a)
    }
  } else {
    params <- build_copula_params(construct)
    marg <- construct$marginals
    marg_x_target <- if (!is.null(marg)) marg$targets[[1]] else ""
    marg_y_target <- if (!is.null(marg)) marg$targets[[2]] else ""
    marg_x_params <- if (!is.null(marg)) vapply(marg$params[[1]], parse_num, numeric(1)) else numeric(0)
    marg_y_params <- if (!is.null(marg)) vapply(marg$params[[2]], parse_num, numeric(1)) else numeric(0)
    for (a in assertions) {
      args <- if (is.null(a$args)) list() else a$args
      actual <- dispatch_copula(target, params, a$method, args, marg_x_target, marg_x_params,
                                 marg_y_target, marg_y_params)
      check_assertion(actual, a)
    }
  }
}

# --- mcmc_sampler path -------------------------------------------------------------------
# Inherently STATEFUL (unlike multivariate_distribution's/bivariate_copula's per-assertion
# dispatch): one bf_mcmc_run_ call per case builds the model via the registry, configures the
# sampler from construct$settings, and samples() ONCE; every assertion in the case reads the
# single returned list. See fixtures/README.md's mcmc_sampler schema for the full method list
# and tolerance policy.

dispatch_mcmc <- function(result, method, args) {
  i1 <- function(x) as.integer(x) + 1L  # 0-based fixture index -> 1-based R index
  switch(method,
    posterior_mean       = result$posterior_mean[[i1(args[[1]])]],
    posterior_sd          = result$posterior_sd[[i1(args[[1]])]],
    posterior_median       = result$posterior_median[[i1(args[[1]])]],
    posterior_lower_ci     = result$posterior_lower_ci[[i1(args[[1]])]],
    posterior_upper_ci     = result$posterior_upper_ci[[i1(args[[1]])]],
    chain_value    = result$chains[[i1(args[[1]])]][i1(args[[2]]), i1(args[[3]])],
    chain_fitness  = result$chain_fitness[[i1(args[[1]])]][[i1(args[[2]])]],
    map_value      = result$map_values[[i1(args[[1]])]],
    map_fitness    = result$map_fitness[[1]],
    acceptance_rate      = result$acceptance_rates[[i1(args[[1]])]],
    mean_log_likelihood  = result$mean_log_likelihood[[i1(args[[1]])]],
    rhat = result$rhat[[i1(args[[1]])]],
    ess  = result$ess[[i1(args[[1]])]],
    stop(sprintf("unknown mcmc_sampler fixture method: %s", method))
  )
}

run_mcmc_case <- function(target, construct, assertions, datasets) {
  ns <- asNamespace("bestfitr")
  model <- construct$model
  data <- as.double(unlist(datasets[[model$dataset]]))
  settings <- if (!is.null(construct$settings)) construct$settings else list()
  result <- ns$bf_mcmc_run_(target, model$name, model$family, data, settings)
  for (a in assertions) {
    args <- if (is.null(a$args)) list() else a$args
    actual <- dispatch_mcmc(result, a$method, args)
    check_assertion(actual, a)
  }
}

# --- bootstrap path ------------------------------------------------------------------------
# Inherently STATEFUL like mcmc_sampler: one bf_bootstrap_run_ call per case builds the model
# via the registry, runs() (or run_with_studentized_bootstrap()) ONCE, and computes confidence
# intervals ONCE; every assertion in the case reads the single returned list. See
# fixtures/README.md's bootstrap schema for the full method list and tolerance policy.

dispatch_bootstrap <- function(result, method, args) {
  i1 <- function(x) as.integer(x) + 1L  # 0-based fixture index -> 1-based R index
  switch(method,
    statistic_lower_ci   = result$statistic_lower_ci[[i1(args[[1]])]],
    statistic_upper_ci   = result$statistic_upper_ci[[i1(args[[1]])]],
    parameter_lower_ci   = result$parameter_lower_ci[[i1(args[[1]])]],
    parameter_upper_ci   = result$parameter_upper_ci[[i1(args[[1]])]],
    population_estimate  = result$population_estimate[[i1(args[[1]])]],
    valid_count          = result$valid_count[[i1(args[[1]])]],
    replicate_value      = result$replicate_values[i1(args[[1]]), i1(args[[2]])],
    stop(sprintf("unknown bootstrap fixture method: %s", method))
  )
}

run_bootstrap_case <- function(construct, assertions, datasets) {
  ns <- asNamespace("bestfitr")
  dataset <- if (!is.null(construct$dataset)) as.double(unlist(datasets[[construct$dataset]])) else numeric(0)
  probabilities <- vapply(construct$probabilities, parse_num, numeric(1))
  mu <- if (!is.null(construct$mu)) construct$mu else 0
  sigma <- if (!is.null(construct$sigma)) construct$sigma else 0
  sample_size <- if (!is.null(construct$sample_size)) as.integer(construct$sample_size) else 0L
  max_retries <- if (!is.null(construct$max_retries)) as.integer(construct$max_retries) else 20L
  run <- if (!is.null(construct$run)) construct$run else "regular"
  alpha <- if (!is.null(construct$alpha)) construct$alpha else 0.1

  result <- ns$bf_bootstrap_run_(construct$model, mu, sigma, sample_size, probabilities, dataset,
    as.integer(construct$replicates), as.integer(construct$seed), max_retries, run,
    construct$ci_method, alpha)
  for (a in assertions) {
    args <- if (is.null(a$args)) list() else a$args
    actual <- dispatch_bootstrap(result, a$method, args)
    check_assertion(actual, a)
  }
}

# --- model_estimation path -------------------------------------------------------------------
# Inherently STATEFUL like mcmc_sampler/bootstrap: one bf_estimation_run_ (ML/MAP),
# bf_estimation_bayes_run_ (BayesianAnalysis, T12), or bf_model_simulate_ (Simulation, M13)
# call per case builds the model, runs its one stateful call (estimate() or the seeded
# ISimulatable draw), and returns the full result surface; every assertion in the case reads
# that single cached list. See fixtures/README.md's model_estimation section for the full
# method list and the `bic`/`chain_value` design notes. `bic` is the one exception to the
# "cached list" contract for ML/MAP: it takes an actual sample size `n` (C# `GetBIC(
# sampleSize)`), read live from the fixture's `args[[1]]` at dispatch time via
# `bf_estimation_bic_`, not precomputed alongside the rest.
#
# M13: `construct.model` is no longer a flat {family, dataset} pair -- it can name any of the
# four Phase 5 model types, a full censored DataFrame, nonstationary trend specs, and explicit
# parameter values. The parsed spec is re-serialized to JSON (digits = I(17) round-trips
# doubles exactly) and handed to the SHARED C++ builder (bestfit/models/model_spec.hpp), the
# same code path the C++ runner and the Python glue use; only the `dataset` reference is still
# resolved here, like every other fixture kind.

dispatch_estimation <- function(result, method, args, ctx) {
  i1 <- function(x) as.integer(x) + 1L  # 0-based fixture index -> 1-based R index
  switch(method,
    parameter          = result$parameters[[i1(args[[1]])]],
    max_log_likelihood = result$max_log_likelihood[[1]],
    aic                = result$aic[[1]],
    bic                = ctx$bic_fn(as.integer(args[[1]])),  # args[[1]] is a sample size n, not an index
    covariance         = result$covariance[i1(args[[1]]), i1(args[[2]])],
    standard_error     = result$standard_errors[[i1(args[[1]])]],
    correlation        = result$correlation[i1(args[[1]]), i1(args[[2]])],
    dic                = result$dic[[1]],
    waic               = result$waic[[1]],
    looic              = result$looic[[1]],
    posterior_mean     = result$posterior_mean[[i1(args[[1]])]],
    # chain_value [chain, iter, param]: bf_estimation_bayes_run_ returns a flattened
    # `chain_values` vector + `chain_dims` (n_chains, n_iterations, n_params); recover the
    # row-major (chain, iter, param) index the C++/Python/C# access order uses.
    chain_value = {
      d <- result$chain_dims
      idx <- args[[1]] * d[2] * d[3] + args[[2]] * d[3] + args[[3]] + 1L
      result$chain_values[[idx]]
    },
    # simulated_value [i]: the seeded ISimulatable draw cached by bf_model_simulate_ (M13).
    simulated_value = result$simulated[[i1(args[[1]])]],
    # The M14 DataFrame surface (works under any target -- it reads the model, not the
    # estimator): ctx$df_fn() lazily builds the frame surface ONCE per case via
    # bf_model_data_frame_ and memoizes it (the bic lazy-rebuild precedent).
    number_of_low_outliers = ctx$df_fn()$number_of_low_outliers[[1]],
    low_outlier_threshold = ctx$df_fn()$low_outlier_threshold[[1]],
    # plotting_position [kind, i]: kind is "exact" | "interval" | "uncertain", in spec order.
    plotting_position = ctx$df_fn()[[paste0("pp_", args[[1]])]][[i1(args[[2]])]],
    stop(sprintf("unknown model_estimation fixture method: %s", method))
  )
}

run_estimation_case <- function(target, construct, assertions, datasets) {
  ns <- asNamespace("bestfitr")
  model <- construct$model
  # Re-serialize the parsed spec for the shared C++ builder (see the path comment above).
  model_json <- as.character(jsonlite::toJSON(model, auto_unbox = TRUE, digits = I(17)))
  data <- if (!is.null(model$dataset)) as.double(unlist(datasets[[model$dataset]])) else numeric(0)

  # The M14 DataFrame surface: lazily build + memoize the frame-surface list (only cases
  # that actually assert a data-frame method pay for the rebuild; see bf_model_data_frame_).
  df_env <- new.env(parent = emptyenv())
  df_fn <- function() {
    if (is.null(df_env$df)) df_env$df <- ns$bf_model_data_frame_(model_json, data)
    df_env$df
  }

  if (target == "Simulation") {
    seed <- if (!is.null(construct$seed)) as.integer(construct$seed) else -1L
    draws <- ns$bf_model_simulate_(model_json, data, as.integer(construct$sample_size), seed)
    result <- list(simulated = draws)
    ctx <- list()
  } else if (target == "BayesianAnalysis") {
    sampler <- if (!is.null(construct$sampler)) construct$sampler else "DEMCzs"
    s <- construct$settings
    geti <- function(name, default = -1L) if (!is.null(s[[name]])) as.integer(s[[name]]) else default
    result <- ns$bf_estimation_bayes_run_(
      model_json, data, sampler,
      seed = geti("seed"), iterations = geti("iterations"),
      warmup_iterations = geti("warmup_iterations"), number_of_chains = geti("number_of_chains"),
      thinning_interval = geti("thinning_interval"), initial_iterations = geti("initial_iterations"),
      output_length = geti("output_length")
    )
    ctx <- list()
  } else {
    optimizer <- if (!is.null(construct$optimizer)) construct$optimizer else "DifferentialEvolution"
    result <- ns$bf_estimation_run_(target, model_json, data, optimizer)
    ctx <- list(bic_fn = function(n) ns$bf_estimation_bic_(target, model_json, data, optimizer, n))
  }
  ctx$df_fn <- df_fn

  for (a in assertions) {
    args <- if (is.null(a$args)) list() else a$args
    actual <- dispatch_estimation(result, a$method, args, ctx)
    check_assertion(actual, a)
  }
}

test_that("oracle fixtures validate", {
  skip_if_not_installed("jsonlite")
  fdir <- system.file("fixtures", package = "bestfitr")
  files <- list.files(fdir, pattern = "\\.json$", recursive = TRUE, full.names = TRUE)
  expect_gt(length(files), 0)
  for (f in files) {
    spec <- jsonlite::read_json(f, simplifyVector = FALSE)
    # Only validate univariate_distribution, multivariate_distribution,
    # bivariate_copula, mcmc_sampler, bootstrap, and model_estimation fixtures; skip other
    # kinds (e.g. special_function) which are validated in C++ only and are not exposed to
    # the R package.
    if (identical(spec$kind, "model_estimation")) {
      target <- spec$target
      datasets <- spec$datasets
      for (case in spec$cases) {
        run_estimation_case(target, case$construct, case$assertions, datasets)
      }
      next
    }
    if (identical(spec$kind, "bootstrap")) {
      datasets <- spec$datasets
      for (case in spec$cases) {
        run_bootstrap_case(case$construct, case$assertions, datasets)
      }
      next
    }
    if (identical(spec$kind, "mcmc_sampler")) {
      target <- spec$target
      datasets <- spec$datasets
      for (case in spec$cases) {
        run_mcmc_case(target, case$construct, case$assertions, datasets)
      }
      next
    }
    if (identical(spec$kind, "bivariate_copula")) {
      target <- spec$target
      datasets <- spec$datasets
      for (case in spec$cases) {
        run_copula_case(target, case$construct, case$assertions, datasets)
      }
      next
    }
    if (identical(spec$kind, "multivariate_distribution")) {
      target <- spec$target
      for (case in spec$cases) {
        if (identical(target, "MultivariateNormal")) {
          run_mvn_case(case$construct, case$assertions)
        } else {
          for (a in case$assertions) {
            args <- if (is.null(a$args)) list() else a$args
            actual <- dispatch_multivariate(target, case$construct, a$method, args)
            check_assertion(actual, a)
          }
        }
      }
      next
    }
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
