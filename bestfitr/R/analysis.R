#' Bayesian univariate frequency analysis
#'
#' Fit a univariate distribution to a sample with a Bayesian MCMC analysis and return the
#' frequency (quantile) curve, the posterior mean and credible band, and goodness-of-fit
#' scalars. Wraps the shared C++ `UnivariateAnalysis` ported from USACE-RMC RMC.BestFit.
#'
#' Because both the R and Python packages call the identical compiled core with a bit-exact
#' Mersenne Twister, a seeded call returns identical numbers in either language.
#'
#' @param data numeric vector of observations.
#' @param distribution distribution family name (e.g. `"Normal"`, `"GeneralizedExtremeValue"`,
#'   `"LogPearsonTypeIII"`) -- any name the core distribution factory accepts.
#' @param sampler MCMC sampler: `"DEMCz"` (default), `"DEMCzs"`, `"ARWMH"`, or `"NUTS"`.
#' @param iterations number of post-warmup MCMC iterations.
#' @param output_length number of posterior samples used to build the credible band.
#' @param credible_level credible-interval width (e.g. `0.90` for a 90% band).
#' @param seed PRNG seed for the sampler (fixed for reproducibility).
#' @param exceedance_probabilities optional numeric vector of exceedance probabilities at which to
#'   tabulate the curve; when `NULL`, the 25 standard default ordinates are used.
#' @return A named list: `parameters` (point-estimate distribution parameters), `mode_curve`,
#'   `mean_curve`, `lower_ci`, `upper_ci` (one value per exceedance ordinate), and the scalars
#'   `aic`, `bic`, `dic`, `rmse`.
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600)
#' fit <- univariate_analysis(peaks, "Normal", sampler = "DEMCzs",
#'                            iterations = 100, output_length = 400, seed = 12345)
#' fit$parameters
univariate_analysis <- function(data, distribution, sampler = "DEMCz", iterations = 3000L,
                                output_length = 10000L, credible_level = 0.90, seed = 12345L,
                                exceedance_probabilities = NULL) {
  model <- list(family = distribution, dataset = "data")
  model_json <- as.character(jsonlite::toJSON(model, auto_unbox = TRUE, digits = I(17)))
  ep <- if (is.null(exceedance_probabilities)) numeric(0) else as.double(exceedance_probabilities)
  bf_analysis_univariate_run_(
    model_json, as.double(data), sampler, as.integer(iterations), as.integer(output_length),
    as.double(credible_level), as.integer(seed), ep
  )
}

#' Fit and rank candidate distributions
#'
#' Fit each of the 14 candidate distributions to a sample by maximum likelihood and return their
#' goodness-of-fit metrics. Ranking is left to the caller. Wraps the shared C++ `FittingAnalysis`.
#'
#' @param data numeric vector of observations.
#' @return A named list with equal-length vectors `distribution` (candidate name), `aic`, `bic`,
#'   `rmse`, and `converged` (logical, whether the fit succeeded) -- one entry per candidate.
#' @export
#' @examples
#' peaks <- c(45000, 38000, 52000, 61000, 33000, 49000, 55000, 42000, 67000, 39000)
#' gof <- fit_distributions(peaks)
#' gof$distribution[which.min(gof$aic)]
fit_distributions <- function(data) {
  bf_analysis_fit_distributions_(as.double(data))
}

#' Bulletin 17C flood-frequency analysis
#'
#' Fit a Bulletin 17C (log-Pearson Type III) flood-frequency model by the generalized method of
#' moments and return the Cohn-style delta-method confidence intervals, the fitted parameters, and
#' the sandwich covariance. Wraps the shared C++ `Bulletin17CAnalysis`.
#'
#' @param data numeric vector of annual peak observations.
#' @param uncertainty_method uncertainty-quantification method: `"MultivariateNormal"` (default) or
#'   `"Bootstrap"`. The `"LinkedMultivariateNormal"` / `"BiasCorrectedBootstrap"` methods are
#'   deferred and raise an error.
#' @param output_length number of parameter-set draws used for uncertainty quantification.
#' @param seed PRNG seed for the uncertainty draw.
#' @param confidence_level confidence level for the intervals (e.g. `0.90`).
#' @param exceedance_probabilities optional numeric vector of exceedance probabilities; when
#'   `NULL`, the 25 standard default ordinates are used.
#' @return A named list: `exceedance_probabilities`, `point_estimates` (log10 space), `lower_ci`,
#'   `upper_ci` (discharge space), `confidence_level`, `beta1`, `nu`, `quantile_variance`,
#'   `parameters` (fitted location/scale/shape), and `covariance` (the p x p sandwich covariance).
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600,
#'            19200, 13800, 25600, 10500, 16900)
#' ci <- bulletin17c_analysis(peaks, output_length = 200, seed = 12345)
#' ci$confidence_level
bulletin17c_analysis <- function(data, uncertainty_method = "MultivariateNormal",
                                 output_length = 10000L, seed = 12345L, confidence_level = 0.90,
                                 exceedance_probabilities = NULL) {
  model <- list(type = "bulletin17c", family = "LogPearsonTypeIII", dataset = "data")
  model_json <- as.character(jsonlite::toJSON(model, auto_unbox = TRUE, digits = I(17)))
  ep <- if (is.null(exceedance_probabilities)) numeric(0) else as.double(exceedance_probabilities)
  res <- bf_analysis_b17c_run_(
    model_json, as.double(data), uncertainty_method, as.integer(output_length),
    as.integer(seed), as.double(confidence_level), ep
  )
  # Reshape the flat row-major covariance into a p x p matrix (empty when unestimated).
  d <- res$covariance_dim
  if (length(res$covariance) > 0 && d > 0) {
    res$covariance <- matrix(res$covariance, nrow = d, ncol = d, byrow = TRUE)
  }
  res$covariance_dim <- NULL
  res
}
