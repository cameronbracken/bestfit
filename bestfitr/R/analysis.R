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
#' @param thinning_interval MCMC thinning interval; `-1` (default) keeps the sampler's own default.
#' @details The MCMC warmup (burn-in) length is set automatically to `max(50, iterations / 2)`; it
#'   is not a user parameter.
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
                                exceedance_probabilities = NULL, thinning_interval = -1L) {
  model <- list(family = distribution, dataset = "data")
  model_json <- as.character(jsonlite::toJSON(model, auto_unbox = TRUE, digits = I(17)))
  ep <- if (is.null(exceedance_probabilities)) numeric(0) else as.double(exceedance_probabilities)
  bf_analysis_univariate_run_(
    model_json, as.double(data), sampler, as.integer(iterations), as.integer(output_length),
    as.double(credible_level), as.integer(seed), ep, as.integer(thinning_interval)
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

# Internal: assemble the model JSON and call the shared per-family dispatch (D5). The seven
# per-family analyses share the univariate_analysis result surface, so the wrappers below differ
# only in the model spec they build. `training_time_steps` / `forecasting_time_steps` are ignored
# by the univariate-family analyses (passed as -1).
.bf_family_run <- function(analysis_type, model, data, sampler, iterations, output_length,
                           credible_level, seed, exceedance_probabilities, thinning_interval,
                           training_time_steps, forecasting_time_steps) {
  model_json <- as.character(jsonlite::toJSON(model, auto_unbox = TRUE, digits = I(17)))
  ep <- if (is.null(exceedance_probabilities)) numeric(0) else as.double(exceedance_probabilities)
  bf_analysis_family_run_(
    analysis_type, model_json, as.double(data), sampler, as.integer(iterations),
    as.integer(output_length), as.double(credible_level), as.integer(seed), ep,
    as.integer(thinning_interval), as.integer(training_time_steps),
    as.integer(forecasting_time_steps)
  )
}

#' Bayesian mixture-model frequency analysis
#'
#' Fit a finite mixture of 1-3 component distributions to a sample with a Bayesian MCMC analysis
#' and return the frequency (quantile) curve, credible band, and goodness-of-fit scalars. Wraps the
#' shared C++ `MixtureAnalysis`.
#'
#' @param data numeric vector of observations.
#' @param families character vector of component distribution family names (e.g. `c("Normal",
#'   "Normal")`).
#' @param zero_inflated logical; add a zero-inflation component (default `FALSE`).
#' @inheritParams univariate_analysis
#' @return A named list: `parameters`, `mode_curve`, `mean_curve`, `lower_ci`, `upper_ci`, and the
#'   scalars `aic`, `bic`, `dic`, `rmse`.
#' @export
#' @examples
#' x <- c(520, 580, 610, 700, 760, 850, 950, 5000, 5400, 5800, 6300, 6800)
#' fit <- mixture_analysis(x, c("Normal", "Normal"), iterations = 100, output_length = 200,
#'                         seed = 12345, thinning_interval = 1)
#' fit$aic
mixture_analysis <- function(data, families, zero_inflated = FALSE, sampler = "DEMCz",
                             iterations = 3000L, output_length = 10000L, credible_level = 0.90,
                             seed = 12345L, exceedance_probabilities = NULL,
                             thinning_interval = -1L) {
  model <- list(
    type = "mixture", families = as.list(as.character(families)),
    zero_inflated = as.logical(zero_inflated), dataset = "data"
  )
  .bf_family_run(
    "mixture", model, data, sampler, iterations, output_length, credible_level, seed,
    exceedance_probabilities, thinning_interval, -1L, -1L
  )
}

#' Bayesian competing-risk frequency analysis
#'
#' Fit a competing-risks model (the observed maximum of several independent parent distributions)
#' with a Bayesian MCMC analysis. Wraps the shared C++ `CompetingRiskAnalysis`.
#'
#' @param data numeric vector of observations.
#' @param families character vector of parent distribution family names (e.g. `c("Gumbel",
#'   "Weibull")`).
#' @inheritParams univariate_analysis
#' @return A named list with the same shape as [mixture_analysis()].
#' @export
#' @examples
#' x <- c(7872, 8624, 5894, 12540, 8307, 6009, 13320, 10641, 8458, 8838, 11742, 9117)
#' fit <- competing_risk_analysis(x, c("Gumbel", "Weibull"), iterations = 100,
#'                                output_length = 200, seed = 12345, thinning_interval = 1)
#' fit$aic
competing_risk_analysis <- function(data, families, sampler = "DEMCz", iterations = 3000L,
                                    output_length = 10000L, credible_level = 0.90, seed = 12345L,
                                    exceedance_probabilities = NULL, thinning_interval = -1L) {
  model <- list(
    type = "competing_risks", families = as.list(as.character(families)), dataset = "data"
  )
  .bf_family_run(
    "competing_risk", model, data, sampler, iterations, output_length, credible_level, seed,
    exceedance_probabilities, thinning_interval, -1L, -1L
  )
}

#' Bayesian point-process (peaks-over-threshold) frequency analysis
#'
#' Fit a peaks-over-threshold point-process model with a Bayesian MCMC analysis. Wraps the shared
#' C++ `PointProcessAnalysis`.
#'
#' @param data numeric vector of peak observations.
#' @param threshold optional numeric threshold (when `NULL`, the model default cascade is used).
#' @param total_years optional numeric exposure length in years.
#' @inheritParams univariate_analysis
#' @return A named list with the same shape as [mixture_analysis()].
#' @export
#' @examples
#' x <- c(950, 1020, 1130, 1250, 1090, 1430, 1180, 1620, 1300, 1550, 1210)
#' fit <- point_process_analysis(x, threshold = 900, total_years = 20, iterations = 100,
#'                               output_length = 200, seed = 12345, thinning_interval = 1)
#' fit$dic
point_process_analysis <- function(data, threshold = NULL, total_years = NULL, sampler = "DEMCz",
                                   iterations = 3000L, output_length = 10000L, credible_level = 0.90,
                                   seed = 12345L, exceedance_probabilities = NULL,
                                   thinning_interval = -1L) {
  model <- list(type = "point_process", dataset = "data")
  if (!is.null(threshold)) model$threshold <- as.double(threshold)
  if (!is.null(total_years)) model$total_years <- as.double(total_years)
  .bf_family_run(
    "point_process", model, data, sampler, iterations, output_length, credible_level, seed,
    exceedance_probabilities, thinning_interval, -1L, -1L
  )
}

# Internal: shared time-series wrapper body (D5). AR / MA / ARIMA / ARIMAX differ only in the
# model spec; all return the univariate_analysis surface (curves are the posterior forecast).
.bf_time_series_run <- function(analysis_type, model, data, sampler, iterations, output_length,
                                credible_level, seed, thinning_interval, training_time_steps,
                                forecasting_time_steps) {
  .bf_family_run(
    analysis_type, model, data, sampler, iterations, output_length, credible_level, seed, NULL,
    thinning_interval,
    if (is.null(training_time_steps)) -1L else as.integer(training_time_steps),
    as.integer(forecasting_time_steps)
  )
}

#' Bayesian autoregressive AR(p) time-series analysis
#'
#' Fit an AR(p) autoregressive model with a Bayesian MCMC analysis and return the posterior
#' forecast curve, credible band, and goodness-of-fit scalars. Wraps the shared C++ `ARAnalysis`.
#'
#' @param data numeric vector of the observed time series (in sequence order).
#' @param order_p autoregressive order (default `1`).
#' @param include_intercept logical; include an intercept term (default `TRUE`).
#' @param training_time_steps number of leading steps used for calibration; when `NULL` the model
#'   default (`max(30, floor(0.8 * n))`) is used, which is invalid for short series -- set it
#'   explicitly (e.g. `15`) when `n` is small.
#' @param forecasting_time_steps number of steps to forecast past the observed series (default `0`).
#' @inheritParams univariate_analysis
#' @return A named list: `parameters` (posterior point estimate), `mode_curve`, `mean_curve`,
#'   `lower_ci`, `upper_ci` (the forecast + credible band over the observed range plus the forecast
#'   horizon), and the scalars `aic`, `bic`, `dic`, `rmse`.
#' @export
#' @examples
#' x <- c(10.2, 11.5, 9.8, 12.1, 13.4, 11.9, 10.6, 12.8, 14.0, 13.1, 11.7, 12.5, 13.9, 15.2,
#'        14.1, 12.9, 13.6, 15.0, 16.2, 14.8)
#' fit <- ar_analysis(x, order_p = 1, training_time_steps = 15, forecasting_time_steps = 3,
#'                    iterations = 100, output_length = 200, seed = 12345, thinning_interval = 1)
#' length(fit$mode_curve)
ar_analysis <- function(data, order_p = 1L, include_intercept = TRUE, training_time_steps = NULL,
                        forecasting_time_steps = 0L, sampler = "DEMCz", iterations = 3000L,
                        output_length = 10000L, credible_level = 0.90, seed = 12345L,
                        thinning_interval = -1L) {
  model <- list(
    type = "time_series", subtype = "ar", dataset = "data",
    orders = list(p = as.integer(order_p)), include_intercept = as.logical(include_intercept)
  )
  .bf_time_series_run(
    "ar", model, data, sampler, iterations, output_length, credible_level, seed, thinning_interval,
    training_time_steps, forecasting_time_steps
  )
}

#' Bayesian moving-average MA(q) time-series analysis
#'
#' Fit an MA(q) moving-average model with a Bayesian MCMC analysis. Wraps the shared C++
#' `MAAnalysis`.
#'
#' @param data numeric vector of the observed time series (in sequence order).
#' @param order_q moving-average order (default `1`).
#' @inheritParams ar_analysis
#' @return A named list with the same shape as [ar_analysis()].
#' @export
#' @examples
#' x <- c(10.2, 11.5, 9.8, 12.1, 13.4, 11.9, 10.6, 12.8, 14.0, 13.1, 11.7, 12.5, 13.9, 15.2,
#'        14.1, 12.9, 13.6, 15.0, 16.2, 14.8)
#' fit <- ma_analysis(x, order_q = 1, training_time_steps = 15, forecasting_time_steps = 3,
#'                    iterations = 100, output_length = 200, seed = 12345, thinning_interval = 1)
#' fit$rmse
ma_analysis <- function(data, order_q = 1L, include_intercept = TRUE, training_time_steps = NULL,
                        forecasting_time_steps = 0L, sampler = "DEMCz", iterations = 3000L,
                        output_length = 10000L, credible_level = 0.90, seed = 12345L,
                        thinning_interval = -1L) {
  model <- list(
    type = "time_series", subtype = "ma", dataset = "data",
    orders = list(q = as.integer(order_q)), include_intercept = as.logical(include_intercept)
  )
  .bf_time_series_run(
    "ma", model, data, sampler, iterations, output_length, credible_level, seed, thinning_interval,
    training_time_steps, forecasting_time_steps
  )
}

#' Bayesian ARIMA(p,d,q) time-series analysis
#'
#' Fit an ARIMA(p,d,q) model with a Bayesian MCMC analysis. Wraps the shared C++ `ARIMAAnalysis`.
#'
#' @param data numeric vector of the observed time series (in sequence order).
#' @param order_p autoregressive order (default `1`).
#' @param order_d differencing order (default `0`).
#' @param order_q moving-average order (default `1`).
#' @inheritParams ar_analysis
#' @return A named list with the same shape as [ar_analysis()].
#' @export
#' @examples
#' x <- c(10.2, 11.5, 9.8, 12.1, 13.4, 11.9, 10.6, 12.8, 14.0, 13.1, 11.7, 12.5, 13.9, 15.2,
#'        14.1, 12.9, 13.6, 15.0, 16.2, 14.8)
#' fit <- arima_analysis(x, order_p = 1, order_d = 1, order_q = 1, training_time_steps = 15,
#'                       forecasting_time_steps = 3, iterations = 100, output_length = 200,
#'                       seed = 12345, thinning_interval = 1)
#' fit$aic
arima_analysis <- function(data, order_p = 1L, order_d = 0L, order_q = 1L, include_intercept = TRUE,
                           training_time_steps = NULL, forecasting_time_steps = 0L,
                           sampler = "DEMCz", iterations = 3000L, output_length = 10000L,
                           credible_level = 0.90, seed = 12345L, thinning_interval = -1L) {
  model <- list(
    type = "time_series", subtype = "arima", dataset = "data",
    orders = list(p = as.integer(order_p), d = as.integer(order_d), q = as.integer(order_q)),
    include_intercept = as.logical(include_intercept)
  )
  .bf_time_series_run(
    "arima", model, data, sampler, iterations, output_length, credible_level, seed,
    thinning_interval, training_time_steps, forecasting_time_steps
  )
}

#' Bayesian ARIMAX(p,d,q) time-series analysis
#'
#' Fit an ARIMAX model (ARIMA with a deterministic trend and optional exogenous covariates) with a
#' Bayesian MCMC analysis. Wraps the shared C++ `ARIMAXAnalysis`. Covariate forecasting past the
#' observed range is not exposed; run fit-only (`forecasting_time_steps = 0`) with covariates.
#'
#' @param data numeric vector of the observed time series (in sequence order).
#' @param order_p autoregressive order (default `1`).
#' @param order_d differencing order (default `0`).
#' @param order_q moving-average order (default `0`).
#' @param order_b exogenous-covariate lag order (default `0`).
#' @param trend deterministic trend: `"None"` (default), `"Linear"`, `"Quadratic"`, or `"Cubic"`.
#' @inheritParams ar_analysis
#' @return A named list with the same shape as [ar_analysis()].
#' @export
#' @examples
#' x <- c(10.2, 11.5, 9.8, 12.1, 13.4, 11.9, 10.6, 12.8, 14.0, 13.1, 11.7, 12.5, 13.9, 15.2,
#'        14.1, 12.9, 13.6, 15.0, 16.2, 14.8)
#' fit <- arimax_analysis(x, trend = "Linear", training_time_steps = 15,
#'                        forecasting_time_steps = 0, iterations = 100, output_length = 200,
#'                        seed = 12345, thinning_interval = 1)
#' fit$bic
arimax_analysis <- function(data, order_p = 1L, order_d = 0L, order_q = 0L, order_b = 0L,
                            trend = "None", include_intercept = TRUE, training_time_steps = NULL,
                            forecasting_time_steps = 0L, sampler = "DEMCz", iterations = 3000L,
                            output_length = 10000L, credible_level = 0.90, seed = 12345L,
                            thinning_interval = -1L) {
  model <- list(
    type = "time_series", subtype = "arimax", dataset = "data",
    orders = list(
      p = as.integer(order_p), d = as.integer(order_d), q = as.integer(order_q),
      b = as.integer(order_b)
    ),
    include_intercept = as.logical(include_intercept), trend = as.character(trend)
  )
  .bf_time_series_run(
    "arimax", model, data, sampler, iterations, output_length, credible_level, seed,
    thinning_interval, training_time_steps, forecasting_time_steps
  )
}

#' Bayesian estimation diagnostics (leverage / influence / prior influence)
#'
#' Fit a univariate distribution to a sample with a Bayesian MCMC analysis and compute the three
#' estimation diagnostics: observation leverage (from the posterior Hessian at the MAP), PSIS-LOO
#' influence (per-observation Pareto-k), and prior influence (prior-vs-data log-likelihood share).
#' Wraps the shared C++ `BayesianAnalysis` diagnostics.
#'
#' @param data numeric vector of observations.
#' @param distribution distribution family name (any name the core distribution factory accepts).
#' @param sampler MCMC sampler: `"DEMCz"` (default), `"DEMCzs"`, `"ARWMH"`, or `"NUTS"`.
#' @param iterations number of post-warmup MCMC iterations.
#' @param output_length number of posterior samples retained.
#' @param seed PRNG seed for the sampler.
#' @param thinning_interval MCMC thinning interval; `-1` (default) keeps the sampler's own default.
#' @param thin_every prior-influence posterior thinning stride (default `10`).
#' @return A named list with three sub-lists: `leverage` (`index`, `leverage`, `fit_influence`,
#'   `variance_influence`, `value` arrays; `prior_leverage`; `total_leverage`,
#'   `total_fit_influence`, `total_variance_influence`), `influence` (`pareto_k`, `elpd_loo` arrays;
#'   `count`, `mean_pareto_k`, `max_pareto_k`, `count_pareto_k_above_05`/`_07`/`_10`,
#'   `proportion_problematic`, `is_reliable`), and `prior_influence` (`count`,
#'   `prior_precision_share`, `total_prior_log_likelihood`, `total_data_log_likelihood`,
#'   `prior_to_data_ratio`, `is_prior_influential`, `mean_prior_precision_share`).
#' @export
#' @examples
#' peaks <- c(12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600)
#' d <- estimation_diagnostics(peaks, "Normal", iterations = 100, output_length = 200,
#'                             seed = 12345, thinning_interval = 1)
#' d$influence$max_pareto_k
estimation_diagnostics <- function(data, distribution, sampler = "DEMCz", iterations = 3000L,
                                   output_length = 10000L, seed = 12345L, thinning_interval = -1L,
                                   thin_every = 10L) {
  model <- list(family = distribution, dataset = "data")
  model_json <- as.character(jsonlite::toJSON(model, auto_unbox = TRUE, digits = I(17)))
  bf_analysis_diagnostics_run_(
    model_json, as.double(data), sampler, as.integer(iterations), as.integer(output_length),
    as.integer(seed), as.integer(thinning_interval), as.integer(thin_every)
  )
}
