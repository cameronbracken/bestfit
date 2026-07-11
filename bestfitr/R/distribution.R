# Public distribution interface over the factory-dispatched univariate glue in
# src/dist.cpp. A `bestfit_dist` is a plain classed list (family + params); every verb
# re-dispatches through the stateless C++ entry points, so no C++ object lifetime leaks
# into R. Composite families (Truncated/Mixture/CompetingRisks/Empirical/KernelDensity)
# are not constructible here yet; they remain internal fixture glue.

#' Create a univariate distribution object
#'
#' Construct a distribution from its family name and parameter vector. The
#' result is a lightweight object accepted by [dist_pdf()], [dist_cdf()],
#' [dist_quantile()], [dist_random()], [dist_moments()], and friends. All 38
#' factory-constructible families of the USACE-RMC Numerics library are
#' supported; see [distribution_names()] for the list.
#'
#' Parameters are positional, in the same order as the C# constructor for the
#' family (for example `Normal` takes `c(mean, sd)` and
#' `GeneralizedExtremeValue` takes `c(location, scale, shape)`). Use
#' [dist_params()] on a constructed object to see the parameter names.
#'
#' @param family the distribution family name, e.g. `"Normal"`,
#'   `"LogNormal"`, `"Gumbel"`, `"GeneralizedExtremeValue"`.
#' @param params numeric vector of parameters, in constructor order.
#' @return An object of class `bestfit_dist`.
#' @seealso [dist_fit()] to estimate one from data, [distribution_names()].
#' @export
#' @examples
#' d <- distribution("Normal", c(100, 15))
#' d
#' dist_cdf(d, 100)
distribution <- function(family, params) {
  if (!is.character(family) || length(family) != 1L) {
    stop("`family` must be a single distribution name; see distribution_names()")
  }
  if (!family %in% bf_dist_names_()) {
    stop(sprintf(
      "unknown distribution family '%s'; see distribution_names() for the %d supported names",
      family, length(bf_dist_names_())
    ))
  }
  params <- as.double(params)
  expected <- bf_dist_parameter_names_(family)$short
  if (length(expected) > 0L && length(params) != length(expected)) {
    stop(sprintf(
      "'%s' expects %d parameters (%s), got %d",
      family, length(expected), paste(expected, collapse = ", "), length(params)
    ))
  }
  structure(list(family = family, params = params), class = "bestfit_dist")
}

#' @export
print.bestfit_dist <- function(x, ...) {
  p <- dist_params(x)
  cat(sprintf(
    "<bestfit_dist> %s(%s)\n", x$family,
    paste(sprintf("%s = %g", names(p), p), collapse = ", ")
  ))
  if (!bf_dist_valid_(x$family, x$params)) {
    cat("  (parameters are not valid for this family)\n")
  }
  invisible(x)
}

check_dist <- function(d) {
  if (!inherits(d, "bestfit_dist")) {
    stop("`d` must be a bestfit_dist object; create one with distribution() or dist_fit()")
  }
  d
}

#' Distribution functions
#'
#' Density, log-density, distribution, quantile, and random-generation
#' functions for a [distribution()] object. All are vectorized over their
#' first numeric argument and evaluate in the shared C++ core, so results are
#' identical to the Python package and to the upstream C# library.
#'
#' `dist_random()` draws from the same seeded Mersenne Twister stream as the
#' C# `GenerateRandomValues(sampleSize, seed)`: a given `seed` reproduces the
#' C# draws bit-for-bit (and matches `bestfitpy` exactly).
#'
#' @param d a `bestfit_dist` object from [distribution()] or [dist_fit()].
#' @param x,q numeric vector of quantiles.
#' @param p numeric vector of probabilities in `(0, 1)`.
#' @param n number of draws.
#' @param seed integer seed for reproducible draws; `NULL` (the default) seeds
#'   from the clock.
#' @return A numeric vector the same length as `x`, `q`, or `p` (`n` for
#'   `dist_random()`).
#' @name dist
#' @examples
#' d <- distribution("Gumbel", c(100, 10))
#' dist_pdf(d, c(95, 100, 120))
#' dist_quantile(d, c(0.5, 0.9, 0.99))
#' dist_random(d, 5, seed = 123)
NULL

#' @rdname dist
#' @export
dist_pdf <- function(d, x) {
  check_dist(d)
  bf_dist_pdf_v_(d$family, d$params, as.double(x))
}

#' @rdname dist
#' @export
dist_log_pdf <- function(d, x) {
  check_dist(d)
  bf_dist_log_pdf_v_(d$family, d$params, as.double(x))
}

#' @rdname dist
#' @export
dist_cdf <- function(d, q) {
  check_dist(d)
  bf_dist_cdf_v_(d$family, d$params, as.double(q))
}

#' @rdname dist
#' @export
dist_quantile <- function(d, p) {
  check_dist(d)
  bf_dist_quantile_v_(d$family, d$params, as.double(p))
}

#' @rdname dist
#' @export
dist_random <- function(d, n, seed = NULL) {
  check_dist(d)
  seed <- if (is.null(seed)) -1L else as.integer(seed)
  bf_dist_random_(d$family, d$params, as.integer(n), seed)
}

#' Distribution properties
#'
#' Moments, parameters, linear moments (L-moments), and log-likelihood of a
#' [distribution()] object.
#'
#' @param d a `bestfit_dist` object from [distribution()] or [dist_fit()].
#' @param data numeric vector of observations.
#' @return `dist_moments()` returns a named numeric vector (`mean`, `median`,
#'   `mode`, `sd`, `skewness`, `kurtosis`, `minimum`, `maximum`; entries are
#'   `NaN` where undefined). `dist_params()` returns the parameter vector named
#'   with the family's short-form parameter names. `dist_lmoments()` returns
#'   the first four L-moments (errors for families without L-moment support).
#'   `dist_log_likelihood()` returns a single numeric value.
#' @name dist_properties
#' @examples
#' d <- distribution("Normal", c(100, 15))
#' dist_moments(d)
#' dist_params(d)
NULL

#' @rdname dist_properties
#' @export
dist_moments <- function(d) {
  check_dist(d)
  bf_dist_moments_(d$family, d$params)
}

#' @rdname dist_properties
#' @export
dist_params <- function(d) {
  check_dist(d)
  out <- d$params
  nm <- bf_dist_parameter_names_(d$family)$short
  if (length(nm) == length(out)) names(out) <- nm
  out
}

#' @rdname dist_properties
#' @export
dist_lmoments <- function(d) {
  check_dist(d)
  bf_dist_linear_moments_(d$family, d$params)
}

#' @rdname dist_properties
#' @export
dist_log_likelihood <- function(d, data) {
  check_dist(d)
  bf_dist_log_likelihood_(d$family, d$params, as.double(data))
}

#' Fit a distribution to data
#'
#' Estimate the parameters of a distribution family from a sample and return
#' the fitted [distribution()] object. Mirrors the C#
#' `Estimate(data, ParameterEstimationMethod)` API of the Numerics library.
#'
#' @param family the distribution family name; see [distribution_names()].
#' @param data numeric vector of observations.
#' @param method estimation method: `"mle"` (maximum likelihood, the default),
#'   `"lmom"` (L-moments), or `"mom"` (product moments). Not every family
#'   supports every method; unsupported combinations error.
#' @return A fitted `bestfit_dist` object.
#' @export
#' @examples
#' d <- distribution("Gumbel", c(100, 10))
#' x <- dist_random(d, 200, seed = 42)
#' dist_fit("Gumbel", x, method = "mle")
dist_fit <- function(family, data, method = c("mle", "lmom", "mom")) {
  method <- match.arg(method)
  params <- bf_dist_fit_(family, as.double(data), method)
  distribution(family, params)
}

#' List the supported distribution families
#'
#' @return A character vector of the distribution family names accepted by
#'   [distribution()] and [dist_fit()].
#' @export
#' @examples
#' distribution_names()
distribution_names <- function() {
  bf_dist_names_()
}
