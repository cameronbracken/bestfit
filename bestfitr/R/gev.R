#' Generalized Extreme Value distribution
#'
#' Density, distribution, and quantile functions for the Generalized Extreme
#' Value (GEV) distribution, parameterised by location \eqn{\xi}, scale
#' \eqn{\alpha}, and shape \eqn{\kappa}. These wrap the shared C++ core ported
#' from the USACE-RMC Numerics library.
#'
#' @param x,q numeric vector of quantiles.
#' @param p numeric vector of probabilities.
#' @param location location parameter \eqn{\xi}.
#' @param scale scale parameter \eqn{\alpha} (must be positive).
#' @param shape shape parameter \eqn{\kappa}.
#'
#' @return A numeric vector the same length as `x`, `q`, or `p`.
#' @name gev
#' @examples
#' dgev(100, location = 100, scale = 10, shape = 0)
#' qgev(0.99, location = 10849, scale = 5745.6, shape = 0.005)
NULL

#' @rdname gev
#' @export
dgev <- function(x, location = 0, scale = 1, shape = 0) {
  bf_gev_pdf_(as.double(x), location, scale, shape)
}

#' @rdname gev
#' @export
pgev <- function(q, location = 0, scale = 1, shape = 0) {
  bf_gev_cdf_(as.double(q), location, scale, shape)
}

#' @rdname gev
#' @export
qgev <- function(p, location = 0, scale = 1, shape = 0) {
  bf_gev_quantile_(as.double(p), location, scale, shape)
}

#' Moments of a GEV distribution
#'
#' @inheritParams gev
#' @return A named numeric vector: `mean`, `median`, `sd`, `skewness`,
#'   `kurtosis`, `minimum`, `maximum`. Entries are `NaN` where the moment is
#'   undefined for the given shape.
#' @export
#' @examples
#' gev_moments(location = 100, scale = 10, shape = 0)
gev_moments <- function(location, scale, shape) {
  bf_gev_moments_(location, scale, shape)
}

#' Fit a GEV distribution
#'
#' Estimate GEV parameters from a sample.
#'
#' @param x numeric vector of observations.
#' @param method estimation method: `"mle"` (maximum likelihood, the default),
#'   `"lmom"` (L-moments), or `"mom"` (product moments).
#' @return A named numeric vector with `location`, `scale`, and `shape`.
#' @export
#' @examples
#' set.seed(1)
#' gev_fit(rnorm(100, 100, 10), method = "lmom")
gev_fit <- function(x, method = c("mle", "lmom", "mom")) {
  method <- match.arg(method)
  bf_gev_fit_(as.double(x), method)
}

#' Standard error of a GEV quantile (maximum likelihood)
#'
#' Delta-method standard error of the `p`-quantile, using the expected
#' (Fisher) information matrix.
#'
#' @inheritParams gev
#' @param sample_size the sample size the fit was based on.
#' @return The standard error (a single numeric value).
#' @export
#' @examples
#' gev_quantile_se(0.99, location = 10849, scale = 5745.6, shape = 0.005,
#'                 sample_size = 62)
gev_quantile_se <- function(p, location, scale, shape, sample_size) {
  sqrt(bf_gev_quantile_variance_(p, location, scale, shape, as.integer(sample_size)))
}
