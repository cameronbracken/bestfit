# Public wrappers over the data-statistics utility glue in src/stats.cpp.

#' Multiple Grubbs-Beck low-outlier test
#'
#' The Multiple Grubbs-Beck test (MGBT) of Bulletin 17C, which identifies
#' potentially influential low floods (PILFs) in an annual-peak-flow record.
#'
#' @param x numeric vector of observations (annual peak flows).
#' @return The number of low outliers detected (an integer).
#' @export
#' @examples
#' d <- distribution("LogNormal", c(3, 0.5))
#' x <- c(dist_random(d, 40, seed = 7), 1, 2)
#' mgbt_test(x)
mgbt_test <- function(x) {
  ch_mgbt_test_(as.double(x))
}

#' Box-Cox transformation
#'
#' The Box-Cox power transformation and its inverse, plus maximum-likelihood
#' estimation of the transformation exponent `lambda`.
#'
#' @param x numeric vector of (positive) observations.
#' @param lambda the transformation exponent.
#' @return `box_cox_lambda()` returns the fitted exponent (searched over
#'   `[-5, 5]`). `box_cox()` and `box_cox_inverse()` return the transformed
#'   vector.
#' @name box_cox
#' @examples
#' d <- distribution("LogNormal", c(3, 0.5))
#' x <- dist_random(d, 100, seed = 11)
#' lambda <- box_cox_lambda(x)
#' y <- box_cox(x, lambda)
#' all.equal(box_cox_inverse(y, lambda), x)
NULL

#' @rdname box_cox
#' @export
box_cox_lambda <- function(x) {
  ch_box_cox_lambda_(as.double(x))
}

#' @rdname box_cox
#' @export
box_cox <- function(x, lambda) {
  ch_box_cox_(as.double(x), lambda)
}

#' @rdname box_cox
#' @export
box_cox_inverse <- function(x, lambda) {
  ch_box_cox_inverse_(as.double(x), lambda)
}

#' Yeo-Johnson transformation
#'
#' The Yeo-Johnson power transformation and its inverse, plus
#' maximum-likelihood estimation of the transformation exponent `lambda`.
#' Unlike [box_cox()], it accepts zero and negative values.
#'
#' @param x numeric vector of observations.
#' @param lambda the transformation exponent.
#' @return `yeo_johnson_lambda()` returns the fitted exponent.
#'   `yeo_johnson()` and `yeo_johnson_inverse()` return the transformed
#'   vector.
#' @name yeo_johnson
#' @examples
#' d <- distribution("Normal", c(0, 2))
#' x <- dist_random(d, 100, seed = 11)
#' lambda <- yeo_johnson_lambda(x)
#' y <- yeo_johnson(x, lambda)
#' all.equal(yeo_johnson_inverse(y, lambda), x)
NULL

#' @rdname yeo_johnson
#' @export
yeo_johnson_lambda <- function(x) {
  ch_yeo_johnson_lambda_(as.double(x))
}

#' @rdname yeo_johnson
#' @export
yeo_johnson <- function(x, lambda) {
  ch_yeo_johnson_(as.double(x), lambda)
}

#' @rdname yeo_johnson
#' @export
yeo_johnson_inverse <- function(x, lambda) {
  ch_yeo_johnson_inverse_(as.double(x), lambda)
}

#' Plotting positions
#'
#' Empirical plotting positions `(i - alpha) / (n + 1 - 2 * alpha)` for a
#' sample of size `n`, by named method or explicit `alpha`.
#'
#' @param n the sample size.
#' @param method one of `"weibull"` (`alpha = 0`, the default), `"median"`
#'   (`0.3175`), `"blom"` (`0.375`), `"cunnane"` (`0.4`), `"gringorten"`
#'   (`0.44`), or `"hazen"` (`0.5`). Ignored when `alpha` is given.
#' @param alpha optional explicit plotting-position parameter in `[0, 0.5]`.
#' @return A numeric vector of `n` non-exceedance probabilities for the
#'   ordered sample.
#' @export
#' @examples
#' plotting_positions(10)
#' plotting_positions(10, method = "cunnane")
plotting_positions <- function(
  n,
  method = c("weibull", "median", "blom", "cunnane", "gringorten", "hazen"),
  alpha = NULL
) {
  if (!is.null(alpha)) {
    return(ch_plotting_positions_alpha_(as.integer(n), as.double(alpha)))
  }
  method <- match.arg(method)
  ch_plotting_positions_(as.integer(n), method)
}

#' Latin hypercube sampling
#'
#' Stratified uniform `[0, 1]` samples via Latin hypercube sampling, using
#' the same seeded Mersenne Twister stream as the C# `LatinHypercube.Random`,
#' so a given `seed` reproduces the C# sample bit-for-bit (and matches
#' `corehydropy` exactly).
#'
#' @param n number of samples (rows).
#' @param dimension number of dimensions (columns).
#' @param seed integer seed for reproducible samples; `NULL` (the default)
#'   seeds from the clock.
#' @param median if `TRUE`, place each sample at the center of its bin
#'   (median LHS) instead of at a random position within it.
#' @return An `n` by `dimension` numeric matrix of probabilities in
#'   `(0, 1)`.
#' @export
#' @examples
#' latin_hypercube(5, 2, seed = 12345)
latin_hypercube <- function(n, dimension = 1, seed = NULL, median = FALSE) {
  seed <- if (is.null(seed)) -1L else as.integer(seed)
  flat <- ch_latin_hypercube_(as.integer(n), as.integer(dimension), seed, isTRUE(median))
  matrix(flat, nrow = n, ncol = dimension, byrow = TRUE)
}
