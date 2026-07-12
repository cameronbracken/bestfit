"""Data-statistics utilities.

Public wrappers over the ``_core`` glue for the Multiple Grubbs-Beck test,
Box-Cox and Yeo-Johnson transforms, plotting positions, and Latin hypercube
sampling.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = [
    "mgbt_test",
    "box_cox_lambda",
    "box_cox",
    "box_cox_inverse",
    "yeo_johnson_lambda",
    "yeo_johnson",
    "yeo_johnson_inverse",
    "plotting_positions",
    "latin_hypercube",
]

_PLOTTING_METHODS = ("weibull", "median", "blom", "cunnane", "gringorten", "hazen")


def _as_list(data) -> list[float]:
    return [float(v) for v in np.asarray(data, dtype=float).ravel()]


def mgbt_test(x) -> int:
    """Multiple Grubbs-Beck low-outlier test.

    The MGBT of Bulletin 17C, which identifies potentially influential low
    floods (PILFs) in an annual-peak-flow record.

    Parameters
    ----------
    x : array-like of float
        Observations (annual peak flows).

    Returns
    -------
    int
        The number of low outliers detected.
    """
    return int(_core.mgbt_test(_as_list(x)))


def box_cox_lambda(x) -> float:
    """Fit the Box-Cox transformation exponent by maximum likelihood.

    Parameters
    ----------
    x : array-like of float
        Positive observations.

    Returns
    -------
    float
        The fitted exponent, searched over ``[-5, 5]``.
    """
    return _core.box_cox_lambda(_as_list(x))


def box_cox(x, lambda_: float):
    """Box-Cox power transformation of ``x`` with exponent ``lambda_``.

    Returns
    -------
    numpy.ndarray
        The transformed values.
    """
    return np.asarray(_core.box_cox(_as_list(x), float(lambda_)))


def box_cox_inverse(x, lambda_: float):
    """Inverse Box-Cox transformation of ``x`` with exponent ``lambda_``."""
    return np.asarray(_core.box_cox_inverse(_as_list(x), float(lambda_)))


def yeo_johnson_lambda(x) -> float:
    """Fit the Yeo-Johnson transformation exponent by maximum likelihood.

    Unlike :func:`box_cox_lambda`, accepts zero and negative values.
    """
    return _core.yeo_johnson_lambda(_as_list(x))


def yeo_johnson(x, lambda_: float):
    """Yeo-Johnson power transformation of ``x`` with exponent ``lambda_``.

    Returns
    -------
    numpy.ndarray
        The transformed values.
    """
    return np.asarray(_core.yeo_johnson(_as_list(x), float(lambda_)))


def yeo_johnson_inverse(x, lambda_: float):
    """Inverse Yeo-Johnson transformation of ``x`` with exponent ``lambda_``."""
    return np.asarray(_core.yeo_johnson_inverse(_as_list(x), float(lambda_)))


def plotting_positions(n: int, method: str = "weibull", alpha: float | None = None):
    """Empirical plotting positions for a sample of size ``n``.

    Computes ``(i - alpha) / (n + 1 - 2 * alpha)`` for ``i = 1..n``.

    Parameters
    ----------
    n : int
        Sample size.
    method : {"weibull", "median", "blom", "cunnane", "gringorten", "hazen"}
        Named plotting-position formula (``alpha`` of 0, 0.3175, 0.375, 0.4,
        0.44, 0.5 respectively). Ignored when ``alpha`` is given.
    alpha : float, optional
        Explicit plotting-position parameter in ``[0, 0.5]``.

    Returns
    -------
    numpy.ndarray
        ``n`` non-exceedance probabilities for the ordered sample.
    """
    if alpha is not None:
        return np.asarray(_core.plotting_positions_alpha(int(n), float(alpha)))
    if method not in _PLOTTING_METHODS:
        raise ValueError(f"unknown plotting-position method '{method}'; use one of {_PLOTTING_METHODS}")
    return np.asarray(_core.plotting_positions(int(n), method))


def latin_hypercube(n: int, dimension: int = 1, seed: int | None = None, median: bool = False):
    """Latin hypercube sample of uniform ``[0, 1]`` probabilities.

    Uses the same seeded Mersenne Twister stream as the C#
    ``LatinHypercube.Random``, so a given ``seed`` reproduces the C# sample
    bit-for-bit (and matches ``bestfitr`` exactly).

    Parameters
    ----------
    n : int
        Number of samples (rows).
    dimension : int
        Number of dimensions (columns).
    seed : int, optional
        Seed for reproducible samples; ``None`` (the default) seeds from the
        clock.
    median : bool
        If True, place each sample at the center of its bin (median LHS)
        instead of at a random position within it.

    Returns
    -------
    numpy.ndarray
        An ``(n, dimension)`` array of probabilities in ``(0, 1)``.
    """
    s = -1 if seed is None else int(seed)
    fn = _core.latin_hypercube_median if median else _core.latin_hypercube
    return np.asarray(fn(int(n), int(dimension), s))
