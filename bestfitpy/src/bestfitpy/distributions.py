"""Public univariate distribution interface.

A thin, stateless wrapper over the factory-dispatched ``_core.dist_*`` glue: every
method re-dispatches with ``(family, params)``, so no C++ object lifetime leaks into
Python. Composite families (Truncated/Mixture/CompetingRisks/Empirical/KernelDensity)
are not constructible here yet; they remain internal fixture glue.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["Distribution", "distribution_names"]


def distribution_names() -> list[str]:
    """List the supported distribution families.

    Returns
    -------
    list of str
        The family names accepted by :class:`Distribution` and
        :meth:`Distribution.fit`, matching the C# class names of the
        USACE-RMC Numerics library (for example ``"Normal"``,
        ``"LogNormal"``, ``"Gumbel"``, ``"GeneralizedExtremeValue"``).
    """
    return list(_core.dist_names())


class Distribution:
    """A univariate distribution from the ported Numerics library.

    Parameters are positional, in the same order as the C# constructor for
    the family (for example ``Normal`` takes ``[mean, sd]`` and
    ``GeneralizedExtremeValue`` takes ``[location, scale, shape]``). Use
    :attr:`parameter_names` to see the names for a family.

    All numeric methods evaluate in the shared C++ core, so results are
    identical to the R package and to the upstream C# library.

    Parameters
    ----------
    family : str
        Distribution family name; see :func:`distribution_names`.
    params : array-like of float
        Parameter vector, in constructor order.

    Examples
    --------
    >>> d = Distribution("Normal", [100, 15])
    >>> d.cdf(100)
    0.5
    >>> d.random(3, seed=123)  # doctest: +SKIP
    array([107.71408450, 108.43058699,  91.52951859])
    """

    def __init__(self, family: str, params) -> None:
        if not isinstance(family, str):
            raise TypeError("family must be a distribution name; see distribution_names()")
        if family not in _core.dist_names():
            raise ValueError(
                f"unknown distribution family '{family}'; "
                "see distribution_names() for the supported names"
            )
        params = [float(v) for v in np.asarray(params, dtype=float).ravel()]
        expected = _core.dist_parameter_names(family)["short"]
        if expected and len(params) != len(expected):
            raise ValueError(
                f"'{family}' expects {len(expected)} parameters "
                f"({', '.join(expected)}), got {len(params)}"
            )
        self._family = family
        self._params = params

    # -- properties -------------------------------------------------------------------

    @property
    def family(self) -> str:
        """str: The distribution family name."""
        return self._family

    @property
    def params(self) -> list[float]:
        """list of float: The parameter vector, in constructor order."""
        return list(self._params)

    @property
    def parameter_names(self) -> dict[str, list[str]]:
        """dict: Parameter names, keys ``"full"`` and ``"short"``."""
        return _core.dist_parameter_names(self._family)

    @property
    def is_valid(self) -> bool:
        """bool: Whether the parameters are valid for this family."""
        return _core.dist_valid(self._family, self._params)

    # -- distribution functions ---------------------------------------------------------

    def pdf(self, x):
        """Probability density at ``x``.

        Parameters
        ----------
        x : float or array-like
            Quantiles.

        Returns
        -------
        float or numpy.ndarray
            Density values, scalar in, scalar out.
        """
        return self._vectorized(_core.dist_pdf_v, x)

    def log_pdf(self, x):
        """Natural log of the probability density at ``x``."""
        return self._vectorized(_core.dist_log_pdf_v, x)

    def cdf(self, x):
        """Cumulative distribution at ``x``."""
        return self._vectorized(_core.dist_cdf_v, x)

    def quantile(self, p):
        """Quantile (inverse CDF) at probability ``p`` in ``(0, 1)``."""
        return self._vectorized(_core.dist_quantile_v, p)

    def random(self, n: int, seed: int | None = None):
        """Draw ``n`` random values.

        Draws come from the same seeded Mersenne Twister stream as the C#
        ``GenerateRandomValues(sampleSize, seed)``: a given ``seed``
        reproduces the C# draws bit-for-bit (and matches ``bestfitr``
        exactly).

        Parameters
        ----------
        n : int
            Number of draws.
        seed : int, optional
            Seed for reproducible draws; ``None`` (the default) seeds from
            the clock.

        Returns
        -------
        numpy.ndarray
            The ``n`` draws.
        """
        s = -1 if seed is None else int(seed)
        return np.asarray(_core.dist_random(self._family, self._params, int(n), s))

    # -- properties of the distribution --------------------------------------------------

    def moments(self) -> dict[str, float]:
        """Moments and support of the distribution.

        Returns
        -------
        dict
            Keys ``mean``, ``median``, ``mode``, ``sd``, ``skewness``,
            ``kurtosis``, ``minimum``, ``maximum``; values are ``nan``
            where undefined.
        """
        m = _core.dist_moments(self._family, self._params)
        order = ["mean", "median", "mode", "sd", "skewness", "kurtosis", "minimum", "maximum"]
        return {k: m[k] for k in order}

    def linear_moments(self) -> list[float]:
        """The first four L-moments.

        Raises
        ------
        ValueError
            If the family has no L-moment support.
        """
        return list(_core.dist_linear_moments(self._family, self._params))

    def log_likelihood(self, data) -> float:
        """Log-likelihood of ``data`` under the distribution.

        Parameters
        ----------
        data : array-like of float
            Observations.
        """
        return _core.dist_log_likelihood(self._family, self._params, _as_list(data))

    # -- estimation -----------------------------------------------------------------------

    @classmethod
    def fit(cls, family: str, data, method: str = "mle") -> "Distribution":
        """Fit a distribution family to data.

        Mirrors the C# ``Estimate(data, ParameterEstimationMethod)`` API of
        the Numerics library.

        Parameters
        ----------
        family : str
            Distribution family name; see :func:`distribution_names`.
        data : array-like of float
            Observations.
        method : {"mle", "lmom", "mom"}
            Estimation method: maximum likelihood (default), L-moments, or
            product moments. Not every family supports every method;
            unsupported combinations raise.

        Returns
        -------
        Distribution
            The fitted distribution.
        """
        params = _core.dist_fit(family, _as_list(data), method)
        return cls(family, params)

    # -- misc -------------------------------------------------------------------------------

    def __repr__(self) -> str:
        names = self.parameter_names["short"]
        if len(names) == len(self._params):
            inner = ", ".join(f"{n} = {v:g}" for n, v in zip(names, self._params))
        else:
            inner = ", ".join(f"{v:g}" for v in self._params)
        return f"Distribution({self._family}({inner}))"

    def _vectorized(self, fn, values):
        arr = np.asarray(values, dtype=float)
        out = np.asarray(fn(self._family, self._params, [float(v) for v in arr.ravel()]))
        if arr.ndim == 0:
            return float(out[0])
        return out.reshape(arr.shape)


def _as_list(data) -> list[float]:
    return [float(v) for v in np.asarray(data, dtype=float).ravel()]
