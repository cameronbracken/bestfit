// ported from: Numerics/Data/Statistics/YeoJohnson.cs @ 2a0357a
//
// Yeo-Johnson power transformation: Transform / Derivative / InverseTransform plus the
// normal-theory LogLikelihood / LogJacobian and the BrentSearch-based FitLambda MLE.
//
// v2.1.4 hardening (upstream-sync Task 2, 2a0357a): FitLambda gained a private
// CanFitLambda pre-check (< 2 points / non-finite / degenerate-constant sample -> NaN, no
// throw -- the prior C++ port threw std::invalid_argument for < 2 points; that throw is
// REMOVED here to match); the Brent objective now clamps a non-finite LogLikelihood to
// -double.MaxValue so the golden-section search stays arithmetically finite even where the
// profile likelihood is undefined; and the found candidate is rejected (NaN returned, not
// merely passed through) if it is non-finite, |lambda| > 5, or LogLikelihood(candidate) is
// itself non-finite. LogLikelihood was hardened the same way: it re-runs CanFitLambda up
// front and returns -Infinity on ANY non-finite intermediate (a transformed value, the
// running sum, the log-Jacobian sum, sse, sigmaSq, or the final ll). Unlike BoxCox,
// CanFitLambda here has no positivity requirement (Yeo-Johnson accepts negative values).
// can_fit_lambda has no C++ "null sample" case (a std::vector reference can't be null);
// size() < 2 covers it.
//
// Also new in v2.1.4: Transform/InverseTransform's negative-value lambda==2 branch is now
// a tolerance band, `Math.Abs(lambda - 2.0) < 1e-8` (was exact `lambda == 2`), so a lambda
// within 1e-8 of 2 takes the log branch instead of the (numerically unstable near lambda=2)
// power-law branch.
#pragma once
#include <cmath>
#include <limits>
#include <vector>

#include "corehydro/numerics/math/optimization/brent_search.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::data {

// Class for performing Yeo-Johnson transformation. This method transforms non-normal
// dependent variables into a normal shape.
class YeoJohnson {
   public:
    // Fit the transform parameter using maximum likelihood estimation.
    // Returns the fitted transformation exponent in [-5, 5], or NaN if the sample cannot
    // support fitting (see can_fit_lambda) or the fitted candidate is not usable.
    static double fit_lambda(const std::vector<double>& values) {
        if (!can_fit_lambda(values)) return std::numeric_limits<double>::quiet_NaN();

        // Keep BrentSearch arithmetic finite even when the profile likelihood is undefined.
        math::optimization::BrentSearch brent(
            [&values](double x) {
                double ll = log_likelihood(values, x);
                return corehydro::numerics::is_finite(ll) ? ll
                                                           : -std::numeric_limits<double>::max();
            },
            -5.0, 5.0);
        brent.maximize();

        double candidate = brent.best_parameter();
        if (!corehydro::numerics::is_finite(candidate) || std::fabs(candidate) > 5.0 ||
            !corehydro::numerics::is_finite(log_likelihood(values, candidate)))
            return std::numeric_limits<double>::quiet_NaN();

        return candidate;
    }

    // The log-likelihood function. The transformed observations are assumed to come from a
    // normal distribution. The change of variable formula is used to write the
    // log-likelihood function. Returns -Infinity if the sample cannot support fitting (see
    // can_fit_lambda), if lambda is non-finite or |lambda| > 5, or if any intermediate
    // computed below is non-finite.
    static double log_likelihood(const std::vector<double>& values, double lambda) {
        if (!can_fit_lambda(values) || !corehydro::numerics::is_finite(lambda) ||
            std::fabs(lambda) > 5.0)
            return -std::numeric_limits<double>::infinity();

        const int n = static_cast<int>(values.size());
        std::vector<double> transformed(static_cast<std::size_t>(n));
        double sum = 0.0;
        double log_jacobian_sum = 0.0;

        // Transform values and compute sum and log-Jacobian
        for (int i = 0; i < n; ++i) {
            double xi = values[static_cast<std::size_t>(i)];
            double yi = transform(xi, lambda);
            if (!corehydro::numerics::is_finite(yi))
                return -std::numeric_limits<double>::infinity();

            transformed[static_cast<std::size_t>(i)] = yi;
            sum += yi;
            if (!corehydro::numerics::is_finite(sum))
                return -std::numeric_limits<double>::infinity();

            // Compute derivative dT/dy for log-Jacobian
            double dTdy;
            if (xi >= 0) {
                dTdy = std::pow(xi + 1, lambda - 1);
            } else {
                dTdy = std::pow(-xi + 1, 1 - lambda);
            }

            // Avoid log of zero or negative values
            if (dTdy > 0)
                log_jacobian_sum += std::log(dTdy);
            else
                return -std::numeric_limits<double>::infinity();  // log-likelihood undefined
            if (!corehydro::numerics::is_finite(log_jacobian_sum))
                return -std::numeric_limits<double>::infinity();
        }

        // Compute mean and SSE
        double mu = sum / n;
        double sse = 0.0;
        for (int i = 0; i < n; ++i) {
            double resid = transformed[static_cast<std::size_t>(i)] - mu;
            sse += resid * resid;
        }
        if (!corehydro::numerics::is_finite(sse) || sse <= 0.0)
            return -std::numeric_limits<double>::infinity();

        double sigma_sq = sse / n;
        if (!corehydro::numerics::is_finite(sigma_sq) || sigma_sq <= 0.0)
            return -std::numeric_limits<double>::infinity();

        double ll = -n / 2.0 * kLogSqrt2PI - n / 2.0 * std::log(sigma_sq) - sse / (2.0 * sigma_sq) +
                    log_jacobian_sum;
        return corehydro::numerics::is_finite(ll) ? ll : -std::numeric_limits<double>::infinity();
    }

    // Computes the Log-Jacobian used to adjust the log-likelihood function.
    static double log_jacobian(const std::vector<double>& values, double lambda) {
        double log_jacobian_sum = 0.0;
        const int n = static_cast<int>(values.size());

        // Transform values and compute sum and log-Jacobian
        for (int i = 0; i < n; ++i) {
            double xi = values[static_cast<std::size_t>(i)];

            // Compute derivative dT/dy for log-Jacobian
            double dTdy;
            if (xi >= 0) {
                dTdy = std::pow(xi + 1, lambda - 1);
            } else {
                dTdy = std::pow(-xi + 1, 1 - lambda);
            }

            // Avoid log of zero or negative values
            log_jacobian_sum += std::log(std::fabs(dTdy));
        }
        return log_jacobian_sum;
    }

    // Returns the Yeo-Johnson transformation of the value.
    // lambda: the transformation exponent. Range -5 to +5.
    static double transform(double value, double lambda) {
        if (std::fabs(lambda) > 5.0) return std::numeric_limits<double>::quiet_NaN();
        if (value >= 0 && std::fabs(lambda) >= 1E-8) {
            return (std::pow(value + 1, lambda) - 1) / lambda;
        } else if (value >= 0 && std::fabs(lambda) < 1E-8) {
            return std::log(value + 1);
        } else if (value < 0 && std::fabs(lambda - 2.0) >= 1E-8) {
            return -(std::pow(-value + 1, 2 - lambda) - 1) / (2 - lambda);
        } else if (value < 0 && std::fabs(lambda - 2.0) < 1E-8) {
            return -std::log(-value + 1);
        } else {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    // Returns the derivative of the Yeo-Johnson transformation with respect to the
    // original value.
    static double derivative(double value, double lambda) {
        if (std::fabs(lambda) > 5.0) return std::numeric_limits<double>::quiet_NaN();
        return value >= 0.0 ? std::pow(value + 1.0, lambda - 1.0)
                            : std::pow(-value + 1.0, 1.0 - lambda);
    }

    // Returns the Yeo-Johnson transformation of each value in the list.
    static std::vector<double> transform(const std::vector<double>& values, double lambda) {
        std::vector<double> new_values;
        new_values.reserve(values.size());
        for (std::size_t i = 0; i < values.size(); ++i)
            new_values.push_back(transform(values[i], lambda));
        return new_values;
    }

    // Returns the inverse of the Yeo-Johnson transformed value.
    static double inverse_transform(double value, double lambda) {
        if (std::fabs(lambda) > 5.0) return std::numeric_limits<double>::quiet_NaN();
        if (value >= 0 && std::fabs(lambda) >= 1E-8) {
            return std::pow(lambda * value + 1, 1 / lambda) - 1;
        } else if (value >= 0 && std::fabs(lambda) < 1E-8) {
            return std::exp(value) - 1;
        } else if (value < 0 && std::fabs(lambda - 2.0) >= 1E-8) {
            return 1 - std::pow(1 - (2 - lambda) * value, 1 / (2 - lambda));
        } else if (value < 0 && std::fabs(lambda - 2.0) < 1E-8) {
            return 1 - std::exp(-value);
        } else {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    // Returns the inverse of each Yeo-Johnson transformed value in the list.
    static std::vector<double> inverse_transform(const std::vector<double>& values,
                                                 double lambda) {
        std::vector<double> new_values;
        new_values.reserve(values.size());
        for (std::size_t i = 0; i < values.size(); ++i)
            new_values.push_back(inverse_transform(values[i], lambda));
        return new_values;
    }

   private:
    // Determines whether a sample can support Yeo-Johnson lambda fitting: at least 2
    // finite values with at least one differing from the first (non-degenerate). Unlike
    // BoxCox::can_fit_lambda, there is no positivity requirement. Mirrors C# CanFitLambda;
    // the C# `values == null` check has no C++ analog (values is a vector reference here,
    // never null), so size() < 2 covers both the null and too-few-points C# cases.
    static bool can_fit_lambda(const std::vector<double>& values) {
        if (values.size() < 2) return false;

        double first = values[0];
        if (!corehydro::numerics::is_finite(first)) return false;

        bool has_different_value = false;
        for (std::size_t i = 1; i < values.size(); ++i) {
            double value = values[i];
            if (!corehydro::numerics::is_finite(value)) return false;
            if (value != first) has_different_value = true;
        }

        return has_different_value;
    }
};

}  // namespace corehydro::numerics::data
