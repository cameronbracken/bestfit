// ported from: Numerics/Data/Statistics/BoxCox.cs @ 2a0357a
//
// Box-Cox power transformation: Transform / InverseTransform (scalar + list), the
// normal-theory LogLikelihood / LogJacobian, and the BrentSearch-based FitLambda MLE.
// Mirrors the already-ported yeo_johnson.hpp structure (same namespace, same
// class-with-static-methods shape, same BrentSearch reuse).
//
// v2.1.4 hardening (upstream-sync Task 2, 2a0357a): FitLambda gained a private
// CanFitLambda pre-check (< 2 points / non-finite / non-positive / degenerate-constant
// sample -> NaN, no throw); the Brent objective now clamps a non-finite LogLikelihood to
// -double.MaxValue so the golden-section search stays arithmetically finite even where the
// profile likelihood is undefined; and the found candidate is rejected (NaN returned, not
// merely passed through) if it is non-finite, |lambda| > 5, or LogLikelihood(candidate) is
// itself non-finite. LogLikelihood was hardened the same way: it re-runs CanFitLambda up
// front and returns -Infinity on ANY non-finite intermediate (a transformed value, mu,
// sumX, sse, sigma, or the final ll) rather than only on a non-positive sigma.
// can_fit_lambda has no C++ "null sample" case (a std::vector reference can't be null);
// size() < 2 covers it. Deviation from C# preserved from the prior port: the ported
// BrentSearch (see brent_search.hpp) folds the Optimizer base in without a Status field,
// so this always evaluates brent.best_parameter() against the post-fit guards below rather
// than checking brent.Status first.
//
// Deviation from the plan/task text: the task list mentions a `Derivative` method, but
// BoxCox.cs @ 2a0357a still does NOT define one (unlike YeoJohnson.cs, which does). The C#
// source governs, so no `derivative` is ported here.
#pragma once
#include <cmath>
#include <limits>
#include <vector>

#include "corehydro/numerics/math/optimization/brent_search.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::data {

// Class for performing Box-Cox transformation. This method transforms non-normal dependent
// variables into a normal shape.
class BoxCox {
   public:
    // Fit the transformation parameter using maximum likelihood estimation.
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
    // can_fit_lambda), if lambda1 is non-finite or |lambda1| > 5, or if any intermediate
    // computed below is non-finite.
    static double log_likelihood(const std::vector<double>& values, double lambda1) {
        if (!can_fit_lambda(values) || !corehydro::numerics::is_finite(lambda1) ||
            std::fabs(lambda1) > 5.0)
            return -std::numeric_limits<double>::infinity();

        const int n = static_cast<int>(values.size());
        std::vector<double> y(static_cast<std::size_t>(n));
        double mu = 0.0;
        double sum_x = 0.0;
        for (int i = 0; i < n; ++i) {
            y[static_cast<std::size_t>(i)] =
                transform(values[static_cast<std::size_t>(i)], lambda1);
            if (!corehydro::numerics::is_finite(y[static_cast<std::size_t>(i)]))
                return -std::numeric_limits<double>::infinity();

            mu += y[static_cast<std::size_t>(i)];
            sum_x += std::log(values[static_cast<std::size_t>(i)]);
        }
        if (!corehydro::numerics::is_finite(mu) || !corehydro::numerics::is_finite(sum_x))
            return -std::numeric_limits<double>::infinity();

        mu = mu / n;
        double sse = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = y[static_cast<std::size_t>(i)] - mu;
            sse += d * d;
        }
        if (!corehydro::numerics::is_finite(sse) || sse <= 0.0)
            return -std::numeric_limits<double>::infinity();

        double sigma = std::sqrt(sse / n);
        if (!corehydro::numerics::is_finite(sigma) || sigma <= 0.0)
            return -std::numeric_limits<double>::infinity();

        double ll = -n / 2.0 * kLogSqrt2PI - n / 2.0 * std::log(sigma * sigma) -
                    1.0 / (2.0 * sigma * sigma) * sse + (lambda1 - 1.0) * sum_x;
        return corehydro::numerics::is_finite(ll) ? ll : -std::numeric_limits<double>::infinity();
    }

    // Computes the Log-Jacobian used to adjust the log-likelihood function.
    // Returns -inf on any non-positive value (Box-Cox undefined for non-positive values).
    static double log_jacobian(const std::vector<double>& values, double lambda) {
        double log_jacobian_sum = 0.0;
        const int n = static_cast<int>(values.size());
        for (int i = 0; i < n; ++i) {
            double xi = values[static_cast<std::size_t>(i)];
            if (xi <= 0) return -std::numeric_limits<double>::infinity();
            // For Box-Cox: log derivative is (lambda - 1) * log(x)
            log_jacobian_sum += (lambda - 1.0) * std::log(xi);
        }
        return log_jacobian_sum;
    }

    // Returns the Box-Cox transformation of the value.
    // lambda: the transformation exponent. Range -5 to +5.
    static double transform(double value, double lambda) {
        if (value <= 0) return std::numeric_limits<double>::quiet_NaN();
        if (std::fabs(lambda) > 5.0) return std::numeric_limits<double>::quiet_NaN();
        if (std::fabs(lambda) < 1e-8) return std::log(value);
        return (std::pow(value, lambda) - 1.0) / lambda;
    }

    // Returns the Box-Cox transformation of each value in the list.
    static std::vector<double> transform(const std::vector<double>& values, double lambda) {
        std::vector<double> new_values;
        new_values.reserve(values.size());
        for (std::size_t i = 0; i < values.size(); ++i)
            new_values.push_back(transform(values[i], lambda));
        return new_values;
    }

    // Returns the reverse of the Box-Cox transformed value.
    static double inverse_transform(double value, double lambda) {
        if (std::fabs(lambda) > 5.0) return std::numeric_limits<double>::quiet_NaN();
        if (std::fabs(lambda) < 1e-8) return std::exp(value);
        return std::pow(value * lambda + 1.0, 1.0 / lambda);
    }

    // Returns the inverse of each Box-Cox transformed value in the list.
    static std::vector<double> inverse_transform(const std::vector<double>& values,
                                                 double lambda) {
        std::vector<double> new_values;
        new_values.reserve(values.size());
        for (std::size_t i = 0; i < values.size(); ++i)
            new_values.push_back(inverse_transform(values[i], lambda));
        return new_values;
    }

   private:
    // Determines whether a sample can support Box-Cox lambda fitting: at least 2 finite,
    // strictly positive values with at least one differing from the first (non-degenerate).
    // Mirrors C# CanFitLambda; the C# `values == null` check has no C++ analog (values is a
    // vector reference here, never null), so size() < 2 covers both the null and
    // too-few-points C# cases.
    static bool can_fit_lambda(const std::vector<double>& values) {
        if (values.size() < 2) return false;

        double first = values[0];
        if (!corehydro::numerics::is_finite(first) || first <= 0.0) return false;

        bool has_different_value = false;
        for (std::size_t i = 1; i < values.size(); ++i) {
            double value = values[i];
            if (!corehydro::numerics::is_finite(value) || value <= 0.0) return false;
            if (value != first) has_different_value = true;
        }

        return has_different_value;
    }
};

}  // namespace corehydro::numerics::data
