// ported from: Numerics/Data/Statistics/BoxCox.cs @ a2c4dbf
//
// Box-Cox power transformation: Transform / InverseTransform (scalar + list), the
// normal-theory LogLikelihood / LogJacobian, and the BrentSearch-based FitLambda MLE.
// Mirrors the already-ported yeo_johnson.hpp structure (same namespace, same
// class-with-static-methods shape, same BrentSearch reuse). The C# `FitLambda(IList, out
// lambda)` overload is folded into a returning form (C++ has no out-parameters); its Status
// check maps to the ported BrentSearch convention of returning best-so-far (see the
// deviation note on fit_lambda below).
//
// Deviation from the plan/task text: the task list mentions a `Derivative` method, but
// BoxCox.cs @ a2c4dbf does NOT define one (unlike YeoJohnson.cs, which does). The C# source
// governs, so no `derivative` is ported here.
#pragma once
#include <cmath>
#include <limits>
#include <vector>

#include "bestfit/numerics/math/optimization/brent_search.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::data {

// Class for performing Box-Cox transformation. This method transforms non-normal dependent
// variables into a normal shape.
class BoxCox {
   public:
    // Fit the transformation parameter using maximum likelihood estimation.
    // Returns the fitted transformation exponent. Range -5 to +5.
    //
    // Deviation from C#: the C# out-parameter overload sets lambda = NaN when
    // brent.Status != Success. The ported BrentSearch (see brent_search.hpp) folds the
    // Optimizer base in without a Status field and returns the best parameter seen even
    // when the iteration cap is hit, so this port always returns brent.best_parameter()
    // (matching yeo_johnson.hpp::fit_lambda).
    static double fit_lambda(const std::vector<double>& values) {
        // Solve with Brent
        math::optimization::BrentSearch brent(
            [&values](double x) { return log_likelihood(values, x); }, -5.0, 5.0);
        brent.maximize();
        return brent.best_parameter();
    }

    // The log-likelihood function. The transformed observations are assumed to come from a
    // normal distribution. The change of variable formula is used to write the
    // log-likelihood function.
    static double log_likelihood(const std::vector<double>& values, double lambda1) {
        const int n = static_cast<int>(values.size());
        std::vector<double> y(static_cast<std::size_t>(n));
        double mu = 0.0;
        double sum_x = 0.0;
        for (int i = 0; i < n; ++i) {
            y[static_cast<std::size_t>(i)] = transform(values[static_cast<std::size_t>(i)], lambda1);
            mu += y[static_cast<std::size_t>(i)];
            sum_x += std::log(values[static_cast<std::size_t>(i)]);
        }
        mu = mu / n;
        double sse = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = y[static_cast<std::size_t>(i)] - mu;
            sse += d * d;
        }
        double sigma = std::sqrt(sse / n);
        double ll = -n / 2.0 * kLogSqrt2PI - n / 2.0 * std::log(sigma * sigma) -
                    1.0 / (2.0 * sigma * sigma) * sse + (lambda1 - 1.0) * sum_x;
        return ll;
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
};

}  // namespace bestfit::numerics::data
