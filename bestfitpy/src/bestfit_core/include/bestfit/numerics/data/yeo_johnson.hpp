// ported from: Numerics/Data/Statistics/YeoJohnson.cs @ a2c4dbf
//
// Yeo-Johnson power transformation: Transform / Derivative / InverseTransform plus the
// normal-theory LogLikelihood / LogJacobian and the BrentSearch-based FitLambda MLE.
// The C# `FitLambda(IList, out lambda)` overload is folded into the returning form (C++
// has no out-parameters); its Status check maps to the ported BrentSearch convention of
// returning best-so-far (see the deviation note on fit_lambda below).
#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/math/optimization/brent_search.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::data {

// Class for performing Yeo-Johnson transformation. This method transforms non-normal
// dependent variables into a normal shape.
class YeoJohnson {
   public:
    // Fit the transform parameter using maximum likelihood estimation.
    // Returns the fitted transformation exponent. Range -5 to +5.
    //
    // Deviation from C#: the C# out-parameter overload sets lambda = NaN when
    // brent.Status != Success. The ported BrentSearch (see brent_search.hpp) folds the
    // Optimizer base in without a Status field and returns the best parameter seen even
    // when the iteration cap is hit (the nelder_mead.hpp precedent), so this port always
    // returns brent.best_parameter().
    static double fit_lambda(const std::vector<double>& values) {
        if (values.size() < 2)
            throw std::invalid_argument("At least 2 values are required to fit lambda.");

        // Solve with Brent
        math::optimization::BrentSearch brent(
            [&values](double x) { return log_likelihood(values, x); }, -5.0, 5.0);
        brent.maximize();
        return brent.best_parameter();
    }

    // The log-likelihood function. The transformed observations are assumed to come from a
    // normal distribution. The change of variable formula is used to write the
    // log-likelihood function.
    static double log_likelihood(const std::vector<double>& values, double lambda) {
        const int n = static_cast<int>(values.size());
        std::vector<double> transformed(static_cast<std::size_t>(n));
        double sum = 0.0;
        double log_jacobian_sum = 0.0;

        // Transform values and compute sum and log-Jacobian
        for (int i = 0; i < n; ++i) {
            double xi = values[static_cast<std::size_t>(i)];
            double yi = transform(xi, lambda);
            transformed[static_cast<std::size_t>(i)] = yi;
            sum += yi;

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
        }

        // Compute mean and SSE
        double mu = sum / n;
        double sse = 0.0;
        for (int i = 0; i < n; ++i) {
            double resid = transformed[static_cast<std::size_t>(i)] - mu;
            sse += resid * resid;
        }

        double sigma_sq = sse / n;
        if (sigma_sq <= 0 || std::isnan(sigma_sq))
            return -std::numeric_limits<double>::infinity();

        return -n / 2.0 * kLogSqrt2PI - n / 2.0 * std::log(sigma_sq) - sse / (2.0 * sigma_sq) +
               log_jacobian_sum;
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
        } else if (value < 0 && lambda != 2) {
            return -(std::pow(-value + 1, 2 - lambda) - 1) / (2 - lambda);
        } else if (value < 0 && lambda == 2) {
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
        } else if (value < 0 && lambda != 2) {
            return 1 - std::pow(1 - (2 - lambda) * value, 1 / (2 - lambda));
        } else if (value < 0 && lambda == 2) {
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
};

}  // namespace bestfit::numerics::data
