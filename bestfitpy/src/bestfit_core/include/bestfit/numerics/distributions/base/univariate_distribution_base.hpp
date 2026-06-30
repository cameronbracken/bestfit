// ported from: Numerics/Distributions/Univariate/Base/UnivariateDistributionBase.cs @ <pending-sha>
//
// Abstract base for every univariate distribution. Declares the distribution-core
// surface as pure virtuals (moments, support, PDF/CDF/InverseCDF, parameters) and
// promotes the shared concrete helpers (Variance, LogPDF, LogLikelihood).
//
// The desktop-app boilerplate of the C# base (XElement serialization, PDF/CDF graph
// builders, equality operators, numerical-integration CentralMoments / Conditional*)
// is intentionally not ported -- those are WPF/analysis concerns, not the math core.
#pragma once
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"

namespace bestfit::numerics::distributions {

class UnivariateDistributionBase {
   public:
    virtual ~UnivariateDistributionBase() = default;

    // --- Identity / parameters ---
    virtual UnivariateDistributionType type() const = 0;
    virtual int number_of_parameters() const = 0;
    virtual std::vector<double> get_parameters() const = 0;
    virtual void set_parameters(const std::vector<double>& parameters) = 0;
    bool parameters_valid() const { return parameters_valid_; }

    // --- Moments / support ---
    virtual double mean() const = 0;
    virtual double median() const = 0;
    virtual double mode() const = 0;
    virtual double standard_deviation() const = 0;
    virtual double skewness() const = 0;
    virtual double kurtosis() const = 0;
    virtual double minimum() const = 0;
    virtual double maximum() const = 0;
    double variance() const {
        double s = standard_deviation();
        return s * s;
    }

    // --- Distribution functions ---
    virtual double pdf(double x) const = 0;
    virtual double cdf(double x) const = 0;
    virtual double inverse_cdf(double probability) const = 0;

    virtual double log_pdf(double x) const {
        double f = pdf(x);
        if (std::isnan(f) || std::isinf(f) || f <= 0.0) return -kInf;
        return std::log(f);
    }

    double log_likelihood(const std::vector<double>& sample) const {
        double ll = 0.0;
        for (double v : sample) ll += log_pdf(v);
        if (std::isnan(ll) || std::isinf(ll)) return -kInf;
        return ll;
    }

    virtual std::unique_ptr<UnivariateDistributionBase> clone() const = 0;

   protected:
    static constexpr double kNearZero = 1E-4;  // assessing if a parameter is near zero
    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    static constexpr double kInf = std::numeric_limits<double>::infinity();

    bool parameters_valid_ = true;
};

}  // namespace bestfit::numerics::distributions
