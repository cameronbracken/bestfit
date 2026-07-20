// ported from: Numerics/Distributions/Univariate/Logistic.cs @ 2a0357a
//
// The Logistic distribution with location ξ and scale α. Logic mirrors the C# source
// method-for-method. IStandardError, IBootstrappable, and the WPF helpers are not ported
// (desktop / uncertainty-analysis concerns). ILinearMomentEstimation is not implemented
// in the C# source and is therefore absent here.
// Re-audited against v2.1.4's "Harden distribution parameter validation" wave: C#'s
// SetParameters now assigns Xi/Alpha directly before computing _parametersValid once from
// the final (location, scale) pair (the old code let the Xi/Alpha property setters
// revalidate against a stale sibling field). This port already assigned xi_/alpha_
// directly and validated once at the end -- pre-aligned, no code change.
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

class Logistic : public UnivariateDistributionBase,
                 public IEstimation,
                 public IMaximumLikelihoodEstimation {
   public:
    Logistic() { set_parameters(0.0, 0.1); }
    Logistic(double location, double scale) { set_parameters(location, scale); }

    double xi() const { return xi_; }
    double alpha() const { return alpha_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Logistic;
    }
    int number_of_parameters() const override { return 2; }
    std::vector<double> get_parameters() const override { return {xi_, alpha_}; }

    void set_parameters(double location, double scale) {
        xi_ = location;
        alpha_ = scale;
        parameters_valid_ = validate(location, scale);
    }
    void set_parameters(const std::vector<double>& p) override { set_parameters(p[0], p[1]); }

    // --- Moments / support ---
    double mean() const override { return xi_; }
    double median() const override { return inverse_cdf(0.5); }
    double mode() const override { return xi_; }
    double standard_deviation() const override { return alpha_ * kPi / std::sqrt(3.0); }
    double skewness() const override { return 0.0; }
    double kurtosis() const override { return 3.0 + 6.0 / 5.0; }
    double minimum() const override { return -kInf; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---
    double pdf(double x) const override {
        double z = (x - xi_) / alpha_;
        double ez = std::exp(-z);
        return 1.0 / alpha_ * ez * std::pow(1.0 + ez, -2.0);
    }

    double cdf(double x) const override {
        double z = (x - xi_) / alpha_;
        return 1.0 / (1.0 + std::exp(-z));
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        return xi_ + alpha_ * std::log(probability / (1.0 - probability));
    }

    // --- Parameter display names (X1; C# Logistic.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Location (\xCE\xBE)", "Scale (\xCE\xB1)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xCE\xBE", "\xCE\xB1"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Logistic>(xi_, alpha_);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            set_parameters(parameters_from_moments(data::product_moments(sample)));
        } else if (method == ParameterEstimationMethod::MaximumLikelihood) {
            set_parameters(mle(sample));
        } else {
            throw std::invalid_argument("estimation method not implemented for Logistic");
        }
    }

    std::vector<double> parameters_from_moments(const std::vector<double>& moments) const {
        double xi = moments[0];
        double alpha = moments[1] * std::sqrt(3.0) / kPi;
        return {xi, alpha};
    }

    void get_parameter_constraints(const std::vector<double>& sample, std::vector<double>& initials,
                                   std::vector<double>& lowers,
                                   std::vector<double>& uppers) const override {
        initials = parameters_from_moments(data::product_moments(sample));
        lowers.resize(2);
        uppers.resize(2);
        // bounds for location
        double xi0 = initials[0] != 0.0 ? initials[0] : kDoubleMachineEpsilon;
        lowers[0] = -std::pow(10.0, std::ceil(std::log10(std::fabs(xi0)) + 1.0));
        uppers[0] =  std::pow(10.0, std::ceil(std::log10(std::fabs(xi0)) + 1.0));
        // bounds for scale
        lowers[1] = kDoubleMachineEpsilon;
        uppers[1] = std::pow(10.0, std::ceil(std::log10(initials[1]) + 1.0));
    }

    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            Logistic lo;
            lo.set_parameters(x[0], x[1]);
            return lo.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 2, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }

   private:
    static bool validate(double location, double scale) {
        if (std::isnan(location) || std::isinf(location)) return false;
        if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0) return false;
        return true;
    }

    double xi_ = 0.0;
    double alpha_ = 0.0;
};

}  // namespace corehydro::numerics::distributions
