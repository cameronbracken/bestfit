// ported from: Numerics/Distributions/Univariate/Exponential.cs @ <pending-sha>
//
// The (two-parameter) Exponential distribution, location ξ and scale α. Logic mirrors
// the C# source method-for-method. The WPF ConditionalMoments helper is not ported.
#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/data/statistics.hpp"
#include "bestfit/numerics/distributions/base/i_estimation.hpp"
#include "bestfit/numerics/distributions/base/i_linear_moment_estimation.hpp"
#include "bestfit/numerics/distributions/base/parameter_estimation_method.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/math/optimization/nelder_mead.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::distributions {

class Exponential : public UnivariateDistributionBase,
                    public IEstimation,
                    public ILinearMomentEstimation {
   public:
    Exponential() { set_parameters(100.0, 10.0); }
    Exponential(double location, double scale) { set_parameters(location, scale); }

    double xi() const { return xi_; }
    double alpha() const { return alpha_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Exponential;
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
    double mean() const override { return xi_ + alpha_; }
    double median() const override { return xi_ - std::log(0.5) * alpha_; }
    double mode() const override { return xi_; }
    double standard_deviation() const override { return alpha_; }
    double skewness() const override { return 2.0; }
    double kurtosis() const override { return 9.0; }
    double minimum() const override { return xi_; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---
    double pdf(double x) const override {
        if (x < minimum() || x > maximum()) return 0.0;
        return 1.0 / alpha_ * std::exp(-((x - xi_) / alpha_));
    }

    double cdf(double x) const override {
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return 1.0;
        return 1.0 - std::exp(-((x - xi_) / alpha_));
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        return xi_ - alpha_ * std::log(1.0 - probability);
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Exponential>(xi_, alpha_);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            set_parameters(parameters_from_moments(data::product_moments(sample)));
        } else if (method == ParameterEstimationMethod::MethodOfLinearMoments) {
            set_parameters(parameters_from_linear_moments(data::linear_moments(sample)));
        } else {
            set_parameters(mle(sample));
        }
    }

    std::vector<double> parameters_from_moments(const std::vector<double>& moments) const {
        return {moments[0] - moments[1], moments[1]};
    }

    std::vector<double> parameters_from_linear_moments(
        const std::vector<double>& moments) const override {
        double L1 = moments[0], L2 = moments[1];
        double alpha = 2.0 * L2;
        double xi = L1 - alpha;
        return {xi, alpha};
    }

    std::vector<double> linear_moments_from_parameters(
        const std::vector<double>& parameters) const override {
        double xi = parameters[0], alpha = parameters[1];
        return {xi + alpha, 0.5 * alpha, 1.0 / 3.0, 1.0 / 6.0};
    }

    void get_parameter_constraints(const std::vector<double>& sample, std::vector<double>& initials,
                                   std::vector<double>& lowers,
                                   std::vector<double>& uppers) const {
        auto moments = data::product_moments(sample);
        double N = static_cast<double>(sample.size());
        double min_data = *std::min_element(sample.begin(), sample.end());
        initials = {(N * min_data - moments[0]) / (N - 1.0), N * (moments[0] - min_data) / (N - 1.0)};
        lowers.assign(2, 0.0);
        uppers.assign(2, 0.0);
        if (initials[0] == 0.0) initials[0] = kDoubleMachineEpsilon;
        lowers[0] = initials[0] - std::pow(10.0, std::ceil(std::log10(std::fabs(initials[0]))));
        uppers[0] = std::pow(10.0, std::ceil(std::log10(initials[0]) + 1.0));
        lowers[1] = kDoubleMachineEpsilon;
        uppers[1] = std::pow(10.0, std::ceil(std::log10(initials[1]) + 1.0));
        if (initials[0] <= lowers[0] || initials[0] >= uppers[0])
            initials[0] = 0.5 * (lowers[0] + uppers[0]);
        if (initials[1] <= lowers[1] || initials[1] >= uppers[1])
            initials[1] = 0.5 * (lowers[1] + uppers[1]);
    }

    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            Exponential e;
            e.set_parameters(x[0], x[1]);
            return e.log_likelihood(sample);
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

}  // namespace bestfit::numerics::distributions
