// ported from: Numerics/Distributions/Univariate/Pert.cs @ a2c4dbf
//
// The PERT distribution defined by min (a), most likely / mode (c) and max (b). It is a
// transformation of the four-parameter Beta distribution: an underlying GeneralizedBeta is
// built via the PERT parameterization and PDF/CDF/InverseCDF delegate to it. Logic mirrors the
// C# source method-for-method. Implements IEstimation with the method of moments and MLE (via
// NelderMead). The method-of-percentiles path (C# ParameterEstimationMethod.MethodOfPercentiles)
// depends on PertPercentile, which is not yet ported, so it is not implemented here. The
// IBootstrappable desktop interface is not ported; IMaximumLikelihoodEstimation is wired in
// Phase 2 (bestfit/numerics/distributions/base/i_maximum_likelihood_estimation.hpp).
#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/distributions/base/i_estimation.hpp"
#include "bestfit/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "bestfit/numerics/distributions/base/parameter_estimation_method.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/distributions/generalized_beta.hpp"
#include "bestfit/numerics/math/optimization/nelder_mead.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::distributions {

class Pert : public UnivariateDistributionBase,
            public IEstimation,
            public IMaximumLikelihoodEstimation {
   public:
    // Constructs a PERT distribution with min = 0, mode = 0.5, max = 1.
    Pert() { set_parameters(0.0, 0.5, 1.0); }
    // Constructs a PERT distribution with the given min, mode and max.
    Pert(double min, double mode, double max) { set_parameters(min, mode, max); }

    const GeneralizedBeta& beta() const { return beta_; }
    double min_val() const { return min_; }
    double most_likely() const { return mode_; }
    double max_val() const { return max_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Pert;
    }
    int number_of_parameters() const override { return 3; }
    std::vector<double> get_parameters() const override { return {min_, mode_, max_}; }

    void set_parameters(double min, double mode, double max) {
        parameters_valid_ = validate(min, mode, max);
        min_ = min;
        mode_ = mode;
        max_ = max;
        if (parameters_valid_) beta_ = GeneralizedBeta::pert(min_, mode_, max_);
    }
    void set_parameters(const std::vector<double>& p) override {
        set_parameters(p[0], p[1], p[2]);
    }

    // --- Moments / support ---
    double mean() const override {
        if (almost_equals(min_, max_) && almost_equals(min_, mode_)) return min_;
        return (min_ + 4.0 * mode_ + max_) / 6.0;
    }

    double median() const override {
        if (almost_equals(min_, max_) && almost_equals(min_, mode_)) return min_;
        return (min_ + 6.0 * mode_ + max_) / 8.0;
    }

    double mode() const override { return mode_; }

    double standard_deviation() const override {
        return std::sqrt((mean() - min_) * (max_ - mean()) / 7.0);
    }

    double skewness() const override { return beta_.skewness(); }
    double kurtosis() const override { return beta_.kurtosis(); }
    double minimum() const override { return min_; }
    double maximum() const override { return max_; }

    // --- Distribution functions ---
    double pdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("Pert: invalid parameters");
        // Guard for the application case where min == max == mode.
        if (almost_equals(min_, max_) && almost_equals(min_, mode_)) return 0.0;
        if (std::isnan(mode_)) return 0.0;
        return beta_.pdf(x);
    }

    double cdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("Pert: invalid parameters");
        if (almost_equals(min_, max_) && almost_equals(min_, mode_)) return 1.0;
        if (std::isnan(mode_)) return 1.0;
        return beta_.cdf(x);
    }

    double inverse_cdf(double probability) const override {
        if (!parameters_valid_) throw std::invalid_argument("Pert: invalid parameters");
        if (almost_equals(min_, max_) && almost_equals(min_, mode_)) return min_;
        if (std::isnan(mode_)) return min_;
        return beta_.inverse_cdf(probability);
    }

    // --- Parameter display names (X1; C# Pert.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Min (a)", "Most Likely (c)", "Max (b)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"a", "c", "b"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Pert>(min_, mode_, max_);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            double s_min = *std::min_element(sample.begin(), sample.end());
            double s_max = *std::max_element(sample.begin(), sample.end());
            double s_mean = std::accumulate(sample.begin(), sample.end(), 0.0) /
                            static_cast<double>(sample.size());
            double s_mode = (s_mean * 6.0 - s_max - s_min) / 4.0;
            s_mode = std::max(s_min, s_mode);
            s_mode = std::min(s_max, s_mode);
            set_parameters(s_min, s_mode, s_max);
        } else if (method == ParameterEstimationMethod::MaximumLikelihood) {
            set_parameters(mle(sample));
        } else {
            throw std::invalid_argument("Pert: unsupported estimation method");
        }
    }

    void get_parameter_constraints(const std::vector<double>& sample,
                                   std::vector<double>& initials, std::vector<double>& lowers,
                                   std::vector<double>& uppers) const override {
        double s_min = *std::min_element(sample.begin(), sample.end());
        double s_max = *std::max_element(sample.begin(), sample.end());
        double s_mean = std::accumulate(sample.begin(), sample.end(), 0.0) /
                        static_cast<double>(sample.size());
        double s_mode = (s_mean * 6.0 - s_max - s_min) / 4.0;

        initials.resize(3);
        lowers.resize(3);
        uppers.resize(3);

        initials[0] = s_min - kDoubleMachineEpsilon;
        initials[1] = s_mode;
        initials[2] = s_max + kDoubleMachineEpsilon;

        lowers[0] = -std::pow(10.0, std::ceil(std::log10(std::fabs(s_min)) + 1.0));
        uppers[0] = s_min;
        lowers[1] = s_min;
        uppers[1] = s_max;
        lowers[2] = s_max;
        uppers[2] = std::pow(10.0, std::ceil(std::log10(std::fabs(s_max)) + 1.0));
    }

    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            Pert n;
            n.set_parameters(x[0], x[1], x[2]);
            return n.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 3, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }

   private:
    static bool almost_equals(double a, double b) {
        return std::fabs(a - b) <= kDoubleMachineEpsilon * std::fmax(std::fabs(a), std::fabs(b));
    }

    static bool validate(double min, double mode, double max) {
        if (std::isnan(min) || std::isinf(min) || std::isnan(max) || std::isinf(max) ||
            min > max)
            return false;
        if (std::isnan(mode) || std::isinf(mode) || mode < min || mode > max) return false;
        return true;
    }

    GeneralizedBeta beta_;
    double min_ = 0.0;
    double mode_ = 0.0;
    double max_ = 0.0;
};

}  // namespace bestfit::numerics::distributions
