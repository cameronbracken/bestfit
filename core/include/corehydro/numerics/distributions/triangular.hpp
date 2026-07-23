// ported from: Numerics/Distributions/Univariate/Triangular.cs @ 2a0357a
//
// The Triangular distribution with min (a), mode (c), and max (b). Logic mirrors the C# source
// method-for-method. Implements IEstimation (MoM and MLE via NelderMead). The IBootstrappable
// interface is a desktop/application concern and is not ported.
#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

class Triangular : public UnivariateDistributionBase,
                   public IEstimation,
                   public IMaximumLikelihoodEstimation {
   public:
    Triangular() { set_parameters(0.0, 0.5, 1.0); }
    Triangular(double min, double mode, double max) { set_parameters(min, mode, max); }

    double min_val() const { return min_; }
    double most_likely() const { return mode_; }
    double max_val() const { return max_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Triangular;
    }
    int number_of_parameters() const override { return 3; }
    std::vector<double> get_parameters() const override { return {min_, mode_, max_}; }

    void set_parameters(double min, double mode, double max) {
        parameters_valid_ = validate(min, mode, max);
        min_ = min;
        mode_ = mode;
        max_ = max;
    }
    void set_parameters(const std::vector<double>& p) override {
        set_parameters(p[0], p[1], p[2]);
    }

    // --- Moments / support ---
    double mean() const override { return (min_ + max_ + mode_) / 3.0; }

    double median() const override {
        double mid = (min_ + max_) / 2.0;
        if (mode_ >= mid) {
            return min_ + std::sqrt((max_ - min_) * (mode_ - min_) / 2.0);
        } else {
            return max_ - std::sqrt((max_ - min_) * (max_ - mode_) / 2.0);
        }
    }

    double mode() const override { return mode_; }

    double standard_deviation() const override {
        return std::sqrt((min_ * min_ + max_ * max_ + mode_ * mode_ -
                          min_ * max_ - min_ * mode_ - max_ * mode_) /
                         18.0);
    }

    double skewness() const override {
        double q = kSqrt2 * (min_ + max_ - 2.0 * mode_) * (2.0 * min_ - max_ - mode_) *
                   (min_ - 2.0 * max_ + mode_);
        double d = 5.0 * std::pow(min_ * min_ + max_ * max_ + mode_ * mode_ -
                                      min_ * max_ - min_ * mode_ - max_ * mode_,
                                  3.0 / 2.0);
        return q / d;
    }

    double kurtosis() const override { return 12.0 / 5.0; }
    double minimum() const override { return min_; }
    double maximum() const override { return max_; }

    // --- Distribution functions ---
    double pdf(double x) const override {
        // Degenerate case: all three parameters equal
        if (std::fabs(min_ - max_) < kNearZero && std::fabs(min_ - mode_) < kNearZero)
            return 0.0;
        if (x < minimum() || x > maximum()) return 0.0;
        if (x >= min_ && x < mode_) {
            return 2.0 * (x - min_) / ((max_ - min_) * (mode_ - min_));
        } else if (x > mode_ && x <= max_) {
            return 2.0 * (max_ - x) / ((max_ - min_) * (max_ - mode_));
        } else if (x == mode_) {
            return 2.0 / (max_ - min_);
        }
        return kNaN;
    }

    double cdf(double x) const override {
        // Degenerate case
        if (std::fabs(min_ - max_) < kNearZero && std::fabs(min_ - mode_) < kNearZero)
            return 1.0;
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return 1.0;
        if (x > min_ && x <= mode_) {
            return (x - min_) * (x - min_) / ((max_ - min_) * (mode_ - min_));
        } else if (x > mode_ && x < max_) {
            return 1.0 - (max_ - x) * (max_ - x) / ((max_ - min_) * (max_ - mode_));
        }
        return kNaN;
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        // Degenerate case
        if (std::fabs(min_ - max_) < kNearZero && std::fabs(min_ - mode_) < kNearZero)
            return min_;
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        double fc = (mode_ - min_) / (max_ - min_);
        if (probability < fc) {
            return min_ + std::sqrt(probability * (max_ - min_) * (mode_ - min_));
        } else {
            return max_ - std::sqrt((1.0 - probability) * (max_ - min_) * (max_ - mode_));
        }
    }

    // --- Parameter display names (X1; C# Triangular.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Min (a)", "Most Likely (c)", "Max (b)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"a", "c", "b"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Triangular>(min_, mode_, max_);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            double s_min = *std::min_element(sample.begin(), sample.end());
            double s_max = *std::max_element(sample.begin(), sample.end());
            double s_mean = std::accumulate(sample.begin(), sample.end(), 0.0) /
                            static_cast<double>(sample.size());
            double s_mode = s_mean * 3.0 - s_max - s_min;
            s_mode = std::max(s_min, s_mode);
            s_mode = std::min(s_max, s_mode);
            set_parameters(s_min, s_mode, s_max);
        } else if (method == ParameterEstimationMethod::MaximumLikelihood) {
            set_parameters(mle(sample));
        } else {
            throw std::invalid_argument("Triangular: unsupported estimation method");
        }
    }

    void get_parameter_constraints(const std::vector<double>& sample,
                                   std::vector<double>& initials, std::vector<double>& lowers,
                                   std::vector<double>& uppers) const override {
        double s_min = *std::min_element(sample.begin(), sample.end());
        double s_max = *std::max_element(sample.begin(), sample.end());
        double s_mean = std::accumulate(sample.begin(), sample.end(), 0.0) /
                        static_cast<double>(sample.size());
        double s_mode = s_mean * 3.0 - s_max - s_min;

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
            Triangular t;
            t.set_parameters(x[0], x[1], x[2]);
            return t.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 3, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }

   private:
    static bool validate(double min, double mode, double max) {
        if (std::isnan(min) || std::isinf(min) || std::isnan(max) || std::isinf(max) ||
            min > max)
            return false;
        if (std::isnan(mode) || std::isinf(mode) || mode < min || mode > max) return false;
        return true;
    }

    double min_ = 0.0;
    double mode_ = 0.0;
    double max_ = 0.0;
};

}  // namespace corehydro::numerics::distributions
