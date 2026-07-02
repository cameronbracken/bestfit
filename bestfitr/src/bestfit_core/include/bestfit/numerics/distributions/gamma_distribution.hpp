// ported from: Numerics/Distributions/Univariate/GammaDistribution.cs @ a2c4dbf
//
// Gamma distribution with scale θ (theta) and shape κ (kappa). Logic mirrors the C#
// source method-for-method. The WPF helpers, IBootstrappable, IStandardError,
// ConditionalMoments, WilsonHilferty, and PartialKp are not ported (desktop / uncertainty
// analysis concerns).
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/data/statistics.hpp"
#include "bestfit/numerics/distributions/base/i_estimation.hpp"
#include "bestfit/numerics/distributions/base/i_linear_moment_estimation.hpp"
#include "bestfit/numerics/distributions/base/parameter_estimation_method.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/math/optimization/nelder_mead.hpp"
#include "bestfit/numerics/math/special/gamma.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::distributions {

namespace sf = bestfit::numerics::math::special;

class GammaDistribution : public UnivariateDistributionBase,
                          public IEstimation,
                          public ILinearMomentEstimation {
   public:
    // Constructs a Gamma distribution with scale θ = 10 and shape κ = 2.
    GammaDistribution() { set_parameters(10.0, 2.0); }

    // Constructs a Gamma distribution with given scale θ and shape κ.
    GammaDistribution(double scale, double shape) { set_parameters(scale, shape); }

    double theta() const { return theta_; }
    double kappa() const { return kappa_; }
    double rate()  const { return 1.0 / theta_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::GammaDistribution;
    }
    int number_of_parameters() const override { return 2; }
    std::vector<double> get_parameters() const override { return {theta_, kappa_}; }

    void set_parameters(double scale, double shape) {
        parameters_valid_ = validate(scale, shape);
        theta_ = scale;
        kappa_ = shape;
    }
    void set_parameters(const std::vector<double>& p) override { set_parameters(p[0], p[1]); }

    // --- Moments / support ---
    double mean() const override { return kappa_ * theta_; }

    double median() const override { return inverse_cdf(0.5); }

    double mode() const override {
        if (kappa_ > 1.0)
            return (kappa_ - 1.0) * theta_;
        return std::numeric_limits<double>::quiet_NaN();
    }

    double standard_deviation() const override {
        return std::sqrt(kappa_ * theta_ * theta_);
    }

    double skewness() const override { return 2.0 / std::sqrt(kappa_); }

    double kurtosis() const override { return 3.0 + 6.0 / kappa_; }

    double minimum() const override { return 0.0; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---
    double pdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("GammaDistribution: invalid parameters");
        if (x < minimum() || x > maximum()) return 0.0;
        return std::exp(-x / theta_ + (kappa_ - 1.0) * std::log(x)
                        - kappa_ * std::log(theta_) - sf::log_gamma(kappa_));
    }

    double cdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("GammaDistribution: invalid parameters");
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return 1.0;
        return sf::lower_incomplete(kappa_, x / theta_);
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (!parameters_valid_) throw std::invalid_argument("GammaDistribution: invalid parameters");
        return sf::inverse_lower_incomplete(kappa_, probability) * theta_;
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<GammaDistribution>(theta_, kappa_);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            set_parameters(parameters_from_moments(data::product_moments(sample)));
        } else if (method == ParameterEstimationMethod::MethodOfLinearMoments) {
            set_parameters(parameters_from_linear_moments(data::linear_moments(sample)));
        } else if (method == ParameterEstimationMethod::MaximumLikelihood) {
            set_parameters(mle(sample));
        } else {
            throw std::invalid_argument("GammaDistribution: unsupported estimation method");
        }
    }

    // ParametersFromMoments: theta = sd^2 / mean, kappa = mean^2 / sd^2
    std::vector<double> parameters_from_moments(const std::vector<double>& moments) const {
        double mean = moments[0];
        double sd   = moments[1];
        double theta = 1.0 / (mean / (sd * sd));
        double kappa = (mean * mean) / (sd * sd);
        return {theta, kappa};
    }

    // ParametersFromLinearMoments: rational-function approximation (Hosking)
    std::vector<double> parameters_from_linear_moments(
        const std::vector<double>& moments) const override {
        constexpr double A1 = -0.3080, A2 = -0.05812, A3 = 0.01765;
        constexpr double B1 = 0.7213,  B2 = -0.5947, B3 = -2.1817, B4 = 1.2113;
        double L1 = moments[0];
        double L2 = moments[1];
        double CV = L2 / L1;
        double T, theta, kappa;
        if (CV < 0.5) {
            T = kPi * CV * CV;
            kappa = (1.0 + A1 * T) / (T * (1.0 + T * (A2 + T * A3)));
        } else {
            T = 1.0 - CV;
            kappa = T * (B1 + T * B2) / (1.0 + T * (B3 + T * B4));
        }
        theta = L1 / kappa;
        return {theta, kappa};
    }

    // LinearMomentsFromParameters: Hosking approximation accurate to 1e-6
    std::vector<double> linear_moments_from_parameters(
        const std::vector<double>& parameters) const override {
        double alpha = parameters[1];  // shape
        double beta  = parameters[0];  // scale
        double L1 = alpha * beta;
        double L2 = std::fabs(std::pow(kPi, -0.5) * beta
                              * sf::function(alpha + 0.5) / sf::function(alpha));
        constexpr double A0 = 0.32573501, A1 = 0.1686915,  A2 = 0.078327243, A3 = -0.0029120539;
        constexpr double B1 = 0.46697102, B2 = 0.24255406;
        constexpr double C0 = 0.12260172, C1 = 0.05373013,  C2 = 0.043384378, C3 = 0.011101277;
        constexpr double D1 = 0.18324466, D2 = 0.20166036;
        constexpr double E1 = 2.3807576,  E2 = 1.5931792,   E3 = 0.11618371;
        constexpr double F1 = 5.1533299,  F2 = 7.142526,    F3 = 1.9745056;
        constexpr double G1 = 2.1235833,  G2 = 4.1670213,   G3 = 3.1925299;
        constexpr double H1 = 9.0551443,  H2 = 26.649995,   H3 = 26.193668;
        double T3, T4;
        if (alpha >= 1.0) {
            T3 = std::pow(alpha, -0.5)
                 * (A0 + A1 * std::pow(alpha, -1) + A2 * std::pow(alpha, -2) + A3 * std::pow(alpha, -3))
                 / (1.0 + B1 * std::pow(alpha, -1) + B2 * std::pow(alpha, -2));
            T4 = (C0 + C1 * std::pow(alpha, -1) + C2 * std::pow(alpha, -2) + C3 * std::pow(alpha, -3))
                 / (1.0 + D1 * std::pow(alpha, -1) + D2 * std::pow(alpha, -2));
        } else {
            T3 = (1.0 + E1 * alpha + E2 * alpha * alpha + E3 * alpha * alpha * alpha)
                 / (1.0 + F1 * alpha + F2 * alpha * alpha + F3 * alpha * alpha * alpha);
            T4 = (1.0 + G1 * alpha + G2 * alpha * alpha + G3 * alpha * alpha * alpha)
                 / (1.0 + H1 * alpha + H2 * alpha * alpha + H3 * alpha * alpha * alpha);
        }
        return {L1, L2, T3, T4};
    }

    void get_parameter_constraints(const std::vector<double>& sample,
                                   std::vector<double>& initials,
                                   std::vector<double>& lowers,
                                   std::vector<double>& uppers) const {
        initials = parameters_from_moments(data::product_moments(sample));
        lowers.resize(2);
        uppers.resize(2);
        lowers[0] = kDoubleMachineEpsilon;
        uppers[0] = std::pow(10.0, std::ceil(std::log10(initials[0]) + 1.0));
        lowers[1] = kDoubleMachineEpsilon;
        uppers[1] = std::pow(10.0, std::ceil(std::log10(initials[1]) + 1.0));
    }

    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            GammaDistribution g;
            g.set_parameters(x[0], x[1]);
            return g.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 2, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }

   private:
    static bool validate(double scale, double shape) {
        if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0) return false;
        if (std::isnan(shape) || std::isinf(shape) || shape <= 0.0) return false;
        return true;
    }

    double theta_ = 10.0;
    double kappa_ = 2.0;
};

}  // namespace bestfit::numerics::distributions
