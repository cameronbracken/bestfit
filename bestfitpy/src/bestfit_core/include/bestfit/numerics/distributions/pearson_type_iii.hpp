// ported from: Numerics/Distributions/Univariate/PearsonTypeIII.cs @ a2c4dbf
//
// Pearson Type III distribution parameterized by mean µ, standard deviation σ,
// and skew γ. Reparameterizes to a shifted Gamma distribution. Mirrors the C# source
// method-for-method. QuantileVariance/ParameterCovariance/Bootstrap/ConditionalMoments
// (desktop / uncertainty-analysis concerns) are not ported.
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/data/statistics.hpp"
#include "bestfit/numerics/distributions/base/i_estimation.hpp"
#include "bestfit/numerics/distributions/base/i_linear_moment_estimation.hpp"
#include "bestfit/numerics/distributions/base/parameter_estimation_method.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "bestfit/numerics/math/optimization/nelder_mead.hpp"
#include "bestfit/numerics/math/special/gamma.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::distributions {

namespace sf = bestfit::numerics::math::special;

class PearsonTypeIII : public UnivariateDistributionBase,
                       public IEstimation,
                       public ILinearMomentEstimation {
   public:
    // Constructs a Pearson Type III distribution with mean=100, sd=10, skew=0.
    PearsonTypeIII() { set_parameters(100.0, 10.0, 0.0); }

    // Constructs a Pearson Type III distribution with given mean, sd, and skew.
    PearsonTypeIII(double mean, double standard_deviation, double skew) {
        set_parameters(mean, standard_deviation, skew);
    }

    double mu()    const { return mu_; }
    double sigma() const { return sigma_; }
    double gamma_param() const { return gamma_; }

    // Derived gamma-parameterization accessors (mirrors C# Xi/Beta/Alpha properties).
    double xi()    const { return mu_ - 2.0 * sigma_ / gamma_; }
    double beta()  const { return 0.5 * sigma_ * gamma_; }
    double alpha() const { return 4.0 / (gamma_ * gamma_); }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::PearsonTypeIII;
    }
    int number_of_parameters() const override { return 3; }
    std::vector<double> get_parameters() const override { return {mu_, sigma_, gamma_}; }

    void set_parameters(double mean, double standard_deviation, double skew) {
        parameters_valid_ = validate(mean, standard_deviation, skew);
        mu_    = mean;
        sigma_ = standard_deviation;
        gamma_ = skew;
    }
    void set_parameters(const std::vector<double>& p) override {
        set_parameters(p[0], p[1], p[2]);
    }

    // --- Moments / support ---
    double mean() const override { return mu_; }

    double median() const override { return inverse_cdf(0.5); }

    double mode() const override {
        if (std::fabs(gamma_) <= kNearZero) {
            return mu_;  // Normal branch
        }
        return xi() + (alpha() - 1.0) * beta();
    }

    double standard_deviation() const override { return sigma_; }

    double skewness() const override { return gamma_; }

    double kurtosis() const override { return 3.0 + 6.0 / alpha(); }

    double minimum() const override {
        if (std::fabs(gamma_) <= kNearZero) {
            return -kInf;
        } else if (beta() > 0.0) {
            return xi();
        } else {
            return -kInf;
        }
    }

    double maximum() const override {
        if (std::fabs(gamma_) <= kNearZero) {
            return kInf;
        } else if (beta() > 0.0) {
            return kInf;
        } else {
            return xi();
        }
    }

    // --- Distribution functions ---
    double pdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("PearsonTypeIII: invalid parameters");
        if (x < minimum() || x > maximum()) return 0.0;
        if (std::fabs(gamma_) <= kNearZero) {
            // Normal branch
            double z = (x - mu_) / sigma_;
            return std::exp(-0.5 * z * z) / (kSqrt2PI * sigma_);
        }
        // Gamma branch
        double abs_beta = std::fabs(beta());
        if (beta() > 0.0) {
            double shifted_x = x - xi();
            return std::exp(-shifted_x / abs_beta
                            + (alpha() - 1.0) * std::log(shifted_x)
                            - alpha() * std::log(abs_beta)
                            - sf::log_gamma(alpha()));
        } else {
            double shifted_x = xi() - x;
            return std::exp(-shifted_x / abs_beta
                            + (alpha() - 1.0) * std::log(shifted_x)
                            - alpha() * std::log(abs_beta)
                            - sf::log_gamma(alpha()));
        }
    }

    double cdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("PearsonTypeIII: invalid parameters");
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return 1.0;
        if (std::fabs(gamma_) <= kNearZero) {
            return 0.5 * (1.0 + std::erf((x - mu_) / (sigma_ * kSqrt2)));
        } else if (beta() > 0.0) {
            double shifted_x = x - xi();
            return sf::lower_incomplete(alpha(), shifted_x / std::fabs(beta()));
        } else {
            double shifted_x = xi() - x;
            return 1.0 - sf::lower_incomplete(alpha(), shifted_x / std::fabs(beta()));
        }
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (!parameters_valid_) throw std::invalid_argument("PearsonTypeIII: invalid parameters");
        if (std::fabs(gamma_) <= kNearZero) {
            return mu_ + sigma_ * Normal::standard_z(probability);
        } else if (beta() > 0.0) {
            return xi() + sf::inverse_lower_incomplete(alpha(), probability) * std::fabs(beta());
        } else {
            return xi() - sf::inverse_lower_incomplete(alpha(), 1.0 - probability) * std::fabs(beta());
        }
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<PearsonTypeIII>(mu_, sigma_, gamma_);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            auto moments = data::product_moments(sample);
            set_parameters(moments[0], moments[1], moments[2]);
        } else if (method == ParameterEstimationMethod::MethodOfLinearMoments) {
            set_parameters(parameters_from_linear_moments(data::linear_moments(sample)));
        } else if (method == ParameterEstimationMethod::MaximumLikelihood) {
            set_parameters(mle(sample));
        } else {
            throw std::invalid_argument("PearsonTypeIII: unsupported estimation method");
        }
    }

    // ParametersFromLinearMoments: rational-function approximation (Hosking).
    // Mirrors C# ParametersFromLinearMoments exactly.
    std::vector<double> parameters_from_linear_moments(
        const std::vector<double>& moments) const override {
        double L1 = moments[0];
        double L2 = moments[1];
        double T3 = moments[2];
        double alpha_val = kNaN;
        double z;
        if (std::fabs(T3) > 0.0 && std::fabs(T3) < 1.0 / 3.0) {
            z = 3.0 * kPi * T3 * T3;
            alpha_val = (1.0 + 0.2906 * z)
                        / (z + 0.1882 * z * z + 0.0442 * z * z * z);
        } else if (std::fabs(T3) >= 1.0 / 3.0 && std::fabs(T3) < 1.0) {
            z = 1.0 - std::fabs(T3);
            alpha_val = (0.36067 * z - 0.59567 * z * z + 0.25361 * z * z * z)
                        / (1.0 - 2.78861 * z + 2.56096 * z * z - 0.77045 * z * z * z);
        }
        double gamma_val = 2.0 * std::pow(alpha_val, -0.5)
                           * (T3 >= 0.0 ? 1.0 : -1.0);
        double sigma_val = L2 * std::pow(kPi, 0.5) * std::pow(alpha_val, 0.5)
                           * sf::function(alpha_val) / sf::function(alpha_val + 0.5);
        double mu_val = L1;
        return {mu_val, sigma_val, gamma_val};
    }

    // LinearMomentsFromParameters: rational-function approximation (Hosking, C# mirror).
    std::vector<double> linear_moments_from_parameters(
        const std::vector<double>& parameters) const override {
        double mu_val    = parameters[0];
        double sigma_val = parameters[1];
        double gamma_val = parameters[2];
        double xi_val    = mu_val - 2.0 * sigma_val / gamma_val;
        double alpha_val = 4.0 / (gamma_val * gamma_val);
        double beta_val  = 0.5 * sigma_val * gamma_val;
        double L1 = xi_val + alpha_val * beta_val;
        double L2 = std::fabs(std::pow(kPi, -0.5) * beta_val
                              * sf::function(alpha_val + 0.5) / sf::function(alpha_val));
        // Approximations accurate to 1e-6 (mirrors C# exactly)
        constexpr double A0 = 0.32573501,  A1 = 0.1686915,   A2 = 0.078327243, A3 = -0.0029120539;
        constexpr double B1 = 0.46697102,  B2 = 0.24255406;
        constexpr double C0 = 0.12260172,  C1 = 0.05373013,  C2 = 0.043384378, C3 = 0.011101277;
        constexpr double D1 = 0.18324466,  D2 = 0.20166036;
        constexpr double E1 = 2.3807576,   E2 = 1.5931792,   E3 = 0.11618371;
        constexpr double F1 = 5.1533299,   F2 = 7.142526,    F3 = 1.9745056;
        constexpr double G1 = 2.1235833,   G2 = 4.1670213,   G3 = 3.1925299;
        constexpr double H1 = 9.0551443,   H2 = 26.649995,   H3 = 26.193668;
        double T3, T4;
        if (alpha_val >= 1.0) {
            T3 = std::pow(alpha_val, -0.5)
                 * (A0 + A1 * std::pow(alpha_val, -1) + A2 * std::pow(alpha_val, -2)
                    + A3 * std::pow(alpha_val, -3))
                 / (1.0 + B1 * std::pow(alpha_val, -1) + B2 * std::pow(alpha_val, -2));
            T4 = (C0 + C1 * std::pow(alpha_val, -1) + C2 * std::pow(alpha_val, -2)
                  + C3 * std::pow(alpha_val, -3))
                 / (1.0 + D1 * std::pow(alpha_val, -1) + D2 * std::pow(alpha_val, -2));
        } else {
            T3 = (1.0 + E1 * alpha_val + E2 * alpha_val * alpha_val
                  + E3 * alpha_val * alpha_val * alpha_val)
                 / (1.0 + F1 * alpha_val + F2 * alpha_val * alpha_val
                    + F3 * alpha_val * alpha_val * alpha_val);
            T4 = (1.0 + G1 * alpha_val + G2 * alpha_val * alpha_val
                  + G3 * alpha_val * alpha_val * alpha_val)
                 / (1.0 + H1 * alpha_val + H2 * alpha_val * alpha_val
                    + H3 * alpha_val * alpha_val * alpha_val);
        }
        // L-skewness sign follows gamma sign (positive gamma → positive T3)
        T3 = (gamma_val >= 0.0) ? T3 : -T3;
        return {L1, L2, T3, T4};
    }

    void get_parameter_constraints(const std::vector<double>& sample,
                                   std::vector<double>& initials,
                                   std::vector<double>& lowers,
                                   std::vector<double>& uppers) const {
        auto moments = data::product_moments(sample);
        initials = {moments[0], moments[1], moments[2]};
        lowers.resize(3);
        uppers.resize(3);
        // Bounds of mean
        lowers[0] = -std::pow(10.0, std::ceil(std::log10(initials[0]) + 1.0));
        uppers[0] =  std::pow(10.0, std::ceil(std::log10(initials[0]) + 1.0));
        // Bounds of standard deviation
        lowers[1] = kDoubleMachineEpsilon;
        uppers[1] = std::pow(10.0, std::ceil(std::log10(initials[1]) + 1.0));
        // Bounds of skew
        lowers[2] = -6.0;
        uppers[2] =  6.0;
        // Correct initial value of skew if necessary
        if (initials[2] <= lowers[2] || initials[2] >= uppers[2]) {
            initials[2] = 0.01;
        }
    }

    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            PearsonTypeIII p3;
            p3.set_parameters(x[0], x[1], x[2]);
            return p3.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 3, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }

   private:
    static bool validate(double mean, double sd, double skew) {
        if (std::isnan(mean) || std::isinf(mean)) return false;
        if (std::isnan(sd)   || std::isinf(sd)   || sd <= 0.0) return false;
        if (std::isnan(skew) || std::isinf(skew)) return false;
        if (skew > 6.0 || skew < -6.0) return false;
        return true;
    }

    double mu_    = 100.0;
    double sigma_ = 10.0;
    double gamma_ = 0.0;
};

}  // namespace bestfit::numerics::distributions
