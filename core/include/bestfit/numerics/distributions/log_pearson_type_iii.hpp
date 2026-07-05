// ported from: Numerics/Distributions/Univariate/LogPearsonTypeIII.cs @ a2c4dbf
//
// Log-Pearson Type III distribution parameterized by mean µ, standard deviation σ,
// and skew γ of the log-transformed (base-10) data. Wraps PearsonTypeIII in log10
// space. Mirrors the C# source method-for-method. Standard USACE flood-frequency
// distribution (Bulletin 17C).
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/data/statistics.hpp"
#include "bestfit/numerics/distributions/base/i_estimation.hpp"
#include "bestfit/numerics/distributions/base/i_linear_moment_estimation.hpp"
#include "bestfit/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "bestfit/numerics/distributions/base/parameter_estimation_method.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/math/optimization/nelder_mead.hpp"
#include "bestfit/numerics/math/special/erf.hpp"
#include "bestfit/numerics/math/special/gamma.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::distributions {

namespace sf = bestfit::numerics::math::special;

class LogPearsonTypeIII : public UnivariateDistributionBase,
                          public IEstimation,
                          public ILinearMomentEstimation,
                          public IMaximumLikelihoodEstimation {
   public:
    // Constructs a LogPearsonTypeIII with mean(log)=3, sd(log)=0.5, skew(log)=0.
    LogPearsonTypeIII() { set_parameters(3.0, 0.5, 0.0); }

    // Constructs a LogPearsonTypeIII with given mean, sd, and skew of log10(X).
    LogPearsonTypeIII(double mean_of_log, double sd_of_log, double skew_of_log) {
        set_parameters(mean_of_log, sd_of_log, skew_of_log);
    }

    double mu()    const { return mu_; }
    double sigma() const { return sigma_; }
    double gamma_param() const { return gamma_; }

    // Derived gamma-parameterization (log-space shifted Gamma) – mirrors C# Xi/Beta/Alpha.
    double xi()    const { return mu_ - 2.0 * sigma_ / gamma_; }
    double beta()  const { return 0.5 * sigma_ * gamma_; }
    double alpha() const { return 4.0 / (gamma_ * gamma_); }

    // Log-correction factor: K = 1 / ln(base), base = 10.
    // log_base(x) = ln(x) * K  (log10(x) = ln(x)/ln(10))
    // K = 1/ln(10)
    double k_factor() const { return 1.0 / std::log(kBase); }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::LogPearsonTypeIII;
    }
    int number_of_parameters() const override { return 3; }
    std::vector<double> get_parameters() const override { return {mu_, sigma_, gamma_}; }

    void set_parameters(double mean_of_log, double sd_of_log, double skew_of_log) {
        parameters_valid_ = validate(mean_of_log, sd_of_log, skew_of_log);
        mu_    = mean_of_log;
        sigma_ = sd_of_log;
        gamma_ = skew_of_log;
    }
    void set_parameters(const std::vector<double>& p) override {
        set_parameters(p[0], p[1], p[2]);
    }

    // --- Moments / support ---
    // Mean of X (not of log X). Mirrors C# Mean property.
    double mean() const override {
        double ln_b = std::log(kBase);
        if (std::fabs(gamma_) <= kNearZero) {
            // Log-Normal case: E[X] = exp((µ + 0.5*σ²*ln(b)) * ln(b))
            return std::exp((mu_ + 0.5 * sigma_ * sigma_ * ln_b) * ln_b);
        } else {
            double ln_mean = xi() * ln_b - alpha() * std::log(1.0 - beta() * ln_b);
            return std::exp(ln_mean);
        }
    }

    double median() const override { return inverse_cdf(0.5); }

    // Mode of X. Mirrors C# Mode property.
    double mode() const override {
        double K = k_factor();
        if (std::fabs(gamma_) <= kNearZero) {
            return std::exp(mu_ / K);
        } else {
            return std::exp((xi() + (alpha() - 1.0) * beta()) / K);
        }
    }

    // Standard deviation of X. Mirrors C# StandardDeviation property.
    double standard_deviation() const override {
        double ln_b = std::log(kBase);
        if (std::fabs(gamma_) <= kNearZero) {
            double a = sigma_ * sigma_ * ln_b;
            double log_prefactor = (2.0 * mu_ + a) * ln_b;
            double exp_a = std::exp(a * ln_b);
            double variance = std::exp(log_prefactor) * (exp_a - 1.0);
            return std::sqrt(variance);
        } else {
            double t1 = -alpha() * std::log(1.0 - 2.0 * beta() * ln_b);
            double t2 = -2.0 * alpha() * std::log(1.0 - beta() * ln_b);
            double max_t = std::max(t1, t2);
            double diff = std::exp(t1 - max_t) - std::exp(t2 - max_t);
            double log_variance = 2.0 * xi() * ln_b + max_t + std::log(diff);
            return std::sqrt(std::exp(log_variance));
        }
    }

    // Skewness of X. Mirrors C# Skewness property.
    double skewness() const override {
        double ln_b = std::log(kBase);
        if (std::fabs(gamma_) <= kNearZero) {
            // Log-Normal case
            double a = sigma_ * sigma_ * ln_b;
            double mu1 = (mu_ + 0.5 * a) * ln_b;
            double mu2 = (2.0 * mu_ + 2.0 * a) * ln_b;
            double mu3 = (3.0 * mu_ + 4.5 * a) * ln_b;
            double m1 = std::exp(mu1);
            double m2 = std::exp(mu2);
            double m3 = std::exp(mu3);
            double third_central = m3 - 3.0 * m2 * m1 + 2.0 * m1 * m1 * m1;
            double sd = standard_deviation();
            return third_central / (sd * sd * sd);
        } else {
            // LP3 case
            double t1 = 1.0 - beta() * ln_b;
            double t2 = 1.0 - 2.0 * beta() * ln_b;
            double t3 = 1.0 - 3.0 * beta() * ln_b;
            double m1 = std::pow(t1, -alpha());
            double m2 = std::pow(t2, -alpha());
            double m3 = std::pow(t3, -alpha());
            double third_central = m3 - 3.0 * m2 * m1 + 2.0 * m1 * m1 * m1;
            double prefactor = std::pow(kBase, 3.0 * xi());
            double sd = standard_deviation();
            return prefactor * third_central / (sd * sd * sd);
        }
    }

    // Kurtosis of X. Mirrors C# Kurtosis property.
    double kurtosis() const override {
        double ln_b = std::log(kBase);
        if (std::fabs(gamma_) <= kNearZero) {
            // Log-Normal case
            double a = sigma_ * sigma_ * ln_b;
            double mu1 = (mu_ + 0.5 * a) * ln_b;
            double mu2 = (2.0 * mu_ + 2.0 * a) * ln_b;
            double mu3 = (3.0 * mu_ + 4.5 * a) * ln_b;
            double mu4 = (4.0 * mu_ + 8.0 * a) * ln_b;
            double m1 = std::exp(mu1);
            double m2 = std::exp(mu2);
            double m3 = std::exp(mu3);
            double m4 = std::exp(mu4);
            double fourth_central = m4 - 4.0 * m3 * m1 + 6.0 * m2 * m1 * m1
                                    - 3.0 * m1 * m1 * m1 * m1;
            double sd = standard_deviation();
            return fourth_central / std::pow(sd, 4.0);
        } else {
            // LP3 case
            double t1 = 1.0 - beta() * ln_b;
            double t2 = 1.0 - 2.0 * beta() * ln_b;
            double t3 = 1.0 - 3.0 * beta() * ln_b;
            double t4 = 1.0 - 4.0 * beta() * ln_b;
            double m1 = std::pow(t1, -alpha());
            double m2 = std::pow(t2, -alpha());
            double m3 = std::pow(t3, -alpha());
            double m4 = std::pow(t4, -alpha());
            double fourth_central = m4 - 4.0 * m3 * m1 + 6.0 * m2 * m1 * m1
                                    - 3.0 * m1 * m1 * m1 * m1;
            double prefactor = std::pow(kBase, 4.0 * xi());
            double sd = standard_deviation();
            return prefactor * fourth_central / std::pow(sd, 4.0);
        }
    }

    // Minimum of X. Mirrors C# Minimum property.
    double minimum() const override {
        double K = k_factor();
        if (std::fabs(gamma_) <= kNearZero) {
            return 0.0;
        } else if (beta() > 0.0) {
            return std::exp(xi() / K);
        } else {
            return 0.0;
        }
    }

    // Maximum of X. Mirrors C# Maximum property.
    double maximum() const override {
        double K = k_factor();
        if (std::fabs(gamma_) <= kNearZero) {
            return kInf;
        } else if (beta() > 0.0) {
            return kInf;
        } else {
            return std::exp(xi() / K);
        }
    }

    // --- Distribution functions ---
    double pdf(double x) const override {
        if (!parameters_valid_)
            throw std::invalid_argument("LogPearsonTypeIII: invalid parameters");
        if (x < minimum() || x > maximum()) return 0.0;
        double K = k_factor();
        if (std::fabs(gamma_) <= kNearZero) {
            // Log-Normal branch
            double log10x = std::log(x) / std::log(kBase);  // log10(x) = log_base(x)
            double d = (log10x - mu_) / sigma_;
            return std::exp(-0.5 * d * d) / (kSqrt2PI * sigma_) * (K / x);
        } else if (beta() > 0.0) {
            double shifted_x = std::log(x) / std::log(kBase) - xi();
            double abs_beta = std::fabs(beta());
            return std::exp(-shifted_x / abs_beta
                            + (alpha() - 1.0) * std::log(shifted_x)
                            - alpha() * std::log(abs_beta)
                            - sf::log_gamma(alpha()))
                   * (K / x);
        } else {
            double shifted_x = xi() - std::log(x) / std::log(kBase);
            double abs_beta = std::fabs(beta());
            return std::exp(-shifted_x / abs_beta
                            + (alpha() - 1.0) * std::log(shifted_x)
                            - alpha() * std::log(abs_beta)
                            - sf::log_gamma(alpha()))
                   * (K / x);
        }
    }

    double cdf(double x) const override {
        if (!parameters_valid_)
            throw std::invalid_argument("LogPearsonTypeIII: invalid parameters");
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return 1.0;
        if (std::fabs(gamma_) <= kNearZero) {
            double log10x = std::log(x) / std::log(kBase);
            return 0.5 * (1.0 + std::erf((log10x - mu_) / (sigma_ * kSqrt2)));
        } else if (beta() > 0.0) {
            double shifted_x = std::log(x) / std::log(kBase) - xi();
            return sf::lower_incomplete(alpha(), shifted_x / std::fabs(beta()));
        } else {
            double shifted_x = xi() - std::log(x) / std::log(kBase);
            return 1.0 - sf::lower_incomplete(alpha(), shifted_x / std::fabs(beta()));
        }
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (!parameters_valid_)
            throw std::invalid_argument("LogPearsonTypeIII: invalid parameters");
        double K = k_factor();
        if (std::fabs(gamma_) <= kNearZero) {
            // Log-Normal branch: InverseCDF using erfc
            return std::exp((mu_ - sigma_ * kSqrt2
                             * sf::erf::inverse_erfc(2.0 * probability))
                            / K);
        } else if (beta() > 0.0) {
            return std::exp(
                (xi() + sf::inverse_lower_incomplete(alpha(), probability) * std::fabs(beta()))
                / K);
        } else {
            return std::exp(
                (xi() - sf::inverse_lower_incomplete(alpha(), 1.0 - probability)
                       * std::fabs(beta()))
                / K);
        }
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<LogPearsonTypeIII>(mu_, sigma_, gamma_);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            // Indirect MoM: transform to log10, compute product moments.
            auto log_sample = transform_log(sample);
            auto moments = data::product_moments(log_sample);
            set_parameters(moments[0], moments[1], moments[2]);
        } else if (method == ParameterEstimationMethod::MethodOfLinearMoments) {
            // Indirect L-moments: transform to log10, compute L-moments, then fit.
            auto log_sample = transform_log(sample);
            auto lmom = data::linear_moments(log_sample);
            set_parameters(parameters_from_linear_moments(lmom));
        } else if (method == ParameterEstimationMethod::MaximumLikelihood) {
            set_parameters(mle(sample));
        } else {
            throw std::invalid_argument("LogPearsonTypeIII: unsupported estimation method");
        }
    }

    // ParametersFromMoments (C# LogPearsonTypeIII.cs:559): the LP3 is parameterized by
    // the first three (log10-space) moments (C# moments.ToArray().Subset(0, 2)).
    // Added in B4 for the Bulletin 17C GMM track.
    std::vector<double> parameters_from_moments(const std::vector<double>& moments) const {
        return {moments[0], moments[1], moments[2]};
    }

    // MomentsFromParameters (C# LogPearsonTypeIII.cs:565): {Mean, StandardDeviation,
    // Skewness, Kurtosis} of an LP3 built from the parameters. Note these are the
    // REAL-SPACE moments of X, not the log-space parameters (upstream asymmetry; C#
    // governs). Added in B4.
    std::vector<double> moments_from_parameters(const std::vector<double>& parameters) const {
        LogPearsonTypeIII dist;
        dist.set_parameters(parameters);
        double m1 = dist.mean();
        double m2 = dist.standard_deviation();
        double m3 = dist.skewness();
        double m4 = dist.kurtosis();
        return {m1, m2, m3, m4};
    }

    // ParametersFromLinearMoments: rational-function approximation (Hosking).
    // Mirrors C# ParametersFromLinearMoments exactly (same code as PearsonTypeIII).
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
        double mu_val    = L1;
        double gamma_val = 2.0 * std::pow(alpha_val, -0.5)
                           * (T3 >= 0.0 ? 1.0 : -1.0);
        double sigma_val;
        if (alpha_val < 100.0) {
            sigma_val = L2 * std::pow(kPi, 0.5) * std::pow(alpha_val, 0.5)
                        * sf::function(alpha_val) / sf::function(alpha_val + 0.5);
        } else {
            sigma_val = std::sqrt(kPi) * L2
                        / (1.0 - 1.0 / (8.0 * alpha_val)
                           + 1.0 / (128.0 * alpha_val * alpha_val));
        }
        return {mu_val, sigma_val, gamma_val};
    }

    // LinearMomentsFromParameters: rational-function approximation (Hosking).
    // Returns L-moments of log(X) (the underlying PT3). Mirrors C# exactly.
    std::vector<double> linear_moments_from_parameters(
        const std::vector<double>& parameters) const override {
        double mu_val    = parameters[0];
        double sigma_val = parameters[1];
        double gamma_val = parameters[2];
        double xi_val    = mu_val - 2.0 * sigma_val / gamma_val;
        double alpha_val = 4.0 / (gamma_val * gamma_val);
        double beta_val  = 0.5 * sigma_val * gamma_val;
        double L1 = xi_val + alpha_val * beta_val;
        double L2;
        if (alpha_val < 100.0) {
            L2 = std::fabs(std::pow(kPi, -0.5) * beta_val
                           * sf::function(alpha_val + 0.5) / sf::function(alpha_val));
        } else {
            L2 = std::sqrt(kPi) * std::fabs(beta_val)
                 / (1.0 - 1.0 / (8.0 * alpha_val)
                    + 1.0 / (128.0 * alpha_val * alpha_val));
        }
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
        return {L1, L2, T3, T4};
    }

    // GetParameterConstraints: mirrors C# GetParameterConstraints for MLE.
    void get_parameter_constraints(const std::vector<double>& sample,
                                   std::vector<double>& initials,
                                   std::vector<double>& lowers,
                                   std::vector<double>& uppers) const override {
        // Use indirect MoM for initial values
        auto log_sample = transform_log(sample);
        auto mom = data::product_moments(log_sample);
        initials = {mom[0], mom[1], mom[2]};
        lowers.resize(3);
        uppers.resize(3);
        double K = k_factor();
        // Bounds of mu
        double real_mu = std::exp(initials[0] / K);
        lowers[0] = kDoubleMachineEpsilon;
        uppers[0] = std::ceil(std::log(std::pow(10.0, std::ceil(std::log10(real_mu) + 1.0)))
                              / std::log(kBase));
        if (std::isnan(uppers[0])) uppers[0] = 5.0;
        // Bounds of sigma
        double real_sigma = std::exp(initials[1] / K);
        lowers[1] = kDoubleMachineEpsilon;
        uppers[1] = std::ceil(std::log(std::pow(10.0, std::ceil(std::log10(real_sigma) + 1.0)))
                              / std::log(kBase));
        if (std::isnan(uppers[1])) uppers[1] = 4.0;
        // Bounds of gamma
        lowers[2] = -6.0;
        uppers[2] = 6.0;
        // Correct initial skew if out of range
        if (initials[2] <= lowers[2] || initials[2] >= uppers[2]) {
            initials[2] = 0.01;
        }
    }

    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            LogPearsonTypeIII lp3;
            lp3.set_parameters(x[0], x[1], x[2]);
            return lp3.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 3, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }

   private:
    // Transform sample to log-base (base 10). Values <= 0 map to log10(0.01) as in C#.
    std::vector<double> transform_log(const std::vector<double>& sample) const {
        std::vector<double> result;
        result.reserve(sample.size());
        for (double v : sample) {
            if (v > 0.0) {
                result.push_back(std::log(v) / std::log(kBase));
            } else {
                result.push_back(std::log(0.01) / std::log(kBase));
            }
        }
        return result;
    }

    static bool validate(double mu, double sigma, double gamma) {
        if (std::isnan(mu)    || std::isinf(mu))    return false;
        if (std::isnan(sigma) || std::isinf(sigma) || sigma <= 0.0) return false;
        if (std::isnan(gamma) || std::isinf(gamma)) return false;
        if (gamma > 6.0 || gamma < -6.0) return false;
        return true;
    }

    static constexpr double kBase    = 10.0;
    static constexpr double kSqrt2   = 1.4142135623730951;
    static constexpr double kSqrt2PI = 2.5066282746310002;

    double mu_    = 3.0;
    double sigma_ = 0.5;
    double gamma_ = 0.0;
};

}  // namespace bestfit::numerics::distributions
