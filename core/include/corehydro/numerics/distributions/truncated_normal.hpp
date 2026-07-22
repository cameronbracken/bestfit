// ported from: Numerics/Distributions/Univariate/TruncatedNormal.cs @ 2a0357a
//
// The Truncated Normal (Gaussian) distribution with parameters mu, sigma, min, max.
// Uses the standard normal CDF/PDF (Normal.StandardCDF / StandardPDF / StandardZ in C#).
// Logic mirrors the C# source method-for-method. Only IEstimation (MoM) is implemented.
// ILinearMomentEstimation is not implemented (C# class does not implement it).
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

class TruncatedNormal : public UnivariateDistributionBase, public IEstimation {
   public:
    // Default constructor: mu=0.5, sigma=0.2, min=0, max=1 (mirrors C# default)
    TruncatedNormal() { set_parameters(0.5, 0.2, 0.0, 1.0); }
    TruncatedNormal(double mean, double standard_deviation, double min_val, double max_val) {
        set_parameters(mean, standard_deviation, min_val, max_val);
    }

    double mu() const { return mu_; }
    double sigma() const { return sigma_; }
    double min_param() const { return min_; }
    double max_param() const { return max_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::TruncatedNormal;
    }
    int number_of_parameters() const override { return 4; }
    std::vector<double> get_parameters() const override { return {mu_, sigma_, min_, max_}; }

    void set_parameters(double mean, double sigma, double min_val, double max_val) {
        if (sigma < 1E-16 && std::signbit(sigma) == false) sigma = 1E-16;
        mu_ = mean;
        sigma_ = sigma;
        min_ = min_val;
        max_ = max_val;
        parameters_valid_ = validate(mean, sigma, min_val, max_val);
    }
    void set_parameters(const std::vector<double>& p) override {
        set_parameters(p[0], p[1], p[2], p[3]);
    }

    // --- Moments / support ---
    double mean() const override {
        double alph = (min_ - mu_) / sigma_;
        double bet = (max_ - mu_) / sigma_;
        double Z = std_cdf(bet) - std_cdf(alph);
        if (Z == 0.0) return mu_;
        return mu_ + sigma_ * (std_pdf(alph) - std_pdf(bet)) / Z;
    }

    double median() const override {
        double alph = (min_ - mu_) / sigma_;
        double bet = (max_ - mu_) / sigma_;
        double Z = std_cdf(bet) - std_cdf(alph);
        if (Z == 0.0) return mu_;
        return mu_ + sigma_ * std_z((std_cdf(alph) + std_cdf(bet)) / 2.0);
    }

    double mode() const override {
        if (mu_ < min_) return min_;
        if (mu_ >= min_ && mu_ <= max_) return mu_;
        if (mu_ > max_) return max_;
        return 0.0;
    }

    double standard_deviation() const override {
        double alph = (min_ - mu_) / sigma_;
        double bet = (max_ - mu_) / sigma_;
        double Z = std_cdf(bet) - std_cdf(alph);
        if (Z == 0.0) return sigma_;
        double term1 = (alph * std_pdf(alph) - bet * std_pdf(bet)) / Z;
        double term2 = (std_pdf(alph) - std_pdf(bet)) / Z;
        return std::sqrt(sigma_ * sigma_ * (1.0 + term1 - term2 * term2));
    }

    // Skewness: Sugiura & Gomi 1985 formula
    double skewness() const override {
        double K0 = (min_ - mu_) / sigma_;
        double K1 = (max_ - mu_) / sigma_;
        double Z = std_cdf(K1) - std_cdf(K0);
        if (Z == 0.0) return 0.0;
        double Z0 = std_pdf(K0) / Z;
        double Z1 = std_pdf(K1) / Z;
        double V = 1.0 - (K1 * Z1 - K0 * Z0) - (Z1 - Z0) * (Z1 - Z0);
        return -(1.0 / std::pow(V, 1.5)) *
               (2.0 * (Z1 - Z0) * (Z1 - Z0) * (Z1 - Z0) +
                (3.0 * K1 * Z1 - 3.0 * K0 * Z0 - 1.0) * (Z1 - Z0) +
                K1 * K1 * Z1 - K0 * K0 * Z0);
    }

    // Kurtosis: Sugiura & Gomi 1985 formula
    double kurtosis() const override {
        double K0 = (min_ - mu_) / sigma_;
        double K1 = (max_ - mu_) / sigma_;
        double Z = std_cdf(K1) - std_cdf(K0);
        if (Z == 0.0) return 3.0;
        double Z0 = std_pdf(K0) / Z;
        double Z1 = std_pdf(K1) / Z;
        double V = 1.0 - (K1 * Z1 - K0 * Z0) - (Z1 - Z0) * (Z1 - Z0);
        return (1.0 / (V * V)) *
               (-3.0 * std::pow(Z1 - Z0, 4) -
                6.0 * (K1 * Z1 - K0 * Z0) * (Z1 - Z0) * (Z1 - Z0) -
                2.0 * (Z1 - Z0) * (Z1 - Z0) -
                4.0 * (K1 * K1 * Z1 - K0 * K0 * Z0) * (Z1 - Z0) -
                3.0 * (K1 * Z1 - K0 * Z0) -
                (K1 * K1 * K1 * Z1 - K0 * K0 * K0 * Z0) + 3.0);
    }

    double minimum() const override { return min_; }
    double maximum() const override { return max_; }

    // --- Distribution functions ---
    // PDF(x) = StandardPDF((x-mu)/sigma) / (sigma * Z)
    double pdf(double x) const override {
        if (x < min_ || x > max_) return 0.0;
        double Xi = (x - mu_) / sigma_;
        double alph = (min_ - mu_) / sigma_;
        double bet = (max_ - mu_) / sigma_;
        double Z = std_cdf(bet) - std_cdf(alph);
        if (Z == 0.0) return std_pdf(Xi);
        return std_pdf(Xi) / (sigma_ * Z);
    }

    // CDF(x) = (Phi((x-mu)/sigma) - Phi(alpha)) / Z
    double cdf(double x) const override {
        if (x <= min_) return 0.0;
        if (x >= max_) return 1.0;
        double Xi = (x - mu_) / sigma_;
        double alph = (min_ - mu_) / sigma_;
        double bet = (max_ - mu_) / sigma_;
        double Z = std_cdf(bet) - std_cdf(alph);
        if (Z == 0.0) return std_cdf(Xi);
        return (std_cdf(Xi) - std_cdf(alph)) / Z;
    }

    // InverseCDF: mu + sigma * StandardZ(Phi(alpha) + p * Z)
    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return min_;
        if (probability == 1.0) return max_;
        double alph = (min_ - mu_) / sigma_;
        double bet = (max_ - mu_) / sigma_;
        double Z = std_cdf(bet) - std_cdf(alph);
        if (Z == 0.0) return mu_ + sigma_ * std_z(probability);
        return mu_ + sigma_ * std_z(std_cdf(alph) + probability * Z);
    }

    // --- Parameter display names (X1; C# TruncatedNormal.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Mean (\xC2\xB5)", "Std Dev (\xCF\x83)", "Min", "Max"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xC2\xB5", "\xCF\x83", "Min", "Max"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<TruncatedNormal>(mu_, sigma_, min_, max_);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            auto mom = data::product_moments(sample);
            double mn = *std::min_element(sample.begin(), sample.end());
            double mx = *std::max_element(sample.begin(), sample.end());
            set_parameters(mom[0], mom[1], mn, mx);
        } else {
            throw std::invalid_argument("TruncatedNormal only supports MethodOfMoments");
        }
    }

    // --- Standard Normal helpers (mirrors C# Normal.StandardCDF / StandardPDF / StandardZ) ---
    static double std_cdf(double x) {
        return 0.5 * (1.0 + std::erf(x / kSqrt2));
    }

    static double std_pdf(double x) {
        return std::exp(-0.5 * x * x) / kSqrt2PI;
    }

    static double std_z(double p) {
        return r8_normal_01_cdf_inverse(p);
    }

   private:
    static bool validate(double mean, double sigma, double min_val, double max_val) {
        if (std::isnan(mean) || std::isinf(mean)) return false;
        if (std::isnan(sigma) || std::isinf(sigma) || sigma <= 0.0) return false;
        if (std::isnan(min_val) || std::isnan(max_val) ||
            std::isinf(min_val) || std::isinf(max_val) || min_val > max_val) return false;
        return true;
    }

    // Wichura AS241 standard-normal quantile
    static double r8_normal_01_cdf_inverse(double p) {
        static const double a[8] = {3.3871328727963666080,    1.3314166789178437745e+2,
                                     1.9715909503065514427e+3,  1.3731693765509461125e+4,
                                     4.5921953931549871457e+4,  6.7265770927008700853e+4,
                                     3.3430575583588128105e+4,  2.5090809287301226727e+3};
        static const double b[8] = {1.0,                       4.2313330701600911252e+1,
                                     6.8718700749205790830e+2,  5.3941960214247511077e+3,
                                     2.1213794301586595867e+4,  3.9307895800092710610e+4,
                                     2.8729085735721942674e+4,  5.2264952788528545610e+3};
        static const double c[8] = {1.42343711074968357734,    4.63033784615654529590,
                                     5.76949722146069140550,    3.64784832476320460504,
                                     1.27045825245236838258,    2.41780725177450611770e-1,
                                     2.27238449892691845833e-2, 7.74545014278341407640e-4};
        static const double d[8] = {1.0,                       2.05319162663775882187,
                                     1.67638483018380384940,    6.89767334985100004550e-1,
                                     1.48103976427480074590e-1, 1.51986665636164571966e-2,
                                     5.47593808499534494600e-4, 1.05075007164441684324e-9};
        static const double e[8] = {6.65790464350110377720,    5.46378491116411436990,
                                     1.78482653991729133580,    2.96560571828504891230e-1,
                                     2.65321895265761230930e-2, 1.24266094738807843860e-3,
                                     2.71155556874348757815e-5, 2.01033439929228813265e-7};
        static const double f[8] = {1.0,                       5.99832206555887937690e-1,
                                     1.36929880922735805310e-1, 1.48753612908506148525e-2,
                                     7.86869131145613259100e-4, 1.84631831751005468180e-5,
                                     1.42151175831644588870e-7, 2.04426310338993978564e-15};
        if (p <= 0.0) return -kInf;
        if (p >= 1.0) return kInf;
        auto poly = [](const double* coeffs, int n, double x) {
            double v = 0.0;
            for (int i = n - 1; 0 <= i; --i) v = v * x + coeffs[i];
            return v;
        };
        double q = p - 0.5;
        double r, value;
        if (std::fabs(q) <= 0.425) {
            r = 0.180625 - q * q;
            value = q * poly(a, 8, r) / poly(b, 8, r);
        } else {
            r = q < 0.0 ? p : 1.0 - p;
            r = std::sqrt(-std::log(r));
            if (r <= 5.0) {
                r -= 1.6;
                value = poly(c, 8, r) / poly(d, 8, r);
            } else {
                r -= 5.0;
                value = poly(e, 8, r) / poly(f, 8, r);
            }
            if (q < 0.0) value = -value;
        }
        return value;
    }

    double mu_ = 0.5;
    double sigma_ = 0.2;
    double min_ = 0.0;
    double max_ = 1.0;
};

}  // namespace corehydro::numerics::distributions
