// ported from: Numerics/Distributions/Univariate/Normal.cs @ <pending-sha>
//
// The Normal (Gaussian) distribution, parameters µ (location) and σ (scale).
// CDF uses std::erf (matches the C# Erf.Function = regularized lower-incomplete gamma to
// well within oracle tolerances); InverseCDF ports Wichura's AS241 (r8_normal_01_cdf_inverse)
// verbatim, the same routine the C# uses. Logic mirrors the C# source method-for-method;
// the WPF confidence-interval helpers (Normal/NoncentralT/MonteCarlo) are not ported.
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
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::distributions {

class Normal : public UnivariateDistributionBase,
               public IEstimation,
               public ILinearMomentEstimation {
   public:
    Normal() { set_parameters(0.0, 1.0); }
    explicit Normal(double mean) { set_parameters(mean, 1.0); }
    Normal(double mean, double standard_deviation) { set_parameters(mean, standard_deviation); }

    double mu() const { return mu_; }
    double sigma() const { return sigma_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override { return UnivariateDistributionType::Normal; }
    int number_of_parameters() const override { return 2; }
    std::vector<double> get_parameters() const override { return {mu_, sigma_}; }

    void set_parameters(double location, double scale) {
        if (scale < 1E-16 && std::signbit(scale) == false) scale = 1E-16;
        mu_ = location;
        sigma_ = scale;
        parameters_valid_ = validate(location, scale);
    }
    void set_parameters(const std::vector<double>& p) override { set_parameters(p[0], p[1]); }

    // --- Moments / support ---
    double mean() const override { return mu_; }
    double median() const override { return mu_; }
    double mode() const override { return mu_; }
    double standard_deviation() const override { return sigma_; }
    double skewness() const override { return 0.0; }
    double kurtosis() const override { return 3.0; }
    double minimum() const override { return -kInf; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---
    double pdf(double x) const override {
        double z = (x - mu_) / sigma_;
        return std::exp(-0.5 * z * z) / (kSqrt2PI * sigma_);
    }

    double cdf(double x) const override {
        return 0.5 * (1.0 + std::erf((x - mu_) / (sigma_ * kSqrt2)));
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        return mu_ + sigma_ * standard_z(probability);
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Normal>(mu_, sigma_);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            set_parameters(data::product_moments(sample));  // {mean, sd, ...}; first two used
        } else if (method == ParameterEstimationMethod::MethodOfLinearMoments) {
            set_parameters(parameters_from_linear_moments(data::linear_moments(sample)));
        } else {
            set_parameters(mle(sample));
        }
    }

    std::vector<double> parameters_from_linear_moments(
        const std::vector<double>& moments) const override {
        double mu = moments[0];
        double sigma = moments[1] * std::sqrt(kPi);
        return {mu, sigma};
    }

    std::vector<double> linear_moments_from_parameters(
        const std::vector<double>& parameters) const override {
        double L1 = parameters[0];
        double L2 = parameters[1] * std::pow(kPi, -0.5);
        double T3 = 0.0;
        double T4 = 30.0 * std::pow(kPi, -1.0) * std::atan(kSqrt2) - 9.0;
        return {L1, L2, T3, T4};
    }

    void get_parameter_constraints(const std::vector<double>& sample, std::vector<double>& initials,
                                   std::vector<double>& lowers,
                                   std::vector<double>& uppers) const {
        auto moments = data::product_moments(sample);
        initials = {moments[0], moments[1]};
        lowers.assign(2, 0.0);
        uppers.assign(2, 0.0);
        if (initials[0] == 0.0) initials[0] = kDoubleMachineEpsilon;
        double locExp = std::ceil(std::log10(std::fabs(initials[0])) + 1.0);
        lowers[0] = -std::pow(10.0, locExp);
        uppers[0] = std::pow(10.0, locExp);
        lowers[1] = kDoubleMachineEpsilon;
        uppers[1] = std::pow(10.0, std::ceil(std::log10(initials[1]) + 1.0));
    }

    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            Normal n;
            n.set_parameters(x[0], x[1]);
            return n.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 2, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }

    // --- Standard normal helpers (static, mirror the C# public API) ---

    // Standard normal CDF Φ(z). Mirrors Normal.StandardCDF(Z) in C#.
    static double standard_cdf(double z) {
        return 0.5 * (1.0 + std::erf(z / kSqrt2));
    }

    // Z variate of the standard normal for a probability (Wichura AS241).
    static double standard_z(double probability) {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        return r8_normal_01_cdf_inverse(probability);
    }

   private:
    static bool validate(double location, double scale) {
        if (std::isnan(location) || std::isinf(location)) return false;
        if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0) return false;
        return true;
    }

    // R8POLY_VALUE evaluates a polynomial p(x) = a[0] + a[1] x + ... + a[n-1] x^(n-1).
    static double r8poly_value(int n, const double a[], double x) {
        double value = 0.0;
        for (int i = n - 1; 0 <= i; --i) value = value * x + a[i];
        return value;
    }

    // R8_NORMAL_01_CDF_INVERSE inverts the standard normal CDF (Wichura, AS241, 1988).
    // Accurate to ~1 part in 1e16. Original FORTRAN77 by Michael Wichura; C++ by John Burkardt.
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
        if (1.0 <= p) return kInf;

        double q = p - 0.5;
        double r, value;
        if (std::fabs(q) <= 0.425) {
            r = 0.180625 - q * q;
            value = q * r8poly_value(8, a, r) / r8poly_value(8, b, r);
        } else {
            r = q < 0.0 ? p : 1.0 - p;
            r = std::sqrt(-std::log(r));
            if (r <= 5.0) {
                r = r - 1.6;
                value = r8poly_value(8, c, r) / r8poly_value(8, d, r);
            } else {
                r = r - 5.0;
                value = r8poly_value(8, e, r) / r8poly_value(8, f, r);
            }
            if (q < 0.0) value = -value;
        }
        return value;
    }

    double mu_ = 0.0;
    double sigma_ = 1.0;
};

}  // namespace bestfit::numerics::distributions
