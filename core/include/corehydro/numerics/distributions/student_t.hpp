// ported from: Numerics/Distributions/Univariate/StudentT.cs @ a2c4dbf
//
// The Student's t distribution with parameters µ (location), σ (scale), and ν (degrees of freedom).
// PDF via log-gamma formula; CDF via regularized incomplete beta with the t→beta argument transform
// betaX = (Z + sqrt(Z²+ν)) / (2·sqrt(Z²+ν)); InverseCDF via two Beta.IncompleteInverse paths split
// at p=0.25/0.75 matching the Accord Math Library algorithm (mirrored from C# source).
// No estimation interfaces upstream. Logic mirrors the C# source method-for-method.
//
// CDF/InverseCDF beta transform notes (both copied from Accord Math Library via C# source):
//   CDF: for nu < 1e8, transform Z=(x-mu)/sigma to betaX as above; feed Beta.Incomplete(nu/2, nu/2, betaX).
//   InverseCDF mid path (0.25 < p < 0.75): uses IncompleteInverse(0.5, nu/2, |1-2p|), t=sqrt(nu*z/(1-z)).
//   InverseCDF tail path (p <= 0.25 or p >= 0.75): uses IncompleteInverse(nu/2, 0.5, 2*p_adj), t=sqrt(nu/z - nu).
#pragma once
#include <string>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/math/special/beta.hpp"
#include "corehydro/numerics/math/special/gamma.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

namespace sf_st = corehydro::numerics::math::special;

class StudentT : public UnivariateDistributionBase {
   public:
    // Constructs a standard Student's t distribution with 10 degrees of freedom (mu=0, sigma=1).
    StudentT() { set_parameters(0.0, 1.0, 10.0); }

    // Constructs a standard Student's t with given degrees of freedom (mu=0, sigma=1).
    explicit StudentT(double degrees_of_freedom) { set_parameters(0.0, 1.0, degrees_of_freedom); }

    // Constructs a Student's t distribution with given location, scale, and degrees of freedom.
    StudentT(double mu, double sigma, double degrees_of_freedom) {
        set_parameters(mu, sigma, degrees_of_freedom);
    }

    double mu() const { return mu_; }
    double sigma() const { return sigma_; }
    double degrees_of_freedom() const { return nu_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::StudentT;
    }
    int number_of_parameters() const override { return 3; }
    std::vector<double> get_parameters() const override { return {mu_, sigma_, nu_}; }

    void set_parameters(double mu, double sigma, double v) {
        mu_ = mu;
        sigma_ = sigma;
        nu_ = v;
        parameters_valid_ = validate(mu, sigma, v);
    }
    void set_parameters(const std::vector<double>& p) override {
        set_parameters(p[0], p[1], p[2]);
    }

    // --- Moments / support ---

    // Mean exists only when nu > 1; otherwise NaN (C# returns double.NaN).
    double mean() const override {
        if (nu_ > 1.0) return mu_;
        return kNaN;
    }

    // Median = mu (always, per C#).
    double median() const override { return mu_; }

    // Mode = mu (always, per C#).
    double mode() const override { return mu_; }

    // SD = sqrt(sigma^2 * nu / (nu-2)) for nu>2; +Inf for 1<nu<=2; NaN for nu<=1.
    double standard_deviation() const override {
        if (nu_ > 2.0) return std::sqrt(sigma_ * sigma_ * nu_ / (nu_ - 2.0));
        if (nu_ > 1.0) return kInf;
        return kNaN;
    }

    // Skewness = 0 for nu>3; NaN otherwise.
    double skewness() const override {
        if (nu_ > 3.0) return 0.0;
        return kNaN;
    }

    // Kurtosis = 3 + 6/(nu-4) for nu>4; +Inf for 2<nu<=4; NaN for nu<=2.
    double kurtosis() const override {
        if (nu_ > 4.0) return 3.0 + 6.0 / (nu_ - 4.0);
        if (nu_ > 2.0) return kInf;
        return kNaN;
    }

    double minimum() const override { return -kInf; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---

    // PDF: for nu >= 1e8, delegate to standard normal PDF. Otherwise use log-gamma formula.
    // Note: the C# implementation does NOT divide by sigma when sigma != 1; it uses Z=(x-mu)/sigma
    // in the formula and returns T_nu(Z) — the standard t density evaluated at Z, not T_nu(Z)/sigma.
    // This mirrors the C# source exactly (verified against Test_StudentT.cs oracle values).
    double pdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("StudentT: invalid parameters");
        double Z = (x - mu_) / sigma_;
        if (nu_ >= 1.0e8) {
            // Large-df fallback: standard normal PDF at Z (mirrors C# Normal.StandardPDF(Z) — no /sigma)
            return std::exp(-0.5 * Z * Z) / kSqrt2PI;
        }
        return std::exp(sf_st::log_gamma((nu_ + 1.0) / 2.0) -
                        sf_st::log_gamma(nu_ / 2.0)) *
               std::pow(1.0 + Z * Z / nu_, -0.5 * (nu_ + 1.0)) /
               std::sqrt(nu_ * kPi);
    }

    // CDF: for nu >= 1e8, delegate to standard normal CDF. Otherwise use the incomplete beta
    // transform: betaX = (Z + sqrt(Z^2+nu)) / (2*sqrt(Z^2+nu)), result = Beta.Incomplete(nu/2, nu/2, betaX).
    double cdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("StudentT: invalid parameters");
        double Z = (x - mu_) / sigma_;
        if (nu_ >= 1.0e8) {
            return standard_z_cdf(Z);
        }
        double inner = std::sqrt(Z * Z + nu_);
        double betaX = (Z + inner) / (2.0 * inner);
        return sf_st::beta::incomplete(nu_ / 2.0, nu_ / 2.0, betaX);
    }

    // InverseCDF: implements the Accord Math Library algorithm mirrored from the C# source.
    // Mid-range path (0.25 < p < 0.75): IncompleteInverse(0.5, nu/2, |1-2p|), t = sqrt(nu*z/(1-z)).
    // Tail path (p <= 0.25 or p >= 0.75): reflect if p > 0.5, then IncompleteInverse(nu/2, 0.5, 2p),
    //   t = sqrt(nu/z - nu); guard against overflow per the C# double.MaxValue * z < nu check.
    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (!parameters_valid_) throw std::invalid_argument("StudentT: invalid parameters");

        if (probability > 0.25 && probability < 0.75) {
            if (probability == 0.5) return mu_;
            double z = sf_st::beta::incomplete_inverse(
                0.5, 0.5 * nu_, std::fabs(1.0 - 2.0 * probability));
            double t = std::sqrt(nu_ * z / (1.0 - z));
            if (probability < 0.5) t = -t;
            return mu_ + sigma_ * t;
        } else {
            int rflg = -1;
            double p = probability;
            if (p >= 0.5) {
                p = 1.0 - p;
                rflg = 1;
            }
            double z = sf_st::beta::incomplete_inverse(0.5 * nu_, 0.5, 2.0 * p);
            double t = std::sqrt(nu_ / z - nu_);
            if (std::numeric_limits<double>::max() * z < nu_) {
                // Overflow guard: return rflg * double.MaxValue (no mu, no sigma — mirrors C# exactly)
                return static_cast<double>(rflg) * std::numeric_limits<double>::max();
            }
            return mu_ + sigma_ * static_cast<double>(rflg) * t;
        }
    }

    // --- Parameter display names (X1; C# StudentT.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Location (\xC2\xB5)", "Scale (\xCF\x83)", "Degrees of Freedom (\xCE\xBD)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xC2\xB5", "\xCF\x83", "\xCE\xBD"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<StudentT>(mu_, sigma_, nu_);
    }

    // --- Standard normal CDF helper (for large-df fallback, mirrors C# Normal.StandardCDF) ---
    static double standard_z_cdf(double z) {
        return 0.5 * (1.0 + std::erf(z / kSqrt2));
    }

   private:
    static bool validate(double mu, double sigma, double v) {
        if (std::isnan(mu) || std::isinf(mu)) return false;
        if (std::isnan(sigma) || std::isinf(sigma) || sigma <= 0.0) return false;
        if (v < 1.0) return false;
        return true;
    }

    double mu_ = 0.0;
    double sigma_ = 1.0;
    double nu_ = 10.0;
};

}  // namespace corehydro::numerics::distributions
