// ported from: Numerics/Distributions/Univariate/PearsonTypeIII.cs @ a2c4dbf
//
// Pearson Type III distribution parameterized by mean µ, standard deviation σ,
// and skew γ. Reparameterizes to a shifted Gamma distribution. Mirrors the C# source
// method-for-method. QuantileVariance/ParameterCovariance/Bootstrap
// (desktop / uncertainty-analysis concerns) are not ported. B4 adds
// ParametersFromMoments/MomentsFromParameters, QuantileGradientForMoments, and the
// ConditionalMoments override for the Bulletin 17C GMM track.
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_linear_moment_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/gamma_distribution.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/math/special/factorial.hpp"
#include "corehydro/numerics/math/special/gamma.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

namespace sf = corehydro::numerics::math::special;

class PearsonTypeIII : public UnivariateDistributionBase,
                       public IEstimation,
                       public ILinearMomentEstimation,
                       public IMaximumLikelihoodEstimation {
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

    // --- Parameter display names (X1; C# PearsonTypeIII.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Mean (\xC2\xB5)", "Std Dev (\xCF\x83)", "Skew (\xCE\xB3)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xC2\xB5", "\xCF\x83", "\xCE\xB3"};
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

    // ParametersFromMoments (C# PearsonTypeIII.cs:389): the PT3 is parameterized by its
    // first three moments (C# moments.ToArray().Subset(0, 2)).
    std::vector<double> parameters_from_moments(const std::vector<double>& moments) const {
        return {moments[0], moments[1], moments[2]};
    }

    // MomentsFromParameters (C# PearsonTypeIII.cs:395): {Mean, StandardDeviation,
    // Skewness, Kurtosis} of a PT3 built from the parameters.
    std::vector<double> moments_from_parameters(const std::vector<double>& parameters) const {
        PearsonTypeIII dist;
        dist.set_parameters(parameters);
        double m1 = dist.mean();
        double m2 = dist.standard_deviation();
        double m3 = dist.skewness();
        double m4 = dist.kurtosis();
        return {m1, m2, m3, m4};
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
        return {L1, L2, T3, T4};
    }

    void get_parameter_constraints(const std::vector<double>& sample,
                                   std::vector<double>& initials,
                                   std::vector<double>& lowers,
                                   std::vector<double>& uppers) const override {
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

    // Returns a list of partial derivatives of X given probability with respect to each
    // moment (C# PearsonTypeIII.QuantileGradientForMoments, line 774). Q(p) = mu +
    // sigma*Kp(skew, p), so the gradient is {1, Kp, sigma*dKp/dskew}. C#
    // ValidateParameters(..., true) throw -> std::invalid_argument.
    std::vector<double> quantile_gradient_for_moments(double probability) const {
        // Validate parameters
        if (!parameters_valid_) throw std::invalid_argument("PearsonTypeIII: invalid parameters");
        return {
            1.0,
            GammaDistribution::frequency_factor_kp(gamma_, probability),
            sigma_ * GammaDistribution::partial_kp(gamma_, probability)
        };
    }

    // ConditionalMoments override (C# PearsonTypeIII.cs:820): delegates to the smooth
    // Normal <-> Wilson-Hilferty <-> Exact blend below; NaN quadruple on hard numerical
    // failure or when the window carries no probability mass.
    std::vector<double> conditional_moments(double a, double b) const override {
        if (a >= b) return {kNaN, kNaN, kNaN, kNaN};

        std::vector<double> m_pearson;
        auto pearson = try_pearson_conditional_moments(a, b, m_pearson, 1e-15);
        if (!pearson.first || !pearson.second) return {kNaN, kNaN, kNaN, kNaN};
        return m_pearson;
    }

   private:
    // Pearson Type III conditional moments with a smooth Normal <-> WH <-> Exact blend
    // (C# TryPearsonConditionalMoments). Fills `moments` with [E[X], m2, m3, m4] where
    // mk are central moments about the **unconditional** mean mu0. Returns
    // {stable, divisable}: {false, false} on hard numerical failure (guard), {true,
    // false} when probability mass was too small, {false, true} when an exception was
    // swallowed (mirrors the C# catch), else {true, true}.
    std::pair<bool, bool> try_pearson_conditional_moments(double a, double b,
                                                          std::vector<double>& moments,
                                                          double p_min) const {
        namespace fct = math::special::factorial;
        moments.clear();

        // ---- small math helpers -------------------------------------------------
        auto Phi = [](double x) { return 0.5 * (1.0 + std::erf(x / kSqrt2)); };
        auto phi = [](double x) { return std::exp(-0.5 * x * x) / kSqrt2PI; };

        // Standardized truncated-normal integrals J_0..J_N with recursion
        auto trunc_std_normal_j = [&](double a_star, double b_star, int N, double* J) {
            bool a_neg_inf = std::isinf(a_star) && a_star < 0.0;
            bool b_pos_inf = std::isinf(b_star) && b_star > 0.0;
            double phiA = a_neg_inf ? 0.0 : phi(a_star);
            double phiB = b_pos_inf ? 0.0 : phi(b_star);
            double PhiA = a_neg_inf ? 0.0 : Phi(a_star);
            double PhiB = b_pos_inf ? 1.0 : Phi(b_star);

            J[0] = PhiB - PhiA;  // D
            if (N == 0) return;
            J[1] = phiA - phiB;  // E[T] * D
            for (int n = 2; n <= N; n++) {
                double termA = a_neg_inf ? 0.0 : std::pow(a_star, n - 1) * phiA;
                double termB = b_pos_inf ? 0.0 : std::pow(b_star, n - 1) * phiB;
                J[n] = termA - termB + (n - 1) * J[n - 2];  // E[T^n] * D
            }
        };

        // E[(mu + sigma*T)^n | a* <= T <= b*], given standardized J's
        auto ez_pow = [](int n, double mu, double sigma, const double* J) {
            double D = J[0];
            if (!(D > 0)) return kNaN;

            double sum = 0.0;
            for (int j = 0; j <= n; j++) {
                double bin = fct::binomial_coefficient(n, j);
                double Ej = J[j] / D;  // E[T^j | trunc]
                sum += bin * std::pow(mu, n - j) * std::pow(sigma, j) * Ej;
            }
            return sum;
        };

        // smooth logistic weights (differentiable)
        auto sigmoid = [](double x, double x0, double s) {
            return 1.0 / (1.0 + std::exp((x - x0) / s));
        };

        // ---- guard Gamma(alpha) -------------------------------------------------
        double lg_alpha = sf::log_gamma(alpha());
        if (std::isnan(lg_alpha) || std::isinf(lg_alpha)) return {false, false};

        try {
            // Map [a,b] in X to [L,U] in Y for unit-scale Gamma Y ~ Gamma(Alpha, 1)
            bool pos = beta() > 0.0;
            double abs_beta = std::fabs(beta());

            double L, U;
            if (pos) {
                L = std::max(0.0, (a - xi()) / abs_beta);
                U = (std::isinf(b) && b > 0.0) ? kInf : std::max(0.0, (b - xi()) / abs_beta);
            } else {
                L = std::max(0.0, (xi() - b) / abs_beta);
                U = (std::isinf(a) && a > 0.0) ? kInf : std::max(0.0, (xi() - a) / abs_beta);
            }

            // =====================================================================
            // (A) Exact truncated-Gamma path (numerically stable centralization)
            // =====================================================================
            bool exact_ok = true;
            double EX_exact = kNaN, m2_exact = kNaN, m3_exact = kNaN, m4_exact = kNaN;

            auto cdf_gamma = [&](int r, double x) {
                double y = (pos ? (x - xi()) : (xi() - x)) / abs_beta;
                double P = sf::lower_incomplete(alpha() + r, y);
                return pos ? P : (1.0 - P);
            };

            double EY[5] = {0.0, 0.0, 0.0, 0.0, 0.0};  // E[Y^r | L<=Y<=U], r=0..4
            {
                double dP[5];
                for (int r = 0; r <= 4; r++) dP[r] = cdf_gamma(r, b) - cdf_gamma(r, a);

                double P0 = dP[0];
                if (!(P0 > p_min)) {
                    exact_ok = false;
                } else {
                    double gr[5];  // Gamma(alpha+r)/Gamma(alpha)
                    for (int r = 0; r <= 4; r++)
                        gr[r] = std::exp(sf::log_gamma(alpha() + r) - lg_alpha);

                    for (int r = 0; r <= 4; r++)
                        EY[r] = gr[r] * (dP[r] / P0);  // truncated raw moments of Y
                }
            }

            if (exact_ok) {
                double EY1 = EY[1], EY2 = EY[2], EY3 = EY[3], EY4 = EY[4];
                EX_exact = xi() + beta() * EY1;

                // conditional central moments of Y about its conditional mean
                double c2Y = std::max(0.0, EY2 - EY1 * EY1);
                double c3Y = EY3 - 3.0 * EY1 * EY2 + 2.0 * EY1 * EY1 * EY1;
                double c4Y = EY4 - 4.0 * EY1 * EY3 + 6.0 * EY1 * EY1 * EY2 -
                             3.0 * std::pow(EY1, 4);

                // shift to moments about alpha (unconditional mean of Y)
                double dY = EY1 - alpha();

                double b1 = beta();
                double b2 = b1 * b1;
                double b3 = b2 * b1;
                double b4 = b3 * b1;
                m2_exact = b2 * (c2Y + dY * dY);
                m3_exact = b3 * (c3Y + 3.0 * dY * c2Y + dY * dY * dY);
                m4_exact = b4 * (c4Y + 4.0 * dY * c3Y + 6.0 * dY * dY * c2Y + std::pow(dY, 4));
            }

            // =====================================================================
            // (B) Wilson-Hilferty (Gamma ~ Z^3) truncated-Normal path
            // =====================================================================
            bool wh_ok = true;
            double EX_wh = kNaN, m2_wh = kNaN, m3_wh = kNaN, m4_wh = kNaN;

            if (!(alpha() > 0.0) || std::isinf(alpha()) || std::isnan(alpha())) {
                wh_ok = false;
            } else {
                // Z ~ N(muZ, sigZ^2); Y ~ Alpha * Z^3
                double muZ = 1.0 - 1.0 / (9.0 * alpha());
                double sigZ = std::sqrt(1.0 / (9.0 * alpha()));

                double ell = (L <= 0.0) ? 0.0 : std::pow(L / alpha(), 1.0 / 3.0);
                double uuu = (std::isinf(U) && U > 0.0)
                                 ? kInf
                                 : (U <= 0.0 ? 0.0 : std::pow(U / alpha(), 1.0 / 3.0));

                double a_star_z = (std::isinf(ell) && ell < 0.0) ? -kInf : (ell - muZ) / sigZ;
                double b_star_z = (std::isinf(uuu) && uuu > 0.0) ? kInf : (uuu - muZ) / sigZ;

                double Jz[13];
                trunc_std_normal_j(a_star_z, b_star_z, 12, Jz);
                double Dz = Jz[0];
                if (!(Dz > p_min) || std::isnan(Dz)) {
                    wh_ok = false;
                } else {
                    // Moments of Z we need
                    double EZ3 = ez_pow(3, muZ, sigZ, Jz);
                    double EZ6 = ez_pow(6, muZ, sigZ, Jz);
                    double EZ9 = ez_pow(9, muZ, sigZ, Jz);
                    double EZ12 = ez_pow(12, muZ, sigZ, Jz);

                    if (!is_finite(EZ3) || !is_finite(EZ6) || !is_finite(EZ9) ||
                        !is_finite(EZ12)) {
                        wh_ok = false;
                    } else {
                        // Mean
                        EX_wh = xi() + beta() * (alpha() * EZ3);

                        // Central moments about mu0 using W = Z^3 - 1
                        double EW2 = (EZ6 - 2.0 * EZ3 + 1.0);
                        double EW3 = (EZ9 - 3.0 * EZ6 + 3.0 * EZ3 - 1.0);
                        double EW4 = (EZ12 - 4.0 * EZ9 + 6.0 * EZ6 - 4.0 * EZ3 + 1.0);

                        double BA = beta() * alpha();
                        double BA2 = BA * BA, BA3 = BA2 * BA, BA4 = BA3 * BA;
                        m2_wh = BA2 * EW2;
                        m3_wh = BA3 * EW3;
                        m4_wh = BA4 * EW4;
                    }
                }
            }

            // =====================================================================
            // (C) Normal small-skew path: Y ~ N(alpha, alpha) truncated to [L,U]
            // =====================================================================
            bool normal_ok = true;
            double EX_n = kNaN, m2_n = kNaN, m3_n = kNaN, m4_n = kNaN;

            {
                if (!(alpha() > 0.0) || std::isnan(alpha()) || std::isinf(alpha())) {
                    normal_ok = false;
                } else {
                    double s = std::sqrt(alpha());
                    double a_star = (L - alpha()) / s;
                    double b_star = (std::isinf(U) && U > 0.0) ? kInf : (U - alpha()) / s;

                    double J[5];
                    trunc_std_normal_j(a_star, b_star, 4, J);
                    double D = J[0];
                    if (!(D > p_min)) {
                        normal_ok = false;
                    } else {
                        // E[T^m]
                        double ET[5];
                        for (int m = 0; m <= 4; m++) ET[m] = J[m] / D;

                        // E[Y^j], j=0..4 with Y = alpha + s T
                        double EYnorm[5];
                        EYnorm[0] = 1.0;
                        for (int j = 1; j <= 4; j++) {
                            double sum = 0.0;
                            for (int m = 0; m <= j; m++) {
                                double bin = fct::binomial_coefficient(j, m);
                                sum += bin * std::pow(alpha(), j - m) * std::pow(s, m) * ET[m];
                            }
                            EYnorm[j] = sum;
                        }

                        EX_n = xi() + beta() * EYnorm[1];

                        // central moments about alpha for Y
                        double CY[5];
                        CY[0] = 1.0;
                        CY[1] = 0.0;  // centered at alpha
                        for (int k = 2; k <= 4; k++) {
                            double sum = 0.0;
                            for (int m = 0; m <= k; m++) {
                                double bin = fct::binomial_coefficient(k, m);
                                sum += bin * std::pow(-alpha(), k - m) * EYnorm[m];
                            }
                            CY[k] = sum;
                        }

                        double b1 = beta(), b2 = b1 * b1, b3 = b2 * b1, b4 = b3 * b1;
                        m2_n = b2 * CY[2];
                        m3_n = b3 * CY[3];
                        m4_n = b4 * CY[4];

                        if (!(is_finite(EX_n) && is_finite(m2_n) && is_finite(m3_n) &&
                              is_finite(m4_n)))
                            normal_ok = false;
                    }
                }
            }

            // If no path has usable mass, return divisable=false
            if (!exact_ok && !wh_ok && !normal_ok) return {true, false};

            // =====================================================================
            // (D) Smooth weights and blend
            // =====================================================================
            // Unconditional skew magnitude for Gamma(alpha,.): |gamma1| = 2 / sqrt(alpha)
            double abs_gamma1 = 2.0 / std::sqrt(alpha());

            // WH <-> Exact around 0.10 (scale 0.03)
            double wWH = sigmoid(abs_gamma1, 0.10, 0.03);  // near 1 at very small skew (favor WH)
            if (!wh_ok) wWH = 0.0;
            if (!exact_ok) wWH = 1.0;

            double EX_we = wWH * EX_wh + (1.0 - wWH) * EX_exact;
            double m2_we = wWH * m2_wh + (1.0 - wWH) * m2_exact;
            double m3_we = wWH * m3_wh + (1.0 - wWH) * m3_exact;
            double m4_we = wWH * m4_wh + (1.0 - wWH) * m4_exact;

            // Normal small-skew vs (WH/Exact) around 1e-2 (narrow scale)
            double wN = sigmoid(abs_gamma1, 1e-3, 3e-4);  // near 1 below ~1e-3 (favor Normal)
            if (!normal_ok) wN = 0.0;

            double EX = wN * EX_n + (1.0 - wN) * EX_we;
            double m2 = wN * m2_n + (1.0 - wN) * m2_we;
            double m3 = wN * m3_n + (1.0 - wN) * m3_we;
            double m4 = wN * m4_n + (1.0 - wN) * m4_we;

            moments = {EX, m2, m3, m4};
            return {true, true};
        } catch (...) {
            // C# swallows any exception here (silent no-throw guard, per the port rule)
            return {false, true};
        }
    }

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

}  // namespace corehydro::numerics::distributions
