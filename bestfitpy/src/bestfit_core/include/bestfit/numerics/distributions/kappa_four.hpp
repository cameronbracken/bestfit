// ported from: Numerics/Distributions/Univariate/KappaFour.cs @ a2c4dbf
//
// Kappa-4 distribution parameterized by location ξ, scale α, shape κ, shape h.
// InverseCDF has closed form with limit branches for h→0 and κ→0 that mirror the C#
// exactly. Moments use numerical integration (stratified bins) matching C# CentralMoments.
// Special cases: h=-1 → Generalized Logistic; h=0 → GEV; h=1 → Generalized Pareto.
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

class KappaFour : public UnivariateDistributionBase,
                  public IEstimation,
                  public ILinearMomentEstimation {
   public:
    // Constructs a Kappa-4 distribution with ξ=100, α=10, κ=0, h=0.
    KappaFour() { set_parameters({100.0, 10.0, 0.0, 0.0}); }

    // Constructs a Kappa-4 distribution with given parameters.
    KappaFour(double location, double scale, double shape, double shape2) {
        set_parameters({location, scale, shape, shape2});
    }

    double xi()    const { return xi_; }
    double alpha() const { return alpha_; }
    double kappa() const { return kappa_; }
    double hondo() const { return hondo_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::KappaFour;
    }
    int number_of_parameters() const override { return 4; }
    std::vector<double> get_parameters() const override {
        return {xi_, alpha_, kappa_, hondo_};
    }

    void set_parameters(const std::vector<double>& p) override {
        parameters_valid_ = validate(p);
        xi_    = p[0];
        alpha_ = p[1];
        kappa_ = p[2];
        hondo_ = p[3];
        moments_computed_ = false;
    }

    // --- Moments / support (numerical integration, mirrors C# CentralMoments(1000)) ---
    double mean() const override {
        ensure_moments();
        return u_[0];
    }

    double median() const override { return inverse_cdf(0.5); }

    // Mode via Brent maximization (mirrors C# Mode property).
    double mode() const override {
        double lo = inverse_cdf(0.001);
        double hi = inverse_cdf(0.999);
        // Simple golden-section search for mode
        const double phi = 0.6180339887498949;
        double a = lo, b = hi;
        double c = b - phi * (b - a);
        double d = a + phi * (b - a);
        for (int i = 0; i < 200; i++) {
            if (pdf(c) < pdf(d)) {
                a = c;
            } else {
                b = d;
            }
            c = b - phi * (b - a);
            d = a + phi * (b - a);
        }
        return (a + b) / 2.0;
    }

    double standard_deviation() const override {
        ensure_moments();
        return u_[1];
    }

    double skewness() const override {
        ensure_moments();
        return u_[2];
    }

    double kurtosis() const override {
        ensure_moments();
        return u_[3];
    }

    // Minimum. Mirrors C# Minimum property.
    double minimum() const override {
        if (hondo_ <= 0.0 && kappa_ < 0.0) {
            return xi_ + alpha_ / kappa_;
        } else if (hondo_ > 0.0 && kappa_ != 0.0) {
            return xi_ + alpha_ / kappa_ * (1.0 - std::pow(hondo_, -kappa_));
        } else if (hondo_ > 0.0 && kappa_ == 0.0) {
            return xi_ + alpha_ * std::log(hondo_);
        } else if (hondo_ <= 0.0 && kappa_ >= 0.0) {
            return -kInf;
        }
        return kNaN;
    }

    // Maximum. Mirrors C# Maximum property.
    double maximum() const override {
        if (kappa_ <= 0.0) {
            return kInf;
        } else {
            return xi_ + alpha_ / kappa_;
        }
    }

    // --- Distribution functions ---
    double pdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("KappaFour: invalid parameters");
        if (x < minimum() || x > maximum()) return 0.0;
        double F = cdf(x);
        double y = (x - xi_) / alpha_;
        double yy = 1.0 - kappa_ * y;
        return (1.0 / alpha_) * std::pow(yy, 1.0 / kappa_ - 1.0) * std::pow(F, 1.0 - hondo_);
    }

    double cdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("KappaFour: invalid parameters");
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return 1.0;
        double y = (x - xi_) / alpha_;
        double yy = 1.0 - kappa_ * y;
        if (kappa_ != 0.0 && hondo_ != 0.0) {
            return std::pow(1.0 - hondo_ * std::pow(yy, 1.0 / kappa_), 1.0 / hondo_);
        } else if (kappa_ != 0.0 && hondo_ == 0.0) {
            return std::exp(-std::pow(yy, 1.0 / kappa_));
        } else if (kappa_ == 0.0 && hondo_ != 0.0) {
            return std::pow(1.0 - hondo_ * std::exp(-y), 1.0 / hondo_);
        } else {
            // kappa==0, hondo==0: Gumbel
            return std::exp(-std::exp(-y));
        }
    }

    // InverseCDF with limit branches for h→0 and κ→0. Mirrors C# InverseCDF.
    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (!parameters_valid_) throw std::invalid_argument("KappaFour: invalid parameters");

        if (kappa_ != 0.0 && hondo_ != 0.0) {
            // General case
            return xi_ + alpha_ / kappa_
                   * (1.0 - std::pow((1.0 - std::pow(probability, hondo_)) / hondo_, kappa_));
        } else if (kappa_ != 0.0 && hondo_ == 0.0) {
            // h→0 limit (GEV)
            return xi_ + (alpha_ / kappa_)
                   * (1.0 - std::pow(-std::log(probability), kappa_));
        } else if (kappa_ == 0.0 && hondo_ != 0.0) {
            // κ→0 limit
            return xi_ - alpha_ * std::log(1.0 - std::pow(probability, hondo_) / hondo_);
        } else {
            // κ=0, h=0 (Gumbel)
            return xi_ - alpha_ * std::log(-std::log(probability));
        }
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<KappaFour>(xi_, alpha_, kappa_, hondo_);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        if (method == ParameterEstimationMethod::MethodOfLinearMoments) {
            set_parameters(parameters_from_linear_moments(data::linear_moments(sample)));
        } else if (method == ParameterEstimationMethod::MaximumLikelihood) {
            set_parameters(mle(sample));
        } else {
            throw std::invalid_argument("KappaFour: unsupported estimation method");
        }
    }

    // ParametersFromLinearMoments: Newton-Raphson iteration, ported directly from Fortran
    // (Hosking, IBM Research Report RC20525, Version 3, August 1996). Mirrors C# exactly.
    std::vector<double> parameters_from_linear_moments(
        const std::vector<double>& moments) const override {
        double L1 = moments[0];
        double L2 = moments[1];
        double T3 = moments[2];
        double T4 = moments[3];

        constexpr double eps = 1e-6;
        constexpr int maxit = 20, maxsr = 10;

        if (L2 <= 0.0)
            throw std::invalid_argument("L-moments invalid.");
        if (std::fabs(T3) >= 1.0 || std::fabs(T4) >= 1.0)
            throw std::invalid_argument("L-moments invalid.");
        if (T4 <= (5.0 * T3 * T3 - 1.0) / 4.0)
            throw std::invalid_argument("L-moments invalid.");
        if (T4 >= (5.0 * T3 * T3 + 1.0) / 6.0)
            throw std::invalid_argument(
                "(TAU-3, TAU-4) lies above the Generalized Logistic.");

        // Initial values
        double G = (1.0 - 3.0 * T3) / (1.0 + T3);
        double H = 1.001;
        double Z = G + H * 0.725;
        double XDIST = 10.0, DIST = 0.0;
        double U1 = 0, U2 = 0, U3 = 0, U4 = 0;
        double ALAM2 = 0, ALAM3 = 0, ALAM4 = 0;
        double TAU3 = 0, TAU4 = 0;
        double E1 = 0, E2 = 0;
        double DEL1 = 0, DEL2 = 0;
        double XG = 0, XH = 0, XZ, RHH;
        double U1G, U2G, U3G, U4G, U1H, U2H, U3H, U4H;
        double DL2G, DL2H, DL3G, DL3H, DL4G, DL4H;
        double D11, D12, D21, D22, DET, H11, H12, H21, H22;
        double FACTOR;

        for (int i = 1; i < maxit; i++) {
            for (int j = 1; j <= maxsr; j++) {
                if (G > 53.0)
                    throw std::runtime_error(
                        "Iteration encountered numerical difficulties - overflow.");
                if (H <= 0.0) {
                    U1 = std::exp(sf::log_gamma(-1.0 / H - G) - sf::log_gamma(-1.0 / H + 1.0));
                    U2 = std::exp(sf::log_gamma(-2.0 / H - G) - sf::log_gamma(-2.0 / H + 1.0));
                    U3 = std::exp(sf::log_gamma(-3.0 / H - G) - sf::log_gamma(-3.0 / H + 1.0));
                    U4 = std::exp(sf::log_gamma(-4.0 / H - G) - sf::log_gamma(-4.0 / H + 1.0));
                } else {
                    U1 = std::exp(sf::log_gamma(1.0 / H) - sf::log_gamma(1.0 / H + 1.0 + G));
                    U2 = std::exp(sf::log_gamma(2.0 / H) - sf::log_gamma(2.0 / H + 1.0 + G));
                    U3 = std::exp(sf::log_gamma(3.0 / H) - sf::log_gamma(3.0 / H + 1.0 + G));
                    U4 = std::exp(sf::log_gamma(4.0 / H) - sf::log_gamma(4.0 / H + 1.0 + G));
                }
                ALAM2 = U1 - 2.0 * U2;
                ALAM3 = -U1 + 6.0 * U2 - 6.0 * U3;
                ALAM4 = U1 - 12.0 * U2 + 30.0 * U3 - 20.0 * U4;
                if (ALAM2 == 0.0)
                    throw std::runtime_error(
                        "Iteration encountered numerical difficulties - overflow.");
                TAU3 = ALAM3 / ALAM2;
                TAU4 = ALAM4 / ALAM2;
                E1 = TAU3 - T3;
                E2 = TAU4 - T4;

                DIST = std::max(std::fabs(E1), std::fabs(E2));
                if (DIST < XDIST) break;

                DEL1 *= 0.5;
                DEL2 *= 0.5;
                G = XG - DEL1;
                H = XH - DEL2;

                if (j == maxsr)
                    throw std::runtime_error(
                        "Iteration encountered numerical difficulties - overflow.");
            }

            if (DIST < eps) break;

            XG = G; XH = H; XZ = Z; XDIST = DIST;
            RHH = 1.0 / (H * H);

            if (H > 0.0) {
                U1G = -U1 * sf::digamma(1.0 / H + 1.0 + G);
                U2G = -U2 * sf::digamma(2.0 / H + 1.0 + G);
                U3G = -U3 * sf::digamma(3.0 / H + 1.0 + G);
                U4G = -U4 * sf::digamma(4.0 / H + 1.0 + G);
                U1H = RHH * (-U1G - U1 * sf::digamma(1.0 / H));
                U2H = 2.0 * RHH * (-U2G - U2 * sf::digamma(2.0 / H));
                U3H = 3.0 * RHH * (-U3G - U3 * sf::digamma(3.0 / H));
                U4H = 4.0 * RHH * (-U4G - U4 * sf::digamma(4.0 / H));
            } else {
                U1G = -U1 * sf::digamma(-1.0 / H - G);
                U2G = -U2 * sf::digamma(-2.0 / H - G);
                U3G = -U3 * sf::digamma(-3.0 / H - G);
                U4G = -U4 * sf::digamma(-4.0 / H - G);
                U1H = RHH * (-U1G - U1 * sf::digamma(-1.0 / H + 1.0));
                U2H = 2.0 * RHH * (-U2G - U2 * sf::digamma(-2.0 / H + 1.0));
                U3H = 3.0 * RHH * (-U3G - U3 * sf::digamma(-3.0 / H + 1.0));
                U4H = 4.0 * RHH * (-U4G - U4 * sf::digamma(-4.0 / H + 1.0));
            }

            DL2G = U1G - 2.0 * U2G;
            DL2H = U1H - 2.0 * U2H;
            DL3G = -U1G + 6.0 * U2G - 6.0 * U3G;
            DL3H = -U1H + 6.0 * U2H - 6.0 * U3H;
            DL4G = U1G - 12.0 * U2G + 30.0 * U3G - 20.0 * U4G;
            DL4H = U1H - 12.0 * U2H + 30.0 * U3H - 20.0 * U4H;
            D11 = (DL3G - TAU3 * DL2G) / ALAM2;
            D12 = (DL3H - TAU3 * DL2H) / ALAM2;
            D21 = (DL4G - TAU4 * DL2G) / ALAM2;
            D22 = (DL4H - TAU4 * DL2H) / ALAM2;
            DET = D11 * D22 - D12 * D21;
            H11 = D22 / DET;
            H12 = -D12 / DET;
            H21 = -D21 / DET;
            H22 = D11 / DET;
            DEL1 = E1 * H11 + E2 * H12;
            DEL2 = E1 * H21 + E2 * H22;

            G = XG - DEL1;
            H = XH - DEL2;
            Z = G + H * 0.725;

            FACTOR = 1.0;
            if (G <= -1.0) FACTOR = 0.8 * (XG + 1.0) / DEL1;
            if (H <= -1.0) FACTOR = std::min(FACTOR, 0.8 * (XH + 1.0) / DEL2);
            if (Z <= -1.0) FACTOR = std::min(FACTOR, 0.8 * (XZ + 1.0) / (XZ - Z));
            if (H <= 0.0 && G * H <= -1.0)
                FACTOR = std::min(FACTOR, 0.8 * (XG * XH + 1.0) / (XG * XH - G * H));
            if (FACTOR != 1.0) {
                DEL1 *= FACTOR;
                DEL2 *= FACTOR;
                G = XG - DEL1;
                H = XH - DEL2;
                Z = G + H * 0.725;
            }

            if (i == maxit - 1)
                throw std::runtime_error("Iterations failed to converge.");
        }

        double hondo_r = H;
        double kappa_r = G;
        double TEMP = sf::log_gamma(1.0 + G);
        if (TEMP > 170.0)
            throw std::runtime_error("Overflow in xi/alpha calculation.");
        double GAM = std::exp(TEMP);
        TEMP = (1.0 + G) * std::log(std::fabs(H));
        if (TEMP > 170.0)
            throw std::runtime_error("Overflow in xi/alpha calculation.");
        double HH = std::exp(TEMP);
        double alpha_r = L2 * G * HH / (ALAM2 * GAM);
        double xi_r = L1 - alpha_r / G * (1.0 - GAM * U1 / HH);

        return {xi_r, alpha_r, kappa_r, hondo_r};
    }

    // LinearMomentsFromParameters. Mirrors C# exactly, including kappa→0 guard.
    std::vector<double> linear_moments_from_parameters(
        const std::vector<double>& parameters) const override {
        double xi    = parameters[0];
        double alpha = parameters[1];
        double kappa = parameters[2];
        double hondo = parameters[3];

        if ((kappa < -1.0 && hondo >= 0.0) ||
            (hondo < 0.0 && (kappa <= -1.0 || kappa >= -1.0 / hondo))) {
            throw std::invalid_argument("L-moments can only be defined for hondo (h) >= 0 and "
                                        "kappa (k) > -1, or if h < 0 and -1 < k < -1/h.");
        }

        // Guard: mirror C# kappa == 0.0 → 1e-100
        if (kappa == 0.0) kappa = 1e-100;

        double L1, L2, T3, T4;
        if (hondo == 0.0) {
            L1 = xi + alpha * (1.0 - sf::function(1.0 + kappa)) / kappa;
            L2 = alpha * (1.0 - std::pow(2.0, -kappa)) * sf::function(1.0 + kappa) / kappa;
            T3 = 2.0 * (1.0 - std::pow(3.0, -kappa)) / (1.0 - std::pow(2.0, -kappa)) - 3.0;
            T4 = (5.0 * (1.0 - std::pow(4.0, -kappa)) - 10.0 * (1.0 - std::pow(3.0, -kappa))
                  + 6.0 * (1.0 - std::pow(2.0, -kappa)))
                 / (1.0 - std::pow(2.0, -kappa));
        } else {
            double g[4];
            for (int r = 1; r <= 4; r++) {
                if (hondo > 0.0) {
                    g[r - 1] = r * sf::function(1.0 + kappa) * sf::function(r / hondo)
                               / (std::pow(hondo, 1.0 + kappa)
                                  * sf::function(1.0 + kappa + r / hondo));
                } else {
                    g[r - 1] = r * sf::function(1.0 + kappa)
                               * sf::function(-kappa - r / hondo)
                               / (std::pow(-hondo, 1.0 + kappa)
                                  * sf::function(1.0 - r / hondo));
                }
            }
            L1 = xi + alpha * (1.0 - g[0]) / kappa;
            L2 = alpha * (g[0] - g[1]) / kappa;
            T3 = (-g[0] + 3.0 * g[1] - 2.0 * g[2]) / (g[0] - g[1]);
            T4 = -(-g[0] + 6.0 * g[1] - 10.0 * g[2] + 5.0 * g[3]) / (g[0] - g[1]);
        }
        return {L1, L2, T3, T4};
    }

    // GetParameterConstraints for MLE.
    void get_parameter_constraints(const std::vector<double>& sample,
                                   std::vector<double>& initials,
                                   std::vector<double>& lowers,
                                   std::vector<double>& uppers) const {
        initials.resize(4);
        lowers.resize(4);
        uppers.resize(4);
        try {
            auto lm = data::linear_moments(sample);
            initials = parameters_from_linear_moments(lm);

            if (initials[0] == 0.0) initials[0] = kDoubleMachineEpsilon;
            lowers[0] = -std::pow(10.0, std::ceil(std::log10(std::fabs(initials[0])) + 1.0));
            uppers[0] =  std::pow(10.0, std::ceil(std::log10(std::fabs(initials[0])) + 1.0));
            lowers[1] = kDoubleMachineEpsilon;
            uppers[1] =  std::pow(10.0, std::ceil(std::log10(std::fabs(initials[1])) + 1.0));
            lowers[2] = -10.0; uppers[2] = 10.0;
            lowers[3] = -2.0;  uppers[3] = 2.0;

            if (initials[2] <= lowers[2] || initials[2] >= uppers[2]) initials[2] = 0.0;
            if (initials[3] <= lowers[3] || initials[3] >= uppers[3]) initials[3] = 0.0;
        } catch (...) {
            // Fall back: use simple bounds
            initials = {sample[0], 1.0, 0.0, 0.0};
            double abs_xi = std::fabs(initials[0]);
            if (abs_xi == 0.0) abs_xi = 1.0;
            lowers[0] = -std::pow(10.0, std::ceil(std::log10(abs_xi) + 1.0));
            uppers[0] =  std::pow(10.0, std::ceil(std::log10(abs_xi) + 1.0));
            lowers[1] = kDoubleMachineEpsilon; uppers[1] = 1000.0;
            lowers[2] = -10.0; uppers[2] = 10.0;
            lowers[3] = -2.0;  uppers[3] = 2.0;
        }
    }

    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            KappaFour k4;
            k4.set_parameters(x);
            return k4.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 4, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }

   private:
    // Numerical moments via stratified integration — mirrors C# CentralMoments(int steps=1000).
    void compute_moments() const {
        const int steps = 1000;
        double a = inverse_cdf(1e-8);
        double b = inverse_cdf(1.0 - 1e-8);
        if (a >= b) {
            u_[0] = a; u_[1] = kNaN; u_[2] = kNaN; u_[3] = kNaN;
            moments_computed_ = true;
            return;
        }
        double width = (b - a) / steps;
        // Stratified bin integration (mirrors C# Stratify.XValues pattern)
        auto upper = [&](int i) { return a + (i + 1) * width; };
        auto lower = [&](int i) { return a + i * width; };
        auto midpt = [&](int i) { return a + (i + 0.5) * width; };

        double sumU1 = 0, sumU2 = 0;
        double dF0 = cdf(upper(0));
        sumU1 += upper(0) * dF0;
        sumU2 += upper(0) * upper(0) * dF0;

        for (int i = 1; i < steps - 1; i++) {
            double dF = cdf(upper(i)) - cdf(lower(i));
            double m  = midpt(i);
            sumU1 += m * dF;
            sumU2 += m * m * dF;
        }

        double dFlast = 1.0 - cdf(lower(steps - 1));
        double xlast  = lower(steps - 1);
        sumU1 += xlast * dFlast;
        sumU2 += xlast * xlast * dFlast;

        double u1 = sumU1;
        double u2 = std::sqrt(sumU2 - u1 * u1);

        double sumU3 = 0, sumU4 = 0;
        {
            double z = (upper(0) - u1) / u2;
            sumU3 += z * z * z * dF0;
            sumU4 += z * z * z * z * dF0;
        }
        for (int i = 1; i < steps - 1; i++) {
            double dF = cdf(upper(i)) - cdf(lower(i));
            double z  = (midpt(i) - u1) / u2;
            sumU3 += z * z * z * dF;
            sumU4 += z * z * z * z * dF;
        }
        {
            double z = (xlast - u1) / u2;
            sumU3 += z * z * z * dFlast;
            sumU4 += z * z * z * z * dFlast;
        }

        u_[0] = u1;
        u_[1] = u2;
        u_[2] = sumU3;
        u_[3] = sumU4;
        moments_computed_ = true;
    }

    void ensure_moments() const {
        if (!moments_computed_) compute_moments();
    }

    static bool validate(const std::vector<double>& p) {
        if (p.size() < 4) return false;
        if (std::isnan(p[0]) || std::isinf(p[0])) return false;
        if (std::isnan(p[1]) || std::isinf(p[1]) || p[1] <= 0.0) return false;
        if (std::isnan(p[2]) || std::isinf(p[2])) return false;
        if (std::isnan(p[3]) || std::isinf(p[3])) return false;
        return true;
    }

    double xi_    = 100.0;
    double alpha_ = 10.0;
    double kappa_ = 0.0;
    double hondo_ = 0.0;

    mutable bool moments_computed_ = false;
    mutable double u_[4] = {kNaN, kNaN, kNaN, kNaN};
};

}  // namespace bestfit::numerics::distributions
