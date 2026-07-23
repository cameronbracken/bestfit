// ported from: Numerics/Distributions/Univariate/NoncentralT.cs @ 2a0357a
//
// Re-audited against v2.1.4's "Harden distribution parameter validation" wave: (1)
// ValidateParameters now rejects NaN/Infinity in `v` explicitly -- the old `v < 1.0d` check
// is false for both NaN and +Infinity, so a non-finite degrees-of-freedom could silently
// read as valid; validate() below adds the missing isnan/isinf short-circuit (a genuine
// behavior fix, not just a header/provenance bump). (2) SetParameters' assign-then-validate
// ordering was already correct in this port (nu_/lambda_ assigned directly, single validate
// call at the end) -- no change there. (3) C# also renamed NCT_CDF/NCTDist/NCT_INV to
// EvaluateCdfWithFallback/EvaluateCdfSeries/InverseCdf and de-gotoed NCTDist's internal
// control flow (dead labels/variables renamed for readability) -- per the task brief this
// refactor is behavior-preserving and is deliberately NOT mirrored cosmetically here; only
// the validation fix above is ported. Confirmed the existing oracle fixture values are
// unchanged (same AS 243 math, just renamed locals/goto-free control flow in C#).
//
// The Noncentral t probability distribution with parameters ν (degrees of freedom) and
// μ (noncentrality). CDF via the twin-series algorithm AS 243 (Guenther 1978 / Lenth 1989).
// PDF computed from the CDF recurrence: for x != 0, PDF(x) = (v/x)*(NCT_CDF(x*sqrt(1+2/v), v+2, mu) - CDF(x));
// for x == 0, PDF(0) = Gamma((v+1)/2) / (sqrt(pi*v)*Gamma(v/2)) * exp(-mu^2/2).
// InverseCDF via bracket-and-Brent root-finding with a Jennett-Welch initial guess.
// Mode via golden-section / parabolic Brent minimization of -PDF on [Q(0.001), Q(0.999)].
// Skewness and Kurtosis via numerical CentralMoments (mirrors C# CentralMoments(1e-8)).
// No IEstimation or ILinearMomentEstimation – C# implements neither.
// Logic mirrors the C# source method-for-method.
#pragma once
#include <string>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/math/rootfinding/brent.hpp"
#include "corehydro/numerics/math/special/beta.hpp"
#include "corehydro/numerics/math/special/gamma.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

namespace sf_nct = corehydro::numerics::math::special;

class NoncentralT : public UnivariateDistributionBase {
   public:
    // Default: 10 degrees of freedom, noncentrality = 0.
    NoncentralT() { set_parameters(10.0, 0.0); }

    // Constructs NCT with given degrees of freedom and noncentrality parameter.
    NoncentralT(double degrees_of_freedom, double noncentrality) {
        set_parameters(degrees_of_freedom, noncentrality);
    }

    double degrees_of_freedom() const { return nu_; }
    double noncentrality() const { return lambda_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::NoncentralT;
    }
    int number_of_parameters() const override { return 2; }
    std::vector<double> get_parameters() const override { return {nu_, lambda_}; }

    void set_parameters(double v, double mu) {
        nu_ = v;
        lambda_ = mu;
        parameters_valid_ = validate(v, mu);
    }
    void set_parameters(const std::vector<double>& p) override { set_parameters(p[0], p[1]); }

    // --- Moments / support ---

    // Mean = lambda * sqrt(v/2) * Gamma((v-1)/2) / Gamma(v/2)  when v > 1; else NaN.
    double mean() const override {
        if (nu_ > 1.0) {
            return lambda_ * std::sqrt(nu_ / 2.0) *
                   sf_nct::function((nu_ - 1.0) / 2.0) / sf_nct::function(nu_ / 2.0);
        }
        return kNaN;
    }

    double median() const override { return inverse_cdf(0.5); }

    // Mode: maximize PDF on [Q(0.001), Q(0.999)] via Brent's parabolic/golden-section optimizer.
    double mode() const override {
        double lo = inverse_cdf(0.001);
        double hi = inverse_cdf(0.999);
        return brent_maximize([this](double x) { return pdf(x); }, lo, hi);
    }

    // SD = sqrt(v*(1+lambda^2)/(v-2) - lambda^2/2*(Gamma((v-1)/2)/Gamma(v/2))^2) for v > 2.
    double standard_deviation() const override {
        if (nu_ > 2.0) {
            double a = nu_ * (1.0 + lambda_ * lambda_) / (nu_ - 2.0);
            double b = nu_ * lambda_ * lambda_ / 2.0;
            double c = sf_nct::function((nu_ - 1.0) / 2.0) / sf_nct::function(nu_ / 2.0);
            return std::sqrt(a - b * c * c);
        }
        return kNaN;
    }

    // Skewness and Kurtosis via numerical CentralMoments (mirrors C# CentralMoments(1e-8)),
    // but with a composite Gauss-Legendre quadrature in place of C#'s AdaptiveGaussKronrod
    // (not yet ported). Accurate for near-symmetric cases (small |lambda|); for large |lambda|
    // with small df these can diverge from the C# value and are NOT oracle-verified there.
    // Revisit once the adaptive integrator (planned foundation task A7) is ported.
    // (B9 additive: re-expose the base trapezoidal central_moments(int steps) overload,
    // which the local central_moments(double) helper below would otherwise name-hide.)
    using UnivariateDistributionBase::central_moments;
    double skewness() const override { return central_moments(1e-8)[2]; }
    double kurtosis() const override { return central_moments(1e-8)[3]; }

    double minimum() const override { return -kInf; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---

    // PDF: for x != 0 uses the CDF recurrence from the C# source:
    //   PDF(x) = (v/x) * (NCT_CDF(x*sqrt(1+2/v), v+2, mu) - CDF(x,v,mu))
    // For x == 0: PDF(0) = Gamma((v+1)/2) / (sqrt(pi*v)*Gamma(v/2)) * exp(-mu^2/2).
    double pdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("NoncentralT: invalid parameters");
        double v = nu_;
        double u = lambda_;
        if (x != 0.0) {
            double A = nct_cdf(x * std::sqrt(1.0 + 2.0 / v), v + 2.0, u);
            double B = nct_cdf(x, v, u);
            double C = v / x;
            return C * (A - B);
        } else {
            double A = sf_nct::function((v + 1.0) / 2.0);
            double B = std::sqrt(kPi * v) * sf_nct::function(v / 2.0);
            double C = std::exp(-(u * u) / 2.0);
            return (A / B) * C;
        }
    }

    // CDF delegates to nct_dist — the AS 243 twin-series algorithm.
    double cdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("NoncentralT: invalid parameters");
        return nct_dist(x, nu_, lambda_);
    }

    // InverseCDF via bracket-then-Brent root-finding with Jennett-Welch initial guess.
    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (!parameters_valid_) throw std::invalid_argument("NoncentralT: invalid parameters");
        return nct_inv(probability, nu_, lambda_);
    }

    // --- Parameter display names (X1; C# NoncentralT.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Degrees of Freedom (\xCE\xBD)", "Noncentrality (\xCE\xBC)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xCE\xBD", "\xCE\xBC"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<NoncentralT>(nu_, lambda_);
    }

   private:
    // --- Validate parameters: nu >= 1 (and finite) and mu finite ---
    static bool validate(double v, double mu) {
        // `v < 1.0` alone is false for both NaN and +Infinity, so the explicit isnan/isinf
        // check is required -- this is the v2.1.4 fix (see file header).
        if (std::isnan(v) || std::isinf(v) || v < 1.0) return false;
        if (std::isnan(mu) || std::isinf(mu)) return false;
        return true;
    }

    // Standard normal CDF using erfc: mirrors Normal.StandardCDF -> MultivariateNormal.MVNPHI.
    // Both evaluate erf-based CDFs; results match to well within 1e-10.
    static double standard_normal_cdf(double z) {
        return 0.5 * (1.0 + std::erf(z / kSqrt2));
    }

    // --- NCT_CDF: wrapper around nct_dist that catches non-convergence (ArgumentException)
    //     and falls back to the normal approximation from the C# source.
    double nct_cdf(double t, double df, double delta) const {
        double ans;
        bool failed = false;
        try {
            ans = nct_dist_checked(t, df, delta, failed);
        } catch (...) {
            failed = true;
        }
        if (failed) {
            // Normal approximation fallback (mirrors C# catch block in NCT_CDF)
            double Z = (t * (1.0 - 1.0 / (4.0 * df)) - delta) /
                       std::sqrt(1.0 + t * t / (2.0 * df));
            ans = standard_normal_cdf(Z);
            if (ans < 0.0) ans = 0.0;
            if (ans > 1.0) ans = 1.0;
        }
        return ans;
    }

    // --- NCTDist: AS 243 twin-series CDF.  Throws std::invalid_argument on non-convergence
    //     (maps to C# ArgumentException).
    static double nct_dist(double t, double df, double delta) {
        bool failed = false;
        double result = nct_dist_checked(t, df, delta, failed);
        if (failed)
            throw std::invalid_argument("NoncentralT: AS243 series failed to converge");
        return result;
    }

    // Core AS 243 algorithm. On non-convergence sets failed=true; caller decides how to handle.
    static double nct_dist_checked(double t, double df, double delta, bool& failed) {
        // ALGORITHM AS 243  APPL. STATIST. (1989), VOL.38, NO. 1
        // Translated from William A. Huber's C# port, which itself follows the FORTRAN.
        //
        // Constants matching the C# source exactly:
        constexpr int ITRMAX = 10000;
        constexpr double Errmax = 0.000000001;
        constexpr double zero = 0.0;
        constexpr double half = 0.5;
        constexpr double one = 1.0;
        constexpr double two = 2.0;
        // R2PI = 1/{Gamma(1.5)*sqrt(2)} = sqrt(2/pi)
        constexpr double r2pi = 0.797884560802865;
        // ALNRPI = ln(sqrt(pi))
        constexpr double alnrpi = 0.5723649429247;

        failed = false;
        double TNC = zero;
        double TT = t;
        double DEL = delta;
        bool NEGDEL = false;

        if (t < zero) {
            NEGDEL = true;
            TT = -TT;
            DEL = -DEL;
        }

        // Compute x = t^2 / (t^2 + df); handles x <= 0 (t == 0) branch
        double x = t * t / (t * t + df);
        int N = 0;

        if (x > zero) {
            double LAMBDA = DEL * DEL;
            double P = half * std::exp(-half * LAMBDA);
            double q = r2pi * P * DEL;
            double s = half - P;
            double a = half;
            double b = half * df;
            double RXB = std::pow(one - x, b);
            // ALBETA = ln(B(a,b)) = ln(sqrt(pi)) + log_gamma(b) - log_gamma(a+b)
            double ALBETA = alnrpi + sf_nct::log_gamma(b) - sf_nct::log_gamma(a + b);
            // IncompleteRatio(x, a, b, ALBETA) = I_x(a,b) the regularized incomplete beta
            double XODD = sf_nct::beta::incomplete(a, b, x);
            double GODD = two * RXB * std::exp(a * std::log(x) - ALBETA);
            double XEVEN = one - RXB;
            double GEVEN = b * x * RXB;
            TNC = P * XODD + q * XEVEN;
            double ERRBD;
            N = 1;
            do {
                a = a + one;
                XODD = XODD - GODD;
                XEVEN = XEVEN - GEVEN;
                GODD = GODD * x * (a + b - one) / a;
                GEVEN = GEVEN * x * (a + b - half) / (a + half);
                P = P * LAMBDA / (two * N);
                q = q * LAMBDA / (two * N + one);
                s = s - P;
                N = N + 1;
                TNC = TNC + P * XODD + q * XEVEN;
                ERRBD = two * s * (XODD - GODD);
            } while (ERRBD > Errmax && N <= ITRMAX);
        }
        // Twenty: label — check iteration count (only relevant when x > 0)
        if (x > zero && N > ITRMAX) {
            failed = true;
            return 0.0;
        }

        if (NEGDEL) {
            // TNC = Normal.StandardCDF(DEL) - TNC
            TNC = standard_normal_cdf(DEL) - TNC;
        } else {
            TNC = TNC + (1.0 - standard_normal_cdf(DEL));
        }
        return TNC;
    }

    // --- NCT_INV: inverse via bracket + Brent root-finding.
    //     Mirrors C# NCT_INV method-for-method.
    double nct_inv(double p, double df, double delta) const {
        constexpr double ytol = 0.0000001;
        constexpr double xtol = 0.0000001;
        constexpr int iterMax = 50;

        double t0 = nct_inv0(p, df, delta);
        double y0 = nct_dist(t0, df, delta) - p;
        double tInc;
        if (y0 > ytol) {
            tInc = -1.0;
        } else if (y0 < -ytol) {
            tInc = 1.0;
        } else {
            return t0;
        }

        // Find a bracket through overshooting (secant method).
        double t1 = t0 + tInc;
        double y1 = nct_dist(t1, df, delta) - p;
        int iter = 0;
        while ((y0 < 0.0) != (y1 > 0.0) &&
               std::fabs(t1 - t0) > xtol &&
               iter < iterMax) {
            double Slope = (y1 - y0) / (t1 - t0);
            if (Slope == 0.0) {
                return (t1 + t0) / 2.0;
            }
            double t2 = t0 - 2.0 * y0 / Slope;
            t0 = t1;
            t1 = t2;
            y0 = y1;
            y1 = nct_dist(t1, df, delta) - p;
            iter = iter + 1;
        }

        // Solve for T using Brent root-finding.
        double lo = std::min(t0, t1);
        double hi = std::max(t0, t1);
        double ANS = math::rootfinding::solve(
            [&](double x) { return nct_dist(x, df, delta) - p; },
            lo, hi, xtol, 1000, /*report_failure=*/false);
        return ANS;
    }

    // --- NCTInv0: initial estimate using the Jennett-Welch approximation.
    //     Mirrors C# NCTInv0 method-for-method.
    double nct_inv0(double P, double N, double D) const {
        // Jennett & Welch approximation, formula (14.1), Johnson & Kotz Vol. 2.
        double z = sf_nct::detail::normal_standard_z(P);
        double b = std::exp(sf_nct::log_gamma((N + 1.0) / 2.0) - sf_nct::log_gamma(N / 2.0)) *
                   std::sqrt(2.0 / N);
        double u2 = z * z;
        double b2 = b * b;
        double denom = b2 - u2 * (1.0 - b2);
        double disc = b2 + (1.0 - b2) * (D * D - u2);
        if (disc > 0.0 && std::fabs(denom) > 1e-12) {
            return (D * b + z * std::sqrt(disc)) / denom;
        } else {
            // Fallback: offset central-t quantile by noncentrality
            // Mirror C#: new StudentT(N).InverseCDF(P) + D
            // Central t quantile via incomplete beta inverse (same as StudentT::inverse_cdf
            // with mu=0, sigma=1) — use the mid-range path.
            return student_t_quantile(P, N) + D;
        }
    }

    // Central t quantile (mu=0, sigma=1, nu=N) — mirrors StudentT::inverse_cdf.
    static double student_t_quantile(double probability, double nu) {
        if (probability > 0.25 && probability < 0.75) {
            if (probability == 0.5) return 0.0;
            double z = sf_nct::beta::incomplete_inverse(
                0.5, 0.5 * nu, std::fabs(1.0 - 2.0 * probability));
            double t = std::sqrt(nu * z / (1.0 - z));
            if (probability < 0.5) t = -t;
            return t;
        } else {
            int rflg = -1;
            double p = probability;
            if (p >= 0.5) {
                p = 1.0 - p;
                rflg = 1;
            }
            double z = sf_nct::beta::incomplete_inverse(0.5 * nu, 0.5, 2.0 * p);
            if (std::numeric_limits<double>::max() * z < nu) {
                return static_cast<double>(rflg) * std::numeric_limits<double>::max();
            }
            double t = std::sqrt(nu / z - nu);
            return static_cast<double>(rflg) * t;
        }
    }

    // --- Brent's parabolic + golden-section maximizer.
    //     Mirrors BrentSearch.Optimize() with sign-flip for maximization.
    //     Used only for Mode computation.
    static double brent_maximize(const std::function<double(double)>& f, double ax, double cx) {
        constexpr double CGOLD = 0.381966;
        constexpr double ZEPS = corehydro::numerics::kDoubleMachineEpsilon * 1.0e-3;
        constexpr int max_iter = 500;
        constexpr double tol = 1e-8;

        double bx = 0.5 * (ax + cx);
        double a = (ax < cx ? ax : cx);
        double b = (ax > cx ? ax : cx);
        double x = bx, w = bx, v = bx;
        // Maximize f = minimize -f
        double fx = -f(x), fw = fx, fv = fx;
        double e = 0.0, d = 0.0;

        for (int i = 0; i < max_iter; ++i) {
            double xm = 0.5 * (a + b);
            double tol2 = 2.0 * (tol * std::fabs(x) + ZEPS);
            if (std::fabs(x - xm) <= tol2 - 0.5 * (b - a)) break;

            double p = 0.0, q = 0.0, r = 0.0, u = 0.0;
            if (std::fabs(e) > tol * std::fabs(x) + ZEPS) {
                r = (x - w) * (fx - fv);
                q = (x - v) * (fx - fw);
                p = (x - v) * q - (x - w) * r;
                q = 2.0 * (q - r);
                if (q > 0.0) p = -p;
                q = std::fabs(q);
                double etemp = e;
                e = d;
                if (std::fabs(p) >= std::fabs(0.5 * q * etemp) ||
                    p <= q * (a - x) || p >= q * (b - x)) {
                    e = (x >= xm) ? (a - x) : (b - x);
                    d = CGOLD * e;
                } else {
                    d = p / q;
                    u = x + d;
                    if ((u - a) < tol2 || (b - u) < tol2)
                        d = (xm >= x) ? std::fabs(tol * std::fabs(x) + ZEPS)
                                      : -(std::fabs(tol * std::fabs(x) + ZEPS));
                }
            } else {
                e = (x >= xm) ? (a - x) : (b - x);
                d = CGOLD * e;
            }
            double tol1 = tol * std::fabs(x) + ZEPS;
            u = (std::fabs(d) >= tol1) ? (x + d) : (x + (d >= 0 ? tol1 : -tol1));
            double fu = -f(u);

            if (fu <= fx) {
                if (u < x) b = x; else a = x;
                v = w; fv = fw;
                w = x; fw = fx;
                x = u; fx = fu;
            } else {
                if (u < x) a = u; else b = u;
                if (fu <= fw || w == x) {
                    v = w; fv = fw;
                    w = u; fw = fu;
                } else if (fu <= fv || v == x || v == w) {
                    v = u; fv = fu;
                }
            }
        }
        return x;
    }

    // --- Numerical central moments mirroring C# CentralMoments(1e-8).
    //     C# uses AdaptiveGaussKronrod on [InverseCDF(1e-16), InverseCDF(1-1e-16)].
    //     We mirror using composite Gauss-Legendre (7-point rule) on the same interval
    //     via x = Q(p) variable change: E[f(X)] = integral_0^1 f(Q(p)) dp.
    //     Partition [1e-16, 1-1e-16] into sub-intervals to resolve the tails.
    //     Returns {u1, u2, u3, u4} matching C# CentralMoments indices:
    //       [0]=mean, [1]=sd, [2]=skewness, [3]=kurtosis.
    std::vector<double> central_moments(double /*eps*/) const {
        // 7-point Gauss-Legendre nodes and weights on [-1,1]
        static constexpr double gl_nodes[7] = {
            -0.9491079123427585, -0.7415311855993945, -0.4058451513773972,
             0.0,
             0.4058451513773972,  0.7415311855993945,  0.9491079123427585};
        static constexpr double gl_weights[7] = {
             0.1294849661688697,  0.2797053914892767,  0.3818300505051189,
             0.4179591836734694,
             0.3818300505051189,  0.2797053914892767,  0.1294849661688697};

        // Break the probability interval into sub-segments to resolve the tails.
        // Fine-grained near 0 and 1 to capture the heavy-tailed 4th moment accurately.
        static const double breakpoints[] = {
            1e-16, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3,
            0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.5,
            0.70, 0.80, 0.90, 0.95, 0.98, 0.99,
            1.0-5e-3, 1.0-1e-3, 1.0-5e-4, 1.0-1e-4, 1.0-1e-5, 1.0-1e-6,
            1.0-1e-8, 1.0-1e-10, 1.0-1e-12, 1.0-1e-16};
        constexpr int n_breaks = 33;

        // First pass: compute mean via E[X] = integral x*pdf(x) dx = integral Q(p) dp
        double u1 = 0.0;
        for (int seg = 0; seg + 1 < n_breaks; ++seg) {
            double pa = breakpoints[seg], pb = breakpoints[seg + 1];
            double mid = 0.5 * (pa + pb), half_len = 0.5 * (pb - pa);
            for (int k = 0; k < 7; ++k) {
                double p = mid + half_len * gl_nodes[k];
                double x = inverse_cdf(p);
                u1 += half_len * gl_weights[k] * x;
            }
        }

        // Second pass: compute variance = E[(X-u1)^2]
        double u2_sq = 0.0;
        for (int seg = 0; seg + 1 < n_breaks; ++seg) {
            double pa = breakpoints[seg], pb = breakpoints[seg + 1];
            double mid = 0.5 * (pa + pb), half_len = 0.5 * (pb - pa);
            for (int k = 0; k < 7; ++k) {
                double p = mid + half_len * gl_nodes[k];
                double x = inverse_cdf(p) - u1;
                u2_sq += half_len * gl_weights[k] * x * x;
            }
        }
        double u2 = std::sqrt(u2_sq);

        // Third pass: compute standardised skewness and kurtosis
        double sum3 = 0.0, sum4 = 0.0;
        for (int seg = 0; seg + 1 < n_breaks; ++seg) {
            double pa = breakpoints[seg], pb = breakpoints[seg + 1];
            double mid = 0.5 * (pa + pb), half_len = 0.5 * (pb - pa);
            for (int k = 0; k < 7; ++k) {
                double p = mid + half_len * gl_nodes[k];
                double z = (u2 > 0.0) ? (inverse_cdf(p) - u1) / u2 : 0.0;
                double z2 = z * z;
                sum3 += half_len * gl_weights[k] * z * z2;
                sum4 += half_len * gl_weights[k] * z2 * z2;
            }
        }

        return {u1, u2, sum3, sum4};
    }

    double nu_ = 10.0;
    double lambda_ = 0.0;
};

}  // namespace corehydro::numerics::distributions
