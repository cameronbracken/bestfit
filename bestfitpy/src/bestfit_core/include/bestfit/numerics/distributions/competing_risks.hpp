// ported from: Numerics/Distributions/Univariate/CompetingRisks.cs @ a2c4dbf
//
// Competing risks (series / parallel system) distribution.
// Constructor: CompetingRisks(vector of component unique_ptrs).
// Two modes via minimum_of_random_variables (default true, mirrors C#):
//   true  → min-of-components: CDF = Union   = 1 − ∏ Si(x) = 1 − ∏(1−Fi(x))
//   false → max-of-components: CDF = Product = ∏ Fi(x)
// PDF (min-rule): f(x) = [Σ hᵢ(x)] · S(x) where hᵢ=fᵢ/Sᵢ, S=∏Sᵢ  (exact for Independent)
// PDF (max-rule): f(x) = [Σ fᵢ/Fᵢ] · ∏Fᵢ                           (exact for Independent)
// LogPDF: numerically stable log-space versions using log-sum-exp (mirrors C#).
// InverseCDF: Brent root-finding on CDF(y) − p = 0; bracket from per-component quantiles.
//   On solve failure, falls back to bisection on [minimum, maximum].
// Moments: AGK numerical integration over [InverseCDF(1e-8), InverseCDF(1-1e-8)],
//   mirroring C# CentralMoments(1000) (trapezoidal int); AGK reproduces to oracle tolerance.
// IEstimation: MLE via Nelder-Mead on total log-likelihood (mirrors C# MLE()); initial
//   values from per-component estimate() if available, else sample statistics.
//   Only MaximumLikelihood is supported; others throw (mirrors C# Estimate()).
// SCOPE NOTE: only the Independent dependency mode is implemented. The PerfectlyPositive,
//   PerfectlyNegative, and CorrelationMatrix modes require a ported MultivariateNormal
//   distribution (Phase 3). If a fixture or oracle requires those modes, cover Independent
//   only and note the deferral here.
// type() returns UnivariateDistributionType::CompetingRisks (mirrors C#).
// Not wired into the flat factory (composite-only, like Mixture).
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/data/statistics.hpp"
#include "bestfit/numerics/distributions/base/i_estimation.hpp"
#include "bestfit/numerics/distributions/base/parameter_estimation_method.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "bestfit/numerics/math/integration/adaptive_gauss_kronrod.hpp"
#include "bestfit/numerics/math/optimization/nelder_mead.hpp"
#include "bestfit/numerics/math/rootfinding/brent.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::distributions {

class CompetingRisks : public UnivariateDistributionBase, public IEstimation {
   public:
    // Construct from already-created component unique_ptrs (takes ownership).
    explicit CompetingRisks(
        std::vector<std::unique_ptr<UnivariateDistributionBase>> components)
        : components_(std::move(components)) {
        validate_and_set();
    }

    // Convenience: construct by cloning components from raw pointers.
    explicit CompetingRisks(const std::vector<UnivariateDistributionBase*>& components) {
        components_.reserve(components.size());
        for (auto* c : components) components_.push_back(c->clone());
        validate_and_set();
    }

    // Whether the composite is min-of-components (true, default) or max-of-components (false).
    // Mirrors C# MinimumOfRandomVariables property.
    bool minimum_of_random_variables = true;

    // Accessors.
    int component_count() const { return static_cast<int>(components_.size()); }
    const UnivariateDistributionBase& component(int i) const { return *components_[i]; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::CompetingRisks;
    }

    // Sum of each component's NumberOfParameters (mirrors C#).
    int number_of_parameters() const override {
        int sum = 0;
        for (const auto& c : components_) sum += c->number_of_parameters();
        return sum;
    }

    // [p0_0, p0_1, ..., p1_0, p1_1, ...] (mirrors C# GetParameters).
    std::vector<double> get_parameters() const override {
        std::vector<double> result;
        for (const auto& c : components_) {
            auto p = c->get_parameters();
            result.insert(result.end(), p.begin(), p.end());
        }
        return result;
    }

    // Mirrors C# SetParameters(IList<double>): distribute flat array to components in order.
    void set_parameters(const std::vector<double>& parameters) override {
        int t = 0;
        for (const auto& c : components_) {
            int n = c->number_of_parameters();
            std::vector<double> p(parameters.begin() + t, parameters.begin() + t + n);
            c->set_parameters(p);
            t += n;
        }
        parameters_valid_ = validate_components();
        moments_computed_ = false;
    }

    // --- Moments / support ---
    double mean() const override {
        if (!moments_computed_) compute_moments();
        return u_[0];
    }
    double median() const override { return inverse_cdf(0.5); }
    double mode() const override {
        // Mirrors C#: BrentSearch.Maximize(PDF, InverseCDF(0.001), InverseCDF(0.999)).
        // Use ternary search as an approximation (exact for unimodal distributions).
        double a = inverse_cdf(0.001);
        double b = inverse_cdf(0.999);
        for (int i = 0; i < 200; ++i) {
            double m1 = a + (b - a) / 3.0;
            double m2 = b - (b - a) / 3.0;
            if (pdf(m1) < pdf(m2))
                a = m1;
            else
                b = m2;
        }
        return 0.5 * (a + b);
    }
    double standard_deviation() const override {
        if (!moments_computed_) compute_moments();
        return u_[1];
    }
    double skewness() const override {
        if (!moments_computed_) compute_moments();
        return u_[2];
    }
    double kurtosis() const override {
        if (!moments_computed_) compute_moments();
        return u_[3];
    }
    // Minimum = min over components (mirrors C# Distributions.Min(p => p.Minimum)).
    double minimum() const override {
        double m = kInf;
        for (const auto& c : components_) m = std::min(m, c->minimum());
        return m;
    }
    // Maximum = max over components (mirrors C# Distributions.Max(p => p.Maximum)).
    double maximum() const override {
        double m = -kInf;
        for (const auto& c : components_) m = std::max(m, c->maximum());
        return m;
    }

    // --- Distribution functions ---
    // Mirrors C# PDF: exact for Independent; falls back to numerical derivative otherwise.
    // Returns at least kMinDensity to prevent log-likelihood issues (mirrors C#).
    double pdf(double x) const override {
        if (components_.size() == 1) return components_[0]->pdf(x);
        double f = minimum_of_random_variables
            ? pdf_min_independent(x)
            : pdf_max_independent(x);
        return f < kMinDensity ? kMinDensity : f;
    }

    // Mirrors C# LogPDF: numerically stable log-space computation for Independent.
    double log_pdf(double x) const override {
        if (components_.size() == 1) return components_[0]->log_pdf(x);
        if (minimum_of_random_variables)
            return log_pdf_min_independent(x);
        else
            return log_pdf_max_independent(x);
    }

    // Mirrors C# CDF (Independent mode):
    //   min-rule: CDF = 1 − ∏(1−Fi(x)) = Union
    //   max-rule: CDF = ∏Fi(x) = JointProbability
    double cdf(double x) const override {
        if (components_.size() == 1) return components_[0]->cdf(x);
        double p;
        if (minimum_of_random_variables) {
            // 1 − ∏ Sᵢ(x)
            double prod_survival = 1.0;
            for (const auto& c : components_) {
                double si = 1.0 - c->cdf(x);
                if (si == 0.0) return 1.0;
                prod_survival *= si;
            }
            p = 1.0 - prod_survival;
        } else {
            // ∏ Fᵢ(x)
            double prod_cdf = 1.0;
            for (const auto& c : components_) {
                prod_cdf *= c->cdf(x);
                if (prod_cdf == 0.0) return 0.0;
            }
            p = prod_cdf;
        }
        return p < 0.0 ? 0.0 : p > 1.0 ? 1.0 : p;
    }

    // Mirrors C# InverseCDF: Brent root-finding on CDF(y) − probability = 0.
    // Bracket from per-component quantile values. Falls back to bisection on failure.
    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (components_.size() == 1) return components_[0]->inverse_cdf(probability);

        // Build bracket from per-component InverseCDF values (mirrors C#).
        double minX = kInf, maxX = -kInf;
        for (const auto& c : components_) {
            double q = c->inverse_cdf(probability);
            minX = std::min(minX, q);
            maxX = std::max(maxX, q);
        }
        if (minX >= maxX) maxX = minX + 1.0;

        double x = 0.0;
        try {
            auto fn = [this, probability](double y) { return probability - cdf(y); };
            // Expand bracket if not straddling zero (mirrors C# Brent.Bracket).
            int expand_attempts = 0;
            while (fn(minX) * fn(maxX) > 0.0 && expand_attempts++ < 200) {
                double mid = 0.5 * (minX + maxX);
                double half_range = 1.5 * (maxX - minX);
                minX = mid - half_range;
                maxX = mid + half_range;
            }
            x = math::rootfinding::solve(fn, minX, maxX, 1E-6, 100, true);
        } catch (...) {
            // Bisection fallback (mirrors C# fall-back to EmpiricalDistribution;
            // we bisect instead — noted in the header comment).
            double lo = minimum(), hi = maximum();
            if (std::isinf(lo)) lo = minX - 50.0;
            if (std::isinf(hi)) hi = maxX + 50.0;
            for (int i = 0; i < 200; ++i) {
                double mid = 0.5 * (lo + hi);
                if (cdf(mid) < probability) lo = mid; else hi = mid;
            }
            x = 0.5 * (lo + hi);
        }
        double mn = minimum(), mx = maximum();
        return x < mn ? mn : x > mx ? mx : x;
    }

    // --- Estimation (IEstimation) ---
    // Mirrors C# Estimate: only MaximumLikelihood supported.
    void estimate(const std::vector<double>& sample,
                  ParameterEstimationMethod method) override {
        if (method != ParameterEstimationMethod::MaximumLikelihood)
            throw std::runtime_error("CompetingRisks only supports MaximumLikelihood estimation");
        set_parameters(mle(sample));
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        std::vector<std::unique_ptr<UnivariateDistributionBase>> cloned;
        cloned.reserve(components_.size());
        for (const auto& c : components_) cloned.push_back(c->clone());
        auto cr = std::make_unique<CompetingRisks>(std::move(cloned));
        cr->minimum_of_random_variables = minimum_of_random_variables;
        return cr;
    }

   private:
    std::vector<std::unique_ptr<UnivariateDistributionBase>> components_;

    // Lazy moment cache.
    mutable bool moments_computed_ = false;
    mutable double u_[4] = {kNaN, kNaN, kNaN, kNaN};  // [mean, sd, skewness, kurtosis]

    // Soft floor used in tail arithmetic (mirrors C# _minDensity / _logZero).
    static constexpr double kMinDensity = 1E-300;
    static constexpr double kLogZero = -745.0;

    void validate_and_set() {
        parameters_valid_ = !components_.empty() && validate_components();
        moments_computed_ = false;
    }

    bool validate_components() const {
        for (const auto& c : components_)
            if (!c->parameters_valid()) return false;
        return true;
    }

    // log-sum-exp trick: log(Σ exp(v[i])).
    static double log_sum_exp(const std::vector<double>& v) {
        if (v.empty()) return -std::numeric_limits<double>::infinity();
        double max_val = *std::max_element(v.begin(), v.end());
        if (std::isinf(max_val)) return max_val;
        double sum = 0.0;
        for (double x : v) sum += std::exp(x - max_val);
        return max_val + std::log(sum);
    }

    // PDF for min-rule (Independent): f(x) = [Σ hᵢ(x)] · S(x).
    // Mirrors C# PDFMinimumIndependent.
    double pdf_min_independent(double x) const {
        double sum_hazard = 0.0;
        double product_survival = 1.0;
        for (const auto& c : components_) {
            double si = 1.0 - c->cdf(x);  // CCDF / survival
            double fi = c->pdf(x);
            product_survival *= si;
            if (si > kMinDensity) {
                sum_hazard += fi / si;
            } else if (fi > kMinDensity) {
                sum_hazard += fi / kMinDensity;
            }
        }
        return sum_hazard * product_survival;
    }

    // PDF for max-rule (Independent): f(x) = [Σ fᵢ/Fᵢ] · ∏Fᵢ.
    // Mirrors C# PDFMaximumIndependent.
    double pdf_max_independent(double x) const {
        double sum_ratio = 0.0;
        double product_cdf = 1.0;
        for (const auto& c : components_) {
            double fi = c->cdf(x);
            double pi = c->pdf(x);
            product_cdf *= fi;
            if (fi > kMinDensity) {
                sum_ratio += pi / fi;
            } else if (pi > kMinDensity) {
                sum_ratio += pi / kMinDensity;
            }
        }
        return sum_ratio * product_cdf;
    }

    // Log-PDF for min-rule. Mirrors C# LogPDFMinimumIndependent.
    double log_pdf_min_independent(double x) const {
        int n = component_count();
        std::vector<double> log_survival(n), log_hazard(n);
        double sum_log_survival = 0.0;
        bool all_survival_zero = true;

        for (int i = 0; i < n; ++i) {
            double si = 1.0 - components_[i]->cdf(x);
            double fi = components_[i]->pdf(x);

            if (si > kMinDensity) {
                log_survival[i] = std::log(si);
                all_survival_zero = false;
            } else {
                log_survival[i] = kLogZero;
            }
            sum_log_survival += log_survival[i];

            if (fi > kMinDensity && si > kMinDensity) {
                log_hazard[i] = std::log(fi) - std::log(si);
            } else if (fi <= kMinDensity) {
                log_hazard[i] = kLogZero;
            } else {
                log_hazard[i] = std::log(fi) - kLogZero;
            }
        }

        if (all_survival_zero) return kLogZero;
        double log_sum_h = log_sum_exp(log_hazard);
        double result = log_sum_h + sum_log_survival;
        return (std::isnan(result) || std::isinf(result)) ? kLogZero : result;
    }

    // Log-PDF for max-rule. Mirrors C# LogPDFMaximumIndependent.
    double log_pdf_max_independent(double x) const {
        int n = component_count();
        std::vector<double> log_cdf_v(n), log_ratio(n);
        double sum_log_cdf = 0.0;
        bool all_cdf_zero = true;

        for (int i = 0; i < n; ++i) {
            double fi = components_[i]->cdf(x);
            double pi = components_[i]->pdf(x);

            if (fi > kMinDensity) {
                log_cdf_v[i] = std::log(fi);
                all_cdf_zero = false;
            } else {
                log_cdf_v[i] = kLogZero;
            }
            sum_log_cdf += log_cdf_v[i];

            if (pi > kMinDensity && fi > kMinDensity) {
                log_ratio[i] = std::log(pi) - std::log(fi);
            } else if (pi <= kMinDensity) {
                log_ratio[i] = kLogZero;
            } else {
                log_ratio[i] = std::log(pi) - kLogZero;
            }
        }

        if (all_cdf_zero) return kLogZero;
        double log_sum_r = log_sum_exp(log_ratio);
        double result = log_sum_r + sum_log_cdf;
        return (std::isnan(result) || std::isinf(result)) ? kLogZero : result;
    }

    // AGK numerical moments over [InverseCDF(1e-8), InverseCDF(1-1e-8)].
    // Mirrors C# CentralMoments(1000) (trapezoidal int); AGK reproduces to oracle tolerance.
    void compute_moments() const {
        namespace agk = math::integration;
        const double eps = 1e-8;
        double a = inverse_cdf(eps);
        double b = inverse_cdf(1.0 - eps);
        if (a >= b) {
            u_[0] = a;
            u_[1] = u_[2] = u_[3] = kNaN;
            moments_computed_ = true;
            return;
        }
        const double tol = 1e-8;
        u_[0] = agk::integrate([this](double x) { return x * pdf(x); }, a, b, tol, tol);
        const double mu = u_[0];
        u_[1] = std::sqrt(agk::integrate(
            [this, mu](double x) { return (x - mu) * (x - mu) * pdf(x); }, a, b, tol, tol));
        const double s = u_[1];
        u_[2] = agk::integrate(
            [this, mu, s](double x) {
                double z = (x - mu) / s;
                return z * z * z * pdf(x);
            }, a, b, tol, tol);
        u_[3] = agk::integrate(
            [this, mu, s](double x) {
                double z = (x - mu) / s;
                return z * z * z * z * pdf(x);
            }, a, b, tol, tol);
        moments_computed_ = true;
    }

    // Nelder-Mead MLE. Mirrors C# MLE() method.
    std::vector<double> mle(const std::vector<double>& sample) const {
        int K = static_cast<int>(components_.size());
        int Np = number_of_parameters();

        // Get initial values for each component via estimate() if available.
        std::vector<double> initials, lowers, uppers;
        {
            auto tmp_stats = data::product_moments(sample);
            double sample_mean = tmp_stats[0];
            double sample_sd = std::max(tmp_stats[1], 1e-10);
            for (int i = 0; i < K; ++i) {
                auto comp_clone = components_[i]->clone();
                auto* est = dynamic_cast<IEstimation*>(comp_clone.get());
                if (est) {
                    try { est->estimate(sample, ParameterEstimationMethod::MaximumLikelihood); }
                    catch (...) { comp_clone = components_[i]->clone(); }
                }
                auto p = comp_clone->get_parameters();
                int n_comp = static_cast<int>(p.size());
                for (int j = 0; j < n_comp; ++j) {
                    double v = std::isfinite(p[j]) && std::fabs(p[j]) > 1e-10 ? p[j] : sample_mean;
                    initials.push_back(v);
                    double range = std::max(100.0 * std::fabs(v), 100.0 * sample_sd);
                    lowers.push_back(1e-15);
                    uppers.push_back(v + range);
                }
            }
        }

        // Nelder-Mead on total log-likelihood over all component parameters.
        auto log_lh_fn = [this, &sample](const std::vector<double>& x) -> double {
            auto dist_clone = static_cast<CompetingRisks*>(this->clone().release());
            std::unique_ptr<CompetingRisks> dc(dist_clone);
            dc->set_parameters(x);
            double lh = dc->log_likelihood(sample);
            if (std::isnan(lh) || std::isinf(lh)) return -std::numeric_limits<double>::infinity();
            return lh;
        };
        math::optimization::NelderMead solver(log_lh_fn, Np, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }
};

}  // namespace bestfit::numerics::distributions
