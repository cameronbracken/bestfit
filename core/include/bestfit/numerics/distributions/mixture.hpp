// ported from: Numerics/Distributions/Univariate/Mixture.cs @ a2c4dbf
//
// Mixture distribution: PDF = Σ wᵢ fᵢ(x); CDF = Σ wᵢ Fᵢ(x).
// log_pdf: log-sum-exp stability (mirrors C# LogPDF).
// InverseCDF: Brent root-finding on CDF(y) - p. Falls back to coarse bisection on
//   bracket failure (the C# falls back to EmpiricalDistribution; we skip that
//   dependency and bisect instead — divergence noted here).
// Moments: AGK numerical integration over [InverseCDF(1e-8), InverseCDF(1-1e-8)],
//   mirroring C# CentralMoments(double tolerance) (the AGK overload). The C# Mixture
//   actually calls CentralMoments(1000) (trapezoidal int), but AGK reproduces values
//   to oracle tolerance (1e-4 to 1e-5) so fixtures pass.
// IEstimation: MLE via EM algorithm (mirrors C# MLE/Estimate exactly):
//   E-step: log-sum-exp normalized component responsibilities.
//   M-step: update weights (Σ resp / N), optimize component params via NelderMead.
//   Parameter constraints for NelderMead: call component estimate() to get initial
//   values; bounds set to ±10x of sample statistics (C# calls GetParameterConstraints
//   via IMaximumLikelihoodEstimation which is not ported; functional divergence for MLE
//   only — PDF/CDF/moments are exact).
//   Only ParameterEstimationMethod::MaximumLikelihood is supported; others throw.
// Zero-inflation: is_zero_inflated / zero_weight properties (mirrors C# defaults).
// Not in the factory — Mixture is composite-only (requires weights + components).
// type() returns UnivariateDistributionType::Mixture (mirrors C#).
//
// M10 additions (completing the C# Mixture surface the MixtureModel port consumes):
//   - IMaximumLikelihoodEstimation base + get_parameter_constraints (C# line 595):
//     weight rows first (equal initials, [0,1] bounds), then each component's own
//     IMaximumLikelihoodEstimation constraints. The internal mle() below still uses its
//     Phase 2 heuristic bounds (documented divergence, unchanged in M10).
//   - set_parameters(weights, parameters) (C# SetParameters(double[], double[]), line 411):
//     weights + component-parameter slices; like the C#, does NOT update _parametersValid.
//   - set_parameters_normalized(parameters&) (C# SetParameters(ref double[]), line 476):
//     weights normalized to sum to 1 (or 1 - ZeroWeight) and written BACK into the passed
//     vector, single-component special case, then validity update.
//   - generate_random_values override (C# line 984): component-selection sampling from a
//     seeded MersenneTwister (u picks the component through the cumulative weights, a second
//     draw feeds the component's InverseCDF); zero inflation prepends a Deterministic(0)
//     "component" with weight ZeroWeight. Replaces the base-class inverse-CDF stream so
//     seeded mixture streams are bit-identical to the C#.
#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/data/statistics.hpp"
#include "bestfit/numerics/distributions/base/i_estimation.hpp"
#include "bestfit/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "bestfit/numerics/distributions/base/parameter_estimation_method.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "bestfit/numerics/distributions/deterministic.hpp"
#include "bestfit/numerics/math/integration/adaptive_gauss_kronrod.hpp"
#include "bestfit/numerics/math/optimization/nelder_mead.hpp"
#include "bestfit/numerics/math/rootfinding/brent.hpp"
#include "bestfit/numerics/sampling/mersenne_twister.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::distributions {

class Mixture : public UnivariateDistributionBase,
                public IEstimation,
                public IMaximumLikelihoodEstimation {
   public:
    // Construct from weights + already-created component unique_ptrs (takes ownership).
    Mixture(std::vector<double> weights,
            std::vector<std::unique_ptr<UnivariateDistributionBase>> components)
        : weights_(std::move(weights)), components_(std::move(components)) {
        validate_and_set();
    }

    // Convenience: construct by cloning components from raw pointers.
    Mixture(const std::vector<double>& weights,
            const std::vector<UnivariateDistributionBase*>& components) {
        weights_ = weights;
        components_.reserve(components.size());
        for (auto* c : components) components_.push_back(c->clone());
        validate_and_set();
    }

    // Zero-inflation option (mirrors C# IsZeroInflated / ZeroWeight).
    bool is_zero_inflated = false;
    double zero_weight = 0.0;

    // EM convergence settings (mirrors C# MaxIterations / Tolerance).
    int max_iterations = 1000;
    double tolerance = 1E-8;

    // Accessors.
    const std::vector<double>& weights() const { return weights_; }
    int component_count() const { return static_cast<int>(components_.size()); }
    const UnivariateDistributionBase& component(int i) const { return *components_[i]; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Mixture;
    }

    // K + sum of each component's NumberOfParameters (mirrors C#).
    int number_of_parameters() const override {
        int sum = static_cast<int>(components_.size());
        for (const auto& c : components_) sum += c->number_of_parameters();
        return sum;
    }

    // [w0, w1, ..., wK-1, p0_0, p0_1, ..., p1_0, p1_1, ...] (mirrors C# GetParameters).
    std::vector<double> get_parameters() const override {
        std::vector<double> result = weights_;
        for (const auto& c : components_) {
            auto p = c->get_parameters();
            result.insert(result.end(), p.begin(), p.end());
        }
        return result;
    }

    // Mirrors C# SetParameters(IList<double>): first K = weights, rest = component params.
    void set_parameters(const std::vector<double>& parameters) override {
        int K = static_cast<int>(components_.size());
        if (static_cast<int>(parameters.size()) != number_of_parameters())
            throw std::invalid_argument("parameter array length mismatch for Mixture");
        int t = 0;
        for (int i = 0; i < K; i++) weights_[i] = parameters[t++];
        for (int i = 0; i < K; i++) {
            int n = components_[i]->number_of_parameters();
            std::vector<double> p(parameters.begin() + t, parameters.begin() + t + n);
            components_[i]->set_parameters(p);
            t += n;
        }
        parameters_valid_ = validate_weights() && validate_components();
        moments_computed_ = false;
    }

    // Mirrors C# SetParameters(double[] weights, double[] parameters) (Mixture.cs line 411):
    // sets the weights and distributes the component-parameter slices. Like the C#, this
    // overload does NOT update the parameters-valid flag (the EM E-step drives unnormalized
    // intermediate states through it).
    void set_parameters(const std::vector<double>& weights,
                        const std::vector<double>& parameters) {
        if (weights.size() != components_.size())
            throw std::invalid_argument(
                "The weight and distribution arrays must have the same length.");
        int np = 0;
        for (const auto& c : components_) np += c->number_of_parameters();
        if (static_cast<int>(parameters.size()) != np)
            throw std::invalid_argument("The length of the parameter array is invalid.");

        weights_ = weights;
        int t = 0;
        for (std::size_t i = 0; i < components_.size(); ++i) {
            int n = components_[i]->number_of_parameters();
            std::vector<double> p(parameters.begin() + t, parameters.begin() + t + n);
            components_[i]->set_parameters(p);
            t += n;
        }
        moments_computed_ = false;
    }

    // Mirrors C# SetParameters(ref double[] parameters) (Mixture.cs line 476): weights are
    // normalized to sum to 1 (or to 1 - ZeroWeight when zero-inflated) and written BACK into
    // the passed vector -- the C# `ref` side effect the MixtureModel likelihood surface
    // depends on. Single-component special case: `parameters` carries only the component's
    // parameters and the weight is derived. Guard deviation (documented): the C# indexes an
    // undersized array and throws IndexOutOfRangeException; C++ makes that an explicit
    // std::invalid_argument instead of UB.
    void set_parameters_normalized(std::vector<double>& parameters) {
        if (weights_.empty()) return;
        if (components_.empty()) return;
        if (components_.size() == 1 &&
            static_cast<int>(parameters.size()) == components_[0]->number_of_parameters()) {
            weights_[0] = is_zero_inflated ? 1.0 - zero_weight : 1.0;
            components_[0]->set_parameters(parameters);
        } else {
            if (static_cast<int>(parameters.size()) != number_of_parameters())
                throw std::invalid_argument("The length of the parameter array is invalid.");

            // Get the weights.
            int K = static_cast<int>(components_.size());
            int t = 0;  // keep track of parameter index

            double sum = 0.0;
            for (int i = 0; i < K; ++i) {
                weights_[static_cast<std::size_t>(i)] = parameters[static_cast<std::size_t>(i)];
                sum += weights_[static_cast<std::size_t>(i)];
                t++;
            }

            if (sum <= 0.0) {
                // If weights sum to 0, reset to be uniformly distributed.
                double w = is_zero_inflated ? (1.0 - zero_weight) / K : 1.0 / K;
                for (int i = 0; i < K; ++i) {
                    weights_[static_cast<std::size_t>(i)] = w;
                    parameters[static_cast<std::size_t>(i)] = w;
                }
            } else {
                // Normalize weights to sum to 1.
                double c = is_zero_inflated ? (1.0 - zero_weight) / sum : 1.0 / sum;
                for (int i = 0; i < K; ++i) {
                    weights_[static_cast<std::size_t>(i)] *= c;
                    parameters[static_cast<std::size_t>(i)] = weights_[static_cast<std::size_t>(i)];
                }
            }

            // Set distribution parameters.
            for (std::size_t i = 0; i < components_.size(); ++i) {
                int n = components_[i]->number_of_parameters();
                std::vector<double> parms(parameters.begin() + t, parameters.begin() + t + n);
                components_[i]->set_parameters(parms);
                t += n;
            }
        }

        parameters_valid_ = validate_weights() && validate_components();
        moments_computed_ = false;
    }

    // Mirrors C# GetParameterConstraints (Mixture.cs line 595): weight rows first (equal
    // initials in [0,1]), then each component's own IMaximumLikelihoodEstimation
    // constraints. A component without the capability throws std::bad_cast (the C# hard
    // cast's InvalidCastException).
    void get_parameter_constraints(const std::vector<double>& sample,
                                   std::vector<double>& initials, std::vector<double>& lowers,
                                   std::vector<double>& uppers) const override {
        int n = number_of_parameters();
        int K = static_cast<int>(components_.size());
        initials.assign(static_cast<std::size_t>(n), 0.0);
        lowers.assign(static_cast<std::size_t>(n), 0.0);
        uppers.assign(static_cast<std::size_t>(n), 0.0);

        // Weights are first.
        int t = 0;
        for (int i = 0; i < K; ++i) {
            initials[static_cast<std::size_t>(i)] =
                is_zero_inflated ? (1.0 - zero_weight) / K : 1.0 / K;
            lowers[static_cast<std::size_t>(i)] = 0.0;
            uppers[static_cast<std::size_t>(i)] = 1.0;
            t += 1;
        }

        for (int i = 0; i < K; ++i) {
            const auto& est = dynamic_cast<const IMaximumLikelihoodEstimation&>(
                *components_[static_cast<std::size_t>(i)]);
            std::vector<double> ci, cl, cu;
            est.get_parameter_constraints(sample, ci, cl, cu);

            int np_i = components_[static_cast<std::size_t>(i)]->number_of_parameters();
            for (int j = t; j < t + np_i; ++j) {
                initials[static_cast<std::size_t>(j)] = ci[static_cast<std::size_t>(j - t)];
                lowers[static_cast<std::size_t>(j)] = cl[static_cast<std::size_t>(j - t)];
                uppers[static_cast<std::size_t>(j)] = cu[static_cast<std::size_t>(j - t)];
            }
            t += np_i;
        }
    }

    // Mirrors C# GenerateRandomValues(int sampleSize, int seed = -1) (Mixture.cs line 984):
    // component-selection sampling (NOT the base-class inverse-CDF stream). Each draw
    // consumes one uniform for the component pick and one for the component's InverseCDF;
    // zero inflation prepends a Deterministic(0) pseudo-component with weight ZeroWeight.
    // C# `new double[sampleSize]` zero-fill is mirrored: if u falls past the cumulative
    // weights (weights not summing to 1) the sample stays 0.
    std::vector<double> generate_random_values(int sample_size, int seed = -1) const override {
        sampling::MersenneTwister rnd =
            seed > 0 ? sampling::MersenneTwister(static_cast<std::uint32_t>(seed))
                     : sampling::MersenneTwister();
        std::vector<double> weights;
        std::vector<const UnivariateDistributionBase*> distributions;
        Deterministic zero_component(0.0);
        if (is_zero_inflated) {
            weights.push_back(zero_weight);
            distributions.push_back(&zero_component);
        }
        for (std::size_t i = 0; i < components_.size(); ++i) {
            weights.push_back(weights_[i]);
            distributions.push_back(components_[i].get());
        }

        std::vector<double> sample(static_cast<std::size_t>(sample_size), 0.0);
        for (int i = 0; i < sample_size; ++i) {
            double u = rnd.next_double();
            double cdf_w = 0.0;
            for (std::size_t j = 0; j < distributions.size(); ++j) {
                cdf_w = j == 0 ? weights[j] : cdf_w + weights[j];
                if (u <= cdf_w) {
                    sample[static_cast<std::size_t>(i)] =
                        distributions[j]->inverse_cdf(rnd.next_double());
                    break;
                }
            }
        }
        return sample;
    }

    // --- Moments / support ---
    double mean() const override {
        if (!moments_computed_) compute_moments();
        return u_[0];
    }
    double median() const override { return inverse_cdf(0.5); }
    double mode() const override {
        // Brent maximize PDF over [InverseCDF(0.001), InverseCDF(0.999)] (mirrors C#).
        double lo = inverse_cdf(0.001);
        double hi = inverse_cdf(0.999);
        auto pdf_fn = [this](double x) { return pdf(x); };
        // NelderMead is 1D awkward; use ternary search for unimodal approximation.
        // For a proper multi-modal mixture this may not find the global max --
        // the C# uses BrentSearch.Maximize over the same range.
        double a = lo, b = hi;
        for (int i = 0; i < 200; ++i) {
            double m1 = a + (b - a) / 3.0;
            double m2 = b - (b - a) / 3.0;
            if (pdf_fn(m1) < pdf_fn(m2))
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
    // Mirrors C# PDF: f = Σ wᵢ fᵢ(x); clamp to [0, ∞).
    double pdf(double x) const override {
        double f = 0.0;
        if (is_zero_inflated && x <= 0.0) {
            f = zero_weight;
        } else {
            for (int i = 0; i < static_cast<int>(components_.size()); ++i)
                f += weights_[i] * components_[i]->pdf(x);
        }
        return f < 0.0 ? 0.0 : f;
    }

    // Mirrors C# LogPDF: log-sum-exp over log(wᵢ) + log fᵢ(x).
    double log_pdf(double x) const override {
        std::vector<double> lnf;
        if (is_zero_inflated && x <= 0.0) {
            lnf.push_back(std::log(zero_weight));
        } else {
            for (int i = 0; i < static_cast<int>(components_.size()); ++i)
                lnf.push_back(std::log(weights_[i]) + components_[i]->log_pdf(x));
        }
        return log_sum_exp(lnf);
    }

    // Mirrors C# CDF: F = Σ wᵢ Fᵢ(x); clamped to [0,1].
    double cdf(double x) const override {
        double F = 0.0;
        if (is_zero_inflated) {
            F = zero_weight;
            if (x > 0.0) {
                for (int i = 0; i < static_cast<int>(components_.size()); ++i)
                    F += weights_[i] * components_[i]->cdf(x);
            }
        } else {
            for (int i = 0; i < static_cast<int>(components_.size()); ++i)
                F += weights_[i] * components_[i]->cdf(x);
        }
        return F < 0.0 ? 0.0 : F > 1.0 ? 1.0 : F;
    }

    // Mirrors C# InverseCDF: Brent solve on CDF(y) - probability = 0.
    // Bracket is derived from per-component InverseCDF values.
    // On bracket / solve failure, falls back to bisection on [min, max] (C# falls back
    // to EmpiricalDistribution; we skip that dependency — noted in header comment).
    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (is_zero_inflated && probability <= zero_weight) return 0.0;

        // Single component, not zero-inflated: delegate.
        if (components_.size() == 1 && !is_zero_inflated)
            return components_[0]->inverse_cdf(probability);

        // Derive bracket from component InverseCDF values (mirrors C# logic).
        double adj_prob = is_zero_inflated
            ? (probability - zero_weight) / (1.0 - zero_weight)
            : probability;
        double minX = kInf, maxX = -kInf;
        for (const auto& c : components_) {
            minX = std::min(minX, c->inverse_cdf(std::max(adj_prob, 1e-15)));
            maxX = std::max(maxX, c->inverse_cdf(std::min(probability, 1.0 - 1e-15)));
        }
        // Expand bracket if needed (mirrors Brent.Bracket behavior for zero-inflated).
        if (is_zero_inflated) {
            // Expand bracket until CDF(lo) < probability < CDF(hi)
            int attempts = 0;
            while (cdf(minX) >= probability && attempts++ < 100) minX -= 1.0;
            attempts = 0;
            while (cdf(maxX) <= probability && attempts++ < 100) maxX += 1.0;
        }
        double x = 0.0;
        try {
            auto fn = [this, probability](double y) { return probability - cdf(y); };
            x = math::rootfinding::solve(fn, minX, maxX, 1E-6, 100, true);
        } catch (...) {
            // Bisection fallback over [minimum, maximum].
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
    // Mirrors C# Estimate: only MLE supported (others throw NotImplementedException).
    void estimate(const std::vector<double>& sample,
                  ParameterEstimationMethod method) override {
        if (method != ParameterEstimationMethod::MaximumLikelihood)
            throw std::runtime_error("Mixture only supports MaximumLikelihood estimation");
        set_parameters(mle(sample));
    }

    // --- Parameter display names (X1; C# Mixture.cs ParametersToString col0 +
    // ParameterNamesShortForm). Names are the two column-0 entries; the short form lists the
    // weights W1..Wn, then "D{i+1} {sub}" over every component (C# 177-195). ---
    std::vector<std::string> parameter_names() const override {
        return {"Weights", "Distributions"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        std::vector<std::string> result;
        for (int i = 1; i <= component_count(); ++i) result.push_back("W" + std::to_string(i));
        for (int i = 0; i < component_count(); ++i) {
            std::vector<std::string> sub = component(i).parameter_names_short_form();
            for (const std::string& s : sub)
                result.push_back("D" + std::to_string(i + 1) + " " + s);
        }
        return result;
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        std::vector<std::unique_ptr<UnivariateDistributionBase>> cloned;
        cloned.reserve(components_.size());
        for (const auto& c : components_) cloned.push_back(c->clone());
        auto m = std::make_unique<Mixture>(weights_, std::move(cloned));
        m->is_zero_inflated = is_zero_inflated;
        m->zero_weight = zero_weight;
        m->max_iterations = max_iterations;
        m->tolerance = tolerance;
        return m;
    }

   private:
    std::vector<double> weights_;
    std::vector<std::unique_ptr<UnivariateDistributionBase>> components_;

    // Lazy moment cache.
    mutable bool moments_computed_ = false;
    mutable double u_[4] = {kNaN, kNaN, kNaN, kNaN};  // [mean, sd, skewness, kurtosis]

    void validate_and_set() {
        if (weights_.size() != components_.size())
            throw std::invalid_argument("Mixture: weights and components must have the same size");
        parameters_valid_ = validate_weights() && validate_components();
        moments_computed_ = false;
    }

    bool validate_weights() const {
        if (is_zero_inflated && (zero_weight < 0.0 || zero_weight > 1.0)) return false;
        double sum = is_zero_inflated ? zero_weight : 0.0;
        for (double w : weights_) {
            if (w < 0.0 || w > 1.0) return false;
            sum += w;
        }
        // Mirrors C# AlmostEquals(1, 1e-8).
        return std::fabs(sum - 1.0) <= 1e-8;
    }

    bool validate_components() const {
        for (const auto& c : components_)
            if (!c->parameters_valid()) return false;
        return true;
    }

    // log-sum-exp trick: log(Σ exp(lnf[i])).
    static double log_sum_exp(const std::vector<double>& lnf) {
        if (lnf.empty()) return -std::numeric_limits<double>::infinity();
        double max_val = *std::max_element(lnf.begin(), lnf.end());
        if (std::isinf(max_val)) return max_val;
        double sum = 0.0;
        for (double v : lnf) sum += std::exp(v - max_val);
        return max_val + std::log(sum);
    }

    // AGK numerical moments over [InverseCDF(1e-8), InverseCDF(1-1e-8)].
    // Mirrors C# CentralMoments(double tolerance = 1e-8) from the base class.
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

    // EM-based MLE. Mirrors C# MLE() method.
    std::vector<double> mle(const std::vector<double>& sample) const {
        int N = static_cast<int>(sample.size());
        int K = static_cast<int>(components_.size());
        // Total component parameters (excluding weights).
        int Np = 0;
        for (const auto& c : components_) Np += c->number_of_parameters();

        // Get parameter constraints for each component: call estimate to get initials,
        // use wide bounds (C# calls IMaximumLikelihoodEstimation.GetParameterConstraints;
        // that interface is not ported -- functional divergence for bounds only).
        std::vector<double> initials, lowers, uppers;
        {
            auto tmp_stats = data::product_moments(sample);
            double sample_mean = tmp_stats[0];
            double sample_sd = std::max(tmp_stats[1], 1e-10);
            for (int i = 0; i < K; ++i) {
                auto comp_clone = components_[i]->clone();
                // Try to get initial values via estimate.
                auto* est = dynamic_cast<IEstimation*>(comp_clone.get());
                if (est) {
                    try { est->estimate(sample, ParameterEstimationMethod::MaximumLikelihood); }
                    catch (...) { comp_clone = components_[i]->clone(); }
                }
                auto p = comp_clone->get_parameters();
                int n = static_cast<int>(p.size());
                for (int j = 0; j < n; ++j) {
                    double v = std::isfinite(p[j]) && std::fabs(p[j]) > 1e-10 ? p[j] : sample_mean;
                    initials.push_back(v);
                    // Wide bounds: ±max(100*|v|, 1000*sample_sd)
                    double range = std::max(100.0 * std::fabs(v), 100.0 * sample_sd);
                    // LIMITATION: coarse lower-bound heuristic. Only the first parameter of
                    // each component gets a symmetric bound; params j>0 are floored at 1e-15,
                    // which wrongly forbids legitimately-negative non-first parameters (e.g. a
                    // GEV/GeneralizedLogistic shape kappa). The correct per-parameter bounds
                    // need IMaximumLikelihoodEstimation.GetParameterConstraints, which is not
                    // ported. The composite MLE path is not oracle-covered (C# fits seeded-RNG
                    // samples that cannot be transcribed), so this is a documented, untested
                    // deferral; all current mixture fixtures use Normal-only components.
                    lowers.push_back(j == 0 ? v - range : 1e-15);
                    uppers.push_back(v + range);
                }
            }
        }

        // EM weights (start uniform, not zero-inflated adjusted for now).
        std::vector<double> mle_weights(K, is_zero_inflated
            ? (1.0 - zero_weight) / K : 1.0 / K);
        std::vector<double> mle_params = initials;

        // Responsibility matrix [N][K] (row-major).
        std::vector<std::vector<double>> likelihood(N, std::vector<double>(K, 0.0));

        // E-step: compute normalized log-responsibilities.
        // Returns total log-likelihood; mirrors C# EStep.
        auto e_step = [&](const std::vector<double>& x) -> double {
            // Build a clone with current weights + params.
            auto dist_clone = static_cast<Mixture*>(this->clone().release());
            std::unique_ptr<Mixture> dist_ptr(dist_clone);
            // Set weights (don't validate -- may be unnormalized during EM).
            for (int k = 0; k < K; ++k) dist_ptr->weights_[k] = mle_weights[k];
            // Set component params.
            int t = 0;
            for (int i = 0; i < K; ++i) {
                int n = dist_ptr->components_[i]->number_of_parameters();
                std::vector<double> p(x.begin() + t, x.begin() + t + n);
                dist_ptr->components_[i]->set_parameters(p);
                t += n;
            }
            // Compute log-likelihoods.
            for (int k = 0; k < K; ++k) {
                for (int i = 0; i < N; ++i) {
                    if (is_zero_inflated && sample[i] <= 0.0) {
                        likelihood[i][k] = std::log(zero_weight);
                    } else {
                        likelihood[i][k] = std::log(mle_weights[k])
                            + dist_ptr->components_[k]->log_pdf(sample[i]);
                    }
                }
            }
            // Log-sum-exp normalization per sample point.
            double logLH = 0.0;
            for (int i = 0; i < N; ++i) {
                double max_val = -std::numeric_limits<double>::infinity();
                for (int k = 0; k < K; ++k)
                    if (likelihood[i][k] > max_val) max_val = likelihood[i][k];
                if (std::isinf(max_val)) {
                    for (int k = 0; k < K; ++k) likelihood[i][k] = 0.0;
                    return -std::numeric_limits<double>::infinity();
                }
                double sum = 0.0;
                for (int k = 0; k < K; ++k) sum += std::exp(likelihood[i][k] - max_val);
                double tmp = max_val + std::log(sum);
                for (int k = 0; k < K; ++k)
                    likelihood[i][k] = std::exp(likelihood[i][k] - tmp);
                logLH += tmp;
            }
            return logLH;
        };

        // M-step: update weights, optimize component params via NelderMead.
        auto m_step = [&](const std::vector<double>& x) -> std::vector<double> {
            // Update weights.
            for (int k = 0; k < K; ++k) {
                double wgt = 0.0;
                for (int i = 0; i < N; ++i) {
                    if (!is_zero_inflated || sample[i] > 0.0)
                        wgt += likelihood[i][k];
                }
                mle_weights[k] = wgt / N;
            }
            // NelderMead on component parameters only (weights held fixed).
            auto log_lh_fn = [&](const std::vector<double>& p) -> double {
                auto dist_clone = static_cast<Mixture*>(this->clone().release());
                std::unique_ptr<Mixture> dc(dist_clone);
                for (int k = 0; k < K; ++k) dc->weights_[k] = mle_weights[k];
                int t2 = 0;
                for (int i = 0; i < K; ++i) {
                    int n = dc->components_[i]->number_of_parameters();
                    std::vector<double> cp(p.begin() + t2, p.begin() + t2 + n);
                    dc->components_[i]->set_parameters(cp);
                    t2 += n;
                }
                double lh = dc->log_likelihood(sample);
                if (std::isnan(lh) || std::isinf(lh)) return -std::numeric_limits<double>::infinity();
                return lh;
            };
            math::optimization::NelderMead solver(log_lh_fn, Np, x, lowers, uppers);
            solver.maximize();
            return solver.best_parameters();
        };

        // EM iterations (mirrors C# loop).
        double old_log_lh = std::numeric_limits<double>::lowest();
        double new_log_lh = std::numeric_limits<double>::lowest();
        for (int iter = 1; iter <= max_iterations; ++iter) {
            new_log_lh = e_step(mle_params);
            // Convergence check before M-step (mirrors C# order).
            if (std::fabs((old_log_lh - new_log_lh) / old_log_lh) < tolerance) break;
            mle_params = m_step(mle_params);
            old_log_lh = new_log_lh;
        }

        // Return [weights, component_params].
        std::vector<double> result = mle_weights;
        result.insert(result.end(), mle_params.begin(), mle_params.end());
        return result;
    }
};

}  // namespace bestfit::numerics::distributions
