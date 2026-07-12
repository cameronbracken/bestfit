// ported from: src/RMC.BestFit/Diagnostics/PriorInfluenceDiagnostics.cs @ fc28c0c
//
// Provides influence diagnostics for the prior components in a Bayesian analysis: how each prior
// component (parameter prior, quantile prior, Jeffreys scale prior, ...) contributes to the
// posterior, the prior-to-data log-likelihood ratio, and a reparameterization-invariant
// per-parameter prior-precision share. The primary constructor evaluates these across a thinned
// posterior MCMC sample.
//
// C++ mapping notes (structural mirroring):
//   - The C# `IModel` is realized in this core by `corehydro::models::ModelBase` (there is no
//     separate IModel header; the estimators all hold `ModelBase&`), so the primary constructor
//     takes `ModelBase& model`. `MCMCResults` is
//     `corehydro::numerics::sampling::mcmc::MCMCResults` (its `Output` -> `.output`, each sample's
//     `.Values` -> `.values`). `Model.DataLogLikelihood(double[])` takes a NON-const reference
//     here (M14 mutable-point convention), so ComputeFromPosterior copies each posterior sample
//     into a mutable working vector before calling `data_log_likelihood`.
//   - The nested C# `readonly struct PriorComponentSummary` ports as a getter-only value class
//     with non-`const` members (same rationale as data_component.hpp / influence_diagnostics.hpp).
//   - `int topN = int.MaxValue` ports as `int top_n = INT_MAX`.
//   - `Dictionary<PriorComponentType, double>` ports as `std::map<PriorComponentType, double>`.
//   - The C# `prior.Variance` throws for improper priors and is caught; here `variance()` is a
//     non-throwing member on UnivariateDistributionBase (std_dev^2), so the try/catch degrades to
//     the same finite-check fall-through (a non-finite or non-positive variance => +infinity =>
//     zero precision), matching the C# behavior for flat / Jeffreys-1/scale priors.
//
// SKIPPED (deliberate): the `XElement` constructors on PriorInfluenceDiagnostics /
// PriorComponentSummary (C# 85 / 509) and `ToXElement()` on both (C# 453 / 587) -- XML, there is
// no XElement port in this core. `PriorComponentSummary.ToString()` (C# 600) -- presentation-only
// (`:F2` assembly), not oracle-checked and not called by any compute path; skipped.
#pragma once
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/models/support/model_base.hpp"
#include "corehydro/models/support/prior_component.hpp"
#include "corehydro/numerics/sampling/mcmc/support/mcmc_results.hpp"

namespace corehydro::diagnostics {

class PriorInfluenceDiagnostics {
   public:
    using MCMCResults = corehydro::numerics::sampling::mcmc::MCMCResults;

    // ------------------------------------------------------------------------------------
    // Nested value type (C# `public readonly struct PriorComponentSummary`, C# 483)
    // ------------------------------------------------------------------------------------

    // Summary statistics for a prior component across posterior samples (C# 494).
    class PriorComponentSummary {
       public:
        PriorComponentSummary(std::string name, corehydro::models::PriorComponentType type,
                              double mean, double standard_deviation, double min, double max)
            : name_(std::move(name)),
              type_(type),
              mean_log_likelihood_(mean),
              standard_deviation_(standard_deviation),
              min_log_likelihood_(min),
              max_log_likelihood_(max) {}

        const std::string& name() const { return name_; }
        corehydro::models::PriorComponentType type() const { return type_; }
        double mean_log_likelihood() const { return mean_log_likelihood_; }
        double standard_deviation() const { return standard_deviation_; }
        double min_log_likelihood() const { return min_log_likelihood_; }
        double max_log_likelihood() const { return max_log_likelihood_; }

        // Coefficient of variation |std dev / mean| (C# 580); 0 when the mean is near zero.
        double coefficient_of_variation() const {
            return std::fabs(mean_log_likelihood_) > 1e-10
                       ? std::fabs(standard_deviation_ / mean_log_likelihood_)
                       : 0.0;
        }

       private:
        std::string name_;
        corehydro::models::PriorComponentType type_;
        double mean_log_likelihood_;
        double standard_deviation_;
        double min_log_likelihood_;
        double max_log_likelihood_;
    };

    // ------------------------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------------------------

    // Creates an empty prior influence diagnostics instance (C# 49): empty components.
    PriorInfluenceDiagnostics() = default;

    // Creates prior influence diagnostics from pre-computed component summaries (C# 59). The C#
    // ArgumentNullException guard has no analogue (C++ vectors are never null). Note (matching
    // the upstream test contract): this constructor does NOT derive Total* from the component
    // means -- those stay 0, so the ratio is 0 and the prior is not flagged influential.
    explicit PriorInfluenceDiagnostics(std::vector<PriorComponentSummary> components)
        : components_(std::move(components)) {
        compute_summary_statistics();
    }

    // Creates prior influence diagnostics from MCMC posterior samples (C# 72). `IModel` is
    // `ModelBase&`; the C# null guards have no analogue for a reference. `thin_every` thins the
    // posterior; default 10.
    PriorInfluenceDiagnostics(corehydro::models::ModelBase& model, const MCMCResults& results,
                              int thin_every = 10) {
        compute_from_posterior(model, results, thin_every);
    }

    // ------------------------------------------------------------------------------------
    // Properties (C# 115-186)
    // ------------------------------------------------------------------------------------

    const std::vector<PriorComponentSummary>& components() const { return components_; }
    int count() const { return static_cast<int>(components_.size()); }
    double total_prior_log_likelihood() const { return total_prior_log_likelihood_; }
    double total_data_log_likelihood() const { return total_data_log_likelihood_; }
    double prior_to_data_ratio() const { return prior_to_data_ratio_; }

    // Whether the prior appears influential based on the prior-to-data ratio (C# 157): the prior
    // contributes more than 20% of the total log-likelihood magnitude (strictly greater).
    bool is_prior_influential() const { return prior_to_data_ratio_ > 0.2; }

    // Per-parameter share of posterior precision contributed by the prior, in [0, 1] (C# 180).
    const std::vector<double>& prior_precision_share() const { return prior_precision_share_; }

    // Mean of prior_precision_share across parameters (C# 186).
    double mean_prior_precision_share() const { return mean_prior_precision_share_; }

    // ------------------------------------------------------------------------------------
    // Methods (C# 367-447)
    // ------------------------------------------------------------------------------------

    // Component summaries of the given prior type (C# 367).
    std::vector<PriorComponentSummary> get_components_by_type(
        corehydro::models::PriorComponentType type) const {
        std::vector<PriorComponentSummary> result;
        for (const auto& c : components_)
            if (c.type() == type) result.push_back(c);
        return result;
    }

    // Components sorted by mean log-likelihood ascending (most constraining first), truncated to
    // `top_n` (C# 380).
    std::vector<PriorComponentSummary> get_most_constraining_components(int top_n = INT_MAX) const {
        std::vector<PriorComponentSummary> sorted = components_;
        std::stable_sort(sorted.begin(), sorted.end(),
                         [](const PriorComponentSummary& a, const PriorComponentSummary& b) {
                             return a.mean_log_likelihood() < b.mean_log_likelihood();
                         });
        int take = std::min(top_n, static_cast<int>(sorted.size()));
        if (take < 0) take = 0;
        sorted.erase(sorted.begin() + take, sorted.end());
        return sorted;
    }

    // Total mean log-likelihood contribution per prior type (C# 395).
    std::map<corehydro::models::PriorComponentType, double> get_contribution_by_type() const {
        std::map<corehydro::models::PriorComponentType, double> result;
        for (const auto& c : components_) result[c.type()] += c.mean_log_likelihood();
        return result;
    }

    // Human-readable summary of the prior influence diagnostics (C# 416). Pure string assembly;
    // the header ("Prior Influence Summary:") / "No prior components" keywords are preserved.
    std::string get_summary() const {
        if (components_.empty()) return "No prior components available for diagnostics.";

        std::string out;
        out += "Prior Influence Summary:\n";
        out += "  Total Prior LL: " + format_f2(total_prior_log_likelihood_) + "\n";
        out += "  Total Data LL: " + format_f2(total_data_log_likelihood_) + "\n";
        out += "  Prior/Total Ratio: " + format_f2(prior_to_data_ratio_ * 100.0) + "%\n";
        out += "  Prior is ";
        out += is_prior_influential() ? "INFLUENTIAL" : "not dominant";
        out += "\n\nContributions by Type:";

        // Sort contributions ascending by value (matching the C# `OrderBy(kv => kv.Value)`).
        auto by_type = get_contribution_by_type();
        std::vector<std::pair<corehydro::models::PriorComponentType, double>> ordered(by_type.begin(),
                                                                                    by_type.end());
        std::stable_sort(ordered.begin(), ordered.end(),
                         [](const auto& a, const auto& b) { return a.second < b.second; });
        for (const auto& kv : ordered) {
            out += "\n  " + prior_type_name(kv.first) + ": " + format_f2(kv.second);
        }
        return out;
    }

    // The component summary at the given index (C# `this[int]`, C# 447).
    const PriorComponentSummary& operator[](int index) const {
        return components_[static_cast<std::size_t>(index)];
    }

   private:
    // Computes prior influence diagnostics from MCMC posterior samples (C# 195).
    void compute_from_posterior(corehydro::models::ModelBase& model, const MCMCResults& results,
                                int thin_every) {
        const auto& samples = results.output;
        int s_count = static_cast<int>(samples.size());
        int step = std::max(1, thin_every);

        // Collect prior components across samples (insertion-ordered to mirror the C#
        // Dictionary enumeration order of first insertion).
        std::vector<std::string> order;
        std::map<std::string, std::vector<double>> component_dict;
        std::map<std::string, corehydro::models::PriorComponentType> type_dict;
        std::vector<double> data_log_likelihoods;

        for (int s = 0; s < s_count; s += step) {
            const std::vector<double>& values = samples[static_cast<std::size_t>(s)].values;

            std::vector<corehydro::models::PriorComponent> prior_components =
                model.pointwise_prior_log_likelihood(values);
            for (const auto& comp : prior_components) {
                if (component_dict.find(comp.name()) == component_dict.end()) {
                    component_dict[comp.name()] = std::vector<double>();
                    type_dict[comp.name()] = comp.type();
                    order.push_back(comp.name());
                }
                component_dict[comp.name()].push_back(comp.log_likelihood());
            }

            // Mutable working copy (data_log_likelihood takes a non-const reference; see header).
            std::vector<double> mutable_values = values;
            data_log_likelihoods.push_back(model.data_log_likelihood(mutable_values));
        }

        // Create component summaries (insertion order).
        components_.clear();
        components_.reserve(order.size());
        for (const std::string& name : order) {
            const std::vector<double>& values = component_dict[name];
            components_.emplace_back(name, type_dict[name], mean(values), compute_std_dev(values),
                                     min_of(values), max_of(values));
        }

        // Overall statistics at the posterior mean/mode.
        total_prior_log_likelihood_ = 0.0;
        for (const auto& c : components_) total_prior_log_likelihood_ += c.mean_log_likelihood();
        total_data_log_likelihood_ = mean(data_log_likelihoods);
        compute_summary_statistics();
        compute_prior_precision_share(model, results, step);
    }

    // Computes the per-parameter share of posterior precision contributed by the prior (C# 258).
    // Reparameterization-invariant under linear changes; bounded in [0, 1].
    void compute_prior_precision_share(corehydro::models::ModelBase& model,
                                       const MCMCResults& results, int step) {
        int p = model.number_of_parameters();
        if (p == 0 || results.output.empty()) {
            prior_precision_share_.clear();
            mean_prior_precision_share_ = 0.0;
            return;
        }

        const auto& samples = results.output;
        int s_count = static_cast<int>(samples.size());

        // Posterior variance per parameter from the MCMC samples.
        std::vector<double> post_var(static_cast<std::size_t>(p));
        for (int i = 0; i < p; ++i) {
            std::size_t si = static_cast<std::size_t>(i);
            double m = 0.0;
            int count = 0;
            for (int s = 0; s < s_count; s += step) {
                m += samples[static_cast<std::size_t>(s)].values[si];
                ++count;
            }
            if (count <= 1) {
                post_var[si] = std::numeric_limits<double>::infinity();
                continue;
            }
            m /= count;
            double sum_sq = 0.0;
            for (int s = 0; s < s_count; s += step) {
                double d = samples[static_cast<std::size_t>(s)].values[si] - m;
                sum_sq += d * d;
            }
            post_var[si] = sum_sq / (count - 1);
        }

        // Prior variance per parameter -- from the prior distribution's analytical variance when
        // finite, otherwise treated as infinite (flat / improper prior => zero precision).
        std::vector<double> prior_var(static_cast<std::size_t>(p));
        for (int i = 0; i < p; ++i) {
            std::size_t si = static_cast<std::size_t>(i);
            double v = std::numeric_limits<double>::infinity();
            try {
                v = model.parameters()[si].prior_distribution().variance();
            } catch (const std::exception&) {
                // Improper prior -- variance undefined (e.g., flat / Jeffreys-1/scale); the
                // fall-through treats it as positive infinity (C# Debug.WriteLine here).
            }
            if (!std::isfinite(v) || v <= 0) v = std::numeric_limits<double>::infinity();
            prior_var[si] = v;
        }

        // Share: precision_prior / max(precision_post, precision_prior), clamped to [0, 1].
        prior_precision_share_.assign(static_cast<std::size_t>(p), 0.0);
        double sum = 0.0;
        int finite_count = 0;
        for (int i = 0; i < p; ++i) {
            std::size_t si = static_cast<std::size_t>(i);
            double prec_prior = std::isinf(prior_var[si]) ? 0.0 : 1.0 / prior_var[si];
            double prec_post = post_var[si] > 0 ? 1.0 / post_var[si] : 0.0;
            double denom = std::max(prec_post, prec_prior);
            double share = denom > 0 ? prec_prior / denom : 0.0;
            if (share < 0) share = 0;
            if (share > 1) share = 1;
            prior_precision_share_[si] = share;
            if (std::isfinite(share)) {
                sum += share;
                ++finite_count;
            }
        }
        mean_prior_precision_share_ = finite_count > 0 ? sum / finite_count : 0.0;
    }

    // Computes the prior-to-data ratio from the component data (C# 334): |prior| / (|prior| +
    // |data|), or 0 when there are no components.
    void compute_summary_statistics() {
        if (components_.empty()) {
            prior_to_data_ratio_ = 0.0;
            return;
        }

        double abs_prior = std::fabs(total_prior_log_likelihood_);
        double abs_data = std::fabs(total_data_log_likelihood_);
        double total = abs_prior + abs_data;
        prior_to_data_ratio_ = total > 0 ? abs_prior / total : 0.0;
    }

    // Standard deviation of an array (C# 352): 0 when fewer than 2 values.
    static double compute_std_dev(const std::vector<double>& values) {
        if (values.size() <= 1) return 0.0;
        double m = mean(values);
        double sum_sq = 0.0;
        for (double v : values) sum_sq += (v - m) * (v - m);
        return std::sqrt(sum_sq / (values.size() - 1));
    }

    static double mean(const std::vector<double>& values) {
        if (values.empty()) return 0.0;
        double sum = 0.0;
        for (double v : values) sum += v;
        return sum / static_cast<double>(values.size());
    }

    static double min_of(const std::vector<double>& values) {
        double m = std::numeric_limits<double>::infinity();
        for (double v : values)
            if (v < m) m = v;
        return m;
    }

    static double max_of(const std::vector<double>& values) {
        double m = -std::numeric_limits<double>::infinity();
        for (double v : values)
            if (v > m) m = v;
        return m;
    }

    // Presentation-only `:F2` approximation (not oracle-checked; see header).
    static std::string format_f2(double value) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.2f", value);
        return std::string(buf);
    }

    // Presentation-only prior-type name (mirrors the C# enum ToString(); not oracle-checked).
    static std::string prior_type_name(corehydro::models::PriorComponentType type) {
        switch (type) {
            case corehydro::models::PriorComponentType::ParameterPrior:
                return "ParameterPrior";
            case corehydro::models::PriorComponentType::QuantilePrior:
                return "QuantilePrior";
            case corehydro::models::PriorComponentType::JeffreysScalePrior:
                return "JeffreysScalePrior";
            case corehydro::models::PriorComponentType::SpatialError:
                return "SpatialError";
            case corehydro::models::PriorComponentType::Jacobian:
                return "Jacobian";
            case corehydro::models::PriorComponentType::OtherPenalty:
                return "OtherPenalty";
            default:
                return "Unknown";
        }
    }

    // ------------------------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------------------------
    std::vector<PriorComponentSummary> components_;
    double total_prior_log_likelihood_ = 0.0;
    double total_data_log_likelihood_ = 0.0;
    double prior_to_data_ratio_ = 0.0;
    std::vector<double> prior_precision_share_;
    double mean_prior_precision_share_ = 0.0;
};

}  // namespace corehydro::diagnostics
