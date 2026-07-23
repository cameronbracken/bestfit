// ported from: src/RMC.BestFit/Diagnostics/LeverageDiagnostics.cs @ c2e6192
//
// Provides influence diagnostics that decompose each observation's and prior's impact at the
// MAP estimate into two dimensions: fit influence (Cook's Distance) and variance influence
// (normalized leverage). At the MAP estimate the posterior Fisher information decomposes
// additively J_post = sum_i J_i + J_prior; leverage = FitInfluence + VarianceInfluence, with
// FitInfluence = g_i' J_post^{-1} g_i / p (Cook's Distance) and VarianceInfluence given by the
// trace method for observations (|tr(J_post^{-1} . J_i)| / p) and the generalized-variance
// determinant method for priors. See the C# class remarks for the full derivation and
// references (Cook 1977; Marshall & Spiegelhalter 2025; Wei, Hu & Fung 1998).
//
// C++ mapping notes (structural mirroring):
//   - The C# `IModel` is realized in this core by `corehydro::models::ModelBase` (there is no
//     separate IModel header; the estimators all hold `ModelBase&`), so the fitting
//     constructor takes `ModelBase& model`. `model.LogLikelihood` is
//     `model.log_likelihood(std::vector<double>&)` (non-const ref, M14 mutable-point
//     convention); the numerical-Hessian helper is `NumericalDiff::compute_hessian`, which
//     takes exactly that `std::function<double(std::vector<double>&)>` signature -- reused
//     rather than duplicating the finite-difference math.
//   - The two nested C# `public readonly struct`s (ObservationLeverage, PriorComponentLeverage)
//     port as getter-only value classes, matching DataComponent/PriorComponent house style
//     (see data_component.hpp's rationale for not marking members `const`: it would delete the
//     copy/move assignment `std::vector<...>` needs for the UpdatePercentages back-fill).
//
// SKIPPED (deliberate, XML/serialization -- there is no XElement port in this core): the
// `XElement` constructors on LeverageDiagnostics / ObservationLeverage / PriorComponentLeverage
// (C# 134 / 804 / 937), and `ToXElement()` on all three (C# 269 / 883 / 997). Every other
// member of the C# class is ported.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/estimation/numerical_diff.hpp"
#include "corehydro/models/support/data_component.hpp"
#include "corehydro/models/support/model_base.hpp"
#include "corehydro/models/support/prior_component.hpp"
#include "corehydro/numerics/math/linalg/lu_decomposition.hpp"  // defines Matrix::inverse/determinant
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/matrix_regularization.hpp"

namespace corehydro::diagnostics {

class LeverageDiagnostics {
   public:
    using Matrix = corehydro::numerics::math::linalg::Matrix;

    // ------------------------------------------------------------------------------------
    // Nested value types (C# `public readonly struct`s)
    // ------------------------------------------------------------------------------------

    // Represents the influence of a single observation decomposed into fit and variance
    // components (C# ObservationLeverage, 766). The full-field constructor (C# 782) defaults
    // `count` to 1 and `name` to none, so the "short" call used at C# test line 96 is the same
    // constructor with those defaults.
    class ObservationLeverage {
       public:
        ObservationLeverage(int index, double leverage, double percent_of_total,
                            double fit_influence, double variance_influence,
                            double percent_fit_of_total, double percent_variance_of_total,
                            double value, corehydro::models::DataComponentType data_type,
                            int count = 1, std::optional<std::string> name = std::nullopt)
            : index_(index),
              leverage_(leverage),
              percent_of_total_(percent_of_total),
              fit_influence_(fit_influence),
              variance_influence_(variance_influence),
              percent_fit_of_total_(percent_fit_of_total),
              percent_variance_of_total_(percent_variance_of_total),
              value_(value),
              data_type_(data_type),
              count_(count),
              name_(std::move(name)) {}

        int index() const { return index_; }
        double leverage() const { return leverage_; }
        double percent_of_total() const { return percent_of_total_; }
        double fit_influence() const { return fit_influence_; }
        double variance_influence() const { return variance_influence_; }
        double percent_fit_of_total() const { return percent_fit_of_total_; }
        double percent_variance_of_total() const { return percent_variance_of_total_; }
        double value() const { return value_; }
        corehydro::models::DataComponentType data_type() const { return data_type_; }
        int count() const { return count_; }
        const std::optional<std::string>& name() const { return name_; }

       private:
        int index_;
        double leverage_;
        double percent_of_total_;
        double fit_influence_;
        double variance_influence_;
        double percent_fit_of_total_;
        double percent_variance_of_total_;
        double value_;
        corehydro::models::DataComponentType data_type_;
        int count_;
        std::optional<std::string> name_;
    };

    // Represents the influence of a single prior component decomposed into fit and variance
    // components (C# PriorComponentLeverage, 905; full constructor C# 918).
    class PriorComponentLeverage {
       public:
        PriorComponentLeverage(std::string name, corehydro::models::PriorComponentType type,
                               double leverage, double percent_of_total, double fit_influence,
                               double variance_influence, double percent_fit_of_total,
                               double percent_variance_of_total)
            : name_(std::move(name)),
              type_(type),
              leverage_(leverage),
              percent_of_total_(percent_of_total),
              fit_influence_(fit_influence),
              variance_influence_(variance_influence),
              percent_fit_of_total_(percent_fit_of_total),
              percent_variance_of_total_(percent_variance_of_total) {}

        const std::string& name() const { return name_; }
        corehydro::models::PriorComponentType type() const { return type_; }
        double leverage() const { return leverage_; }
        double percent_of_total() const { return percent_of_total_; }
        double fit_influence() const { return fit_influence_; }
        double variance_influence() const { return variance_influence_; }
        double percent_fit_of_total() const { return percent_fit_of_total_; }
        double percent_variance_of_total() const { return percent_variance_of_total_; }

       private:
        std::string name_;
        corehydro::models::PriorComponentType type_;
        double leverage_;
        double percent_of_total_;
        double fit_influence_;
        double variance_influence_;
        double percent_fit_of_total_;
        double percent_variance_of_total_;
    };

    // ------------------------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------------------------

    // Constructs an empty leverage diagnostics instance (C# 79): empty arrays, all scalar
    // totals 0 (via the default member initializers below).
    LeverageDiagnostics() = default;

    // Constructs leverage diagnostics by computing the Hessian numerically at the given MAP
    // parameter values (C# 98). The C# `IModel model` is `ModelBase& model` here; the C#
    // null-argument guards are structurally unrepresentable for a reference. Throws
    // `std::invalid_argument` (C# ArgumentException) when `map_values` length does not match
    // the model parameter count.
    LeverageDiagnostics(corehydro::models::ModelBase& model, const std::vector<double>& map_values)
        : number_of_parameters_(static_cast<int>(map_values.size())) {
        if (static_cast<int>(map_values.size()) != model.number_of_parameters())
            throw std::invalid_argument(
                "mapValues length must match model parameter count.");
        compute_leverages(model, map_values);
    }

    // Constructs leverage diagnostics from pre-computed observation and prior component
    // leverages (C# 121). C#'s null-coalesce to empty is implicit here (C++ vectors are never
    // null); computes summary statistics and back-fills each ordinate's percentages.
    LeverageDiagnostics(std::vector<ObservationLeverage> observations,
                        std::vector<PriorComponentLeverage> prior_components,
                        int number_of_parameters)
        : observations_(std::move(observations)),
          prior_components_(std::move(prior_components)),
          number_of_parameters_(number_of_parameters) {
        compute_summary_statistics();
        update_percentages();
    }

    // ------------------------------------------------------------------------------------
    // Properties (C# 166-226)
    // ------------------------------------------------------------------------------------

    const std::vector<ObservationLeverage>& observations() const { return observations_; }
    const std::vector<PriorComponentLeverage>& prior_components() const {
        return prior_components_;
    }
    int number_of_parameters() const { return number_of_parameters_; }
    int count() const { return static_cast<int>(observations_.size()); }
    double total_observation_leverage() const { return total_observation_leverage_; }
    double total_prior_leverage() const { return total_prior_leverage_; }
    double total_leverage() const { return total_leverage_; }
    double observation_fit_influence() const { return observation_fit_influence_; }
    double observation_variance_influence() const { return observation_variance_influence_; }
    double prior_fit_influence() const { return prior_fit_influence_; }
    double prior_variance_influence() const { return prior_variance_influence_; }
    double total_fit_influence() const { return total_fit_influence_; }
    double total_variance_influence() const { return total_variance_influence_; }

    // ------------------------------------------------------------------------------------
    // Methods
    // ------------------------------------------------------------------------------------

    // Returns the most influential observations sorted by leverage descending (C# 237).
    // `top_n` larger than the count returns all; an empty instance returns empty.
    std::vector<ObservationLeverage> get_most_influential_observations(int top_n) const {
        std::vector<ObservationLeverage> sorted = observations_;
        std::stable_sort(sorted.begin(), sorted.end(),
                         [](const ObservationLeverage& a, const ObservationLeverage& b) {
                             return a.leverage() > b.leverage();
                         });
        int take = std::min(top_n, static_cast<int>(sorted.size()));
        if (take < 0) take = 0;
        // resize() would require a default constructor (DefaultInsertable); erase the tail
        // instead since `take <= size`.
        sorted.erase(sorted.begin() + take, sorted.end());
        return sorted;
    }

    // Returns a human-readable summary of the leverage diagnostics (C# 249). Presentation text
    // only; the `:F1` format is approximated with `%.1f` (not oracle-checked).
    std::string get_summary() const {
        if (total_leverage_ <= 0) return "No leverage diagnostics available.";

        double total_fv = total_fit_influence_ + total_variance_influence_;
        if (total_fv <= 0) total_fv = total_leverage_;  // fallback

        double obs_var_pct = total_fv > 0 ? observation_variance_influence_ / total_fv * 100.0 : 0;
        double obs_fit_pct = total_fv > 0 ? observation_fit_influence_ / total_fv * 100.0 : 0;
        double prior_var_pct = total_fv > 0 ? prior_variance_influence_ / total_fv * 100.0 : 0;
        double prior_fit_pct = total_fv > 0 ? prior_fit_influence_ / total_fv * 100.0 : 0;

        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "p = %d. Data: %.1f%% of variance info, %.1f%% of fit influence. "
                      "Priors: %.1f%% of variance info, %.1f%% of fit influence.",
                      number_of_parameters_, obs_var_pct, obs_fit_pct, prior_var_pct,
                      prior_fit_pct);
        return std::string(buf);
    }

    // ------------------------------------------------------------------------------------
    // Public statics (C# 585, 599) -- called by GMM penalty diagnostics
    // ------------------------------------------------------------------------------------

    // Public accessor for the numerical Hessian (C# 585). Delegates to the ported
    // NumericalDiff::compute_hessian (the private numerical-Hessian in this port), not a
    // duplicated finite-difference routine.
    static Matrix compute_numerical_hessian_public(
        const std::function<double(std::vector<double>&)>& function,
        const std::vector<double>& parameters, int p) {
        return corehydro::estimation::NumericalDiff::compute_hessian(function, parameters, p);
    }

    // Public accessor for generalized variance influence (C# 599). Computes
    // |log(|det(Sigma_{-k})|) - log(|det(Sigma)|)| / p by evaluating the reduced objective's
    // Hessian.
    static double compute_gen_var_public(
        const std::function<double(std::vector<double>&)>& log_likelihood_without,
        const Matrix& sigma, const std::vector<double>& map_values, int p) {
        return compute_generalized_variance_influence(log_likelihood_without, sigma, map_values, p);
    }

   private:
    // ------------------------------------------------------------------------------------
    // Private methods (C# 297-757)
    // ------------------------------------------------------------------------------------

    // Computes leverages from the model and MAP values using a numerical Hessian via central
    // differences (C# 297). On any inversion or computation failure the arrays are cleared
    // (silent guard; the C# logs via Debug.WriteLine).
    void compute_leverages(corehydro::models::ModelBase& model,
                           const std::vector<double>& map_values) {
        int p = static_cast<int>(map_values.size());

        try {
            // Step 1: posterior Hessian via central differences on Model.LogLikelihood.
            Matrix hessian = compute_numerical_hessian_public(
                [&model](std::vector<double>& x) { return model.log_likelihood(x); }, map_values,
                p);

            // Step 2: invert the negative Hessian (Fisher information at MAP).
            Matrix neg_hessian = hessian * -1.0;
            Matrix hessian_inv(p, p);
            try {
                hessian_inv = neg_hessian.inverse();
            } catch (const std::exception&) {
                // Direct inversion failed -- try regularization (C# Debug.WriteLine here).
                try {
                    neg_hessian = corehydro::numerics::math::linalg::MatrixRegularization::
                        make_symmetric_positive_definite(neg_hessian);
                    hessian_inv = neg_hessian.inverse();
                } catch (const std::exception&) {
                    // Regularized inversion also failed -- empty result (silent guard).
                    observations_.clear();
                    prior_components_.clear();
                    return;
                }
            }

            // Step 3: per-observation Cook's D and variance influence.
            compute_observation_leverages(model, map_values, p, hessian_inv);

            // Step 4: per-prior-component Cook's D and variance influence.
            compute_prior_component_leverages(model, map_values, p, hessian_inv);

            // Step 5: summary statistics and percentages.
            compute_summary_statistics();
            update_percentages();
        } catch (const std::exception&) {
            observations_.clear();
            prior_components_.clear();
        }
    }

    // Computes per-observation score vectors and their leverages (C# 381), following the exact
    // forward/backward caching + flat-spot escalation pattern of the C# source.
    void compute_observation_leverages(corehydro::models::ModelBase& model,
                                       const std::vector<double>& map_values, int p,
                                       const Matrix& hessian_inv) {
        namespace nd = corehydro::estimation;
        std::vector<double> base_pointwise_ll = model.pointwise_data_log_likelihood(map_values);
        int n = static_cast<int>(base_pointwise_ll.size());

        std::vector<double> perturbed_params = map_values;
        std::vector<double> step = nd::NumericalDiff::compute_step_sizes(map_values);

        // Cache forward/backward evaluations for diagonal terms, with flat-spot detection and
        // step escalation per parameter.
        std::vector<std::vector<double>> fwd_vals(static_cast<std::size_t>(p));
        std::vector<std::vector<double>> bwd_vals(static_cast<std::size_t>(p));
        std::vector<bool> have(static_cast<std::size_t>(p), false);  // C# `fwdVals[j] == null`
        for (int j = 0; j < p; ++j) {
            std::size_t sj = static_cast<std::size_t>(j);
            double h = step[sj];
            while (h <= nd::NumericalDiff::kMaxStep) {
                perturbed_params[sj] = map_values[sj] + h;
                fwd_vals[sj] = model.pointwise_data_log_likelihood(perturbed_params);
                perturbed_params[sj] = map_values[sj] - h;
                bwd_vals[sj] = model.pointwise_data_log_likelihood(perturbed_params);
                perturbed_params[sj] = map_values[sj];
                have[sj] = true;

                double total_diff = 0;
                double total_scale = 0;
                for (int i = 0; i < n; ++i) {
                    std::size_t si = static_cast<std::size_t>(i);
                    total_diff += std::fabs(fwd_vals[sj][si] - bwd_vals[sj][si]);
                    total_scale += std::max(std::fabs(fwd_vals[sj][si]), std::fabs(bwd_vals[sj][si]));
                }
                if (total_diff >= nd::NumericalDiff::kFlatSpotRelTol * std::max(total_scale, 1.0))
                    break;

                h *= nd::NumericalDiff::kStepGrowthFactor;
            }
            step[sj] = std::min(h, nd::NumericalDiff::kMaxStep);

            // If the initial step > MaxStep the loop never ran -- compute with the capped step.
            if (!have[sj]) {
                perturbed_params[sj] = map_values[sj] + step[sj];
                fwd_vals[sj] = model.pointwise_data_log_likelihood(perturbed_params);
                perturbed_params[sj] = map_values[sj] - step[sj];
                bwd_vals[sj] = model.pointwise_data_log_likelihood(perturbed_params);
                perturbed_params[sj] = map_values[sj];
            }
        }

        // Data components for observation metadata (optional -- not all models provide it).
        std::vector<corehydro::models::DataComponent> data_components;
        bool have_data_components = false;
        try {
            data_components = model.pointwise_data_log_likelihood_components(map_values);
            have_data_components = true;
        } catch (const std::exception&) {
            // C# catches NotImplementedException (optional metadata) and logs other failures;
            // silent no-throw guard here, leaving data_components absent.
        }

        // Fit Influence = g_i' J_post^{-1} g_i / p (Cook's Distance, score-based).
        // Variance Influence = |tr(J_post^{-1} . J_i)| / p (trace method).
        observations_.clear();
        observations_.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            std::size_t si = static_cast<std::size_t>(i);

            // Build J_i (per-obs Fisher info) from cached perturbation data -- diagonal only.
            Matrix ji(p, p);
            for (int j = 0; j < p; ++j) {
                std::size_t sj = static_cast<std::size_t>(j);
                ji(j, j) = -(fwd_vals[sj][si] - 2.0 * base_pointwise_ll[si] + bwd_vals[sj][si]) /
                           (step[sj] * step[sj]);
            }

            double variance_influence = std::fabs(trace_product(hessian_inv, ji, p)) / p;

            std::vector<double> score_vector(static_cast<std::size_t>(p));
            for (int j = 0; j < p; ++j) {
                std::size_t sj = static_cast<std::size_t>(j);
                score_vector[sj] = (fwd_vals[sj][si] - bwd_vals[sj][si]) / (2.0 * step[sj]);
            }

            double fit_influence =
                p > 0 ? compute_quadratic_form(score_vector, hessian_inv, p) / p : 0;
            double leverage = fit_influence + variance_influence;

            double value = 0;
            auto data_type = corehydro::models::DataComponentType::Exact;
            int count = 1;
            std::optional<std::string> name = std::nullopt;
            if (have_data_components && i < static_cast<int>(data_components.size())) {
                value = data_components[si].value();
                data_type = data_components[si].type();
                count = data_components[si].count();
                name = data_components[si].name();
            }

            observations_.emplace_back(i, leverage, 0.0, fit_influence, variance_influence, 0.0,
                                       0.0, value, data_type, count, name);
        }
    }

    // Computes per-prior-component score vectors and their leverages (C# 498). Uses the
    // generalized-variance determinant method for variance influence (priors can contribute a
    // large fraction of total information, making the linear trace approximation inaccurate).
    void compute_prior_component_leverages(corehydro::models::ModelBase& model,
                                           const std::vector<double>& map_values, int p,
                                           const Matrix& hessian_inv) {
        std::vector<corehydro::models::PriorComponent> base_components;
        try {
            base_components = model.pointwise_prior_log_likelihood(map_values);
        } catch (const std::exception&) {
            // C# Debug.WriteLine here; silent guard.
            prior_components_.clear();
            return;
        }

        if (base_components.empty()) {
            prior_components_.clear();
            return;
        }

        int num_components = static_cast<int>(base_components.size());
        prior_components_.clear();
        prior_components_.reserve(static_cast<std::size_t>(num_components));
        for (int k = 0; k < num_components; ++k) {
            std::size_t sk = static_cast<std::size_t>(k);
            int captured_k = k;
            try {
                // Scalar function for the score vector: PointwisePriorLogLikelihood(theta)[k].
                auto prior_func = [&model, captured_k](std::vector<double>& theta) -> double {
                    auto comps = model.pointwise_prior_log_likelihood(theta);
                    return captured_k < static_cast<int>(comps.size())
                               ? comps[static_cast<std::size_t>(captured_k)].log_likelihood()
                               : 0.0;
                };

                // Log-likelihood with this prior removed (data + all other priors).
                auto ll_without_prior = [&model, captured_k](std::vector<double>& theta) -> double {
                    double data_ll = model.data_log_likelihood(theta);
                    auto prior_comps = model.pointwise_prior_log_likelihood(theta);
                    double prior_ll = 0;
                    for (int idx = 0; idx < static_cast<int>(prior_comps.size()); ++idx)
                        if (idx != captured_k)
                            prior_ll += prior_comps[static_cast<std::size_t>(idx)].log_likelihood();
                    return data_ll + prior_ll;
                };
                double variance_influence = compute_generalized_variance_influence(
                    ll_without_prior, hessian_inv, map_values, p);

                // Fit Influence (Cook's D): score vector via central differences.
                std::vector<double> score_vector(static_cast<std::size_t>(p));
                std::vector<double> perturbed_params = map_values;
                for (int j = 0; j < p; ++j) {
                    std::size_t sj = static_cast<std::size_t>(j);
                    double h = std::max(std::fabs(map_values[sj]) * 1e-4, 1e-3);
                    perturbed_params[sj] = map_values[sj] + h;
                    double fwd = prior_func(perturbed_params);
                    perturbed_params[sj] = map_values[sj] - h;
                    double bwd = prior_func(perturbed_params);
                    score_vector[sj] = (fwd - bwd) / (2.0 * h);
                    perturbed_params[sj] = map_values[sj];
                }

                double fit_influence =
                    p > 0 ? compute_quadratic_form(score_vector, hessian_inv, p) / p : 0;
                double leverage = fit_influence + variance_influence;

                prior_components_.emplace_back(base_components[sk].name(),
                                               base_components[sk].type(), leverage, 0.0,
                                               fit_influence, variance_influence, 0.0, 0.0);
            } catch (const std::exception&) {
                // C# Debug.WriteLine here; zero component on failure.
                prior_components_.emplace_back(base_components[sk].name(),
                                               base_components[sk].type(), 0.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0);
            }
        }
    }

    // Computes tr(A . B) without forming the full matrix product (C# 613).
    static double trace_product(const Matrix& a, const Matrix& b, int p) {
        double trace = 0;
        for (int i = 0; i < p; ++i)
            for (int j = 0; j < p; ++j) trace += a(i, j) * b(j, i);
        return trace;
    }

    // Computes the variance influence of a prior component using the generalized variance
    // (determinant) method (C# 638): |log(|det(Sigma_{-k})|) - log(|det(Sigma)|)| / p.
    static double compute_generalized_variance_influence(
        const std::function<double(std::vector<double>&)>& log_likelihood_without,
        const Matrix& sigma, const std::vector<double>& map_values, int p) {
        try {
            double det_sigma = sigma.determinant();
            double abs_det_sigma = std::fabs(det_sigma);

            // Hessian of the reduced log-likelihood, negate, invert -> Sigma_{-k}.
            Matrix hessian_reduced = corehydro::estimation::NumericalDiff::compute_hessian(
                log_likelihood_without, map_values, p);
            Matrix neg_hessian_reduced = hessian_reduced * -1.0;
            Matrix sigma_reduced(p, p);
            try {
                sigma_reduced = neg_hessian_reduced.inverse();
            } catch (const std::exception&) {
                neg_hessian_reduced = corehydro::numerics::math::linalg::MatrixRegularization::
                    make_symmetric_positive_definite(neg_hessian_reduced);
                sigma_reduced = neg_hessian_reduced.inverse();
            }

            double det_sigma_reduced = sigma_reduced.determinant();
            double abs_det_reduced = std::fabs(det_sigma_reduced);

            // Guard against log(0): 1e-300 threshold is above double min but well below any
            // physically meaningful determinant value.
            double result = 0;
            if (abs_det_sigma > 1e-300 && abs_det_reduced > 1e-300)
                result = std::fabs(std::log(abs_det_reduced) - std::log(abs_det_sigma)) / p;

            return result;
        } catch (const std::exception&) {
            // C# Debug.WriteLine here; silent guard.
            return 0;
        }
    }

    // Computes the quadratic form g' A g, clamped to non-negative (C# 689).
    static double compute_quadratic_form(const std::vector<double>& g, const Matrix& a, int p) {
        double result = 0;
        for (int j = 0; j < p; ++j) {
            std::size_t sj = static_cast<std::size_t>(j);
            double tmp = 0;
            for (int k = 0; k < p; ++k) tmp += a(j, k) * g[static_cast<std::size_t>(k)];
            result += g[sj] * tmp;
        }
        // Clamp to non-negative: rounding can produce small negative values for PD A.
        return std::max(0.0, result);
    }

    // Computes summary statistics from observation and prior component leverages (C# 707).
    void compute_summary_statistics() {
        total_observation_leverage_ = 0;
        for (const auto& o : observations_) total_observation_leverage_ += o.leverage();
        total_prior_leverage_ = 0;
        for (const auto& pc : prior_components_) total_prior_leverage_ += pc.leverage();
        total_leverage_ = total_observation_leverage_ + total_prior_leverage_;

        observation_fit_influence_ = 0;
        observation_variance_influence_ = 0;
        for (const auto& o : observations_) {
            observation_fit_influence_ += o.fit_influence();
            observation_variance_influence_ += o.variance_influence();
        }
        prior_fit_influence_ = 0;
        prior_variance_influence_ = 0;
        for (const auto& pc : prior_components_) {
            prior_fit_influence_ += pc.fit_influence();
            prior_variance_influence_ += pc.variance_influence();
        }
        total_fit_influence_ = observation_fit_influence_ + prior_fit_influence_;
        total_variance_influence_ = observation_variance_influence_ + prior_variance_influence_;

        // C# Debug.WriteLine warning when the leverage sum deviates significantly from p;
        // silent no-op here (no compute-path side effect).
    }

    // Updates percentage values on observations and prior components after totals are known
    // (C# 730).
    void update_percentages() {
        if (total_leverage_ <= 0) return;

        double total_fv = total_fit_influence_ + total_variance_influence_;
        if (total_fv <= 0) total_fv = total_leverage_;  // fallback for pre-computed leverages

        for (auto& obs : observations_) {
            obs = ObservationLeverage(
                obs.index(), obs.leverage(), obs.leverage() / total_leverage_ * 100.0,
                obs.fit_influence(), obs.variance_influence(),
                total_fv > 0 ? obs.fit_influence() / total_fv * 100.0 : 0,
                total_fv > 0 ? obs.variance_influence() / total_fv * 100.0 : 0, obs.value(),
                obs.data_type(), obs.count(), obs.name());
        }

        for (auto& pc : prior_components_) {
            pc = PriorComponentLeverage(
                pc.name(), pc.type(), pc.leverage(), pc.leverage() / total_leverage_ * 100.0,
                pc.fit_influence(), pc.variance_influence(),
                total_fv > 0 ? pc.fit_influence() / total_fv * 100.0 : 0,
                total_fv > 0 ? pc.variance_influence() / total_fv * 100.0 : 0);
        }
    }

    // ------------------------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------------------------
    std::vector<ObservationLeverage> observations_;
    std::vector<PriorComponentLeverage> prior_components_;
    int number_of_parameters_ = 0;
    double total_observation_leverage_ = 0;
    double total_prior_leverage_ = 0;
    double total_leverage_ = 0;
    double observation_fit_influence_ = 0;
    double observation_variance_influence_ = 0;
    double prior_fit_influence_ = 0;
    double prior_variance_influence_ = 0;
    double total_fit_influence_ = 0;
    double total_variance_influence_ = 0;
};

}  // namespace corehydro::diagnostics
