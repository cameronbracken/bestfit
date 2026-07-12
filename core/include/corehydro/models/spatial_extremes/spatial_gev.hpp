// ported from: upstream/RMC-BestFit/src/RMC.BestFit/Models/SpatialExtremes/SpatialGEV.cs @ fc28c0c
//
// Spatial Generalized Extreme Value (GEV) model following Renard's Bayesian Hierarchical Model
// framework. Four levels assembled from the earlier Phase-7a leaves:
//   Level 1 (Data):    Y_ij | theta_j ~ GEV(xi_j, alpha_j, kappa_j) via the ported
//                      GeneralizedExtremeValue leaf.
//   Level 2 (Process): per-parameter spatial trends (GeneralLinearFunction, log-link default for
//                      location + scale), latent GP spatial errors (S3 SpatialRegressionErrors),
//                      and optional Gaussian-copula spatial dependence (S2 GaussianCopula).
//   Level 3 (Priors):  Uniform hyperparameter priors seeded from at-site data moments.
//
// Namespace (S3 precedent, C# governs): the C# namespace is RMC.BestFit.Models.SpatialExtremes,
// and every header this file #includes (correlation models, GaussianCopula,
// SpatialRegressionErrors, GeneralLinearFunction lives in models::trend_functions) places the
// spatial-extremes types in corehydro::models::spatial_extremes. SpatialGEV joins them there,
// correcting the plan/requirements shorthand corehydro::models::SpatialGEV.
//
// -------------------------------------------------------------------------------------------
// PRESERVED NON-CANONICAL SPATIAL-ERROR DECOMPOSITION (C# remarks SpatialGEV.cs:833-866 and
// 1122-1134, transcribed verbatim below). Do NOT "fix" it:
//
//   This model intentionally INCLUDES the Gaussian-process spatial-error log densities
//   (location / scale / shape errors) in the returned DATA likelihood -- historically they
//   have been treated as "data" so that the marginal site likelihood plus spatial dependence
//   is a single integrable quantity. As a consequence:
//     * pointwise_data_log_likelihood_components does NOT add the spatial-error contributions
//       (it is per-site, not per-process), so its Sum() does NOT match data_log_likelihood.
//     * pointwise_prior_log_likelihood DOES emit the spatial-error contributions as
//       PriorComponentType::SpatialError, so its Sum() does NOT match ModelBase's
//       prior_log_likelihood (which only sums parameter priors).
//     * WAIC and LOO-CV (computed from pointwise_data_log_likelihood_components) therefore
//       EXCLUDE the spatial-error term -- they reflect the marginal site-by-site predictive
//       performance only, not the joint spatial process.
//     * A consumer that adds data_log_likelihood + pointwise_prior_log_likelihood.Sum() would
//       double-count the spatial-error term. Always compute the joint via the inherited
//       ModelBase::log_likelihood (= data_log_likelihood + prior_log_likelihood) instead.
//   This asymmetry is a known design tradeoff, deliberately violating the canonical
//   pointwise-vs-scalar identity to surface the per-error-process contribution to the prior
//   diagnostics panel.
// -------------------------------------------------------------------------------------------
//
// SIMPLIFICATIONS vs. the C# (documented; not behavior changes):
//   - Thread-local clones: the C# clones trend/copula/error models per LL evaluation because it
//     runs parallel MCMC chains. The C++ core is single-threaded by design, so the "thread-safe"
//     / parallel remarks port as a plain per-call clone. compute_log_likelihood_internal keeps
//     the clone-into-locals structure (it is what makes the const evaluation side-effect free),
//     just without the concurrency motivation.
//   - pointwise_prior_log_likelihood: the C# mutates `this` (SetParameterValues -> evaluate ->
//     restore, guarded by Debug.WriteLine best-effort restores). This port computes the same
//     quantity WITHOUT mutating `this`: parameter priors come from the flat parameters_ list
//     (their priors are value-copied at set_default_parameters time) evaluated at the supplied
//     values, and the spatial-error contributions come from CLONED error models set to the
//     supplied parameter slots. The swallowed-exception restore guards therefore have no port.
//   - generate_random_values: the C# re-applies SetParameterValues(current values) before
//     sampling; that call only rebuilds internal MVN state the sampler never reads (the draw
//     path reads trend Predict + error GetError, i.e. parameter values only), so it is a no-op
//     for the sampled quantities and is omitted to keep the method const.
//
// Marshaling order (oracle-visible, C# SetParameterValues SpatialGEV.cs:507-569): the flat
// parameter vector is consumed as
//   copula -> location -> scale -> shape -> location-errors -> scale-errors -> shape-errors,
// each block present only when its gate (Use*  &&  non-null child) is on. parameters_ is
// assembled in exactly this order in set_default_parameters, and set_parameter_values
// distributes the incoming vector to the child models in the same order.
//
// Aliasing note: in C# the flat Parameters list holds *references* to the child models'
// ModelParameter objects. std::vector<ModelParameter> holds values, so this port stores VALUE
// COPIES in parameters_ and keeps them in sync -- set_parameter_values writes each value into
// BOTH the child model (for Predict / GetError / LogPDF) and parameters_[i] (for the estimators'
// bounds/prior reads). This is exactly the pattern GaussianCopula (S2) and SpatialRegressionErrors
// (S3) already use for their delegated parameters.
//
// Deliberately NOT ported (project-wide conventions): the XElement ctor / ToXElement()
// (XML serialization), INotifyPropertyChanged / PropertyChanged plumbing, and the
// SpatialGEVAnalysis wrapper class (separate file, not in S4 scope). The C# ArgumentNullException
// null-guards on data/coordinates/trends are vacuous here (by-value / const-ref parameters cannot
// be null); the dimension-mismatch guard IS ported. TryParseBool string-config parsing has no
// port (it only feeds the XElement ctor).
#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/models/spatial_extremes/copula_models/gaussian_copula.hpp"
#include "corehydro/models/spatial_extremes/copula_models/spatial_regression_errors.hpp"
#include "corehydro/models/spatial_extremes/spatial_correlation/correlation_function_type.hpp"
#include "corehydro/models/support/data_component.hpp"
#include "corehydro/models/support/model_base.hpp"
#include "corehydro/models/support/prior_component.hpp"
#include "corehydro/models/support/simulatable.hpp"
#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/models/trend_functions/general_linear_function.hpp"
#include "corehydro/numerics/data/correlation.hpp"
#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/generalized_extreme_value.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models::spatial_extremes {

class SpatialGEV : public ModelBase, public ISimulatable<std::vector<double>> {
   public:
    using Grid = std::vector<std::vector<double>>;

    // Constructs a new spatial GEV model. `at_site_data` is [observations][sites] (row-major),
    // `coordinates` is [sites][2] as (X,Y) or (Lat,Lon). The C# null guards on data / coordinates
    // / trend models are vacuous (by-value parameters); the sites-vs-coordinates dimension guard
    // is ported (C# ArgumentException -> std::invalid_argument).
    SpatialGEV(Grid at_site_data, Grid coordinates,
               trend_functions::GeneralLinearFunction location,
               trend_functions::GeneralLinearFunction scale,
               trend_functions::GeneralLinearFunction shape)
        : at_site_data_(std::move(at_site_data)),
          coordinates_(std::move(coordinates)),
          location_(std::move(location)),
          scale_(std::move(scale)),
          shape_(std::move(shape)) {
        int data_sites = at_site_data_.empty() ? 0 : static_cast<int>(at_site_data_[0].size());
        if (data_sites != static_cast<int>(coordinates_.size())) {
            throw std::invalid_argument(
                "Number of sites in data must match number of coordinates.");
        }

        // Options default (C# ctor 90-95).
        use_copula_dependence_ = false;
        use_location_errors_ = false;
        use_scale_errors_ = false;
        use_shape_errors_ = false;
        use_log_link_for_location_ = true;  // Default to log-link for location.
        use_log_link_for_scale_ = true;     // Default to log-link for scale.

        // Initialize weights to 1.0 (equal weighting).
        site_weights_.assign(static_cast<std::size_t>(sites()), 1.0);

        set_default_parameters();
    }

    // Move-only: the copula / error children are held via unique_ptr (move-only S2/S3 types).
    // Deep copies go through clone().
    SpatialGEV(SpatialGEV&&) = default;
    SpatialGEV& operator=(SpatialGEV&&) = default;
    SpatialGEV(const SpatialGEV&) = delete;
    SpatialGEV& operator=(const SpatialGEV&) = delete;
    ~SpatialGEV() override = default;

    // --- Members (C# properties). ---
    const Grid& at_site_data() const { return at_site_data_; }
    const Grid& coordinates() const { return coordinates_; }

    // Number of sites (columns in the data matrix).
    int sites() const {
        return at_site_data_.empty() ? 0 : static_cast<int>(at_site_data_[0].size());
    }
    // Number of observations (rows in the data matrix).
    int observations() const { return static_cast<int>(at_site_data_.size()); }

    trend_functions::GeneralLinearFunction& location() { return location_; }
    const trend_functions::GeneralLinearFunction& location() const { return location_; }
    trend_functions::GeneralLinearFunction& scale() { return scale_; }
    const trend_functions::GeneralLinearFunction& scale() const { return scale_; }
    trend_functions::GeneralLinearFunction& shape() { return shape_; }
    const trend_functions::GeneralLinearFunction& shape() const { return shape_; }

    // Nullable copula / error children (nullptr == C# null).
    GaussianCopula* spatial_dependence() { return spatial_dependence_.get(); }
    const GaussianCopula* spatial_dependence() const { return spatial_dependence_.get(); }
    void set_spatial_dependence(GaussianCopula copula) {
        spatial_dependence_ = std::make_unique<GaussianCopula>(std::move(copula));
    }

    SpatialRegressionErrors* location_errors() { return location_errors_.get(); }
    const SpatialRegressionErrors* location_errors() const { return location_errors_.get(); }
    void set_location_errors(SpatialRegressionErrors errors) {
        location_errors_ = std::make_unique<SpatialRegressionErrors>(std::move(errors));
    }
    SpatialRegressionErrors* scale_errors() { return scale_errors_.get(); }
    const SpatialRegressionErrors* scale_errors() const { return scale_errors_.get(); }
    void set_scale_errors(SpatialRegressionErrors errors) {
        scale_errors_ = std::make_unique<SpatialRegressionErrors>(std::move(errors));
    }
    SpatialRegressionErrors* shape_errors() { return shape_errors_.get(); }
    const SpatialRegressionErrors* shape_errors() const { return shape_errors_.get(); }
    void set_shape_errors(SpatialRegressionErrors errors) {
        shape_errors_ = std::make_unique<SpatialRegressionErrors>(std::move(errors));
    }

    // Option flags.
    bool use_copula_dependence() const { return use_copula_dependence_; }
    void set_use_copula_dependence(bool v) { use_copula_dependence_ = v; }
    bool use_location_errors() const { return use_location_errors_; }
    void set_use_location_errors(bool v) { use_location_errors_ = v; }
    bool use_scale_errors() const { return use_scale_errors_; }
    void set_use_scale_errors(bool v) { use_scale_errors_ = v; }
    bool use_shape_errors() const { return use_shape_errors_; }
    void set_use_shape_errors(bool v) { use_shape_errors_ = v; }
    bool use_log_link_for_location() const { return use_log_link_for_location_; }
    void set_use_log_link_for_location(bool v) { use_log_link_for_location_ = v; }
    bool use_log_link_for_scale() const { return use_log_link_for_scale_; }
    void set_use_log_link_for_scale(bool v) { use_log_link_for_scale_ = v; }

    // Site-specific weights for weighted likelihood (mutable + const; tests write in place).
    std::vector<double>& site_weights() { return site_weights_; }
    const std::vector<double>& site_weights() const { return site_weights_; }
    void set_site_weights(std::vector<double> w) { site_weights_ = std::move(w); }

    // --- SetDefaultParameters (C# 318-504). Computes method-of-moments hyperparameter seeds
    // from at-site data (mean -> location, std -> scale, 0 -> shape) and assembles the flat
    // parameter list in marshaling order. ---
    void set_default_parameters() override {
        // Method-of-moments style estimates from at-site data (per-site mean / std, count
        // weighted). Cheaper than per-site MLE, adequate to seed the sampler (C# remark).
        double nt = 0;
        double sum_loc = 0, sum_scl = 0, sum_shp = 0;
        std::vector<double> loc_list, scl_list, shp_list;

        for (int j = 0; j < sites(); ++j) {
            std::vector<double> site_data;
            for (int i = 0; i < observations(); ++i) {
                double v = at_site_data_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                if (!std::isnan(v)) site_data.push_back(v);
            }
            if (!site_data.empty()) {
                nt += static_cast<double>(site_data.size());
                double mean = 0.0;
                for (double v : site_data) mean += v;
                mean /= static_cast<double>(site_data.size());
                double sum_sq = 0.0;
                for (double v : site_data) sum_sq += (v - mean) * (v - mean);
                double std_dev =
                    site_data.size() > 1
                        ? std::sqrt(sum_sq / static_cast<double>(site_data.size() - 1))
                        : std::max(std::fabs(mean), 1e-3);
                double shape_est = 0.0;  // Weak default near 0; MCMC explores [-0.5, 0.5].

                sum_loc += mean * static_cast<double>(site_data.size());
                sum_scl += std_dev * static_cast<double>(site_data.size());
                sum_shp += shape_est * static_cast<double>(site_data.size());

                loc_list.push_back(mean);
                scl_list.push_back(std_dev);
                shp_list.push_back(shape_est);
            }
        }

        double avg_loc = nt > 0 ? sum_loc / nt : 0.0;
        double avg_scl = nt > 0 ? std::max(sum_scl / nt, 1e-3) : 1.0;
        double avg_shp = nt > 0 ? sum_shp / nt : 0.0;

        std::vector<ModelParameter>& location_params = location_.parameters();
        std::vector<ModelParameter>& scale_params = scale_.parameters();
        std::vector<ModelParameter>& shape_params = shape_.parameters();

        // Location intercept.
        if (use_log_link_for_location_) {
            location_params[0].set_value(std::log(std::max(avg_loc, 0.01)));
            location_params[0].set_lower_bound(std::log(0.01));
            location_params[0].set_upper_bound(std::ceil(std::log(std::fabs(avg_loc)) + 3.0));
        } else {
            location_params[0].set_value(avg_loc);
            double range = std::pow(10, std::ceil(std::log10(std::fabs(avg_loc)) + 1));
            location_params[0].set_lower_bound(-range);
            location_params[0].set_upper_bound(range);
        }
        location_params[0].set_prior_distribution(
            std::make_unique<numerics::distributions::Uniform>(
                location_params[0].lower_bound(), location_params[0].upper_bound()));

        // Scale intercept.
        if (use_log_link_for_scale_) {
            scale_params[0].set_value(std::log(std::max(avg_scl, 0.01)));
            scale_params[0].set_lower_bound(std::log(0.01));
            scale_params[0].set_upper_bound(std::ceil(std::log(std::fabs(avg_scl)) + 3.0));
        } else {
            scale_params[0].set_value(avg_scl);
            scale_params[0].set_lower_bound(numerics::kDoubleMachineEpsilon);
            scale_params[0].set_upper_bound(
                std::pow(10, std::ceil(std::log10(std::fabs(avg_scl)) + 1)));
        }
        scale_params[0].set_prior_distribution(std::make_unique<numerics::distributions::Uniform>(
            scale_params[0].lower_bound(), scale_params[0].upper_bound()));

        // Shape intercept.
        shape_params[0].set_value(avg_shp);
        shape_params[0].set_lower_bound(-0.5);
        shape_params[0].set_upper_bound(0.5);
        shape_params[0].set_prior_distribution(
            std::make_unique<numerics::distributions::Uniform>(-0.5, 0.5));

        // Assemble the flat parameter list in marshaling order (value copies; see header note).
        parameters_.clear();

        if (use_copula_dependence_ && spatial_dependence_ != nullptr)
            append_params(spatial_dependence_->parameters());

        append_params(location_params);
        append_params(scale_params);
        append_params(shape_params);

        if (use_location_errors_ && location_errors_ != nullptr) {
            double max_loc_error = std::ceil((numerics::data::maximum(loc_list) - avg_loc) * 3);
            location_errors_->set_default_parameters(std::max(max_loc_error, 1.0));
            append_params(location_errors_->parameters());
        }
        if (use_scale_errors_ && scale_errors_ != nullptr) {
            double max_scl_error = std::ceil((numerics::data::maximum(scl_list) - avg_scl) * 3);
            scale_errors_->set_default_parameters(std::max(max_scl_error, 1.0));
            append_params(scale_errors_->parameters());
        }
        if (use_shape_errors_ && shape_errors_ != nullptr) {
            double max_shp_error = std::ceil((numerics::data::maximum(shp_list) - avg_shp) * 3);
            shape_errors_->set_default_parameters(std::max(max_shp_error, 0.5));
            append_params(shape_errors_->parameters());
        }
    }

    // --- SetParameterValues (C# 507-569). Distributes the incoming vector to the child models
    // in marshaling order AND syncs the flat parameters_ values (see header aliasing note). ---
    void set_parameter_values(const std::vector<double>& parameters) override {
        if (parameters.size() != static_cast<std::size_t>(number_of_parameters())) {
            throw std::invalid_argument(
                "Expected " + std::to_string(number_of_parameters()) + " parameters but got " +
                std::to_string(parameters.size()) + ".");
        }

        int index = 0;

        if (use_copula_dependence_ && spatial_dependence_ != nullptr) {
            std::vector<double> cop_params;
            for (int i = 0; i < spatial_dependence_->number_of_parameters(); ++i)
                cop_params.push_back(parameters[static_cast<std::size_t>(index++)]);
            spatial_dependence_->set_parameter_values(cop_params);
        }

        index = set_child_values(location_, parameters, index);
        index = set_child_values(scale_, parameters, index);
        index = set_child_values(shape_, parameters, index);

        if (use_location_errors_ && location_errors_ != nullptr)
            index = set_error_values(*location_errors_, parameters, index);
        if (use_scale_errors_ && scale_errors_ != nullptr)
            index = set_error_values(*scale_errors_, parameters, index);
        if (use_shape_errors_ && shape_errors_ != nullptr)
            index = set_error_values(*shape_errors_, parameters, index);

        // Sync the flat list values (priors/bounds already match from set_default_parameters).
        for (std::size_t i = 0; i < parameters_.size(); ++i) parameters_[i].set_value(parameters[i]);
    }

    // --- GetGEVParameters (C# 576-597): assemble (xi, alpha, kappa) for a site from the
    // current member state. ---
    std::vector<double> get_gev_parameters(int site_index) const {
        if (site_index < 0 || site_index >= sites())
            throw std::out_of_range("site index out of range");
        return gev_parameters_local(site_index, location_, scale_, shape_, location_errors_.get(),
                                    scale_errors_.get(), shape_errors_.get());
    }

    // --- DataLogLikelihood (C# 868-876): data likelihood only (base combines with priors).
    // INCLUDES the spatial-error log densities -- see the preserved-decomposition note. ---
    double data_log_likelihood(std::vector<double>& parameters) const override {
        if (parameters.size() != static_cast<std::size_t>(number_of_parameters()))
            return -std::numeric_limits<double>::infinity();
        return compute_log_likelihood_internal(parameters);
    }

    // --- PointwiseDataLogLikelihood (C# 883-1090): one value per observation (time step across
    // all sites). Does NOT include the spatial-error term (per-observation, not per-process). ---
    std::vector<double> pointwise_data_log_likelihood(
        const std::vector<double>& parameters) const override {
        std::vector<double> result(static_cast<std::size_t>(observations()),
                                   -std::numeric_limits<double>::infinity());
        if (parameters.size() != static_cast<std::size_t>(number_of_parameters())) return result;

        std::vector<double> weights = site_weights_snapshot();
        auto loc = location_;
        auto scl = scale_;
        auto shp = shape_;
        std::optional<GaussianCopula> cop;
        std::optional<SpatialRegressionErrors> le, se, she;
        distribute_locals(parameters, loc, scl, shp, cop, le, se, she);
        SpatialRegressionErrors* lep = le ? &*le : nullptr;
        SpatialRegressionErrors* sep = se ? &*se : nullptr;
        SpatialRegressionErrors* shp_ep = she ? &*she : nullptr;

        numerics::distributions::GeneralizedExtremeValue local_gev;

        // Cache GEV parameters per site.
        std::vector<std::vector<double>> gev_cache(static_cast<std::size_t>(sites()));
        for (int j = 0; j < sites(); ++j)
            gev_cache[static_cast<std::size_t>(j)] =
                gev_parameters_local(j, loc, scl, shp, lep, sep, shp_ep);

        const bool use_copula = use_copula_dependence_ && cop.has_value();
        for (int i = 0; i < observations(); ++i) {
            double obs_log_lh = 0.0;
            std::vector<double> z(static_cast<std::size_t>(sites()), 0.0);
            bool has_data = false;
            bool valid = true;

            for (int j = 0; j < sites(); ++j) {
                double x =
                    at_site_data_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                if (std::isnan(x)) continue;
                has_data = true;
                const std::vector<double>& gp = gev_cache[static_cast<std::size_t>(j)];
                if (gp[1] <= 0) {
                    valid = false;
                    break;
                }
                local_gev.set_parameters(gp);
                double marg = local_gev.log_pdf(x);
                if (!numerics::is_finite(marg)) {
                    valid = false;
                    break;
                }
                obs_log_lh += weights[static_cast<std::size_t>(j)] * marg;
                if (use_copula) {
                    double u = local_gev.cdf(x);
                    z[static_cast<std::size_t>(j)] =
                        numerics::distributions::Normal::standard_z(u);
                }
            }

            if (!valid) {
                result[static_cast<std::size_t>(i)] = -std::numeric_limits<double>::infinity();
                continue;
            }
            if (use_copula && has_data) {
                double cop_log_lh = cop->log_pdf(z);
                if (!numerics::is_finite(cop_log_lh)) {
                    result[static_cast<std::size_t>(i)] =
                        -std::numeric_limits<double>::infinity();
                    continue;
                }
                obs_log_lh += cop_log_lh;
            }
            result[static_cast<std::size_t>(i)] = obs_log_lh;
        }
        return result;
    }

    // --- PointwiseDataLogLikelihoodComponents (C# 1093-1120). ---
    std::vector<DataComponent> pointwise_data_log_likelihood_components(
        const std::vector<double>& parameters) const override {
        std::vector<double> log_liks = pointwise_data_log_likelihood(parameters);
        std::vector<DataComponent> result;
        result.reserve(log_liks.size());
        for (std::size_t i = 0; i < log_liks.size(); ++i) {
            double value = 0.0;
            for (int j = 0; j < sites(); ++j) {
                double x = at_site_data_[i][static_cast<std::size_t>(j)];
                if (!std::isnan(x)) {
                    value = x;
                    break;
                }
            }
            result.emplace_back(static_cast<int>(i), log_liks[i], value,
                                DataComponentType::Exact, sites(),
                                "Obs " + std::to_string(i + 1));
        }
        return result;
    }

    // --- PointwisePriorLogLikelihood (C# 1135-1210). Emits per-parameter priors AND the
    // SpatialError components (the deliberately-broken decomposition). Computed WITHOUT mutating
    // `this` -- see the header simplification note. ---
    std::vector<PriorComponent> pointwise_prior_log_likelihood(
        const std::vector<double>& parameters) const override {
        std::vector<PriorComponent> result;
        if (parameters.size() != static_cast<std::size_t>(number_of_parameters())) return result;

        // Parameter priors from the flat list (priors value-copied at set_default_parameters).
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            double ll = parameters_[i].prior_distribution().log_pdf(parameters[i]);
            const std::string& param_name = parameters_[i].owner_name().empty()
                                                ? parameters_[i].name()
                                                : parameters_[i].owner_name();
            result.emplace_back("Parameter Prior: " + param_name, ll,
                                PriorComponentType::ParameterPrior);
        }

        // Spatial error priors (Gaussian process priors), from cloned error models set to the
        // supplied parameter slots.
        auto loc = location_;
        auto scl = scale_;
        auto shp = shape_;
        std::optional<GaussianCopula> cop;
        std::optional<SpatialRegressionErrors> le, se, she;
        distribute_locals(parameters, loc, scl, shp, cop, le, se, she);

        if (le) {
            double v = le->log_pdf();
            if (numerics::is_finite(v))
                result.emplace_back("Spatial Error: Location", v,
                                    PriorComponentType::SpatialError);
        }
        if (se) {
            double v = se->log_pdf();
            if (numerics::is_finite(v))
                result.emplace_back("Spatial Error: Scale", v, PriorComponentType::SpatialError);
        }
        if (she) {
            double v = she->log_pdf();
            if (numerics::is_finite(v))
                result.emplace_back("Spatial Error: Shape", v, PriorComponentType::SpatialError);
        }
        return result;
    }

    // --- PDF / CDF / InverseCDF at a site (C# 1217-1246). ---
    double pdf(double x, int site_index) const {
        numerics::distributions::GeneralizedExtremeValue gev;
        gev.set_parameters(get_gev_parameters(site_index));
        return gev.pdf(x);
    }
    double cdf(double x, int site_index) const {
        numerics::distributions::GeneralizedExtremeValue gev;
        gev.set_parameters(get_gev_parameters(site_index));
        return gev.cdf(x);
    }
    double inverse_cdf(double probability, int site_index) const {
        numerics::distributions::GeneralizedExtremeValue gev;
        gev.set_parameters(get_gev_parameters(site_index));
        return gev.inverse_cdf(probability);
    }

    // --- Validate (C# 1311-1371). ---
    ValidationResult validate() const override {
        ValidationResult result;
        if (sites() < 2) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: At least 2 sites are required for spatial modeling.");
        }
        if (use_copula_dependence_ && spatial_dependence_ == nullptr) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Copula dependence enabled but SpatialDependence is null.");
        }
        if (use_location_errors_ && location_errors_ == nullptr) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Location errors enabled but LocationErrors is null.");
        }
        if (use_scale_errors_ && scale_errors_ == nullptr) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Scale errors enabled but ScaleErrors is null.");
        }
        if (use_shape_errors_ && shape_errors_ == nullptr) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Shape errors enabled but ShapeErrors is null.");
        }
        if (site_weights_.size() != static_cast<std::size_t>(sites())) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Site weights must be specified for all sites.");
        }
        return result;
    }

    // --- ComputeIntersiteCorrelation (C# 1454-1494): Pearson correlation from pairwise-complete
    // observations. ---
    Grid compute_intersite_correlation() const {
        int n = sites();
        Grid corr(static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(n), 0.0));
        for (int i = 0; i < n; ++i) {
            corr[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = 1.0;
            for (int j = i + 1; j < n; ++j) {
                std::vector<double> pairs_i, pairs_j;
                for (int t = 0; t < observations(); ++t) {
                    double xi = at_site_data_[static_cast<std::size_t>(t)][static_cast<std::size_t>(i)];
                    double xj = at_site_data_[static_cast<std::size_t>(t)][static_cast<std::size_t>(j)];
                    if (!std::isnan(xi) && !std::isnan(xj)) {
                        pairs_i.push_back(xi);
                        pairs_j.push_back(xj);
                    }
                }
                double c = 0.0;
                if (pairs_i.size() >= 2) c = numerics::data::pearson(pairs_i, pairs_j);
                corr[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = c;
                corr[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = c;
            }
        }
        return corr;
    }

    // --- ComputeEffectiveSampleSize (C# 1506-1531). Empty matrix == C# null (compute from data). ---
    double compute_effective_sample_size(const Grid& correlation_matrix = {}) const {
        const Grid corr = correlation_matrix.empty() ? compute_intersite_correlation()
                                                     : correlation_matrix;
        double avg_corr = average_abs_offdiagonal(corr);
        double nominal_n = static_cast<double>(observations()) * static_cast<double>(sites());
        return nominal_n / (1.0 + (sites() - 1) * avg_corr);
    }

    // --- ComputeVarianceInflationFactor (C# 1628-1650). ---
    double compute_variance_inflation_factor(const Grid& correlation_matrix = {}) const {
        const Grid corr = correlation_matrix.empty() ? compute_intersite_correlation()
                                                     : correlation_matrix;
        double avg_corr = average_abs_offdiagonal(corr);
        return 1.0 + (sites() - 1) * avg_corr;
    }

    // --- ComputeEffectiveSampleSizeWeights (C# 1401-1444). ---
    void compute_effective_sample_size_weights(const Grid& correlation_matrix = {}) {
        const Grid corr = correlation_matrix.empty() ? compute_intersite_correlation()
                                                     : correlation_matrix;
        if (static_cast<int>(corr.size()) != sites() ||
            (!corr.empty() && static_cast<int>(corr[0].size()) != sites())) {
            throw std::invalid_argument("Correlation matrix must be Sites x Sites.");
        }

        std::vector<double> weights(static_cast<std::size_t>(sites()), 0.0);
        double sum_weights = 0.0;
        for (int j = 0; j < sites(); ++j) {
            double sum_corr = 0.0;
            int count = 0;
            for (int k = 0; k < sites(); ++k) {
                if (k != j) {
                    sum_corr += std::fabs(
                        corr[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)]);
                    ++count;
                }
            }
            double avg_corr = count > 0 ? sum_corr / count : 0.0;
            weights[static_cast<std::size_t>(j)] = 1.0 / (1.0 + (sites() - 1) * avg_corr);
            sum_weights += weights[static_cast<std::size_t>(j)];
        }
        for (int j = 0; j < sites(); ++j)
            weights[static_cast<std::size_t>(j)] =
                weights[static_cast<std::size_t>(j)] * sites() / sum_weights;
        site_weights_ = weights;
    }

    // --- ConfigureForProperCoverage (C# 1571-1612): the recommended Bayesian configuration. ---
    void configure_for_proper_coverage(
        CorrelationFunctionType correlation_type = CorrelationFunctionType::Exponential,
        bool include_scale_errors = false, bool include_shape_errors = false,
        bool use_weighted_likelihood = false) {
        use_copula_dependence_ = true;
        set_spatial_dependence(GaussianCopula(coordinates_, correlation_type));

        use_location_errors_ = true;
        set_location_errors(SpatialRegressionErrors(coordinates_, correlation_type));

        if (include_scale_errors) {
            use_scale_errors_ = true;
            set_scale_errors(SpatialRegressionErrors(coordinates_, correlation_type));
        }
        if (include_shape_errors) {
            use_shape_errors_ = true;
            set_shape_errors(SpatialRegressionErrors(coordinates_, correlation_type));
        }

        if (use_weighted_likelihood) {
            compute_effective_sample_size_weights();
        } else {
            for (int i = 0; i < sites(); ++i) site_weights_[static_cast<std::size_t>(i)] = 1.0;
        }

        set_default_parameters();
    }

    // --- PredictAtUngauged (C# 1674-1716): trend + kriged spatial errors at a new coordinate.
    // Returns (GEVParams [xi, alpha, kappa], ErrorVariances [var_xi, var_alpha, var_kappa]). ---
    std::pair<std::vector<double>, std::vector<double>> predict_at_ungauged(
        const std::vector<double>& coordinates,
        const std::vector<double>& covariates = {}) const {
        if (coordinates.size() != 2)
            throw std::invalid_argument("Coordinates must be a 2-element array [X, Y].");

        double loc_trend = location_.predict_with_covariates(covariates);
        double scl_trend = scale_.predict_with_covariates(covariates);
        double shp_trend = shape_.predict_with_covariates(covariates);

        double loc_err = 0, scl_err = 0, shp_err = 0;
        double loc_err_var = 0, scl_err_var = 0, shp_err_var = 0;

        if (use_location_errors_ && location_errors_ != nullptr) {
            auto [mean, variance] = location_errors_->get_kriging_prediction(coordinates);
            loc_err = mean;
            loc_err_var = variance;
        }
        if (use_scale_errors_ && scale_errors_ != nullptr) {
            auto [mean, variance] = scale_errors_->get_kriging_prediction(coordinates);
            scl_err = mean;
            scl_err_var = variance;
        }
        if (use_shape_errors_ && shape_errors_ != nullptr) {
            auto [mean, variance] = shape_errors_->get_kriging_prediction(coordinates);
            shp_err = mean;
            shp_err_var = variance;
        }

        double xi = use_log_link_for_location_ ? std::exp(loc_trend + loc_err)
                                               : loc_trend + loc_err;
        double alpha = use_log_link_for_scale_
                           ? std::exp(scl_trend + scl_err)
                           : std::max(scl_trend + scl_err, numerics::kDoubleMachineEpsilon);
        double kappa = shp_trend + shp_err;

        return {{xi, alpha, kappa}, {loc_err_var, scl_err_var, shp_err_var}};
    }

    // --- GetGEVAtUngauged (C# 1724-1728). ---
    numerics::distributions::GeneralizedExtremeValue get_gev_at_ungauged(
        const std::vector<double>& coordinates,
        const std::vector<double>& covariates = {}) const {
        auto [gev_params, err_var] = predict_at_ungauged(coordinates, covariates);
        return numerics::distributions::GeneralizedExtremeValue(gev_params[0], gev_params[1],
                                                                gev_params[2]);
    }

    // --- GenerateRandomValues (C# 1748-1782): seeded per-site GEV sampling. The returned array
    // groups samples by site (site-0 samples, then site-1, ...); length = sampleSize * Sites.
    // Const per the ISimulatable contract (see header note on the omitted SetParameterValues). ---
    std::vector<double> generate_random_values(int sample_size, int seed = -1) const override {
        if (sample_size <= 0)
            throw std::out_of_range("Sample size must be positive.");
        if (parameters_.empty())
            throw std::runtime_error("Parameters must be set before generating random values.");
        if (sites() <= 0) throw std::runtime_error("At least one site must be defined.");

        numerics::sampling::MersenneTwister rng =
            seed > 0 ? numerics::sampling::MersenneTwister(static_cast<std::uint32_t>(seed))
                     : numerics::sampling::MersenneTwister();

        std::vector<double> result(static_cast<std::size_t>(sample_size) *
                                   static_cast<std::size_t>(sites()));
        std::size_t result_index = 0;
        for (int s = 0; s < sites(); ++s) {
            std::vector<double> gp = get_gev_parameters(s);
            numerics::distributions::GeneralizedExtremeValue gev(gp[0], gp[1], gp[2]);
            for (int i = 0; i < sample_size; ++i)
                result[result_index++] = gev.inverse_cdf(rng.next_double());
        }
        return result;
    }

    // --- Clone (C# 1249-1282): deep, independent copy. Preserves the source's current parameter
    // values (does NOT re-run set_default_parameters). ---
    SpatialGEV clone() const {
        SpatialGEV c(at_site_data_, coordinates_, clone_trend(location_), clone_trend(scale_),
                     clone_trend(shape_));
        c.use_copula_dependence_ = use_copula_dependence_;
        c.use_location_errors_ = use_location_errors_;
        c.use_scale_errors_ = use_scale_errors_;
        c.use_shape_errors_ = use_shape_errors_;
        c.use_log_link_for_location_ = use_log_link_for_location_;
        c.use_log_link_for_scale_ = use_log_link_for_scale_;
        c.site_weights_ = site_weights_;

        if (spatial_dependence_ != nullptr)
            c.spatial_dependence_ = std::make_unique<GaussianCopula>(spatial_dependence_->clone());
        if (location_errors_ != nullptr)
            c.location_errors_ =
                std::make_unique<SpatialRegressionErrors>(location_errors_->clone());
        if (scale_errors_ != nullptr)
            c.scale_errors_ = std::make_unique<SpatialRegressionErrors>(scale_errors_->clone());
        if (shape_errors_ != nullptr)
            c.shape_errors_ = std::make_unique<SpatialRegressionErrors>(shape_errors_->clone());

        // The trend Clone()s already deep-copied the parameter values; rebuild the flat list to
        // reflect this clone's (possibly copula/error-augmented) configuration, then restore the
        // source's current values so Clone() preserves them (C# 1274-1281 rationale).
        std::vector<double> values;
        values.reserve(parameters_.size());
        for (const ModelParameter& p : parameters_) values.push_back(p.value());
        c.set_default_parameters();
        if (values.size() == static_cast<std::size_t>(c.number_of_parameters()))
            c.set_parameter_values(values);
        return c;
    }

   private:
    // Appends value copies of a child model's parameters to the flat list.
    void append_params(const std::vector<ModelParameter>& src) {
        for (const ModelParameter& p : src) parameters_.push_back(p);
    }

    // Distributes a slice of `parameters` to a trend child; returns the advanced index.
    static int set_child_values(trend_functions::GeneralLinearFunction& child,
                                const std::vector<double>& parameters, int index) {
        std::vector<double> vals;
        for (int i = 0; i < child.number_of_parameters(); ++i)
            vals.push_back(parameters[static_cast<std::size_t>(index++)]);
        child.set_parameter_values(vals);
        return index;
    }
    static int set_error_values(SpatialRegressionErrors& child,
                                const std::vector<double>& parameters, int index) {
        std::vector<double> vals;
        for (int i = 0; i < child.number_of_parameters(); ++i)
            vals.push_back(parameters[static_cast<std::size_t>(index++)]);
        child.set_parameter_values(vals);
        return index;
    }

    // Deep-copies a trend model via its polymorphic clone(), downcast back to the concrete type
    // (SpatialGEV only ever stores GeneralLinearFunction trends).
    static trend_functions::GeneralLinearFunction clone_trend(
        const trend_functions::GeneralLinearFunction& src) {
        return src;  // value copy = deep copy (ModelParameter deep-copies its prior)
    }

    // GetGEVParametersLocal (C# 611-631): assemble (xi, alpha, kappa) from the supplied trend /
    // error models, honoring the link functions and error gates.
    std::vector<double> gev_parameters_local(
        int site_index, const trend_functions::GeneralLinearFunction& loc,
        const trend_functions::GeneralLinearFunction& scl,
        const trend_functions::GeneralLinearFunction& shp, const SpatialRegressionErrors* loc_err,
        const SpatialRegressionErrors* scl_err, const SpatialRegressionErrors* shp_err) const {
        double loc_trend = loc.predict(site_index);
        double loc_error =
            (use_location_errors_ && loc_err != nullptr) ? loc_err->get_error(site_index) : 0.0;
        double xi = use_log_link_for_location_ ? std::exp(loc_trend + loc_error)
                                               : loc_trend + loc_error;

        double scl_trend = scl.predict(site_index);
        double scl_error =
            (use_scale_errors_ && scl_err != nullptr) ? scl_err->get_error(site_index) : 0.0;
        double alpha = use_log_link_for_scale_
                           ? std::exp(scl_trend + scl_error)
                           : std::max(scl_trend + scl_error, numerics::kDoubleMachineEpsilon);

        double kappa = shp.predict(site_index);
        if (use_shape_errors_ && shp_err != nullptr) kappa += shp_err->get_error(site_index);

        return {xi, alpha, kappa};
    }

    // Snapshots SiteWeights once per LL evaluation (C# localSiteWeights).
    std::vector<double> site_weights_snapshot() const {
        return site_weights_.empty()
                   ? std::vector<double>(static_cast<std::size_t>(sites()), 0.0)
                   : site_weights_;
    }

    // Clones trend/copula/error children into locals and distributes `parameters` to them in
    // marshaling order (the shared body of ComputeLogLikelihoodInternal / PointwiseData* and the
    // spatial-error half of PointwisePrior*).
    void distribute_locals(const std::vector<double>& parameters,
                           trend_functions::GeneralLinearFunction& loc,
                           trend_functions::GeneralLinearFunction& scl,
                           trend_functions::GeneralLinearFunction& shp,
                           std::optional<GaussianCopula>& cop,
                           std::optional<SpatialRegressionErrors>& le,
                           std::optional<SpatialRegressionErrors>& se,
                           std::optional<SpatialRegressionErrors>& she) const {
        if (use_copula_dependence_ && spatial_dependence_ != nullptr)
            cop = spatial_dependence_->clone();
        if (use_location_errors_ && location_errors_ != nullptr) le = location_errors_->clone();
        if (use_scale_errors_ && scale_errors_ != nullptr) se = scale_errors_->clone();
        if (use_shape_errors_ && shape_errors_ != nullptr) she = shape_errors_->clone();

        int index = 0;
        if (cop) {
            std::vector<double> cp;
            for (int i = 0; i < cop->number_of_parameters(); ++i)
                cp.push_back(parameters[static_cast<std::size_t>(index++)]);
            cop->set_parameter_values(cp);
        }
        index = set_child_values(loc, parameters, index);
        index = set_child_values(scl, parameters, index);
        index = set_child_values(shp, parameters, index);
        if (le) index = set_error_values(*le, parameters, index);
        if (se) index = set_error_values(*se, parameters, index);
        if (she) index = set_error_values(*she, parameters, index);
    }

    // ComputeLogLikelihoodInternal (C# 639-831): data likelihood on cloned locals, INCLUDING the
    // spatial-error log densities (preserved non-canonical decomposition).
    double compute_log_likelihood_internal(const std::vector<double>& parameters) const {
        std::vector<double> weights = site_weights_snapshot();
        auto loc = location_;
        auto scl = scale_;
        auto shp = shape_;
        std::optional<GaussianCopula> cop;
        std::optional<SpatialRegressionErrors> le, se, she;
        distribute_locals(parameters, loc, scl, shp, cop, le, se, she);
        SpatialRegressionErrors* lep = le ? &*le : nullptr;
        SpatialRegressionErrors* sep = se ? &*se : nullptr;
        SpatialRegressionErrors* shp_ep = she ? &*she : nullptr;

        numerics::distributions::GeneralizedExtremeValue local_gev;
        double log_lh = 0.0;
        const double neg_inf = -std::numeric_limits<double>::infinity();
        const bool use_copula = use_copula_dependence_ && cop.has_value();

        for (int i = 0; i < observations(); ++i) {
            std::vector<double> z(static_cast<std::size_t>(sites()), 0.0);
            bool has_data = false;
            for (int j = 0; j < sites(); ++j) {
                double x =
                    at_site_data_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                if (std::isnan(x)) continue;
                has_data = true;
                std::vector<double> gp = gev_parameters_local(j, loc, scl, shp, lep, sep, shp_ep);
                if (gp[1] <= 0) return neg_inf;
                local_gev.set_parameters(gp);
                double marg = local_gev.log_pdf(x);
                if (!numerics::is_finite(marg)) return neg_inf;
                log_lh += weights[static_cast<std::size_t>(j)] * marg;
                if (use_copula) {
                    double u = local_gev.cdf(x);
                    z[static_cast<std::size_t>(j)] =
                        numerics::distributions::Normal::standard_z(u);
                }
            }
            if (use_copula && has_data) {
                double cop_log_lh = cop->log_pdf(z);
                if (!numerics::is_finite(cop_log_lh)) return neg_inf;
                log_lh += cop_log_lh;
            }
        }

        // Spatial error contributions (Gaussian process priors) -- counted in the DATA likelihood
        // by the preserved non-canonical decomposition (see file header).
        if (lep != nullptr) {
            double v = le->log_pdf();
            if (!numerics::is_finite(v)) return neg_inf;
            log_lh += v;
        }
        if (sep != nullptr) {
            double v = se->log_pdf();
            if (!numerics::is_finite(v)) return neg_inf;
            log_lh += v;
        }
        if (shp_ep != nullptr) {
            double v = she->log_pdf();
            if (!numerics::is_finite(v)) return neg_inf;
            log_lh += v;
        }
        return log_lh;
    }

    // Average absolute off-diagonal correlation (upper triangle), C# ComputeEffectiveSampleSize /
    // ComputeVarianceInflationFactor share this.
    double average_abs_offdiagonal(const Grid& corr) const {
        double sum_corr = 0.0;
        int count = 0;
        for (int i = 0; i < sites(); ++i)
            for (int j = i + 1; j < sites(); ++j) {
                sum_corr += std::fabs(
                    corr[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
                ++count;
            }
        return count > 0 ? sum_corr / count : 0.0;
    }

    Grid at_site_data_;
    Grid coordinates_;
    trend_functions::GeneralLinearFunction location_;
    trend_functions::GeneralLinearFunction scale_;
    trend_functions::GeneralLinearFunction shape_;
    std::unique_ptr<GaussianCopula> spatial_dependence_;
    std::unique_ptr<SpatialRegressionErrors> location_errors_;
    std::unique_ptr<SpatialRegressionErrors> scale_errors_;
    std::unique_ptr<SpatialRegressionErrors> shape_errors_;
    bool use_copula_dependence_ = false;
    bool use_location_errors_ = false;
    bool use_scale_errors_ = false;
    bool use_shape_errors_ = false;
    bool use_log_link_for_location_ = true;
    bool use_log_link_for_scale_ = true;
    std::vector<double> site_weights_;
};

}  // namespace corehydro::models::spatial_extremes
