// ported from: RMC-BestFit/src/RMC.BestFit/Models/RatingCurve/RatingCurve.cs @ fc28c0c
//
// Stage-discharge rating curve model using the BaRatin matrix-of-controls framework in ADDITION
// mode (controls accumulate as stage rises), on top of ModelBase (the shared likelihood surface
// the Estimation layer optimizes/samples) and ISimulatable<std::vector<double>> (C#
// ISimulatable<double[]>):
//
//   Q(h) = sum_k alpha_k * (h - xi_k)^beta_k * 1{h > xi_k}   for k in 1..NumberOfSegments (1-3)
//
// where alpha_k = 10^(log10alpha_k) and xi_k = h_k is control k's activation stage (xi_1 = h1 is
// the main-channel cease-to-flow stage, fit directly; for k >= 2 the BaRatin continuity derivation
// collapses the "b" offset to the activation breakpoint, so xi_k = h_k). Parameter vector layout
// (length = 3*NumberOfSegments + 1):
//   1 seg: [h1, log10alpha1, beta1, sigma]
//   2 seg: [h1, log10alpha1, beta1, h2, log10alpha2, beta2, sigma]
//   3 seg: [h1, log10alpha1, beta1, h2, log10alpha2, beta2, h3, log10alpha3, beta3, sigma]
// Errors are modeled in log10-space with residual sigma. Uniform parameter priors are set in
// SetDefaultParameters; the optional Jeffreys 1/sigma scale prior (UseJeffreysRuleForScale,
// default true) is added on top in PriorLogLikelihood / PointwisePriorLogLikelihood.
//
// Structural mirroring: the class/member/method layout follows RatingCurve.cs. The property
// setters that invalidate the aligned-observations cache and re-run SetDefaultParameters() on
// change (StageData/DischargeData/NumberOfSegments, C# lines 169/196/223) preserve that effective
// behavior here; the never-mutate rule is RELAXED for these stateful model objects (per
// .claude/CLAUDE.md), matching the upstream mutable WPF-binding design.
//
// Deliberately NOT ported (documented; C# governs -- deviations noted in task-R1-report.md):
//   - XML: the XElement ctor (C#:118), ToXElement (C#:1063), and the UseJeffreysRuleForScale /
//     NumberOfSegments / UseDefaultFlatPriors XElement attributes -- project-wide non-port.
//   - INotifyPropertyChanged / RaisePropertyChange / Parameter_PropertyChanged: WPF data-binding
//     plumbing, ported as silent no-ops (parameters_ is a plain std::vector, no change
//     notification is threaded through it).
//   - StageData_CollectionChanged / DischargeData_CollectionChanged auto-retrain (C#:270/281):
//     the ported TimeSeries adapter raises no CollectionChanged event; in-place ordinate mutation
//     after construction does NOT auto-invalidate the aligned cache here. Reassigning the series
//     via set_stage_data / set_discharge_data DOES invalidate it (the reassignment path is ported).
//   - IModel Clone() (C#:1041): the C++ core has no virtual IModel::Clone (see model_base.hpp) and
//     no R1 fit path needs a clone, so it is omitted (T1/S4 precedent).
//   - SetParameterValues override (C#:461): the ModelBase base already throws on a wrong-length
//     vector; the C# null-guard is vacuous for a const std::vector& (no null), so the base body is
//     inherited unchanged.
//   - The RatingCurveAnalysis wrapper is a separate class, NOT in R1 scope.
//
// RNG deviation (documented): the seeded Predict(parameters, stage, seed) overload (C#:915) draws
// its stochastic noise from System.Random (a .NET LCG with no ported equivalent); this port
// substitutes the ported MersenneTwister. Same-seed reproducibility and different-seed divergence
// hold, but the exact seeded VALUES are not C#-reproducible (a P4 concern). GenerateRandomValues
// (C#:1172) and GenerateSyntheticData (C#:1243) use the ported MersenneTwister exactly as the C#
// does (bit-exact -- the C# already uses Numerics.Sampling.MersenneTwister there).
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "bestfit/models/support/data_component.hpp"
#include "bestfit/models/support/model_base.hpp"
#include "bestfit/models/support/model_parameter.hpp"
#include "bestfit/models/support/prior_component.hpp"
#include "bestfit/models/support/simulatable.hpp"
#include "bestfit/models/support/subscript_formatter.hpp"
#include "bestfit/models/support/validation_result.hpp"
#include "bestfit/numerics/data/statistics.hpp"
#include "bestfit/numerics/data/time_series/time_series.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "bestfit/numerics/distributions/uniform.hpp"
#include "bestfit/numerics/sampling/mersenne_twister.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::models {

class RatingCurve : public ModelBase, public ISimulatable<std::vector<double>> {
   public:
    using TimeSeries = numerics::data::TimeSeries;

    // One date-aligned observation (C# tuple (DateTime Date, double Stage, double Discharge)).
    struct AlignedObservation {
        long date;
        double stage;
        double discharge;
    };

    // Generated synthetic (stage, discharge) pair of series (C# tuple return, C#:1243).
    struct SyntheticData {
        TimeSeries stage_data;
        TimeSeries discharge_data;
    };

    // Minimum number of date-aligned (stage, discharge) pairs required to fit (C#:530).
    static constexpr int kMinimumAlignedObservations = 10;

    // --- Construction (C#:92-110). ---

    // Single-segment rating curve with default parameters (C#:92).
    RatingCurve() {
        number_of_segments_ = 1;
        set_default_parameters();
    }

    // Rating curve over a stage series + a discharge series (C#:104). Setting the series
    // invalidates the aligned cache and (per the C# setters) triggers SetDefaultParameters.
    RatingCurve(const TimeSeries& stage_data, const TimeSeries& discharge_data,
                int number_of_segments = 1) {
        stage_data_ = stage_data;
        discharge_data_ = discharge_data;
        aligned_observations_cache_.reset();
        number_of_segments_ = number_of_segments;
        set_default_parameters();
    }

    // --- Properties. ---

    bool has_stage_data() const { return stage_data_.has_value(); }
    const TimeSeries& stage_data() const { return *stage_data_; }
    void set_stage_data(const TimeSeries& value) {
        // C# unsubscribes/subscribes CollectionChanged here -> no-op in this port.
        stage_data_ = value;
        aligned_observations_cache_.reset();
        // RaisePropertyChange -> no-op.
        if (use_default_flat_priors()) set_default_parameters();
    }

    bool has_discharge_data() const { return discharge_data_.has_value(); }
    const TimeSeries& discharge_data() const { return *discharge_data_; }
    void set_discharge_data(const TimeSeries& value) {
        discharge_data_ = value;
        aligned_observations_cache_.reset();
        if (use_default_flat_priors()) set_default_parameters();
    }

    int number_of_segments() const { return number_of_segments_; }
    void set_number_of_segments(int value) {
        if (number_of_segments_ != value) {
            number_of_segments_ = value;
            // Rebuild the parameter list to match the new segment count (C#:237).
            set_default_parameters();
        }
    }

    bool use_jeffreys_rule_for_scale() const { return use_jeffreys_rule_for_scale_; }
    void set_use_jeffreys_rule_for_scale(bool value) {
        if (use_jeffreys_rule_for_scale_ != value) use_jeffreys_rule_for_scale_ = value;
    }

    // Convenience: current parameter values (C# Parameters.Select(x => x.Value).ToArray()).
    std::vector<double> parameter_values() const {
        std::vector<double> v;
        v.reserve(parameters_.size());
        for (const auto& p : parameters_) v.push_back(p.value());
        return v;
    }

    // --- SetDefaultParameters (C#:290). ---
    void set_default_parameters() override {
        // Get stage constraints.
        double min_h = 0, max_h = 10, range_h = 10;
        if (stage_data_ && stage_data_->count() > 0) {
            min_h = stage_data_->min_value();
            max_h = max_value(*stage_data_);
            range_h = max_h - min_h;
        }

        // Get discharge constraints for the log-space standard deviation.
        double sigma_ub = 2.0;  // default for log-space errors
        if (discharge_data_ && discharge_data_->count() > 0) {
            std::vector<double> log_q;
            for (int i = 0; i < discharge_data_->count(); ++i) {
                double v = (*discharge_data_)[i].value();
                if (v > 0) log_q.push_back(std::log10(v));
            }
            if (log_q.size() > 1) {
                double std_dev_log_q = numerics::data::standard_deviation(log_q);
                sigma_ub = std::ceil(std_dev_log_q * 3);  // 3x std dev for the upper bound
            }
        }

        // Zero-flow stage bounds for segment 1 (h1, the main-channel cease-to-flow stage).
        double h1_min = min_h - range_h;
        double h1_max = min_h + 0.1 * range_h;

        parameters_.clear();

        // Segment 1 location (h1).
        parameters_.emplace_back(
            /*owner_name=*/"", /*name=*/std::string("Zero-Flow Stage (h") + to_subscript(1) + ")",
            /*value=*/min_h - 0.1 * range_h, /*lower_bound=*/h1_min, /*upper_bound=*/h1_max,
            std::make_unique<numerics::distributions::Uniform>(h1_min, h1_max));

        // Segment 1 coefficient log10(alpha1) and exponent beta1.
        parameters_.emplace_back(
            /*owner_name=*/"", /*name=*/std::string("Coefficient (\xCE\xB1") + to_subscript(1) + ")",
            /*value=*/0.0, /*lower_bound=*/-5.0, /*upper_bound=*/5.0,
            std::make_unique<numerics::distributions::Uniform>(-5.0, 5.0));
        parameters_.emplace_back(
            /*owner_name=*/"", /*name=*/std::string("Exponent (\xCE\xB2") + to_subscript(1) + ")",
            /*value=*/2.0, /*lower_bound=*/0.5, /*upper_bound=*/5.0,
            std::make_unique<numerics::distributions::Uniform>(0.5, 5.0), /*is_positive=*/true);

        if (number_of_segments_ >= 2) {
            double h2_min = min_h + 0.2 * range_h;
            double h2_max = min_h + 0.7 * range_h;
            double h2_default = min_h + 0.45 * range_h;

            parameters_.emplace_back(
                /*owner_name=*/"",
                /*name=*/std::string("Activation Stage (h") + to_subscript(2) + ")",
                /*value=*/h2_default, /*lower_bound=*/h2_min, /*upper_bound=*/h2_max,
                std::make_unique<numerics::distributions::Uniform>(h2_min, h2_max));
            parameters_.emplace_back(
                /*owner_name=*/"",
                /*name=*/std::string("Coefficient (\xCE\xB1") + to_subscript(2) + ")",
                /*value=*/0.0, /*lower_bound=*/-5.0, /*upper_bound=*/5.0,
                std::make_unique<numerics::distributions::Uniform>(-5.0, 5.0));
            parameters_.emplace_back(
                /*owner_name=*/"",
                /*name=*/std::string("Exponent (\xCE\xB2") + to_subscript(2) + ")",
                /*value=*/2.0, /*lower_bound=*/0.5, /*upper_bound=*/5.0,
                std::make_unique<numerics::distributions::Uniform>(0.5, 5.0), /*is_positive=*/true);
        }

        if (number_of_segments_ >= 3) {
            double h3_min = min_h + 0.5 * range_h;
            double h3_max = max_h;
            double h3_default = min_h + 0.75 * range_h;

            parameters_.emplace_back(
                /*owner_name=*/"",
                /*name=*/std::string("Activation Stage (h") + to_subscript(3) + ")",
                /*value=*/h3_default, /*lower_bound=*/h3_min, /*upper_bound=*/h3_max,
                std::make_unique<numerics::distributions::Uniform>(h3_min, h3_max));
            parameters_.emplace_back(
                /*owner_name=*/"",
                /*name=*/std::string("Coefficient (\xCE\xB1") + to_subscript(3) + ")",
                /*value=*/0.0, /*lower_bound=*/-5.0, /*upper_bound=*/5.0,
                std::make_unique<numerics::distributions::Uniform>(-5.0, 5.0));
            parameters_.emplace_back(
                /*owner_name=*/"",
                /*name=*/std::string("Exponent (\xCE\xB2") + to_subscript(3) + ")",
                /*value=*/2.0, /*lower_bound=*/0.5, /*upper_bound=*/5.0,
                std::make_unique<numerics::distributions::Uniform>(0.5, 5.0), /*is_positive=*/true);
        }

        // Scale parameter (log-space standard deviation).
        parameters_.emplace_back(
            /*owner_name=*/"", /*name=*/"Scale (\xCF\x83)", /*value=*/sigma_ub / 3,
            /*lower_bound=*/numerics::kDoubleMachineEpsilon, /*upper_bound=*/sigma_ub,
            std::make_unique<numerics::distributions::Uniform>(numerics::kDoubleMachineEpsilon,
                                                               sigma_ub),
            /*is_positive=*/true);
    }

    // --- GetAlignedObservations (C#:498): date-inner-join, lazily cached. ---
    const std::vector<AlignedObservation>& get_aligned_observations() const {
        if (aligned_observations_cache_) return *aligned_observations_cache_;

        std::vector<AlignedObservation> result;
        if (!stage_data_ || !discharge_data_) {
            aligned_observations_cache_ = std::move(result);
            return *aligned_observations_cache_;
        }

        std::unordered_map<long, double> discharge_by_date;
        discharge_by_date.reserve(static_cast<std::size_t>(discharge_data_->count()));
        for (int j = 0; j < discharge_data_->count(); ++j)
            discharge_by_date[(*discharge_data_)[j].index()] = (*discharge_data_)[j].value();

        for (int i = 0; i < stage_data_->count(); ++i) {
            long idx = (*stage_data_)[i].index();
            auto it = discharge_by_date.find(idx);
            if (it != discharge_by_date.end())
                result.push_back({idx, (*stage_data_)[i].value(), it->second});
        }
        std::sort(result.begin(), result.end(),
                  [](const AlignedObservation& a, const AlignedObservation& b) {
                      return a.date < b.date;
                  });

        aligned_observations_cache_ = std::move(result);
        return *aligned_observations_cache_;
    }

    // --- DataLogLikelihood (C#:540): log10-space residuals ~ Normal(0, sigma). ---
    double data_log_likelihood(std::vector<double>& parameters) const override {
        if (!stage_data_ || !discharge_data_) return -std::numeric_limits<double>::infinity();

        const auto& aligned = get_aligned_observations();
        if (static_cast<int>(aligned.size()) < kMinimumAlignedObservations)
            return -std::numeric_limits<double>::infinity();

        for (double v : parameters)
            if (std::isnan(v)) return -std::numeric_limits<double>::infinity();

        if (!validate_segment_ordering(parameters))
            return -std::numeric_limits<double>::infinity();

        double sigma = parameters.back();
        numerics::distributions::Normal norm_dist(0.0, sigma);
        double log_lh = 0.0;

        for (const auto& obs : aligned) {
            double pred_q = predict(parameters, obs.stage);
            if (pred_q <= 0) return -std::numeric_limits<double>::infinity();
            double residual = std::log10(obs.discharge) - std::log10(pred_q);
            log_lh += norm_dist.log_pdf(residual);
        }
        return log_lh;
    }

    // --- PointwiseDataLogLikelihood (C#:590). ---
    std::vector<double> pointwise_data_log_likelihood(
        const std::vector<double>& parameters) const override {
        if (!stage_data_ || !discharge_data_) return {};

        const auto& aligned = get_aligned_observations();
        int n = static_cast<int>(aligned.size());
        if (n == 0) return {};

        for (double v : parameters)
            if (std::isnan(v))
                return std::vector<double>(static_cast<std::size_t>(n),
                                           -std::numeric_limits<double>::infinity());

        if (!validate_segment_ordering(parameters))
            return std::vector<double>(static_cast<std::size_t>(n),
                                       -std::numeric_limits<double>::infinity());

        double sigma = parameters.back();
        numerics::distributions::Normal norm_dist(0.0, sigma);
        std::vector<double> result(static_cast<std::size_t>(n));

        for (int i = 0; i < n; ++i) {
            double pred_q = predict(parameters, aligned[static_cast<std::size_t>(i)].stage);
            if (pred_q <= 0) {
                result[static_cast<std::size_t>(i)] = -std::numeric_limits<double>::infinity();
                continue;
            }
            double residual =
                std::log10(aligned[static_cast<std::size_t>(i)].discharge) - std::log10(pred_q);
            result[static_cast<std::size_t>(i)] = norm_dist.log_pdf(residual);
        }
        return result;
    }

    // --- PointwiseDataLogLikelihoodComponents (C#:649). ---
    std::vector<DataComponent> pointwise_data_log_likelihood_components(
        const std::vector<double>& parameters) const override {
        if (!stage_data_ || !discharge_data_) return {};

        const auto& aligned = get_aligned_observations();
        int n = static_cast<int>(aligned.size());
        std::vector<DataComponent> result;
        result.reserve(static_cast<std::size_t>(n));
        if (n == 0) return result;

        for (double v : parameters) {
            if (std::isnan(v)) {
                for (int j = 0; j < n; ++j)
                    result.emplace_back(j, -std::numeric_limits<double>::infinity(),
                                        aligned[static_cast<std::size_t>(j)].stage,
                                        DataComponentType::Exact, 1, std::to_string(j));
                return result;
            }
        }

        if (!validate_segment_ordering(parameters)) {
            for (int i = 0; i < n; ++i)
                result.emplace_back(i, -std::numeric_limits<double>::infinity(),
                                    aligned[static_cast<std::size_t>(i)].stage,
                                    DataComponentType::Exact, 1, std::to_string(i));
            return result;
        }

        double sigma = parameters.back();
        numerics::distributions::Normal norm_dist(0.0, sigma);

        for (int i = 0; i < n; ++i) {
            double stage = aligned[static_cast<std::size_t>(i)].stage;
            double pred_q = predict(parameters, stage);
            double log_lh;
            if (pred_q <= 0) {
                log_lh = -std::numeric_limits<double>::infinity();
            } else {
                double residual =
                    std::log10(aligned[static_cast<std::size_t>(i)].discharge) - std::log10(pred_q);
                log_lh = norm_dist.log_pdf(residual);
            }
            result.emplace_back(i, log_lh, stage, DataComponentType::Exact, 1,
                                std::string("Stage=") + format_f2(stage));
        }
        return result;
    }

    // --- PriorLogLikelihood (C#:708): parameter priors + optional Jeffreys 1/sigma. ---
    double prior_log_likelihood(std::vector<double>& parameters) const override {
        if (parameters.size() < parameters_.size())
            return -std::numeric_limits<double>::infinity();

        double sigma = parameters.back();
        double log_lh = 0.0;
        for (std::size_t i = 0; i < parameters_.size(); ++i)
            log_lh += parameters_[i].prior_distribution().log_pdf(parameters[i]);

        if (use_jeffreys_rule_for_scale_)
            log_lh -= sigma > 0 ? std::log(sigma) : std::numeric_limits<double>::infinity();

        if (!numerics::is_finite(log_lh)) return -std::numeric_limits<double>::infinity();
        return log_lh;
    }

    // --- PointwisePriorLogLikelihood (C#:735). ---
    std::vector<PriorComponent> pointwise_prior_log_likelihood(
        const std::vector<double>& parameters) const override {
        std::vector<PriorComponent> result;
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            double ll = parameters_[i].prior_distribution().log_pdf(parameters[i]);
            const std::string& param_name = parameters_[i].owner_name().empty()
                                                ? parameters_[i].name()
                                                : parameters_[i].owner_name();
            result.emplace_back("Parameter Prior: " + param_name, ll,
                                PriorComponentType::ParameterPrior);
        }
        if (use_jeffreys_rule_for_scale_) {
            double sigma = parameters.back();
            double ll = sigma > 0 ? -std::log(sigma) : -std::numeric_limits<double>::infinity();
            result.emplace_back("Jeffreys Scale: \xCF\x83", ll,
                                PriorComponentType::JeffreysScalePrior);
        }
        return result;
    }

    // --- Predict (C#:820): addition-mode segment sum, real-space discharge. ---
    double predict(const std::vector<double>& parameters, double stage) const {
        double h1 = parameters[0];
        double depth1 = stage - h1;
        if (depth1 <= 0) return 0.0;

        double q = std::pow(10, parameters[1]) * std::pow(depth1, parameters[2]);

        if (number_of_segments_ >= 2) {
            double depth2 = stage - parameters[3];
            if (depth2 > 0) q += std::pow(10, parameters[4]) * std::pow(depth2, parameters[5]);
        }
        if (number_of_segments_ >= 3) {
            double depth3 = stage - parameters[6];
            if (depth3 > 0) q += std::pow(10, parameters[7]) * std::pow(depth3, parameters[8]);
        }
        return q;
    }

    // --- Seeded Predict (C#:915): adds log10-space Normal(0, sigma) noise. MersenneTwister
    // substituted for System.Random (see header). ---
    double predict(const std::vector<double>& parameters, double stage, int seed) const {
        numerics::sampling::MersenneTwister prng(static_cast<std::uint32_t>(seed));
        double sigma = parameters.back();
        numerics::distributions::Normal err_dist(0.0, sigma);
        double log_q = std::log10(predict(parameters, stage));
        log_q += err_dist.inverse_cdf(prng.next_double());
        return std::pow(10, log_q);
    }

    // --- GetLog10Alpha (C#:861): indexer into the fit parameters. ---
    double get_log10_alpha(int segment_one_based, const std::vector<double>& parameters) const {
        if (segment_one_based < 1 || segment_one_based > number_of_segments_)
            throw std::out_of_range("Segment index must be between 1 and " +
                                    std::to_string(number_of_segments_) + ".");
        switch (segment_one_based) {
            case 1:
                return parameters[1];
            case 2:
                return parameters[4];
            case 3:
                return parameters[7];
            default:
                throw std::out_of_range("segment_one_based");
        }
    }

    // --- GetLocation (C#:893): xi_k offset (h1 fit; h_k activation stage for k >= 2). ---
    double get_location(int segment_one_based, const std::vector<double>& parameters) const {
        if (segment_one_based < 1 || segment_one_based > number_of_segments_)
            throw std::out_of_range("Segment index must be between 1 and " +
                                    std::to_string(number_of_segments_) + ".");
        switch (segment_one_based) {
            case 1:
                return parameters[0];
            case 2:
                return parameters[3];
            case 3:
                return parameters[6];
            default:
                return std::numeric_limits<double>::quiet_NaN();
        }
    }

    // --- Residuals (C#:975): log10-space residuals over the aligned observations. ---
    std::vector<double> residuals(const std::vector<double>& parameters) const {
        const auto& aligned = get_aligned_observations();
        std::vector<double> res(aligned.size());
        for (std::size_t i = 0; i < aligned.size(); ++i) {
            double pred_q = predict(parameters, aligned[i].stage);
            res[i] = pred_q <= 0 ? std::numeric_limits<double>::quiet_NaN()
                                 : std::log10(aligned[i].discharge) - std::log10(pred_q);
        }
        return res;
    }

    // --- FittedValues (C#:1005): log10-space fitted values over the aligned observations. ---
    std::vector<double> fitted_values(const std::vector<double>& parameters) const {
        const auto& aligned = get_aligned_observations();
        std::vector<double> fitted(aligned.size());
        for (std::size_t i = 0; i < aligned.size(); ++i) {
            double pred_q = predict(parameters, aligned[i].stage);
            fitted[i] = pred_q <= 0 ? std::numeric_limits<double>::quiet_NaN() : std::log10(pred_q);
        }
        return fitted;
    }

    // --- GenerateRatingTable (C#:1025): [i][0] stage, [i][1] discharge. ---
    std::vector<std::array<double, 2>> generate_rating_table(const std::vector<double>& parameters,
                                                             double min_stage, double max_stage,
                                                             int num_points = 100) const {
        std::vector<std::array<double, 2>> table(static_cast<std::size_t>(num_points));
        double step_size = (max_stage - min_stage) / (num_points - 1);
        for (int i = 0; i < num_points; ++i) {
            double stage = min_stage + i * step_size;
            table[static_cast<std::size_t>(i)][0] = stage;
            table[static_cast<std::size_t>(i)][1] = predict(parameters, stage);
        }
        return table;
    }

    // --- Validate (C#:1079). ---
    ValidationResult validate() const override {
        ValidationResult result;  // is_valid = true

        if (!stage_data_) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Stage time series is missing. Please select a valid time series.");
        }
        if (!discharge_data_) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Discharge time series is missing. Please select a valid time series.");
        }

        if (stage_data_ && discharge_data_) {
            int aligned_count = static_cast<int>(get_aligned_observations().size());
            if (aligned_count < kMinimumAlignedObservations) {
                result.is_valid = false;
                result.validation_messages.push_back(
                    "Error: Stage and discharge time series must share at least " +
                    std::to_string(kMinimumAlignedObservations) +
                    " common dates to fit a rating curve (found " + std::to_string(aligned_count) +
                    ").");
            }
            bool any_nonpositive = false;
            for (int i = 0; i < discharge_data_->count(); ++i)
                if ((*discharge_data_)[i].value() <= 0) any_nonpositive = true;
            if (any_nonpositive) {
                result.is_valid = false;
                result.validation_messages.push_back(
                    "Error: All discharge values must be positive (log-space model requires Q > "
                    "0).");
            }
        }

        if (number_of_segments_ < 1 || number_of_segments_ > 3) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Number of segments must be between 1 and 3.");
        }

        int expected_params = number_of_segments_ * 3 + 1;
        if (static_cast<int>(parameters_.size()) != expected_params) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Expected " + std::to_string(expected_params) + " parameters but found " +
                std::to_string(parameters_.size()) + ".");
        }

        for (const auto& p : parameters_) {
            ValidationResult v = p.validate();
            if (!v.is_valid) {
                result.is_valid = false;
                for (const auto& m : v.validation_messages) result.validation_messages.push_back(m);
            }
        }

        if (!parameters_.empty()) {
            std::vector<double> param_values = parameter_values();
            if (!validate_segment_ordering(param_values)) {
                result.is_valid = false;
                // C# message uses Greek subscript glyphs; ASCII-normalized here (keeps the
                // "Segment ordering" token the tests key on). See task-R1-report.md.
                result.validation_messages.push_back(
                    "Error: Segment ordering invalid. Require h1 < h2 < h3.");
            }
        }

        return result;
    }

    // --- GenerateRandomValues (C#:1172): ISimulatable entry point. Bit-exact MersenneTwister. ---
    std::vector<double> generate_random_values(int sample_size, int seed = -1) const override {
        if (sample_size <= 0) throw std::out_of_range("Sample size must be positive.");
        if (!stage_data_ || stage_data_->count() == 0)
            throw std::runtime_error(
                "StageData cannot be null or empty when generating random values.");
        if (parameters_.empty())
            throw std::runtime_error("Parameters must be set before generating random values.");

        numerics::sampling::MersenneTwister rng =
            seed > 0 ? numerics::sampling::MersenneTwister(static_cast<std::uint32_t>(seed))
                     : numerics::sampling::MersenneTwister();
        numerics::distributions::Normal normal(0.0, 1.0);

        double sigma = parameters_.back().value();
        std::vector<double> param_values = parameter_values();

        std::vector<double> result(static_cast<std::size_t>(sample_size));
        for (int i = 0; i < sample_size; ++i) {
            int stage_index = rng.next(0, stage_data_->count());
            double stage = (*stage_data_)[stage_index].value();
            double pred_q = predict(param_values, stage);
            double log_q = std::log10(pred_q) + sigma * normal.inverse_cdf(rng.next_double());
            result[static_cast<std::size_t>(i)] = std::pow(10, log_q);
        }
        return result;
    }

    // --- GenerateSyntheticData (C#:1243): synthetic (stage, discharge) series with log-normal
    // error. Bit-exact MersenneTwister (as in the C#). ---
    SyntheticData generate_synthetic_data(int sample_size, double min_stage, double max_stage,
                                          int seed = -1) const {
        if (sample_size <= 0) throw std::out_of_range("Sample size must be positive.");
        if (min_stage >= max_stage)
            throw std::out_of_range("Minimum stage must be less than maximum stage.");
        if (parameters_.empty())
            throw std::runtime_error("Parameters must be set before generating synthetic data.");

        numerics::sampling::MersenneTwister rng =
            seed > 0 ? numerics::sampling::MersenneTwister(static_cast<std::uint32_t>(seed))
                     : numerics::sampling::MersenneTwister();

        double sigma = parameters_.back().value();
        numerics::distributions::Normal normal(0.0, sigma);
        std::vector<double> param_values = parameter_values();

        std::vector<std::pair<double, double>> pairs(static_cast<std::size_t>(sample_size));
        for (int i = 0; i < sample_size; ++i) {
            double stage = min_stage + rng.next_double() * (max_stage - min_stage);
            double pred_q = predict(param_values, stage);
            double log_q = std::log10(pred_q) + normal.inverse_cdf(rng.next_double());
            pairs[static_cast<std::size_t>(i)] = {stage, std::pow(10, log_q)};
        }

        std::sort(pairs.begin(), pairs.end(),
                  [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                      return a.first < b.first;
                  });

        std::vector<double> sorted_stages(static_cast<std::size_t>(sample_size));
        std::vector<double> sorted_discharges(static_cast<std::size_t>(sample_size));
        for (int i = 0; i < sample_size; ++i) {
            sorted_stages[static_cast<std::size_t>(i)] = pairs[static_cast<std::size_t>(i)].first;
            sorted_discharges[static_cast<std::size_t>(i)] =
                pairs[static_cast<std::size_t>(i)].second;
        }

        // C# starts the index at DateTime(2000,1,1); the integer-index adapter uses 0 (the date is
        // only a join key, never used in arithmetic -- see time_series.hpp).
        return {TimeSeries(numerics::data::TimeInterval::OneDay, 0, sorted_stages),
                TimeSeries(numerics::data::TimeInterval::OneDay, 0, sorted_discharges)};
    }

   private:
    // --- ValidateSegmentOrdering (C#:771): enforce h1 < h2 < h3 over the present segments. ---
    bool validate_segment_ordering(const std::vector<double>& parameters) const {
        if (number_of_segments_ == 1) return true;

        int expected_length = 3 * number_of_segments_ + 1;
        if (static_cast<int>(parameters.size()) < expected_length) return false;

        double h1 = parameters[0];
        double h2 = parameters[3];
        if (number_of_segments_ == 2) return h1 < h2;
        if (number_of_segments_ == 3) {
            double h3 = parameters[6];
            return h1 < h2 && h2 < h3;
        }
        return true;
    }

    // Max value skipping NaN (C# TimeSeries.MaxValue; the ported adapter omits it, so compute it
    // locally here -- SetDefaultParameters is the only consumer).
    static double max_value(const TimeSeries& series) {
        double max = std::numeric_limits<double>::lowest();
        for (int i = 0; i < series.count(); ++i) {
            double v = series[i].value();
            if (!std::isnan(v) && v > max) max = v;
        }
        return max;
    }

    // Approximates C#'s $"Stage={stage:F2}" (fixed 2-decimal; not oracle-checked).
    static std::string format_f2(double value) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.2f", value);
        return std::string(buf);
    }

    std::optional<TimeSeries> stage_data_;
    std::optional<TimeSeries> discharge_data_;
    int number_of_segments_ = 1;
    bool use_jeffreys_rule_for_scale_ = true;

    // Lazily-built, invalidated on series reassignment (C#:160). Mutable so const likelihood
    // methods can populate it.
    mutable std::optional<std::vector<AlignedObservation>> aligned_observations_cache_;
};

}  // namespace bestfit::models
