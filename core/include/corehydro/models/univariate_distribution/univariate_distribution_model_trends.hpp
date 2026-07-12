// ported from: RMC-BestFit/src/RMC.BestFit/Models/UnivariateDistribution/UnivariateDistribution.cs
// @ fc28c0c (the trend-driven surfaces: SetDefaultParameters line 571, SetTrendModel line
// 794, NonstationaryData_LogLikelihood line 1256, NonstationaryPointwiseLogLikelihood-
// Components line 1509, NonstationaryPointwiseLogLikelihood line 1716)
//
// Out-of-line definitions of UnivariateDistributionModel's trend-driven members, split
// from univariate_distribution_model.hpp purely for file size (the data_frame_plotting.hpp
// precedent); this header is included by the model header after the class definition and
// must not be included directly.
//
// Notes shared by the bodies here:
//   - The C# duplicates the constraint-block math between SetDefaultParameters and
//     SetTrendModel; both copies are ported as-is (structural mirroring beats DRY for
//     upstream diffability, the repo-wide porting convention).
//   - The C# `TrendModels[i] is LinearTrend` type tests map to switches on
//     ITrendModel::type() (each concrete trend has a unique TrendModelType).
//   - The C# computes xmin/xmax from the four series but overrides the result with
//     uppers[i] (its own comment, UnivariateDistribution.cs 648-652 / 904-908); the dead
//     computation is not ported.
//   - The C# wraps the constraint block in try/catch and rethrows
//     InvalidOperationException; here any std::exception is rethrown as
//     std::runtime_error with the same message (file-header exception mapping).
//   - The nonstationary likelihoods dispatch each full-time-series ordinate through the
//     C# `data is ExactData` pattern chain, mapped to dynamic_cast (the data types all
//     derive from Data directly).
//   - Parameter-validity checks probe on a throwaway clone (see the model file header).
#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// (Included by univariate_distribution_model.hpp; all types it needs are declared there.)

namespace corehydro::models {

// C# `SetDefaultParameters` (line 571), both paths.
inline void UnivariateDistributionModel::set_default_parameters() {
    if (distribution_ == nullptr) return;

    // (Handler removal for Parameters and the TrendModels is INPC plumbing, skipped.)

    try {
        // Get default parameter values. C# guard: DataFrame != null &&
        // DataFrame.Validate().IsValid && ExactSeries != null && ExactSeries.Count > 0.
        if (has_data_frame() && data_frame().validate().is_valid &&
            data_frame().exact_series().count() > 0) {
            auto* ml_estimator =
                dynamic_cast<numerics::distributions::IMaximumLikelihoodEstimation*>(
                    distribution_.get());
            if (ml_estimator == nullptr) {
                // C# would fail the (IMaximumLikelihoodEstimation) cast, wrapped by the
                // catch into InvalidOperationException.
                throw std::runtime_error(
                    "UnivariateDistributionModel: distribution does not implement "
                    "IMaximumLikelihoodEstimation (GetParameterConstraints unavailable)");
            }

            std::vector<double> initials;
            std::vector<double> lowers;
            std::vector<double> uppers;
            ml_estimator->get_parameter_constraints(
                data_frame().exact_series().values_to_list(), initials, lowers, uppers);

            // Make sure the full time series exists (C# 609). The C++ lazy getter below
            // also rebuilds when stale; this explicit call mirrors the C# guard.
            if (is_nonstationary_ &&
                static_cast<int>(data_frame().full_time_series().size()) !=
                    data_frame().total_record_length()) {
                data_frame().create_full_time_series();
            }

            // If no time series, use default constraints (C# 613-616).
            const std::vector<std::unique_ptr<Data>>& fts = data_frame().full_time_series();
            double tmin = fts.size() > 2 ? fts.front()->index() : 1;
            double tmax = fts.size() > 2 ? fts.back()->index() : 2;
            double trange = tmax - tmin;
            double tmid = (tmin + tmax) / 2.0;

            for (int i = 0; i < distribution_->number_of_parameters(); ++i) {
                ITrendModel& trend = *trend_models_[static_cast<std::size_t>(i)];
                trend.set_default_parameters();
                trend.set_start_index(static_cast<int>(tmin));

                // Set location parameters (C# 625-629): IsPositive flags scale-type
                // parameters positive off the constraint lower bound (C# line 628).
                ModelParameter& p0 = trend.parameters()[0];
                p0.set_value(initials[static_cast<std::size_t>(i)]);
                p0.set_lower_bound(lowers[static_cast<std::size_t>(i)]);
                p0.set_upper_bound(uppers[static_cast<std::size_t>(i)]);
                p0.set_is_positive(lowers[static_cast<std::size_t>(i)] ==
                                   numerics::kDoubleMachineEpsilon);
                p0.set_prior_distribution(std::make_unique<numerics::distributions::Uniform>(
                    lowers[static_cast<std::size_t>(i)], uppers[static_cast<std::size_t>(i)]));

                // (The C# xmin/xmax computation here is dead -- see the header note.)
                double xrange = uppers[static_cast<std::size_t>(i)];

                double beta = 5.0 / trange;
                // Guard against zero/very small beta (C# 656-658).
                double abs_beta = std::abs(beta);
                double scale = abs_beta > 1e-15
                                   ? std::pow(10.0, std::floor(std::log10(abs_beta)))
                                   : 1e-15;
                beta = std::ceil(beta / scale) * scale;

                double tdelta1 = xrange / trange;
                double tdelta2 = xrange / std::pow(trange / 2.0, 2);
                double tdelta3 = xrange / std::pow(trange / 2.0, 3);
                // Guard against zero/very small tdelta values (C# 664-666).
                tdelta1 = std::abs(tdelta1) > 1e-15
                              ? std::pow(10.0, std::floor(std::log10(std::abs(tdelta1)) + 1))
                              : 1e-15;
                tdelta2 = std::abs(tdelta2) > 1e-15
                              ? std::pow(10.0, std::floor(std::log10(std::abs(tdelta2)) + 1))
                              : 1e-15;
                tdelta3 = std::abs(tdelta3) > 1e-15
                              ? std::pow(10.0, std::floor(std::log10(std::abs(tdelta3)) + 1))
                              : 1e-15;

                std::vector<ModelParameter>& tp = trend.parameters();
                switch (trend.type()) {
                    // Linear Trend (C# 669-675).
                    case TrendModelType::Linear:
                        tp[1].set_value(0.0);
                        tp[1].set_lower_bound(-tdelta1);
                        tp[1].set_upper_bound(tdelta1);
                        tp[1].set_prior_distribution(
                            std::make_unique<numerics::distributions::Uniform>(-tdelta1,
                                                                               tdelta1));
                        break;
                    // Quadratic Trend (C# 677-688).
                    case TrendModelType::Quadratic:
                        tp[1].set_value(0.0);
                        tp[1].set_lower_bound(-tdelta1);
                        tp[1].set_upper_bound(tdelta1);
                        tp[1].set_prior_distribution(
                            std::make_unique<numerics::distributions::Uniform>(-tdelta1,
                                                                               tdelta1));
                        tp[2].set_value(0.0);
                        tp[2].set_lower_bound(-tdelta2);
                        tp[2].set_upper_bound(tdelta2);
                        tp[2].set_prior_distribution(
                            std::make_unique<numerics::distributions::Uniform>(-tdelta2,
                                                                               tdelta2));
                        break;
                    // Cubic Trend (C# 690-706).
                    case TrendModelType::Cubic:
                        tp[1].set_value(0.0);
                        tp[1].set_lower_bound(-tdelta1);
                        tp[1].set_upper_bound(tdelta1);
                        tp[1].set_prior_distribution(
                            std::make_unique<numerics::distributions::Uniform>(-tdelta1,
                                                                               tdelta1));
                        tp[2].set_value(0.0);
                        tp[2].set_lower_bound(-tdelta2);
                        tp[2].set_upper_bound(tdelta2);
                        tp[2].set_prior_distribution(
                            std::make_unique<numerics::distributions::Uniform>(-tdelta2,
                                                                               tdelta2));
                        tp[3].set_value(0.0);
                        tp[3].set_lower_bound(-tdelta3);
                        tp[3].set_upper_bound(tdelta3);
                        tp[3].set_prior_distribution(
                            std::make_unique<numerics::distributions::Uniform>(-tdelta3,
                                                                               tdelta3));
                        break;
                    // Exponential and Logistic Trend (C# 708-714).
                    case TrendModelType::Exponential:
                    case TrendModelType::Logistic:
                        tp[1].set_value(0.0);
                        tp[1].set_lower_bound(-beta);
                        tp[1].set_upper_bound(beta);
                        tp[1].set_prior_distribution(
                            std::make_unique<numerics::distributions::Uniform>(-beta, beta));
                        break;
                    // Sinusoidal (C# 716-721; the C# sets bounds but not Value here).
                    case TrendModelType::Sinusoidal:
                        tp[1].set_lower_bound(0.0);
                        tp[1].set_upper_bound(uppers[static_cast<std::size_t>(i)] / 2.0);
                        tp[1].set_prior_distribution(
                            std::make_unique<numerics::distributions::Uniform>(
                                0.0, uppers[static_cast<std::size_t>(i)] / 2.0));
                        break;
                    // Step Function (C# 723-736).
                    case TrendModelType::StepFunction:
                        tp[1].set_value(initials[static_cast<std::size_t>(i)]);
                        tp[1].set_lower_bound(lowers[static_cast<std::size_t>(i)]);
                        tp[1].set_upper_bound(uppers[static_cast<std::size_t>(i)]);
                        // >>> C# line 728 (the M8 review ledger item): the step LEVEL is
                        // flagged positive off the constraint lower bound, like the
                        // location parameter above.
                        tp[1].set_is_positive(lowers[static_cast<std::size_t>(i)] ==
                                              numerics::kDoubleMachineEpsilon);
                        tp[1].set_prior_distribution(
                            std::make_unique<numerics::distributions::Uniform>(
                                lowers[static_cast<std::size_t>(i)],
                                uppers[static_cast<std::size_t>(i)]));
                        // Change point.
                        tp[2].set_value(tmid);
                        tp[2].set_lower_bound(tmin);
                        tp[2].set_upper_bound(tmax);
                        tp[2].set_prior_distribution(
                            std::make_unique<numerics::distributions::Uniform>(tmin, tmax));
                        break;
                    default:
                        break;  // Constant / Power / Reciprocal: no extra defaults in C#.
                }
            }

            // Retained Phase 4 deviation (see the model file header): the C# leaves the
            // distribution at its previous parameters here.
            distribution_->set_parameters(initials);
        }
    } catch (const std::exception&) {
        throw std::runtime_error(
            "Failed to set default parameters for the univariate distribution.");
    }

    // OwnerName wiring from Distribution.ParameterNames (C# 747-749; X1). Each trend's
    // set_owner_name propagates the name onto that trend's ModelParameters before the
    // parameters_ list is rebuilt below (so PriorInfluenceDiagnostics keys distinctly).
    {
        std::vector<std::string> parameter_names = distribution_->parameter_names();
        for (std::size_t i = 0;
             i < static_cast<std::size_t>(distribution_->number_of_parameters()) &&
             i < parameter_names.size();
             ++i) {
            trend_models_[i]->set_owner_name(parameter_names[i]);
        }
    }

    // C# 751-753: _parameters = the concatenation of all trend parameters. The C# AddRange
    // ALIASES the trend's ModelParameter objects; the C++ value type COPIES (see the model
    // file header on the aliasing divergence).
    parameters_.clear();
    for (std::size_t i = 0; i < static_cast<std::size_t>(distribution_->number_of_parameters());
         ++i) {
        for (const ModelParameter& mp : trend_models_[i]->parameters()) parameters_.push_back(mp);
    }

    // If not nonstationary, remove the name string (C# 756-760). The C# clears through the
    // alias, so the trend's stored parameter is cleared too; stationary trends are all
    // ConstantTrend (one parameter each), so _parameters[i] == TrendModels[i].Parameters[0].
    if (!is_nonstationary_) {
        for (std::size_t i = 0;
             i < static_cast<std::size_t>(distribution_->number_of_parameters()); ++i) {
            parameters_[i].set_name("");
            trend_models_[i]->parameters()[0].set_name("");
        }
    }

    // (Handler re-adds, C# 762-780: INPC plumbing, skipped.)
}

// C# `SetTrendModel(int index, TrendModelType type)` (line 794).
inline void UnivariateDistributionModel::set_trend_model(int index, TrendModelType type) {
    if (distribution_ == nullptr)
        throw std::runtime_error("Distribution must be set before setting a trend model.");

    if (index < 0 || index >= distribution_->number_of_parameters())
        throw std::out_of_range("The index is out of range.");  // C# ArgumentOutOfRange

    // (Handler removal for Parameters / the replaced trend: INPC plumbing, skipped.)

    std::unique_ptr<ITrendModel> model = make_trend_model(type);

    try {
        // Get default parameter values (same guard as SetDefaultParameters).
        if (has_data_frame() && data_frame().validate().is_valid &&
            data_frame().exact_series().count() > 0) {
            auto* ml_estimator =
                dynamic_cast<numerics::distributions::IMaximumLikelihoodEstimation*>(
                    distribution_.get());
            if (ml_estimator == nullptr) {
                throw std::runtime_error(
                    "UnivariateDistributionModel: distribution does not implement "
                    "IMaximumLikelihoodEstimation (GetParameterConstraints unavailable)");
            }

            std::vector<double> initials;
            std::vector<double> lowers;
            std::vector<double> uppers;
            ml_estimator->get_parameter_constraints(
                data_frame().exact_series().values_to_list(), initials, lowers, uppers);

            // Make sure the full time series exists (C# 868).
            if (is_nonstationary_ &&
                static_cast<int>(data_frame().full_time_series().size()) !=
                    data_frame().total_record_length()) {
                data_frame().create_full_time_series();
            }

            // Get time constraints (C# 872-875).
            const std::vector<std::unique_ptr<Data>>& fts = data_frame().full_time_series();
            double tmin = fts.size() > 2 ? fts.front()->index() : 1;
            double tmax = fts.size() > 2 ? fts.back()->index() : 2;
            double trange = tmax - tmin;
            double tmid = (tmin + tmax) / 2.0;

            model->set_default_parameters();
            model->set_start_index(static_cast<int>(tmin));

            // Set location parameters (C# 881-885).
            std::vector<ModelParameter>& mp = model->parameters();
            mp[0].set_value(initials[static_cast<std::size_t>(index)]);
            mp[0].set_lower_bound(lowers[static_cast<std::size_t>(index)]);
            mp[0].set_upper_bound(uppers[static_cast<std::size_t>(index)]);
            mp[0].set_is_positive(lowers[static_cast<std::size_t>(index)] ==
                                  numerics::kDoubleMachineEpsilon);
            mp[0].set_prior_distribution(std::make_unique<numerics::distributions::Uniform>(
                lowers[static_cast<std::size_t>(index)],
                uppers[static_cast<std::size_t>(index)]));

            // (The C# xmin/xmax computation here is dead -- see the header note.)
            double xrange = uppers[static_cast<std::size_t>(index)];

            double beta = 5.0 / trange;
            // Guard against zero/very small beta (C# 911-914).
            double abs_beta = std::abs(beta);
            double scale =
                abs_beta > 1e-15 ? std::pow(10.0, std::floor(std::log10(abs_beta))) : 1e-15;
            beta = std::ceil(beta / scale) * scale;

            double tdelta1 = xrange / trange;
            double tdelta2 = xrange / std::pow(trange / 2.0, 2);
            double tdelta3 = xrange / std::pow(trange / 2.0, 3);
            // Guard against zero/very small tdelta values (C# 920-922).
            tdelta1 = std::abs(tdelta1) > 1e-15
                          ? std::pow(10.0, std::floor(std::log10(std::abs(tdelta1)) + 1))
                          : 1e-15;
            tdelta2 = std::abs(tdelta2) > 1e-15
                          ? std::pow(10.0, std::floor(std::log10(std::abs(tdelta2)) + 1))
                          : 1e-15;
            tdelta3 = std::abs(tdelta3) > 1e-15
                          ? std::pow(10.0, std::floor(std::log10(std::abs(tdelta3)) + 1))
                          : 1e-15;

            switch (model->type()) {
                // Linear Trend (C# 926-932).
                case TrendModelType::Linear:
                    mp[1].set_value(0.0);
                    mp[1].set_lower_bound(-tdelta1);
                    mp[1].set_upper_bound(tdelta1);
                    mp[1].set_prior_distribution(
                        std::make_unique<numerics::distributions::Uniform>(-tdelta1, tdelta1));
                    break;
                // Quadratic Trend (C# 934-945).
                case TrendModelType::Quadratic:
                    mp[1].set_value(0.0);
                    mp[1].set_lower_bound(-tdelta1);
                    mp[1].set_upper_bound(tdelta1);
                    mp[1].set_prior_distribution(
                        std::make_unique<numerics::distributions::Uniform>(-tdelta1, tdelta1));
                    mp[2].set_value(0.0);
                    mp[2].set_lower_bound(-tdelta2);
                    mp[2].set_upper_bound(tdelta2);
                    mp[2].set_prior_distribution(
                        std::make_unique<numerics::distributions::Uniform>(-tdelta2, tdelta2));
                    break;
                // Cubic Trend (C# 947-963).
                case TrendModelType::Cubic:
                    mp[1].set_value(0.0);
                    mp[1].set_lower_bound(-tdelta1);
                    mp[1].set_upper_bound(tdelta1);
                    mp[1].set_prior_distribution(
                        std::make_unique<numerics::distributions::Uniform>(-tdelta1, tdelta1));
                    mp[2].set_value(0.0);
                    mp[2].set_lower_bound(-tdelta2);
                    mp[2].set_upper_bound(tdelta2);
                    mp[2].set_prior_distribution(
                        std::make_unique<numerics::distributions::Uniform>(-tdelta2, tdelta2));
                    mp[3].set_value(0.0);
                    mp[3].set_lower_bound(-tdelta3);
                    mp[3].set_upper_bound(tdelta3);
                    mp[3].set_prior_distribution(
                        std::make_unique<numerics::distributions::Uniform>(-tdelta3, tdelta3));
                    break;
                // Exponential and Logistic Trend (C# 965-971).
                case TrendModelType::Exponential:
                case TrendModelType::Logistic:
                    mp[1].set_value(0.0);
                    mp[1].set_lower_bound(-beta);
                    mp[1].set_upper_bound(beta);
                    mp[1].set_prior_distribution(
                        std::make_unique<numerics::distributions::Uniform>(-beta, beta));
                    break;
                // Sinusoidal (C# 973-978; unlike SetDefaultParameters, the C# also sets
                // Value = the bound midpoint here).
                case TrendModelType::Sinusoidal:
                    mp[1].set_lower_bound(0.0);
                    mp[1].set_upper_bound(uppers[static_cast<std::size_t>(index)] / 2.0);
                    mp[1].set_value(0.5 * (mp[1].lower_bound() + mp[1].upper_bound()));
                    mp[1].set_prior_distribution(
                        std::make_unique<numerics::distributions::Uniform>(
                            0.0, uppers[static_cast<std::size_t>(index)] / 2.0));
                    break;
                // Step Function (C# 981-994).
                case TrendModelType::StepFunction:
                    mp[1].set_value(initials[static_cast<std::size_t>(index)]);
                    mp[1].set_lower_bound(lowers[static_cast<std::size_t>(index)]);
                    mp[1].set_upper_bound(uppers[static_cast<std::size_t>(index)]);
                    // >>> C# line 986 (the M8 review ledger item): IsPositive off the
                    // constraint lower bound on the step level.
                    mp[1].set_is_positive(lowers[static_cast<std::size_t>(index)] ==
                                          numerics::kDoubleMachineEpsilon);
                    mp[1].set_prior_distribution(
                        std::make_unique<numerics::distributions::Uniform>(
                            lowers[static_cast<std::size_t>(index)],
                            uppers[static_cast<std::size_t>(index)]));
                    // Change point.
                    mp[2].set_value(tmid);
                    mp[2].set_lower_bound(tmin);
                    mp[2].set_upper_bound(tmax);
                    mp[2].set_prior_distribution(
                        std::make_unique<numerics::distributions::Uniform>(tmin, tmax));
                    break;
                default:
                    break;  // Constant / Power / Reciprocal: no extra defaults in C#.
            }
        }
    } catch (const std::exception&) {
        throw std::runtime_error("Failed to set the trend model defaults.");
    }

    trend_models_[static_cast<std::size_t>(index)] = std::move(model);
    // OwnerName wiring from ParameterNames (C# 1004; X1): name the replaced trend, propagating
    // onto its ModelParameters.
    {
        std::vector<std::string> parameter_names = distribution_->parameter_names();
        if (static_cast<std::size_t>(index) < parameter_names.size())
            trend_models_[static_cast<std::size_t>(index)]->set_owner_name(
                parameter_names[static_cast<std::size_t>(index)]);
    }

    // Reset the Parameters list (C# 1007-1011; copies, see the aliasing note).
    parameters_.clear();
    for (std::size_t i = 0; i < static_cast<std::size_t>(distribution_->number_of_parameters());
         ++i) {
        for (const ModelParameter& mp2 : trend_models_[i]->parameters())
            parameters_.push_back(mp2);
    }

    // (Handler re-adds, C# 1014-1018: INPC plumbing, skipped.)
}

// C# `NonstationaryData_LogLikelihood(model, parameters)` (line 1256).
inline double UnivariateDistributionModel::nonstationary_data_log_likelihood(
    DistributionBase& model, const std::vector<double>& p) const {
    if (!has_data_frame())
        throw std::runtime_error("DataFrame must be set before computing log likelihood.");

    // The C# caches the FullTimeSeries reference once per call (the hot-path fix); the C++
    // lazy const getter is likewise evaluated once here.
    const std::vector<std::unique_ptr<Data>>& full_time_series = data_frame().full_time_series();
    if (full_time_series.empty()) {
        throw std::runtime_error(
            "FullTimeSeries is not populated for nonstationary log likelihood.");
    }
    std::size_t series_count = full_time_series.size();
    double low_outlier_threshold = data_frame().low_outlier_threshold();

    double log_lh = 0.0;

    // Build trend models with proposed parameters (C# 1276-1289).
    std::size_t t = 0;
    std::vector<std::unique_ptr<ITrendModel>> trend_models_copy;
    trend_models_copy.reserve(trend_models_.size());
    for (std::size_t i = 0; i < trend_models_.size(); ++i) {
        std::size_t n = static_cast<std::size_t>(trend_models_[i]->number_of_parameters());
        std::vector<double> parms(p.begin() + static_cast<std::ptrdiff_t>(t),
                                  p.begin() + static_cast<std::ptrdiff_t>(t + n));
        std::unique_ptr<ITrendModel> clone = trend_models_[i]->clone();
        clone->set_parameter_values(parms);
        trend_models_copy.push_back(std::move(clone));
        t += n;
    }

    // *** Compute the full log-likelihood *** (C# 1291-1355; one reused values buffer).
    std::size_t num_parameters = static_cast<std::size_t>(model.number_of_parameters());
    std::vector<double> values(num_parameters);
    for (std::size_t i = 0; i < series_count; ++i) {
        // Get data at time-step i.
        const Data& data = *full_time_series[i];

        // Get parameters.
        for (std::size_t j = 0; j < num_parameters; ++j)
            values[j] = trend_models_copy[j]->predict(data.index());

        // Check if parameters are valid (probe clone; C# ValidateParameters-before-set).
        std::unique_ptr<DistributionBase> probe = model.clone();
        probe->set_parameters(values);
        if (!probe->parameters_valid()) return -std::numeric_limits<double>::infinity();

        // Set parameters.
        model.set_parameters(values);

        // exact data
        if (const auto* exact = dynamic_cast<const ExactData*>(&data)) {
            if (!exact->is_low_outlier()) {
                log_lh += model.log_likelihood(exact->value());
            } else {
                log_lh += model.log_likelihood_left_censored(low_outlier_threshold, 1);
            }
        }
        // uncertain Data
        else if (const auto* uncertain = dynamic_cast<const UncertainData*>(&data)) {
            const DistributionBase& dist = uncertain->distribution();
            double lower_probability = 1E-8;
            double upper_probability = 1.0 - 1E-8;
            double a = dist.inverse_cdf(lower_probability);
            double b = dist.inverse_cdf(upper_probability);
            double mass = upper_probability - lower_probability;
            if (numerics::is_finite(a) && numerics::is_finite(b) && numerics::is_finite(mass) &&
                mass > 0.0 && a < b) {
                double ep = numerics::math::integration::Integration::gauss_legendre20(
                                [&](double q) { return dist.pdf(q) * model.pdf(q); }, a, b) /
                            mass;
                log_lh += ep > 0 ? std::log(ep) : -std::numeric_limits<double>::infinity();
            } else {
                log_lh += -std::numeric_limits<double>::infinity();
            }
        }
        // interval data
        else if (const auto* interval = dynamic_cast<const IntervalData*>(&data)) {
            log_lh += model.log_likelihood_intervals(interval->lower_value(),
                                                     interval->upper_value());
        }
        // threshold data
        else if (const auto* threshold = dynamic_cast<const ThresholdData*>(&data)) {
            if (threshold->number_below() > 0)
                log_lh += model.log_likelihood_left_censored(threshold->value(),
                                                             threshold->number_below());
            if (threshold->number_above() > 0)
                log_lh += model.log_likelihood_right_censored(threshold->value(),
                                                              threshold->number_above());
        }
    }

    return log_lh;
}

// C# `NonstationaryPointwiseLogLikelihood` (line 1716).
inline std::vector<double> UnivariateDistributionModel::nonstationary_pointwise_log_likelihood(
    DistributionBase& model, const std::vector<double>& p) const {
    if (!has_data_frame())
        throw std::runtime_error(
            "DataFrame must be set before computing pointwise log likelihood.");

    // See nonstationary_data_log_likelihood for the FullTimeSeries caching rationale.
    const std::vector<std::unique_ptr<Data>>& full_time_series = data_frame().full_time_series();
    if (full_time_series.empty()) {
        throw std::runtime_error(
            "FullTimeSeries is not populated for nonstationary log likelihood.");
    }
    std::size_t series_count = full_time_series.size();
    double low_outlier_threshold = data_frame().low_outlier_threshold();
    const double neg_inf = -std::numeric_limits<double>::infinity();

    // Build trend models with proposed parameters.
    std::size_t t = 0;
    std::vector<std::unique_ptr<ITrendModel>> trend_models_copy;
    trend_models_copy.reserve(trend_models_.size());
    for (std::size_t i = 0; i < trend_models_.size(); ++i) {
        std::size_t n = static_cast<std::size_t>(trend_models_[i]->number_of_parameters());
        std::vector<double> parms(p.begin() + static_cast<std::ptrdiff_t>(t),
                                  p.begin() + static_cast<std::ptrdiff_t>(t + n));
        std::unique_ptr<ITrendModel> clone = trend_models_[i]->clone();
        clone->set_parameter_values(parms);
        trend_models_copy.push_back(std::move(clone));
        t += n;
    }

    std::vector<double> result(series_count);

    // Compute pointwise log-likelihood for each time step.
    std::size_t num_parameters = static_cast<std::size_t>(model.number_of_parameters());
    std::vector<double> values(num_parameters);
    for (std::size_t i = 0; i < series_count; ++i) {
        const Data& data = *full_time_series[i];

        // Get parameters at this time step.
        for (std::size_t j = 0; j < num_parameters; ++j)
            values[j] = trend_models_copy[j]->predict(data.index());

        // Check if parameters are valid (this variant marks only the ordinate invalid and
        // continues, unlike the total's early return -- exactly the C# difference).
        std::unique_ptr<DistributionBase> probe = model.clone();
        probe->set_parameters(values);
        if (!probe->parameters_valid()) {
            result[i] = neg_inf;
            continue;
        }

        // Set parameters.
        model.set_parameters(values);

        // Compute log-likelihood based on data type.
        if (const auto* exact = dynamic_cast<const ExactData*>(&data)) {
            if (!exact->is_low_outlier()) {
                result[i] = model.log_likelihood(exact->value());
            } else {
                result[i] = model.log_likelihood_left_censored(low_outlier_threshold, 1);
            }
        } else if (const auto* uncertain = dynamic_cast<const UncertainData*>(&data)) {
            const DistributionBase& dist = uncertain->distribution();
            double lower_probability = 1E-8;
            double upper_probability = 1.0 - 1E-8;
            double a = dist.inverse_cdf(lower_probability);
            double b = dist.inverse_cdf(upper_probability);
            double mass = upper_probability - lower_probability;
            if (numerics::is_finite(a) && numerics::is_finite(b) && numerics::is_finite(mass) &&
                mass > 0.0 && a < b) {
                double ep = numerics::math::integration::Integration::gauss_legendre20(
                                [&](double q) { return dist.pdf(q) * model.pdf(q); }, a, b) /
                            mass;
                result[i] = ep > 0 ? std::log(ep) : neg_inf;
            } else {
                result[i] = neg_inf;
            }
        } else if (const auto* interval = dynamic_cast<const IntervalData*>(&data)) {
            result[i] = model.log_likelihood_intervals(interval->lower_value(),
                                                       interval->upper_value());
        } else if (const auto* threshold = dynamic_cast<const ThresholdData*>(&data)) {
            double log_lh = 0.0;
            if (threshold->number_below() > 0)
                log_lh += model.log_likelihood_left_censored(threshold->value(),
                                                             threshold->number_below());
            if (threshold->number_above() > 0)
                log_lh += model.log_likelihood_right_censored(threshold->value(),
                                                              threshold->number_above());
            result[i] = log_lh;
        }
    }

    return result;
}

// C# `NonstationaryPointwiseLogLikelihoodComponents` (line 1509).
inline std::vector<DataComponent>
UnivariateDistributionModel::nonstationary_pointwise_log_likelihood_components(
    DistributionBase& model, const std::vector<double>& p) const {
    if (!has_data_frame())
        throw std::runtime_error(
            "DataFrame must be set before computing pointwise log likelihood components.");

    // See nonstationary_data_log_likelihood for the FullTimeSeries caching rationale.
    const std::vector<std::unique_ptr<Data>>& full_time_series = data_frame().full_time_series();
    if (full_time_series.empty()) {
        throw std::runtime_error(
            "FullTimeSeries is not populated for nonstationary log likelihood.");
    }
    std::size_t series_count = full_time_series.size();
    double low_outlier_threshold = data_frame().low_outlier_threshold();
    const double neg_inf = -std::numeric_limits<double>::infinity();

    // Build trend models with proposed parameters.
    std::size_t t = 0;
    std::vector<std::unique_ptr<ITrendModel>> trend_models_copy;
    trend_models_copy.reserve(trend_models_.size());
    for (std::size_t i = 0; i < trend_models_.size(); ++i) {
        std::size_t n = static_cast<std::size_t>(trend_models_[i]->number_of_parameters());
        std::vector<double> parms(p.begin() + static_cast<std::ptrdiff_t>(t),
                                  p.begin() + static_cast<std::ptrdiff_t>(t + n));
        std::unique_ptr<ITrendModel> clone = trend_models_[i]->clone();
        clone->set_parameter_values(parms);
        trend_models_copy.push_back(std::move(clone));
        t += n;
    }

    std::vector<DataComponent> result;
    result.reserve(series_count);

    // Compute pointwise log-likelihood components for each time step.
    std::size_t num_parameters = static_cast<std::size_t>(model.number_of_parameters());
    std::vector<double> values(num_parameters);
    for (std::size_t i = 0; i < series_count; ++i) {
        const Data& data = *full_time_series[i];
        int idx = static_cast<int>(i);

        // Get parameters at this time step.
        for (std::size_t j = 0; j < num_parameters; ++j)
            values[j] = trend_models_copy[j]->predict(data.index());

        // Check if parameters are valid.
        std::unique_ptr<DistributionBase> probe = model.clone();
        probe->set_parameters(values);
        if (!probe->parameters_valid()) {
            // Add invalid component based on data type (C# 1554-1563).
            if (const auto* exact = dynamic_cast<const ExactData*>(&data)) {
                result.emplace_back(idx, neg_inf, exact->value(), DataComponentType::Exact, 1,
                                    std::to_string(data.index()));
            } else if (const auto* uncertain = dynamic_cast<const UncertainData*>(&data)) {
                result.emplace_back(idx, neg_inf, uncertain->distribution().mean(),
                                    DataComponentType::Uncertain, 1,
                                    std::to_string(data.index()));
            } else if (const auto* interval = dynamic_cast<const IntervalData*>(&data)) {
                result.emplace_back(idx, neg_inf,
                                    (interval->lower_value() + interval->upper_value()) / 2.0,
                                    DataComponentType::Interval, 1,
                                    std::to_string(data.index()));
            } else if (const auto* threshold = dynamic_cast<const ThresholdData*>(&data)) {
                result.emplace_back(idx, neg_inf, threshold->value(),
                                    DataComponentType::LeftCensored,
                                    threshold->number_below() + threshold->number_above(),
                                    "Threshold");
            }
            continue;
        }

        // Set parameters.
        model.set_parameters(values);

        // Compute log-likelihood based on data type.
        if (const auto* exact = dynamic_cast<const ExactData*>(&data)) {
            double log_lh;
            double value = exact->value();

            if (!exact->is_low_outlier()) {
                log_lh = model.log_likelihood(exact->value());
            } else {
                log_lh = model.log_likelihood_left_censored(low_outlier_threshold, 1);
                value = low_outlier_threshold;
            }

            result.emplace_back(idx, log_lh, value, DataComponentType::Exact, 1,
                                std::to_string(data.index()));
        } else if (const auto* uncertain = dynamic_cast<const UncertainData*>(&data)) {
            const DistributionBase& dist = uncertain->distribution();
            double lower_probability = 1E-8;
            double upper_probability = 1.0 - 1E-8;
            double a = dist.inverse_cdf(lower_probability);
            double b = dist.inverse_cdf(upper_probability);
            double mass = upper_probability - lower_probability;
            double log_lh = neg_inf;
            if (numerics::is_finite(a) && numerics::is_finite(b) && numerics::is_finite(mass) &&
                mass > 0.0 && a < b) {
                double ep = numerics::math::integration::Integration::gauss_legendre20(
                                [&](double q) { return dist.pdf(q) * model.pdf(q); }, a, b) /
                            mass;
                log_lh = ep > 0 ? std::log(ep) : neg_inf;
            }
            result.emplace_back(idx, log_lh, dist.mean(), DataComponentType::Uncertain, 1,
                                std::to_string(data.index()));
        } else if (const auto* interval = dynamic_cast<const IntervalData*>(&data)) {
            double log_lh = model.log_likelihood_intervals(interval->lower_value(),
                                                           interval->upper_value());
            double midpoint = (interval->lower_value() + interval->upper_value()) / 2.0;

            result.emplace_back(idx, log_lh, midpoint, DataComponentType::Interval, 1,
                                std::to_string(data.index()));
        } else if (const auto* threshold = dynamic_cast<const ThresholdData*>(&data)) {
            double log_lh = 0.0;
            if (threshold->number_below() > 0)
                log_lh += model.log_likelihood_left_censored(threshold->value(),
                                                             threshold->number_below());
            if (threshold->number_above() > 0)
                log_lh += model.log_likelihood_right_censored(threshold->value(),
                                                              threshold->number_above());

            int total_count = threshold->number_below() + threshold->number_above();
            result.emplace_back(idx, log_lh, threshold->value(),
                                DataComponentType::LeftCensored, total_count, "Threshold");
        }
    }

    return result;
}

}  // namespace corehydro::models
