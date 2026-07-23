// ported from: RMC.BestFit/Models/TimeSeries/MovingAverage.cs @ c2e6192
//
// v2.0.0 (upstream-sync Task 16, f140c4d + 0d6821d): SetTrainingData's BoxCox/YeoJohnson
// branches now guard against a non-finite fitted lambda (Task 2's hardened fit_lambda already
// returns NaN instead of throwing for a degenerate/unsupported sample -- see box_cox.hpp /
// yeo_johnson.hpp). On failure: lambda_/log_jacobian_ reset to 0, training_time_series_
// replaced with an empty series, transform_fit_validation_message_ set to the C#-exact message
// text, and the branch returns early (the happy-path subset/transform loop never runs). Validate
// appends that message (and flips is_valid false) as the LAST check, mirroring the C# ordering.
// Deliberately NOT ported: the C# `catch (ArithmeticException)` branch around FitLambda (with
// its "Solver message: ..." suffix) has no C++ analog -- the ported fit_lambda never throws, it
// always returns a double (possibly NaN), so only the C# `!double.IsFinite(_lambda)` branch
// (the plain message, no solver-text suffix) is reachable here. Also SKIPPED (XML/GUI-only, no
// oracle-visible C++ surface, not in this task's scope): the XElement ctor's
// TrainingTimeSteps/UseDefaultTrainingSteps attribute restoration + explicit SetTrainingData()
// call. Task 21 CLOSED the two v2.0.0 deltas this note previously deferred:
// ResetDefaultTrainingStepsForNewTimeSeries (the TimeSeries-setter training-window reset) and the
// TimeInterval.Irregular Validate guard are both ported below -- both are model-layer library
// surface reachable from the public setters, not GUI/XML.
//
// Moving-average MA(q) time-series model:
//   Y(t) = mu + eps(t) + theta_1*eps(t-1) + ... + theta_q*eps(t-q),  eps(t) ~ N(0, sigma^2)
// on top of ModelBase and ISimulatable<std::vector<double>> (C# ISimulatable<double[]>). Supports
// None/Logarithmic/Box-Cox/Yeo-Johnson data transforms, a training/forecasting split, and the
// optional Jeffreys (1/sigma) scale prior.
//
// Structural mirroring: the class/member/method layout follows MovingAverage.cs. The property
// setters that re-run SetDefaultParameters()/SetTrainingData() on change (Order/IncludeIntercept/
// TransformType/TimeSeries/TrainingTimeSteps) preserve that effective behavior; the never-mutate
// rule is RELAXED for these stateful model objects (per .claude/CLAUDE.md), matching the upstream
// mutable WPF-binding design.
//
// KEY DIFFERENCES FROM AutoRegressive (C# governs, verified against MovingAverage.cs):
//   - Error recursion eps[t] = y[t] - (mu + sum theta_q*eps[t-q]) with CSS warm-up
//     (min(t, Order)), NOT the AR mean-centered residual recursion.
//   - DataLogLikelihood sums ALL t = 0..N (no t >= Order skip; MovingAverage.cs:551); the training
//     guard is Count == 0 (not Count <= Order).
//   - Pointwise count == TrainingTimeSteps (all observations), not TrainingTimeSteps - Order.
//   - SetDefaultParameters uses the data-driven bounds when Count > 0 (not Count > Order).
//   - SetTrainingData's transform subset starts at index 0 (AR starts at Order).
//   - IsInvertible (order 1 -> |theta|<1; else sum|theta|<1) replaces IsStationary; Validate has
//     no "length <= order" guard and its warning mentions "invertibility".
//   - GenerateRandomValues seeds the MersenneTwister on seed >= 0 (AR uses seed > 0), draws from
//     Normal(0,1) and scales by sigma (AR draws from Normal(0,sigma) directly).
//
// Deliberately NOT ported (same rationale as auto_regressive.hpp): XML, INotifyPropertyChanged,
// IModel Clone(), GenerateRandomSeries. RNG deviation (Predict): C# System.Random has no ported
// equivalent; the ported MersenneTwister is substituted (determinism/divergence hold; exact
// seeded forecast values are a P4 concern). See task-T1-report.md.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/models/support/data_component.hpp"
#include "corehydro/models/support/model_base.hpp"
#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/models/support/prior_component.hpp"
#include "corehydro/models/support/simulatable.hpp"
#include "corehydro/models/support/subscript_formatter.hpp"
#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/models/time_series/transform_type.hpp"
#include "corehydro/numerics/data/box_cox.hpp"
#include "corehydro/numerics/data/time_series/time_series.hpp"
#include "corehydro/numerics/data/yeo_johnson.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models {

class MovingAverage : public ModelBase, public ISimulatable<std::vector<double>> {
   public:
    using TimeSeries = numerics::data::TimeSeries;

    // Prediction decomposition (C# tuple (double[] Y, double[] InterceptPart, double[] MAPart)).
    struct PredictResult {
        std::vector<double> y;
        std::vector<double> intercept_part;
        std::vector<double> ma_part;
    };

    // --- Construction (MovingAverage.cs:44-62). ---

    MovingAverage() {
        order_ = 1;
        include_intercept_ = true;
        set_default_parameters();
    }

    explicit MovingAverage(const TimeSeries& time_series, int order = 1,
                           bool include_intercept = true) {
        order_ = order;
        include_intercept_ = include_intercept;
        set_time_series(time_series);
    }

    // --- Properties. ---

    bool has_time_series() const { return time_series_.has_value(); }
    const TimeSeries& time_series() const { return *time_series_; }
    void set_time_series(const TimeSeries& value) {
        // C# unsubscribes/subscribes CollectionChanged here -> no-op in this port.
        time_series_ = value;
        reset_default_training_steps_for_new_time_series();
        set_training_data();
        if (use_default_flat_priors()) set_default_parameters();
    }

    int order() const { return order_; }
    void set_order(int value) {
        if (order_ != value) {
            order_ = value;
            set_default_parameters();
        }
    }

    bool include_intercept() const { return include_intercept_; }
    void set_include_intercept(bool value) {
        if (include_intercept_ != value) {
            include_intercept_ = value;
            set_default_parameters();
        }
    }

    Transform transform_type() const { return transform_type_; }
    void set_transform_type(Transform value) {
        if (transform_type_ != value) {
            transform_type_ = value;
            set_training_data();
            set_default_parameters();
        }
    }

    bool has_training_time_series() const { return training_time_series_.has_value(); }
    const TimeSeries& training_time_series() const { return *training_time_series_; }

    int training_time_steps() const { return training_time_steps_; }
    void set_training_time_steps(int value) {
        if (training_time_steps_ != value) {
            training_time_steps_ = value;
            set_training_data();
            if (use_default_flat_priors()) set_default_parameters();
        }
    }

    bool use_default_training_steps() const { return use_default_training_steps_; }
    void set_use_default_training_steps(bool value) {
        if (use_default_training_steps_ != value) {
            use_default_training_steps_ = value;
            if (use_default_training_steps_ && time_series_) set_default_training_steps();
        }
    }

    int forecasting_time_steps() const {
        return time_series_ ? time_series_->count() - training_time_steps_ : 0;
    }

    bool use_jeffreys_rule_for_scale() const { return use_jeffreys_rule_for_scale_; }
    void set_use_jeffreys_rule_for_scale(bool value) {
        if (use_jeffreys_rule_for_scale_ != value) use_jeffreys_rule_for_scale_ = value;
    }

    void set_transform_parameters(double lambda1 = 0, double lambda2 = 0) {
        lambda_ = lambda1;
        lambda2_ = lambda2;
    }

    std::vector<double> parameter_values() const {
        std::vector<double> v;
        v.reserve(parameters_.size());
        for (const auto& p : parameters_) v.push_back(p.value());
        return v;
    }

    // --- SetDefaultParameters (C#:404). Data-driven bounds when Count > 0 (differs from AR). ---
    void set_default_parameters() override {
        parameters_.clear();

        double mean = 0, sigma = 1, min = -10, max = 10, sigma_ub = 10;

        if (training_time_series_ && training_time_series_->count() > 0) {
            mean = training_time_series_->mean_value();
            sigma = training_time_series_->standard_deviation();

            double sign = (mean > 0) - (mean < 0);
            double temp_min = sign * std::pow(10.0, std::floor(std::log10(std::fabs(mean)) - 1.0));
            double temp_max = sign * std::pow(10.0, std::ceil(std::log10(std::fabs(mean)) + 1.0));
            min = std::min(temp_min, temp_max);
            max = std::max(temp_min, temp_max);

            if (std::isnan(min) || std::isinf(min)) min = -1000;
            if (std::isnan(max) || std::isinf(max)) max = 1000;
            if (min >= max) {
                min = mean - 100;
                max = mean + 100;
            }

            sigma_ub = std::pow(10.0, std::ceil(std::log10(sigma) + 1.0));
            if (std::isnan(sigma_ub) || std::isinf(sigma_ub)) sigma_ub = 100;
        }

        if (include_intercept_) {
            parameters_.emplace_back(
                /*owner_name=*/"", /*name=*/"Intercept (\xCE\xBC)", /*value=*/mean,
                /*lower_bound=*/min, /*upper_bound=*/max,
                std::make_unique<numerics::distributions::Uniform>(min, max));
        }

        for (int i = 1; i <= order_; ++i) {
            parameters_.emplace_back(
                /*owner_name=*/"", /*name=*/std::string("MA (\xCE\xB8") + to_subscript(i) + ")",
                /*value=*/0.0, /*lower_bound=*/-2.0, /*upper_bound=*/2.0,
                std::make_unique<numerics::distributions::Uniform>(-2.0, 2.0));
        }

        parameters_.emplace_back(
            /*owner_name=*/"", /*name=*/"Scale (\xCF\x83)", /*value=*/sigma,
            /*lower_bound=*/numerics::kDoubleMachineEpsilon, /*upper_bound=*/sigma_ub,
            std::make_unique<numerics::distributions::Uniform>(numerics::kDoubleMachineEpsilon,
                                                               sigma_ub),
            /*is_positive=*/true);
    }

    // --- Residuals (C#:493): MA error recursion, CSS warm-up eps=0 (min(t, Order)). ---
    std::vector<double> residuals(const std::vector<double>& parameters) const {
        int effective = training_time_series_
                            ? std::min(training_time_steps_, training_time_series_->count())
                            : training_time_steps_;
        if (effective < 0) effective = 0;
        std::vector<double> epsilon(static_cast<std::size_t>(effective), 0.0);

        int k = 0;
        double mu = include_intercept_ ? parameters[static_cast<std::size_t>(k++)] : 0.0;
        std::vector<double> theta(static_cast<std::size_t>(order_));
        for (int i = 0; i < order_; ++i)
            theta[static_cast<std::size_t>(i)] = parameters[static_cast<std::size_t>(k++)];

        for (int t = 0; t < effective; ++t) {
            double prediction = mu;
            for (int q = 1; q <= std::min(t, order_); ++q)
                prediction += theta[static_cast<std::size_t>(q - 1)] *
                              epsilon[static_cast<std::size_t>(t - q)];
            epsilon[static_cast<std::size_t>(t)] = (*training_time_series_)[t].value() - prediction;
        }
        return epsilon;
    }

    // --- DataLogLikelihood (C#:531): CSS-conditional Gaussian over ALL t + logJacobian. ---
    double data_log_likelihood(std::vector<double>& parameters) const override {
        if (!training_time_series_ || training_time_series_->count() == 0)
            return -std::numeric_limits<double>::infinity();

        for (double v : parameters)
            if (std::isnan(v)) return -std::numeric_limits<double>::infinity();

        std::vector<double> res = residuals(parameters);
        double sigma = parameters.back();
        if (sigma <= 0) return -std::numeric_limits<double>::infinity();
        numerics::distributions::Normal norm(0.0, sigma);
        double log_lh = 0.0;
        for (int t = 0; t < static_cast<int>(res.size()); ++t)
            log_lh += norm.log_pdf(res[static_cast<std::size_t>(t)]);
        return log_lh + log_jacobian_;
    }

    // --- PointwiseDataLogLikelihood (C#:560): one entry per training observation. ---
    std::vector<double> pointwise_data_log_likelihood(
        const std::vector<double>& parameters) const override {
        if (!training_time_series_ || training_time_series_->count() == 0) return {};

        int effective = std::min(training_time_steps_, training_time_series_->count());
        int n = effective;
        if (n <= 0) return {};

        for (double v : parameters)
            if (std::isnan(v))
                return std::vector<double>(static_cast<std::size_t>(n),
                                           -std::numeric_limits<double>::infinity());

        std::vector<double> res = residuals(parameters);
        double sigma = parameters.back();
        numerics::distributions::Normal norm(0.0, sigma);
        std::vector<double> result(static_cast<std::size_t>(n));

        double jacobian_per_obs = log_jacobian_ / n;
        for (int t = 0; t < n; ++t)
            result[static_cast<std::size_t>(t)] =
                norm.log_pdf(res[static_cast<std::size_t>(t)]) + jacobian_per_obs;
        return result;
    }

    // --- PointwiseDataLogLikelihoodComponents (C#:594). ---
    std::vector<DataComponent> pointwise_data_log_likelihood_components(
        const std::vector<double>& parameters) const override {
        if (!training_time_series_ || training_time_series_->count() == 0) return {};

        int effective = std::min(training_time_steps_, training_time_series_->count());
        int n = effective;
        if (n <= 0) return {};

        std::vector<DataComponent> result;
        result.reserve(static_cast<std::size_t>(n));
        std::vector<double> response = training_time_series_->values_to_array();

        for (double v : parameters) {
            if (std::isnan(v)) {
                for (int j = 0; j < n; ++j) {
                    double value = j < static_cast<int>(response.size())
                                       ? response[static_cast<std::size_t>(j)]
                                       : 0.0;
                    result.emplace_back(j, -std::numeric_limits<double>::infinity(), value,
                                        DataComponentType::Exact, 1,
                                        std::string("t=") + std::to_string(j));
                }
                return result;
            }
        }

        std::vector<double> res = residuals(parameters);
        double sigma = parameters.back();
        numerics::distributions::Normal norm(0.0, sigma);
        double jacobian_per_obs = log_jacobian_ / n;

        for (int t = 0; t < n; ++t) {
            double log_lh = norm.log_pdf(res[static_cast<std::size_t>(t)]) + jacobian_per_obs;
            double value = t < static_cast<int>(response.size())
                               ? response[static_cast<std::size_t>(t)]
                               : 0.0;
            result.emplace_back(t, log_lh, value, DataComponentType::Exact, 1,
                                std::string("t=") + std::to_string(t));
        }
        return result;
    }

    // --- PriorLogLikelihood (C#:635): parameter priors + optional Jeffreys 1/sigma. ---
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

    // --- PointwisePriorLogLikelihood (C#:660). ---
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
            result.emplace_back("Jeffreys' rule for \xCF\x83", ll,
                                PriorComponentType::ParameterPrior);
        }
        return result;
    }

    // --- Predict (C#:688). Returns Y + component decomposition. ---
    PredictResult predict_components(const std::vector<double>& parameters, int forecast_steps = 0,
                                     int seed = -1) const {
        if (!time_series_) throw std::runtime_error("TimeSeries must be set.");

        int total_steps = training_time_steps_ + forecast_steps;
        // Deliberate deviation from C#: a negative total (an invalid negative forecast_steps)
        // returns empty component vectors here, where C#'s `new double[totalSteps]` would throw
        // on the negative allocation. Unreachable on normal paths (total_steps >= 0).
        if (total_steps < 0) total_steps = 0;

        std::vector<double> y(static_cast<std::size_t>(total_steps), 0.0);
        std::vector<double> intercept_part(static_cast<std::size_t>(total_steps), 0.0);
        std::vector<double> ma_part(static_cast<std::size_t>(total_steps), 0.0);
        std::vector<double> epsilon(static_cast<std::size_t>(total_steps), 0.0);

        int k = 0;
        double mu = include_intercept_ ? parameters[static_cast<std::size_t>(k++)] : 0.0;
        std::vector<double> theta(static_cast<std::size_t>(order_));
        for (int i = 0; i < order_; ++i)
            theta[static_cast<std::size_t>(i)] = parameters[static_cast<std::size_t>(k++)];
        double sigma = parameters[static_cast<std::size_t>(k)];

        std::optional<numerics::sampling::MersenneTwister> prng;
        std::optional<numerics::distributions::Normal> err_dist;
        if (seed >= 0) {
            prng.emplace(static_cast<std::uint32_t>(seed));
            err_dist.emplace(0.0, sigma);
        }

        for (int t = 0; t < total_steps; ++t) {
            intercept_part[static_cast<std::size_t>(t)] = mu;

            double ma = 0.0;
            for (int q = 1; q <= std::min(t, order_); ++q)
                ma += theta[static_cast<std::size_t>(q - 1)] * epsilon[static_cast<std::size_t>(t - q)];
            ma_part[static_cast<std::size_t>(t)] = ma;

            double prediction = mu + ma;
            y[static_cast<std::size_t>(t)] = prediction;

            bool in_fit_window = training_time_series_ && t < training_time_steps_ &&
                                 t < training_time_series_->count();
            if (in_fit_window)
                epsilon[static_cast<std::size_t>(t)] = (*training_time_series_)[t].value() - prediction;
            else
                epsilon[static_cast<std::size_t>(t)] = 0.0;

            if (prng) {
                double error = err_dist->inverse_cdf(prng->next_double());
                y[static_cast<std::size_t>(t)] += error;
                if (!in_fit_window) epsilon[static_cast<std::size_t>(t)] = error;
            }
        }

        if (transform_type_ == Transform::Logarithmic || transform_type_ == Transform::BoxCox) {
            for (int t = 0; t < total_steps; ++t)
                y[static_cast<std::size_t>(t)] =
                    numerics::data::BoxCox::inverse_transform(y[static_cast<std::size_t>(t)], lambda_);
        } else if (transform_type_ == Transform::YeoJohnson) {
            for (int t = 0; t < total_steps; ++t)
                y[static_cast<std::size_t>(t)] = numerics::data::YeoJohnson::inverse_transform(
                    y[static_cast<std::size_t>(t)], lambda_);
        }

        return {std::move(y), std::move(intercept_part), std::move(ma_part)};
    }

    std::vector<double> predict(int forecast_steps = 0, int seed = -1) const {
        return predict_components(parameter_values(), forecast_steps, seed).y;
    }

    // --- IsInvertible (C#:852). ---
    bool is_invertible() const {
        if (parameters_.empty()) return true;

        int k = include_intercept_ ? 1 : 0;
        if (static_cast<int>(parameters_.size()) <= k) return true;

        std::vector<double> theta(static_cast<std::size_t>(order_));
        for (int i = 0; i < order_; ++i)
            theta[static_cast<std::size_t>(i)] = parameters_[static_cast<std::size_t>(k + i)].value();

        if (order_ == 1) return std::fabs(theta[0]) < 1.0;

        double sum_abs_theta = 0.0;
        for (int i = 0; i < order_; ++i)
            sum_abs_theta += std::fabs(theta[static_cast<std::size_t>(i)]);
        return sum_abs_theta < 1.0;
    }

    // --- Validate (C#:875). No "length <= order" guard; warning mentions "invertibility". ---
    ValidationResult validate() const override {
        ValidationResult result;

        if (!time_series_) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: Time series is null.");
            return result;
        }

        if (time_series_->count() < 10) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Time series must have at least 10 observations.");
        }
        if (time_series_->time_interval() == numerics::data::TimeInterval::Irregular) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Time series analysis requires a regular time interval. Resample or "
                "convert the series to a regular interval before estimating.");
        }
        if (order_ < 1 || order_ > 10) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: MA order must be between 1 and 10.");
        }
        if (training_time_steps_ < number_of_parameters()) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Training time steps (" + std::to_string(training_time_steps_) +
                ") must be at least equal to the number of parameters (" +
                std::to_string(number_of_parameters()) + ").");
        }
        if (training_time_steps_ > time_series_->count()) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Training time steps cannot exceed time series length.");
        }

        for (const auto& p : parameters_) {
            ValidationResult v = p.validate();
            if (!v.is_valid) {
                result.is_valid = false;
                for (const auto& m : v.validation_messages) result.validation_messages.push_back(m);
            }
        }
        if (!is_invertible()) {
            // C# message uses Greek/em-dash glyphs; ASCII-normalized here (keeps the "Warning" /
            // "invertibility" tokens the tests key on). See task-T1-report.md.
            result.validation_messages.push_back(
                "Warning: MA parameters do not satisfy the sum-of-absolute-values invertibility "
                "sufficient condition (sum|theta_i| < 1). The model may still be invertible - this "
                "check is conservative for orders >= 2 - but uniqueness of the MA representation is "
                "not guaranteed if it is not.");
        }
        if (transform_fit_validation_message_) {
            result.is_valid = false;
            result.validation_messages.push_back(*transform_fit_validation_message_);
        }
        return result;
    }

    // --- GenerateRandomValues (C#:933): ISimulatable entry point. Bit-exact MersenneTwister. ---
    std::vector<double> generate_random_values(int sample_size, int seed = -1) const override {
        if (sample_size <= 0)
            throw std::out_of_range("Sample size must be positive.");

        numerics::sampling::MersenneTwister rng =
            seed >= 0 ? numerics::sampling::MersenneTwister(static_cast<std::uint32_t>(seed))
                      : numerics::sampling::MersenneTwister();
        numerics::distributions::Normal normal(0.0, 1.0);

        int param_index = 0;
        double intercept = 0.0;
        if (include_intercept_) {
            intercept = parameters_[static_cast<std::size_t>(param_index)].value();
            param_index++;
        }
        std::vector<double> theta(static_cast<std::size_t>(order_));
        for (int i = 0; i < order_; ++i)
            theta[static_cast<std::size_t>(i)] =
                parameters_[static_cast<std::size_t>(param_index + i)].value();
        param_index += order_;

        double sigma = parameters_[static_cast<std::size_t>(param_index)].value();

        std::vector<double> epsilon(static_cast<std::size_t>(sample_size));
        for (int i = 0; i < sample_size; ++i)
            epsilon[static_cast<std::size_t>(i)] = sigma * normal.inverse_cdf(rng.next_double());

        std::vector<double> result(static_cast<std::size_t>(sample_size));
        for (int t = 0; t < sample_size; ++t) {
            double value = intercept + epsilon[static_cast<std::size_t>(t)];
            for (int j = 0; j < std::min(t, order_); ++j)
                value += theta[static_cast<std::size_t>(j)] *
                         epsilon[static_cast<std::size_t>(t - 1 - j)];
            result[static_cast<std::size_t>(t)] = value;
        }
        return result;
    }

   private:
    // --- ResetDefaultTrainingStepsForNewTimeSeries (MovingAverage.cs:347) -----------------------------------
    //
    // v2.0.0: attaching a DIFFERENT response series is a new calibration problem, so the setter
    // discards any manual training-window edit and restores the default split. An empty series
    // zeroes the window (the C# `TimeSeries = null` arm -- this port holds the series BY VALUE in
    // a std::optional, so "no usable series" is the empty series, never a null reference).
    // C# RaisePropertyChange calls -> no-ops here.
    //
    // Divergence, unavoidable and inert for this port: C# short-circuits the whole setter on
    // `ReferenceEquals(_timeSeries, value)`, so re-assigning the SAME object preserves a manual
    // window. Value semantics have no object identity to test, so re-assigning an equal series
    // here resets. The public surface never re-assigns the same series (the fixture/binding
    // builders assign once at construction), and C#'s companion CollectionChanged hook -- which
    // preserves the manual window when the ATTACHED series is edited in place -- has no analog
    // either, since this port hands out the series by const reference and cannot be edited in
    // place.
    void reset_default_training_steps_for_new_time_series() {
        use_default_training_steps_ = true;
        if (!time_series_ || time_series_->count() == 0) {
            training_time_steps_ = 0;
            return;
        }
        set_default_training_steps();
    }

    void set_default_training_steps() {
        if (!time_series_ || time_series_->count() == 0) return;
        int min_steps = std::max(30, static_cast<int>(parameters_.size()));
        set_training_time_steps(
            std::max(min_steps, static_cast<int>(std::floor(0.8 * time_series_->count()))));
    }

    // SetTrainingData (C#:333). MA transform subset starts at index 0 (AR starts at Order).
    void set_training_data() {
        transform_fit_validation_message_.reset();
        if (!time_series_ || training_time_steps_ == 0) return;

        int effective = std::min(training_time_steps_, time_series_->count());
        TimeSeries tts(time_series_->time_interval());

        if (transform_type_ == Transform::None) {
            lambda_ = 0;
            log_jacobian_ = 0;
            for (int i = 0; i < effective; ++i) tts.add((*time_series_)[i].clone());
        } else if (transform_type_ == Transform::Logarithmic) {
            lambda_ = 0;
            std::vector<double> data = subset(time_series_->values_to_array(), 0, effective - 1);
            log_jacobian_ = numerics::data::BoxCox::log_jacobian(data, lambda_);
            for (int i = 0; i < effective; ++i) {
                tts.add((*time_series_)[i].clone());
                tts[i].set_value(numerics::data::BoxCox::transform((*time_series_)[i].value(), lambda_));
            }
        } else if (transform_type_ == Transform::BoxCox) {
            lambda_ = numerics::data::BoxCox::fit_lambda(time_series_->values_to_list());
            if (!numerics::is_finite(lambda_)) {
                lambda_ = 0;
                log_jacobian_ = 0;
                training_time_series_ = TimeSeries(time_series_->time_interval());
                transform_fit_validation_message_ =
                    "Error: Box-Cox lambda estimation failed. Select a different transform or "
                    "revise the time-series data.";
                return;
            }
            std::vector<double> data = subset(time_series_->values_to_array(), 0, effective - 1);
            log_jacobian_ = numerics::data::BoxCox::log_jacobian(data, lambda_);
            for (int i = 0; i < effective; ++i) {
                tts.add((*time_series_)[i].clone());
                tts[i].set_value(numerics::data::BoxCox::transform((*time_series_)[i].value(), lambda_));
            }
        } else if (transform_type_ == Transform::YeoJohnson) {
            lambda_ = numerics::data::YeoJohnson::fit_lambda(time_series_->values_to_list());
            if (!numerics::is_finite(lambda_)) {
                lambda_ = 0;
                log_jacobian_ = 0;
                training_time_series_ = TimeSeries(time_series_->time_interval());
                transform_fit_validation_message_ =
                    "Error: Yeo-Johnson lambda estimation failed. Select a different transform "
                    "or revise the time-series data.";
                return;
            }
            std::vector<double> data = subset(time_series_->values_to_array(), 0, effective - 1);
            log_jacobian_ = numerics::data::YeoJohnson::log_jacobian(data, lambda_);
            for (int i = 0; i < effective; ++i) {
                tts.add((*time_series_)[i].clone());
                tts[i].set_value(
                    numerics::data::YeoJohnson::transform((*time_series_)[i].value(), lambda_));
            }
        }

        training_time_series_ = std::move(tts);
    }

    static std::vector<double> subset(const std::vector<double>& v, int start, int end) {
        std::vector<double> out;
        if (start < 0) start = 0;
        if (end >= static_cast<int>(v.size())) end = static_cast<int>(v.size()) - 1;
        for (int i = start; i <= end; ++i) out.push_back(v[static_cast<std::size_t>(i)]);
        return out;
    }

    std::optional<TimeSeries> time_series_;
    std::optional<TimeSeries> training_time_series_;
    int order_ = 1;
    bool include_intercept_ = true;

    Transform transform_type_ = Transform::None;
    double lambda_ = 0;
    double lambda2_ = 0;
    double log_jacobian_ = 0;
    // v2.0.0: set by set_training_data() when BoxCox/YeoJohnson FitLambda returns a non-finite
    // lambda; surfaced by validate() (C# _transformFitValidationMessage).
    std::optional<std::string> transform_fit_validation_message_;

    int training_time_steps_ = 0;
    bool use_default_training_steps_ = true;
    bool use_jeffreys_rule_for_scale_ = true;
};

}  // namespace corehydro::models
