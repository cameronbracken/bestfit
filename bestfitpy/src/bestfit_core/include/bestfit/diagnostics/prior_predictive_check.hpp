// ported from: RMC-BestFit/src/RMC.BestFit/Diagnostics/PriorPredictiveCheck.cs @ fc28c0c
//
// Prior predictive checking for Bayesian models: draw parameter sets from the priors, generate
// simulated datasets from those draws, and summarize the resulting predictive distribution.
//
// TEMPLATE-ON-MODEL (C++ mapping of the C# generic-over-IModel design): the C# class holds an
// `IModel` and, per draw, calls `model.Clone()` (returning `IModel`), `SetParameterValues`, then
// casts to `ISimulatable<double[]>` and calls `GenerateRandomValues`. This port's ModelBase has
// no virtual `clone()` (the concrete models each own a value-returning `clone()`), so the check
// is a class template parameterized on the concrete model type `TModel`. `TModel` is constrained
// by static_assert to derive from both `bestfit::models::ModelBase` (for `parameters()` /
// `prior_log_likelihood()`) and `bestfit::models::ISimulatable<std::vector<double>>` (for
// `generate_random_values()`), which is the compile-time analogue of the C# ctor's
// `model is not ISimulatable<double[]>` runtime guard. The model is held by CONST REFERENCE
// (non-owning), mirroring the C# reference field; the caller must keep the model alive for the
// check's lifetime.
//
// SEED / NUMBER-OF-DRAWS (C# governs): the C# ctor takes only the model; `Seed` (default 12345)
// and `NumberOfDraws` (default 1000) are settable properties, NOT ctor parameters. This port
// follows the C# ctor exactly -- seed_/number_of_draws_ are members with getters/setters, not
// ctor args (the task brief's "seed = 12345" ctor parameter does not exist in the C# source).
//
// PARALLELISM: every C# `Parallel.For` ports to a serial loop -- the loop bodies are independent
// writes into pre-sized arrays, so the serialized result is numerically identical (the RNG draws
// are pre-generated off a single stream before the loop, exactly as the C# does, so seeded
// reproducibility is preserved).
//
// NULL-PRIOR FALLBACK (C# 171-187): the C# has a third branch for parameters whose
// `PriorDistribution is null`, falling back to uniform-within-bounds sampling. In this port a
// `ModelParameter` ALWAYS owns a prior distribution (default Uniform; there is no null state
// without editing the oracle-locked ModelParameter), so that branch is unreachable and is not
// ported. The reachable branches (fixed value; sample-from-prior-then-clamp) are transcribed
// verbatim, and both consume the same pre-generated uniform, so the RNG stream is unaffected.
//
// DROPPED (no numerical content): the `Debug.WriteLine` diagnostic in the per-draw catch (C# 265)
// ports as a silent no-throw skip, per the global swallowed-guard rule.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "bestfit/diagnostics/predictive_summary.hpp"
#include "bestfit/models/support/model_base.hpp"
#include "bestfit/models/support/simulatable.hpp"
#include "bestfit/numerics/data/statistics.hpp"
#include "bestfit/numerics/math/optimization/support/parameter_set.hpp"
#include "bestfit/numerics/sampling/mersenne_twister.hpp"
#include "bestfit/numerics/utilities/extension_methods.hpp"

namespace bestfit::diagnostics {

template <typename TModel>
class PriorPredictiveCheck {
    static_assert(std::is_base_of<bestfit::models::ModelBase, TModel>::value,
                  "TModel must derive from bestfit::models::ModelBase.");
    static_assert(
        std::is_base_of<bestfit::models::ISimulatable<std::vector<double>>, TModel>::value,
        "TModel must implement ISimulatable<std::vector<double>> for data generation.");

    using ParameterSet = bestfit::numerics::math::optimization::ParameterSet;
    using MersenneTwister = bestfit::numerics::sampling::MersenneTwister;

   public:
    // C# ctor `PriorPredictiveCheck(IModel model)` (C# 61): the ISimulatable guard is a
    // compile-time static_assert here (see file header). The C# ArgumentNullException guard has
    // no C++ analogue (the model is a reference, never null).
    explicit PriorPredictiveCheck(const TModel& model) : model_(model) {}

    // C# `Model` (C# 84).
    const TModel& model() const { return model_; }

    // C# `Seed` (C# 89-93): default 12345.
    int seed() const { return seed_; }
    void set_seed(int seed) { seed_ = seed; }

    // C# `NumberOfDraws` (C# 101-110): default 1000; setter throws (C# ArgumentOutOfRangeException
    // -> std::out_of_range) when value < 1.
    int number_of_draws() const { return number_of_draws_; }
    void set_number_of_draws(int number_of_draws) {
        if (number_of_draws < 1)
            throw std::out_of_range("Number of draws must be at least 1.");
        number_of_draws_ = number_of_draws;
    }

    // C# `SampleFromPriors()` (C# 137): samples one ParameterSet per draw; Fitness holds the
    // negative prior log-likelihood.
    std::vector<ParameterSet> sample_from_priors() const {
        const auto& parameters = model_.parameters();
        int num_params = static_cast<int>(parameters.size());

        // Pre-generate all uniform random values (2-D sub-generator-per-column stream).
        MersenneTwister rng(static_cast<std::uint32_t>(seed_));
        auto uniforms = bestfit::numerics::utilities::next_doubles(rng, number_of_draws_, num_params);

        std::vector<ParameterSet> result(static_cast<std::size_t>(number_of_draws_));

        for (int i = 0; i < number_of_draws_; ++i) {
            std::vector<double> values(static_cast<std::size_t>(num_params));

            for (int j = 0; j < num_params; ++j) {
                const auto& param = parameters[static_cast<std::size_t>(j)];

                if (param.is_fixed()) {
                    values[static_cast<std::size_t>(j)] = param.value();
                } else {
                    // Sample from the prior via inverse CDF, then clamp to bounds (C# 162-170).
                    // The null-prior fallback branch is unreachable in this port (see header).
                    double u = uniforms[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                    double v = param.prior_distribution().inverse_cdf(u);
                    v = std::max(param.lower_bound(), std::min(param.upper_bound(), v));
                    values[static_cast<std::size_t>(j)] = v;
                }
            }

            // Prior log-likelihood for this parameter set; Fitness = -priorLL (C# 191-192).
            double prior_ll = model_.prior_log_likelihood(values);
            result[static_cast<std::size_t>(i)] = ParameterSet(std::move(values), -prior_ll);
        }

        return result;
    }

    // C# `GeneratePriorPredictive(int sampleSize)` (C# 225): one simulated dataset per valid
    // prior draw. Throws (C# ArgumentOutOfRangeException -> std::out_of_range) when sampleSize < 1.
    std::vector<std::vector<double>> generate_prior_predictive(int sample_size) const {
        if (sample_size < 1)
            throw std::out_of_range("Sample size must be at least 1.");

        auto prior_samples = sample_from_priors();
        int num_samples = static_cast<int>(prior_samples.size());

        MersenneTwister rng(static_cast<std::uint32_t>(seed_ + 1));
        auto seeds = bestfit::numerics::utilities::next_integers(rng, num_samples);

        std::vector<std::vector<double>> result;
        result.reserve(static_cast<std::size_t>(num_samples));

        for (int i = 0; i < num_samples; ++i) {
            try {
                const auto& prior_params = prior_samples[static_cast<std::size_t>(i)];

                // Skip invalid draws (Fitness = -priorLL, so invalid if +Infinity or NaN, C# 248).
                double f = prior_params.fitness;
                if ((std::isinf(f) && f > 0.0) || std::isnan(f)) continue;

                TModel clone = model_.clone();
                clone.set_parameter_values(prior_params.values);

                auto data = clone.generate_random_values(sample_size,
                                                         seeds[static_cast<std::size_t>(i)]);
                if (!data.empty()) result.push_back(std::move(data));
            } catch (...) {
                // Invalid parameter combination -- silently skip (C# 262-266 swallowed guard).
            }
        }

        return result;
    }

    // C# `ComputeSummary(int sampleSize)` (C# 285): prior predictive summary statistics.
    PredictiveSummary compute_summary(int sample_size) const {
        return compute_summary_from_data(generate_prior_predictive(sample_size));
    }

   private:
    // C# private `ComputeSummaryFromData(List<double[]>)` (C# 299): quantiles [2.5, 25, 50, 75,
    // 97.5]% of the mean/SD/min/max statistics across the predictive datasets.
    static PredictiveSummary compute_summary_from_data(
        const std::vector<std::vector<double>>& predictive_data) {
        if (predictive_data.empty()) return PredictiveSummary();

        std::size_t n = predictive_data.size();
        const std::vector<double> percents = {0.025, 0.25, 0.5, 0.75, 0.975};
        std::vector<double> means(n), sds(n), mins(n), maxs(n);
        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

        for (std::size_t i = 0; i < n; ++i) {
            const auto& data = predictive_data[i];
            std::vector<double> valid;
            valid.reserve(data.size());
            for (double x : data)
                if (std::isfinite(x)) valid.push_back(x);

            if (valid.empty()) {
                means[i] = sds[i] = mins[i] = maxs[i] = kNaN;
                continue;
            }
            means[i] = bestfit::numerics::data::mean(valid);
            sds[i] = bestfit::numerics::data::standard_deviation(valid);
            mins[i] = *std::min_element(valid.begin(), valid.end());
            maxs[i] = *std::max_element(valid.begin(), valid.end());
        }

        PredictiveSummary summary;
        summary.number_of_valid_draws = static_cast<int>(predictive_data.size());
        summary.mean_quantiles = percentiles(means, percents);
        summary.sd_quantiles = percentiles(sds, percents);
        summary.min_quantiles = percentiles(mins, percents);
        summary.max_quantiles = percentiles(maxs, percents);
        return summary;
    }

    // C# `Statistics.Percentile(IList<double> data, IList<double> k)` (Statistics.cs 573): sort
    // once, then evaluate each requested percentile on the sorted copy (Type-7 interpolation).
    static std::vector<double> percentiles(std::vector<double> data,
                                           const std::vector<double>& ks) {
        std::sort(data.begin(), data.end());
        std::vector<double> result(ks.size());
        for (std::size_t i = 0; i < ks.size(); ++i)
            result[i] = bestfit::numerics::data::percentile(data, ks[i], /*data_is_sorted=*/true);
        return result;
    }

    const TModel& model_;
    int seed_ = 12345;
    int number_of_draws_ = 1000;
};

}  // namespace bestfit::diagnostics
