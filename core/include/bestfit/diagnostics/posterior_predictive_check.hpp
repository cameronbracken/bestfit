// ported from: RMC-BestFit/src/RMC.BestFit/Diagnostics/PosteriorPredictiveCheck.cs @ fc28c0c
//
// Posterior predictive checking for Bayesian models: draw parameter sets from the supplied
// posterior samples, generate replicate datasets, and compare observed-data test statistics to
// their posterior predictive distribution via p-values.
//
// TEMPLATE-ON-MODEL: same rationale as prior_predictive_check.hpp -- the C# generic-over-IModel
// design ports as a class template on the concrete model type `TModel`, constrained by
// static_assert to derive from both `bestfit::models::ModelBase` and
// `bestfit::models::ISimulatable<std::vector<double>>` (the compile-time analogue of the C#
// `model is not ISimulatable<double[]>` runtime guard). The model is held by const reference
// (non-owning), mirroring the C# reference field.
//
// TWO CONSTRUCTORS (C# 63 and 95): the primary ctor takes the posterior samples as a
// `std::vector<ParameterSet>` (C# `IList<ParameterSet>`, the MCMCResults.Output format); the
// overload takes an `MCMCResults` and reads its `output` field. Both throw std::invalid_argument
// (C# ArgumentException) on empty samples / empty observed data. The C# ArgumentNullException
// guards have no C++ analogue (the model is a reference; the vectors are never null, only empty --
// the empty cases are the ported throws).
//
// SEED (C# governs): the C# ctors do NOT take a seed -- `Seed` (default 12345) is a settable
// property. This port follows the C# exactly (seed_ member + getter/setter, not a ctor param).
//
// PARALLELISM: every C# `Parallel.For` / `Parallel.ForEach` ports to a serial loop; the RNG
// draws (replicate indices + data seeds) are pre-generated off a single stream before the loop,
// exactly as the C# does, so seeded reproducibility is preserved.
//
// REPLICATES.COUNT DIVISOR (C# 274-286, load-bearing): ComputePValue divides by
// `replicates.size()`, the ACTUAL number of successfully generated replicates, NOT the requested
// `number_of_replicates` -- GenerateReplicates silently skips failed draws, so the returned list
// can be shorter. Transcribed exactly. When zero replicates survive, ComputePValue returns NaN
// and ComputeCommonPValues returns a default (all-NaN) result.
//
// SILENT SKIP (C# 214-220): the per-replicate try/catch logs via Debug.WriteLine and continues;
// this port swallows the exception and skips the replicate (silent no-throw), per the global rule.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "bestfit/diagnostics/predictive_check_results.hpp"
#include "bestfit/diagnostics/predictive_summary.hpp"
#include "bestfit/models/support/model_base.hpp"
#include "bestfit/models/support/simulatable.hpp"
#include "bestfit/numerics/data/statistics.hpp"
#include "bestfit/numerics/math/optimization/support/parameter_set.hpp"
#include "bestfit/numerics/sampling/mcmc/support/mcmc_results.hpp"
#include "bestfit/numerics/sampling/mersenne_twister.hpp"
#include "bestfit/numerics/utilities/extension_methods.hpp"

namespace bestfit::diagnostics {

namespace detail {

// Port of Numerics `Statistics.Skewness(IList<double>)` (Statistics.cs 292): the bias-corrected
// (G1) sample skewness `sqrt(n(n-1))/(n-2) * m3/m2^{3/2}`, where m2/m3 are the population second
// and third central moments. Returns NaN for an empty sequence. Kept file-local (the shared
// statistics header exposes mean/SD/percentile but not this standalone skewness).
inline double skewness(const std::vector<double>& data) {
    if (data.empty()) return std::numeric_limits<double>::quiet_NaN();
    double mean = bestfit::numerics::data::mean(data);
    double n = static_cast<double>(data.size());
    double s2 = 0.0, s3 = 0.0;
    for (double x : data) {
        double xm = x - mean;
        s2 += xm * xm;
        s3 += xm * xm * xm;
    }
    double m2 = s2 / n;
    double m3 = s3 / n;
    double g = m3 / std::pow(m2, 3.0 / 2.0);
    double a = std::sqrt(n * (n - 1.0));
    double b = n - 2.0;
    return a / b * g;
}

}  // namespace detail

template <typename TModel>
class PosteriorPredictiveCheck {
    static_assert(std::is_base_of<bestfit::models::ModelBase, TModel>::value,
                  "TModel must derive from bestfit::models::ModelBase.");
    static_assert(
        std::is_base_of<bestfit::models::ISimulatable<std::vector<double>>, TModel>::value,
        "TModel must implement ISimulatable<std::vector<double>> for data generation.");

    using ParameterSet = bestfit::numerics::math::optimization::ParameterSet;
    using MersenneTwister = bestfit::numerics::sampling::MersenneTwister;
    using MCMCResults = bestfit::numerics::sampling::mcmc::MCMCResults;

   public:
    // C# primary ctor (C# 63): model + posterior samples (MCMCResults.Output format) + observed
    // data. Throws std::invalid_argument (C# ArgumentException) on empty samples / observed.
    PosteriorPredictiveCheck(const TModel& model, std::vector<ParameterSet> posterior_samples,
                             std::vector<double> observed_data)
        : model_(model),
          posterior_samples_(std::move(posterior_samples)),
          observed_data_(std::move(observed_data)) {
        if (posterior_samples_.empty())
            throw std::invalid_argument("Posterior samples cannot be empty.");
        if (observed_data_.empty())
            throw std::invalid_argument("Observed data cannot be empty.");
    }

    // C# MCMCResults overload (C# 95): reads the posterior draws off `mcmcResults.Output`.
    // Throws std::invalid_argument (C# ArgumentException) when the results carry no samples.
    PosteriorPredictiveCheck(const TModel& model, const MCMCResults& mcmc_results,
                             std::vector<double> observed_data)
        : model_(model), observed_data_(std::move(observed_data)) {
        if (mcmc_results.output.empty())
            throw std::invalid_argument("MCMC results contain no samples.");
        if (observed_data_.empty())
            throw std::invalid_argument("Observed data cannot be empty.");
        posterior_samples_ = mcmc_results.output;
    }

    // C# `Model` (C# 131).
    const TModel& model() const { return model_; }

    // C# `NumberOfPosteriorSamples` (C# 136).
    int number_of_posterior_samples() const { return static_cast<int>(posterior_samples_.size()); }

    // C# `SampleSize` (C# 141).
    int sample_size() const { return static_cast<int>(observed_data_.size()); }

    // C# `Seed` (C# 146-150): default 12345.
    int seed() const { return seed_; }
    void set_seed(int seed) { seed_ = seed; }

    // C# `GenerateReplicates(int numberOfReplicates)` (C# 179): one replicate dataset per drawn
    // posterior sample. Throws (C# ArgumentOutOfRangeException -> std::out_of_range) when
    // numberOfReplicates < 1. Failed draws are silently skipped, so the result can be shorter
    // than requested.
    std::vector<std::vector<double>> generate_replicates(int number_of_replicates) const {
        if (number_of_replicates < 1)
            throw std::out_of_range("Must generate at least 1 replicate.");

        MersenneTwister rng(static_cast<std::uint32_t>(seed_));
        int n = static_cast<int>(observed_data_.size());

        // Pre-generate posterior indices, then data seeds -- same stream order as C# (C# 188-189).
        auto indices = bestfit::numerics::utilities::next_integers(
            rng, 0, static_cast<int>(posterior_samples_.size()), number_of_replicates);
        auto seeds = bestfit::numerics::utilities::next_integers(rng, number_of_replicates);

        std::vector<std::vector<double>> result;
        result.reserve(static_cast<std::size_t>(number_of_replicates));

        for (int rep = 0; rep < number_of_replicates; ++rep) {
            try {
                int idx = indices[static_cast<std::size_t>(rep)];
                int data_seed = seeds[static_cast<std::size_t>(rep)];
                const auto& posterior_params = posterior_samples_[static_cast<std::size_t>(idx)];

                TModel clone = model_.clone();
                clone.set_parameter_values(posterior_params.values);

                auto y_rep = clone.generate_random_values(n, data_seed);
                if (!y_rep.empty()) result.push_back(std::move(y_rep));
            } catch (...) {
                // Invalid parameters -- silently skip (C# 214-220 swallowed guard).
            }
        }

        return result;
    }

    // C# `ComputePValue(Func<double[],double>, int numberOfReplicates = 1000)` (C# 260):
    // `#{ T(y_rep) >= T(y_obs) } / replicates.Count`. Throws std::invalid_argument (C#
    // ArgumentNullException) when the test statistic is empty. Returns NaN when no replicate
    // survives. The divisor is the ACTUAL replicate count, not numberOfReplicates (see header).
    double compute_p_value(const std::function<double(const std::vector<double>&)>& test_statistic,
                           int number_of_replicates = 1000) const {
        if (!test_statistic)
            throw std::invalid_argument("Test statistic cannot be null.");

        double t_obs = test_statistic(observed_data_);

        auto replicates = generate_replicates(number_of_replicates);
        if (replicates.empty()) return std::numeric_limits<double>::quiet_NaN();

        int count_ge = 0;
        for (const auto& rep : replicates)
            if (test_statistic(rep) >= t_obs) ++count_ge;

        return static_cast<double>(count_ge) / static_cast<double>(replicates.size());
    }

    // C# `ComputeCommonPValues(int numberOfReplicates = 1000)` (C# 299): p-values for the
    // mean/SD/skewness/min/max test statistics. Returns a default (all-NaN) result when no
    // replicate survives.
    PredictiveCheckResults compute_common_p_values(int number_of_replicates = 1000) const {
        auto replicates = generate_replicates(number_of_replicates);
        if (replicates.empty()) return PredictiveCheckResults();

        double obs_mean = bestfit::numerics::data::mean(observed_data_);
        double obs_sd = bestfit::numerics::data::standard_deviation(observed_data_);
        double obs_skew = detail::skewness(observed_data_);
        double obs_min = *std::min_element(observed_data_.begin(), observed_data_.end());
        double obs_max = *std::max_element(observed_data_.begin(), observed_data_.end());

        int count_mean = 0, count_sd = 0, count_skew = 0, count_min = 0, count_max = 0;
        for (const auto& y_rep : replicates) {
            if (bestfit::numerics::data::mean(y_rep) >= obs_mean) ++count_mean;
            if (bestfit::numerics::data::standard_deviation(y_rep) >= obs_sd) ++count_sd;
            if (detail::skewness(y_rep) >= obs_skew) ++count_skew;
            if (*std::min_element(y_rep.begin(), y_rep.end()) >= obs_min) ++count_min;
            if (*std::max_element(y_rep.begin(), y_rep.end()) >= obs_max) ++count_max;
        }

        double n = static_cast<double>(replicates.size());
        PredictiveCheckResults results;
        results.number_of_replicates = static_cast<int>(replicates.size());
        results.mean_p_value = count_mean / n;
        results.sd_p_value = count_sd / n;
        results.skewness_p_value = count_skew / n;
        results.min_p_value = count_min / n;
        results.max_p_value = count_max / n;
        return results;
    }

    // C# `ComputeSummary(int numberOfReplicates = 1000)` (C# 345): posterior predictive summary
    // statistics.
    PredictiveSummary compute_summary(int number_of_replicates = 1000) const {
        return compute_summary_from_data(generate_replicates(number_of_replicates));
    }

   private:
    // C# private `ComputeSummaryFromData(List<double[]>)` (C# 356): quantiles [2.5, 25, 50, 75,
    // 97.5]% of the mean/SD/min/max statistics across the replicate datasets.
    static PredictiveSummary compute_summary_from_data(
        const std::vector<std::vector<double>>& replicates) {
        if (replicates.empty()) return PredictiveSummary();

        std::size_t n = replicates.size();
        const std::vector<double> percents = {0.025, 0.25, 0.5, 0.75, 0.975};
        std::vector<double> means(n), sds(n), mins(n), maxs(n);
        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

        for (std::size_t i = 0; i < n; ++i) {
            const auto& data = replicates[i];
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
        summary.number_of_valid_draws = static_cast<int>(replicates.size());
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
    std::vector<ParameterSet> posterior_samples_;
    std::vector<double> observed_data_;
    int seed_ = 12345;
};

}  // namespace bestfit::diagnostics
