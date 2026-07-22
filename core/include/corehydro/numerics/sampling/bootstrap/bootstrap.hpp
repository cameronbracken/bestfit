// ported from: Numerics/Sampling/Bootstrap/Bootstrap.cs @ 2a0357a
//
// A general-purpose bootstrap class for parametric or non-parametric bootstrap analysis.
// This port covers the REGULAR (non-pivotal) workflow only, per the task brief:
//   - Run(), RunDoubleBootstrap(), RunWithStudentizedBootstrap()
//   - GetConfidenceIntervals (Percentile / BiasCorrected / BCa / Normal / BootstrapT)
//   - the cube-root default Transform/InverseTransform
//   - the leave-one-out jackknife / BCa acceleration-constant machinery
// The covariance-aware PIVOTAL bootstrap (`RunPivotalBootstrap`, `TransformPivotalBootstrap`,
// `GetRawPivotalConfidenceIntervals`, `TryCreatePivotalDraw`, `TryApplyInvalidPolicy`,
// `LinkCovariance`, `CreatePivotalLinks`, the second (`BootstrapFit`) constructor, and every
// `Pivotal*` property/field) is a DOCUMENTED OMISSION -- ported in P3.11. Because that region
// is skipped, this port also drops the corresponding pieces of shared state
// (`_originalCovariance`, `_rawBootstrap*`, `_pivotalLinks`, `_pivotalDiagnostics`) and the
// `BootstrapRunType.Pivotal` enum member/branch; `GetConfidenceIntervals`'s pivotal-run-type
// branch is therefore unreachable code in the C# source under this port's scope and is
// likewise omitted.
//
// Namespace note: see bootstrap_results.hpp's header -- C# `Bootstrap<TData>` lives in the
// flat `Numerics.Sampling` namespace despite sitting in a `Sampling/Bootstrap/` folder; this
// port keeps that flat namespace (`corehydro::numerics::sampling`) for parity.
//
// Delegate ports: C#'s `Func<...>` properties become `std::function` public data members
// (default-constructed/empty, i.e. falsy via `operator bool`, unless assigned -- mirrors a
// null C# delegate). `Transform`/`InverseTransform` default to the cube-root/cube pair
// (`x => Math.Pow(x, 1d / 3d)` / `x => Math.Pow(x, 3d)`), transcribed via `std::pow` (not
// `std::cbrt`, which -- unlike `Math.Pow` -- returns the real cube root for negative `x`;
// `std::pow(x, 1.0/3.0)` matches `Math.Pow`'s NaN-for-negative-base behavior exactly).
//
// Threading: `Parallel.For(0, Replicates, idx => {...})` becomes a plain serial `for` loop.
// This is PROVABLY order-independent: each replicate `idx` reads only `seeds[idx]` (an
// up-front array, computed once before the loop) and writes only to `_bootstrapParameterSets
// [idx]`/`_bootstrapStatistics[idx, *]`/`_validFlags[idx]` -- no replicate reads or mutates
// another replicate's slot, so serial vs. parallel execution order cannot change any output
// value. `Interlocked.Increment(ref _failedCount)` becomes a plain `++failed_count_` (safe
// under the same no-cross-replicate-aliasing argument).
//
// Per-replicate seeding cascade (transcribed verbatim, load-bearing for cross-language
// reproducibility): ONE master `MersenneTwister(PRNGSeed)` draws `Replicates` seeds up front
// via `NextIntegers` (`utilities::next_integers`, in order); replicate `idx`'s FIRST attempt
// seeds a fresh `MersenneTwister` from `seeds[idx]`; a failed attempt's retry `r` (1-based
// count) reseeds from `seeds[idx] + 10 * r`. The `+` is computed in `uint32_t` (matching C#'s
// unchecked `int` addition, which wraps mod 2^32 in two's-complement -- the identical bit
// pattern to unsigned wraparound) to avoid signed-integer-overflow UB while staying bit-exact
// with the C# stream for every seed value, including near-`int.MaxValue` ones.
//
// BCa HAZARD (see `compute_acceleration_constants`'s own comment): C#'s
// `ComputeAccelerationConstants` uses `Tools.ParallelAdd` inside its own `Parallel.For` --  an
// order-DEPENDENT floating-point reduction that is NOT bit-reproducible even run-to-run in
// the real C# library (confirmed by running the oracle emitter twice and diffing `--dump`
// output). This port replaces it with a plain serial accumulation in jackknife-index order --
// deterministic within this port, but not guaranteed to match either C# run bit-for-bit. The
// BCa fixture case therefore uses a LOOSE tolerance sized from the measured C# run-to-run
// wobble (see fixtures/README.md's bootstrap schema).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"
#include "corehydro/numerics/sampling/bootstrap/bootstrap_results.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/tools.hpp"
#include "corehydro/numerics/utilities/extension_methods.hpp"

namespace corehydro::numerics::sampling {

namespace opt = corehydro::numerics::math::optimization;

template <typename TData>
class Bootstrap {
   public:
    // --- Delegates (C# `Func<...>` properties) ---------------------------------------------
    using ResampleFn = std::function<TData(const TData&, const opt::ParameterSet&, MersenneTwister&)>;
    using FitFn = std::function<opt::ParameterSet(const TData&)>;
    using StatisticFn = std::function<std::vector<double>(const opt::ParameterSet&)>;
    using JackknifeFn = std::function<TData(const TData&, int)>;
    using SampleSizeFn = std::function<int(const TData&)>;
    using TransformFn = std::function<double(double)>;

    // --- Construction ------------------------------------------------------------------------

    // Constructs a new regular bootstrap analysis.
    Bootstrap(TData original_data, opt::ParameterSet original_parameters)
        : original_data_(std::move(original_data)), original_parameters_(std::move(original_parameters)) {}

    // --- Properties ----------------------------------------------------------------------------

    // Delegate for resampling the original data given the current parameters and PRNG.
    ResampleFn resample_function;
    // Delegate for fitting a model to data and returning a parameter set.
    FitFn fit_function;
    // Delegate for extracting statistics from a fitted parameter set.
    StatisticFn statistic_function;
    // Optional delegate for computing a leave-one-out jackknife sample. Required for BCa.
    JackknifeFn jackknife_function;
    // Optional delegate returning the number of observations in the data. Required for BCa.
    SampleSizeFn sample_size_function;

    // Optional transform applied to statistic values before Normal/BootstrapT CI computation.
    // Default cube-root (see file header for why `std::pow`, not `std::cbrt`).
    TransformFn transform = [](double x) { return std::pow(x, 1.0 / 3.0); };
    // Optional inverse transform corresponding to `transform`. Default cube.
    TransformFn inverse_transform = [](double x) { return std::pow(x, 3.0); };

    // Number of bootstrap replicates. Default = 10,000.
    int replicates = 10000;
    // PRNG seed for reproducibility. Default = 12345.
    int prng_seed = 12345;
    // Maximum number of retries for a failed bootstrap replicate. Default = 20.
    int max_retries = 20;
    // Number of inner bootstrap replicates for Bootstrap-t standard error estimation.
    // Default = 300.
    int inner_replicates = 300;

    // The active bootstrapped model parameter sets.
    const std::vector<opt::ParameterSet>& bootstrap_parameter_sets() const { return bootstrap_parameter_sets_; }
    // The active bootstrapped statistics, [replicate][statistic].
    const std::vector<std::vector<double>>& bootstrap_statistics() const { return bootstrap_statistics_; }
    // The number of replicates that failed after all retries.
    int failed_replicates() const { return failed_count_; }

    // --- Run Methods -----------------------------------------------------------------------

    // Runs the regular bootstrap procedure with error handling and retry logic.
    void run() {
        validate_core_delegates();
        validate_replication_settings();
        initialize_state();

        MersenneTwister prng(static_cast<std::uint32_t>(prng_seed));
        std::vector<int> seeds = utilities::next_integers(prng, replicates);

        for (int idx = 0; idx < replicates; ++idx) {
            bool succeeded = false;
            for (int retry = 0; retry < max_retries; ++retry) {
                try {
                    MersenneTwister rng(replicate_seed(seeds[static_cast<std::size_t>(idx)], retry));
                    TData sample = resample_function(original_data_, original_parameters_, rng);
                    opt::ParameterSet fit_result = fit_function(sample);
                    if (!has_expected_finite_parameter_values(fit_result, num_params_)) continue;
                    std::vector<double> stat = validate_statistics(statistic_function(fit_result), num_stats_);

                    bootstrap_parameter_sets_[static_cast<std::size_t>(idx)] = fit_result;
                    for (int k = 0; k < num_stats_; ++k)
                        bootstrap_statistics_[static_cast<std::size_t>(idx)][static_cast<std::size_t>(k)] = stat[static_cast<std::size_t>(k)];
                    valid_flags_[static_cast<std::size_t>(idx)] = true;
                    succeeded = true;
                } catch (...) {
                    // Retry failed regular bootstrap replicates.
                }
                if (succeeded) break;
            }
            if (!succeeded) mark_failed(idx);
        }

        run_type_ = BootstrapRunType::Regular;
    }

    // Runs the double bootstrap procedure with bias correction.
    void run_double_bootstrap(int inner_reps = 300) {
        validate_core_delegates();
        validate_replication_settings();
        if (inner_reps < 1)
            throw std::invalid_argument("The number of inner replicates must be positive.");

        initialize_state();

        MersenneTwister prng(static_cast<std::uint32_t>(prng_seed));
        std::vector<int> seeds = utilities::next_integers(prng, replicates);

        for (int idx = 0; idx < replicates; ++idx) {
            bool succeeded = false;
            for (int retry = 0; retry < max_retries; ++retry) {
                try {
                    MersenneTwister rng(replicate_seed(seeds[static_cast<std::size_t>(idx)], retry));

                    TData outer_sample = resample_function(original_data_, original_parameters_, rng);
                    opt::ParameterSet outer_fit = fit_function(outer_sample);
                    if (!has_expected_finite_parameter_values(outer_fit, num_params_)) continue;
                    std::vector<double> outer_stat = validate_statistics(statistic_function(outer_fit), num_stats_);

                    int p = num_params_;
                    std::vector<double> params_inner_sum(static_cast<std::size_t>(p), 0.0);
                    std::vector<double> stats_inner_sum(static_cast<std::size_t>(num_stats_), 0.0);
                    int valid_inner = 0;

                    for (int k = 0; k < inner_reps; ++k) {
                        try {
                            TData inner_sample = resample_function(outer_sample, outer_fit, rng);
                            opt::ParameterSet inner_fit = fit_function(inner_sample);
                            if (!has_expected_finite_parameter_values(inner_fit, p)) continue;
                            std::vector<double> inner_stat = validate_statistics(statistic_function(inner_fit), num_stats_);

                            for (int i = 0; i < p; ++i)
                                params_inner_sum[static_cast<std::size_t>(i)] += inner_fit.values[static_cast<std::size_t>(i)];
                            for (int i = 0; i < num_stats_; ++i)
                                stats_inner_sum[static_cast<std::size_t>(i)] += inner_stat[static_cast<std::size_t>(i)];
                            ++valid_inner;
                        } catch (...) {
                            // Skip failed inner replicate.
                        }
                    }

                    if (valid_inner == 0) continue;

                    std::vector<double> bias_corrected_parms(static_cast<std::size_t>(p));
                    for (int i = 0; i < p; ++i) {
                        double inner_mean = params_inner_sum[static_cast<std::size_t>(i)] / valid_inner;
                        bias_corrected_parms[static_cast<std::size_t>(i)] =
                            outer_fit.values[static_cast<std::size_t>(i)] - (inner_mean - outer_fit.values[static_cast<std::size_t>(i)]);
                    }

                    std::vector<double> bias_corrected_stats(static_cast<std::size_t>(num_stats_));
                    for (int i = 0; i < num_stats_; ++i) {
                        double inner_mean = stats_inner_sum[static_cast<std::size_t>(i)] / valid_inner;
                        bias_corrected_stats[static_cast<std::size_t>(i)] =
                            outer_stat[static_cast<std::size_t>(i)] - (inner_mean - outer_stat[static_cast<std::size_t>(i)]);
                    }

                    bootstrap_parameter_sets_[static_cast<std::size_t>(idx)] =
                        opt::ParameterSet(bias_corrected_parms, outer_fit.fitness, outer_fit.weight);
                    for (int k = 0; k < num_stats_; ++k)
                        bootstrap_statistics_[static_cast<std::size_t>(idx)][static_cast<std::size_t>(k)] = bias_corrected_stats[static_cast<std::size_t>(k)];
                    valid_flags_[static_cast<std::size_t>(idx)] = true;
                    succeeded = true;
                } catch (...) {
                    // Retry failed outer replicate.
                }
                if (succeeded) break;
            }
            if (!succeeded) mark_failed(idx);
        }

        run_type_ = BootstrapRunType::DoubleBootstrap;
    }

    // Runs the regular bootstrap procedure with nested inner bootstrap for studentized
    // Bootstrap-t confidence intervals.
    void run_with_studentized_bootstrap() {
        validate_core_delegates();
        validate_replication_settings();
        if (inner_replicates < 1)
            throw std::invalid_argument("The number of inner replicates must be positive.");

        std::vector<double> original_stats = validate_statistics(statistic_function(original_parameters_));
        parent_statistics_ = original_stats;
        num_stats_ = static_cast<int>(original_stats.size());
        num_params_ = static_cast<int>(original_parameters_.values.size());

        std::vector<double> pop_transformed(static_cast<std::size_t>(num_stats_));
        for (int i = 0; i < num_stats_; ++i)
            pop_transformed[static_cast<std::size_t>(i)] = apply_transform(original_stats[static_cast<std::size_t>(i)]);

        bootstrap_parameter_sets_.assign(static_cast<std::size_t>(replicates), opt::ParameterSet{});
        bootstrap_statistics_.assign(static_cast<std::size_t>(replicates),
                                      std::vector<double>(static_cast<std::size_t>(num_stats_), 0.0));
        valid_flags_.assign(static_cast<std::size_t>(replicates), false);
        failed_count_ = 0;
        studentized_values_ =
            std::vector<std::vector<double>>(static_cast<std::size_t>(replicates), std::vector<double>(static_cast<std::size_t>(num_stats_), 0.0));
        transformed_statistics_ =
            std::vector<std::vector<double>>(static_cast<std::size_t>(replicates), std::vector<double>(static_cast<std::size_t>(num_stats_), 0.0));

        MersenneTwister prng(static_cast<std::uint32_t>(prng_seed));
        std::vector<int> seeds = utilities::next_integers(prng, replicates);

        for (int idx = 0; idx < replicates; ++idx) {
            bool succeeded = false;
            for (int retry = 0; retry < max_retries; ++retry) {
                try {
                    MersenneTwister rng(replicate_seed(seeds[static_cast<std::size_t>(idx)], retry));
                    TData sample = resample_function(original_data_, original_parameters_, rng);
                    opt::ParameterSet outer_fit = fit_function(sample);
                    if (!has_expected_finite_parameter_values(outer_fit, num_params_)) continue;
                    std::vector<double> outer_stats = validate_statistics(statistic_function(outer_fit), num_stats_);

                    bootstrap_parameter_sets_[static_cast<std::size_t>(idx)] = outer_fit;
                    for (int k = 0; k < num_stats_; ++k)
                        bootstrap_statistics_[static_cast<std::size_t>(idx)][static_cast<std::size_t>(k)] = outer_stats[static_cast<std::size_t>(k)];

                    std::vector<double> outer_transformed(static_cast<std::size_t>(num_stats_));
                    for (int j = 0; j < num_stats_; ++j)
                        outer_transformed[static_cast<std::size_t>(j)] = apply_transform(outer_stats[static_cast<std::size_t>(j)]);

                    // The inner PRNG is seeded from `seeds[idx]` DIRECTLY -- NOT offset by
                    // `retry` -- matching C#'s `new MersenneTwister(seeds[idx])` exactly.
                    MersenneTwister inner_prng(static_cast<std::uint32_t>(seeds[static_cast<std::size_t>(idx)]));
                    std::vector<int> inner_seeds = utilities::next_integers(inner_prng, inner_replicates);
                    std::vector<std::vector<double>> inner_transformed(
                        static_cast<std::size_t>(inner_replicates), std::vector<double>(static_cast<std::size_t>(num_stats_)));
                    int valid_inner = 0;

                    for (int k = 0; k < inner_replicates; ++k) {
                        try {
                            MersenneTwister inner_rng(static_cast<std::uint32_t>(inner_seeds[static_cast<std::size_t>(k)]));
                            TData inner_sample = resample_function(sample, outer_fit, inner_rng);
                            opt::ParameterSet inner_fit = fit_function(inner_sample);
                            if (!has_expected_finite_parameter_values(inner_fit, num_params_)) {
                                for (int j = 0; j < num_stats_; ++j)
                                    inner_transformed[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)] = kNaN;
                                continue;
                            }
                            std::vector<double> inner_stats = validate_statistics(statistic_function(inner_fit), num_stats_);

                            for (int j = 0; j < num_stats_; ++j)
                                inner_transformed[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)] =
                                    apply_transform(inner_stats[static_cast<std::size_t>(j)]);
                            ++valid_inner;
                        } catch (...) {
                            for (int j = 0; j < num_stats_; ++j)
                                inner_transformed[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)] = kNaN;
                        }
                    }

                    if (valid_inner < 2) continue;

                    for (int j = 0; j < num_stats_; ++j) {
                        std::vector<double> col = get_column(inner_transformed, j);
                        std::vector<double> valid_col;
                        valid_col.reserve(col.size());
                        for (double v : col)
                            if (corehydro::numerics::is_finite(v)) valid_col.push_back(v);
                        double se = valid_col.size() > 1 ? data::standard_deviation(valid_col) : kNaN;
                        (*transformed_statistics_)[static_cast<std::size_t>(idx)][static_cast<std::size_t>(j)] =
                            outer_transformed[static_cast<std::size_t>(j)];
                        (*studentized_values_)[static_cast<std::size_t>(idx)][static_cast<std::size_t>(j)] =
                            se > 0 ? (pop_transformed[static_cast<std::size_t>(j)] - outer_transformed[static_cast<std::size_t>(j)]) / se : kNaN;
                    }

                    valid_flags_[static_cast<std::size_t>(idx)] = true;
                    succeeded = true;
                } catch (...) {
                    // Retry failed outer replicate.
                }
                if (succeeded) break;
            }

            if (!succeeded) {
                mark_failed(idx);
                for (int j = 0; j < num_stats_; ++j) {
                    (*transformed_statistics_)[static_cast<std::size_t>(idx)][static_cast<std::size_t>(j)] = kNaN;
                    (*studentized_values_)[static_cast<std::size_t>(idx)][static_cast<std::size_t>(j)] = kNaN;
                }
            }
        }

        run_type_ = BootstrapRunType::Studentized;
    }

    // --- Confidence Intervals ----------------------------------------------------------------

    // Computes bootstrap confidence intervals using the specified method.
    BootstrapResults get_confidence_intervals(BootstrapCIMethod method, double alpha = 0.1) {
        validate_confidence_interval_request(method, alpha);

        if (!statistic_function) throw std::runtime_error("StatisticFunction must be set.");
        std::vector<double> original_stats = validate_statistics(statistic_function(original_parameters_));

        std::optional<std::vector<double>> accel_constants;
        if (method == BootstrapCIMethod::BCa) accel_constants = compute_acceleration_constants(original_stats);

        BootstrapResults results;
        results.method = method;
        results.alpha = alpha;
        results.statistic_results.resize(static_cast<std::size_t>(num_stats_));
        results.parameter_results.resize(static_cast<std::size_t>(num_params_));
        results.failed_replicates = failed_count_;

        for (int i = 0; i < num_stats_; ++i) {
            std::vector<double> values = get_column(bootstrap_statistics_, i);
            switch (method) {
                case BootstrapCIMethod::Percentile:
                    results.statistic_results[static_cast<std::size_t>(i)] =
                        compute_percentile_ci(values, original_stats[static_cast<std::size_t>(i)], alpha);
                    break;
                case BootstrapCIMethod::BiasCorrected:
                    results.statistic_results[static_cast<std::size_t>(i)] =
                        compute_bias_corrected_ci(values, original_stats[static_cast<std::size_t>(i)], alpha);
                    break;
                case BootstrapCIMethod::BCa:
                    results.statistic_results[static_cast<std::size_t>(i)] = compute_bca_ci(
                        values, original_stats[static_cast<std::size_t>(i)], alpha, (*accel_constants)[static_cast<std::size_t>(i)]);
                    break;
                case BootstrapCIMethod::Normal:
                    results.statistic_results[static_cast<std::size_t>(i)] =
                        compute_normal_ci(values, original_stats[static_cast<std::size_t>(i)], alpha);
                    break;
                case BootstrapCIMethod::BootstrapT:
                    results.statistic_results[static_cast<std::size_t>(i)] =
                        compute_bootstrap_t_ci(i, original_stats[static_cast<std::size_t>(i)], alpha);
                    break;
            }
        }

        for (int i = 0; i < num_params_; ++i) {
            std::vector<double> values;
            values.reserve(bootstrap_parameter_sets_.size());
            for (const auto& ps : bootstrap_parameter_sets_) values.push_back(ps.values[static_cast<std::size_t>(i)]);
            results.parameter_results[static_cast<std::size_t>(i)] =
                compute_percentile_ci(values, original_parameters_.values[static_cast<std::size_t>(i)], alpha);
        }

        return results;
    }

   private:
    // Identifies the workflow that produced the active bootstrap results. `Pivotal` is
    // omitted -- see file header.
    enum class BootstrapRunType { None, Regular, DoubleBootstrap, Studentized };

    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    TData original_data_;
    opt::ParameterSet original_parameters_;
    std::vector<opt::ParameterSet> bootstrap_parameter_sets_;
    std::vector<std::vector<double>> bootstrap_statistics_;  // [replicate][statistic]
    int num_stats_ = 0;
    int num_params_ = 0;
    int failed_count_ = 0;
    std::vector<bool> valid_flags_;
    std::optional<std::vector<std::vector<double>>> studentized_values_;
    std::optional<std::vector<std::vector<double>>> transformed_statistics_;
    std::optional<std::vector<double>> parent_statistics_;
    BootstrapRunType run_type_ = BootstrapRunType::None;

    // --- CI Methods --------------------------------------------------------------------------

    // Computes percentile confidence intervals for a single statistic or parameter.
    BootstrapStatisticResult compute_percentile_ci(const std::vector<double>& values, double population_estimate,
                                                    double alpha) const {
        std::vector<double> valid;
        valid.reserve(values.size());
        for (double v : values)
            if (corehydro::numerics::is_finite(v)) valid.push_back(v);
        std::sort(valid.begin(), valid.end());

        double lower_p = alpha / 2.0;
        double upper_p = 1.0 - alpha / 2.0;

        BootstrapStatisticResult r;
        r.population_estimate = population_estimate;
        r.lower_ci = !valid.empty() ? data::percentile(valid, lower_p, true) : kNaN;
        r.upper_ci = !valid.empty() ? data::percentile(valid, upper_p, true) : kNaN;
        r.valid_count = static_cast<int>(valid.size());
        r.total_count = static_cast<int>(values.size());
        r.standard_error = valid.size() > 1 ? data::standard_deviation(valid) : kNaN;
        r.mean = !valid.empty() ? data::mean(valid) : kNaN;
        return r;
    }

    // Computes bias-corrected confidence intervals for a single statistic.
    BootstrapStatisticResult compute_bias_corrected_ci(const std::vector<double>& values, double population_estimate,
                                                        double alpha) const {
        std::vector<double> valid;
        valid.reserve(values.size());
        for (double v : values)
            if (corehydro::numerics::is_finite(v)) valid.push_back(v);
        int valid_n = static_cast<int>(valid.size());
        if (valid_n == 0) return empty_result(population_estimate, static_cast<int>(values.size()));

        int count_leq = 0;
        for (int i = 0; i < valid_n; ++i)
            if (valid[static_cast<std::size_t>(i)] <= population_estimate) ++count_leq;
        double p0 = static_cast<double>(count_leq) / (valid_n + 1);

        std::sort(valid.begin(), valid.end());

        double z0 = distributions::Normal::standard_z(p0);
        double z_lower = distributions::Normal::standard_z(alpha / 2.0);
        double z_upper = distributions::Normal::standard_z(1.0 - alpha / 2.0);
        double bc_lower = distributions::Normal::standard_cdf(2.0 * z0 + z_lower);
        double bc_upper = distributions::Normal::standard_cdf(2.0 * z0 + z_upper);

        BootstrapStatisticResult r;
        r.population_estimate = population_estimate;
        r.lower_ci = data::percentile(valid, bc_lower, true);
        r.upper_ci = data::percentile(valid, bc_upper, true);
        r.valid_count = valid_n;
        r.total_count = static_cast<int>(values.size());
        r.standard_error = valid_n > 1 ? data::standard_deviation(valid) : kNaN;
        r.mean = data::mean(valid);
        return r;
    }

    // Computes bias-corrected and accelerated confidence intervals for a single statistic.
    BootstrapStatisticResult compute_bca_ci(const std::vector<double>& values, double population_estimate,
                                             double alpha, double acceleration) const {
        std::vector<double> valid;
        valid.reserve(values.size());
        for (double v : values)
            if (corehydro::numerics::is_finite(v)) valid.push_back(v);
        int valid_n = static_cast<int>(valid.size());
        if (valid_n == 0) return empty_result(population_estimate, static_cast<int>(values.size()));

        int count_leq = 0;
        for (int i = 0; i < valid_n; ++i)
            if (valid[static_cast<std::size_t>(i)] <= population_estimate) ++count_leq;
        double p0 = static_cast<double>(count_leq + 1) / (valid_n + 1);

        std::sort(valid.begin(), valid.end());

        double z0 = distributions::Normal::standard_z(p0);
        double z_lower = distributions::Normal::standard_z(alpha / 2.0);
        double z_upper = distributions::Normal::standard_z(1.0 - alpha / 2.0);

        double num_lower = z0 + z_lower;
        double den_lower = 1.0 - acceleration * num_lower;
        double bc_lower = distributions::Normal::standard_cdf(z0 + num_lower / den_lower);

        double num_upper = z0 + z_upper;
        double den_upper = 1.0 - acceleration * num_upper;
        double bc_upper = distributions::Normal::standard_cdf(z0 + num_upper / den_upper);

        BootstrapStatisticResult r;
        r.population_estimate = population_estimate;
        r.lower_ci = data::percentile(valid, bc_lower, true);
        r.upper_ci = data::percentile(valid, bc_upper, true);
        r.valid_count = valid_n;
        r.total_count = static_cast<int>(values.size());
        r.standard_error = valid_n > 1 ? data::standard_deviation(valid) : kNaN;
        r.mean = data::mean(valid);
        return r;
    }

    // Computes Normal confidence intervals for a single statistic using `transform`.
    BootstrapStatisticResult compute_normal_ci(const std::vector<double>& values, double population_estimate,
                                                double alpha) const {
        double pop_transformed = apply_transform(population_estimate);
        std::vector<double> valid_slice;
        valid_slice.reserve(values.size());
        for (double v : values)
            if (corehydro::numerics::is_finite(v)) valid_slice.push_back(apply_transform(v));

        if (valid_slice.size() < 2) return empty_result(population_estimate, static_cast<int>(values.size()));

        double se = data::standard_deviation(valid_slice);
        double z_lower = distributions::Normal::standard_z(alpha / 2.0);
        double z_upper = distributions::Normal::standard_z(1.0 - alpha / 2.0);

        double lower_transformed = pop_transformed + se * z_lower;
        double upper_transformed = pop_transformed + se * z_upper;

        BootstrapStatisticResult r;
        r.population_estimate = population_estimate;
        r.lower_ci = apply_inverse_transform(lower_transformed);
        r.upper_ci = apply_inverse_transform(upper_transformed);
        r.valid_count = static_cast<int>(valid_slice.size());
        r.total_count = static_cast<int>(values.size());
        r.standard_error = se;
        r.mean = data::mean(valid_slice);
        return r;
    }

    // Computes Bootstrap-t confidence intervals for a single statistic.
    BootstrapStatisticResult compute_bootstrap_t_ci(int statistic_index, double population_estimate,
                                                      double alpha) const {
        double pop_transformed = apply_transform(population_estimate);
        std::vector<double> x_col = get_column(*transformed_statistics_, statistic_index);
        std::vector<double> t_col = get_column(*studentized_values_, statistic_index);

        std::vector<double> valid_x, valid_t;
        valid_x.reserve(x_col.size());
        valid_t.reserve(t_col.size());
        for (double v : x_col)
            if (corehydro::numerics::is_finite(v)) valid_x.push_back(v);
        for (double v : t_col)
            if (corehydro::numerics::is_finite(v)) valid_t.push_back(v);

        if (valid_t.size() < 2) return empty_result(population_estimate, replicates);

        double se = data::standard_deviation(valid_x);
        std::sort(valid_t.begin(), valid_t.end());

        double t_lower = data::percentile(valid_t, alpha / 2.0, true);
        double t_upper = data::percentile(valid_t, 1.0 - alpha / 2.0, true);

        BootstrapStatisticResult r;
        r.population_estimate = population_estimate;
        r.lower_ci = apply_inverse_transform(pop_transformed + se * t_lower);
        r.upper_ci = apply_inverse_transform(pop_transformed + se * t_upper);
        r.valid_count = static_cast<int>(valid_t.size());
        r.total_count = replicates;
        r.standard_error = se;
        r.mean = data::mean(valid_x);
        return r;
    }

    // --- BCa Support ---------------------------------------------------------------------------

    // Computes acceleration constants for each statistic using leave-one-out jackknife
    // samples. See file header's BCa HAZARD note: this is a plain serial sum, NOT a
    // reproduction of C#'s order-dependent `Tools.ParallelAdd` reduction.
    std::vector<double> compute_acceleration_constants(const std::vector<double>& population_estimates) const {
        int n = sample_size_function(original_data_);
        std::vector<double> i2(static_cast<std::size_t>(num_stats_), 0.0);
        std::vector<double> i3(static_cast<std::size_t>(num_stats_), 0.0);
        std::vector<double> a(static_cast<std::size_t>(num_stats_), 0.0);

        for (int idx = 0; idx < n; ++idx) {
            try {
                TData jack_data = jackknife_function(original_data_, idx);
                opt::ParameterSet jack_fit = fit_function(jack_data);
                std::vector<double> jack_stats = statistic_function(jack_fit);

                for (int i = 0; i < num_stats_; ++i) {
                    double diff = population_estimates[static_cast<std::size_t>(i)] - jack_stats[static_cast<std::size_t>(i)];
                    i2[static_cast<std::size_t>(i)] += diff * diff;
                    i3[static_cast<std::size_t>(i)] += diff * diff * diff;
                }
            } catch (...) {
                // Skip failed jackknife samples.
            }
        }

        for (int i = 0; i < num_stats_; ++i)
            a[static_cast<std::size_t>(i)] = i3[static_cast<std::size_t>(i)] / (std::pow(i2[static_cast<std::size_t>(i)], 1.5) * 6.0);

        return a;
    }

    // --- Private Helpers ---------------------------------------------------------------------

    // Combines a replicate's up-front seed with its (0-based) retry count, in `uint32_t`
    // (see file header's seeding-cascade note).
    static std::uint32_t replicate_seed(int seed, int retry) {
        return static_cast<std::uint32_t>(seed) + static_cast<std::uint32_t>(10 * retry);
    }

    void validate_core_delegates() const {
        if (!resample_function) throw std::runtime_error("ResampleFunction must be set.");
        if (!fit_function) throw std::runtime_error("FitFunction must be set.");
        if (!statistic_function) throw std::runtime_error("StatisticFunction must be set.");
    }

    void validate_replication_settings() const {
        if (replicates < 1) throw std::invalid_argument("The number of replicates must be positive.");
        if (max_retries < 1) throw std::invalid_argument("The maximum retry count must be positive.");
    }

    // Initializes regular-bootstrap state arrays before a `run()` or `run_double_bootstrap()`.
    void initialize_state() {
        std::vector<double> original_stats = validate_statistics(statistic_function(original_parameters_));
        parent_statistics_ = original_stats;
        num_stats_ = static_cast<int>(original_stats.size());
        num_params_ = static_cast<int>(original_parameters_.values.size());

        bootstrap_parameter_sets_.assign(static_cast<std::size_t>(replicates), opt::ParameterSet{});
        bootstrap_statistics_.assign(static_cast<std::size_t>(replicates),
                                      std::vector<double>(static_cast<std::size_t>(num_stats_), 0.0));
        valid_flags_.assign(static_cast<std::size_t>(replicates), false);
        failed_count_ = 0;
        studentized_values_.reset();
        transformed_statistics_.reset();
        run_type_ = BootstrapRunType::None;
    }

    // Marks a replicate as failed with NaN parameter and statistic values.
    void mark_failed(int idx) {
        std::vector<double> nan_params(static_cast<std::size_t>(num_params_), kNaN);
        bootstrap_parameter_sets_[static_cast<std::size_t>(idx)] = opt::ParameterSet(nan_params, kNaN);
        for (int k = 0; k < num_stats_; ++k) bootstrap_statistics_[static_cast<std::size_t>(idx)][static_cast<std::size_t>(k)] = kNaN;
        valid_flags_[static_cast<std::size_t>(idx)] = false;
        ++failed_count_;  // plain increment replacing Interlocked.Increment -- see file header
    }

    double apply_transform(double value) const { return transform ? transform(value) : value; }
    double apply_inverse_transform(double value) const { return inverse_transform ? inverse_transform(value) : value; }

    static BootstrapStatisticResult empty_result(double population_estimate, int total_count) {
        BootstrapStatisticResult r;
        r.population_estimate = population_estimate;
        r.lower_ci = kNaN;
        r.upper_ci = kNaN;
        r.valid_count = 0;
        r.total_count = total_count;
        r.standard_error = kNaN;
        r.mean = kNaN;
        return r;
    }

    // Extracts column `col` from a [replicate][stat]-indexed 2D vector (this port's
    // representation of C#'s `double[,]`; mirrors `ExtensionMethods.GetColumn`, which is not
    // otherwise ported -- see extension_methods.hpp's own header note).
    static std::vector<double> get_column(const std::vector<std::vector<double>>& matrix, int col) {
        std::vector<double> values(matrix.size());
        for (std::size_t i = 0; i < matrix.size(); ++i) values[i] = matrix[i][static_cast<std::size_t>(col)];
        return values;
    }

    // Validates statistic values returned by `statistic_function`. The C# overload's null
    // check has no C++ equivalent (a `std::vector` cannot be null); every other check is
    // transcribed.
    static std::vector<double> validate_statistics(const std::vector<double>& statistics,
                                                     std::optional<int> expected_length = std::nullopt) {
        if (statistics.empty())
            throw std::runtime_error("The statistic function must return at least one statistic.");
        if (expected_length.has_value() && static_cast<int>(statistics.size()) != *expected_length)
            throw std::runtime_error("The statistic function must return the same number of statistics for every draw.");
        if (!contains_only_finite_values(statistics))
            throw std::runtime_error("The statistic function returned a non-finite value.");
        return statistics;
    }

    static bool has_expected_finite_parameter_values(const opt::ParameterSet& parameters, int expected_length) {
        return static_cast<int>(parameters.values.size()) == expected_length &&
               contains_only_finite_values(parameters.values);
    }

    static bool contains_only_finite_values(const std::vector<double>& values) {
        for (double v : values)
            if (!corehydro::numerics::is_finite(v)) return false;
        return true;
    }

    // Validates a confidence interval request against the last bootstrap run mode. The
    // pivotal-run-type branch is omitted -- see file header (`run_type_` never becomes
    // `Pivotal` in this port).
    void validate_confidence_interval_request(BootstrapCIMethod method, double alpha) const {
        if (run_type_ == BootstrapRunType::None)
            throw std::runtime_error("A bootstrap run must be completed before requesting confidence intervals.");
        if (alpha <= 0.0 || alpha >= 1.0) throw std::invalid_argument("Alpha must be between 0 and 1.");

        if (method == BootstrapCIMethod::BCa && (!jackknife_function || !sample_size_function))
            throw std::runtime_error("JackknifeFunction and SampleSizeFunction must be set for BCa method.");
        if (method == BootstrapCIMethod::BootstrapT && !studentized_values_.has_value())
            throw std::runtime_error(
                "RunWithStudentizedBootstrap() must be called before requesting Bootstrap-t confidence intervals.");
    }
};

}  // namespace corehydro::numerics::sampling
