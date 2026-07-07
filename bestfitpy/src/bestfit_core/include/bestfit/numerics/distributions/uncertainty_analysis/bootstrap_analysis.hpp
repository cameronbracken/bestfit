// ported from: Numerics/Distributions/Univariate/Uncertainty Analysis/BootstrapAnalysis.cs @ a2c4dbf
//
// The standalone frequentist parametric-bootstrap uncertainty engine. Given an IBootstrappable
// univariate distribution, a parameter-estimation method, a bootstrap sample size and a replication
// count, it resamples the distribution `Replications` times and derives quantile confidence bands by
// five methods (Percentile, Bias-Corrected, Normal cube-root, Bootstrap-t, BCa) plus a full
// UncertaintyAnalysisResults (mode/CI/mean curves) via Estimate().
//
// DEVIATIONS from the C# source, all deliberate:
//  * Threading REMOVED: every C# `Parallel.For` / `Tools.ParallelAdd` reduction becomes a plain
//    serial loop. The reductions are order-independent sums / independent element writes, so the
//    serial form is numerically identical to the parallel one (same as the ported
//    UncertaintyAnalysisResults). Result: the async `RunAsync`-style methods are ordinary synchronous
//    calls.
//  * The owned `Distribution` is stored as an owning std::unique_ptr<UnivariateDistributionBase>
//    (a clone of the ctor argument), with cached IBootstrappable*/IEstimation* views. The C#
//    holds the caller's object; cloning keeps ownership simple and the parameters BCa mutates
//    (SampleSize + an in-place Estimate on the sample data) local to this analysis.
//  * The `IUnivariateDistribution[]? distributions = null` optional arguments become a
//    `const std::vector<const UnivariateDistributionBase*>*` (nullptr => generate internally),
//    mirroring the C# "pass a shared bootstrapped set or let me build one" contract. Owning
//    generated sets are held in a local and viewed as non-owning pointers, matching how the C#
//    array of polymorphic distributions is passed by reference.
//  * The C# `XValues[idx] != double.NaN` guards are a C#-language no-op (NaN != anything is always
//    true); the effective test is `value <= population` (NaN compares false), transcribed as that.
//  * `Debug`/swallowed fit failures -> silent no-throw, exactly as the C# try/catch arms already do
//    (a failed replicate becomes a null distribution / NaN column).
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/data/interpolation/linear.hpp"
#include "bestfit/numerics/data/interpolation/transform.hpp"
#include "bestfit/numerics/data/statistics.hpp"
#include "bestfit/numerics/distributions/base/i_bootstrappable.hpp"
#include "bestfit/numerics/distributions/base/i_estimation.hpp"
#include "bestfit/numerics/distributions/base/parameter_estimation_method.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "bestfit/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"
#include "bestfit/numerics/math/optimization/support/parameter_set.hpp"
#include "bestfit/numerics/sampling/mersenne_twister.hpp"
#include "bestfit/numerics/utilities/extension_methods.hpp"

namespace bestfit::numerics {

class BootstrapAnalysis {
    using UnivariateDistributionBase = distributions::UnivariateDistributionBase;
    using IBootstrappable = distributions::IBootstrappable;
    using IEstimation = distributions::IEstimation;
    using ParameterEstimationMethod = distributions::ParameterEstimationMethod;
    using ParameterSet = math::optimization::ParameterSet;
    using DistPtr = std::unique_ptr<UnivariateDistributionBase>;
    using DistView = std::vector<const UnivariateDistributionBase*>;

    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

   public:
    // Construct a new Bootstrap Analysis. The distribution must be IBootstrappable (the C#
    // `distribution as IBootstrappable == null` guard). sampleSize >= 10, replications >= 100.
    BootstrapAnalysis(const UnivariateDistributionBase& distribution,
                      ParameterEstimationMethod estimationMethod, int sampleSize,
                      int replications = 10000, int seed = 12345) {
        if (dynamic_cast<const IBootstrappable*>(&distribution) == nullptr)
            throw std::invalid_argument("The distribution must implement IBootstrappable.");
        if (sampleSize < 10)
            throw std::out_of_range("The sample size must at least 10.");
        if (replications < 100)
            throw std::out_of_range("The number of bootstrap replications must be at least 100.");

        distribution_ = distribution.clone();
        bootstrappable_ = dynamic_cast<IBootstrappable*>(distribution_.get());
        estimation_ = dynamic_cast<IEstimation*>(distribution_.get());
        estimation_method_ = estimationMethod;
        sample_size_ = sampleSize;
        replications_ = replications;
        prng_seed_ = seed;
    }

    // --- Accessors (mirror the C# get properties) ---
    ParameterEstimationMethod estimation_method() const { return estimation_method_; }
    int sample_size() const { return sample_size_; }
    int replications() const { return replications_; }
    int prng_seed() const { return prng_seed_; }

    // Bootstrap a list of fitted distributions.
    std::vector<DistPtr> distributions() {
        std::vector<DistPtr> boot(static_cast<std::size_t>(replications_));
        sampling::MersenneTwister r(static_cast<std::uint32_t>(prng_seed_));
        auto seeds = utilities::next_integers(r, replications_);
        for (int idx = 0; idx < replications_; ++idx) {
            DistPtr result;
            bool failed = false;
            for (int m = 0; m < retries_; ++m) {
                try {
                    result = bootstrappable_->bootstrap(estimation_method_, sample_size_,
                                                        seeds[static_cast<std::size_t>(idx)] + 10 * m);
                    failed = false;
                } catch (...) {
                    failed = true;
                }
                if (!failed) break;
            }
            // MLE and certain L-moments methods can fail; on fail, leave null.
            if (failed) result.reset();
            boot[static_cast<std::size_t>(idx)] = std::move(result);
        }
        return boot;
    }

    // Return a list of distributions given an array of parameter sets.
    std::vector<DistPtr> distributions(const std::vector<ParameterSet>& parameterSets) {
        std::vector<DistPtr> boot(parameterSets.size());
        for (std::size_t idx = 0; idx < parameterSets.size(); ++idx) {
            try {
                auto dist = distribution_->clone();
                dist->set_parameters(parameterSets[idx].values);
                boot[idx] = std::move(dist);
            } catch (...) {
                boot[idx].reset();
            }
        }
        return boot;
    }

    // Bootstrap an array of distribution parameters [B][NumberOfParameters].
    std::vector<std::vector<double>> parameters(const DistView* distributions_in = nullptr) {
        std::vector<DistPtr> owned;
        DistView view = resolve(distributions_in, owned);
        int np = distribution_->number_of_parameters();
        std::vector<std::vector<double>> out(view.size(),
                                             std::vector<double>(static_cast<std::size_t>(np)));
        for (std::size_t idx = 0; idx < view.size(); ++idx) {
            if (view[idx] != nullptr) {
                auto p = view[idx]->get_parameters();
                for (int i = 0; i < np; ++i) out[idx][static_cast<std::size_t>(i)] = p[static_cast<std::size_t>(i)];
            } else {
                for (int i = 0; i < np; ++i) out[idx][static_cast<std::size_t>(i)] = kNaN;
            }
        }
        return out;
    }

    // Bootstrap an array of distribution parameter sets.
    std::vector<ParameterSet> parameter_sets(const DistView* distributions_in = nullptr) {
        std::vector<DistPtr> owned;
        DistView view = resolve(distributions_in, owned);
        int np = distribution_->number_of_parameters();
        std::vector<ParameterSet> out(view.size());
        for (std::size_t idx = 0; idx < view.size(); ++idx) {
            if (view[idx] != nullptr) {
                out[idx] = ParameterSet(view[idx]->get_parameters(), kNaN);
            } else {
                out[idx] = ParameterSet(std::vector<double>(static_cast<std::size_t>(np), kNaN), kNaN);
            }
        }
        return out;
    }

    // Bootstrap a list of product moments for each bootstrapped sample [B][4].
    std::vector<std::array<double, 4>> product_moments() {
        std::vector<std::array<double, 4>> out(static_cast<std::size_t>(replications_));
        sampling::MersenneTwister r(static_cast<std::uint32_t>(prng_seed_));
        auto seeds = utilities::next_integers(r, replications_);
        for (int idx = 0; idx < replications_; ++idx) {
            auto moments = data::product_moments(
                distribution_->generate_random_values(sample_size_, seeds[static_cast<std::size_t>(idx)]));
            for (int i = 0; i < 4; ++i)
                out[static_cast<std::size_t>(idx)][static_cast<std::size_t>(i)] = moments[static_cast<std::size_t>(i)];
        }
        return out;
    }

    // Bootstrap a list of linear moments for each bootstrapped sample [B][4].
    std::vector<std::array<double, 4>> linear_moments() {
        std::vector<std::array<double, 4>> out(static_cast<std::size_t>(replications_));
        sampling::MersenneTwister r(static_cast<std::uint32_t>(prng_seed_));
        auto seeds = utilities::next_integers(r, replications_);
        for (int idx = 0; idx < replications_; ++idx) {
            auto moments = data::linear_moments(
                distribution_->generate_random_values(sample_size_, seeds[static_cast<std::size_t>(idx)]));
            for (int i = 0; i < 4; ++i)
                out[static_cast<std::size_t>(idx)][static_cast<std::size_t>(i)] = moments[static_cast<std::size_t>(i)];
        }
        return out;
    }

    // Bootstrap a list of quantiles given input non-exceedance probabilities [B][p].
    std::vector<std::vector<double>> quantiles(const std::vector<double>& probabilities) {
        auto owned = distributions();
        DistView view = to_view(owned);
        std::vector<std::vector<double>> out(view.size(),
                                             std::vector<double>(probabilities.size()));
        for (std::size_t i = 0; i < probabilities.size(); ++i)
            for (std::size_t idx = 0; idx < view.size(); ++idx)
                out[idx][i] = view[idx] != nullptr ? view[idx]->inverse_cdf(probabilities[i]) : kNaN;
        return out;
    }

    // Bootstrap a list of non-exceedance probabilities given input quantile values [B][q].
    std::vector<std::vector<double>> probabilities(const std::vector<double>& quantiles_in) {
        auto owned = distributions();
        DistView view = to_view(owned);
        std::vector<std::vector<double>> out(view.size(),
                                             std::vector<double>(quantiles_in.size()));
        for (std::size_t i = 0; i < quantiles_in.size(); ++i)
            for (std::size_t idx = 0; idx < view.size(); ++idx)
                out[idx][i] = view[idx] != nullptr ? view[idx]->cdf(quantiles_in[i]) : kNaN;
        return out;
    }

    // Bootstrap full uncertainty analysis results using the percentile method.
    distributions::UncertaintyAnalysisResults estimate(const std::vector<double>& probabilities,
                                                       double alpha = 0.1,
                                                       const DistView* distributions_in = nullptr,
                                                       bool recordParameterSets = true) {
        distributions::UncertaintyAnalysisResults results;
        results.parent_distribution = distribution_.get();

        // Mode curve.
        results.mode_curve.assign(probabilities.size(), 0.0);
        for (std::size_t i = 0; i < probabilities.size(); ++i)
            results.mode_curve[i] = distribution_->inverse_cdf(probabilities[i]);

        // Bootstrapped list of distributions (shared or generated).
        std::vector<DistPtr> owned;
        DistView view = resolve(distributions_in, owned);

        // Parameter sets.
        if (recordParameterSets) results.parameter_sets = parameter_sets(&view);

        // Confidence intervals.
        results.confidence_intervals = percentile_quantile_ci(probabilities, alpha, &view);

        // Log-spaced quantile grid (C# governs the exact bin-count rule).
        auto minMax = compute_min_max_quantiles(0.001, 1.0 - 1e-9, view);
        std::vector<double> quantiles;
        double shift = minMax[0] <= 0.0 ? std::abs(minMax[0]) + 1.0 : 0.0;
        double min = minMax[0] + shift;
        double max = minMax[1] + shift;
        int order = static_cast<int>(std::floor(std::log10(max) - std::log10(min)));
        int bins = std::max(200, std::min(1000, 100 * order));
        double delta = (std::log10(max) - std::log10(min)) / (bins - 1);
        double x = std::log10(min);
        quantiles.push_back(std::pow(10.0, x) - shift);
        for (int i = 1; i <= bins - 1; ++i) {
            x = std::log10(quantiles[static_cast<std::size_t>(i - 1)] + shift) + delta;
            quantiles.push_back(std::pow(10.0, x) - shift);
        }

        // Mean curve.
        results.mean_curve = expected_probabilities(quantiles, probabilities, &view);

        return results;
    }

    // Bootstrap the expected non-exceedance probabilities, interpolated to the desired
    // probabilities (the mean/predictive curve builder).
    std::vector<double> expected_probabilities(const std::vector<double>& quantiles,
                                               const std::vector<double>& probabilities,
                                               const DistView* distributions_in = nullptr) {
        std::vector<DistPtr> owned;
        DistView view = resolve(distributions_in, owned);

        std::vector<double> quants = quantiles;
        std::sort(quants.begin(), quants.end());
        std::vector<double> expected(quantiles.size());
        for (std::size_t i = 0; i < quantiles.size(); ++i) {
            double total = 0.0;
            for (std::size_t j = 0; j < view.size(); ++j)
                if (view[j] != nullptr) total += view[j]->cdf(quants[i]);
            expected[i] = total / static_cast<double>(view.size());
        }

        double minY = std::numeric_limits<double>::max();
        double maxY = std::numeric_limits<double>::lowest();
        std::vector<double> yVals{quantiles[0]};
        std::vector<double> xVals{expected[0]};
        for (std::size_t i = 1; i < quantiles.size(); ++i) {
            if (expected[i] > xVals.back()) {
                minY = std::min(minY, quantiles[i]);
                maxY = std::max(maxY, quantiles[i]);
                yVals.push_back(quantiles[i]);
                xVals.push_back(expected[i]);
            }
        }
        bool useLogTransform = minY > 0.0 && (std::log10(maxY) - std::log10(minY)) > 1.0;

        data::Linear linint(xVals, yVals);
        linint.x_transform = data::Transform::NormalZ;
        linint.y_transform = useLogTransform ? data::Transform::Logarithmic : data::Transform::None;
        return linint.interpolate(probabilities);
    }

    // Bootstrap the expected non-exceedance probabilities given the input quantile values (overload
    // without interpolation to probabilities).
    std::vector<double> expected_probabilities(const std::vector<double>& quantiles,
                                               const DistView* distributions_in) {
        std::vector<DistPtr> owned;
        DistView view = resolve(distributions_in, owned);
        std::vector<double> quants = quantiles;
        std::sort(quants.begin(), quants.end());
        std::vector<double> expected(quantiles.size());
        for (std::size_t i = 0; i < quantiles.size(); ++i) {
            double total = 0.0;
            for (std::size_t j = 0; j < view.size(); ++j)
                if (view[j] != nullptr) total += view[j]->cdf(quants[i]);
            expected[i] = total / static_cast<double>(view.size());
        }
        return expected;
    }

    // Returns the min and max quantiles from a bootstrap analysis {min, max}.
    std::array<double, 2> compute_min_max_quantiles(double minProbability, double maxProbability,
                                                    const DistView& distributions_in) {
        std::array<double, 2> output = {std::numeric_limits<double>::max(),
                                        std::numeric_limits<double>::lowest()};
        for (std::size_t j = 0; j < distributions_in.size(); ++j) {
            if (distributions_in[j] != nullptr) {
                double minX = distributions_in[j]->inverse_cdf(minProbability);
                double maxX = distributions_in[j]->inverse_cdf(maxProbability);
                if (minX < output[0]) output[0] = minX;
                if (maxX > output[1]) output[1] = maxX;
            }
        }
        return output;
    }

    // --- Confidence-interval methods ([p][2] = {lower, upper}) ---

    // Percentile method.
    std::vector<std::array<double, 2>> percentile_quantile_ci(
        const std::vector<double>& probabilities, double alpha = 0.1,
        const DistView* distributions_in = nullptr) {
        std::vector<DistPtr> owned;
        DistView view = resolve(distributions_in, owned);
        std::array<double, 2> CIs = {alpha / 2.0, 1.0 - alpha / 2.0};
        std::vector<std::array<double, 2>> output(probabilities.size());
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            std::vector<double> validValues = valid_quantiles(view, probabilities[i]);
            std::sort(validValues.begin(), validValues.end());
            for (int j = 0; j < 2; ++j)
                output[i][static_cast<std::size_t>(j)] =
                    data::percentile(validValues, CIs[static_cast<std::size_t>(j)], true);
        }
        return output;
    }

    // Bias-corrected percentile method.
    std::vector<std::array<double, 2>> bias_corrected_quantile_ci(
        const std::vector<double>& probabilities, double alpha = 0.1,
        const DistView* distributions_in = nullptr) {
        std::vector<double> populationXValues(probabilities.size());
        for (std::size_t i = 0; i < probabilities.size(); ++i)
            populationXValues[i] = distribution_->inverse_cdf(probabilities[i]);

        std::vector<DistPtr> owned;
        DistView view = resolve(distributions_in, owned);
        std::array<double, 2> CIs = {alpha / 2.0, 1.0 - alpha / 2.0};
        std::vector<std::array<double, 2>> output(probabilities.size());
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            double P0 = 0.0;
            std::vector<double> XValues(view.size());
            for (std::size_t idx = 0; idx < view.size(); ++idx) {
                XValues[idx] = view[idx] != nullptr ? view[idx]->inverse_cdf(probabilities[i]) : kNaN;
                // C# `XValues[idx] != double.NaN` is a no-op; effective test is value <= population.
                if (XValues[idx] <= populationXValues[i]) P0 += 1.0;
            }
            P0 = P0 / (static_cast<double>(view.size()) + 1.0);

            std::vector<double> validValues = filter_valid(XValues);
            std::sort(validValues.begin(), validValues.end());
            for (int j = 0; j < 2; ++j) {
                double Z0 = distributions::Normal::standard_z(P0);
                double Z = distributions::Normal::standard_z(CIs[static_cast<std::size_t>(j)]);
                double BC = distributions::Normal::standard_cdf(2.0 * Z0 + Z);
                output[i][static_cast<std::size_t>(j)] = data::percentile(validValues, BC, true);
            }
        }
        return output;
    }

    // Normal (standard) method with a cube-root transform.
    std::vector<std::array<double, 2>> normal_quantile_ci(
        const std::vector<double>& probabilities, double alpha = 0.1,
        const DistView* distributions_in = nullptr) {
        std::vector<double> populationXValues(probabilities.size());
        for (std::size_t i = 0; i < probabilities.size(); ++i)
            populationXValues[i] = std::pow(distribution_->inverse_cdf(probabilities[i]), 1.0 / 3.0);

        std::vector<DistPtr> owned;
        DistView view = resolve(distributions_in, owned);
        std::array<double, 2> CIs = {alpha / 2.0, 1.0 - alpha / 2.0};
        std::vector<std::array<double, 2>> output(probabilities.size());
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            std::vector<double> XValues(view.size());
            for (std::size_t idx = 0; idx < view.size(); ++idx)
                XValues[idx] = view[idx] != nullptr
                                   ? std::pow(view[idx]->inverse_cdf(probabilities[i]), 1.0 / 3.0)
                                   : kNaN;
            std::vector<double> validValues = filter_valid(XValues);
            double SE = data::standard_deviation(validValues);
            for (int j = 0; j < 2; ++j) {
                double Z = distributions::Normal::standard_z(CIs[static_cast<std::size_t>(j)]);
                output[i][static_cast<std::size_t>(j)] = std::pow(populationXValues[i] + SE * Z, 3.0);
            }
        }
        return output;
    }

    // Bias-corrected and accelerated (BCa) percentile method (jackknife acceleration constants).
    std::vector<std::array<double, 2>> bca_quantile_ci(const std::vector<double>& sampleData,
                                                       const std::vector<double>& probabilities,
                                                       double alpha = 0.1) {
        std::array<double, 2> CIs = {alpha / 2.0, 1.0 - alpha / 2.0};
        std::vector<std::array<double, 2>> output(probabilities.size());

        // Estimate distribution (mutates SampleSize and the stored distribution's parameters).
        sample_size_ = static_cast<int>(sampleData.size());
        estimation_->estimate(sampleData, estimation_method_);

        std::vector<double> populationXValues(probabilities.size());
        for (std::size_t i = 0; i < probabilities.size(); ++i)
            populationXValues[i] = distribution_->inverse_cdf(probabilities[i]);

        auto a = acceleration_constants(sampleData, probabilities, populationXValues);

        auto owned = distributions();
        DistView view = to_view(owned);
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            double P0 = 0.0;
            std::vector<double> XValues(static_cast<std::size_t>(replications_));
            for (int idx = 0; idx < replications_; ++idx) {
                XValues[static_cast<std::size_t>(idx)] =
                    view[static_cast<std::size_t>(idx)] != nullptr
                        ? view[static_cast<std::size_t>(idx)]->inverse_cdf(probabilities[i])
                        : kNaN;
                if (XValues[static_cast<std::size_t>(idx)] <= populationXValues[i]) P0 += 1.0;
            }
            P0 = (P0 + 1.0) / (static_cast<double>(replications_) + 1.0);

            std::vector<double> validValues = filter_valid(XValues);
            std::sort(validValues.begin(), validValues.end());
            for (int j = 0; j < 2; ++j) {
                double Z0 = distributions::Normal::standard_z(P0);
                double Z = distributions::Normal::standard_z(CIs[static_cast<std::size_t>(j)]);
                double num = Z0 + Z;
                double den = 1.0 - a[i] * (Z0 + Z);
                double BC = distributions::Normal::standard_cdf(Z0 + num / den);
                output[i][static_cast<std::size_t>(j)] = data::percentile(validValues, BC, true);
            }
        }
        return output;
    }

    // Bootstrap-t (Student-t) method with cube-root transform + inner bootstrap standard error.
    std::vector<std::array<double, 2>> bootstrap_t_quantile_ci(
        const std::vector<double>& probabilities, double alpha = 0.1) {
        std::size_t p = probabilities.size();
        std::vector<double> populationXValues(p);
        for (std::size_t i = 0; i < p; ++i)
            populationXValues[i] = std::pow(distribution_->inverse_cdf(probabilities[i]), 1.0 / 3.0);

        std::vector<std::vector<double>> xValues(static_cast<std::size_t>(replications_),
                                                 std::vector<double>(p));
        std::vector<std::vector<double>> studentT(static_cast<std::size_t>(replications_),
                                                  std::vector<double>(p));
        std::array<double, 2> CIs = {alpha / 2.0, 1.0 - alpha / 2.0};
        std::vector<std::array<double, 2>> output(p);

        sampling::MersenneTwister r(static_cast<std::uint32_t>(prng_seed_));
        auto seeds = utilities::next_integers(r, replications_);
        for (int i = 0; i < replications_; ++i) {
            std::size_t ui = static_cast<std::size_t>(i);
            try {
                auto newDistribution = distribution_->clone();
                auto sample =
                    newDistribution->generate_random_values(sample_size_, seeds[ui]);
                auto* est = dynamic_cast<IEstimation*>(newDistribution.get());
                est->estimate(sample, estimation_method_);

                std::vector<double> bootXValues(p);
                for (std::size_t j = 0; j < p; ++j)
                    bootXValues[j] = std::pow(newDistribution->inverse_cdf(probabilities[j]), 1.0 / 3.0);

                auto bootSE = bootstrap_standard_error(*newDistribution, probabilities, 300, seeds[ui]);
                for (std::size_t j = 0; j < p; ++j) {
                    xValues[ui][j] = bootXValues[j];
                    studentT[ui][j] = (populationXValues[j] - bootXValues[j]) / bootSE[j];
                }
            } catch (...) {
                for (std::size_t j = 0; j < p; ++j) {
                    xValues[ui][j] = kNaN;
                    studentT[ui][j] = kNaN;
                }
            }
        }

        for (std::size_t i = 0; i < p; ++i) {
            std::vector<double> XValues;
            std::vector<double> TValues;
            for (int k = 0; k < replications_; ++k) {
                std::size_t uk = static_cast<std::size_t>(k);
                if (!std::isnan(xValues[uk][i])) {
                    XValues.push_back(xValues[uk][i]);
                    TValues.push_back(studentT[uk][i]);
                }
            }
            double SE = data::standard_deviation(XValues);
            std::sort(TValues.begin(), TValues.end());
            for (int j = 0; j < 2; ++j) {
                double T = data::percentile(TValues, CIs[static_cast<std::size_t>(j)], true);
                output[i][static_cast<std::size_t>(j)] = std::pow(populationXValues[i] + SE * T, 3.0);
            }
        }
        return output;
    }

   private:
    // Estimates the acceleration constants for each probability (jackknife).
    std::vector<double> acceleration_constants(const std::vector<double>& sampleData,
                                               const std::vector<double>& probabilities,
                                               const std::vector<double>& thetaHats) {
        std::size_t N = sampleData.size();
        std::size_t p = probabilities.size();
        std::vector<double> I2(p, 0.0);
        std::vector<double> I3(p, 0.0);
        std::vector<double> a(p, 0.0);

        for (std::size_t idx = 0; idx < N; ++idx) {
            std::vector<double> jackSample;
            jackSample.reserve(N - 1);
            for (std::size_t k = 0; k < N; ++k)
                if (k != idx) jackSample.push_back(sampleData[k]);

            auto newDistribution = distribution_->clone();
            try {
                auto* est = dynamic_cast<IEstimation*>(newDistribution.get());
                est->estimate(jackSample, estimation_method_);
                for (std::size_t i = 0; i < p; ++i) {
                    double thetaJack = newDistribution->inverse_cdf(probabilities[i]);
                    I2[i] += std::pow(thetaHats[i] - thetaJack, 2.0);
                    I3[i] += std::pow(thetaHats[i] - thetaJack, 3.0);
                }
            } catch (...) {
                // MLE and certain L-moments methods can fail to find a solution.
            }
        }
        for (std::size_t i = 0; i < p; ++i)
            a[i] = I3[i] / (std::pow(I2[i], 1.5) * 6.0);
        return a;
    }

    // Estimates the standard error for each probability using the parametric bootstrap (300 reps).
    std::vector<double> bootstrap_standard_error(const UnivariateDistributionBase& parentDist,
                                                 const std::vector<double>& probabilities,
                                                 int replications = 300, int seed = 12345) {
        std::size_t p = probabilities.size();
        sampling::MersenneTwister r(static_cast<std::uint32_t>(seed));
        auto seeds = utilities::next_integers(r, replications);
        std::vector<std::vector<double>> xValues(static_cast<std::size_t>(replications),
                                                 std::vector<double>(p, kNaN));
        std::vector<double> se(p);
        for (int i = 0; i < replications; ++i) {
            std::size_t ui = static_cast<std::size_t>(i);
            try {
                auto bootDist = parentDist.clone();
                auto sample = bootDist->generate_random_values(sample_size_, seeds[ui]);
                auto* est = dynamic_cast<IEstimation*>(bootDist.get());
                est->estimate(sample, estimation_method_);
                for (std::size_t j = 0; j < p; ++j)
                    xValues[ui][j] = std::pow(bootDist->inverse_cdf(probabilities[j]), 1.0 / 3.0);
            } catch (...) {
                // On fail, leave the NaN-initialized column.
            }
        }
        for (std::size_t i = 0; i < p; ++i) {
            std::vector<double> col(static_cast<std::size_t>(replications));
            for (int k = 0; k < replications; ++k) col[static_cast<std::size_t>(k)] = xValues[static_cast<std::size_t>(k)][i];
            se[i] = data::standard_deviation(col);
        }
        return se;
    }

    // Build a non-owning view of an owning distribution vector.
    static DistView to_view(const std::vector<DistPtr>& owned) {
        DistView view;
        view.reserve(owned.size());
        for (const auto& d : owned) view.push_back(d.get());
        return view;
    }

    // Resolve the optional distributions argument: use the caller's view when provided, else
    // generate an owning set (stored in `owned` to keep it alive) and view that.
    DistView resolve(const DistView* distributions_in, std::vector<DistPtr>& owned) {
        if (distributions_in != nullptr) return *distributions_in;
        owned = distributions();
        return to_view(owned);
    }

    // Collect the inverse-CDF quantiles at `probability` across the view, dropping null dists.
    static std::vector<double> valid_quantiles(const DistView& view, double probability) {
        std::vector<double> x(view.size());
        for (std::size_t idx = 0; idx < view.size(); ++idx)
            x[idx] = view[idx] != nullptr ? view[idx]->inverse_cdf(probability) : kNaN;
        return filter_valid(x);
    }

    // Drop NaN entries (order-preserving), mirroring the C# valid-value compaction.
    static std::vector<double> filter_valid(const std::vector<double>& values) {
        std::vector<double> out;
        out.reserve(values.size());
        for (double v : values)
            if (!std::isnan(v)) out.push_back(v);
        return out;
    }

    DistPtr distribution_;
    IBootstrappable* bootstrappable_ = nullptr;
    IEstimation* estimation_ = nullptr;
    ParameterEstimationMethod estimation_method_ = ParameterEstimationMethod::MethodOfMoments;
    int sample_size_ = 0;
    int replications_ = 0;
    int prng_seed_ = 0;
    int retries_ = 20;
};

}  // namespace bestfit::numerics
