// ported from: Numerics/Sampling/Stratify.cs @ 2a0357a
//
// Stratifies the x-axis into a list of StratificationBin objects on a linear or log10 scale,
// per a StratificationOptions description. Feeds the profile-likelihood confidence-interval
// grids used by the BestFit MLE/MAP estimators.
//
// In scope: XValues(StratificationOptions, isLogarithmic) and XValues(list<StratificationOptions>,
// isLogarithmic) -- the two overloads the estimators need.
//
// Omitted (severable, tracked for a later Analyses/uncertainty phase): XToExceedanceProbability,
// XToProbability, Probabilities (both overloads), MultivariateProbabilities,
// ExceedanceProbabilities (both overloads), ProbabilityToX, ExceedanceProbabilityToX, and the
// ImportanceDistribution enum they depend on. These pull in importance-sampling distributions
// (Gumbel/Normal/Uniform), MultivariateNormal + Latin Hypercube Sampling, correlation/Cholesky
// machinery, and RNG seeding, none of which any in-scope estimator calls -- only later
// Analyses/uncertainty-quantification phases will.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "corehydro/numerics/sampling/stratification_bin.hpp"
#include "corehydro/numerics/sampling/stratification_options.hpp"

namespace corehydro::numerics::sampling {

class Stratify {
   public:
    // Returns a list of stratified x value bins.
    // Returns empty if options.is_valid() is false or options.is_probability() is true.
    static std::vector<StratificationBin> XValues(const StratificationOptions& options,
                                                   bool is_logarithmic = false) {
        std::vector<StratificationBin> bins;
        if (!options.is_valid() || options.is_probability()) return bins;

        double delta, offset, min, max, xl, xu;

        if (is_logarithmic) {
            // first determine the offset
            offset = options.lower_bound() <= 0.0 ? 1.0 - options.lower_bound() : 0.0;
            // get delta
            min = std::log10(options.lower_bound() + offset);
            max = std::log10(options.upper_bound() + offset);
            delta = (max - min) / options.number_of_bins();
            // stratify first bin
            xl = std::pow(10.0, min) - offset;
            xu = std::pow(10.0, std::log10(xl + offset) + delta) - offset;
            bins.emplace_back(xl, xu);
            // stratify remaining bins
            for (int i = 1; i < options.number_of_bins(); ++i) {
                xl = xu;
                xu = std::pow(10.0, std::log10(xl + offset) + delta) - offset;
                bins.emplace_back(xl, xu);
            }
        } else {
            // get delta
            delta = (options.upper_bound() - options.lower_bound()) / options.number_of_bins();
            // stratify first bin
            xl = options.lower_bound();
            xu = xl + delta;
            bins.emplace_back(xl, xu);
            // stratify remaining bins
            for (int i = 1; i < options.number_of_bins(); ++i) {
                xl = xu;
                xu = xl + delta;
                bins.emplace_back(xl, xu);
            }
        }

        return bins;
    }

    // Returns a list of stratified x value bins for a list of stratification options, sorted
    // ascending by upper_bound before concatenating each option's XValues().
    static std::vector<StratificationBin> XValues(std::vector<StratificationOptions> options,
                                                   bool is_logarithmic = false) {
        if (options.empty()) return {};

        std::stable_sort(options.begin(), options.end(),
                          [](const StratificationOptions& a, const StratificationOptions& b) {
                              return a.upper_bound() < b.upper_bound();
                          });

        std::vector<StratificationBin> bins;
        for (const auto& opt : options) {
            auto opt_bins = XValues(opt, is_logarithmic);
            bins.insert(bins.end(), opt_bins.begin(), opt_bins.end());
        }
        return bins;
    }
};

}  // namespace corehydro::numerics::sampling
