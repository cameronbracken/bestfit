// bestfit ADDITION -- no upstream C# counterpart.
//
// A small, CLOSED registry mapping a named bootstrap "model family" to a fully-configured
// `Bootstrap<std::vector<double>>` instance whose delegates match a specific upstream
// `Test_Bootstrap.cs` test construction VERBATIM. Mirrors `mcmc/model_registry.hpp`'s role for
// MCMC fixtures: the C# equivalent is inline construction code living only inside
// `Test_Bootstrap.cs`'s private `CreateNormalBootstrap()` helper (used by every Percentile/
// Normal/BiasCorrected/BootstrapT case) and `Test_BCaCI()` (the BCa case), not in the library
// itself. This gives the fixture runners and the oracle emitter a single, shared place to
// build the exact same model the fixtures reference by name.
//
// Registry (one entry): "normal_quantiles" -- parametric bootstrap of a `Normal(mu, sigma)`
// distribution fitted by Method of Moments, evaluated at a fixed set of quantile
// probabilities:
//   - ResampleFunction: draws `resample_size` values from `Normal(ps[0], ps[1])` via
//     `generate_random_values(resample_size, rng.next())` -- `resample_size` is a FIXED
//     closure-captured value (`_sampleSize`/`sampleData.Length` in C#), not derived from the
//     `data` argument, matching both `CreateNormalBootstrap` and `Test_BCaCI` ignoring their
//     own `data` parameter.
//   - FitFunction: Method-of-Moments `Normal` fit; throws if the fitted parameters are
//     invalid (mirrors `if (!d.ParametersValid) throw ...`).
//   - StatisticFunction: `Normal(ps[0], ps[1]).InverseCDF(p)` for each `p` in `probabilities`.
//   - `sample_data`, when non-empty, supplies a REAL sample (`Test_BCaCI`'s literal 100-value
//     array): `mu`/`sigma` are then IGNORED and instead fit from `sample_data` via Method of
//     Moments (mirrors `Test_BCaCI`'s own `((IEstimation)dist).Estimate(sampleData, ...)` call
//     before constructing `Bootstrap`), `resample_size` becomes `sample_data.size()`, and
//     JackknifeFunction/SampleSizeFunction are wired to leave-one-out over `sample_data`
//     (required by the BCa CI method -- `ComputeAccelerationConstants` needs both). When
//     `sample_data` is empty, `original_data` is unused (matches `CreateNormalBootstrap`'s
//     `new Bootstrap<double[]>(null, parms)`) and JackknifeFunction/SampleSizeFunction are
//     left unset -- BCa is not requestable on that configuration, mirroring the C# test's own
//     `Test_BCa_RequiresJackknifeFunction`.
#pragma once
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "bestfit/numerics/distributions/base/parameter_estimation_method.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "bestfit/numerics/sampling/bootstrap/bootstrap.hpp"

namespace bestfit::numerics::sampling {

// Builds the "normal_quantiles" bootstrap model. See the file header for the full
// `sample_data` semantics.
inline Bootstrap<std::vector<double>> build_normal_quantiles_bootstrap(
    double mu, double sigma, int sample_size, const std::vector<double>& probabilities,
    const std::vector<double>& sample_data = {}) {
    double fit_mu = mu;
    double fit_sigma = sigma;
    int resample_size = sample_size;
    std::vector<double> original_data;

    if (!sample_data.empty()) {
        distributions::Normal probe;
        probe.estimate(sample_data, distributions::ParameterEstimationMethod::MethodOfMoments);
        fit_mu = probe.mu();
        fit_sigma = probe.sigma();
        resample_size = static_cast<int>(sample_data.size());
        original_data = sample_data;
    }

    math::optimization::ParameterSet parms({fit_mu, fit_sigma}, std::numeric_limits<double>::quiet_NaN());
    Bootstrap<std::vector<double>> boot(original_data, parms);

    boot.resample_function = [resample_size](const std::vector<double>& /*data*/,
                                              const math::optimization::ParameterSet& ps, MersenneTwister& rng) {
        distributions::Normal d(ps.values[0], ps.values[1]);
        return d.generate_random_values(resample_size, rng.next());
    };

    boot.fit_function = [](const std::vector<double>& sample) {
        distributions::Normal d;
        d.estimate(sample, distributions::ParameterEstimationMethod::MethodOfMoments);
        if (!d.parameters_valid()) throw std::runtime_error("Invalid parameters.");
        return math::optimization::ParameterSet(d.get_parameters(), std::numeric_limits<double>::quiet_NaN());
    };

    boot.statistic_function = [probabilities](const math::optimization::ParameterSet& ps) {
        distributions::Normal d(ps.values[0], ps.values[1]);
        std::vector<double> result(probabilities.size());
        for (std::size_t i = 0; i < probabilities.size(); ++i) result[i] = d.inverse_cdf(probabilities[i]);
        return result;
    };

    if (!sample_data.empty()) {
        boot.jackknife_function = [](const std::vector<double>& data, int idx) {
            std::vector<double> out;
            out.reserve(data.size() > 0 ? data.size() - 1 : 0);
            for (int i = 0; i < static_cast<int>(data.size()); ++i)
                if (i != idx) out.push_back(data[static_cast<std::size_t>(i)]);
            return out;
        };
        boot.sample_size_function = [](const std::vector<double>& data) { return static_cast<int>(data.size()); };
    }

    return boot;
}

// Builds a named bootstrap model. Closed registry -- currently one entry.
inline Bootstrap<std::vector<double>> build_bootstrap_model(const std::string& name, double mu, double sigma,
                                                              int sample_size,
                                                              const std::vector<double>& probabilities,
                                                              const std::vector<double>& sample_data = {}) {
    if (name == "normal_quantiles")
        return build_normal_quantiles_bootstrap(mu, sigma, sample_size, probabilities, sample_data);
    throw std::invalid_argument("unknown bootstrap model registry entry: " + name);
}

}  // namespace bestfit::numerics::sampling
