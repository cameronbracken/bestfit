// ported from: RMC-BestFit/src/RMC.BestFit/Models/UnivariateDistribution/
//              Bulletin17CDistribution.cs @ c2e6192
//
// v2.0.0 (upstream-sync Task 18): re-verified against c2e6192 -- `git diff fc28c0c..v2.0.0`
// over this region (MomentConditions through QuantileVariance) is mojibake-only doc-comment
// re-encoding (stray UTF-8 math glyphs like mu/sigma/times collapsed to '?'/'�' by a
// non-UTF-8-safe commit somewhere in the file's history) plus one stale parameter-name doc
// fix (SetPenaltyFunction's `seed` -> `prng`, in the companion class header, not this
// file). No method body in this slice changed; the provenance bump records re-verification,
// not a port.
//
// The Bulletin17CDistribution MOMENT MACHINERY (Task B10): the out-of-line bodies of the
// methods declared in bulletin17c_distribution.hpp -- MomentConditions (C# 1043),
// RepairErrors (1223), Mu456_Pearson3_FromSigmaGamma (1280),
// ConditionalMomentsForUncertainData (1325), MomentConditionsForUncertainData (1418),
// UpdateMomentMeanCovariance (1490), CensoringAsymmetryScore (1691),
// WeightedErrorDirectionScore (1901), WeightedErrorDirectionScoreFromLinked (2090),
// PointwiseMomentConditionsImpl (2128), QuantileGradient (2315), QuantileVariance (2388).
//
// This is an .ipp-style companion split out ONLY to keep the class header under the repo
// file-size cap; the class stays one class mirroring the one C# file. It is included at
// the bottom of bulletin17c_distribution.hpp; including this file directly also works
// (it pulls the class header in first, and the #pragma once pair breaks the cycle).
//
// Port notes for THIS slice:
// - `parameters = _linkController.InverseLink(parameters)` on entry: the parameter is
//   taken BY VALUE and reassigned, mirroring the C# array-reference reassignment.
// - `model.ValidateParameters(parameters, false)` + `model.SetParameters(parameters)`:
//   the ported set_parameters performs the same non-throwing validation and records it in
//   parameters_valid(), so the pair ports as set_parameters + a parameters_valid() check
//   on the throwaway working model (the UnivariateDistributionModel probe precedent).
// - The C# log-space remap (LP3 -> PT3 with isLog10, LogNormal -> Normal with isLog10,
//   else Distribution.Clone()) is written INLINE in each method, exactly as the C#
//   repeats it (the brief's keep-it-inline instruction).
// - Tools.Log10 -> numerics::clamped_log10 (the established Data.Log10Value mapping);
//   Tools.DoubleMachineEpsilon -> numerics::kDoubleMachineEpsilon; double.MaxValue ->
//   std::numeric_limits<double>::max().
// - `conditionalMoments.Fill(0)` fills the reusable buffer in place (std::fill); the
//   `conditionalMoments = model.ConditionalMoments(...)` reassignment can grow the buffer
//   to the base method's 4 entries, exactly as the C# array reference is replaced -- the
//   extra entries are never read (all consumers gate on q and c.Length, as upstream).
// - The C# local functions (ExpectedError, AddMeasurementErrorSecondMoment,
//   AccumulatePull, AccumulateDirection, StoreG, MomentError) port as lambdas in the same
//   positions.
// - CensoringAsymmetryScore's `errs[j] >= 0 -> negPull` / `else -> posPull` accumulation
//   is transcribed VERBATIM (C# 1766-1772), including the sign convention.
// - `((IStandardError)model).QuantileGradient(...)` (C# 2355): no IStandardError mixin
//   exists in the port (Phase 4 decision), so the cast ports as a static per-type
//   dispatch over the reachable families (Exponential, Gamma, Normal -- LogNormal is
//   remapped to Normal before the cast); an unreachable type throws std::runtime_error
//   where the failed C# cast would raise InvalidCastException.
// - C# `double[,]` (QuantileVariance's covarianceMatrix) -> the row-major Matrix2D, the
//   JacobianFunction convention.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/models/univariate_distribution/bulletin17c_distribution.hpp"
#include "corehydro/numerics/math/integration/integration.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models {

// Computes the GMM moment condition vector G and its covariance matrix S (C# line 1043).
inline estimation::MomentConditionResult Bulletin17CDistribution::moment_conditions(
    std::vector<double> parameters) const {
    using numerics::math::linalg::Matrix;
    using numerics::math::linalg::Vector;

    parameters = link_controller_.inverse_link(parameters);

    int q = number_of_parameters();
    int n = data_frame().total_record_length();

    // Create result holders
    Vector mean(q);
    Matrix covariance(q);

    // Bessel small-sample correction factors for non-outlier exact data
    int ns = 0;
    for (std::size_t k = 0; k < data_frame().exact_series().count(); k++)
        if (!data_frame().exact_series()[k].is_low_outlier()) ns++;
    double c2 = ns >= 2 ? ns / static_cast<double>(ns - 1) : 1.0;
    double c3 = ns >= 3 ? static_cast<double>(ns * ns) / ((ns - 1) * (ns - 2)) : 1.0;

    // Configure distribution model and log-space flag
    std::unique_ptr<DistributionBase> model;
    bool is_log10 = false;

    // Disable model-based covariance when low outliers are present, since the
    // unconditional mu4-mu6 formulas don't account for the truncated contribution
    bool use_model_covariance = data_frame().number_of_low_outliers() == 0;

    if (distribution_type() == DistributionType::LogPearsonTypeIII) {
        model = create_distribution(DistributionType::PearsonTypeIII);
        is_log10 = true;
    } else if (distribution_type() == DistributionType::LogNormal) {
        model = create_distribution(DistributionType::Normal);
        is_log10 = true;
    } else {
        model = distribution_->clone();
    }

    // Validate parameters without throwing -- this is called thousands of times during
    // GMM optimization and exception overhead is significant (C# ValidateParameters
    // (parameters, false) + SetParameters; see the file header note).
    model->set_parameters(parameters);
    if (!model->parameters_valid()) {
        // Invalid parameters -- signal the optimizer to reject this parameter set
        for (int i = 0; i < q; i++) mean[i] = std::numeric_limits<double>::max();
        return {mean, covariance};
    }

    // Integration bounds from distribution support
    double min = model->inverse_cdf(numerics::kDoubleMachineEpsilon);
    double max = model->inverse_cdf(1 - numerics::kDoubleMachineEpsilon);

    // Unconditional central moments of the fitted distribution
    double mu = model->mean();
    double sigma = model->standard_deviation();
    double sigma2 = model->variance();
    double skewness = model->skewness();

    // Guard against non-finite moments from extreme parameter values during optimization
    if (!numerics::is_finite(mu) || !numerics::is_finite(sigma2) || sigma <= 0) {
        for (int i = 0; i < q; i++) mean[i] = std::numeric_limits<double>::max();
        return {mean, covariance};
    }

    // Store unconditional moments: [mu, sigma^2, mu3]
    std::vector<double> unconditional_moments(static_cast<std::size_t>(q));
    unconditional_moments[0] = mu;
    if (q >= 2) unconditional_moments[1] = sigma2;
    if (q >= 3) unconditional_moments[2] = skewness * sigma * sigma * sigma;

    // Reusable buffer for conditional moments
    std::vector<double> conditional_moments(static_cast<std::size_t>(q));

    // Low outlier moments -- treated as left-censored below the low outlier threshold
    if (data_frame().number_of_low_outliers() > 0) {
        double lower = min;
        double upper = std::min(
            max, is_log10 ? numerics::clamped_log10(data_frame().low_outlier_threshold())
                          : data_frame().low_outlier_threshold());
        if (lower >= upper)
            std::fill(conditional_moments.begin(), conditional_moments.end(), 0.0);
        else
            conditional_moments = model->conditional_moments(lower, upper);
        update_moment_mean_covariance(conditional_moments, unconditional_moments, mean,
                                      covariance, data_frame().number_of_low_outliers(), true,
                                      false);
    }

    // Exact data -- non-outlier observed values with Bessel corrections
    for (std::size_t k = 0; k < data_frame().exact_series().count(); k++) {
        const ExactData& data = data_frame().exact_series()[k];
        if (!data.is_low_outlier()) {
            double value = is_log10 ? data.log10_value() : data.value();
            double d = value - unconditional_moments[0];
            double d2 = d * d;
            conditional_moments[0] = value;
            if (q >= 2) conditional_moments[1] = c2 * d2;
            if (q >= 3) conditional_moments[2] = c3 * d2 * d;
            update_moment_mean_covariance(conditional_moments, unconditional_moments, mean,
                                          covariance, 1, false, use_model_covariance, sigma,
                                          skewness);
        }
    }

    // Uncertain data -- integrate over measurement error distribution
    for (std::size_t k = 0; k < data_frame().uncertain_series().count(); k++) {
        const UncertainData& data = data_frame().uncertain_series()[k];
        Matrix measurement_error_second_moment(q);
        std::vector<double> uncertain_moments = conditional_moments_for_uncertain_data(
            unconditional_moments, data, is_log10, &measurement_error_second_moment);

        // Law of total variance for uncertain observations: keep the usual
        // hydrologic/process covariance contribution, then add the ME-induced second
        // moment. The row update handles whether the conditional mean outer product has
        // already been included by the selected covariance path.
        update_moment_mean_covariance(uncertain_moments, unconditional_moments, mean,
                                      covariance, 1, false, use_model_covariance, sigma,
                                      skewness, &measurement_error_second_moment);
    }

    // Interval data -- conditional moments over [L, U]
    for (std::size_t k = 0; k < data_frame().interval_series().count(); k++) {
        const IntervalData& data = data_frame().interval_series()[k];
        double lower = std::max(min, is_log10 ? data.log10_lower_value() : data.lower_value());
        double upper = std::min(max, is_log10 ? data.log10_upper_value() : data.upper_value());
        if (lower >= upper)
            std::fill(conditional_moments.begin(), conditional_moments.end(), 0.0);
        else
            conditional_moments = model->conditional_moments(lower, upper);
        update_moment_mean_covariance(conditional_moments, unconditional_moments, mean,
                                      covariance, 1, true, false);
    }

    // Threshold data -- left-censored (Y < threshold) and right-censored (Y >= threshold)
    for (std::size_t k = 0; k < data_frame().threshold_series().count(); k++) {
        const ThresholdData& data = data_frame().threshold_series()[k];
        // Left censored
        if (data.number_below() > 0) {
            double lower = min;
            double upper = std::min(max, is_log10 ? data.log10_value() : data.value());
            if (lower >= upper)
                std::fill(conditional_moments.begin(), conditional_moments.end(), 0.0);
            else
                conditional_moments = model->conditional_moments(lower, upper);
            update_moment_mean_covariance(conditional_moments, unconditional_moments, mean,
                                          covariance, data.number_below(), true, false);
        }
        // Right censored
        if (data.number_above() > 0) {
            double lower = std::min(max, is_log10 ? data.log10_value() : data.value());
            double upper = max;
            if (lower >= upper)
                std::fill(conditional_moments.begin(), conditional_moments.end(), 0.0);
            else
                conditional_moments = model->conditional_moments(lower, upper);
            update_moment_mean_covariance(conditional_moments, unconditional_moments, mean,
                                          covariance, data.number_above(), true, false);
        }
    }

    // Detect non-finite values and signal failure to the optimizer
    repair_errors(mean, covariance);

    // Normalize to sample means and form covariance: Cov(g) = E[gg'] - E[g]E[g]'
    mean = mean / static_cast<double>(n);
    covariance = covariance / static_cast<double>(n);
    covariance = covariance - Matrix::outer(mean, mean);
    return {mean, covariance};
}

// Detects non-finite values in the moment condition vector or covariance accumulator and
// replaces them with the optimizer-rejection sentinels (C# line 1223).
inline void Bulletin17CDistribution::repair_errors(
    numerics::math::linalg::Vector& errors, numerics::math::linalg::Matrix& covariance) const {
    bool has_bad_value = false;

    // Check the moment condition vector
    for (int i = 0; i < errors.length(); i++) {
        if (!numerics::is_finite(errors[i])) {
            has_bad_value = true;
            break;
        }
    }

    // Check the covariance matrix
    if (!has_bad_value) {
        int rows = covariance.number_of_rows();
        int cols = covariance.number_of_columns();
        for (int i = 0; i < rows && !has_bad_value; i++)
            for (int j = 0; j < cols && !has_bad_value; j++)
                if (!numerics::is_finite(covariance(i, j))) has_bad_value = true;
    }

    if (has_bad_value) {
        for (int i = 0; i < errors.length(); i++)
            errors[i] = std::numeric_limits<double>::max();
        for (int i = 0; i < covariance.number_of_rows(); i++)
            for (int j = 0; j < covariance.number_of_columns(); j++) covariance(i, j) = 0.0;
    }
}

// Computes the 4th-6th central moments of a Pearson Type III distribution from sigma and
// skew gamma (C# line 1280).
inline void Bulletin17CDistribution::mu456_pearson3_from_sigma_gamma(
    double sigma, double gamma, double& mu4, double& mu5, double& mu6) {
    double s2 = sigma * sigma;
    double s4 = s2 * s2;
    double s5 = s4 * sigma;
    double s6 = s4 * s2;
    double g2 = gamma * gamma;
    double g4 = g2 * g2;

    mu4 = s4 * (3.0 + 1.5 * g2);
    mu5 = s5 * gamma * (10.0 + 3.0 * g2);
    mu6 = s6 * (15.0 + 32.5 * g2 + 7.5 * g4);  // 65/2 = 32.5; 15/2 = 7.5
}

// Computes conditional central moments for an uncertain data observation by integrating
// over the measurement error distribution with 20-point Gauss-Legendre quadrature
// (C# line 1325).
inline std::vector<double> Bulletin17CDistribution::conditional_moments_for_uncertain_data(
    const std::vector<double>& unconditional_moments, const UncertainData& uncertain_data,
    bool is_log10, numerics::math::linalg::Matrix* measurement_error_second_moment) const {
    using numerics::math::integration::Integration;

    const DistributionBase& dist = uncertain_data.distribution();
    double a = dist.inverse_cdf(1E-8);
    double b = dist.inverse_cdf(1 - 1E-8);
    double mass = dist.cdf(b) - dist.cdf(a);

    int q = static_cast<int>(unconditional_moments.size());
    if (a >= b) return std::vector<double>(static_cast<std::size_t>(q), 0.0);

    double mu = unconditional_moments[0];
    double mu2 = q >= 2 ? unconditional_moments[1] : 0.0;
    double mu3 = q >= 3 ? unconditional_moments[2] : 0.0;
    std::vector<double> moments(static_cast<std::size_t>(q), 0.0);

    // E[Y | error_dist] normalized by retained ME probability mass.
    moments[0] = Integration::gauss_legendre20(
                     [&](double x) {
                         return (is_log10 ? numerics::clamped_log10(x) : x) * dist.pdf(x);
                     },
                     a, b) /
                 mass;

    // E[(Y - mu)^2 | error_dist]
    if (q >= 2) {
        moments[1] = Integration::gauss_legendre20(
                         [&](double x) {
                             double v = is_log10 ? numerics::clamped_log10(x) : x;
                             double d = v - mu;
                             return d * d * dist.pdf(x);
                         },
                         a, b) /
                     mass;
    }

    // E[(Y - mu)^3 | error_dist]
    if (q >= 3) {
        moments[2] = Integration::gauss_legendre20(
                         [&](double x) {
                             double v = is_log10 ? numerics::clamped_log10(x) : x;
                             double d = v - mu;
                             return d * d * d * dist.pdf(x);
                         },
                         a, b) /
                     mass;
    }

    if (measurement_error_second_moment != nullptr) {
        // For uncertain rows only, integrate E_ME[gg'] so the GMM S matrix carries
        // measurement-error variance. UpdateMomentMeanCovariance adds this term by the
        // law of total variance without replacing the hydrologic/process term.
        auto moment_error = [&](double x, int index) {
            double v = is_log10 ? numerics::clamped_log10(x) : x;
            double d = v - mu;

            if (index == 0) return d;
            if (index == 1) return d * d - mu2;
            return d * d * d - mu3;
        };

        for (int i = 0; i < q; i++) {
            for (int j = i; j < q; j++) {
                double value = Integration::gauss_legendre20(
                                   [&](double x) {
                                       double gi = moment_error(x, i);
                                       double gj = moment_error(x, j);
                                       return gi * gj * dist.pdf(x);
                                   },
                                   a, b) /
                               mass;

                (*measurement_error_second_moment)(i, j) = value;
                (*measurement_error_second_moment)(j, i) = value;
            }
        }
    }

    return moments;
}

// Computes moment condition errors (g-vector) for an uncertain data observation
// (C# line 1418).
inline std::vector<double> Bulletin17CDistribution::moment_conditions_for_uncertain_data(
    const std::vector<double>& unconditional_moments, const UncertainData& uncertain_data,
    bool is_log10) const {
    std::vector<double> moments =
        conditional_moments_for_uncertain_data(unconditional_moments, uncertain_data, is_log10);
    std::vector<double> errors(moments.size());
    for (std::size_t i = 0; i < moments.size(); i++)
        errors[i] = moments[i] - unconditional_moments[i];
    return errors;
}

// Accumulates running sums for the moment condition mean vector and second-moment matrix
// (C# line 1490).
inline void Bulletin17CDistribution::update_moment_mean_covariance(
    const std::vector<double>& c, const std::vector<double>& m,
    numerics::math::linalg::Vector& mean, numerics::math::linalg::Matrix& covariance, double w,
    bool is_censored, bool use_model_covariance, double model_sigma, double model_gamma,
    const numerics::math::linalg::Matrix* measurement_error_second_moment) const {
    int q = mean.length();

    // Unconditional low-order (about mu)
    double mu = m[0];
    double mu2 = m[1];
    double mu3 = (q >= 3 && static_cast<int>(m.size()) >= 3) ? m[2] : 0.0;

    // Use model-provided sigma/gamma for higher central moments (mu4-mu6). For q<3
    // distributions (Exponential, Gamma), the moments array m[] doesn't include mu3, so
    // gamma cannot be derived from m[] alone. The caller passes the model's
    // StandardDeviation and Skewness directly.
    double sigma = model_sigma;
    double gamma = model_gamma;

    // "Conditional" pieces provided for this row (about mu for indices >= 1)
    double c1 = c[0] - mu;  // E[(Y-mu)|A] or (Y-mu) for exact row
    double c2 = (static_cast<int>(c.size()) >= 2) ? c[1] : 0.0;  // E[(Y-mu)^2|A] or (Y-mu)^2
    double c3 = (q >= 3 && static_cast<int>(c.size()) >= 3) ? c[2]
                                                            : 0.0;  // E[(Y-mu)^3|A] or (Y-mu)^3

    // E[g|A]
    double eg1 = c1;
    double eg2 = c2 - mu2;
    double eg3 = (q >= 3) ? (c3 - mu3) : 0.0;

    // --- accumulate sum_g (RAW) ---
    mean[0] += w * eg1;
    mean[1] += w * eg2;
    if (q >= 3) mean[2] += w * eg3;

    auto expected_error = [&](int index) {
        if (index == 0) return eg1;
        if (index == 1) return eg2;
        return eg3;
    };

    auto add_measurement_error_second_moment = [&](bool conditional_mean_outer_already_included) {
        if (measurement_error_second_moment == nullptr) return;

        for (int i = 0; i < q; i++) {
            for (int j = 0; j < q; j++) {
                double value = (*measurement_error_second_moment)(i, j);

                // When the fallback path already added E_ME[g]E_ME[g]', subtract that
                // piece here so the added term is only Var_ME(g). When the analytic model
                // path is used, no conditional mean outer product has been added, so the
                // raw E_ME[gg'] term belongs in the accumulator.
                if (conditional_mean_outer_already_included)
                    value -= expected_error(i) * expected_error(j);

                covariance(i, j) += w * value;
            }
        }
    };

    // By default, use the outer-product approximation (works for censored or unsupported
    // families)
    bool used_model_for_this_row = false;

    if (!is_censored && use_model_covariance) {
        // Try to use model mu4..mu6 to get the proper "inflated" variance for EXACT rows.
        double mu4 = 0.0, mu5 = 0.0, mu6 = 0.0;

        switch (distribution_type()) {
            case DistributionType::Exponential:
            case DistributionType::GammaDistribution:
            case DistributionType::PearsonTypeIII:
            case DistributionType::LogPearsonTypeIII:
                // Pearson Type III / Gamma family uses (sigma, gamma). Exponential is the
                // gamma=2 special case.
                mu456_pearson3_from_sigma_gamma(sigma, gamma, mu4, mu5, mu6);
                used_model_for_this_row = true;
                break;

            // If you want to extend to other families, add cases here, e.g.:
            case DistributionType::Normal:
            case DistributionType::LogNormal: {
                // mu4 = 3 s^4, mu5 = 0, mu6 = 15 s^6
                double s2 = mu2;
                double s4 = s2 * s2;
                double s6 = s4 * s2;
                mu4 = 3.0 * s4;
                mu5 = 0.0;
                mu6 = 15.0 * s6;
                used_model_for_this_row = true;
                break;
            }

            default:
                used_model_for_this_row = false;  // fall back to outer product
                break;
        }

        if (used_model_for_this_row) {
            // Per-row E[gg'] using model central moments (unconditional)
            // g = [ g1,      g2,               g3             ]
            //   = [ (Y-mu),  (Y-mu)^2 - mu2,   (Y-mu)^3 - mu3 ]
            //
            // Use model mu3 = gamma sigma^3 instead of mu3 from m[2], which is 0 for
            // q<3 distributions (Exponential, Gamma) since m[] has only q entries.
            double model_mu3 = gamma * sigma * sigma * sigma;

            double m11 = mu2;
            double m12 = model_mu3;
            double m22 = mu4 - mu2 * mu2;

            double m13 = 0.0, m23 = 0.0, m33 = 0.0;
            if (q >= 3) {
                m13 = mu4;                        // E[(Y-mu)((Y-mu)^3-mu3)] = mu4
                m23 = mu5 - mu2 * model_mu3;      // E[((Y-mu)^2-mu2)((Y-mu)^3-mu3)]
                m33 = mu6 - model_mu3 * model_mu3;  // E[((Y-mu)^3-mu3)^2] = mu6 - mu3^2
            }

            covariance(0, 0) += w * m11;
            covariance(0, 1) += w * m12;
            covariance(1, 0) += w * m12;
            covariance(1, 1) += w * m22;

            if (q >= 3) {
                covariance(0, 2) += w * m13;
                covariance(2, 0) += w * m13;
                covariance(1, 2) += w * m23;
                covariance(2, 1) += w * m23;
                covariance(2, 2) += w * m33;
            }

            add_measurement_error_second_moment(false);
            return;  // IMPORTANT: we've completed the model-based path
        }
    }

    // --- Outer-product fallback (censored rows or unsupported families) ---
    {
        double g1 = eg1, g2 = eg2, g3 = eg3;

        double m11 = g1 * g1;
        double m12 = g1 * g2;
        double m22 = g2 * g2;

        double m13 = 0.0, m23 = 0.0, m33 = 0.0;
        if (q >= 3) {
            m13 = g1 * g3;
            m23 = g2 * g3;
            m33 = g3 * g3;
        }

        covariance(0, 0) += w * m11;
        covariance(0, 1) += w * m12;
        covariance(1, 0) += w * m12;
        covariance(1, 1) += w * m22;

        if (q >= 3) {
            covariance(0, 2) += w * m13;
            covariance(2, 0) += w * m13;
            covariance(1, 2) += w * m23;
            covariance(2, 1) += w * m23;
            covariance(2, 2) += w * m33;
        }

        add_measurement_error_second_moment(true);
    }
}

// Computes a directional censoring asymmetry score S in [-1, 1] for each moment condition
// (C# line 1691).
inline std::vector<double> Bulletin17CDistribution::censoring_asymmetry_score(
    std::vector<double> parameters) const {
    parameters = link_controller_.inverse_link(parameters);

    int q = number_of_parameters();

    std::vector<double> pos_pull(static_cast<std::size_t>(q), 0.0);  // right-tail pull
    std::vector<double> neg_pull(static_cast<std::size_t>(q), 0.0);  // left-tail pull
    std::vector<double> score(static_cast<std::size_t>(q), 0.0);

    // Bessel small-sample correction factors for non-outlier exact data
    int ns = 0;
    for (std::size_t k = 0; k < data_frame().exact_series().count(); k++)
        if (!data_frame().exact_series()[k].is_low_outlier()) ns++;
    double c2 = ns >= 2 ? ns / static_cast<double>(ns - 1) : 1.0;
    double c3 = ns >= 3 ? static_cast<double>(ns * ns) / ((ns - 1) * (ns - 2)) : 1.0;

    // Configure distribution model and log-space flag
    std::unique_ptr<DistributionBase> model;
    bool is_log10 = false;

    if (distribution_type() == DistributionType::LogPearsonTypeIII) {
        model = create_distribution(DistributionType::PearsonTypeIII);
        is_log10 = true;
    } else if (distribution_type() == DistributionType::LogNormal) {
        model = create_distribution(DistributionType::Normal);
        is_log10 = true;
    } else {
        model = distribution_->clone();
    }

    // Validate parameters without throwing -- called frequently during optimization
    model->set_parameters(parameters);
    if (!model->parameters_valid()) {
        std::fill(score.begin(), score.end(), std::numeric_limits<double>::quiet_NaN());
        return score;
    }

    // Integration bounds from distribution support
    double min = model->inverse_cdf(numerics::kDoubleMachineEpsilon);
    double max = model->inverse_cdf(1 - numerics::kDoubleMachineEpsilon);

    // Unconditional central moments of the fitted distribution
    double mu = model->mean();
    double sigma = model->standard_deviation();
    double sigma2 = model->variance();
    double skewness = model->skewness();

    // Guard against non-finite moments
    if (!numerics::is_finite(mu) || !numerics::is_finite(sigma2) || sigma <= 0) {
        std::fill(score.begin(), score.end(), std::numeric_limits<double>::quiet_NaN());
        return score;
    }

    // Store unconditional moments: [mu, sigma^2, mu3]
    std::vector<double> unconditional_moments(static_cast<std::size_t>(q));
    unconditional_moments[0] = mu;
    if (q >= 2) unconditional_moments[1] = sigma2;
    if (q >= 3) unconditional_moments[2] = skewness * sigma * sigma * sigma;

    // Reusable buffers
    std::vector<double> errors(static_cast<std::size_t>(q));
    std::vector<double> conditional_moments(static_cast<std::size_t>(q));

    // Helper: accumulate pull for a given observation weight (transcribed VERBATIM,
    // including the >= 0 -> negPull sign convention -- see the file header).
    auto accumulate_pull = [&](const std::vector<double>& errs, double weight) {
        for (int j = 0; j < q; j++) {
            if (errs[static_cast<std::size_t>(j)] >= 0)
                neg_pull[static_cast<std::size_t>(j)] +=
                    weight * std::fabs(errs[static_cast<std::size_t>(j)]);
            else
                pos_pull[static_cast<std::size_t>(j)] +=
                    weight * std::fabs(errs[static_cast<std::size_t>(j)]);
        }
    };

    // Low outlier moments -- left-censored below the threshold
    if (data_frame().number_of_low_outliers() > 0) {
        double lower = min;
        double upper = std::min(
            max, is_log10 ? numerics::clamped_log10(data_frame().low_outlier_threshold())
                          : data_frame().low_outlier_threshold());
        if (lower >= upper)
            std::fill(conditional_moments.begin(), conditional_moments.end(), 0.0);
        else
            conditional_moments = model->conditional_moments(lower, upper);
        for (int j = 0; j < q; j++)
            errors[static_cast<std::size_t>(j)] = conditional_moments[static_cast<std::size_t>(j)] -
                                                  unconditional_moments[static_cast<std::size_t>(j)];
        accumulate_pull(errors, data_frame().number_of_low_outliers());
    }

    // Exact data -- non-outlier observed values with Bessel corrections
    for (std::size_t k = 0; k < data_frame().exact_series().count(); k++) {
        const ExactData& data = data_frame().exact_series()[k];
        if (!data.is_low_outlier()) {
            double value = is_log10 ? data.log10_value() : data.value();
            double d = value - unconditional_moments[0];
            double d2 = d * d;
            conditional_moments[0] = value;
            if (q >= 2) conditional_moments[1] = c2 * d2;
            if (q >= 3) conditional_moments[2] = c3 * d2 * d;
            for (int j = 0; j < q; j++)
                errors[static_cast<std::size_t>(j)] =
                    conditional_moments[static_cast<std::size_t>(j)] -
                    unconditional_moments[static_cast<std::size_t>(j)];
            accumulate_pull(errors, 1);
        }
    }

    // Uncertain data -- integrate over measurement error distribution
    for (std::size_t k = 0; k < data_frame().uncertain_series().count(); k++) {
        std::vector<double> uncertain_errors = moment_conditions_for_uncertain_data(
            unconditional_moments, data_frame().uncertain_series()[k], is_log10);
        accumulate_pull(uncertain_errors, 1);
    }

    // Interval data -- conditional moments over [L, U]
    for (std::size_t k = 0; k < data_frame().interval_series().count(); k++) {
        const IntervalData& data = data_frame().interval_series()[k];
        double lower = std::max(min, is_log10 ? data.log10_lower_value() : data.lower_value());
        double upper = std::min(max, is_log10 ? data.log10_upper_value() : data.upper_value());
        if (lower >= upper)
            std::fill(conditional_moments.begin(), conditional_moments.end(), 0.0);
        else
            conditional_moments = model->conditional_moments(lower, upper);
        for (int j = 0; j < q; j++)
            errors[static_cast<std::size_t>(j)] = conditional_moments[static_cast<std::size_t>(j)] -
                                                  unconditional_moments[static_cast<std::size_t>(j)];
        accumulate_pull(errors, 1);
    }

    // Threshold data -- left-censored and right-censored
    for (std::size_t k = 0; k < data_frame().threshold_series().count(); k++) {
        const ThresholdData& data = data_frame().threshold_series()[k];
        // Left censored
        if (data.number_below() > 0) {
            double lower = min;
            double upper = std::min(max, is_log10 ? data.log10_value() : data.value());
            if (lower >= upper)
                std::fill(conditional_moments.begin(), conditional_moments.end(), 0.0);
            else
                conditional_moments = model->conditional_moments(lower, upper);
            for (int j = 0; j < q; j++)
                errors[static_cast<std::size_t>(j)] =
                    conditional_moments[static_cast<std::size_t>(j)] -
                    unconditional_moments[static_cast<std::size_t>(j)];
            accumulate_pull(errors, data.number_below());
        }
        // Right censored
        if (data.number_above() > 0) {
            double lower = std::min(max, is_log10 ? data.log10_value() : data.value());
            double upper = max;
            if (lower >= upper)
                std::fill(conditional_moments.begin(), conditional_moments.end(), 0.0);
            else
                conditional_moments = model->conditional_moments(lower, upper);
            for (int j = 0; j < q; j++)
                errors[static_cast<std::size_t>(j)] =
                    conditional_moments[static_cast<std::size_t>(j)] -
                    unconditional_moments[static_cast<std::size_t>(j)];
            accumulate_pull(errors, data.number_above());
        }
    }

    for (int j = 0; j < q; j++) {
        std::size_t sj = static_cast<std::size_t>(j);
        double denom = pos_pull[sj] + neg_pull[sj];
        score[sj] = (denom > 0.0) ? (pos_pull[sj] - neg_pull[sj]) / (denom + 1e-12) : 0.0;
    }

    return score;
}

// Computes a weighted error direction score (WEDS) for each parameter (C# line 1901).
// NOTE: deliberately evaluated in the NATURAL parameterization -- no inverse-link step
// (the one scoring method that skips it; see the class-header remarks).
inline std::vector<double> Bulletin17CDistribution::weighted_error_direction_score(
    const std::vector<double>& parameters) const {
    int q = number_of_parameters();

    // weighted counts of positive errors (obs above model expectation) / negative errors
    std::vector<double> w_pos(static_cast<std::size_t>(q), 0.0);
    std::vector<double> w_neg(static_cast<std::size_t>(q), 0.0);
    std::vector<double> score(static_cast<std::size_t>(q), 0.0);

    // Bessel small-sample correction factors for non-outlier exact data
    int ns = 0;
    for (std::size_t k = 0; k < data_frame().exact_series().count(); k++)
        if (!data_frame().exact_series()[k].is_low_outlier()) ns++;
    double c2 = ns >= 2 ? ns / static_cast<double>(ns - 1) : 1.0;
    double c3 = ns >= 3 ? static_cast<double>(ns * ns) / ((ns - 1) * (ns - 2)) : 1.0;

    // Configure distribution model and log-space flag
    std::unique_ptr<DistributionBase> model;
    bool is_log10 = false;

    if (distribution_type() == DistributionType::LogPearsonTypeIII) {
        model = create_distribution(DistributionType::PearsonTypeIII);
        is_log10 = true;
    } else if (distribution_type() == DistributionType::LogNormal) {
        model = create_distribution(DistributionType::Normal);
        is_log10 = true;
    } else {
        model = distribution_->clone();
    }

    // Validate parameters without throwing
    model->set_parameters(parameters);
    if (!model->parameters_valid()) {
        std::fill(score.begin(), score.end(), std::numeric_limits<double>::quiet_NaN());
        return score;
    }

    // Integration bounds from distribution support
    double min = model->inverse_cdf(numerics::kDoubleMachineEpsilon);
    double max = model->inverse_cdf(1 - numerics::kDoubleMachineEpsilon);

    // Unconditional central moments of the fitted distribution
    double mu = model->mean();
    double sigma = model->standard_deviation();
    double sigma2 = model->variance();
    double skewness = model->skewness();

    // Guard against non-finite moments
    if (!numerics::is_finite(mu) || !numerics::is_finite(sigma2) || sigma <= 0) {
        std::fill(score.begin(), score.end(), std::numeric_limits<double>::quiet_NaN());
        return score;
    }

    // Store unconditional moments: [mu, sigma^2, mu3]
    std::vector<double> unconditional_moments(static_cast<std::size_t>(q));
    unconditional_moments[0] = mu;
    if (q >= 2) unconditional_moments[1] = sigma2;
    if (q >= 3) unconditional_moments[2] = skewness * sigma * sigma * sigma;

    // Reusable buffers
    std::vector<double> errors(static_cast<std::size_t>(q));
    std::vector<double> conditional_moments(static_cast<std::size_t>(q));

    // Helper: accumulate weighted direction counts
    auto accumulate_direction = [&](const std::vector<double>& errs, double weight) {
        for (int j = 0; j < q; j++) {
            if (errs[static_cast<std::size_t>(j)] > 0)
                w_pos[static_cast<std::size_t>(j)] += weight;
            else if (errs[static_cast<std::size_t>(j)] < 0)
                w_neg[static_cast<std::size_t>(j)] += weight;
        }
    };

    // Low outlier moments -- left-censored below the threshold
    if (data_frame().number_of_low_outliers() > 0) {
        double lower = min;
        double upper = std::min(
            max, is_log10 ? numerics::clamped_log10(data_frame().low_outlier_threshold())
                          : data_frame().low_outlier_threshold());
        if (lower >= upper)
            std::fill(conditional_moments.begin(), conditional_moments.end(), 0.0);
        else
            conditional_moments = model->conditional_moments(lower, upper);
        for (int j = 0; j < q; j++)
            errors[static_cast<std::size_t>(j)] = conditional_moments[static_cast<std::size_t>(j)] -
                                                  unconditional_moments[static_cast<std::size_t>(j)];
        accumulate_direction(errors, data_frame().number_of_low_outliers());
    }

    // Exact data -- non-outlier observed values with Bessel corrections
    for (std::size_t k = 0; k < data_frame().exact_series().count(); k++) {
        const ExactData& data = data_frame().exact_series()[k];
        if (!data.is_low_outlier()) {
            double value = is_log10 ? data.log10_value() : data.value();
            double d = value - unconditional_moments[0];
            double d2 = d * d;
            conditional_moments[0] = value;
            if (q >= 2) conditional_moments[1] = c2 * d2;
            if (q >= 3) conditional_moments[2] = c3 * d2 * d;
            for (int j = 0; j < q; j++)
                errors[static_cast<std::size_t>(j)] =
                    conditional_moments[static_cast<std::size_t>(j)] -
                    unconditional_moments[static_cast<std::size_t>(j)];
            accumulate_direction(errors, 1);
        }
    }

    // Uncertain data -- integrate over measurement error distribution
    for (std::size_t k = 0; k < data_frame().uncertain_series().count(); k++) {
        std::vector<double> uncertain_errors = moment_conditions_for_uncertain_data(
            unconditional_moments, data_frame().uncertain_series()[k], is_log10);
        accumulate_direction(uncertain_errors, 1);
    }

    // Interval data -- conditional moments over [L, U]
    for (std::size_t k = 0; k < data_frame().interval_series().count(); k++) {
        const IntervalData& data = data_frame().interval_series()[k];
        double lower = std::max(min, is_log10 ? data.log10_lower_value() : data.lower_value());
        double upper = std::min(max, is_log10 ? data.log10_upper_value() : data.upper_value());
        if (lower >= upper)
            std::fill(conditional_moments.begin(), conditional_moments.end(), 0.0);
        else
            conditional_moments = model->conditional_moments(lower, upper);
        for (int j = 0; j < q; j++)
            errors[static_cast<std::size_t>(j)] = conditional_moments[static_cast<std::size_t>(j)] -
                                                  unconditional_moments[static_cast<std::size_t>(j)];
        accumulate_direction(errors, 1);
    }

    // Threshold data -- left-censored and right-censored
    for (std::size_t k = 0; k < data_frame().threshold_series().count(); k++) {
        const ThresholdData& data = data_frame().threshold_series()[k];
        // Left censored
        if (data.number_below() > 0) {
            double lower = min;
            double upper = std::min(max, is_log10 ? data.log10_value() : data.value());
            if (lower >= upper)
                std::fill(conditional_moments.begin(), conditional_moments.end(), 0.0);
            else
                conditional_moments = model->conditional_moments(lower, upper);
            for (int j = 0; j < q; j++)
                errors[static_cast<std::size_t>(j)] =
                    conditional_moments[static_cast<std::size_t>(j)] -
                    unconditional_moments[static_cast<std::size_t>(j)];
            accumulate_direction(errors, data.number_below());
        }
        // Right censored
        if (data.number_above() > 0) {
            double lower = std::min(max, is_log10 ? data.log10_value() : data.value());
            double upper = max;
            if (lower >= upper)
                std::fill(conditional_moments.begin(), conditional_moments.end(), 0.0);
            else
                conditional_moments = model->conditional_moments(lower, upper);
            for (int j = 0; j < q; j++)
                errors[static_cast<std::size_t>(j)] =
                    conditional_moments[static_cast<std::size_t>(j)] -
                    unconditional_moments[static_cast<std::size_t>(j)];
            accumulate_direction(errors, data.number_above());
        }
    }

    for (int j = 0; j < q; j++) {
        std::size_t sj = static_cast<std::size_t>(j);
        double total = w_pos[sj] + w_neg[sj];
        score[sj] = total > 0 ? (w_pos[sj] - w_neg[sj]) / total : 0.0;
    }

    return score;
}

// Computes WEDS from link-space parameters by inverse-linking first (C# line 2090). The
// C# ArgumentNullException for a null array is unrepresentable for a vector reference.
inline std::vector<double> Bulletin17CDistribution::weighted_error_direction_score_from_linked(
    const std::vector<double>& linked_parameters) const {
    return weighted_error_direction_score(link_controller_.inverse_link(linked_parameters));
}

// Computes per-observation moment condition g-vectors for all observations (C# line 2128).
inline numerics::math::linalg::Matrix2D Bulletin17CDistribution::pointwise_moment_conditions_impl(
    std::vector<double> parameters) const {
    using numerics::math::linalg::Matrix2D;

    parameters = link_controller_.inverse_link(parameters);

    int q = number_of_parameters();
    int n = data_frame().total_record_length();
    Matrix2D result(static_cast<std::size_t>(n),
                    std::vector<double>(static_cast<std::size_t>(q), 0.0));

    // Bessel small-sample correction factors for non-outlier exact data
    int ns = 0;
    for (std::size_t k = 0; k < data_frame().exact_series().count(); k++)
        if (!data_frame().exact_series()[k].is_low_outlier()) ns++;
    double c2 = ns >= 2 ? ns / static_cast<double>(ns - 1) : 1.0;
    double c3 = ns >= 3 ? static_cast<double>(ns * ns) / ((ns - 1) * (ns - 2)) : 1.0;

    // Configure distribution model and log-space flag
    std::unique_ptr<DistributionBase> model;
    bool is_log10 = false;

    if (distribution_type() == DistributionType::LogPearsonTypeIII) {
        model = create_distribution(DistributionType::PearsonTypeIII);
        is_log10 = true;
    } else if (distribution_type() == DistributionType::LogNormal) {
        model = create_distribution(DistributionType::Normal);
        is_log10 = true;
    } else {
        model = distribution_->clone();
    }

    // Validate parameters without throwing -- called frequently during optimization
    model->set_parameters(parameters);
    if (!model->parameters_valid()) {
        return result;
    }

    // Integration bounds from distribution support
    double min = model->inverse_cdf(numerics::kDoubleMachineEpsilon);
    double max = model->inverse_cdf(1 - numerics::kDoubleMachineEpsilon);

    // Unconditional central moments of the fitted distribution
    double mu = model->mean();
    double sigma = model->standard_deviation();
    double sigma2 = model->variance();
    double skewness = model->skewness();

    // Guard against non-finite moments from extreme parameter values
    if (!numerics::is_finite(mu) || !numerics::is_finite(sigma2) || sigma <= 0) {
        return result;
    }

    // Store unconditional moments: [mu, sigma^2, mu3]
    std::vector<double> unconditional_moments(static_cast<std::size_t>(q));
    unconditional_moments[0] = mu;
    if (q >= 2) unconditional_moments[1] = sigma2;
    if (q >= 3) unconditional_moments[2] = skewness * sigma * sigma * sigma;

    // Reusable buffer for conditional moments
    std::vector<double> conditional_moments(static_cast<std::size_t>(q));

    // Current row index into the result matrix
    int row = 0;

    // Helper: store g-vector (c[j] - m[j]) into result[row..row+count-1, :]
    auto store_g = [&](const std::vector<double>& c, const std::vector<double>& m, int count) {
        double g0 = c[0] - m[0];
        double g1 = (c.size() >= 2) ? c[1] - m[1] : 0.0;
        double g2 = (q >= 3 && c.size() >= 3) ? c[2] - m[2] : 0.0;
        for (int r = 0; r < count; r++) {
            std::size_t sr = static_cast<std::size_t>(row);
            result[sr][0] = g0;
            if (q >= 2) result[sr][1] = g1;
            if (q >= 3) result[sr][2] = g2;
            row++;
        }
    };

    // Low outlier moments -- treated as left-censored below the low outlier threshold
    if (data_frame().number_of_low_outliers() > 0) {
        double lower = min;
        double upper = std::min(
            max, is_log10 ? numerics::clamped_log10(data_frame().low_outlier_threshold())
                          : data_frame().low_outlier_threshold());
        if (lower >= upper)
            std::fill(conditional_moments.begin(), conditional_moments.end(), 0.0);
        else
            conditional_moments = model->conditional_moments(lower, upper);
        store_g(conditional_moments, unconditional_moments,
                data_frame().number_of_low_outliers());
    }

    // Exact data -- non-outlier observed values with Bessel corrections
    for (std::size_t k = 0; k < data_frame().exact_series().count(); k++) {
        const ExactData& data = data_frame().exact_series()[k];
        if (!data.is_low_outlier()) {
            double value = is_log10 ? data.log10_value() : data.value();
            double d = value - unconditional_moments[0];
            double d2 = d * d;
            conditional_moments[0] = value;
            if (q >= 2) conditional_moments[1] = c2 * d2;
            if (q >= 3) conditional_moments[2] = c3 * d2 * d;
            store_g(conditional_moments, unconditional_moments, 1);
        }
    }

    // Uncertain data -- integrate over measurement error distribution
    for (std::size_t k = 0; k < data_frame().uncertain_series().count(); k++) {
        std::vector<double> uncertain_moments = conditional_moments_for_uncertain_data(
            unconditional_moments, data_frame().uncertain_series()[k], is_log10);
        store_g(uncertain_moments, unconditional_moments, 1);
    }

    // Interval data -- conditional moments over [L, U]
    for (std::size_t k = 0; k < data_frame().interval_series().count(); k++) {
        const IntervalData& data = data_frame().interval_series()[k];
        double lower = std::max(min, is_log10 ? data.log10_lower_value() : data.lower_value());
        double upper = std::min(max, is_log10 ? data.log10_upper_value() : data.upper_value());
        if (lower >= upper)
            std::fill(conditional_moments.begin(), conditional_moments.end(), 0.0);
        else
            conditional_moments = model->conditional_moments(lower, upper);
        store_g(conditional_moments, unconditional_moments, 1);
    }

    // Threshold data -- left-censored (Y < threshold) and right-censored (Y >= threshold)
    for (std::size_t k = 0; k < data_frame().threshold_series().count(); k++) {
        const ThresholdData& data = data_frame().threshold_series()[k];
        // Left censored
        if (data.number_below() > 0) {
            double lower = min;
            double upper = std::min(max, is_log10 ? data.log10_value() : data.value());
            if (lower >= upper)
                std::fill(conditional_moments.begin(), conditional_moments.end(), 0.0);
            else
                conditional_moments = model->conditional_moments(lower, upper);
            store_g(conditional_moments, unconditional_moments, data.number_below());
        }
        // Right censored
        if (data.number_above() > 0) {
            double lower = std::min(max, is_log10 ? data.log10_value() : data.value());
            double upper = max;
            if (lower >= upper)
                std::fill(conditional_moments.begin(), conditional_moments.end(), 0.0);
            else
                conditional_moments = model->conditional_moments(lower, upper);
            store_g(conditional_moments, unconditional_moments, data.number_above());
        }
    }

    return result;
}

// Computes the gradient of the quantile function with respect to the distribution
// parameters (C# line 2315).
inline std::vector<double> Bulletin17CDistribution::quantile_gradient(
    double probability, const std::vector<double>& parameters) const {
    // (The C# ArgumentNullException for a null parameters array is unrepresentable.)
    if (probability <= 0.0 || probability >= 1.0)
        throw std::out_of_range("Probability must be between 0 and 1 exclusive.");

    int p = number_of_parameters();
    if (static_cast<int>(parameters.size()) != p)
        throw std::invalid_argument("Expected " + std::to_string(p) + " parameters but received " +
                                    std::to_string(parameters.size()) + ".");

    // Create the base distribution model. Log-transformed distributions (LP3, LogNormal)
    // use their base distribution (PT3, Normal) since parameters are in log-space.
    std::unique_ptr<DistributionBase> model;
    if (distribution_type() == DistributionType::LogPearsonTypeIII) {
        model = create_distribution(DistributionType::PearsonTypeIII);
    } else if (distribution_type() == DistributionType::LogNormal) {
        model = create_distribution(DistributionType::Normal);
    } else {
        model = distribution_->clone();
    }

    // Validate and set model parameters
    model->set_parameters(parameters);
    if (!model->parameters_valid())
        throw std::invalid_argument("Invalid parameters for quantile gradient computation.");

    // Get quantile gradient dF^-1(p)/dtheta
    if (distribution_type() == DistributionType::PearsonTypeIII ||
        distribution_type() == DistributionType::LogPearsonTypeIII) {
        return dynamic_cast<const numerics::distributions::PearsonTypeIII&>(*model)
            .quantile_gradient_for_moments(probability);
    } else {
        // The C# `((IStandardError)model).QuantileGradient(probability)` cast (line 2355)
        // as a per-type dispatch -- see the file header.
        using namespace numerics::distributions;
        switch (model->type()) {
            case DistributionType::Exponential:
                return dynamic_cast<const Exponential&>(*model).quantile_gradient(probability);
            case DistributionType::GammaDistribution:
                return dynamic_cast<const GammaDistribution&>(*model).quantile_gradient(
                    probability);
            case DistributionType::Normal:
                return dynamic_cast<const Normal&>(*model).quantile_gradient(probability);
            default:
                throw std::runtime_error("Distribution does not implement IStandardError.");
        }
    }
}

// Computes the variance of a quantile estimate using the delta method (C# line 2388).
inline double Bulletin17CDistribution::quantile_variance(
    double probability, const std::vector<double>& parameters,
    const numerics::math::linalg::Matrix2D& covariance_matrix) const {
    // (The C# ArgumentNullException for a null covariance matrix is unrepresentable.)
    int p = number_of_parameters();
    int rows = static_cast<int>(covariance_matrix.size());
    int cols = rows > 0 ? static_cast<int>(covariance_matrix[0].size()) : 0;
    if (rows != p || cols != p)
        throw std::invalid_argument("Covariance matrix must be " + std::to_string(p) + " x " +
                                    std::to_string(p) + " but received " + std::to_string(rows) +
                                    " x " + std::to_string(cols) + ".");

    // Get the quantile gradient vector
    std::vector<double> gradient = quantile_gradient(probability, parameters);

    // Compute quadratic form: Var(Q_p) = g' Sigma g
    double q_var = 0.0;
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < p; j++) {
            q_var += gradient[static_cast<std::size_t>(i)] *
                     gradient[static_cast<std::size_t>(j)] *
                     covariance_matrix[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        }
    }

    return q_var;
}

}  // namespace corehydro::models
