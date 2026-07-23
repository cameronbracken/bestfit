// ported from: RMC-BestFit/src/RMC.BestFit/Models/DataFrame/ThresholdDiagnostics.cs @ c2e6192
//
// Threshold selection diagnostics for Peaks-Over-Threshold (POT) analysis:
//   1. Mean Residual Life plot (MRL) -- the sample mean of excesses above each candidate
//      threshold; linear in the threshold under a valid GPD model. CLT confidence bands.
//   2. Parameter Stability plot -- GPD fitted by MLE at each candidate threshold; the
//      modified scale (sigma* = alpha - kappa * u) and the shape (kappa) should be
//      approximately constant above the true threshold. Delta-method standard errors
//      from the closed-form GPD parameter covariance.
// References: Coles (2001) Section 4.3; Davison & Smith (1990); the R POT package's
// mrlplot()/tcplot().
//
// C# ArgumentNullException/ArgumentException -> std::invalid_argument (the null-data
// case has no counterpart -- the data arrive by const reference). The C# static class
// maps to a class with only static members; the result DTOs (MRLPoint,
// MeanResidualLifeResult, StabilityPoint, ParameterStabilityResult) mirror the C#
// get-only properties as public fields with the same constructor parameter order.
// System.Diagnostics.Debug.WriteLine in the MLE catch block is not ported.
#pragma once
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/generalized_pareto.hpp"
#include "corehydro/numerics/distributions/normal.hpp"

namespace corehydro::models {

// A single point on a Mean Residual Life (MRL) plot: one candidate threshold with the
// sample mean of excesses above it and CLT confidence bounds (C# MRLPoint, line 302).
class MRLPoint {
   public:
    MRLPoint(double threshold, double mean_excess, double lower_ci, double upper_ci,
             int exceedance_count)
        : threshold(threshold),
          mean_excess(mean_excess),
          lower_ci(lower_ci),
          upper_ci(upper_ci),
          exceedance_count(exceedance_count) {}

    double threshold;      // The candidate threshold value u.
    double mean_excess;    // Sample mean of exceedances (x_i - u) for all x_i > u.
    double lower_ci;       // Lower bound of the confidence interval.
    double upper_ci;       // Upper bound of the confidence interval.
    int exceedance_count;  // Number of observations exceeding the threshold.
};

// Results of an MRL computation across a range of thresholds; points are ordered by
// increasing threshold, thresholds with fewer than 5 exceedances excluded
// (C# MeanResidualLifeResult, line 359).
class MeanResidualLifeResult {
   public:
    explicit MeanResidualLifeResult(std::vector<MRLPoint> points)
        : points(std::move(points)) {}

    std::vector<MRLPoint> points;
};

// A single point on a GPD parameter stability plot: MLE-estimated modified scale and
// shape at one candidate threshold, with delta-method confidence intervals
// (C# StabilityPoint, line 388).
class StabilityPoint {
   public:
    StabilityPoint(double threshold, double modified_scale, double modified_scale_lower_ci,
                   double modified_scale_upper_ci, double shape, double shape_lower_ci,
                   double shape_upper_ci, int exceedance_count)
        : threshold(threshold),
          modified_scale(modified_scale),
          modified_scale_lower_ci(modified_scale_lower_ci),
          modified_scale_upper_ci(modified_scale_upper_ci),
          shape(shape),
          shape_lower_ci(shape_lower_ci),
          shape_upper_ci(shape_upper_ci),
          exceedance_count(exceedance_count) {}

    double threshold;                // The candidate threshold value u.
    double modified_scale;           // sigma* = alpha_hat - kappa_hat * u.
    double modified_scale_lower_ci;  // Delta-method lower bound for sigma*.
    double modified_scale_upper_ci;  // Delta-method upper bound for sigma*.
    double shape;                    // The shape parameter kappa estimated by MLE.
    double shape_lower_ci;           // Lower bound for the shape parameter.
    double shape_upper_ci;           // Upper bound for the shape parameter.
    int exceedance_count;            // Number of observations exceeding the threshold.
};

// Results of a GPD parameter stability computation; points are ordered by increasing
// threshold, thresholds with fewer than 10 exceedances or failed MLE excluded
// (C# ParameterStabilityResult, line 479).
class ParameterStabilityResult {
   public:
    explicit ParameterStabilityResult(std::vector<StabilityPoint> points)
        : points(std::move(points)) {}

    std::vector<StabilityPoint> points;
};

// Static diagnostics class (C# static class ThresholdDiagnostics, line 35).
class ThresholdDiagnostics {
   public:
    ThresholdDiagnostics() = delete;

    // Computes the Mean Residual Life plot data for equally spaced candidate thresholds
    // in [u_min, u_max] (C# ComputeMeanResidualLife, line 78). CI = mean_excess +/-
    // z * sd / sqrt(n_u); thresholds with fewer than 5 exceedances are skipped.
    static MeanResidualLifeResult compute_mean_residual_life(
        const std::vector<double>& data, double u_min, double u_max,
        int n_thresholds = 100, double confidence_level = 0.95) {
        validate_inputs(data, u_min, u_max, n_thresholds, confidence_level);

        double z = numerics::distributions::Normal::standard_z(
            (1.0 + confidence_level) / 2.0);
        double step = (u_max - u_min) / (n_thresholds - 1);
        std::vector<MRLPoint> points;

        for (int i = 0; i < n_thresholds; i++) {
            double u = u_min + i * step;
            std::vector<double> excesses = compute_excesses(data, u);
            int n_u = static_cast<int>(excesses.size());

            if (n_u < kMinExceedancesForMRL) continue;

            double mean_excess = compute_mean(excesses);
            double sd = compute_standard_deviation(excesses, mean_excess);
            double se = sd / std::sqrt(static_cast<double>(n_u));

            points.push_back(MRLPoint(u, mean_excess, mean_excess - z * se,
                                      mean_excess + z * se, n_u));
        }

        return MeanResidualLifeResult(std::move(points));
    }

    // Computes the GPD parameter stability plot data for equally spaced candidate
    // thresholds in [u_min, u_max] (C# ComputeParameterStability, line 150). Fits the
    // GPD to the excesses by MLE; plots the modified scale sigma* = alpha - kappa * u
    // (delta-method SE) and the shape kappa. Thresholds with fewer than 10 exceedances,
    // or where the MLE fails, are skipped.
    static ParameterStabilityResult compute_parameter_stability(
        const std::vector<double>& data, double u_min, double u_max,
        int n_thresholds = 50, double confidence_level = 0.95) {
        validate_inputs(data, u_min, u_max, n_thresholds, confidence_level);

        double z = numerics::distributions::Normal::standard_z(
            (1.0 + confidence_level) / 2.0);
        double step = (u_max - u_min) / (n_thresholds - 1);
        std::vector<StabilityPoint> points;

        for (int i = 0; i < n_thresholds; i++) {
            double u = u_min + i * step;
            std::vector<double> excesses = compute_excesses(data, u);
            int n_u = static_cast<int>(excesses.size());

            if (n_u < kMinExceedancesForGPD) continue;

            try {
                // Fit GPD by MLE to the exceedances
                numerics::distributions::GeneralizedPareto gpd;
                gpd.estimate(excesses,
                             numerics::distributions::ParameterEstimationMethod::
                                 MaximumLikelihood);

                double alpha = gpd.alpha();  // scale
                double kappa = gpd.kappa();  // shape

                // Get the parameter covariance matrix (3x3)
                // [0][0]=Var(xi), [1][1]=Var(alpha), [2][2]=Var(kappa),
                // [1][2]=Cov(alpha, kappa)
                auto covar = gpd.parameter_covariance(
                    n_u, numerics::distributions::ParameterEstimationMethod::
                             MaximumLikelihood);

                double var_alpha = covar[1][1];
                double var_kappa = covar[2][2];
                double cov_alpha_kappa = covar[1][2];

                // Modified scale: sigma* = alpha - kappa * u
                double modified_scale = alpha - kappa * u;

                // Standard error of sigma* via delta method:
                // se(sigma*) = sqrt(Var(alpha) - 2*u*Cov(alpha,kappa) + u^2*Var(kappa))
                double var_star =
                    var_alpha - 2.0 * u * cov_alpha_kappa + u * u * var_kappa;

                // Guard against numerical issues yielding negative variance
                double se_star = var_star > 0 ? std::sqrt(var_star) : 0.0;
                double se_kappa = var_kappa > 0 ? std::sqrt(var_kappa) : 0.0;

                points.push_back(StabilityPoint(
                    u, modified_scale, modified_scale - z * se_star,
                    modified_scale + z * se_star, kappa, kappa - z * se_kappa,
                    kappa + z * se_kappa, n_u));
            } catch (...) {
                // MLE may fail for extreme thresholds -- skip this threshold
                continue;
            }
        }

        return ParameterStabilityResult(std::move(points));
    }

   private:
    // The minimum number of exceedances required to compute a mean residual life point.
    static constexpr int kMinExceedancesForMRL = 5;

    // The minimum number of exceedances required to fit a GPD for parameter stability
    // analysis.
    static constexpr int kMinExceedancesForGPD = 10;

    // Validates the common input parameters (C# ValidateInputs, line 228). The C#
    // null-data ArgumentNullException has no counterpart here (const reference).
    static void validate_inputs(const std::vector<double>& data, double u_min,
                                double u_max, int n_thresholds,
                                double confidence_level) {
        if (data.empty()) throw std::invalid_argument("Data must not be empty.");
        if (u_max <= u_min)
            throw std::invalid_argument("uMax (" + std::to_string(u_max) +
                                        ") must be greater than uMin (" +
                                        std::to_string(u_min) + ").");
        if (n_thresholds < 2)
            throw std::invalid_argument("nThresholds must be at least 2.");
        if (confidence_level <= 0.0 || confidence_level >= 1.0)
            throw std::invalid_argument(
                "confidenceLevel must be in the open interval (0, 1).");
    }

    // The list of exceedances (x_i - u) for all data values strictly greater than the
    // threshold (C# ComputeExcesses, line 248).
    static std::vector<double> compute_excesses(const std::vector<double>& data,
                                                double threshold) {
        std::vector<double> excesses;
        for (std::size_t i = 0; i < data.size(); i++) {
            if (data[i] > threshold) excesses.push_back(data[i] - threshold);
        }
        return excesses;
    }

    // Arithmetic mean (C# ComputeMean, line 264).
    static double compute_mean(const std::vector<double>& values) {
        double sum = 0;
        for (std::size_t i = 0; i < values.size(); i++) sum += values[i];
        return sum / static_cast<double>(values.size());
    }

    // Sample standard deviation (Bessel's correction) given a precomputed mean; 0 for
    // fewer than 2 values (C# ComputeStandardDeviation, line 279).
    static double compute_standard_deviation(const std::vector<double>& values,
                                             double mean) {
        if (values.size() < 2) return 0.0;

        double sum_sq = 0;
        for (std::size_t i = 0; i < values.size(); i++) {
            double diff = values[i] - mean;
            sum_sq += diff * diff;
        }
        return std::sqrt(sum_sq / static_cast<double>(values.size() - 1));
    }
};

}  // namespace corehydro::models
