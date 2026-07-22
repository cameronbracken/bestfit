// ported from: RMC-BestFit/src/RMC.BestFit/Analyses/SpatialExtremes/SpatialGEVCrossValidationResults.cs @ c2e6192
//
// Plain result DTO: leave-one-site-out cross-validation diagnostics produced by
// SpatialGEVAnalysis::run_cross_validation(). Each field mirrors the C# property one-for-one
// (snake_case), value-initialized. The C# XML docs carry no numeric content and are dropped. Note
// (C# remark, transcribed): SiteCRPS is NOT yet computed by the driver -- it is allocated zero-filled
// for forward compatibility. The never-mutate rule is relaxed for this DTO (the driver fills it in
// place, mirroring the mutable C# class).
#pragma once
#include <vector>

namespace corehydro::analyses {

struct SpatialGEVCrossValidationResults {
    // Per-site prediction error (predicted - observed) for the T=100 quantile (C# 40).
    std::vector<double> site_prediction_errors;

    // Per-site RMSE across return periods T=2,5,10,25,50,100 (C# 49).
    std::vector<double> site_rmse;

    // Per-site relative bias (predicted - observed)/observed at T=100 (C# 59).
    std::vector<double> site_bias;

    // Per-site Continuous Ranked Probability Score (NOT yet computed; always zero) (C# 75).
    std::vector<double> site_crps;

    // Aggregate metrics across all sites (C# 84/93/102).
    double mean_absolute_error = 0.0;
    double root_mean_square_error = 0.0;
    double mean_bias = 0.0;
};

}  // namespace corehydro::analyses
