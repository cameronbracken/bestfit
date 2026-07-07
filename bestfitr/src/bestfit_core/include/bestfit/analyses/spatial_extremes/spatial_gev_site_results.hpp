// ported from: RMC-BestFit/src/RMC.BestFit/Analyses/SpatialExtremes/SpatialGEVSiteResults.cs @ fc28c0c
//
// Plain result DTO: the GEV (location/scale/shape) parameters and quantile curves for a single site
// (or, when site_index == -1, an ungauged prediction location) produced by SpatialGEVAnalysis. Each
// field mirrors the C# property one-for-one (snake_case), value-initialized. The C# XML docs carry
// no numeric content and are dropped. The never-mutate rule is relaxed for this DTO (the analysis
// fills and, in InflatePosteriorCovariance, edits it in place, mirroring the mutable C# class).
#pragma once
#include <vector>

namespace bestfit::analyses {

struct SpatialGEVSiteResults {
    // Site index (0-based). -1 indicates an ungauged (predicted) location (C# 34).
    int site_index = 0;

    // Site coordinates [X, Y] (C# 39).
    std::vector<double> coordinate;

    // GEV location parameter (xi): posterior mean + credible bounds (C# 48/53/58).
    double location_mean = 0.0;
    double location_lower = 0.0;
    double location_upper = 0.0;

    // GEV scale parameter (alpha): posterior mean + credible bounds (C# 67/72/77).
    double scale_mean = 0.0;
    double scale_lower = 0.0;
    double scale_upper = 0.0;

    // GEV shape parameter (kappa): posterior mean + credible bounds (C# 90/95/100).
    double shape_mean = 0.0;
    double shape_lower = 0.0;
    double shape_upper = 0.0;

    // Exceedance probabilities for the quantile curves (C# 108).
    std::vector<double> probabilities;

    // Quantiles at each probability: posterior mean, credible bounds, and point-estimate/mode
    // (C# 113/118/123/128).
    std::vector<double> quantile_mean;
    std::vector<double> quantile_lower;
    std::vector<double> quantile_upper;
    std::vector<double> quantile_mode;
};

}  // namespace bestfit::analyses
