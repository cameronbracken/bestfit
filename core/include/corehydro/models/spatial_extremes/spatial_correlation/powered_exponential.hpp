// ported from: RMC-BestFit/src/RMC.BestFit/Models/SpatialExtremes/SpatialCorrelation/PoweredExponential.cs @ fc28c0c
//
// Powered exponential (Gaussian) spatial correlation function: rho(h) = exp(-(h/phi)^nu).
// Two parameters, Range and Smoothness. The exponential (nu=1) and Gaussian (nu=2) are special
// cases. See i_correlation_model.hpp for the base-vs-impl DRY decision and the ToXElement skip.
#pragma once
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "corehydro/models/spatial_extremes/spatial_correlation/correlation_function_type.hpp"
#include "corehydro/models/spatial_extremes/spatial_correlation/i_correlation_model.hpp"
#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models::spatial_extremes {

class PoweredExponential : public ICorrelationModel {
   public:
    // Constructs a new powered exponential correlation model (C# ctor):
    //   Range      (Value 10.0, LowerBound machine-eps, UpperBound 500, IsPositive, Uniform(eps,500))
    //   Smoothness (Value 1.5,  LowerBound 0.1,          UpperBound 2.0, IsPositive, Uniform(0.1,2.0))
    PoweredExponential() {
        parameters_.emplace_back(
            /*owner_name=*/"", /*name=*/"Range", /*value=*/10.0,
            /*lower_bound=*/numerics::kDoubleMachineEpsilon, /*upper_bound=*/500.0,
            std::make_unique<numerics::distributions::Uniform>(numerics::kDoubleMachineEpsilon, 500.0),
            /*is_positive=*/true);
        parameters_.emplace_back(
            /*owner_name=*/"", /*name=*/"Smoothness", /*value=*/1.5,
            /*lower_bound=*/0.1, /*upper_bound=*/2.0,
            std::make_unique<numerics::distributions::Uniform>(0.1, 2.0),
            /*is_positive=*/true);
    }

    CorrelationFunctionType type() const override {
        return CorrelationFunctionType::PoweredExponential;
    }

    // Evaluates the correlation function at distance h (C# Evaluate):
    // h<0 throws; h==0 -> 1; range<=0 -> 0; else exp(-pow(h/range, smoothness)).
    double evaluate(double h) const override {
        if (h < 0) throw std::invalid_argument("Distance must be non-negative.");
        if (h == 0) return 1.0;
        double range = parameters_[0].value();
        double smoothness = parameters_[1].value();
        if (range <= 0) return 0.0;
        return std::exp(-std::pow(h / range, smoothness));
    }

    // Returns a deep copy (C# Clone): loops over both parameters copying value/bounds and
    // cloning the prior.
    std::unique_ptr<ICorrelationModel> clone() const override {
        auto result = std::make_unique<PoweredExponential>();
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            result->parameters_[i].set_value(parameters_[i].value());
            result->parameters_[i].set_lower_bound(parameters_[i].lower_bound());
            result->parameters_[i].set_upper_bound(parameters_[i].upper_bound());
            result->parameters_[i].set_prior_distribution(parameters_[i].prior_distribution().clone());
        }
        return result;
    }
};

}  // namespace corehydro::models::spatial_extremes
