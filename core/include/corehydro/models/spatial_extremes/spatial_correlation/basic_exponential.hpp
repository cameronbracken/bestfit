// ported from: RMC-BestFit/src/RMC.BestFit/Models/SpatialExtremes/SpatialCorrelation/BasicExponential.cs @ c2e6192
//
// Exponential spatial correlation function: rho(h) = exp(-h / range). One parameter, Range.
// See i_correlation_model.hpp for the base-vs-impl DRY decision (parameters()/
// number_of_parameters()/set_parameter_values() live on the base) and the ToXElement skip.
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

class BasicExponential : public ICorrelationModel {
   public:
    // Constructs a new exponential correlation model (C# ctor):
    // Range (Value 10.0, LowerBound machine-eps, UpperBound 500, IsPositive, Uniform(eps, 500)).
    BasicExponential() {
        parameters_.emplace_back(
            /*owner_name=*/"", /*name=*/"Range", /*value=*/10.0,
            /*lower_bound=*/numerics::kDoubleMachineEpsilon, /*upper_bound=*/500.0,
            std::make_unique<numerics::distributions::Uniform>(numerics::kDoubleMachineEpsilon, 500.0),
            /*is_positive=*/true);
    }

    CorrelationFunctionType type() const override { return CorrelationFunctionType::Exponential; }

    // Evaluates the correlation function at distance h (C# Evaluate):
    // h<0 throws; h==0 -> 1; range<=0 -> 0; else exp(-h/range).
    double evaluate(double h) const override {
        if (h < 0) throw std::invalid_argument("Distance must be non-negative.");
        if (h == 0) return 1.0;
        double range = parameters_[0].value();
        if (range <= 0) return 0.0;
        return std::exp(-h / range);
    }

    // Returns a deep copy (C# Clone): copies Value/LowerBound/UpperBound and clones the prior.
    std::unique_ptr<ICorrelationModel> clone() const override {
        auto result = std::make_unique<BasicExponential>();
        result->parameters_[0].set_value(parameters_[0].value());
        result->parameters_[0].set_lower_bound(parameters_[0].lower_bound());
        result->parameters_[0].set_upper_bound(parameters_[0].upper_bound());
        result->parameters_[0].set_prior_distribution(parameters_[0].prior_distribution().clone());
        return result;
    }
};

}  // namespace corehydro::models::spatial_extremes
