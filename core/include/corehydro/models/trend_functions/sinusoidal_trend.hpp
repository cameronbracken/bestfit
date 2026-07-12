// ported from: RMC-BestFit/src/RMC.BestFit/Models/TrendFunctions/SinusoidalTrend.cs @ fc28c0c
//
// Sinusoidal trend model: y(t) = alpha + beta * sin(2*pi * gamma * (t - StartIndex) + delta).
// The XElement constructor is deliberately not ported (XML out of scope). See
// linear_trend.hpp for the double-vs-int time-offset note.
#pragma once
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/models/trend_functions/support/trend_model_base.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models::trend_functions {

class SinusoidalTrend final : public TrendModelBase {
   public:
    SinusoidalTrend() { set_default_parameters(); }

    TrendModelType type() const override { return TrendModelType::Sinusoidal; }

    void set_default_parameters() override {
        std::vector<ModelParameter> params;
        {
            ModelParameter p;  // "(alpha)": mean level
            p.set_owner_name(owner_name());
            p.set_name("(\xCE\xB1)");
            params.push_back(std::move(p));
        }
        {
            ModelParameter p;  // "(beta)": amplitude (non-negative), bounds [0, 1],
                               // Value 0.5, Uniform(0, 1) prior
            p.set_owner_name(owner_name());
            p.set_name("(\xCE\xB2)");
            p.set_lower_bound(0.0);
            p.set_upper_bound(1.0);
            p.set_value(0.5);
            p.set_prior_distribution(
                std::make_unique<numerics::distributions::Uniform>(0.0, 1.0));
            params.push_back(std::move(p));
        }
        {
            ModelParameter p;  // "(gamma)": frequency (cycles per time unit), bounds
                               // [DoubleMachineEpsilon, 0.5], Value 0.25, Uniform(0, 0.5) prior
            p.set_owner_name(owner_name());
            p.set_name("(\xCE\xB3)");
            p.set_lower_bound(numerics::kDoubleMachineEpsilon);
            p.set_upper_bound(0.5);
            p.set_value(0.25);
            p.set_prior_distribution(
                std::make_unique<numerics::distributions::Uniform>(0.0, 0.5));
            params.push_back(std::move(p));
        }
        {
            ModelParameter p;  // "(delta)": phase shift, bounds [0, 2*pi], Value pi,
                               // Uniform(0, 2*pi) prior
            p.set_owner_name(owner_name());
            p.set_name("(\xCE\xB4)");
            p.set_lower_bound(0.0);
            p.set_upper_bound(2.0 * numerics::kPi);
            p.set_value(numerics::kPi);
            p.set_prior_distribution(
                std::make_unique<numerics::distributions::Uniform>(0.0, 2.0 * numerics::kPi));
            params.push_back(std::move(p));
        }
        set_parameters(std::move(params));
    }

    double predict(int index) const override {
        const double t = static_cast<double>(index) - static_cast<double>(start_index());
        const double a = parameters()[0].value();
        const double b = parameters()[1].value();
        const double c = parameters()[2].value();
        const double d = parameters()[3].value();

        return a + b * std::sin(2.0 * numerics::kPi * c * t + d);
    }

    std::unique_ptr<ITrendModel> clone() const override {
        auto model = std::make_unique<SinusoidalTrend>();
        model->set_owner_name(owner_name());
        model->set_use_default_flat_priors(use_default_flat_priors());
        model->set_start_index(start_index());
        for (std::size_t i = 0; i < parameters().size(); i++) {
            model->parameters()[i] = parameters()[i].clone();
        }
        return model;
    }
};

}  // namespace corehydro::models::trend_functions
