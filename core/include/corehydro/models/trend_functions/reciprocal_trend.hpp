// ported from: RMC-BestFit/src/RMC.BestFit/Models/TrendFunctions/ReciprocalTrend.cs @ fc28c0c
//
// Reciprocal trend model: y(t) = 1 / (alpha + beta * (t - StartIndex)).
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

namespace corehydro::models::trend_functions {

class ReciprocalTrend final : public TrendModelBase {
   public:
    ReciprocalTrend() { set_default_parameters(); }

    TrendModelType type() const override { return TrendModelType::Reciprocal; }

    void set_default_parameters() override {
        std::vector<ModelParameter> params;
        {
            ModelParameter p;  // "(alpha)"
            p.set_owner_name(owner_name());
            p.set_name("(\xCE\xB1)");
            params.push_back(std::move(p));
        }
        {
            ModelParameter p;  // "(beta)": bounds [-1, 1], Uniform(-1, 1) prior
            p.set_owner_name(owner_name());
            p.set_name("(\xCE\xB2)");
            p.set_lower_bound(-1.0);
            p.set_upper_bound(1.0);
            p.set_prior_distribution(
                std::make_unique<numerics::distributions::Uniform>(-1.0, 1.0));
            params.push_back(std::move(p));
        }
        set_parameters(std::move(params));
    }

    double predict(int index) const override {
        const double t = static_cast<double>(index) - static_cast<double>(start_index());
        const double a = parameters()[0].value();
        const double b = parameters()[1].value();
        double denom = a + b * t;

        if (std::fabs(denom) < kMinDenominatorMagnitude) {
            denom = denom >= 0.0 ? kMinDenominatorMagnitude : -kMinDenominatorMagnitude;
        }

        return 1.0 / denom;
    }

    std::unique_ptr<ITrendModel> clone() const override {
        auto model = std::make_unique<ReciprocalTrend>();
        model->set_owner_name(owner_name());
        model->set_use_default_flat_priors(use_default_flat_priors());
        model->set_start_index(start_index());
        for (std::size_t i = 0; i < parameters().size(); i++) {
            model->parameters()[i] = parameters()[i].clone();
        }
        return model;
    }

   private:
    // C# `MinDenominatorMagnitude`.
    static constexpr double kMinDenominatorMagnitude = 1e-12;
};

}  // namespace corehydro::models::trend_functions
