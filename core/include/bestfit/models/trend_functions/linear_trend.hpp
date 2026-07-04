// ported from: RMC-BestFit/src/RMC.BestFit/Models/TrendFunctions/LinearTrend.cs @ fc28c0c
//
// Linear trend model: y(t) = alpha + beta * (t - StartIndex).
// The XElement constructor is deliberately not ported (XML out of scope).
//
// Deviation: `double t = index - StartIndex` is computed in double (each int cast first)
// rather than C#'s unchecked int subtraction, which silently wraps on overflow; signed
// int overflow is undefined behavior in C++. Only observable when |index - StartIndex|
// exceeds the int range (e.g. the upstream Test_Predict_IntegerOverflow_Handled inputs),
// where C# wraps and C++ returns the exact double difference.
#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "bestfit/models/support/model_parameter.hpp"
#include "bestfit/models/trend_functions/support/trend_model_base.hpp"
#include "bestfit/numerics/distributions/uniform.hpp"

namespace bestfit::models::trend_functions {

class LinearTrend final : public TrendModelBase {
   public:
    LinearTrend() { set_default_parameters(); }

    TrendModelType type() const override { return TrendModelType::Linear; }

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
        return a + b * t;
    }

    std::unique_ptr<ITrendModel> clone() const override {
        auto model = std::make_unique<LinearTrend>();
        model->set_owner_name(owner_name());
        model->set_use_default_flat_priors(use_default_flat_priors());
        model->set_start_index(start_index());
        for (std::size_t i = 0; i < parameters().size(); i++) {
            model->parameters()[i] = parameters()[i].clone();
        }
        return model;
    }
};

}  // namespace bestfit::models::trend_functions
