// ported from: RMC-BestFit/src/RMC.BestFit/Models/TrendFunctions/ConstantTrend.cs @ fc28c0c
//
// Trend model that is constant in time: y(t) = alpha.
// The XElement constructor is deliberately not ported (XML out of scope).
#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "bestfit/models/support/model_parameter.hpp"
#include "bestfit/models/trend_functions/support/trend_model_base.hpp"

namespace bestfit::models::trend_functions {

class ConstantTrend final : public TrendModelBase {
   public:
    ConstantTrend() { set_default_parameters(); }

    TrendModelType type() const override { return TrendModelType::Constant; }

    void set_default_parameters() override {
        std::vector<ModelParameter> params;
        {
            ModelParameter p;  // "(alpha)"
            p.set_owner_name(owner_name());
            p.set_name("(\xCE\xB1)");
            params.push_back(std::move(p));
        }
        set_parameters(std::move(params));
    }

    double predict(int /*index*/) const override { return parameters()[0].value(); }

    std::unique_ptr<ITrendModel> clone() const override {
        auto model = std::make_unique<ConstantTrend>();
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
