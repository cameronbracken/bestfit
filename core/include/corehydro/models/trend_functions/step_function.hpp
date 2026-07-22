// ported from: RMC-BestFit/src/RMC.BestFit/Models/TrendFunctions/StepFunction.cs @ c2e6192
//
// Step-function trend model with a single change point: y(t) = mu_1 for t <= t_s and
// y(t) = mu_2 for t > t_s, where t_s is the step time (the change-point index at which
// the mean shifts).
// The XElement constructor is deliberately not ported (XML out of scope). See
// linear_trend.hpp for the double-vs-int time-offset note.
#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/models/trend_functions/support/trend_model_base.hpp"

namespace corehydro::models::trend_functions {

class StepFunction final : public TrendModelBase {
   public:
    StepFunction() { set_default_parameters(); }

    TrendModelType type() const override { return TrendModelType::StepFunction; }

    void set_default_parameters() override {
        std::vector<ModelParameter> params;
        {
            ModelParameter p;  // "(mu_1)"
            p.set_owner_name(owner_name());
            p.set_name("(\xCE\xBC\xE2\x82\x81)");
            params.push_back(std::move(p));
        }
        {
            ModelParameter p;  // "(mu_2)"
            p.set_owner_name(owner_name());
            p.set_name("(\xCE\xBC\xE2\x82\x82)");
            params.push_back(std::move(p));
        }
        {
            ModelParameter p;  // "(t_s)": change point (index)
            p.set_owner_name(owner_name());
            p.set_name("(t\xE2\x82\x9B)");
            params.push_back(std::move(p));
        }
        set_parameters(std::move(params));
    }

    // Inclusion convention: at the change-point itself (t == tc), the function returns
    // mu_1 (the pre-change value). Strictly after the change-point (t > tc), it returns
    // mu_2. This matches the standard left-continuous step convention.
    double predict(int index) const override {
        const double t = static_cast<double>(index) - static_cast<double>(start_index());
        const double mu1 = parameters()[0].value();
        const double mu2 = parameters()[1].value();
        const double tc = parameters()[2].value() - static_cast<double>(start_index());

        // Step at change-point tc; left-continuous (mu_1 for t <= tc, mu_2 for t > tc).
        if (t <= tc) {
            return mu1;
        }

        return mu2;
    }

    std::unique_ptr<ITrendModel> clone() const override {
        auto model = std::make_unique<StepFunction>();
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
