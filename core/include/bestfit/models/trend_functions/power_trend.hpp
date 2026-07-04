// ported from: RMC-BestFit/src/RMC.BestFit/Models/TrendFunctions/PowerTrend.cs @ fc28c0c
//
// Power-law trend model: y(t) = alpha * (t - StartIndex)^beta.
// The XElement constructor is deliberately not ported (XML out of scope). See
// linear_trend.hpp for the double-vs-int time-offset note.
#pragma once
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "bestfit/models/support/model_parameter.hpp"
#include "bestfit/models/trend_functions/support/trend_model_base.hpp"
#include "bestfit/numerics/distributions/uniform.hpp"

namespace bestfit::models::trend_functions {

class PowerTrend final : public TrendModelBase {
   public:
    PowerTrend() { set_default_parameters(); }

    TrendModelType type() const override { return TrendModelType::Power; }

    void set_default_parameters() override {
        std::vector<ModelParameter> params;
        {
            ModelParameter p;  // "(alpha)"
            p.set_owner_name(owner_name());
            p.set_name("(\xCE\xB1)");
            params.push_back(std::move(p));
        }
        {
            ModelParameter p;  // "(beta)": bounds [-5, 5], Uniform(-5, 5) prior
            p.set_owner_name(owner_name());
            p.set_name("(\xCE\xB2)");
            p.set_lower_bound(-5.0);
            p.set_upper_bound(5.0);
            p.set_prior_distribution(
                std::make_unique<numerics::distributions::Uniform>(-5.0, 5.0));
            params.push_back(std::move(p));
        }
        set_parameters(std::move(params));
    }

    double predict(int index) const override {
        double t = static_cast<double>(index) - static_cast<double>(start_index());
        const double a = parameters()[0].value();
        const double b = parameters()[1].value();

        // Guard against negative t passed in by mistake, which could lead to complex
        // values for non-integer beta. Under normal usage, StartIndex is less than or
        // equal to the current index.
        if (t < 0.0) {
            t = 0.0;
        }

        // pow(0, beta) is +inf for beta < 0 and 1 for beta = 0 (per IEEE-754). The prior
        // allows beta in [-5, 5], so beta < 0 is reachable. Clamp t to a small positive
        // floor when beta < 0 so predict(StartIndex) returns a finite (though still
        // extreme) value at the boundary rather than propagating +inf into downstream
        // log-likelihood evaluation. The floor value 1e-2 was chosen so that for beta=-5
        // the magnitude pow(1e-2, -5) = 1e10 is large enough to be confidently rejected
        // by the data likelihood at the sampler's accept/reject step yet finite enough
        // that intermediate log / numerical-derivative passes don't overflow. The
        // previous floor 1e-12 produced pow(1e-12, -5) = 1e60, which is unphysical and
        // risks intermediate numerical issues even though the LL ultimately rejects.
        if (t == 0.0 && b < 0.0) {
            t = 1e-2;
        }

        return a * std::pow(t, b);
    }

    std::unique_ptr<ITrendModel> clone() const override {
        auto model = std::make_unique<PowerTrend>();
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
