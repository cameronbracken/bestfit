// ported from: RMC-BestFit/src/RMC.BestFit/Models/TrendFunctions/ExponentialTrend.cs @ c2e6192
//
// Exponential trend model: y(t) = alpha * exp(beta * (t - StartIndex)).
// The XElement constructor is deliberately not ported (XML out of scope). See
// linear_trend.hpp for the double-vs-int time-offset note.
#pragma once
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/models/trend_functions/support/trend_model_base.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"

namespace corehydro::models::trend_functions {

class ExponentialTrend final : public TrendModelBase {
   public:
    ExponentialTrend() { set_default_parameters(); }

    TrendModelType type() const override { return TrendModelType::Exponential; }

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

    // When b * t would overflow exp() (|exponent| > 700), the prediction is propagated as
    // +inf / -inf / 0 rather than a finite-but-saturated value. This lets downstream
    // log-likelihood guards detect the unphysical extrapolation and reject it (as -inf)
    // rather than treat a * 1e304 as a valid prediction.
    double predict(int index) const override {
        const double t = static_cast<double>(index) - static_cast<double>(start_index());
        const double a = parameters()[0].value();
        const double b = parameters()[1].value();
        const double exponent = b * t;

        if (exponent > kMaxExponent) {
            return a >= 0.0 ? std::numeric_limits<double>::infinity()
                            : -std::numeric_limits<double>::infinity();
        }
        if (exponent < -kMaxExponent) {
            return 0.0;
        }

        return a * std::exp(exponent);
    }

    std::unique_ptr<ITrendModel> clone() const override {
        auto model = std::make_unique<ExponentialTrend>();
        model->set_owner_name(owner_name());
        model->set_use_default_flat_priors(use_default_flat_priors());
        model->set_start_index(start_index());
        for (std::size_t i = 0; i < parameters().size(); i++) {
            model->parameters()[i] = parameters()[i].clone();
        }
        return model;
    }

   private:
    // C# `MaxExponent`: ~log(1e304), guard against overflow.
    static constexpr double kMaxExponent = 700.0;
};

}  // namespace corehydro::models::trend_functions
