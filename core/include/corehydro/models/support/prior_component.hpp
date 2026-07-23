// ported from: RMC-BestFit/src/RMC.BestFit/Models/Support/PriorComponent.cs @ c2e6192
//
// Represents a single component of the prior log-likelihood for influence diagnostics
// (parameter priors, quantile priors, Jeffreys scale priors, spatial error, Jacobian terms,
// and other penalties). Like DataComponent, the C# type is a `readonly struct`; ported here
// as a getter-only value type (see data_component.hpp for the rationale on not using `const`
// members).
#pragma once
#include <cstdio>
#include <string>
#include <utility>

namespace corehydro::models {

enum class PriorComponentType {
    ParameterPrior,
    QuantilePrior,
    JeffreysScalePrior,
    SpatialError,
    Jacobian,
    OtherPenalty
};

class PriorComponent {
   public:
    explicit PriorComponent(std::string name, double log_likelihood,
                             PriorComponentType type = PriorComponentType::ParameterPrior)
        : name_(std::move(name)), log_likelihood_(log_likelihood), type_(type) {}

    const std::string& name() const { return name_; }
    double log_likelihood() const { return log_likelihood_; }
    PriorComponentType type() const { return type_; }

    // Mirrors the C# `ToString()`: "<Name>: <LogLikelihood:F4>". The exact `:F4` numeric
    // formatting is approximated with `%.4f` (not oracle-checked).
    std::string to_string() const {
        char ll_buf[64];
        std::snprintf(ll_buf, sizeof ll_buf, "%.4f", log_likelihood_);
        return name_ + ": " + ll_buf;
    }

   private:
    std::string name_;
    double log_likelihood_;
    PriorComponentType type_;
};

}  // namespace corehydro::models
