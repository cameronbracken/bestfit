// ported from: Numerics/Functions/Link Functions/YeoJohnsonLink.cs @ a2c4dbf
//
// Yeo-Johnson power-transformation link function, defined on the full real line. Useful
// for skew or shape parameters that do not have a closed-form variance-stabilizing
// link. Lambda is restricted to the YeoJohnson::fit_lambda range [-5, 5]. Wraps the
// ported YeoJohnson transform class (corehydro/numerics/data/yeo_johnson.hpp); lambda is
// fixed at construction (the C# property is get-only). The XElement constructor and
// ToXElement are dropped; the C# null check in the values constructor has no analog
// (a C++ std::vector cannot be null).
// C# ArgumentOutOfRangeException -> std::out_of_range; ArgumentException ->
// std::invalid_argument.
#pragma once
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/yeo_johnson.hpp"
#include "corehydro/numerics/functions/i_link_function.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::functions {

class YeoJohnsonLink : public ILinkFunction {
   public:
    // Initializes a new instance with lambda equal to 1.
    YeoJohnsonLink() : lambda_(1.0) {}

    // Initializes a new instance with the given Yeo-Johnson transformation parameter.
    explicit YeoJohnsonLink(double lambda) : lambda_(validate_lambda(lambda)) {}

    // Initializes a new instance by estimating lambda from representative values.
    explicit YeoJohnsonLink(const std::vector<double>& values) {
        if (values.size() < 2)
            throw std::invalid_argument("At least 2 values are required to fit lambda.");
        for (std::size_t i = 0; i < values.size(); ++i)
            if (!corehydro::numerics::is_finite(values[i]))
                throw std::invalid_argument("Every representative value must be finite.");

        lambda_ = validate_lambda(data::YeoJohnson::fit_lambda(values));
    }

    // Gets the Yeo-Johnson transformation parameter.
    double lambda() const { return lambda_; }

    double link(double x) const override { return data::YeoJohnson::transform(x, lambda_); }

    double inverse_link(double eta) const override {
        return data::YeoJohnson::inverse_transform(eta, lambda_);
    }

    double d_link(double x) const override { return data::YeoJohnson::derivative(x, lambda_); }

   private:
    static constexpr double kMinimumLambda = -5.0;
    static constexpr double kMaximumLambda = 5.0;

    double lambda_;

    // Validates a Yeo-Johnson lambda value and returns it.
    static double validate_lambda(double lambda) {
        if (!corehydro::numerics::is_finite(lambda) || lambda < kMinimumLambda ||
            lambda > kMaximumLambda)
            throw std::out_of_range("Lambda must be finite and in the range [-5, 5].");

        return lambda;
    }
};

}  // namespace corehydro::numerics::functions
