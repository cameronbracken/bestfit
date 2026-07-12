// ported from: Numerics/Distributions/Bivariate Copulas/Base/IArchimedeanCopula.cs @ a2c4dbf
//           +  Numerics/Distributions/Bivariate Copulas/Base/ArchimedeanCopula.cs @ a2c4dbf
//
// Abstract base for every Archimedean copula (Clayton, AMH, Frank, Gumbel, Joe). Folds
// IArchimedeanCopula's members directly into the base, matching the IBivariateCopula/
// BivariateCopula precedent. Declares the five generator-function pure virtuals and
// promotes the shared 1-parameter plumbing (NumberOfCopulaParameters/GetCopulaParameters/
// SetCopulaParameters/ValidateParameter) plus the generic Genest et al. 1986 PDF/CDF/
// InverseCDF built from the generator alone -- concrete copulas that have a closed form
// (e.g. Clayton's CDF/InverseCDF) override just those methods; PDF still resolves through
// the generic formula here via virtual dispatch on the (possibly overridden) CDF.
#pragma once
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/distributions/copulas/base/bivariate_copula.hpp"

namespace corehydro::numerics::distributions::copulas {

class ArchimedeanCopula : public BivariateCopula {
   public:
    // The generator function of the copula.
    virtual double generator(double t) const = 0;
    // The first derivative of the generator function.
    virtual double generator_prime(double t) const = 0;
    // The second derivative of the generator function.
    virtual double generator_prime2(double t) const = 0;
    // The inverse of the generator function.
    virtual double generator_inverse(double t) const = 0;
    // The inverse of the first derivative of the generator function.
    virtual double generator_prime_inverse(double t) const = 0;

    int number_of_copula_parameters() const override { return 1; }
    std::vector<double> get_copula_parameters() const override { return {theta()}; }
    void set_copula_parameters(const std::vector<double>& parameters) override {
        set_theta(parameters[0]);
    }

    // NOTE (upstream bug, transcribed verbatim): the C# ArchimedeanCopula.ValidateParameter
    // final branch does `return new ArgumentOutOfRangeException(nameof(Theta), "Parameter is
    // valid");` instead of `return null;` -- i.e. it constructs and returns a NON-NULL
    // exception object even when theta IS in range. Because BivariateCopula.Theta's setter
    // is `_parametersValid = ValidateParameter(value, false) is null;`, this means
    // `ParametersValid` is unconditionally FALSE for every Archimedean-derived copula
    // (Clayton, AMH, Frank, Gumbel, Joe) regardless of whether theta is actually valid.
    // Confirmed against the real C# library (`new ClaytonCopula(2.0).ParametersValid ==
    // false`) and against NormalCopula/StudentTCopula, whose ValidateParameter correctly
    // `return null;` in the equivalent branch. This does NOT affect PDF/CDF/InverseCDF or
    // any fit (the branch below never throws for a valid theta) -- only the ParametersValid
    // getter is wrong. See docs/upstream-csharp-issues.md for the full writeup; ported
    // bug-for-bug per the "observable behavior governs" rule (the bug is a wrong return
    // value, not a crash/UB, so there is no fidelity reason to deviate).
    std::optional<std::string> validate_parameter(double parameter,
                                                    bool throw_exception) const override {
        if (parameter < theta_minimum()) {
            std::string msg = "The dependency parameter theta (theta) must be greater than or "
                               "equal to " +
                               std::to_string(theta_minimum()) + ".";
            if (throw_exception) throw std::out_of_range(msg);
            return msg;
        }
        if (parameter > theta_maximum()) {
            std::string msg =
                "The dependency parameter theta (theta) must be less than or equal to " +
                std::to_string(theta_maximum()) + ".";
            if (throw_exception) throw std::out_of_range(msg);
            return msg;
        }
        return std::string("Parameter is valid");
    }

    // The Genest et al. (1986) copula density: c(u,v) = -psi''(C(u,v)) psi'(u) psi'(v) /
    // psi'(C(u,v))^3, where psi is the generator. C(u,v) resolves through the (possibly
    // overridden) virtual cdf(), so a concrete class with a closed-form CDF (e.g. Clayton)
    // still gets a consistent PDF here without overriding PDF itself.
    double pdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        double num = -generator_prime2(cdf(u, v)) * generator_prime(u) * generator_prime(v);
        double den = std::pow(generator_prime(cdf(u, v)), 3.0);
        return num / den;
    }

    double cdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        return generator_inverse(generator(u) + generator(v));
    }

    // Genest et al. 1986 conditional-sampling InverseCDF:
    // 1) s = psi'(u) / v; w = (psi')^-1(s).
    // 2) v' = psi^-1(psi(w) - psi(u)).
    // 3) (u, v') is the simulated pair, preserving the dependence structure.
    std::array<double, 2> inverse_cdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        double s = generator_prime(u) / v;
        double w = generator_prime_inverse(s);
        double vv = generator_inverse(generator(w) - generator(u));
        return {u, vv};
    }
};

}  // namespace corehydro::numerics::distributions::copulas
