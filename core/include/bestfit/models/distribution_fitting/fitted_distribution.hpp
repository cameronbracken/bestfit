// ported from: RMC-BestFit/src/RMC.BestFit/Models/DistributionFitting/FittedDistribution.cs @ fc28c0c
//
// A goodness-of-fit DTO: one fitted univariate distribution plus its AIC / BIC / RMSE metrics,
// a fit-succeeded flag, a show-results flag, and an optional failure diagnostic. Produced by
// FittingAnalysis, one instance per candidate distribution. Ranking is left to the consumer.
//
// OWNERSHIP (deviation from the C# GC reference). In C# `Distribution` is a get-only
// `UnivariateDistributionBase?` holding the `.Clone()` FittingAnalysis hands in. Here the DTO OWNS
// that clone through `std::unique_ptr`; `distribution()` returns a non-owning raw pointer (nullptr
// when unset -- the C# has a null-Distribution path, e.g. its ToolTip, so accessors tolerate it).
// The never-mutate rule is RELAXED for this model/DTO object (it mirrors the C# stateful API).
//
// ShowResults CHOICE (C# governs; brief allows either faithful mapping). The C# setter is public
// (backed by INPC), but FittingAnalysis only ever sets ShowResults through the CONSTRUCTOR (to the
// `success` flag). A get-only field is therefore behaviorally equivalent for this scope, so this
// port makes `show_results` a get-only field with NO setter. INPC is dropped project-wide.
//
// ErrorMessage is the one member FittingAnalysis writes AFTER construction (on a failed MLE
// attempt), so it stays a plain mutable get/set field.
//
// SKIPPED (XML/INPC/GUI, dropped project-wide per the A4/A5 precedent):
//   * the `XElement` deserialization constructor and `ToXElement()` -- XML (de)serialization.
//   * `ToolTip` -- a WPF display string over DisplayName/ParameterNames/GetParameters formatting.
//   * `INotifyPropertyChanged` / `PropertyChanged` / `RaisePropertyChange` -- GUI binding.
#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"

namespace bestfit::models {

class FittedDistribution {
   public:
    using UnivariateDistributionBase = numerics::distributions::UnivariateDistributionBase;

    // C# ctor (line 30): `FittedDistribution(distribution, aic = NaN, bic = NaN, rmse = NaN,
    // fitSucceeded = false, showResults = false)`. Takes ownership of the distribution clone the
    // caller (FittingAnalysis) hands in; `distribution` may be null (empty unique_ptr).
    explicit FittedDistribution(
        std::unique_ptr<UnivariateDistributionBase> distribution = nullptr,
        double aic = std::numeric_limits<double>::quiet_NaN(),
        double bic = std::numeric_limits<double>::quiet_NaN(),
        double rmse = std::numeric_limits<double>::quiet_NaN(), bool fit_succeeded = false,
        bool show_results = false)
        : distribution_(std::move(distribution)),
          aic_(aic),
          bic_(bic),
          rmse_(rmse),
          fit_succeeded_(fit_succeeded),
          show_results_(show_results) {}

    // Move-only (owns a unique_ptr). Move enables `fitted_distributions_[idx] = FittedDistribution(...)`.
    FittedDistribution(const FittedDistribution&) = delete;
    FittedDistribution& operator=(const FittedDistribution&) = delete;
    FittedDistribution(FittedDistribution&&) = default;
    FittedDistribution& operator=(FittedDistribution&&) = default;

    // C# get-only `Distribution` (line 100). Non-owning pointer; nullptr when unset.
    const UnivariateDistributionBase* distribution() const { return distribution_.get(); }
    UnivariateDistributionBase* distribution() { return distribution_.get(); }

    // C# get-only `AIC` / `BIC` / `RMSE` (lines 105-120), default NaN.
    double aic() const { return aic_; }
    double bic() const { return bic_; }
    double rmse() const { return rmse_; }

    // C# get-only `FitSucceeded` (line 125), default false.
    bool fit_succeeded() const { return fit_succeeded_; }

    // C# `ShowResults` (line 130) -- get-only field here (see the file header CHOICE note).
    bool show_results() const { return show_results_; }

    // C# `ErrorMessage` (line 93): mutable get/set, default empty. Written by FittingAnalysis on a
    // failed MLE attempt.
    const std::string& error_message() const { return error_message_; }
    void set_error_message(std::string message) { error_message_ = std::move(message); }

   private:
    std::unique_ptr<UnivariateDistributionBase> distribution_;
    double aic_;
    double bic_;
    double rmse_;
    bool fit_succeeded_;
    bool show_results_;
    std::string error_message_;
};

}  // namespace bestfit::models
