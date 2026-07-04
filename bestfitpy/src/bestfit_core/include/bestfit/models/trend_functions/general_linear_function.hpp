// ported from: RMC-BestFit/src/RMC.BestFit/Models/TrendFunctions/GeneralLinearFunction.cs @ fc28c0c
//
// General linear function for covariate modeling:
//   f(x) = beta_0 + beta_1*x_1 + beta_2*x_2 + ... + beta_p*x_p
// where beta_0 is the intercept and beta_i are regression coefficients for covariates x_i.
//
// Unlike time-based trend models that use a scalar index, this model supports arbitrary
// covariate matrices, making it suitable for spatial regression surfaces (location,
// elevation, etc.), covariate effects on distribution parameters, ARMAX exogenous variable
// modeling, and hierarchical Bayesian spatial models (SpatialGEV).
//
// The C# class implements ITrendModel DIRECTLY (not via TrendModelBase); this port mirrors
// that. Notably its OwnerName setter does NOT propagate the new name to the parameters the
// way TrendModelBase's does -- that C# asymmetry is preserved.
//
// Covariate representation: the C# `double[,]? _covariates` is used only for dimension
// queries (GetLength(0)/GetLength(1)), element reads (_covariates[i, j]), and a deep
// Clone(). This port stores `std::optional<numerics::math::linalg::Matrix>`:
//   - `std::nullopt` maps to the C# null (number_of_covariates()/number_of_observations()
//     return 0; set_covariates(std::nullopt) resets to an intercept-only model), and stays
//     distinguishable from an engaged empty matrix exactly as C# null vs `new double[0,0]`.
//   - Matrix::number_of_rows()/number_of_columns()/operator()(i, j) reproduce the three
//     C# array operations, and copying the optional deep-copies the data (the C# Clone()).
// The covariates() getter mirrors the C# property get, which returns the internal array
// reference: element writes through the mutable overload are possible (as in C#), but
// wholesale replacement must go through set_covariates() to get the parameter rebuild
// (in C# the property setter is likewise the only way to trigger the rebuild).
//
// SubscriptFormatter: the C# calls SubscriptFormatter.ToSubscript(j)
// (Models/Support/SubscriptFormatter.cs, deferred as WPF cosmetics), but here the subscript
// digits are part of the parameter NAMES this class produces, so the digit -> U+2080..U+2089
// mapping is ported as a small file-local helper below rather than a public support header.
// Unicode is written as UTF-8 byte escapes (M6 convention; see quadratic_trend.hpp).
//
// Deliberately NOT ported (per the porting policy for this layer):
//   - the XElement constructor and ToXElement() (XML serialization)
//   - INotifyPropertyChanged / PropertyChanged / RaisePropertyChanged (WPF binding
//     plumbing); the setters keep the C# functional side effects but the notification
//     calls are dropped.
//
// C++ mapping notes:
//   - The C# `(ownerName, covariates)` ctor throws ArgumentNullException for a null
//     ownerName; a null std::string is unrepresentable, so that check has no port. The
//     OwnerName setter's `value ?? string.Empty` null-to-empty guard is likewise
//     unrepresentable; its ordinal-equality change check ports.
//   - SetParameterValues' ArgumentNullException branch is unrepresentable (const
//     reference); the count-mismatch ArgumentException ports as std::invalid_argument
//     (M6 convention, trend_model_base.hpp).
//   - Predict's ArgumentOutOfRangeException ports as std::out_of_range (the repo-wide
//     mapping, e.g. mcmc_diagnostics.hpp); PredictWithCovariates' length-mismatch
//     ArgumentException ports as std::invalid_argument. PredictWithCovariates' `covariates
//     == null` guard collapses into the empty-vector case.
//   - `ITrendModel Clone()` maps to `std::unique_ptr<ITrendModel> clone() const`.
#pragma once
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bestfit/models/support/model_parameter.hpp"
#include "bestfit/models/trend_functions/support/i_trend_model.hpp"
#include "bestfit/models/trend_functions/support/trend_model_type.hpp"
#include "bestfit/numerics/distributions/uniform.hpp"
#include "bestfit/numerics/math/linalg/matrix.hpp"

namespace bestfit::models::trend_functions {

namespace general_linear_detail {

// Localized port of Models/Support/SubscriptFormatter.cs @ fc28c0c (ToSubscript): maps the
// ASCII digits of the invariant-culture integer string to their Unicode subscript
// counterparts U+2080..U+2089 (UTF-8 escapes); any non-digit character (e.g. the '-' of a
// negative value) passes through unchanged, as in the C#.
inline std::string to_subscript(int value) {
    static const char* const kSubscripts[10] = {
        "\xE2\x82\x80", "\xE2\x82\x81", "\xE2\x82\x82", "\xE2\x82\x83", "\xE2\x82\x84",
        "\xE2\x82\x85", "\xE2\x82\x86", "\xE2\x82\x87", "\xE2\x82\x88", "\xE2\x82\x89"};
    const std::string s = std::to_string(value);
    std::string result;
    result.reserve(s.size() * 3);
    for (const char c : s) {
        if (c >= '0' && c <= '9') {
            result += kSubscripts[c - '0'];
        } else {
            result += c;
        }
    }
    return result;
}

}  // namespace general_linear_detail

class GeneralLinearFunction final : public ITrendModel {
   public:
    // Initializes a new instance with default settings (intercept only, no covariates).
    GeneralLinearFunction() { set_default_parameters(); }

    // Initializes a new instance for a specific distribution parameter with optional
    // covariates. `owner_name` is the name of the owning distribution parameter (e.g.
    // "Location", "Scale", "Shape"); `covariates` is the covariate matrix where each row
    // corresponds to an observation (site) and each column corresponds to a covariate
    // (std::nullopt for an intercept-only model). The C# ArgumentNullException for a null
    // ownerName is unrepresentable with std::string (see header note).
    explicit GeneralLinearFunction(std::string owner_name,
                                   std::optional<numerics::math::linalg::Matrix> covariates =
                                       std::nullopt)
        : owner_name_(std::move(owner_name)), covariates_(std::move(covariates)) {
        set_default_parameters();
    }

    // --- OwnerName (no propagation to parameters; see header note) ---
    const std::string& owner_name() const override { return owner_name_; }
    void set_owner_name(const std::string& value) override {
        // The C# `value ?? string.Empty` null guard is unrepresentable; the ordinal
        // equality change check ports.
        if (owner_name_ != value) {
            owner_name_ = value;
            // RaisePropertyChanged(nameof(OwnerName)) skipped (INPC not ported).
        }
    }

    // --- Type ---
    TrendModelType type() const override { return TrendModelType::GeneralLinear; }

    // --- StartIndex ---
    // For GeneralLinearFunction, StartIndex is used when the model is evaluated via the
    // predict(int index) method for compatibility with time-based models. When using
    // predict_with_covariates, this property is not used. (C# remark.)
    int start_index() const override { return start_index_; }
    void set_start_index(int value) override {
        if (start_index_ != value) {
            start_index_ = value;
            // RaisePropertyChanged(nameof(StartIndex)) skipped (INPC not ported).
        }
    }

    // --- Parameters ---
    std::vector<ModelParameter>& parameters() override { return parameters_; }
    const std::vector<ModelParameter>& parameters() const override { return parameters_; }

    // --- NumberOfParameters ---
    int number_of_parameters() const override { return static_cast<int>(parameters_.size()); }

    // --- UseDefaultFlatPriors ---
    bool use_default_flat_priors() const override { return use_default_flat_priors_; }
    void set_use_default_flat_priors(bool value) override {
        if (use_default_flat_priors_ != value) {
            use_default_flat_priors_ = value;
            // RaisePropertyChanged(nameof(UseDefaultFlatPriors)) skipped (INPC not ported).
            if (use_default_flat_priors_) {
                set_default_parameters();
            }
        }
    }

    // --- Covariates ---
    // Each row corresponds to an observation (e.g., a site in spatial models); each column
    // corresponds to a covariate variable. The number of columns determines the number of
    // regression coefficients (plus intercept). The getter mirrors the C# property get
    // (a reference to the stored matrix; see the header note on element writes vs
    // wholesale replacement).
    const std::optional<numerics::math::linalg::Matrix>& covariates() const {
        return covariates_;
    }
    std::optional<numerics::math::linalg::Matrix>& covariates() { return covariates_; }
    void set_covariates(std::optional<numerics::math::linalg::Matrix> value) {
        covariates_ = std::move(value);
        // RaisePropertyChanged(nameof(Covariates)) and (nameof(NumberOfCovariates))
        // skipped (INPC not ported).
        // Rebuild parameters when covariates change.
        set_default_parameters();
    }

    // --- NumberOfCovariates: columns in the covariate matrix (0 when none). ---
    int number_of_covariates() const {
        return covariates_.has_value() ? covariates_->number_of_columns() : 0;
    }

    // --- NumberOfObservations: rows in the covariate matrix (0 when none). ---
    int number_of_observations() const {
        return covariates_.has_value() ? covariates_->number_of_rows() : 0;
    }

    // --- SetDefaultParameters ---
    // Creates parameters for the intercept (beta_0) and one coefficient (beta_i) per
    // covariate. Default priors are Uniform(-1, 1) for the regression coefficients. The
    // intercept prior is left unbounded (uses ModelParameter defaults).
    void set_default_parameters() override {
        parameters_.clear();

        // Intercept parameter ("beta_0", owner-qualified "{owner}.beta_0" when the owner
        // name is non-empty)
        {
            ModelParameter p;
            p.set_owner_name(owner_name_);
            p.set_name(owner_name_.empty() ? std::string(kBeta0)
                                           : owner_name_ + "." + kBeta0);
            parameters_.push_back(std::move(p));
        }

        // Covariate coefficients ("beta_1", "beta_2", ...): bounds [-1, 1],
        // Uniform(-1, 1) prior
        const int num_covariates =
            covariates_.has_value() ? covariates_->number_of_columns() : 0;
        for (int j = 1; j <= num_covariates; j++) {
            const std::string beta_name =
                "\xCE\xB2" + general_linear_detail::to_subscript(j);
            ModelParameter p;
            p.set_owner_name(owner_name_);
            p.set_name(owner_name_.empty() ? beta_name : owner_name_ + "." + beta_name);
            p.set_lower_bound(-1.0);
            p.set_upper_bound(1.0);
            p.set_prior_distribution(
                std::make_unique<numerics::distributions::Uniform>(-1.0, 1.0));
            parameters_.push_back(std::move(p));
        }

        // RaisePropertyChanged(nameof(Parameters)) and (nameof(NumberOfParameters))
        // skipped (INPC not ported).
    }

    // --- SetParameterValues ---
    void set_parameter_values(const std::vector<double>& parameters) override {
        // C#'s ArgumentNullException branch is unrepresentable (const reference).
        if (parameters.size() != static_cast<std::size_t>(number_of_parameters())) {
            throw std::invalid_argument("Expected " + std::to_string(number_of_parameters()) +
                                        " parameter(s) but received " +
                                        std::to_string(parameters.size()) + ".");
        }
        for (int i = 0; i < number_of_parameters(); i++) {
            parameters_[static_cast<std::size_t>(i)].set_value(
                parameters[static_cast<std::size_t>(i)]);
        }
        // RaisePropertyChanged(nameof(Parameters)) skipped (INPC not ported).
    }

    // --- Predict ---
    // The index selects a row from the stored covariate matrix; the prediction is
    // beta_0 + beta_1*Covariates[index,0] + beta_2*Covariates[index,1] + ...
    // If no covariates are stored, returns only the intercept (no index check -- the C#
    // bounds check lives inside the `_covariates != null` branch).
    double predict(int index) const override {
        double result = parameters_[0].value();  // Intercept

        if (covariates_.has_value()) {
            if (index < 0 || index >= covariates_->number_of_rows()) {
                throw std::out_of_range(
                    "Index " + std::to_string(index) + " is out of range [0, " +
                    std::to_string(covariates_->number_of_rows() - 1) + "].");
            }

            for (int j = 0; j < covariates_->number_of_columns(); j++) {
                result += parameters_[static_cast<std::size_t>(j) + 1].value() *
                          (*covariates_)(index, j);
            }
        }

        return result;
    }

    // --- PredictWithCovariates ---
    // Evaluates the linear function using provided covariate values directly: useful for
    // spatial prediction at ungauged locations where covariate values are known but there
    // is no corresponding row in the stored covariate matrix. An empty vector (the C#
    // null-or-empty case) returns the intercept; a length mismatch with
    // number_of_covariates() throws.
    double predict_with_covariates(const std::vector<double>& covariates) const {
        double result = parameters_[0].value();  // Intercept

        // If model has no covariates, return just the intercept
        if (number_of_covariates() == 0) {
            return result;
        }

        // Validate covariate vector (the C# `covariates == null` case collapses into the
        // empty case)
        if (covariates.empty()) {
            return result;
        }

        if (static_cast<int>(covariates.size()) != number_of_covariates()) {
            throw std::invalid_argument("Expected " + std::to_string(number_of_covariates()) +
                                        " covariate(s) but received " +
                                        std::to_string(covariates.size()) + ".");
        }

        // Compute linear combination
        for (std::size_t j = 0; j < covariates.size(); j++) {
            result += parameters_[j + 1].value() * covariates[j];
        }

        return result;
    }

    // --- Clone ---
    std::unique_ptr<ITrendModel> clone() const override {
        // Clone covariate matrix if present (copying the optional deep-copies the data);
        // the ctor runs set_default_parameters(), whose output is then replaced wholesale
        // with parameter-by-parameter clones, exactly as the C#.
        auto model = std::make_unique<GeneralLinearFunction>(owner_name_, covariates_);
        model->use_default_flat_priors_ = use_default_flat_priors_;
        model->start_index_ = start_index_;

        // Clone parameters
        model->parameters_.clear();
        for (int i = 0; i < number_of_parameters(); i++) {
            model->parameters_.push_back(parameters_[static_cast<std::size_t>(i)].clone());
        }

        return model;
    }

   private:
    // "beta_0" (U+03B2 U+2080) as UTF-8 byte escapes (M6 convention).
    static constexpr const char* kBeta0 = "\xCE\xB2\xE2\x82\x80";

    std::string owner_name_ = "";
    bool use_default_flat_priors_ = true;
    int start_index_ = 0;
    std::optional<numerics::math::linalg::Matrix> covariates_;
    std::vector<ModelParameter> parameters_;
};

}  // namespace bestfit::models::trend_functions
