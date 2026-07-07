// ported from: RMC-BestFit/src/RMC.BestFit/Analyses/Support/WeightedUnivariateAnalysis.cs @ fc28c0c
//
// Trivial child-wrapper DTO used by CompositeAnalysis: pairs an IUnivariateAnalysis with a weight.
// The wrapped analysis may be any IUnivariateAnalysis sibling (UnivariateAnalysis,
// Bulletin17CAnalysis, ...) EXCEPT another CompositeAnalysis -- that is rejected at the setter
// (C# ArgumentException, WeightedUnivariateAnalysis.cs:97-101) to avoid circular references and the
// undefined semantics of nesting composites.
//
// DEVIATIONS from the C# source, all deliberate:
//  * OWNERSHIP. The C# holds `IUnivariateAnalysis` by GC reference; here it is a NON-owning raw
//    pointer (the composite references, but does not own, its child analyses -- the caller keeps
//    them alive), and the default state is a null pointer (C# `null!`).
//  * INPC DROPPED. `INotifyPropertyChanged` / `PropertyChanged` / `RaisePropertyChange` /
//    `UnivariateAnalysis_PropertyChanged` -- WPF binding cascades, no numerical content.
//  * XML DROPPED. `ToXElement` (WeightedUnivariateAnalysis.cs:140) -- XElement serialization.
//  * Validate() returns `std::pair<bool, std::string>`, mirroring the C# `(bool IsValid, string
//    Message)` tuple EXACTLY (a single joined message, not the ValidationResult list shape).
//
// The composite-child reject uses `is_composite_analysis(...)`, a helper forward-declared here and
// defined inline in composite_analysis.hpp (the only header that completes CompositeAnalysis). This
// breaks the WeightedUnivariateAnalysis <-> CompositeAnalysis include cycle: a
// WeightedUnivariateAnalysis is only ever exercised as a CompositeAnalysis child, so
// composite_analysis.hpp -- which supplies the definition -- is always in the include set wherever
// the setter runs.
#pragma once
#include <stdexcept>
#include <string>
#include <utility>

#include "bestfit/analyses/support/i_univariate_analysis.hpp"

namespace bestfit::analyses {

class CompositeAnalysis;

// Returns true when `analysis` is a CompositeAnalysis. Defined inline in composite_analysis.hpp.
bool is_composite_analysis(const IUnivariateAnalysis* analysis);

class WeightedUnivariateAnalysis {
   public:
    // C# parameterless ctor (WeightedUnivariateAnalysis.cs:33): null analysis, zero weight.
    WeightedUnivariateAnalysis() = default;

    // C# `WeightedUnivariateAnalysis(IUnivariateAnalysis, double)` (:47). Rejects a
    // CompositeAnalysis child via the setter (C# ArgumentException).
    WeightedUnivariateAnalysis(IUnivariateAnalysis* analysis, double weight) {
        set_univariate_analysis(analysis);
        weight_ = weight;
    }

    // C# `Weight` (:72).
    double weight() const { return weight_; }
    void set_weight(double value) { weight_ = value; }

    // C# `UnivariateAnalysis` (:92). The setter rejects a CompositeAnalysis child.
    IUnivariateAnalysis* univariate_analysis() const { return univariate_analysis_; }
    void set_univariate_analysis(IUnivariateAnalysis* value) {
        if (value != nullptr && is_composite_analysis(value)) {
            throw std::invalid_argument(
                "A CompositeAnalysis cannot be used as a child of another CompositeAnalysis. "
                "Composite-of-composite is not supported because it can introduce circular "
                "references and undefined weighting semantics.");
        }
        univariate_analysis_ = value;
    }

    // C# `Validate` (:155): (bool IsValid, string Message). Null / unestimated analysis fails with
    // the shared "requires estimation" message; otherwise delegates to the inner analysis's
    // Validate() and joins its messages with "; ".
    std::pair<bool, std::string> validate() const {
        if (univariate_analysis_ == nullptr)
            return {false, "Error: A selected univariate analysis is invalid or requires estimation."};

        if (!univariate_analysis_->is_estimated())
            return {false, "Error: A selected univariate analysis is invalid or requires estimation."};

        bestfit::models::ValidationResult validation = univariate_analysis_->validate();
        if (!validation.is_valid) {
            std::string joined;
            for (std::size_t i = 0; i < validation.validation_messages.size(); ++i) {
                if (i > 0) joined += "; ";
                joined += validation.validation_messages[i];
            }
            return {false, joined};
        }

        return {true, std::string()};
    }

   private:
    double weight_ = 0.0;
    IUnivariateAnalysis* univariate_analysis_ = nullptr;  // NON-owning (see header)
};

}  // namespace bestfit::analyses
