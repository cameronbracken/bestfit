// ported from: RMC-BestFit/src/RMC.BestFit/Analyses/Support/AnalysisBase.cs @ fc28c0c
//
// Abstract base implementation for analysis classes. This port keeps ONLY the compute contract:
// the pure-virtual run()/validate() every concrete analysis implements, and the IsEstimated state
// (a protected backing field, a public const accessor, and a protected setter). Everything else
// on the C# class is WPF/threading/lifecycle plumbing with no numerical content and is dropped.
//
// IAnalysis is a VIRTUAL base here (deviation from the brief's "no virtual inheritance required"
// note). A concrete analysis derives from BOTH AnalysisBase and IUnivariateAnalysis, and
// IUnivariateAnalysis reaches IAnalysis through IBayesianAnalysis -- so IAnalysis sits at the join
// of a diamond. Virtual inheritance collapses it to a single shared subobject, which (a) makes an
// IAnalysis* upcast from the concrete analysis unambiguous and (b) lets this class's is_estimated()
// satisfy the single IAnalysis::is_estimated for every path. This faithfully maps the C# interface
// model, where the compiler dedupes a repeated interface base automatically.
//
// DEVIATIONS from the C# source, all deliberate (WPF/threading/lifecycle, none numerical):
//  * `INotifyPropertyChanged` / `PropertyChanged` / `RaisePropertyChange` -- DROPPED (GUI binding).
//    The IsEstimated setter's PropertyChanged notification is dropped; the plain assignment is kept.
//  * `AnalysisStarting` / `AnalysisCompleted` events + `OnAnalysisStarting` / `OnAnalysisCompleted`
//    -- DROPPED (GUI run-lifecycle signals).
//  * `CancellationTokenSource _cancellationTokenSource` + the `CancellationTokenSource` property +
//    `ResetCancellationToken()` + `CancelAnalysis()` -- DROPPED (cancellation plumbing).
//  * `SafeProgressReporter` parameter on RunAsync -- DROPPED; `Task RunAsync(...)` -> `void run()`.
//  * `SemaphoreSlim _reprocessGate` + `ReprocessIfEstimated(...)` + `AnalysisRunCompletedEventArgs`
//    -- DROPPED (fire-and-forget UI-thread reprocess scheduling). The C# Debug.WriteLine /
//    swallowed-exception guard lived inside ReprocessIfEstimated and disappears with it; no
//    surviving member needs a guard.
#pragma once

#include "corehydro/analyses/support/i_analysis.hpp"
#include "corehydro/models/support/validation_result.hpp"

namespace corehydro::analyses {

class AnalysisBase : public virtual IAnalysis {
   public:
    ~AnalysisBase() override = default;

    // Concrete analyses implement the compute contract (C# abstract RunAsync / Validate).
    void run() override = 0;
    corehydro::models::ValidationResult validate() const override = 0;

    // Whether the candidate distributions have been successfully fitted to the current data
    // (C# `bool IsEstimated { get; protected set; }`).
    bool is_estimated() const override { return is_estimated_; }

   protected:
    // Backing field for is_estimated() (C# `protected bool _isEstimated = false;`).
    bool is_estimated_ = false;

    // Protected setter (C# `IsEstimated { ...; protected set; }`). The C# setter raised
    // PropertyChanged on change; that notification is dropped, keeping the plain assignment.
    void set_is_estimated(bool value) { is_estimated_ = value; }
};

}  // namespace corehydro::analyses
