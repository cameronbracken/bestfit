// ported from: RMC-BestFit/src/RMC.BestFit/Analyses/Support/IAnalysis.cs @ fc28c0c
//
// Capability interface for all analyses. Mirrors the C# IAnalysis mixin, kept as a pure-virtual
// class with a public virtual destructor and no data (the Phase-4 interface precedent, e.g.
// numerics/distributions/base/i_estimation.hpp). Concrete state lives only in AnalysisBase / the
// concrete analyses.
//
// DEVIATIONS from the C# source, all deliberate (WPF/threading/lifecycle, none numerical):
//  * The `INotifyPropertyChanged` base is DROPPED (GUI data-binding only).
//  * The `AnalysisStarting` / `AnalysisCompleted` events are DROPPED (GUI run-lifecycle signals).
//  * `CancelAnalysis()` is DROPPED (cancellation-token plumbing; the synchronous port has no
//    cancellable long-running task surface).
//  * The C# async `Task RunAsync(SafeProgressReporter?)` collapses to a synchronous
//    `void run()` (project-wide async->sync rule; SafeProgressReporter dropped).
//  * The C# value-tuple `(bool IsValid, List<string> ValidationMessages) Validate()` maps to the
//    shared corehydro::models::ValidationResult struct; validate() is const to match the ported
//    Models/Estimation validate() bodies (C# has no const; the models port fixed it const).
#pragma once

#include "corehydro/models/support/validation_result.hpp"

namespace corehydro::analyses {

class IAnalysis {
   public:
    virtual ~IAnalysis() = default;

    // Run the analysis (C# `Task RunAsync(SafeProgressReporter?)`, async->sync).
    virtual void run() = 0;

    // Validate the current state, returning validity + any messages (C# `Validate()` tuple).
    virtual corehydro::models::ValidationResult validate() const = 0;

    // Whether the analysis currently has valid results (C# `bool IsEstimated { get; }`).
    virtual bool is_estimated() const = 0;
};

}  // namespace corehydro::analyses
