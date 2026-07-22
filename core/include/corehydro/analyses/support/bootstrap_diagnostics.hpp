// ported from: upstream/RMC-BestFit/src/RMC.BestFit/Analyses/Support/BootstrapDiagnostics.cs @ c2e6192
//
// Diagnostic counters collected during the parametric-bootstrap (A8) and pivot-bootstrap (Phase 9)
// uncertainty paths of Bulletin17CAnalysis. The C# updates the integer counters with Interlocked
// operations for its Parallel.For loops; the port runs the bootstrap loop SERIALLY (the estimator-
// port precedent), so plain `++` / `+=` reproduce the same final counts -- no atomics needed.
//
// T19 EXTENSION (net v2.0.0 state, commits 1b424e3/71b7d4b/7efa9d0): AttemptedReplicates (candidate
// count including replacements, sentinel -1 falls back to TotalReplicates), RetainedReplicates
// (parameter sets actually delivered, sentinel -1 falls back to ValidReplicates), TransformFailures
// (pivot link-space transform failures, distinct from a z-limit PivotRejection), the five GMM
// OptimizationStatus counters (RecordGMMStatus), and OptimizerFallbacks (BFGS -> Nelder-Mead count,
// AddOptimizerFallbacks -- see GeneralizedMethodOfMoments::optimizer_fallback_count(), T13).
// ValidReplicates/FailureRate/AverageRetries/AverageFunctionEvaluations are redefined over
// AttemptedReplicates rather than TotalReplicates (see each accessor). NONE of these new counters
// are populated by A8's get_parameter_sets_from_parametric_bootstrap() -- the shipped v2.0.0 C#
// GetParameterSetsFromParametricBootstrap never calls IncrementAttempted / RecordGMMStatus /
// AddOptimizerFallbacks / IncrementTransformFailure / sets RetainedReplicates -- so a diagnostics
// object built by that path always reads the new counters through their legacy-fallback defaults.
// They exist so the class is oracle-complete for the other sampling methods that DO populate them
// (the pivot bootstrap's Phase 1/3 GMM attempts and transform failures, Task 20).
//
// NAMING (documented per the A8 brief): the analyses layer uses snake_case members/accessors
// (mirroring the A7 Bulletin17CAnalysis header), so the C# PascalCase properties become snake_case
// here (`TotalReplicates` -> total_replicates() / set_total_replicates(), `IncrementFailed()` ->
// increment_failed(), etc.). The derived rate accessors guard division by zero exactly as the C#
// (`_totalReplicates > 0 ? ... : 0.0`, now `AttemptedReplicates > 0 ? ... : 0.0` for the three
// redefined-over-attempts rates).
//
// SKIPPED C# surface (each dropped for the shipped scope; see the report):
//   * ToXElement / FromXElement -- the XML (de)serialization surface, dropped project-wide.
//   * Phase1Time / Phase2Time / Phase3Time (TimeSpan) + the Stopwatch plumbing -- run-lifecycle
//     timing; OMITTED entirely (no field), since nothing on the shipped compute path reads them.
//   * The Interlocked atomics -- collapse to plain increments under the serial port.
#pragma once

#include <algorithm>

#include "corehydro/numerics/math/optimization/support/optimization_status.hpp"

namespace corehydro::analyses {

class BootstrapDiagnostics {
   public:
    using OptimizationStatus = corehydro::numerics::math::optimization::OptimizationStatus;

    // --- Counters (C# properties) ----------------------------------------------------------

    // C# `TotalReplicates` (settable): total bootstrap replicates requested.
    int total_replicates() const { return total_replicates_; }
    void set_total_replicates(int value) { total_replicates_ = value; }

    // C# `AttemptedReplicates` (T19): total candidate replicates evaluated, including
    // replacements. Falls back to total_replicates() when never explicitly incremented (the
    // C# -1 "not recorded" sentinel).
    int attempted_replicates() const {
        return attempted_replicates_ >= 0 ? attempted_replicates_ : total_replicates_;
    }

    // C# `FailedReplicates`: replicates that exhausted every retry attempt and fell back to
    // parent parameters (the shipped A8 semantics -- see the file header).
    int failed_replicates() const { return failed_replicates_; }

    // C# `ValidReplicates` (T19 redefinition): max(0, attempted - failed), over
    // attempted_replicates() rather than total_replicates().
    int valid_replicates() const {
        return std::max(0, attempted_replicates() - failed_replicates_);
    }

    // C# `RetainedReplicates` (T19, get/set): the number of parameter sets actually delivered
    // to the results. Falls back to valid_replicates() when never explicitly set (the C# -1
    // "not recorded" sentinel) -- true for every A8 parametric-bootstrap diagnostics object.
    int retained_replicates() const {
        return retained_replicates_ >= 0 ? retained_replicates_ : valid_replicates();
    }
    void set_retained_replicates(int value) { retained_replicates_ = value; }

    // C# `FailureRate` (T19 redefinition): failed / attempted (0 when no attempts).
    double failure_rate() const {
        int attempted = attempted_replicates();
        return attempted > 0 ? static_cast<double>(failed_replicates_) / attempted : 0.0;
    }

    // C# `TotalRetries`: cumulative retry attempts across all replicates.
    int total_retries() const { return total_retries_; }

    // C# `AverageRetries` (T19 redefinition): retries / attempted (0 when no attempts).
    double average_retries() const {
        int attempted = attempted_replicates();
        return attempted > 0 ? static_cast<double>(total_retries_) / attempted : 0.0;
    }

    // C# `TotalFunctionEvaluations`: cumulative GMM function evaluations across all replicates.
    int total_function_evaluations() const { return total_function_evaluations_; }

    // C# `AverageFunctionEvaluations` (T19 redefinition): function evaluations / attempted (0
    // when no attempts).
    double average_function_evaluations() const {
        int attempted = attempted_replicates();
        return attempted > 0 ? static_cast<double>(total_function_evaluations_) / attempted
                             : 0.0;
    }

    // C# `TransformFailures` (T19): pivot draws discarded because the link-space transform
    // failed (e.g. an unfactorable replicate covariance), distinct from a z-limit
    // PivotRejection. Only populated by the pivot (bias-corrected) bootstrap method.
    int transform_failures() const { return transform_failures_; }

    // C# `PivotRejections`: pivot draws rejected past the z-limit (pivot bootstrap only, Phase 9).
    int pivot_rejections() const { return pivot_rejections_; }

    // C# `PivotRejectionRate`: pivot rejections / total (0 when no replicates). NOT redefined
    // over attempted_replicates() -- the C# leaves this one over TotalReplicates.
    double pivot_rejection_rate() const {
        return total_replicates_ > 0
                   ? static_cast<double>(pivot_rejections_) / total_replicates_
                   : 0.0;
    }

    // C# `MahalanobisRejections`: replicates rejected via Mahalanobis-distance outlier detection.
    int mahalanobis_rejections() const { return mahalanobis_rejections_; }

    // C# `MahalanobisRejectionRate`: Mahalanobis rejections / total (0 when no replicates). NOT
    // redefined over attempted_replicates() -- the C# leaves this one over TotalReplicates too.
    double mahalanobis_rejection_rate() const {
        return total_replicates_ > 0
                   ? static_cast<double>(mahalanobis_rejections_) / total_replicates_
                   : 0.0;
    }

    // C# `StatusSuccessCount` / `StatusMaximumIterationsCount` /
    // `StatusMaximumFunctionEvaluationsCount` / `StatusFailureCount` / `StatusNoneCount` (T19):
    // the distribution of terminal OptimizationStatus values across every replicate GMM attempt
    // (including retries), recorded by record_gmm_status().
    int status_success_count() const { return status_success_count_; }
    int status_maximum_iterations_count() const { return status_maximum_iterations_count_; }
    int status_maximum_function_evaluations_count() const {
        return status_maximum_function_evaluations_count_;
    }
    int status_failure_count() const { return status_failure_count_; }
    int status_none_count() const { return status_none_count_; }

    // C# `OptimizerFallbacks` (T19): cumulative count of GMM optimization passes that fell back
    // from BFGS to Nelder-Mead (see GeneralizedMethodOfMoments::optimizer_fallback_count(), T13).
    int optimizer_fallbacks() const { return optimizer_fallbacks_; }

    // --- Mutators (C# thread-safe increment methods; plain increments under the serial port) ---

    // C# `IncrementFailed()`.
    void increment_failed() { ++failed_replicates_; }

    // C# `IncrementAttempted()` (T19): the C# lazily promotes the -1 sentinel to 0 on first
    // call via CompareExchange; the serial port does the same with a plain guard.
    void increment_attempted() {
        if (attempted_replicates_ < 0) attempted_replicates_ = 0;
        ++attempted_replicates_;
    }

    // C# `AddOptimizerFallbacks(int)` (T19).
    void add_optimizer_fallbacks(int count) {
        if (count > 0) optimizer_fallbacks_ += count;
    }

    // C# `IncrementTransformFailure()` (T19).
    void increment_transform_failure() { ++transform_failures_; }

    // C# `RecordGMMStatus(OptimizationStatus)` (T19): routes each terminal optimizer status to
    // its own counter; any status other than the four named ones (i.e. `None`) falls into the
    // `default` arm, matching the C# switch.
    void record_gmm_status(OptimizationStatus status) {
        switch (status) {
            case OptimizationStatus::Success:
                ++status_success_count_;
                break;
            case OptimizationStatus::MaximumIterationsReached:
                ++status_maximum_iterations_count_;
                break;
            case OptimizationStatus::MaximumFunctionEvaluationsReached:
                ++status_maximum_function_evaluations_count_;
                break;
            case OptimizationStatus::Failure:
                ++status_failure_count_;
                break;
            default:
                ++status_none_count_;
                break;
        }
    }

    // C# `AddRetries(int)`.
    void add_retries(int count) { total_retries_ += count; }

    // C# `AddFunctionEvaluations(int)`.
    void add_function_evaluations(int count) { total_function_evaluations_ += count; }

    // C# `IncrementPivotRejection()`.
    void increment_pivot_rejection() { ++pivot_rejections_; }

    // C# `IncrementMahalanobisRejection()`.
    void increment_mahalanobis_rejection() { ++mahalanobis_rejections_; }

   private:
    int total_replicates_ = 0;
    int attempted_replicates_ = -1;    // T19: -1 = "not recorded" sentinel (C# default).
    int failed_replicates_ = 0;
    int retained_replicates_ = -1;     // T19: -1 = "not recorded" sentinel (C# default).
    int transform_failures_ = 0;       // T19.
    int status_success_count_ = 0;                     // T19.
    int status_maximum_iterations_count_ = 0;           // T19.
    int status_maximum_function_evaluations_count_ = 0;  // T19.
    int status_failure_count_ = 0;                      // T19.
    int status_none_count_ = 0;                         // T19.
    int optimizer_fallbacks_ = 0;      // T19.
    int total_retries_ = 0;
    int total_function_evaluations_ = 0;
    int pivot_rejections_ = 0;
    int mahalanobis_rejections_ = 0;
};

}  // namespace corehydro::analyses
