// ported from: upstream/RMC-BestFit/src/RMC.BestFit/Analyses/Support/BootstrapDiagnostics.cs @ fc28c0c
//
// Diagnostic counters collected during the parametric-bootstrap (A8) and pivot-bootstrap (Phase 9)
// uncertainty paths of Bulletin17CAnalysis. The C# updates the integer counters with Interlocked
// operations for its Parallel.For loops; the port runs the bootstrap loop SERIALLY (the estimator-
// port precedent), so plain `++` / `+=` reproduce the same final counts -- no atomics needed.
//
// NAMING (documented per the A8 brief): the analyses layer uses snake_case members/accessors
// (mirroring the A7 Bulletin17CAnalysis header), so the C# PascalCase properties become snake_case
// here (`TotalReplicates` -> total_replicates() / set_total_replicates(), `IncrementFailed()` ->
// increment_failed(), etc.). The derived rate accessors guard division by zero exactly as the C#
// (`_totalReplicates > 0 ? ... : 0.0`).
//
// SKIPPED C# surface (each dropped for the shipped scope; see the report):
//   * ToXElement / FromXElement -- the XML (de)serialization surface, dropped project-wide.
//   * Phase1Time / Phase2Time / Phase3Time (TimeSpan) + the Stopwatch plumbing -- run-lifecycle
//     timing; OMITTED entirely (no field), since nothing on the shipped compute path reads them.
//   * The Interlocked atomics -- collapse to plain increments under the serial port.
#pragma once

namespace bestfit::analyses {

class BootstrapDiagnostics {
   public:
    // --- Counters (C# properties) ----------------------------------------------------------

    // C# `TotalReplicates` (settable): total bootstrap replicates requested.
    int total_replicates() const { return total_replicates_; }
    void set_total_replicates(int value) { total_replicates_ = value; }

    // C# `FailedReplicates`: replicates that failed all retries and fell back to parent params.
    int failed_replicates() const { return failed_replicates_; }

    // C# `ValidReplicates`: successfully-estimated replicates (total - failed).
    int valid_replicates() const { return total_replicates_ - failed_replicates_; }

    // C# `FailureRate`: failed / total (0 when no replicates).
    double failure_rate() const {
        return total_replicates_ > 0
                   ? static_cast<double>(failed_replicates_) / total_replicates_
                   : 0.0;
    }

    // C# `TotalRetries`: cumulative retry attempts across all replicates.
    int total_retries() const { return total_retries_; }

    // C# `AverageRetries`: retries / total (0 when no replicates).
    double average_retries() const {
        return total_replicates_ > 0 ? static_cast<double>(total_retries_) / total_replicates_
                                     : 0.0;
    }

    // C# `TotalFunctionEvaluations`: cumulative GMM function evaluations across all replicates.
    int total_function_evaluations() const { return total_function_evaluations_; }

    // C# `AverageFunctionEvaluations`: function evaluations / total (0 when no replicates).
    double average_function_evaluations() const {
        return total_replicates_ > 0
                   ? static_cast<double>(total_function_evaluations_) / total_replicates_
                   : 0.0;
    }

    // C# `PivotRejections`: pivot draws rejected past the z-limit (pivot bootstrap only, Phase 9).
    int pivot_rejections() const { return pivot_rejections_; }

    // C# `PivotRejectionRate`: pivot rejections / total (0 when no replicates).
    double pivot_rejection_rate() const {
        return total_replicates_ > 0
                   ? static_cast<double>(pivot_rejections_) / total_replicates_
                   : 0.0;
    }

    // C# `MahalanobisRejections`: replicates rejected via Mahalanobis-distance outlier detection.
    int mahalanobis_rejections() const { return mahalanobis_rejections_; }

    // C# `MahalanobisRejectionRate`: Mahalanobis rejections / total (0 when no replicates).
    double mahalanobis_rejection_rate() const {
        return total_replicates_ > 0
                   ? static_cast<double>(mahalanobis_rejections_) / total_replicates_
                   : 0.0;
    }

    // --- Mutators (C# thread-safe increment methods; plain increments under the serial port) ---

    // C# `IncrementFailed()`.
    void increment_failed() { ++failed_replicates_; }

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
    int failed_replicates_ = 0;
    int total_retries_ = 0;
    int total_function_evaluations_ = 0;
    int pivot_rejections_ = 0;
    int mahalanobis_rejections_ = 0;
};

}  // namespace bestfit::analyses
