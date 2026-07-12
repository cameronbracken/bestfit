// ported from: RMC-BestFit/src/RMC.BestFit/Models/UnivariateDistribution/Base/
//              UnivariateDistributionModelBase.cs @ fc28c0c
//
// Abstract base for univariate distribution models: a shared DataFrame input used for
// likelihood evaluation, the Jeffreys-rule-for-scale toggle, and the IQuantilePriors state
// (the pure-virtual surface comes from models/support/i_quantile_priors.hpp; this base
// supplies the backing state and accessors). Concrete models (UnivariateDistributionModel
// here; the Mixture / CompetingRisks / PointProcess siblings in M10-M12) derive from this and
// implement SetDefaultQuantilePriors / ProcessQuantilePriors plus the ModelBase likelihood
// surface.
//
// DATAFRAME OWNERSHIP AND NULLABILITY: the C# `_dataFrame` is a nullable reference; the C++
// DataFrame is a move-only value type, held here as `std::optional<DataFrame>` (empty ==
// C# null, e.g. after the parameterless model constructor). `data_frame()` is an unguarded
// deref like the other non-null-by-contract accessors (QuantilePrior::distribution());
// check `has_data_frame()` first on paths where the frame may be absent. There is no way to
// assign "null" through the setter -- the C# `DataFrame = null` state exists in C++ only as
// the never-set state.
//
// PROCESS-THRESHOLD-SERIES CADENCE (the M4 -> M8 contract; replaces the C# INPC plumbing).
// Reading of the C# event flow, documented here as M5 did for plotting positions:
//   1. Upstream, the DataFrame itself reprocesses thresholds on every series mutation (its
//      own CollectionChanged handlers run CalculatePlottingPositions, which calls
//      ProcessThresholdSeries first), so a frame handed to a model is already processed.
//   2. The concrete C# UnivariateDistribution.DataFrame setter (UnivariateDistribution.cs
//      line 289) calls `_dataFrame.ProcessThresholdSeries()` again at assignment, then --
//      when UseDefaultFlatPriors -- SetDefaultParameters(). (The C# BASE setter, lines
//      94-114, only subscribes and calls SetDefaultParameters; the reprocess-at-set-time
//      behavior comes from the concrete override, and the sibling models get it on
//      subsequent mutations via DataFrame_PropertyChanged.)
//   3. The likelihood entry points (LogLikelihood, DataLogLikelihood, the Pointwise*
//      variants, StationaryData_LogLikelihood) NEVER call ProcessThresholdSeries. The
//      C# guard tests rely on this: they overwrite threshold counts after model
//      construction, and the likelihood must not reprocess/overwrite them.
// Because the C++ DataFrame (M4) replaced INPC with explicit invalidation -- and
// ProcessThresholdSeries is NOT idempotent when explicit points exactly cover
// Duration - NumberAbove -- this base calls process_threshold_series() exactly ONCE per
// model-boundary assignment, in set_data_frame(), and never before likelihood evaluation.
// Callers that mutate the held frame's series in place afterwards own the C# "event"
// obligation: call data_frame().process_threshold_series() themselves (or re-set the frame)
// -- mirroring the upstream once-per-mutation cadence.
//
// The C# DataFrame_PropertyChanged handler (lines 234-249: ignore PlottingParameter /
// PlottingPosition changes, else ProcessThresholdSeries + SetDefaultParameters-if-flat)
// ports as the protected virtual data_frame_property_changed() below; with no events the
// only built-in trigger is set_data_frame(), whose effective body is identical. M9+ (the
// nonstationary path) overrides it for the full-time-series/trend side effects.
//
// The C# QuantilePrior_PropertyChanged handler (lines 265-270) re-ran ProcessQuantilePriors
// whenever any prior in the list changed; with no events the C++ setter
// set_quantile_priors() calls process_quantile_priors() after replacing the list (the
// setter's own C# side effect, line 170), and callers that mutate a stored QuantilePrior in
// place afterwards must call process_quantile_priors() themselves.
//
// SKIPPED (project-wide deferrals): INotifyPropertyChanged / RaisePropertyChange /
// event (un)subscription bodies, and the [Category]/[DisplayName]/[Description]/[Browsable]
// attributes (WPF display concerns).
#pragma once
#include <optional>
#include <utility>
#include <vector>

#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/support/i_quantile_priors.hpp"
#include "corehydro/models/support/model_base.hpp"
#include "corehydro/models/support/quantile_prior.hpp"

namespace corehydro::models {

class UnivariateDistributionModelBase : public ModelBase, public IQuantilePriors {
   public:
    UnivariateDistributionModelBase() = default;
    ~UnivariateDistributionModelBase() override = default;

    // Move-only, like the DataFrame it owns (deep copies are a concrete-model Clone()
    // concern -- M9).
    UnivariateDistributionModelBase(const UnivariateDistributionModelBase&) = delete;
    UnivariateDistributionModelBase& operator=(const UnivariateDistributionModelBase&) = delete;
    UnivariateDistributionModelBase(UnivariateDistributionModelBase&&) = default;
    UnivariateDistributionModelBase& operator=(UnivariateDistributionModelBase&&) = default;

    // --- DataFrame (C# property, lines 94-114; nullability note in the header). ---
    bool has_data_frame() const { return data_frame_.has_value(); }
    DataFrame& data_frame() { return *data_frame_; }
    const DataFrame& data_frame() const { return *data_frame_; }

    // Assigns the input data frame (C# setter, plus the model-boundary
    // process_threshold_series() contract -- see the cadence note in the header). Mirrors
    // the C# order: store, reprocess thresholds, then SetDefaultParameters() when
    // UseDefaultFlatPriors (concrete setter, UnivariateDistribution.cs 275-319, minus the
    // nonstationary block deferred to M9).
    virtual void set_data_frame(DataFrame data_frame) {
        data_frame_ = std::move(data_frame);
        data_frame_->process_threshold_series();
        if (use_default_flat_priors()) set_default_parameters();
    }

    // --- UseJeffreysRuleForScale (C# lines 130-146; default true): prior on the scale
    // parameter sigma proportional to 1/sigma, the Jeffreys noninformative prior for a pure
    // scale parameter. Hoisted here (M8) from the Phase 4 UnivariateDistributionModel. ---
    bool use_jeffreys_rule_for_scale() const { return use_jeffreys_rule_for_scale_; }
    void set_use_jeffreys_rule_for_scale(bool value) { use_jeffreys_rule_for_scale_ = value; }

    // --- IQuantilePriors surface (state + accessors; C# lines 148-213). ---

    std::vector<QuantilePrior>& quantile_priors() override { return quantile_priors_; }
    const std::vector<QuantilePrior>& quantile_priors() const override { return quantile_priors_; }

    // C# setter (lines 151-173): replaces the list, rewires the (unported) INPC handlers,
    // and recomputes the processed priors for use in the likelihood.
    void set_quantile_priors(std::vector<QuantilePrior> quantile_priors) override {
        quantile_priors_ = std::move(quantile_priors);
        process_quantile_priors();
    }

    // C# EnableQuantilePriors (lines 181-198; default false). The setter resets the default
    // quantile priors on change, exactly like the C#.
    bool enable_quantile_priors() const override { return enable_quantile_priors_; }
    void set_enable_quantile_priors(bool enable_quantile_priors) override {
        if (enable_quantile_priors_ != enable_quantile_priors) {
            enable_quantile_priors_ = enable_quantile_priors;
            set_default_quantile_priors();
        }
    }

    // C# UseSingleQuantile (lines 200-213; default false).
    bool use_single_quantile() const override { return use_single_quantile_; }
    void set_use_single_quantile(bool use_single_quantile) override {
        if (use_single_quantile_ != use_single_quantile) {
            use_single_quantile_ = use_single_quantile;
            set_default_quantile_priors();
        }
    }

    // set_default_quantile_priors() / process_quantile_priors() stay pure virtual (C# lines
    // 272-275), declared on IQuantilePriors; concrete models implement them.

   protected:
    // C# DataFrame_PropertyChanged (lines 234-249): the explicit-invalidation equivalent for
    // callers/derived classes that mutate the held frame in place (see the cadence note; the
    // PlottingParameter/PlottingPosition filtering has no C++ trigger to filter). NEVER
    // called from the likelihood paths.
    virtual void data_frame_property_changed() {
        if (!data_frame_) return;
        data_frame_->process_threshold_series();
        if (use_default_flat_priors()) set_default_parameters();
    }

    std::optional<DataFrame> data_frame_;          // C# _dataFrame (line 43)
    std::vector<QuantilePrior> quantile_priors_;   // C# _quantilePriors (line 48)
    std::vector<QuantilePrior> quantile_priors_true_;  // C# _quantilePriorsTrue (line 55)
    bool use_jeffreys_rule_for_scale_ = true;      // C# _useJeffreysRuleForScale (line 60)
    bool enable_quantile_priors_ = false;          // C# _enableQuantilePriors (line 65)
    bool use_single_quantile_ = false;             // C# _useSingleQuantile (line 70)
};

}  // namespace corehydro::models
