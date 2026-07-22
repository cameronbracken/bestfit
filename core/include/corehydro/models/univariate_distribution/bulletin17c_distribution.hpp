// ported from: RMC-BestFit/src/RMC.BestFit/Models/UnivariateDistribution/
//              Bulletin17CDistribution.cs @ c2e6192
//
// Bulletin 17C (B17C) distribution model. Task B9 landed the CONSTRUCTION slice: the ctors,
// the six-family distribution factory + supported-type gate, LinkController wiring,
// parameter initialization (SetInitialParameters / SetDefaultParameters), the penalty
// surface (SetUpQuantilePenalties, SetPenaltyFunction, SetRandomPenaltyFunction),
// SetParameterValues, Validate, Clone, and GenerateRandomValues. Task B10 adds the moment
// machinery: MomentConditions, RepairErrors, Mu456_Pearson3_FromSigmaGamma,
// ConditionalMomentsForUncertainData, MomentConditionsForUncertainData,
// UpdateMomentMeanCovariance, CensoringAsymmetryScore,
// WeightedErrorDirectionScore(+FromLinked), PointwiseMomentConditionsImpl,
// QuantileGradient, and QuantileVariance. The class stays ONE class mirroring the one C#
// file; to keep this header under the repo's file-size cap the moment-machinery method
// BODIES live in the companion header bulletin17c_moment_machinery.hpp (an .ipp-style
// include pulled in at the bottom of this file), with the declarations and their C# line
// anchors below.
//
// Type-system design note (kept from the C#): Bulletin17CDistribution intentionally does
// NOT derive from ModelBase and does NOT implement IModel. The Bulletin 17C procedure is
// fit by Generalized Method of Moments (GMM) only -- there is no log-likelihood and no
// posterior MCMC chain to expose. Implementing only IGMMModel, ISimulatable, and
// IUnivariateModel keeps the surface minimal and prevents Bayesian-API consumers from
// accidentally calling LogLikelihood-shaped methods that would have no defined meaning
// here.
//
// INPC replacement (the DataFrame-port strategy): the C# wires PropertyChanged handlers
// (DataFrame_PropertyChanged, Parameter_PropertyChanged, ParameterPenalty_PropertyChanged,
// QuantilePenalty_PropertyChanged) so that mutations re-run ProcessThresholdSeries /
// SetDefaultParameters automatically. The C++ port replaces the event plumbing with DIRECT
// CALLS at the trigger points the port keeps: set_data_frame() reprocesses thresholds and
// calls set_default_parameters() itself, set_distribution() calls
// set_default_parameters() itself, and callers that mutate the held frame's series in
// place afterwards own the reprocess/reseed obligation (the M4->M8 cadence contract; see
// models/data_frame/data_frame.hpp).
//
// DataFrame nullability: the C# `DataFrame` is a nullable reference; the C++ DataFrame is
// a move-only value type, held as std::optional<DataFrame> (empty == C# null, e.g. after
// the parameterless constructor) -- the UnivariateDistributionModelBase precedent.
// data_frame() is an unguarded deref; check has_data_frame() first where the frame may be
// absent. The C# `ArgumentNullException` for a null DataFrame argument is structurally
// unrepresentable for a by-value parameter and has no port.
//
// Distribution ownership: the C# `(DataFrame, UnivariateDistributionBase)` ctor CLONES the
// supplied distribution to avoid aliasing; the C++ ctor takes a std::unique_ptr and the
// caller transfers sole ownership, so aliasing is impossible by construction (the
// UnivariateDistributionModel precedent) -- the defensive clone is therefore not repeated.
//
// LinkController: default-constructed = no links = identity (B1's null-means-identity
// port). The C# setter replaces null with a fresh default controller; the C++
// LinkController is a non-nullable value, so set_link_controller() takes it by value
// (null unrepresentable). The XElement link (de)serialization paths (which used B2's
// BestFitLinkFunctionFactory) are XML-skipped; the member/InverseLink usage is ported.
//
// ModelParameter names: the C# reads `Distribution.ParametersToString[i, 0]` for the
// parameter Name (and the penalty Name via DisplayName). ParametersToString is not on the
// ported distribution base (Phase 4 decision, display-only surface), so parameter and
// penalty names stay "" -- the MixtureModel precedent (see mixture_model.hpp's header).
//
// ((IMomentEstimation)Distribution).ParametersFromMoments: the ported distributions carry
// `parameters_from_moments` non-virtually (no IMomentEstimation mixin exists in the C++
// port -- a B4 scope decision), so the cast ports as a static per-type dispatch over
// exactly the six supported families (the only types reachable here; an unsupported type
// throws, feeding the same catch the failed C# cast would).
//
// Clone(): the C# clones by round-tripping through XElement to avoid the public
// constructor's SetDefaultParameters() call (which would overwrite user-defined
// penalties). XML is skipped, so the port clones DIRECTLY via a private copy path with
// the same observable result: parameters/penalties copied verbatim (no reseeding), the
// stored PenaltyFunction NOT carried over (the C# XElement round trip does not serialize
// it), and the clone's DataFrame reprocessed exactly as the XElement ctor's DataFrame
// setter does. Two documented deviations: (1) the C# clone SHARES the DataFrame reference
// while the C++ clone deep-copies the value-typed frame (reference sharing is
// unrepresentable; same DataFrame-port precedent); (2) a non-empty LinkController cannot
// yet be deep-copied (the ported ILinkFunction has no clone()), so clone() throws
// std::logic_error if links are installed -- no B9 code path installs links; the WEDS /
// uncertainty slice (B10+) owns the follow-up.
//
// CloneWithDataFrame() (v2.0.0, upstream-sync Task 18, 0dc8594 "Preserve B17C model state
// on bootstrap clones"): C# adds `CloneWithDataFrame(DataFrame dataFrame)`, which returns
// `new Bulletin17CDistribution(dataFrame, ToXElement())` -- the SAME XElement round trip
// Clone() uses, so it restores Distribution/Parameters/ParameterPenalties/
// QuantilePenalties/LinkController from the current snapshot (the XElement ctor's
// `_isDeserializing = true` suppresses SetDefaultParameters), but binds the round trip to
// the CALLER-SUPPLIED frame instead of the source object's own DataFrame. The port mirrors
// this via clone_core_state(), the SAME private deep-copy helper clone() now calls,
// followed by binding + reprocessing the supplied frame instead of the source's -- so both
// methods share one code path for the "restore serialized state, don't rebuild it" contract
// (the point of this task: bootstrap replicates in the T19/20 pivotal-bootstrap workflow
// call this so each replicate estimates from the PARENT's fitted parameters and enabled
// penalties, not from freshly-defaulted ones derived from the resampled data). Per the C#
// doc remarks: assigning the resample through the public DataFrame setter instead would
// invoke SetDefaultParameters, discarding the cloned initial values, disabling every
// parameter penalty, and -- when default-parameter derivation fails for the resampled frame
// -- emptying the parameter list entirely. `CloneWithDataFrame(null)`'s C#
// ArgumentNullException has no port (same structurally-unrepresentable-for-a-value-typed-
// DataFrame-argument precedent noted above for the ctors); the C++ signature takes
// `const DataFrame&` (never null) and deep-copies it, so the caller's original frame is
// left untouched -- matching the "supplied frame bound to the clone" observable behavior
// even though the C# reference-sharing itself has no analog.
//
// SetDefaultParameters ROS-trigger narrowing (v2.0.0, upstream-sync Task 18, c420d48
// "Improving default parameter settings for the B17 distribution"): the shared
// compute_default_initials() helper's nonparametric-ROS-override condition narrows from
// `NumberOfLowOutliers > 0 || UncertainSeries.Count > 0 || IntervalSeries.Count > 0 ||
// ThresholdSeries.Count > 0` to `NumberOfLowOutliers > 0 || ThresholdSeries.Count > 0` --
// uncertain/interval data alone no longer triggers the ROS nonparametric-moment override;
// those fits now seed from the plain IMaximumLikelihoodEstimation constraint initials, like
// exact-only data. GMM starting values (and, since BFGS is not path-independent for every
// data configuration, potentially the converged fit) change for interval/uncertain-only
// fits versus the pre-v2.0.0 port; low-outlier and threshold fits are unaffected (both
// legs of the condition are untouched by the narrowing).
//
// Debug.WriteLine / swallowed-exception guards (SetInitialParameters' catch,
// SetDefaultParameters' catch): silent no-throw guards per the repo convention -- the
// catch BODIES are ported, only the trace text is not.
//
// Deliberately NOT ported (project-wide deferrals): the two XElement ctors (C# lines 106,
// 183), ToXElement (line 2424), INotifyPropertyChanged / RaisePropertyChange and the four
// PropertyChanged handlers (lines 514-581; replaced by the direct-call strategy above),
// the `_isDeserializing` flag itself (XML-only; the effective non-deserializing setter
// behavior is preserved), and the [Category]/[DisplayName]/[Description]/[Browsable]
// attributes.
//
// EXCEPTION-TYPE MAPPING for THIS file: C# ArgumentNullException/ArgumentException ->
// std::invalid_argument; ArgumentOutOfRangeException -> std::out_of_range;
// InvalidOperationException -> std::runtime_error (the model-layer mapping; see
// models/support/simulatable.hpp).
#pragma once
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/estimation/gmm_delegates.hpp"
#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/support/i_gmm_model.hpp"
#include "corehydro/models/support/i_univariate_model.hpp"
#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/models/support/parameter_penalty.hpp"
#include "corehydro/models/support/quantile_penalty.hpp"
#include "corehydro/models/support/simulatable.hpp"
#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/exponential.hpp"
#include "corehydro/numerics/distributions/gamma_distribution.hpp"
#include "corehydro/numerics/distributions/log_normal.hpp"
#include "corehydro/numerics/distributions/log_pearson_type_iii.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/pearson_type_iii.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/functions/link_controller.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models {

class Bulletin17CDistribution : public IGMMModel,
                                public ISimulatable<std::vector<double>>,
                                public IUnivariateModel {
   public:
    using DistributionBase = numerics::distributions::UnivariateDistributionBase;
    using DistributionType = numerics::distributions::UnivariateDistributionType;
    using LinkController = numerics::functions::LinkController;

    // --- Construction --------------------------------------------------------------------

    // Constructs a new Bulletin 17C (B17C) distribution model using a Log-Pearson Type III
    // distribution and no data (C# line 50).
    Bulletin17CDistribution() {
        set_distribution(create_distribution(DistributionType::LogPearsonTypeIII));
        set_up_quantile_penalties();
    }

    // Constructs a new Bulletin 17C (B17C) distribution model from a data frame and a
    // distribution instance (C# line 64). The C# clones the supplied distribution; the C++
    // caller transfers sole ownership instead (see the header note). A null distribution
    // throws std::invalid_argument (C# ArgumentNullException).
    Bulletin17CDistribution(DataFrame data_frame, std::unique_ptr<DistributionBase> distribution) {
        if (distribution == nullptr)
            throw std::invalid_argument("The distribution cannot be null.");

        set_distribution(std::move(distribution));
        set_data_frame(std::move(data_frame));
        set_up_quantile_penalties();
    }

    // Constructs a new Bulletin 17C (B17C) distribution model from a data frame and a
    // distribution type (C# line 82).
    Bulletin17CDistribution(DataFrame data_frame, DistributionType distribution_type) {
        set_distribution(create_distribution(distribution_type));
        set_data_frame(std::move(data_frame));
        set_up_quantile_penalties();
    }

    // (The two XElement constructors, C# lines 106 and 183, are XML-skipped.)

    // The model owns move-only state (DataFrame, unique_ptr distribution); use clone()
    // for a deep copy, as the C# Clone() contract intends. Moves are DELETED (B10, the B9
    // reviewer follow-up): set_penalty_function / set_random_penalty_function store
    // this-capturing closures in penalty_function_, and the accessor-returned delegate
    // closures capture this too, so a defaulted move would leave the moved-into object's
    // stored penalty function pointing at the moved-from object. The C# reference type
    // has no move to mirror; deleting them is the safe structural analog.
    Bulletin17CDistribution(const Bulletin17CDistribution&) = delete;
    Bulletin17CDistribution& operator=(const Bulletin17CDistribution&) = delete;
    Bulletin17CDistribution(Bulletin17CDistribution&&) = delete;
    Bulletin17CDistribution& operator=(Bulletin17CDistribution&&) = delete;
    ~Bulletin17CDistribution() override = default;

    // --- Members -------------------------------------------------------------------------

    // Determines whether the specified distribution type is supported by the Bulletin 17C
    // model (C# line 279): Exponential, Gamma, Log-Normal, Log-Pearson Type III, Normal,
    // and Pearson Type III.
    static bool is_supported_distribution_type(DistributionType distribution_type) {
        return distribution_type == DistributionType::Exponential ||
               distribution_type == DistributionType::GammaDistribution ||
               distribution_type == DistributionType::LogNormal ||
               distribution_type == DistributionType::LogPearsonTypeIII ||
               distribution_type == DistributionType::Normal ||
               distribution_type == DistributionType::PearsonTypeIII;
    }

    // --- LinkController (C# property, line 308): manages per-parameter link functions that
    // transform parameters between real-space and link-space during GMM estimation. The C#
    // setter replaces null with a fresh default; a null value is unrepresentable here. ---
    LinkController& link_controller() { return link_controller_; }
    const LinkController& link_controller() const { return link_controller_; }
    void set_link_controller(LinkController value) { link_controller_ = std::move(value); }

    // --- DataFrame (C# property, line 340; nullability note in the header). ---
    bool has_data_frame() const { return data_frame_.has_value(); }
    DataFrame& data_frame() override { return *data_frame_; }
    const DataFrame& data_frame() const override { return *data_frame_; }

    // Assigns the input data frame (C# setter body: unsubscribe/subscribe events skipped;
    // ProcessThresholdSeries, then SetDefaultParameters when not deserializing -- the port
    // never deserializes).
    void set_data_frame(DataFrame value) {
        data_frame_ = std::move(value);
        data_frame_->process_threshold_series();
        set_default_parameters();
    }

    // --- Distribution: the parent probability distribution (C# property, line 371). Never
    // null after construction; the setter throws on null (C# ArgumentNullException) and
    // triggers SetDefaultParameters (the not-deserializing branch). ---
    const DistributionBase* distribution() const override { return distribution_.get(); }
    DistributionBase* mutable_distribution() { return distribution_.get(); }
    void set_distribution(std::unique_ptr<DistributionBase> value) {
        if (value == nullptr)
            throw std::invalid_argument("The distribution cannot be null.");
        distribution_ = std::move(value);
        set_default_parameters();
    }

    // Bulletin 17C analysis assumes a stationary parent population -- there is no
    // trend-on-parameters concept in the B17C methodology. Always false (C# line 395).
    bool is_nonstationary() const override { return false; }

    // --- DistributionType (C# property, line 400): the setter swaps in a fresh default
    // distribution of the new type (no-op when the type is unchanged). ---
    DistributionType distribution_type() const { return distribution_->type(); }
    void set_distribution_type(DistributionType value) {
        if (distribution_ != nullptr && distribution_->type() == value) return;
        set_distribution(create_distribution(value));
    }

    // The list of model parameters (C# line 412; the private setter's INPC rewiring is
    // replaced by the direct-call strategy).
    std::vector<ModelParameter>& parameters() override { return parameters_; }
    const std::vector<ModelParameter>& parameters() const override { return parameters_; }

    // Returns the number of model parameters (C# line 434).
    int number_of_parameters() const override { return static_cast<int>(parameters_.size()); }

    // Returns the number of moment conditions used in the GMM estimation (C# line 437).
    int number_of_moment_conditions() const override {
        return static_cast<int>(parameters_.size());
    }

    // The collection of parameter penalties (C# line 442).
    std::vector<ParameterPenalty>& parameter_penalties() { return parameter_penalties_; }
    const std::vector<ParameterPenalty>& parameter_penalties() const {
        return parameter_penalties_;
    }

    // The collection of quantile penalties (C# line 467).
    std::vector<QuantilePenalty>& quantile_penalties() { return quantile_penalties_; }
    const std::vector<QuantilePenalty>& quantile_penalties() const {
        return quantile_penalties_;
    }

    // The total sample size (C# line 489: `DataFrame?.TotalRecordLength() ?? 0`).
    int sample_size() const override {
        return has_data_frame() ? data_frame().total_record_length() : 0;
    }

    // The moment condition function (C# line 492: `=> MomentConditions`). The C# method
    // group conversion becomes a this-capturing closure; the caller-lifetime contract is
    // the GMM estimator's (the model must outlive the returned callable).
    estimation::MomentConditionFunction moment_condition_function() const override {
        return [this](const std::vector<double>& parameters) {
            return moment_conditions(parameters);
        };
    }

    // The optional analytical Jacobian function (C# line 495: `=> null`).
    estimation::JacobianFunction jacobian_function() const override {
        return estimation::JacobianFunction{};
    }

    // The optional penalty function (C# line 498: the stored `_penaltyFunction`, empty
    // until set_penalty_function()/set_random_penalty_function() builds it).
    estimation::PenaltyFunction penalty_function() const override { return penalty_function_; }

    // The pointwise moment condition function (C# line 501:
    // `=> PointwiseMomentConditionsImpl`; same closure/lifetime note as above).
    estimation::PointwiseMomentConditionFunction pointwise_moment_conditions() const override {
        return [this](const std::vector<double>& parameters) {
            return pointwise_moment_conditions_impl(parameters);
        };
    }

    // --- Methods -------------------------------------------------------------------------

    // Creates a univariate distribution instance from a distribution type (C# line 591).
    // Throws std::out_of_range (C# ArgumentOutOfRangeException) for unsupported types.
    static std::unique_ptr<DistributionBase> create_distribution(
        DistributionType distribution_type) {
        if (distribution_type == DistributionType::Exponential)
            return std::make_unique<numerics::distributions::Exponential>();
        else if (distribution_type == DistributionType::GammaDistribution)
            return std::make_unique<numerics::distributions::GammaDistribution>();
        else if (distribution_type == DistributionType::LogNormal)
            return std::make_unique<numerics::distributions::LogNormal>();
        else if (distribution_type == DistributionType::LogPearsonTypeIII)
            return std::make_unique<numerics::distributions::LogPearsonTypeIII>();
        else if (distribution_type == DistributionType::Normal)
            return std::make_unique<numerics::distributions::Normal>();
        else if (distribution_type == DistributionType::PearsonTypeIII)
            return std::make_unique<numerics::distributions::PearsonTypeIII>();
        else
            throw std::out_of_range("Unsupported distribution type.");
    }

    // Sets initial parameter values based on the input data and distribution constraints
    // (C# line 612). Any failure (no data frame, failed capability cast, non-finite
    // constraint math) falls back to the midpoint of each parameter's bounds -- the C#
    // catch block, with the Debug.WriteLine trace dropped (silent guard).
    void set_initial_parameters() {
        try {
            std::vector<double> initials;
            std::vector<double> lowers;
            std::vector<double> uppers;
            compute_default_initials(initials, lowers, uppers);

            // Clamp initials to bounds (.at(): the C# List/array indexers throw on a
            // too-short list -- in the try loop that throw feeds the catch fallback,
            // exactly like the C#).
            for (int i = 0; i < distribution_->number_of_parameters(); i++) {
                std::size_t si = static_cast<std::size_t>(i);
                if (initials.at(si) < lowers.at(si) || initials.at(si) > uppers.at(si)) {
                    initials.at(si) = 0.5 * (lowers.at(si) + uppers.at(si));
                }
                parameters_.at(si).set_value(initials.at(si));
            }
        } catch (...) {
            // C#: Debug.WriteLine("Parameter initialization failed.") -- silent guard.
            // Make initials mid point of bounds (.at(): a throw here propagates, exactly
            // like the C# indexer inside the catch block).
            for (int i = 0; i < distribution_->number_of_parameters(); i++) {
                std::size_t si = static_cast<std::size_t>(i);
                parameters_.at(si).set_value(0.5 * (parameters_.at(si).lower_bound() +
                                                    parameters_.at(si).upper_bound()));
            }
        }
    }

    // Sets the default parameters and priors for this model (C# line 683). The empty /
    // invalid-DataFrame branch builds parameter NAME SHELLS (values/bounds left at
    // defaults) so the UI penalty grid has the correct row count before InputData is
    // selected; the full branch seeds values from the distribution constraints (with the
    // nonparametric ROS override for low-outlier/threshold data -- narrowed in v2.0.0, see
    // the file header), Uniform(lower, upper) priors, and one ParameterPenalty per
    // parameter. The catch leaves the lists empty (silent guard; C# Debug.WriteLine
    // dropped).
    void set_default_parameters() override {
        // (Old-handler removal for parameters and parameter penalties is INPC plumbing,
        // skipped.)
        parameters_.clear();
        parameter_penalties_.clear();

        if (distribution_ == nullptr || !has_data_frame() ||
            !data_frame().validate().is_valid || data_frame().exact_series().count() == 0) {
            // Build parameter name shells so the UI penalty grid shows the correct rows
            // even before InputData is selected. Values/bounds are left at defaults.
            // (Names stay "" -- ParametersToString is not on the ported distribution
            // base; see the file header.)
            if (distribution_ != nullptr) {
                for (int i = 0; i < distribution_->number_of_parameters(); i++) {
                    ModelParameter parameter;
                    parameter.set_name("");
                    parameters_.push_back(std::move(parameter));

                    // Add penalty for parameter
                    ParameterPenalty penalty;
                    penalty.set_name(parameters_.back().display_name());
                    parameter_penalties_.push_back(std::move(penalty));
                }
            }
            return;
        }

        try {
            std::vector<double> initials;
            std::vector<double> lowers;
            std::vector<double> uppers;
            compute_default_initials(initials, lowers, uppers);

            // Clamp initials to bounds
            for (int i = 0; i < distribution_->number_of_parameters(); i++) {
                std::size_t si = static_cast<std::size_t>(i);
                if (initials[si] < lowers[si] || initials[si] > uppers[si]) {
                    initials[si] = 0.5 * (lowers[si] + uppers[si]);
                }
            }

            // Build model parameters with uniform priors (Name "" -- see the file header)
            for (int i = 0; i < distribution_->number_of_parameters(); i++) {
                std::size_t si = static_cast<std::size_t>(i);
                parameters_.emplace_back(
                    "", "", initials[si], lowers[si], uppers[si],
                    std::make_unique<numerics::distributions::Uniform>(lowers[si], uppers[si]));

                // Add penalty for parameter
                ParameterPenalty penalty;
                penalty.set_name(parameters_.back().display_name());
                parameter_penalties_.push_back(std::move(penalty));
            }
        } catch (...) {
            // C#: Debug.WriteLine("Bulletin17CDistribution.SetDefaultParameters failed:
            // ...") -- silent guard. If parameter initialization fails, leave parameters
            // empty.
            parameters_.clear();
            parameter_penalties_.clear();
        }
    }

    // Sets the model parameter values (C# line 828, virtual): writes each
    // Parameters[i].Value and pushes the values into the distribution. A wrong-length list
    // throws std::invalid_argument (C# ArgumentException).
    void set_parameter_values(const std::vector<double>& parameters) override {
        if (static_cast<int>(parameters.size()) != number_of_parameters())
            throw std::invalid_argument("The list of parameter values are the wrong length");
        for (int i = 0; i < number_of_parameters(); i++)
            parameters_[static_cast<std::size_t>(i)].set_value(
                parameters[static_cast<std::size_t>(i)]);
        distribution_->set_parameters(parameters);
    }

    // Builds and stores the penalty function (C# line 841). With no enabled penalties the
    // stored function is cleared. Deterministic (prng == nullptr): the closure sums each
    // enabled ParameterPenalty at theta[i] and each enabled QuantilePenalty at the fitted
    // quantile InverseCDF(1 - AEP) of a cloned distribution, with theta =
    // LinkController.InverseLink(parameters). Random (prng != nullptr): every enabled
    // penalty is CLONED and its Mean perturbed by sqrt(MSE) * StandardZ(prng.NextDouble())
    // BEFORE the closure is built -- the C# draw order (all parameter penalties first,
    // then all quantile penalties) is oracle-relevant for seeded streams and is mirrored
    // exactly. The C# `Random` parameter is the port's MersenneTwister (the repo's Random
    // mapping); nullptr = the C# null = deterministic.
    void set_penalty_function(numerics::sampling::MersenneTwister* prng = nullptr) {
        // Determine if there any penalties
        int n_penalty = count_enabled_penalties();
        if (n_penalty == 0) {
            penalty_function_ = nullptr;
            return;
        }

        // (C# dereferences DataFrame here unconditionally -- NullReferenceException on a
        // null frame; an empty optional cannot deref safely, so the throw is explicit.)
        if (!has_data_frame()) throw std::runtime_error("DataFrame is null.");
        int nt = data_frame().total_record_length();

        // if prng is null, penalty function is deterministic
        if (prng == nullptr) {
            penalty_function_ = [this, nt](const std::vector<double>& parameters) {
                std::vector<double> theta = link_controller_.inverse_link(parameters);
                double result = 0;
                for (std::size_t i = 0; i < parameter_penalties_.size(); i++) {
                    if (parameter_penalties_[i].enabled()) {
                        result += parameter_penalties_[i].function(theta[i], nt);
                    }
                }
                for (std::size_t i = 0; i < quantile_penalties_.size(); i++) {
                    if (quantile_penalties_[i].enabled()) {
                        auto dist = distribution_->clone();
                        dist->set_parameters(theta);
                        double aep = quantile_penalties_[i].aep();
                        result +=
                            quantile_penalties_[i].function(dist->inverse_cdf(1 - aep), nt);
                    }
                }
                return result;
            };
            return;
        } else {
            // Create random parameter penalties
            std::vector<ParameterPenalty> param_penalties;
            param_penalties.reserve(parameter_penalties_.size());
            for (std::size_t i = 0; i < parameter_penalties_.size(); i++) {
                param_penalties.push_back(parameter_penalties_[i].clone());
                if (param_penalties[i].enabled()) {
                    param_penalties[i].set_mean(
                        param_penalties[i].mean() +
                        std::sqrt(param_penalties[i].mse()) *
                            numerics::distributions::Normal::standard_z(prng->next_double()));
                }
            }

            // Create random quantile penalties
            std::vector<QuantilePenalty> quant_penalties;
            quant_penalties.reserve(quantile_penalties_.size());
            for (std::size_t i = 0; i < quantile_penalties_.size(); i++) {
                quant_penalties.push_back(quantile_penalties_[i].clone());
                if (quant_penalties[i].enabled()) {
                    quant_penalties[i].set_mean(
                        quant_penalties[i].mean() +
                        std::sqrt(quant_penalties[i].mse()) *
                            numerics::distributions::Normal::standard_z(prng->next_double()));
                }
            }

            // Create randomized penalty function
            penalty_function_ = [this, nt, param_penalties = std::move(param_penalties),
                                 quant_penalties = std::move(quant_penalties)](
                                    const std::vector<double>& parameters) {
                std::vector<double> theta = link_controller_.inverse_link(parameters);
                double result = 0;
                for (std::size_t i = 0; i < param_penalties.size(); i++) {
                    if (param_penalties[i].enabled()) {
                        result += param_penalties[i].function(theta[i], nt);
                    }
                }
                for (std::size_t i = 0; i < quant_penalties.size(); i++) {
                    if (quant_penalties[i].enabled()) {
                        auto dist = distribution_->clone();
                        dist->set_parameters(theta);
                        double aep = quant_penalties[i].aep();
                        result += quant_penalties[i].function(dist->inverse_cdf(1 - aep), nt);
                    }
                }
                return result;
            };
            return;
        }
    }

    // Sets a randomized penalty function for bootstrap uncertainty analysis (C# line 937):
    // like the random variant of set_penalty_function, but the penalty Means are recentered
    // at the parent parameters (parameter penalties) / the parent distribution's quantile
    // (quantile penalties, log10-transformed when UseLog10). A wrong-length parent vector
    // throws std::out_of_range (C# ArgumentOutOfRangeException); a null prng throws
    // std::invalid_argument (C# ArgumentNullException). Draw order as above.
    void set_random_penalty_function(const std::vector<double>& parent_parameters,
                                     numerics::sampling::MersenneTwister* prng) {
        // Determine if there any penalties
        int n_penalty = count_enabled_penalties();
        if (n_penalty == 0) {
            penalty_function_ = nullptr;
            return;
        }
        // Make sure inputs are valid
        if (static_cast<int>(parent_parameters.size()) != number_of_parameters())
            throw std::out_of_range("The parent parameter length is invalid.");
        if (prng == nullptr) throw std::invalid_argument("The prng cannot be null.");

        // (C# dereferences DataFrame here unconditionally -- see set_penalty_function.)
        if (!has_data_frame()) throw std::runtime_error("DataFrame is null.");
        int nt = data_frame().total_record_length();

        // Create random parameter penalties centered at the parent parameters
        std::vector<ParameterPenalty> param_penalties;
        param_penalties.reserve(parameter_penalties_.size());
        for (std::size_t i = 0; i < parameter_penalties_.size(); i++) {
            param_penalties.push_back(parameter_penalties_[i].clone());
            if (param_penalties[i].enabled()) {
                param_penalties[i].set_mean(
                    parent_parameters[i] +
                    std::sqrt(param_penalties[i].mse()) *
                        numerics::distributions::Normal::standard_z(prng->next_double()));
            }
        }

        // Create random quantile penalties centered at the parent quantile
        auto parent_dist = distribution_->clone();
        parent_dist->set_parameters(parent_parameters);
        std::vector<QuantilePenalty> quant_penalties;
        quant_penalties.reserve(quantile_penalties_.size());
        for (std::size_t i = 0; i < quantile_penalties_.size(); i++) {
            quant_penalties.push_back(quantile_penalties_[i].clone());
            if (quant_penalties[i].enabled()) {
                double q_mean =
                    quant_penalties[i].use_log10()
                        ? numerics::clamped_log10(
                              parent_dist->inverse_cdf(1 - quant_penalties[i].aep()))
                        : parent_dist->inverse_cdf(1 - quant_penalties[i].aep());
                quant_penalties[i].set_mean(
                    q_mean + std::sqrt(quant_penalties[i].mse()) *
                                 numerics::distributions::Normal::standard_z(
                                     prng->next_double()));
            }
        }

        // Create bootstrap penalty function
        penalty_function_ = [this, nt, param_penalties = std::move(param_penalties),
                             quant_penalties = std::move(quant_penalties)](
                                const std::vector<double>& parameters) {
            std::vector<double> theta = link_controller_.inverse_link(parameters);
            double result = 0;
            for (std::size_t i = 0; i < param_penalties.size(); i++) {
                if (param_penalties[i].enabled()) {
                    result += param_penalties[i].function(theta[i], nt);
                }
            }
            for (std::size_t i = 0; i < quant_penalties.size(); i++) {
                if (quant_penalties[i].enabled()) {
                    auto dist = distribution_->clone();
                    dist->set_parameters(theta);
                    double aep = quant_penalties[i].aep();
                    result += quant_penalties[i].function(dist->inverse_cdf(1 - aep), nt);
                }
            }
            return result;
        };
    }

    // --- Moment machinery (B10; bodies in bulletin17c_moment_machinery.hpp) --------------

    // Computes the GMM moment condition vector G and its covariance matrix S (C# line
    // 1043): g(Y; theta) = [Y - mu, (Y - mu)^2 - sigma^2, (Y - mu)^3 - mu3], accumulated
    // per data type (low outliers, exact, uncertain, interval, threshold), repaired for
    // non-finite values, then normalized to G = sum(g)/n and S = E[gg'] - G G'. The
    // parameter vector is in LINK space (inverse-linked on entry). Invalid parameters
    // return a double-max-filled G (the optimizer-rejection sentinel), not a throw.
    estimation::MomentConditionResult moment_conditions(std::vector<double> parameters) const;

    // Computes a directional censoring asymmetry score in [-1, 1] per moment condition
    // (C# line 1691): posPull/negPull decomposition over all data types with the final
    // (posPull - negPull) / (posPull + negPull + 1e-12) normalization. Link-space
    // parameters; a NaN-filled array signals invalid parameters.
    std::vector<double> censoring_asymmetry_score(std::vector<double> parameters) const;

    // Computes the weighted error direction score (WEDS) per parameter (C# line 1901).
    // Deliberately operates on NATURAL-space parameters -- this is the one scoring method
    // that does NOT inverse-link, so the diagnostic stays invariant to temporary
    // uncertainty-link choices (see the C# doc remarks). NaN-filled on invalid parameters.
    std::vector<double> weighted_error_direction_score(
        const std::vector<double>& parameters) const;

    // Computes WEDS from link-space parameters by inverse-linking through the current
    // LinkController first, then delegating (C# line 2090).
    std::vector<double> weighted_error_direction_score_from_linked(
        const std::vector<double>& linked_parameters) const;

    // Computes the gradient of the quantile function with respect to the distribution
    // parameters (C# line 2315). PT3/LP3 use PearsonTypeIII::quantile_gradient_for_moments
    // (moment-space gradient); the other families use the IStandardError QuantileGradient
    // (C# line 2355 cast -- a per-type dispatch here, see the companion header). Guards:
    // probability outside (0, 1) -> std::out_of_range; wrong parameter arity or invalid
    // parameters -> std::invalid_argument (C# ArgumentOutOfRange/ArgumentException).
    std::vector<double> quantile_gradient(double probability,
                                          const std::vector<double>& parameters) const;

    // Computes the variance of a quantile estimate with the delta method Var(Q_p) =
    // g' Sigma g via quantile_gradient (C# line 2388). A covariance matrix whose
    // dimensions do not match NumberOfParameters throws std::invalid_argument.
    double quantile_variance(double probability, const std::vector<double>& parameters,
                             const numerics::math::linalg::Matrix2D& covariance_matrix) const;

    // Return a deep copy of the model (C# line 2413; the XElement round trip becomes a
    // direct deep clone -- see the header note on the preserved observable behavior and
    // the two documented deviations).
    std::unique_ptr<IGMMModel> clone() const override {
        std::unique_ptr<Bulletin17CDistribution> copy = clone_core_state();
        if (has_data_frame()) {
            copy->data_frame_ = data_frame().clone();
            // The C# XElement ctor's DataFrame setter reprocesses thresholds even while
            // deserializing.
            copy->data_frame_->process_threshold_series();
        }
        return copy;
    }

    // Creates a deep clone of this model bound to the SUPPLIED data frame, preserving the
    // current parameters, parameter penalties, quantile penalties, and link controller
    // without any data-driven re-initialization (C# line 2451, v2.0.0 addition -- see the
    // header note above clone() for the full rationale and the two documented deviations
    // it shares with clone()). Shares clone_core_state() with clone(); the only difference
    // is which frame gets bound and reprocessed -- the caller's, not this instance's own.
    std::unique_ptr<Bulletin17CDistribution> clone_with_data_frame(
        const DataFrame& data_frame) const {
        std::unique_ptr<Bulletin17CDistribution> copy = clone_core_state();
        copy->data_frame_ = data_frame.clone();
        // Mirrors the C# DataFrame setter's ProcessThresholdSeries call; SetDefaultParameters
        // stays suppressed by the RawTag ctor's deserialization-equivalent bypass, exactly
        // like clone().
        copy->data_frame_->process_threshold_series();
        return copy;
    }

    // Validates the current state of the object and reports any issues found (C# line
    // 2469). Ported checks: null DataFrame (early return), DataFrame.Validate, null /
    // unsupported distribution, the retained measurement-error quantile-bound guard on
    // uncertain data, the log-distribution non-positive-data checks (exact +
    // uncertain), per-ModelParameter validation, per-ParameterPenalty and
    // per-QuantilePenalty validation, and the quantile-penalty AEP/Mean ordering
    // cross-checks. Nothing in the C# method touches unported members, so no clause is
    // skipped.
    ValidationResult validate() const override {
        ValidationResult result;

        // DataFrame checks
        if (!has_data_frame()) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: The input DataFrame is null.");
            return result;
        }

        // Validate data frame
        ValidationResult data_valid = data_frame().validate();
        if (!data_valid.is_valid) {
            result.is_valid = false;
            result.validation_messages.insert(result.validation_messages.end(),
                                              data_valid.validation_messages.begin(),
                                              data_valid.validation_messages.end());
        }

        // Distribution checks
        if (distribution_ == nullptr) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: The parent distribution is null.");
        }

        // Distribution type check. (The C# interpolates the enum NAME via
        // Distribution.Type.ToString(); no enum-name table is ported, so the numeric enum
        // value stands in -- the message text is not oracle-checked.)
        if (distribution_ != nullptr && !is_supported_distribution_type(distribution_->type())) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Distribution type '" +
                std::to_string(static_cast<int>(distribution_->type())) +
                "' is not supported by the Bulletin 17C model. Supported types: "
                "Exponential, Gamma, Log-Normal, Log-Pearson Type III, Normal, Pearson "
                "Type III.");
        }

        // Validate retained ME quantile bounds directly. This guard uses 1E-8 to avoid
        // InverseCDF(0/1) infinities and matches the B17C ME moment integration window.
        for (std::size_t k = 0; k < data_frame().uncertain_series().count(); k++) {
            const UncertainData& data = data_frame().uncertain_series()[k];
            const DistributionBase& dist = data.distribution();
            double lower_probability =
                std::isinf(dist.minimum()) && dist.minimum() < 0 ? 1E-8 : 0.0;
            double upper_probability =
                std::isinf(dist.maximum()) && dist.maximum() > 0 ? 1.0 - 1E-8 : 1.0;
            double lower =
                lower_probability > 0.0 ? dist.inverse_cdf(lower_probability) : dist.minimum();
            double upper =
                upper_probability < 1.0 ? dist.inverse_cdf(upper_probability) : dist.maximum();
            double mass = upper_probability - lower_probability;

            if (!numerics::is_finite(lower) || !numerics::is_finite(upper) ||
                !numerics::is_finite(mass) || mass <= 0.0 || lower >= upper) {
                result.is_valid = false;
                result.validation_messages.push_back(
                    "Error: Uncertain data at index " + std::to_string(data.index()) +
                    " has invalid measurement-error integration bounds at the 1E-8 "
                    "probability window.");
            }
        }

        // Log-distribution non-positive data checks
        if (distribution_ != nullptr &&
            (distribution_type() == DistributionType::LogNormal ||
             distribution_type() == DistributionType::LogPearsonTypeIII)) {
            bool non_positive_exact = false;
            if (data_frame().exact_series().count() > 0) {
                for (std::size_t k = 0; k < data_frame().exact_series().count(); k++) {
                    const ExactData& x = data_frame().exact_series()[k];
                    if (!x.is_low_outlier() && x.value() <= 0.0) {
                        non_positive_exact = true;
                        break;
                    }
                }
            }

            bool non_positive_uncertain = false;
            for (std::size_t k = 0; k < data_frame().uncertain_series().count(); k++) {
                const DistributionBase& dist =
                    data_frame().uncertain_series()[k].distribution();
                double lower = dist.inverse_cdf(1E-8);
                double upper = dist.inverse_cdf(1 - 1E-8);

                // Log-space B17C validation checks retained support at 1E-8 so endpoint
                // infinities do not masquerade as data-support failures.
                if (!numerics::is_finite(lower) || !numerics::is_finite(upper) ||
                    lower >= upper || lower <= 0.0) {
                    non_positive_uncertain = true;
                    break;
                }
            }

            if (non_positive_exact || non_positive_uncertain) {
                result.is_valid = false;
                result.validation_messages.push_back(
                    "Error: Log-based distributions cannot be used because some data "
                    "values are non-positive.");
            }
        }

        // Parameter validation
        for (std::size_t i = 0; i < parameters_.size(); i++) {
            ValidationResult valid = parameters_[i].validate();
            if (!valid.is_valid) {
                result.is_valid = false;
                result.validation_messages.insert(result.validation_messages.end(),
                                                  valid.validation_messages.begin(),
                                                  valid.validation_messages.end());
            }
        }

        // Parameter penalty validation
        for (std::size_t i = 0; i < parameter_penalties_.size(); i++) {
            ValidationResult valid = parameter_penalties_[i].validate();
            if (!valid.is_valid) {
                result.is_valid = false;
                result.validation_messages.insert(result.validation_messages.end(),
                                                  valid.validation_messages.begin(),
                                                  valid.validation_messages.end());
            }
        }

        // Quantile penalty validation
        for (std::size_t i = 0; i < quantile_penalties_.size(); i++) {
            ValidationResult valid = quantile_penalties_[i].validate();
            if (!valid.is_valid) {
                result.is_valid = false;
                result.validation_messages.insert(result.validation_messages.end(),
                                                  valid.validation_messages.begin(),
                                                  valid.validation_messages.end());
            }

            // Cross-validate quantile penalty ordering
            if (i >= 1) {
                if (quantile_penalties_[i].aep() >= quantile_penalties_[i - 1].aep()) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: Quantile penalties must have strictly decreasing annual "
                        "exceedance probabilities (AEP).");
                }

                if (quantile_penalties_[i].mean() <= quantile_penalties_[i - 1].mean()) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: Quantile penalty means must increase with decreasing "
                        "annual exceedance probability.");
                }
            }
        }

        return result;
    }

    // Generates random samples from the underlying distribution (C# line 2621): thin
    // delegate to Distribution.GenerateRandomValues. A non-positive sample size throws
    // std::out_of_range (C# ArgumentOutOfRangeException); a missing distribution throws
    // std::runtime_error (C# InvalidOperationException).
    std::vector<double> generate_random_values(int sample_size, int seed = -1) const override {
        if (sample_size <= 0)
            throw std::out_of_range("Sample size must be positive.");
        if (distribution_ == nullptr)
            throw std::runtime_error(
                "Distribution cannot be null when generating random values.");

        return distribution_->generate_random_values(sample_size, seed);
    }

   private:
    // Tag for the raw (clone) constructor: no distribution, no reseeding -- the C#
    // _isDeserializing suppression path.
    struct RawTag {};
    explicit Bulletin17CDistribution(RawTag) {}

    // The shared body of clone() / clone_with_data_frame() (C# line 2413 / 2451): both round
    // trip through the SAME "restore the serialized snapshot" contract, differing only in
    // which DataFrame gets bound afterward. Deep-copies Distribution/Parameters/
    // ParameterPenalties/QuantilePenalties via the RawTag ctor (bypassing
    // set_default_parameters(), exactly like the C# XElement ctor's
    // `_isDeserializing = true` suppression). DataFrame, PenaltyFunction, and LinkController
    // are the caller's responsibility: PenaltyFunction is never carried over (the C# XElement
    // round trip does not serialize it) and link_controller_ stays default-constructed after
    // the guard below verifies no link is installed (a non-empty LinkController cannot yet be
    // deep-copied -- the ported ILinkFunction has no clone(); B10+ follow-up).
    std::unique_ptr<Bulletin17CDistribution> clone_core_state() const {
        for (const auto& link : link_controller_.links()) {
            if (link != nullptr)
                throw std::logic_error(
                    "Bulletin17CDistribution: deep-copying an installed LinkController link "
                    "is not supported yet (ILinkFunction has no clone(); B10+ follow-up).");
        }

        std::unique_ptr<Bulletin17CDistribution> copy(new Bulletin17CDistribution(RawTag{}));
        copy->distribution_ = distribution_->clone();
        copy->parameters_ = parameters_;                      // ModelParameter deep-copies
        copy->parameter_penalties_ = parameter_penalties_;
        copy->quantile_penalties_ = quantile_penalties_;
        return copy;
    }

    // Sets up the single default quantile penalty (C# line 817; the handler rewiring is
    // INPC plumbing, skipped).
    void set_up_quantile_penalties() { quantile_penalties_.emplace_back(); }

    // The shared enabled-penalty count (C# `ParameterPenalties.Where(x => x.Enabled).Count()
    // + QuantilePenalties.Where(x => x.Enabled).Count()`).
    int count_enabled_penalties() const {
        int n = 0;
        for (const ParameterPenalty& p : parameter_penalties_)
            if (p.enabled()) n++;
        for (const QuantilePenalty& q : quantile_penalties_)
            if (q.enabled()) n++;
        return n;
    }

    // The shared body of SetInitialParameters / SetDefaultParameters (C# duplicates it
    // verbatim in both methods): collect the explicit data values, get the distribution's
    // MLE parameter constraints, and override the initials with nonparametric ROS moment
    // estimates when low outliers or a threshold series are present (v2.0.0, upstream-sync
    // Task 18, c420d48 -- narrowed from also firing on bare uncertain/interval data; see the
    // file header). Throws (into the callers' silent catch guards) when the frame is absent
    // or a capability cast fails -- the C# NullReference/InvalidCast exceptions those same
    // catches swallow.
    void compute_default_initials(std::vector<double>& initials, std::vector<double>& lowers,
                                  std::vector<double>& uppers) const {
        if (!has_data_frame())
            throw std::runtime_error("DataFrame is null.");  // C# NullReferenceException

        // Get data values for parameter constraints
        std::vector<double> data;
        const DataFrame& df = data_frame();
        for (std::size_t i = 0; i < df.exact_series().count(); i++)
            data.push_back(df.exact_series()[i].value());
        for (std::size_t i = 0; i < df.uncertain_series().count(); i++)
            data.push_back(df.uncertain_series()[i].value());
        for (std::size_t i = 0; i < df.interval_series().count(); i++)
            data.push_back(df.interval_series()[i].value());

        // Get typical parameter constraints (initials, lower, upper bounds)
        const auto* ml_estimator =
            dynamic_cast<const numerics::distributions::IMaximumLikelihoodEstimation*>(
                distribution_.get());
        if (ml_estimator == nullptr)
            throw std::runtime_error(
                "Distribution does not implement IMaximumLikelihoodEstimation.");
        ml_estimator->get_parameter_constraints(data, initials, lowers, uppers);

        // Override initials with nonparametric moment estimates when low outliers or a
        // threshold series are present (v2.0.0: uncertain/interval data ALONE no longer
        // triggers this -- see the file header). Use ROS (Regression on Order Statistics)
        // to impute low-outlier values, which avoids the severe moment distortion caused by
        // log-transforming near-zero or zero flows.
        if (df.number_of_low_outliers() > 0 || df.threshold_series().count() > 0) {
            bool use_log10 = distribution_type() == DistributionType::LogNormal ||
                             distribution_type() == DistributionType::LogPearsonTypeIII;
            std::optional<std::vector<double>> np_moments =
                df.get_nonparametric_moments_ros(use_log10);

            if (np_moments.has_value()) {
                bool all_finite = true;
                for (int i = 0; i < distribution_->number_of_parameters(); i++) {
                    if (!numerics::is_finite((*np_moments)[static_cast<std::size_t>(i)])) {
                        all_finite = false;
                        break;
                    }
                }
                if (all_finite) {
                    initials = parameters_from_moments_dispatch(*distribution_, *np_moments);
                }
            }
        }
    }

    // The C# `((IMomentEstimation)Distribution).ParametersFromMoments(...)` cast, as a
    // static per-type dispatch over the six supported families (no IMomentEstimation mixin
    // exists in the C++ port -- see the file header). An unreachable type throws into the
    // callers' silent catch guards, exactly where the failed C# cast would.
    static std::vector<double> parameters_from_moments_dispatch(
        const DistributionBase& dist, const std::vector<double>& moments) {
        using namespace numerics::distributions;
        switch (dist.type()) {
            case DistributionType::Exponential:
                return dynamic_cast<const Exponential&>(dist).parameters_from_moments(moments);
            case DistributionType::GammaDistribution:
                return dynamic_cast<const GammaDistribution&>(dist).parameters_from_moments(
                    moments);
            case DistributionType::LogNormal:
                return dynamic_cast<const LogNormal&>(dist).parameters_from_moments(moments);
            case DistributionType::LogPearsonTypeIII:
                return dynamic_cast<const LogPearsonTypeIII&>(dist).parameters_from_moments(
                    moments);
            case DistributionType::Normal:
                return dynamic_cast<const Normal&>(dist).parameters_from_moments(moments);
            case DistributionType::PearsonTypeIII:
                return dynamic_cast<const PearsonTypeIII&>(dist).parameters_from_moments(
                    moments);
            default:
                throw std::runtime_error(
                    "Distribution does not implement IMomentEstimation.");
        }
    }

    // --- Private moment machinery (B10; bodies in bulletin17c_moment_machinery.hpp) ------

    // Detects non-finite values in the moment condition vector or covariance accumulator
    // and replaces them with the optimizer-rejection sentinels (C# line 1223: errors ->
    // double.MaxValue, covariance -> 0). The C# passes the raw backing arrays
    // (mean.Array / covariance.Array); the port mutates the Vector/Matrix directly.
    void repair_errors(numerics::math::linalg::Vector& errors,
                       numerics::math::linalg::Matrix& covariance) const;

    // Computes the 4th-6th central moments of a Pearson Type III distribution from sigma
    // and skew gamma (C# line 1280): mu4 = s^4(3 + 3g^2/2), mu5 = s^5 g(10 + 3g^2),
    // mu6 = s^6(15 + 65g^2/2 + 15g^4/2). C# out params -> reference params.
    static void mu456_pearson3_from_sigma_gamma(double sigma, double gamma, double& mu4,
                                                double& mu5, double& mu6);

    // Computes conditional central moments for an uncertain observation by 20-point
    // Gauss-Legendre integration over the measurement-error distribution, normalized by
    // the retained probability mass (C# line 1325). The optional matrix receives the raw
    // E_ME[gg'] second moment (C# `Matrix?` -> nullable pointer, nullptr = C# null).
    std::vector<double> conditional_moments_for_uncertain_data(
        const std::vector<double>& unconditional_moments, const UncertainData& uncertain_data,
        bool is_log10,
        numerics::math::linalg::Matrix* measurement_error_second_moment = nullptr) const;

    // Computes moment condition errors (g-vector) for an uncertain observation by
    // delegating to conditional_moments_for_uncertain_data and subtracting the
    // unconditional moments (C# line 1418).
    std::vector<double> moment_conditions_for_uncertain_data(
        const std::vector<double>& unconditional_moments, const UncertainData& uncertain_data,
        bool is_log10) const;

    // Accumulates running sums for the moment condition mean vector and second-moment
    // matrix, one observation (or censored group) per call (C# line 1490): the
    // model-based mu4-mu6 covariance path for supported families vs the outer-product
    // fallback, plus the optional measurement-error second-moment term (law of total
    // variance).
    void update_moment_mean_covariance(
        const std::vector<double>& c, const std::vector<double>& m,
        numerics::math::linalg::Vector& mean, numerics::math::linalg::Matrix& covariance,
        double w, bool is_censored = false, bool use_model_covariance = true,
        double model_sigma = 0.0, double model_gamma = 0.0,
        const numerics::math::linalg::Matrix* measurement_error_second_moment = nullptr) const;

    // Computes per-observation moment condition g-vectors as an [n x q] matrix (C# line
    // 2128; the delegate target of pointwise_moment_conditions()). Row order: low
    // outliers -> exact -> uncertain -> interval -> threshold (NumberBelow + NumberAbove
    // rows each). Column means equal moment_conditions' G (the CON-23 invariant). A
    // zero-filled matrix signals invalid parameters.
    numerics::math::linalg::Matrix2D pointwise_moment_conditions_impl(
        std::vector<double> parameters) const;

    std::optional<DataFrame> data_frame_;                       // C# _dataFrame (line 290)
    std::unique_ptr<DistributionBase> distribution_;            // C# _distribution (line 291)
    std::vector<ModelParameter> parameters_;                    // C# _parameters (line 292)
    std::vector<ParameterPenalty> parameter_penalties_;         // C# _parameterPenalties (293)
    std::vector<QuantilePenalty> quantile_penalties_;           // C# _quantilePenalties (294)
    estimation::PenaltyFunction penalty_function_;              // C# _penaltyFunction (295)
    LinkController link_controller_;                            // C# _linkController (296)
};

}  // namespace corehydro::models

// The B10 moment-machinery method bodies (an .ipp-style companion; see the file header).
#include "corehydro/models/univariate_distribution/bulletin17c_moment_machinery.hpp"
