// ported from: RMC-BestFit/src/RMC.BestFit/Models/BivariateDistribution/BivariateDistribution.cs @ c2e6192
//
// A bivariate joint-distribution model: two pre-fit univariate marginals (X and Y) plus a
// BivariateCopula describing their dependence. The estimators (MLE/MAP/Bayesian) fit ONLY the
// copula parameter(s); the marginals are held FIXED during the copula fit. Derives ModelBase
// (the fittable-model likelihood surface) and ISimulatable<Matrix2D> (paired simulation).
//
// TData / simulation shape: ISimulatable is instantiated on
// `corehydro::numerics::math::linalg::Matrix2D` (std::vector<std::vector<double>>, n rows x 2
// cols; column 0 = X, column 1 = Y) -- the C# `ISimulatable<double[,]>`. Matrix2D is chosen
// over the linalg `Matrix` class because BivariateCopula::generate_random_values already
// returns Matrix2D in exactly this (row = sample, col 0/1 = x/y) convention, so simulation is
// a zero-conversion pass-through; wrapping it in a `Matrix` would be gratuitous.
//
// Marginals: held as NON-OWNING raw pointers `IUnivariateModel*` (marginal_x_/marginal_y_),
// the faithful mapping of the C# reference-typed `IUnivariateModel MarginalX/Y` auto-property
// (assigning aliases; the caller keeps ownership and the marginal object identity is
// preserved -- the C# `Assert.AreSame` tests). The caller MUST keep the marginals alive for
// the lifetime of the BivariateDistribution.
//
// Invalidation strategy (INPC removal, following the M4/M5 precedent): the C# marginal-property
// setters wire INotifyPropertyChanged handlers (MarginalX_PropertyChanged, C# 146-196) so a
// later mutation of a marginal re-runs SetDefaultParameters. That reactive plumbing is DROPPED
// here; the C++ setter simply stores the marginal pointer and re-runs SetDefaultParameters when
// UseDefaultFlatPriors (the immediate side effect the C# setter also performs). A caller who
// mutates a marginal AFTER assignment must re-assign it (or call set_default_parameters())
// to re-sync -- the same contract the rest of the ported model layer carries for INPC-free
// binding objects. RaisePropertyChange / ToXElement / the XElement ctor are likewise not ported.
//
// CopulaEstimationMethod: reuses the ported Numerics enum
// `numerics::distributions::copulas::CopulaEstimationMethod`. The C# BestFit source does NOT
// declare its own enum -- BivariateDistribution.cs `using Numerics.Distributions.Copulas;` and
// references `CopulaEstimationMethod.InferenceFromMargins` / `.PseudoLikelihood` off the
// Numerics enum. That enum additionally carries `FullLikelihood`, which this model never
// selects; mirroring the C#, SetSampleData treats "not PseudoLikelihood" as the raw-value
// (IFM) branch and DataLogLikelihood only sums under the two explicitly-handled members
// (a FullLikelihood setting would contribute an all-zero likelihood, exactly as the C#
// if/else-if leaves it). Default: InferenceFromMargins.
//
// CreateCopula: reuses the ported `copulas::create_copula(CopulaType)` factory (DRY) -- the C#
// `CreateCopula` and the ported factory both default-construct the concrete copula per type
// with identical seed parameters, so there is nothing extra to reproduce.
//
// PseudoLikelihood auto-plotting (v2.0.0, this class's only behavior change since fc28c0c):
// SetSampleData now validates the paired PSEUDO observations (each marginal's
// PlottingPositionComplement) strictly inside (0, 1) whenever CopulaEstimationMethod is
// PseudoLikelihood. A freshly-built marginal's exact data carries the un-computed default
// plotting position (0.0 -> complement 1.0, NOT interior), so the very first PseudoLikelihood
// SetSampleData call is always invalid; on invalidity, the affected marginal's
// DataFrame::calculate_plotting_positions() is run once and the paired sample is rebuilt from
// the recomputed positions. The result is cached per marginal DataFrame identity + Task 12's
// data_frame().plotting_position_version(), so an unchanged sample skips re-validation on every
// subsequent call (the IFM/hot-path cost is untouched). RETIRES the audit finding this port
// carried since P4/P5 (docs/upstream-csharp-issues.md, reconciled by Task 22): a model-level
// PseudoLikelihood MLE fit previously could never succeed in EITHER language because the shared
// build path never triggered CalculatePlottingPositions -- it now does, faithfully, in both.
//
// SKIPPED (C# line spans + reason):
//   - XML ctor (C# 60-95), ToXElement (698-718): XML serialization is a project-wide non-port.
//     The XElement-round-trip tests are consequently not transcribed.
//   - INPC plumbing: MarginalX_PropertyChanged / MarginalY_PropertyChanged (247-270),
//     RaisePropertyChange calls throughout, the [Category]/[DisplayName]/[Description]/
//     [Browsable] attributes: WPF data-binding, not ported (matching every other ported model).
//   - ParameterNameFor's generic fallback (355) is reachable only by a hypothetical
//     3+-parameter copula; kept verbatim for fidelity.
//   - GetSampleDataAlignmentCounts (v2.0.0 addition) + its CountPairedExactData helper: a
//     GUI-only diagnostic (reports X/Y/paired observation counts for a properties panel), never
//     consumed by the likelihood/estimation surface. Not ported (matching the RatingCurve
//     GetDataAlignmentCounts precedent in rating_curve.hpp).
//
// EXCEPTION-TYPE MAPPING for this file: C# ArgumentNullException/ArgumentException ->
// std::invalid_argument; ArgumentOutOfRangeException -> std::out_of_range;
// InvalidOperationException -> std::runtime_error. The SafeLogPDF/SafeIfmLogPDF swallowed-
// exception guards (C# Debug.WriteLine + return -inf) port as silent no-throw guards.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/support/data_component.hpp"
#include "corehydro/models/support/i_univariate_model.hpp"
#include "corehydro/models/support/model_base.hpp"
#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/models/support/simulatable.hpp"
#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/numerics/distributions/copulas/base/bivariate_copula.hpp"
#include "corehydro/numerics/distributions/copulas/base/copula_estimation_method.hpp"
#include "corehydro/numerics/distributions/copulas/base/copula_factory.hpp"
#include "corehydro/numerics/distributions/copulas/base/copula_type.hpp"
#include "corehydro/numerics/distributions/copulas/student_t_copula.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models {

class BivariateDistribution : public ModelBase,
                              public ISimulatable<numerics::math::linalg::Matrix2D> {
   public:
    using BivariateCopula = numerics::distributions::copulas::BivariateCopula;
    using CopulaType = numerics::distributions::copulas::CopulaType;
    using CopulaEstimationMethod = numerics::distributions::copulas::CopulaEstimationMethod;
    using StudentTCopula = numerics::distributions::copulas::StudentTCopula;
    using Matrix2D = numerics::math::linalg::Matrix2D;
    using UnivariateDistributionBase = numerics::distributions::UnivariateDistributionBase;

    // C# parameterless ctor (line 33): a default Normal copula and no marginals.
    BivariateDistribution() { set_copula(create_copula(CopulaType::Normal)); }

    // C# `BivariateDistribution(IUnivariateModel, IUnivariateModel, CopulaType)` (line 44).
    // Marginals are aliased (non-owning), the copula is built from the type, then the
    // parameters are defaulted (mirroring the C# setter side effects + explicit call).
    BivariateDistribution(IUnivariateModel& marginal_x, IUnivariateModel& marginal_y,
                          CopulaType copula_type) {
        set_marginal_x(&marginal_x);
        set_marginal_y(&marginal_y);
        set_copula(create_copula(copula_type));
        set_default_parameters();
    }

    // Move-only (holds a unique_ptr<BivariateCopula>); deep copies go through clone().
    BivariateDistribution(BivariateDistribution&&) noexcept = default;
    BivariateDistribution& operator=(BivariateDistribution&&) noexcept = default;
    ~BivariateDistribution() override = default;

    // --- CreateCopula (C# static, line 277): reuse the ported factory (DRY). ------------
    // C# throws ArgumentOutOfRangeException for an unsupported type; the ported factory
    // throws std::invalid_argument.
    static std::unique_ptr<BivariateCopula> create_copula(CopulaType copula_type) {
        return numerics::distributions::copulas::create_copula(copula_type);
    }

    // --- Copula property (C# line 115). --------------------------------------------------
    BivariateCopula& copula() { return *copula_; }
    const BivariateCopula& copula() const { return *copula_; }

    // C# setter (line 118): null-check, store, then SetDefaultParameters().
    void set_copula(std::unique_ptr<BivariateCopula> copula) {
        if (copula == nullptr)
            throw std::invalid_argument("value");  // C# ArgumentNullException
        copula_ = std::move(copula);
        set_default_parameters();
    }

    // C# `CopulaType` property (line 129): getter reads Copula.Type; setter rebuilds Copula
    // via the factory.
    CopulaType copula_type() const { return copula_->type(); }
    void set_copula_type(CopulaType value) { set_copula(create_copula(value)); }

    // --- Marginals (C# lines 146-203; INPC handlers dropped, see file header). -----------
    IUnivariateModel* marginal_x() { return marginal_x_; }
    const IUnivariateModel* marginal_x() const { return marginal_x_; }
    void set_marginal_x(IUnivariateModel* value) {
        marginal_x_ = value;
        if (marginal_x_ != nullptr && use_default_flat_priors()) set_default_parameters();
    }

    IUnivariateModel* marginal_y() { return marginal_y_; }
    const IUnivariateModel* marginal_y() const { return marginal_y_; }
    void set_marginal_y(IUnivariateModel* value) {
        marginal_y_ = value;
        if (marginal_y_ != nullptr && use_default_flat_priors()) set_default_parameters();
    }

    // --- CopulaEstimationMethod (C# line 222): setter re-runs SetSampleData on change. ----
    CopulaEstimationMethod copula_estimation_method() const { return copula_estimation_method_; }
    void set_copula_estimation_method(CopulaEstimationMethod value) {
        if (copula_estimation_method_ != value) {
            copula_estimation_method_ = value;
            set_sample_data();
        }
    }

    // --- SetSampleData (C# line 394, v2.0.0). ---------------------------------------------
    // Two-pointer index merge of the two marginals' non-low-outlier exact series (both sorted
    // by index first, via get_eligible_exact_data()). PseudoLikelihood stores
    // PlottingPositionComplement pairs (unit interval); every other method (IFM) stores raw
    // Value pairs. Under PseudoLikelihood, the pseudo sample is validated strictly inside
    // (0, 1) once per distinct (marginal DataFrame identity, plotting_position_version()) pair;
    // an invalid marginal gets CalculatePlottingPositions() run once and the sample rebuilt.
    void set_sample_data() {
        bool use_pseudo_likelihood =
            copula_estimation_method_ == CopulaEstimationMethod::PseudoLikelihood;

        sample_data_x_.clear();
        sample_data_y_.clear();

        // DEVIATION (raw-pointer nullability, documented): the C# `MarginalX!.DataFrame!`
        // null-forgiving access assumes both marginals are already set (and carry a DataFrame)
        // whenever SetSampleData runs -- true on every path the C# actually reaches (the
        // two-marginal ctor, SetDefaultParameters' own null-guarded call, CopulaEstimationMethod's
        // setter after construction), though even the real C# would NRE on a marginal with a
        // NULL DataFrame under PseudoLikelihood (the null flows through GetEligibleExactData's
        // guard unharmed, but is deref'd unconditionally at the bottom via
        // `dataFrameX.PlottingPositionVersion`). A raw IUnivariateModel* CAN be null here (e.g.
        // the parameterless ctor followed directly by set_copula_estimation_method()), and
        // `data_frame()` is an UNGUARDED dereference of a possibly-absent optional (see
        // i_univariate_model.hpp / the base header's nullability note) -- calling it before
        // confirming a frame is attached would be undefined behavior, not a catchable C# NRE.
        // validate() safely reports invalid (without touching data_frame()) when none is
        // attached, so gate on it here -- same guard the OLD (pre-v2.0.0) SetSampleData used,
        // now hoisted above the DataFrame identity/version-cache access this port also needs.
        if (marginal_x_ == nullptr || !marginal_x_->validate().is_valid) return;
        if (marginal_y_ == nullptr || !marginal_y_->validate().is_valid) return;

        DataFrame* data_frame_x = &marginal_x_->data_frame();
        DataFrame* data_frame_y = &marginal_y_->data_frame();

        std::vector<const ExactData*> data_x, data_y;
        std::tie(data_x, data_y) = get_eligible_exact_data();

        bool requires_pseudo_validation =
            use_pseudo_likelihood &&
            (validated_pseudo_data_frame_x_ != data_frame_x ||
             validated_pseudo_version_x_ != data_frame_x->plotting_position_version() ||
             validated_pseudo_data_frame_y_ != data_frame_y ||
             validated_pseudo_version_y_ != data_frame_y->plotting_position_version());

        bool invalid_x = false, invalid_y = false;
        if (requires_pseudo_validation) {
            std::tie(invalid_x, invalid_y) =
                add_paired_sample_data_and_validate(data_x, data_y, sample_data_x_, sample_data_y_);
        } else {
            add_paired_sample_data(data_x, data_y, use_pseudo_likelihood, sample_data_x_,
                                   sample_data_y_);
        }

        if (use_pseudo_likelihood && (invalid_x || invalid_y)) {
            if (invalid_x) data_frame_x->calculate_plotting_positions();
            if (invalid_y && (data_frame_x != data_frame_y || !invalid_x))
                data_frame_y->calculate_plotting_positions();

            sample_data_x_.clear();
            sample_data_y_.clear();
            std::tie(data_x, data_y) = get_eligible_exact_data();
            add_paired_sample_data(data_x, data_y, true, sample_data_x_, sample_data_y_);
        }

        if (use_pseudo_likelihood) {
            validated_pseudo_data_frame_x_ = data_frame_x;
            validated_pseudo_data_frame_y_ = data_frame_y;
            validated_pseudo_version_x_ = data_frame_x->plotting_position_version();
            validated_pseudo_version_y_ = data_frame_y->plotting_position_version();
        }
    }

    // --- SetDefaultParameters (C# line 293). ---------------------------------------------
    // One ModelParameter per copula parameter, bounds from Copula.ParameterConstraints,
    // Uniform prior over those bounds. The C# `MarginalX.DataFrame is null` guard is folded
    // into the Validate() check: the ported IUnivariateModel exposes no DataFrame-null probe,
    // and validate().is_valid already fails on a null frame for every ported marginal
    // implementation (so the subsequent set_sample_data data_frame() deref is guarded).
    void set_default_parameters() override {
        // (Handler removal for the old parameters is INPC plumbing, skipped.)
        parameters_.clear();

        if (copula_ == nullptr) return;
        if (marginal_x_ == nullptr || !marginal_x_->validate().is_valid) return;
        if (marginal_y_ == nullptr || !marginal_y_->validate().is_valid) return;

        // Build the joint sample from the marginals.
        set_sample_data();
        if (sample_data_x_.empty() || sample_data_y_.empty()) return;

        numerics::math::linalg::Matrix2D bounds =
            copula_->parameter_constraints(sample_data_x_, sample_data_y_);
        int n_params = copula_->number_of_copula_parameters();

        for (int i = 0; i < n_params; ++i) {
            double lower = bounds[static_cast<std::size_t>(i)][0];
            double upper = bounds[static_cast<std::size_t>(i)][1];

            ModelParameter mp;
            mp.set_name(parameter_name_for(*copula_, i));
            mp.set_value(initial_value_for(*copula_, i, lower, upper));
            mp.set_lower_bound(lower);
            mp.set_upper_bound(upper);
            mp.set_is_positive(lower == numerics::kDoubleMachineEpsilon);
            mp.set_prior_distribution(
                std::make_unique<numerics::distributions::Uniform>(lower, upper));
            parameters_.push_back(std::move(mp));
        }
    }

    // --- DataLogLikelihood (C# line 455). ------------------------------------------------
    // Fits ONLY the copula parameters; the marginals stay fixed. A cloned copula carries the
    // proposed parameters and CLONES of the marginals' fitted distributions; each per-sample
    // evaluation is exception-shielded to -inf.
    double data_log_likelihood(std::vector<double>& parameters) const override {
        const UnivariateDistributionBase* dist_x =
            marginal_x_ != nullptr ? marginal_x_->distribution() : nullptr;
        const UnivariateDistributionBase* dist_y =
            marginal_y_ != nullptr ? marginal_y_->distribution() : nullptr;
        if (copula_ == nullptr || dist_x == nullptr || dist_y == nullptr) {
            return -std::numeric_limits<double>::infinity();
        }
        if (parameters.size() !=
            static_cast<std::size_t>(copula_->number_of_copula_parameters())) {
            return -std::numeric_limits<double>::infinity();
        }

        std::unique_ptr<BivariateCopula> model = copula_->clone();
        model->set_copula_parameters(parameters);
        model->marginal_distribution_x = dist_x->clone();
        model->marginal_distribution_y = dist_y->clone();

        double log_lh = 0.0;
        if (copula_estimation_method_ == CopulaEstimationMethod::PseudoLikelihood) {
            for (std::size_t i = 0; i < sample_data_x_.size(); ++i)
                log_lh += safe_log_pdf(*model, sample_data_x_[i], sample_data_y_[i]);
        } else if (copula_estimation_method_ == CopulaEstimationMethod::InferenceFromMargins) {
            for (std::size_t i = 0; i < sample_data_x_.size(); ++i)
                log_lh += safe_ifm_log_pdf(*model, sample_data_x_[i], sample_data_y_[i]);
        }
        if (!numerics::is_finite(log_lh)) return -std::numeric_limits<double>::infinity();
        return log_lh;
    }

    // --- PointwiseDataLogLikelihood (C# line 502). ---------------------------------------
    std::vector<double> pointwise_data_log_likelihood(
        const std::vector<double>& parameters) const override {
        std::size_t n = sample_data_x_.size();
        if (n == 0) return {};

        std::vector<double> result(n);
        // NOTE (parity): C# (BivariateDistribution.cs:509) does NOT null-check Copula here; the
        // added `copula_ == nullptr ||` guard is defensive and unreachable, since set_copula()
        // throws on null so copula_ is never null post-construction.
        if (copula_ == nullptr ||
            parameters.size() !=
                static_cast<std::size_t>(copula_->number_of_copula_parameters())) {
            for (std::size_t i = 0; i < n; ++i)
                result[i] = -std::numeric_limits<double>::infinity();
            return result;
        }

        std::unique_ptr<BivariateCopula> model = clone_model_with_marginals(parameters);
        if (model == nullptr) {
            for (std::size_t i = 0; i < n; ++i)
                result[i] = -std::numeric_limits<double>::infinity();
            return result;
        }

        if (copula_estimation_method_ == CopulaEstimationMethod::PseudoLikelihood) {
            for (std::size_t i = 0; i < n; ++i)
                result[i] = safe_log_pdf(*model, sample_data_x_[i], sample_data_y_[i]);
        } else if (copula_estimation_method_ == CopulaEstimationMethod::InferenceFromMargins) {
            for (std::size_t i = 0; i < n; ++i)
                result[i] = safe_ifm_log_pdf(*model, sample_data_x_[i], sample_data_y_[i]);
        }
        return result;
    }

    // --- PointwiseDataLogLikelihoodComponents (C# line 546). -----------------------------
    std::vector<DataComponent> pointwise_data_log_likelihood_components(
        const std::vector<double>& parameters) const override {
        std::size_t n = sample_data_x_.size();
        std::vector<DataComponent> result;
        if (n == 0) return result;
        result.reserve(n);

        // NOTE (parity): C# (BivariateDistribution.cs:555) does NOT null-check Copula here; the
        // added `copula_ == nullptr ||` guard is defensive and unreachable, since set_copula()
        // throws on null so copula_ is never null post-construction.
        if (copula_ == nullptr ||
            parameters.size() !=
                static_cast<std::size_t>(copula_->number_of_copula_parameters())) {
            for (std::size_t i = 0; i < n; ++i)
                result.emplace_back(static_cast<int>(i),
                                    -std::numeric_limits<double>::infinity(), sample_data_x_[i],
                                    DataComponentType::Exact, 1, std::to_string(i));
            return result;
        }

        std::unique_ptr<BivariateCopula> model = clone_model_with_marginals(parameters);
        if (model == nullptr) {
            for (std::size_t i = 0; i < n; ++i)
                result.emplace_back(static_cast<int>(i),
                                    -std::numeric_limits<double>::infinity(), sample_data_x_[i],
                                    DataComponentType::Exact, 1, std::to_string(i));
            return result;
        }

        for (std::size_t i = 0; i < n; ++i) {
            double log_lh =
                copula_estimation_method_ == CopulaEstimationMethod::PseudoLikelihood
                    ? safe_log_pdf(*model, sample_data_x_[i], sample_data_y_[i])
                    : safe_ifm_log_pdf(*model, sample_data_x_[i], sample_data_y_[i]);
            result.emplace_back(static_cast<int>(i), log_lh, sample_data_x_[i],
                                DataComponentType::Exact, 1, std::to_string(i));
        }
        return result;
    }

    // --- SetParameterValues (C# line 641): writes the model parameters AND fans them out to
    // the underlying copula (both theta and, for StudentT, degrees of freedom). -----------
    void set_parameter_values(const std::vector<double>& parameters) override {
        if (parameters.size() != static_cast<std::size_t>(number_of_parameters()))
            throw std::invalid_argument("The length of the parameter list in incorrect.");

        for (std::size_t i = 0; i < parameters.size(); ++i)
            parameters_[i].set_value(parameters[i]);

        copula_->set_copula_parameters(parameters);
    }

    // --- GenerateRandomValues (C# line 661, the ISimulatable<double[,]> surface). ---------
    // A cloned copula carries CLONES of the marginals' fitted distributions, then delegates to
    // BivariateCopula::generate_random_values (row i = sample i's (x, y) pair). DEVIATION: the
    // C# aliases MarginalX.Distribution into the working copula; the ported IUnivariateModel
    // exposes the marginal distribution as a raw CONST pointer, which cannot be aliased into
    // the copula's non-const shared_ptr, so the distributions are CLONED. Generation only
    // READS the marginals (inverse_cdf), so a clone is behaviorally identical.
    Matrix2D generate_random_values(int sample_size, int seed = -1) const override {
        if (sample_size <= 0)
            throw std::out_of_range("Sample size must be positive.");
        if (copula_ == nullptr)
            throw std::runtime_error("Copula cannot be null when generating random values.");
        const UnivariateDistributionBase* dist_x =
            marginal_x_ != nullptr ? marginal_x_->distribution() : nullptr;
        const UnivariateDistributionBase* dist_y =
            marginal_y_ != nullptr ? marginal_y_->distribution() : nullptr;
        if (marginal_x_ == nullptr || marginal_y_ == nullptr || dist_x == nullptr ||
            dist_y == nullptr) {
            throw std::runtime_error(
                "Both marginal distributions must be specified before generating random values.");
        }

        std::unique_ptr<BivariateCopula> copula = copula_->clone();
        copula->marginal_distribution_x = dist_x->clone();
        copula->marginal_distribution_y = dist_y->clone();
        return copula->generate_random_values(sample_size, seed);
    }

    // --- Clone (C# line 676): aliases the marginals, deep-copies the copula + parameters,
    // then rebuilds the sample data. C# returns IModel; the C++ clone returns the concrete
    // model by value (ModelBase carries no clone(), the Phase 4 decision). ------------------
    BivariateDistribution clone() const {
        BivariateDistribution result;  // default Normal copula, empty parameters
        result.marginal_x_ = marginal_x_;  // alias (C# `_marginalX = MarginalX`)
        result.marginal_y_ = marginal_y_;
        result.copula_ = copula_->clone();
        result.set_use_default_flat_priors(use_default_flat_priors());

        std::vector<ModelParameter> parms;
        parms.reserve(parameters_.size());
        for (std::size_t i = 0; i < parameters_.size(); ++i)
            parms.push_back(parameters_[i].clone());
        result.parameters_ = std::move(parms);

        result.copula_estimation_method_ = copula_estimation_method_;
        result.set_sample_data();
        return result;
    }

    // --- Validate (C# line 721). ---------------------------------------------------------
    ValidationResult validate() const override {
        ValidationResult result;

        // Marginal checks.
        if (marginal_x_ == nullptr) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Marginal distribution for X is not specified.");
        } else {
            ValidationResult mx = marginal_x_->validate();
            if (!mx.is_valid) {
                result.is_valid = false;
                result.validation_messages.push_back("Error: Marginal-X validation failed:");
                result.validation_messages.insert(result.validation_messages.end(),
                                                  mx.validation_messages.begin(),
                                                  mx.validation_messages.end());
            }
        }

        if (marginal_y_ == nullptr) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Marginal distribution for Y is not specified.");
        } else {
            ValidationResult my = marginal_y_->validate();
            if (!my.is_valid) {
                result.is_valid = false;
                result.validation_messages.push_back("Error: Marginal-Y validation failed:");
                result.validation_messages.insert(result.validation_messages.end(),
                                                  my.validation_messages.begin(),
                                                  my.validation_messages.end());
            }
        }

        // Copula.
        if (copula_ == nullptr) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: Copula is not specified.");
        }

        // Parameter checks.
        int expected_param_count = copula_ != nullptr ? copula_->number_of_copula_parameters() : 0;
        if (parameters_.size() != static_cast<std::size_t>(expected_param_count)) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Copula parameter collection is missing or has an unexpected size. "
                "Expected " +
                std::to_string(expected_param_count) + " parameter(s) for the current copula.");
        } else {
            for (const ModelParameter& p : parameters_) {
                if (p.lower_bound() > p.upper_bound()) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: " + p.name() +
                        " has inconsistent bounds. LowerBound must be less than or equal to "
                        "UpperBound.");
                }
                if (p.value() < p.lower_bound() || p.value() > p.upper_bound()) {
                    result.is_valid = false;
                    result.validation_messages.push_back("Error: " + p.name() +
                                                         " is outside its specified bounds.");
                }
                if (!p.prior_distribution().parameters_valid()) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: " + p.name() +
                        " prior distribution is not defined or is invalid.");
                }
            }
        }

        // Sample data checks (read-only; the series are populated as a side effect of marginal
        // assignment, not here).
        if (sample_data_x_.empty() || sample_data_y_.empty()) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: No overlapping, non-outlier exact data is available to estimate the "
                "copula.");
        } else if (sample_data_x_.size() != sample_data_y_.size()) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The paired sample data for the copula is inconsistent. X and Y sample "
                "sizes do not match.");
        }

        // Pseudo likelihood requires strictly-interior unit-interval data.
        if (copula_estimation_method_ == CopulaEstimationMethod::PseudoLikelihood &&
            !sample_data_x_.empty()) {
            bool out_of_range = false;
            for (double u : sample_data_x_)
                if (u <= 0.0 || u >= 1.0) out_of_range = true;
            for (double v : sample_data_y_)
                if (v <= 0.0 || v >= 1.0) out_of_range = true;
            if (out_of_range) {
                result.is_valid = false;
                result.validation_messages.push_back(
                    "Error: Pseudo likelihood estimation requires plotting positions strictly "
                    "between 0 and 1.");
            }
        }

        return result;
    }

   private:
    // C# ParameterNameFor (line 351): index 0 is the dependency parameter; StudentT index 1 is
    // the degrees of freedom; the generic fallback future-proofs multi-parameter copulas.
    static std::string parameter_name_for(const BivariateCopula& copula, int index) {
        if (index == 0) return "Dependency (θ)";  // "Dependency (theta)"
        if (index == 1 && dynamic_cast<const StudentTCopula*>(&copula) != nullptr)
            return "DegreesOfFreedom";
        return "Parameter" + std::to_string(index + 1);
    }

    // C# InitialValueFor (line 367): the midpoint of [lower, upper], except StudentT's
    // degrees-of-freedom (index 1) which defaults to 5.
    static double initial_value_for(const BivariateCopula& copula, int index, double lower,
                                    double upper) {
        if (index == 1 && dynamic_cast<const StudentTCopula*>(&copula) != nullptr) return 5.0;
        return 0.5 * (lower + upper);
    }

    // C# SafeLogPDF (line 598): pseudo-likelihood per-sample guard. Silent no-throw wrapper:
    // any thrown exception (C# Debug.WriteLine + return) or non-finite value -> -inf.
    static double safe_log_pdf(const BivariateCopula& model, double u, double v) {
        try {
            double log_lh = model.log_pdf(u, v);
            return numerics::is_finite(log_lh) ? log_lh
                                               : -std::numeric_limits<double>::infinity();
        } catch (...) {
            return -std::numeric_limits<double>::infinity();
        }
    }

    // C# SafeIfmLogPDF (line 623): IFM per-sample guard. Evaluates each marginal CDF to obtain
    // (u, v), then the copula LogPDF; silent no-throw wrapper -> -inf on any throw / non-finite.
    static double safe_ifm_log_pdf(const BivariateCopula& model, double x, double y) {
        try {
            double u = model.marginal_distribution_x->cdf(x);
            double v = model.marginal_distribution_y->cdf(y);
            if (!numerics::is_finite(u) || !numerics::is_finite(v))
                return -std::numeric_limits<double>::infinity();
            double log_lh = model.log_pdf(u, v);
            return numerics::is_finite(log_lh) ? log_lh
                                               : -std::numeric_limits<double>::infinity();
        } catch (...) {
            return -std::numeric_limits<double>::infinity();
        }
    }

    // Shared setup for the pointwise methods: a cloned copula with the proposed parameters and
    // cloned marginal distributions. Returns nullptr if a marginal distribution is unavailable
    // (mirrors the C# `MarginalX.Distribution!` non-null assumption -- the guard folds a null
    // marginal into the all -inf pointwise result the callers already produce on a bad shape).
    std::unique_ptr<BivariateCopula> clone_model_with_marginals(
        const std::vector<double>& parameters) const {
        const UnivariateDistributionBase* dist_x =
            marginal_x_ != nullptr ? marginal_x_->distribution() : nullptr;
        const UnivariateDistributionBase* dist_y =
            marginal_y_ != nullptr ? marginal_y_->distribution() : nullptr;
        if (dist_x == nullptr || dist_y == nullptr) return nullptr;

        std::unique_ptr<BivariateCopula> model = copula_->clone();
        model->set_copula_parameters(parameters);
        model->marginal_distribution_x = dist_x->clone();
        model->marginal_distribution_y = dist_y->clone();
        return model;
    }

    // Collect pointers to the non-low-outlier exact ordinates (index-sorted by the caller).
    static std::vector<const ExactData*> non_low_outlier(const ExactSeries& series) {
        std::vector<const ExactData*> result;
        for (std::size_t i = 0; i < series.count(); ++i)
            if (!series[i].is_low_outlier()) result.push_back(&series[i]);
        return result;
    }

    // C# GetEligibleExactData (line 491, v2.0.0): sorted, non-low-outlier exact data from both
    // marginals. Empty pairs when either marginal is missing or invalid (folds the C#
    // `MarginalX.DataFrame is null` half of the guard into validate(), same as before -- see
    // the file header).
    std::pair<std::vector<const ExactData*>, std::vector<const ExactData*>>
    get_eligible_exact_data() const {
        if (marginal_x_ == nullptr || !marginal_x_->validate().is_valid) return {};
        if (marginal_y_ == nullptr || !marginal_y_->validate().is_valid) return {};

        marginal_x_->data_frame().exact_series().sort_by_index();
        marginal_y_->data_frame().exact_series().sort_by_index();

        return {non_low_outlier(marginal_x_->data_frame().exact_series()),
                non_low_outlier(marginal_y_->data_frame().exact_series())};
    }

    // C# AddPairedSampleData (line 558, v2.0.0): two-pointer index merge; adds
    // PlottingPositionComplement pairs under PseudoLikelihood, raw Value pairs otherwise. No
    // validation -- the hot path for an already-validated pseudo sample (or any IFM sample).
    static void add_paired_sample_data(const std::vector<const ExactData*>& data_x,
                                       const std::vector<const ExactData*>& data_y,
                                       bool use_pseudo_likelihood,
                                       std::vector<double>& sample_data_x,
                                       std::vector<double>& sample_data_y) {
        std::size_t i = 0, j = 0;
        while (i < data_x.size() && j < data_y.size()) {
            int idx_x = data_x[i]->index();
            int idx_y = data_y[j]->index();
            if (idx_x == idx_y) {
                sample_data_x.push_back(use_pseudo_likelihood ? data_x[i]->plotting_position_complement()
                                                              : data_x[i]->value());
                sample_data_y.push_back(use_pseudo_likelihood ? data_y[j]->plotting_position_complement()
                                                              : data_y[j]->value());
                ++i;
                ++j;
            } else if (idx_x < idx_y) {
                ++i;
            } else {
                ++j;
            }
        }
    }

    // C# AddPairedSampleDataAndValidate (line 602, v2.0.0): same merge, but always adds pseudo
    // (PlottingPositionComplement) pairs and flags whether either marginal supplied a paired
    // value outside the open interval (0, 1).
    static std::pair<bool, bool> add_paired_sample_data_and_validate(
        const std::vector<const ExactData*>& data_x, const std::vector<const ExactData*>& data_y,
        std::vector<double>& sample_data_x, std::vector<double>& sample_data_y) {
        std::size_t i = 0, j = 0;
        bool invalid_x = false, invalid_y = false;
        while (i < data_x.size() && j < data_y.size()) {
            int idx_x = data_x[i]->index();
            int idx_y = data_y[j]->index();
            if (idx_x == idx_y) {
                double value_x = data_x[i]->plotting_position_complement();
                double value_y = data_y[j]->plotting_position_complement();
                invalid_x = invalid_x || !(value_x > 0.0 && value_x < 1.0);
                invalid_y = invalid_y || !(value_y > 0.0 && value_y < 1.0);
                sample_data_x.push_back(value_x);
                sample_data_y.push_back(value_y);
                ++i;
                ++j;
            } else if (idx_x < idx_y) {
                ++i;
            } else {
                ++j;
            }
        }
        return {invalid_x, invalid_y};
    }

    std::unique_ptr<BivariateCopula> copula_;
    IUnivariateModel* marginal_x_ = nullptr;
    IUnivariateModel* marginal_y_ = nullptr;
    std::vector<double> sample_data_x_;
    std::vector<double> sample_data_y_;
    CopulaEstimationMethod copula_estimation_method_ =
        CopulaEstimationMethod::InferenceFromMargins;

    // Pseudo-likelihood validation cache (C# `_validatedPseudoDataFrameX/Y` +
    // `_validatedPseudoVersionX/Y`, v2.0.0): identity of the last-validated marginal DataFrame
    // plus its plotting_position_version() at validation time, so an unchanged sample skips
    // re-validation on every subsequent SetSampleData call. Reset to defaults on Clone() (a
    // fresh clone re-validates once, matching the C#: Clone() never copies these fields).
    //
    // KNOWN GAP (see data_frame.hpp's plotting_position_version() accessor comment for detail):
    // in the real C#, a BARE `ExactData.PlottingPosition = x` mutation (bypassing
    // CalculatePlottingPositions entirely) also bumps the target DataFrame's version stamp via
    // an INPC handler, so this cache correctly detects it as stale on the next SetSampleData
    // call. This port's `Data::set_plotting_position()` has no back-pointer to its owning
    // DataFrame, so it cannot bump plotting_position_version() here -- a caller who bypasses
    // calculate_plotting_positions() this way will NOT see this cache invalidate. No fixture
    // exercises that specific sequence; the oracle-verified PseudoLikelihood scenario is the
    // first-validation auto-repair from never-computed (default 0.0) positions, which this
    // cache handles identically to the C#.
    DataFrame* validated_pseudo_data_frame_x_ = nullptr;
    DataFrame* validated_pseudo_data_frame_y_ = nullptr;
    std::int64_t validated_pseudo_version_x_ = -1;
    std::int64_t validated_pseudo_version_y_ = -1;
};

}  // namespace corehydro::models
