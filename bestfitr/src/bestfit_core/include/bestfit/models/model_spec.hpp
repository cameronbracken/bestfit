// bestfit ADDITION -- no upstream C# counterpart.
//
// The shared `model_estimation` fixture model builder. A fixture case's `construct.model`
// object (see fixtures/README.md, "model_estimation") names one of the four Phase 5 model
// types and carries its data (a flat `dataset` reference or a full `data_frame` of exact /
// interval / threshold / uncertain arrays), optional nonstationary trend specs, and optional
// explicit parameter values. Like mcmc/model_registry.hpp, this header exists so the three
// fixture harnesses (C++ test runner, bestfitr glue, bestfitpy glue) build the EXACT same
// model from the same spec through ONE construction path: each harness re-serializes its
// native `construct.model` structure to a JSON string (nlohmann `dump()` in C++,
// `jsonlite::toJSON(digits = I(17))` in R, `json.dumps()` in Python -- all three round-trip
// doubles exactly) and calls `build_model_from_json`, which parses with models/json_lite.hpp
// and dispatches on the spec's `type` discriminator. There is no closed model-name registry:
// the "registry" is this one dispatch plus the same univariate distribution factory every
// other fixture kind uses.
//
// Spec shape (all types; `type` defaults to "univariate_distribution"):
//   { "type": "univariate_distribution", "family": "<factory name>",
//     "dataset": "<datasets key>" | "data_frame": { ... },
//     "trends": [ { "parameter": p, "type": "<TrendModelType name>",
//                   "start_index"?: i, "values"?: [ ... ] } ],
//     "parameter_values"?: [ ... ] }
//   { "type": "mixture", "families": [ ... ], "zero_inflated"?: b, <data>, ... }
//   { "type": "competing_risks", "families": [ ... ], <data>, ... }
//   { "type": "point_process", <data>, "use_defaults"?: b, "threshold"?: x,
//     "total_years"?: y, ... }
// `dataset` is resolved by the CALLING harness (the file-level `datasets` map never reaches
// this builder); the resolved values arrive as the `dataset` argument and become an
// ExactSeries with sequential 0-based indexes -- exactly the Phase 4 vector-ctor path.
// `data_frame` arrays carry their values inline:
//   exact:     { "index": i, "value": x, "is_low_outlier"?: b }
//   interval:  { "index": i, "lower": lo, "value": x, "upper": hi }
//   threshold: { "start_index": i, "end_index": j, "value": x, "number_above": n }
//   uncertain: { "index": i, "distribution": { "family": "<name>", "parameters": [ ... ] } }
// plus an optional frame-level "low_outlier_threshold" and an optional frame-level
// "mgbt_low_outliers" flag (M14) that triggers the public set_low_outliers_from_mgbt() path
// after the series are assigned.
//
// `trends` attach via UnivariateDistributionModel::set_trend_model (which supplies the
// data-driven defaults exactly like the C# SetTrendModel); `start_index` then overrides the
// trend's anchor, and explicit trend `values` / the model-level `parameter_values` are
// applied through ONE final set_parameter_values call -- the sync-safe setter the model
// header mandates (poking parameters() elements directly would desync the trend copies).
// Attaching any trend first sets is_nonstationary(true).
//
// All four models derive from ModelBase (what the estimators accept) -- including
// CompetingRisksModel, which is NOT an IUnivariateModel (the C# omits it) -- so the builder
// returns std::unique_ptr<ModelBase>. Callers needing the seeded-simulation surface
// dynamic_cast the result to ISimulatable<std::vector<double>>* (implemented by all four).
#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bestfit/models/data_frame/data_frame.hpp"
#include "bestfit/models/json_lite.hpp"
#include "bestfit/models/support/model_base.hpp"
#include "bestfit/models/trend_functions/support/trend_model_type.hpp"
#include "bestfit/models/univariate_distribution/bulletin17c_distribution.hpp"
#include "bestfit/models/univariate_distribution/competing_risks_model.hpp"
#include "bestfit/models/univariate_distribution/mixture_model.hpp"
#include "bestfit/models/univariate_distribution/point_process_model.hpp"
#include "bestfit/models/univariate_distribution/univariate_distribution_model.hpp"
#include "bestfit/models/univariate_distribution/univariate_distribution_model_trends.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_factory.hpp"

namespace bestfit::models::spec {

inline trend_functions::TrendModelType parse_trend_model_type(const std::string& name) {
    using TT = trend_functions::TrendModelType;
    if (name == "Constant") return TT::Constant;
    if (name == "Cubic") return TT::Cubic;
    if (name == "Exponential") return TT::Exponential;
    if (name == "Linear") return TT::Linear;
    if (name == "Logistic") return TT::Logistic;
    if (name == "Power") return TT::Power;
    if (name == "Quadratic") return TT::Quadratic;
    if (name == "Reciprocal") return TT::Reciprocal;
    if (name == "Sinusoidal") return TT::Sinusoidal;
    if (name == "StepFunction") return TT::StepFunction;
    if (name == "GeneralLinear") return TT::GeneralLinear;
    throw std::runtime_error("unknown trend model type: " + name);
}

// A `{ "family": ..., "parameters": [...] }` distribution spec -> a parameterized
// distribution through the same factory every other fixture kind uses.
inline std::unique_ptr<numerics::distributions::UnivariateDistributionBase>
build_spec_distribution(const JsonValue& spec) {
    auto dist = numerics::distributions::create_distribution(spec.at("family").as_string());
    if (spec.contains("parameters")) dist->set_parameters(spec.at("parameters").as_double_vector());
    return dist;
}

// A `data_frame` spec object -> a DataFrame (threshold processing happens later, at the
// model boundary -- every model's set_data_frame runs process_threshold_series itself).
inline DataFrame build_data_frame(const JsonValue& spec) {
    DataFrame df;

    if (spec.contains("exact")) {
        std::vector<ExactData> items;
        for (const JsonValue& e : spec.at("exact").items())
            items.emplace_back(e.at("index").as_int(), e.at("value").as_double(), 0.0,
                               e.value_or("is_low_outlier", false));
        df.set_exact_series(ExactSeries(items));
    }

    if (spec.contains("interval")) {
        std::vector<IntervalData> items;
        for (const JsonValue& e : spec.at("interval").items())
            items.emplace_back(e.at("index").as_int(), e.at("lower").as_double(),
                               e.at("value").as_double(), e.at("upper").as_double());
        df.set_interval_series(IntervalSeries(items));
    }

    if (spec.contains("threshold")) {
        std::vector<ThresholdData> items;
        for (const JsonValue& e : spec.at("threshold").items()) {
            ThresholdData td(e.at("start_index").as_int(), e.at("end_index").as_int(),
                             e.at("value").as_double());
            td.set_number_above(e.at("number_above").as_int());
            items.push_back(td);
        }
        df.set_threshold_series(ThresholdSeries(items));
    }

    if (spec.contains("uncertain")) {
        std::vector<UncertainData> items;
        for (const JsonValue& e : spec.at("uncertain").items())
            items.emplace_back(e.at("index").as_int(),
                               build_spec_distribution(e.at("distribution")));
        df.set_uncertain_series(UncertainSeries(items));
    }

    if (spec.contains("low_outlier_threshold"))
        df.set_low_outlier_threshold(spec.at("low_outlier_threshold").as_double());

    // Optional MGBT trigger (M14): the public set_low_outliers_from_mgbt() path, run at the
    // frame boundary (before the model ctor sees the frame) exactly like a C# caller would --
    // flags low outliers, sets low_outlier_threshold, and thereby left-censors the flagged
    // values in the model's likelihood. Mutually exclusive with explicit `is_low_outlier`
    // flags / `low_outlier_threshold` by fixture convention (MGBT clears both first).
    if (spec.value_or("mgbt_low_outliers", false)) df.set_low_outliers_from_mgbt();

    return df;
}

// Resolves a model spec's data source: the inline `data_frame` object when present,
// otherwise an exact-only frame over the harness-resolved `dataset` values (sequential
// 0-based indexes -- the same frame the Phase 4 vector ctors build).
inline DataFrame build_model_data_frame(const JsonValue& model,
                                        const std::vector<double>& dataset) {
    if (model.contains("data_frame")) return build_data_frame(model.at("data_frame"));
    if (model.contains("dataset")) {
        DataFrame df;
        df.set_exact_series(ExactSeries(dataset));
        return df;
    }
    throw std::runtime_error("model spec requires either 'dataset' or 'data_frame'");
}

// Optional model-level `parameter_values`: one sync-safe set_parameter_values call.
inline void apply_parameter_values(ModelBase& model, const JsonValue& spec) {
    if (!spec.contains("parameter_values")) return;
    model.set_parameter_values(spec.at("parameter_values").as_double_vector());
}

// `families` -> distribution types through the factory (the string->type mapping every
// other fixture kind already relies on).
inline std::vector<numerics::distributions::UnivariateDistributionType> parse_families(
    const JsonValue& model) {
    std::vector<numerics::distributions::UnivariateDistributionType> types;
    for (const JsonValue& f : model.at("families").items())
        types.push_back(numerics::distributions::create_distribution(f.as_string())->type());
    return types;
}

inline std::unique_ptr<ModelBase> build_univariate_distribution_model(
    const JsonValue& model, const std::vector<double>& dataset) {
    auto dist = numerics::distributions::create_distribution(model.at("family").as_string());

    std::unique_ptr<UnivariateDistributionModel> m;
    if (model.contains("data_frame")) {
        m = std::make_unique<UnivariateDistributionModel>(
            build_data_frame(model.at("data_frame")), std::move(dist));
    } else if (model.contains("dataset")) {
        // The Phase 4 vector-ctor shim, byte-for-byte (mle_normal_smoke.json et al.).
        m = std::make_unique<UnivariateDistributionModel>(std::move(dist), dataset);
    } else {
        throw std::runtime_error("model spec requires either 'dataset' or 'data_frame'");
    }

    if (model.contains("trends")) {
        const std::vector<JsonValue>& trends = model.at("trends").items();
        m->set_is_nonstationary(true);

        // Pass 1: attach every trend (set_trend_model supplies the C#-mirrored defaults),
        // then override the anchor where the spec asks.
        for (const JsonValue& t : trends) {
            int p = t.at("parameter").as_int();
            m->set_trend_model(p, parse_trend_model_type(t.at("type").as_string()));
            if (t.contains("start_index"))
                m->trend_models()[static_cast<std::size_t>(p)]->set_start_index(
                    t.at("start_index").as_int());
        }

        // Pass 2 (after the parameter layout is final): explicit per-trend values overwrite
        // their slice of the full parameter vector, applied through ONE
        // set_parameter_values call (the sync-safe setter -- see the file header).
        bool has_values = false;
        std::vector<double> full;
        full.reserve(m->parameters().size());
        for (const ModelParameter& mp : m->parameters()) full.push_back(mp.value());
        for (const JsonValue& t : trends) {
            if (!t.contains("values")) continue;
            has_values = true;
            int p = t.at("parameter").as_int();
            std::size_t offset = 0;
            for (int j = 0; j < p; ++j)
                offset += static_cast<std::size_t>(
                    m->trend_models()[static_cast<std::size_t>(j)]->number_of_parameters());
            std::vector<double> values = t.at("values").as_double_vector();
            std::size_t n = static_cast<std::size_t>(
                m->trend_models()[static_cast<std::size_t>(p)]->number_of_parameters());
            if (values.size() != n)
                throw std::runtime_error(
                    "trend spec 'values' length does not match the trend's parameter count");
            for (std::size_t k = 0; k < n; ++k) full[offset + k] = values[k];
        }
        if (has_values) m->set_parameter_values(full);
    }

    apply_parameter_values(*m, model);
    return m;
}

inline std::unique_ptr<ModelBase> build_mixture_model(const JsonValue& model,
                                                      const std::vector<double>& dataset) {
    auto m = std::make_unique<MixtureModel>(build_model_data_frame(model, dataset),
                                            parse_families(model),
                                            model.value_or("zero_inflated", false));
    apply_parameter_values(*m, model);
    return m;
}

inline std::unique_ptr<ModelBase> build_competing_risks_model(
    const JsonValue& model, const std::vector<double>& dataset) {
    auto m = std::make_unique<CompetingRisksModel>(build_model_data_frame(model, dataset),
                                                   parse_families(model));
    apply_parameter_values(*m, model);
    return m;
}

inline std::unique_ptr<ModelBase> build_point_process_model(
    const JsonValue& model, const std::vector<double>& dataset) {
    // Ctor surface: default-construct (non-seasonal GEV competing-risks distribution),
    // assign the frame (set_data_frame runs the AMS/defaults cascade), then the optional
    // knobs in C#-property order: UseDefaults before the explicit Threshold/TotalYears so
    // an explicit value is never clobbered by the defaults cascade. The deferred seasonal
    // path is deliberately NOT exposed.
    auto m = std::make_unique<PointProcessModel>();
    m->set_data_frame(build_model_data_frame(model, dataset));
    if (model.contains("use_defaults")) m->set_use_defaults(model.at("use_defaults").as_bool());
    if (model.contains("threshold")) m->set_threshold(model.at("threshold").as_double());
    if (model.contains("total_years")) m->set_total_years(model.at("total_years").as_double());
    apply_parameter_values(*m, model);
    return m;
}

// The `construct.model` dispatch: `type` defaults to the Phase 4 behavior.
inline std::unique_ptr<ModelBase> build_model(const JsonValue& model,
                                              const std::vector<double>& dataset) {
    std::string type = model.value_or("type", "univariate_distribution");
    if (type == "univariate_distribution") return build_univariate_distribution_model(model, dataset);
    if (type == "mixture") return build_mixture_model(model, dataset);
    if (type == "competing_risks") return build_competing_risks_model(model, dataset);
    if (type == "point_process") return build_point_process_model(model, dataset);
    throw std::runtime_error("unknown model_estimation model type: " + type);
}

// The single shared entry point all three harnesses call (see the file header).
inline std::unique_ptr<ModelBase> build_model_from_json(const std::string& model_json,
                                                        const std::vector<double>& dataset) {
    return build_model(parse_json(model_json), dataset);
}

// A `type: "bulletin17c"` spec -> a concrete Bulletin17CDistribution (B11).
//
// WIRING DECISION (deviation from the brief -- documented): the brief assumed
// Bulletin17CDistribution "derives from ModelBase" and could ride the build_model_from_json ->
// ModelBase path. It does NOT: it derives from IGMMModel + ISimulatable + IUnivariateModel, and
// NONE of those derive from ModelBase (IUnivariateModel is a deliberately minimal capability
// mixin, i_univariate_model.hpp). So a Bulletin17CDistribution* is not convertible to a
// ModelBase*, and this needs its OWN construction entry point returning the CONCRETE type. That
// concrete pointer is exactly what both consumers want anyway: the GMM glue passes it straight
// as the IGMMModel& the estimator ctor takes (no cross-cast), and the seeded-draw glue calls
// its ISimulatable generate_random_values directly.
//
// The spec carries the distribution `family` (any of the six IsSupportedDistributionType
// families -- Exponential, Gamma, LogNormal, LogPearsonTypeIII (default), Normal,
// PearsonTypeIII; an unsupported type throws from the B17C ctor), the Phase 5 DataFrame block or
// a flat `dataset` (via build_model_data_frame, carrying the same low_outlier_threshold /
// mgbt_low_outliers keys), and optional explicit `parameter_values` (applied last, exactly like
// every other model type).
inline std::unique_ptr<Bulletin17CDistribution> build_bulletin17c_model(
    const JsonValue& model, const std::vector<double>& dataset) {
    DataFrame df = build_model_data_frame(model, dataset);
    numerics::distributions::UnivariateDistributionType type =
        model.contains("family")
            ? numerics::distributions::create_distribution(model.at("family").as_string())->type()
            : numerics::distributions::UnivariateDistributionType::LogPearsonTypeIII;
    auto m = std::make_unique<Bulletin17CDistribution>(std::move(df), type);
    if (model.contains("parameter_values"))
        m->set_parameter_values(model.at("parameter_values").as_double_vector());
    return m;
}

inline std::unique_ptr<Bulletin17CDistribution> build_bulletin17c_from_json(
    const std::string& model_json, const std::vector<double>& dataset) {
    return build_bulletin17c_model(parse_json(model_json), dataset);
}

}  // namespace bestfit::models::spec
