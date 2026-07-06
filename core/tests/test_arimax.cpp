// T3 support ctest (C++-only): the ARIMAX TimeSeries model -- the richest ModelBase family in
// the TimeSeries slice (ARIMA differenced+transformed mean PLUS an inline polynomial trend, an
// inline Fourier seasonality term, and exogenous covariate regression). Its full-fit likelihood
// / MLE / MAP / posterior oracles come from the P4 dotnet emitter, NOT T3, so this file
// transcribes only the STRUCTURAL / DETERMINISM / TRIVIALLY-ANALYTIC assertions (internal
// support test territory -- hardcoded oracles are correct here; public-API oracle values stay
// in fixtures/).
//
// Structural oracles transcribed (values unaltered where they exist) from
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/TimeSeriesModels/ARIMAXTests.cs @ fc28c0c
//
// The C# fixtures draw their data from System.Random (a .NET LCG, not the ported Mersenne
// Twister), so the exact data VALUES are unreproducible in C++. This file regenerates
// structurally-equivalent data (same AR(1)+trend / monthly seasonal+trend shapes, ported
// bit-exact MersenneTwister, positive mean ~1000 so the log/Box-Cox transforms are well
// defined), then applies the same STRUCTURAL assertions (parameter counts/names/bounds,
// finiteness, sentinels, differencing/seasonality/trend layout, seeded-stream determinism,
// validation outcomes). No data-value oracle is claimed for the generated fixtures.
//
// Skipped C# test methods (see task-T3-report.md for the full list + reasons):
//   - XML tests (Test_Constructor_XElement_RestoresModel, Test_ToXElement_*,
//     Test_RoundTrip_PreservesAllProperties, Test_CovariateExtension_XmlSerialization):
//     XML is a project-wide non-port.
//   - Clone tests (Test_Clone_*): the C++ core has no virtual IModel::Clone (see model_base.hpp);
//     no fit path in T3 needs a clone, so clone() is omitted (T1 / T2 / S4 precedent).
//   - Test_SetParameterValues_NullParameters_ThrowsException: VACUOUS (const-ref vector; no null).
//   - The CovariateExtension Predict/Generate tests (Test_Predict_CovariateExtension*,
//     Test_GenerateRandomValues_CovariateExtension*): the covariate forecast-tail extension
//     (BlockBootstrap/KNN) needs the DEFERRED heavy TimeSeries container (ResampleWithBlockBootstrap
//     / ResampleWithKNN); deferred with that container per PHASE7_PLAN.md.
//
// P4 pending (the numeric quantities this file deliberately does NOT oracle; P4 dotnet emitter):
//   - exact ARIMAX DataLogLikelihood / PriorLogLikelihood / pointwise-component numeric values,
//   - exact Residuals values under each trend/seasonality/covariate/Transform/differencing config,
//   - exact Predict() forecast values (incl. the reverse-differencing integration) and the
//     seeded cross-language GenerateRandomValues digest.
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "bestfit/models/support/model_base.hpp"
#include "bestfit/models/support/simulatable.hpp"
#include "bestfit/models/time_series/arimax.hpp"
#include "bestfit/models/time_series/transform_type.hpp"
#include "bestfit/numerics/data/time_series/time_series.hpp"
#include "bestfit/numerics/sampling/mersenne_twister.hpp"
#include "bestfit/numerics/tools.hpp"
#include "check.hpp"

namespace {

using bestfit::models::ARIMAX;
using bestfit::models::Transform;
using bestfit::numerics::data::TimeInterval;
using bestfit::numerics::data::TimeSeries;
using bestfit::numerics::sampling::MersenneTwister;

// Compile-time mirror of Model_InheritsFromModelBase / Model_ImplementsISimulatable.
static_assert(std::is_base_of<bestfit::models::ModelBase, ARIMAX>::value,
              "ARIMAX must derive from ModelBase");
static_assert(std::is_base_of<bestfit::models::ISimulatable<std::vector<double>>, ARIMAX>::value,
              "ARIMAX must implement ISimulatable<std::vector<double>>");

// ---- Test-data helpers (mirror the C# private fixtures; data regenerated with the ported RNG) ----

// 60 annual observations, AR(1)-with-trend, positive mean ~1000 (mirrors CreateSampleTimeSeries).
TimeSeries make_sample_series() {
    TimeSeries ts(TimeInterval::OneYear, 0, 59);
    MersenneTwister rng(12345);
    const double mean = 1000.0, phi = 0.6, sigma = 100.0, trend_slope = 5.0;
    double prev = mean;
    for (int i = 0; i < ts.count(); ++i) {
        double innovation = rng.next_double() * 2.0 - 1.0;
        double trend = trend_slope * i;
        double value = mean + trend + phi * (prev - mean - trend_slope * (i - 1)) + sigma * innovation;
        ts[i].set_value(value);
        prev = value;
    }
    return ts;
}

// 240 monthly observations, seasonal + trend + noise (mirrors CreateMonthlyTimeSeries).
TimeSeries make_monthly_series() {
    TimeSeries ts(TimeInterval::OneMonth, 0, 239);
    MersenneTwister rng(54321);
    for (int i = 0; i < ts.count(); ++i) {
        double seasonal = 50.0 * std::sin(2.0 * bestfit::numerics::kPi * i / 12.0);
        double trend = 0.5 * i;
        double noise = rng.next_double() * 20.0 - 10.0;
        ts[i].set_value(500.0 + seasonal + trend + noise);
    }
    return ts;
}

// 15 annual observations, deterministic 100 + 10i (mirrors CreateShortTimeSeries).
TimeSeries make_short_series() {
    TimeSeries ts(TimeInterval::OneYear, 0, 14);
    for (int i = 0; i < ts.count(); ++i) ts[i].set_value(100.0 + i * 10.0);
    return ts;
}

// Covariate aligned to the target series (mirrors CreateCovariateTimeSeries).
TimeSeries make_covariate_series(const TimeSeries& target) {
    TimeSeries ts(target.time_interval(), 0, target.count() - 1);
    MersenneTwister rng(67890);
    for (int i = 0; i < ts.count(); ++i)
        ts[i].set_value(0.5 * target[i].value() + rng.next_double() * 50.0);
    return ts;
}

// 5 annual observations -> too short for the 10-observation validation guard.
TimeSeries make_tiny_series() {
    TimeSeries ts(TimeInterval::OneYear, 0, 4);
    for (int i = 0; i < ts.count(); ++i) ts[i].set_value(i);
    return ts;
}

bool messages_contain(const bestfit::models::ValidationResult& r, const std::string& needle) {
    for (const auto& m : r.validation_messages)
        if (m.find(needle) != std::string::npos) return true;
    return false;
}

// ============================ Constructors ============================

void test_arimax_constructors() {
    ARIMAX empty;
    CHECK_EQ(empty.ar_order_p(), 1);
    CHECK_EQ(empty.diff_order_d(), 0);
    CHECK_EQ(empty.ma_order_q(), 0);
    CHECK_TRUE(empty.include_intercept());

    TimeSeries ts = make_sample_series();
    ARIMAX m(ts);
    CHECK_EQ(m.time_series().count(), ts.count());  // adapted AreSame -> structural equality
    CHECK_EQ(m.ar_order_p(), 1);
}

// ============================ Properties ============================

void test_arimax_properties() {
    TimeSeries ts = make_sample_series();

    ARIMAX m;
    m.set_time_series(ts);
    CHECK_EQ(m.time_series().count(), ts.count());

    ARIMAX mp(ts);
    mp.set_ar_order_p(3);
    CHECK_EQ(mp.ar_order_p(), 3);

    ARIMAX mq(ts);
    mq.set_ma_order_q(2);
    CHECK_EQ(mq.ma_order_q(), 2);

    ARIMAX md(ts);
    md.set_diff_order_d(1);
    CHECK_EQ(md.diff_order_d(), 1);

    ARIMAX mx(ts);
    mx.set_x_order_b(2);
    CHECK_EQ(mx.x_order_b(), 2);

    ARIMAX mi(ts);
    mi.set_include_intercept(false);
    CHECK_TRUE(!mi.include_intercept());

    ARIMAX mtr(ts);
    mtr.set_trend_type(ARIMAX::Trend::Quadratic);
    CHECK_TRUE(mtr.trend_type() == ARIMAX::Trend::Quadratic);

    TimeSeries monthly = make_monthly_series();
    ARIMAX ms(monthly);
    ms.set_include_seasonality(true);
    CHECK_TRUE(ms.include_seasonality());

    ARIMAX mt(ts);
    mt.set_transform_type(Transform::Logarithmic);
    CHECK_TRUE(mt.transform_type() == Transform::Logarithmic);
}

// ============================ Parameter layout ============================

void test_arimax_number_of_parameters() {
    TimeSeries ts = make_sample_series();

    // AR(1) with intercept: mu + phi + sigma = 3.
    ARIMAX ar1(ts);
    ar1.set_ar_order_p(1);
    ar1.set_ma_order_q(0);
    ar1.set_trend_type(ARIMAX::Trend::None);
    ar1.set_include_seasonality(false);
    CHECK_EQ(ar1.number_of_parameters(), 3);

    // ARMA(1,1) with intercept: mu + phi + theta + sigma = 4.
    ARIMAX arma11(ts);
    arma11.set_ar_order_p(1);
    arma11.set_ma_order_q(1);
    arma11.set_trend_type(ARIMAX::Trend::None);
    CHECK_EQ(arma11.number_of_parameters(), 4);

    // AR(1) + linear trend: mu + gamma + phi + sigma = 4.
    ARIMAX lin(ts);
    lin.set_ar_order_p(1);
    lin.set_ma_order_q(0);
    lin.set_trend_type(ARIMAX::Trend::Linear);
    CHECK_EQ(lin.number_of_parameters(), 4);

    // AR(1) + quadratic trend: mu + gamma1 + gamma2 + phi + sigma = 5.
    ARIMAX quad(ts);
    quad.set_ar_order_p(1);
    quad.set_ma_order_q(0);
    quad.set_trend_type(ARIMAX::Trend::Quadratic);
    CHECK_EQ(quad.number_of_parameters(), 5);

    // Trend increments: +1 linear, +2 quadratic, +3 cubic.
    ARIMAX tr(ts);
    tr.set_trend_type(ARIMAX::Trend::None);
    int base = tr.number_of_parameters();
    tr.set_trend_type(ARIMAX::Trend::Linear);
    CHECK_EQ(tr.number_of_parameters(), base + 1);
    tr.set_trend_type(ARIMAX::Trend::None);
    tr.set_trend_type(ARIMAX::Trend::Quadratic);
    CHECK_EQ(tr.number_of_parameters(), base + 2);
    tr.set_trend_type(ARIMAX::Trend::None);
    tr.set_trend_type(ARIMAX::Trend::Cubic);
    CHECK_EQ(tr.number_of_parameters(), base + 3);

    // Seasonality adds a sin/cos Fourier pair.
    TimeSeries monthly = make_monthly_series();
    ARIMAX seas(monthly);
    seas.set_include_seasonality(false);
    int base_s = seas.number_of_parameters();
    seas.set_include_seasonality(true);
    CHECK_TRUE(seas.number_of_parameters() > base_s);

    // Removing the intercept drops exactly one parameter.
    ARIMAX ni(ts);
    ni.set_include_intercept(false);
    int without = ni.number_of_parameters();
    ni.set_include_intercept(true);
    CHECK_EQ(ni.number_of_parameters(), without + 1);
}

void test_arimax_parameter_names_bounds() {
    TimeSeries ts = make_sample_series();

    ARIMAX m(ts);
    m.set_ar_order_p(2);
    m.set_ma_order_q(1);
    m.set_trend_type(ARIMAX::Trend::None);
    // Layout: Intercept, AR1, AR2, MA1, Scale.
    CHECK_TRUE(m.parameters()[0].name().find("Intercept") != std::string::npos);
    CHECK_TRUE(m.parameters()[1].name().find("AR") != std::string::npos);
    CHECK_TRUE(m.parameters()[2].name().find("AR") != std::string::npos);
    CHECK_TRUE(m.parameters()[3].name().find("MA") != std::string::npos);
    CHECK_TRUE(m.parameters().back().name().find("Scale") != std::string::npos);

    // AR coefficient bounds [-2, 2] (index 1, after intercept).
    CHECK_NEAR(m.parameters()[1].lower_bound(), -2.0, 0.0);
    CHECK_NEAR(m.parameters()[1].upper_bound(), 2.0, 0.0);

    // Scale is positive with a strictly positive lower bound (last parameter).
    const auto& scale = m.parameters().back();
    CHECK_TRUE(scale.is_positive());
    CHECK_TRUE(scale.lower_bound() > 0.0);

    // Trend / seasonality parameter names.
    ARIMAX tr(ts);
    tr.set_trend_type(ARIMAX::Trend::Linear);
    CHECK_TRUE(tr.parameters()[1].name().find("Trend") != std::string::npos);

    TimeSeries monthly = make_monthly_series();
    ARIMAX seas(monthly);
    seas.set_include_seasonality(true);
    bool has_seasonality = false;
    for (const auto& p : seas.parameters())
        if (p.name().find("Seasonality") != std::string::npos) has_seasonality = true;
    CHECK_TRUE(has_seasonality);
}

// ============================ SetParameterValues ============================

void test_arimax_set_parameter_values() {
    TimeSeries ts = make_sample_series();
    ARIMAX m(ts);
    m.set_ar_order_p(1);
    m.set_ma_order_q(0);
    m.set_trend_type(ARIMAX::Trend::None);

    std::vector<double> nv = {1000.0, 0.5, 50.0};
    m.set_parameter_values(nv);
    CHECK_NEAR(m.parameters()[0].value(), 1000.0, 0.0);
    CHECK_NEAR(m.parameters()[1].value(), 0.5, 0.0);
    CHECK_NEAR(m.parameters()[2].value(), 50.0, 0.0);

    CHECK_THROWS(m.set_parameter_values(std::vector<double>{1.0}));
}

// ============================ LogLikelihood ============================

void test_arimax_log_likelihood() {
    TimeSeries ts = make_sample_series();

    ARIMAX m(ts);
    auto p = m.parameter_values();
    double d = m.data_log_likelihood(p);
    CHECK_TRUE(!std::isnan(d));
    CHECK_TRUE(!(std::isinf(d) && d > 0.0));  // not +inf

    // With a linear trend, still finite (not NaN).
    ARIMAX tr(ts);
    tr.set_trend_type(ARIMAX::Trend::Linear);
    auto ptr = tr.parameter_values();
    CHECK_TRUE(!std::isnan(tr.data_log_likelihood(ptr)));

    // ARMA(2,1) finite.
    ARIMAX a21(ts);
    a21.set_ar_order_p(2);
    a21.set_ma_order_q(1);
    auto p21 = a21.parameter_values();
    CHECK_TRUE(!std::isnan(a21.data_log_likelihood(p21)));

    // Null time series -> DataLogLikelihood is 0.0 (ARIMAX sentinel; C# governs, ARIMAXTests.cs:459).
    ARIMAX nullm;
    auto pnull = nullm.parameter_values();
    double rn = nullm.data_log_likelihood(pnull);
    CHECK_NEAR(rn, 0.0, 0.0);

    // Prior finite.
    double pr = m.prior_log_likelihood(p);
    CHECK_TRUE(!std::isnan(pr));
    CHECK_TRUE(!(std::isinf(pr) && pr > 0.0));

    // Pointwise: positive count not exceeding the series length.
    ARIMAX ap(ts);
    ap.set_ar_order_p(2);
    auto pap = ap.parameter_values();
    auto pw = ap.pointwise_data_log_likelihood(pap);
    CHECK_TRUE(static_cast<int>(pw.size()) > 0);
    CHECK_TRUE(static_cast<int>(pw.size()) <= ts.count());

    // Sum of pointwise == total data log-likelihood.
    auto pws = m.pointwise_data_log_likelihood(p);
    double sum = 0.0;
    for (double v : pws) sum += v;
    CHECK_NEAR(sum, m.data_log_likelihood(p), 1e-6);

    // Components all Exact.
    auto comps = m.pointwise_data_log_likelihood_components(p);
    for (const auto& c : comps)
        CHECK_TRUE(c.type() == bestfit::models::DataComponentType::Exact);
}

// ============================ Predict ============================

void test_arimax_predict() {
    TimeSeries ts = make_sample_series();

    ARIMAX m(ts);
    auto p = m.parameter_values();

    auto r0 = m.predict_components(p);
    CHECK_TRUE(r0.y.size() > 0);
    CHECK_TRUE(static_cast<int>(r0.y.size()) >= m.training_time_steps());

    // Same-seed reproducible.
    auto s1 = m.predict_components(p, 0, 12345).y;
    auto s2 = m.predict_components(p, 0, 12345).y;
    for (std::size_t i = 0; i < s1.size(); ++i) CHECK_NEAR(s1[i], s2[i], 1e-10);

    // All tuple components present with equal length.
    ARIMAX m11(ts);
    m11.set_ar_order_p(1);
    m11.set_ma_order_q(1);
    auto p11 = m11.parameter_values();
    auto rc = m11.predict_components(p11);
    std::size_t n = rc.y.size();
    CHECK_EQ(rc.intercept_part.size(), n);
    CHECK_EQ(rc.trend_part.size(), n);
    CHECK_EQ(rc.seasonality_part.size(), n);
    CHECK_EQ(rc.covariate_part.size(), n);
    CHECK_EQ(rc.ar_part.size(), n);
    CHECK_EQ(rc.ma_part.size(), n);
}

// ============================ GenerateRandomValues ============================

void test_arimax_generate_random_values() {
    TimeSeries ts = make_sample_series();
    ARIMAX m(ts);

    CHECK_EQ(static_cast<int>(m.generate_random_values(100, 12345).size()), 100);

    auto g1 = m.generate_random_values(50, 12345);
    auto g2 = m.generate_random_values(50, 12345);
    for (std::size_t i = 0; i < g1.size(); ++i) CHECK_NEAR(g1[i], g2[i], 1e-10);

    // Different seeds diverge somewhere.
    auto d1 = m.generate_random_values(50, 12345);
    auto d2 = m.generate_random_values(50, 54321);
    bool any_diff = false;
    for (std::size_t i = 0; i < d1.size(); ++i)
        if (std::fabs(d1[i] - d2[i]) > 1e-10) any_diff = true;
    CHECK_TRUE(any_diff);

    CHECK_THROWS(m.generate_random_values(0, 12345));
}

// ============================ Validation ============================

void test_arimax_validate() {
    TimeSeries ts = make_sample_series();

    ARIMAX valid(ts);
    CHECK_TRUE(valid.validate().is_valid);

    ARIMAX nullm;
    auto rn = nullm.validate();
    CHECK_TRUE(!rn.is_valid);
    CHECK_TRUE(messages_contain(rn, "Time series"));

    TimeSeries tiny = make_tiny_series();
    auto rs = ARIMAX(tiny).validate();
    CHECK_TRUE(!rs.is_valid);
    CHECK_TRUE(messages_contain(rs, "10 observations"));

    // 15-obs short series: default training steps (30) exceed the series length -> invalid.
    TimeSeries shortts = make_short_series();
    auto rsh = ARIMAX(shortts).validate();
    CHECK_TRUE(!rsh.is_valid);

    // Bad AR order.
    ARIMAX bp(ts);
    bp.set_ar_order_p(-1);
    CHECK_TRUE(!bp.validate().is_valid);

    // Non-stationary AR parameters produce a warning mentioning "stationarity".
    ARIMAX ns(ts);
    ns.set_ar_order_p(1);
    int idx = ns.include_intercept() ? 1 : 0;
    ns.parameters()[idx].set_value(1.5);
    CHECK_TRUE(messages_contain(ns.validate(), "stationarity"));
}

// ============================ Trend / seasonality / differencing / transform ============================

void test_arimax_trend_seasonality_diff_transform() {
    TimeSeries ts = make_sample_series();

    // Streamflow with AR(1) + linear trend: valid + DataLL not -inf.
    ARIMAX sp(ts);
    sp.set_ar_order_p(1);
    sp.set_trend_type(ARIMAX::Trend::Linear);
    CHECK_TRUE(sp.validate().is_valid);
    auto psp = sp.parameter_values();
    double dsp = sp.data_log_likelihood(psp);
    CHECK_TRUE(!(std::isinf(dsp) && dsp < 0.0));

    // Monthly with AR(1) + seasonality: valid + DataLL finite.
    TimeSeries monthly = make_monthly_series();
    ARIMAX ms(monthly);
    ms.set_ar_order_p(1);
    ms.set_include_seasonality(true);
    CHECK_TRUE(ms.validate().is_valid);
    auto pms = ms.parameter_values();
    CHECK_TRUE(!std::isnan(ms.data_log_likelihood(pms)));

    // ARIMA(1,1,1) valid.
    ARIMAX a111(ts);
    a111.set_ar_order_p(1);
    a111.set_diff_order_d(1);
    a111.set_ma_order_q(1);
    CHECK_TRUE(a111.validate().is_valid);

    // First / second differencing: DataLL not NaN.
    ARIMAX d1(ts);
    d1.set_diff_order_d(1);
    auto pd1 = d1.parameter_values();
    CHECK_TRUE(!std::isnan(d1.data_log_likelihood(pd1)));

    ARIMAX d2(ts);
    d2.set_diff_order_d(2);
    auto pd2 = d2.parameter_values();
    CHECK_TRUE(!std::isnan(d2.data_log_likelihood(pd2)));

    // Log transform: DataLL not NaN.
    ARIMAX lt(ts);
    lt.set_transform_type(Transform::Logarithmic);
    auto plt = lt.parameter_values();
    CHECK_TRUE(!std::isnan(lt.data_log_likelihood(plt)));

    // No transform: DataLL not NaN.
    ARIMAX nt(ts);
    nt.set_transform_type(Transform::None);
    auto pnt = nt.parameter_values();
    CHECK_TRUE(!std::isnan(nt.data_log_likelihood(pnt)));

    // High orders can be set.
    ARIMAX ho(ts);
    ho.set_ar_order_p(5);
    ho.set_ma_order_q(3);
    CHECK_EQ(ho.ar_order_p(), 5);
    CHECK_EQ(ho.ma_order_q(), 3);
}

// ============================ Covariates ============================

void test_arimax_covariates() {
    TimeSeries ts = make_sample_series();

    ARIMAX m(ts);
    m.set_ar_order_p(1);
    m.set_ma_order_q(0);
    m.set_x_order_b(0);
    m.set_use_default_training_steps(false);
    m.set_training_time_steps(ts.count());
    m.set_default_parameters();
    int base = m.number_of_parameters();

    TimeSeries cov = make_covariate_series(ts);
    m.set_covariates(std::vector<TimeSeries>{cov});
    CHECK_TRUE(m.number_of_parameters() > base);

    // DataLogLikelihood remains finite after attaching a covariate.
    auto p = m.parameter_values();
    double d = m.data_log_likelihood(p);
    CHECK_TRUE(!std::isnan(d));
    CHECK_TRUE(!(std::isinf(d) && d > 0.0));
}

}  // namespace

int main() {
    test_arimax_constructors();
    test_arimax_properties();
    test_arimax_number_of_parameters();
    test_arimax_parameter_names_bounds();
    test_arimax_set_parameter_values();
    test_arimax_log_likelihood();
    test_arimax_predict();
    test_arimax_generate_random_values();
    test_arimax_validate();
    test_arimax_trend_seasonality_diff_transform();
    test_arimax_covariates();

    return bftest::summary("arimax");
}
