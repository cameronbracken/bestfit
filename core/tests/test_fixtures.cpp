// Generic, fixture-driven validation for the C++ core.
//
// Reads the language-neutral oracle fixtures (the single source of truth shared with
// the R and Python packages) and checks every assertion. No oracle values live here --
// only the dispatch from fixture method names to the core API. The fixtures directory is
// passed as argv[1] (CMake points it at the repo's canonical fixtures/).
//
// Two code paths: the GEV slice keeps its bespoke dispatch (location/scale/shape names,
// standard-error methods); every other distribution goes through the polymorphic
// UnivariateDistributionBase + factory path, which is what new distributions plug into.
// Special functions use a flat target->lambda map.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "bestfit/numerics/data/goodness_of_fit.hpp"
#include "bestfit/numerics/distributions/base/i_estimation.hpp"
#include "bestfit/numerics/distributions/base/i_linear_moment_estimation.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "bestfit/numerics/distributions/empirical_distribution.hpp"
#include "bestfit/numerics/distributions/generalized_extreme_value.hpp"
#include "bestfit/numerics/distributions/kernel_density.hpp"
#include "bestfit/numerics/distributions/competing_risks.hpp"
#include "bestfit/numerics/distributions/mixture.hpp"
#include "bestfit/numerics/distributions/truncated_distribution.hpp"
#include "bestfit/numerics/math/special/beta.hpp"
#include "bestfit/numerics/math/special/bessel.hpp"
#include "bestfit/numerics/math/special/erf.hpp"
#include "bestfit/numerics/math/special/factorial.hpp"
#include "bestfit/numerics/math/special/gamma.hpp"
#include "check.hpp"
#include "third_party/json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;
namespace dist = bestfit::numerics::distributions;
using dist::EstimationMethod;
using dist::GeneralizedExtremeValue;

static double parse_num(const json& v) {
    if (v.is_string()) {
        std::string s = v.get<std::string>();
        if (s == "nan") return std::numeric_limits<double>::quiet_NaN();
        if (s == "inf") return std::numeric_limits<double>::infinity();
        if (s == "-inf") return -std::numeric_limits<double>::infinity();
        throw std::runtime_error("unexpected string value: " + s);
    }
    return v.get<double>();
}

// --- Shared assertion checking ---------------------------------------------------------

static void check_value(double actual, const json& as, const std::string& where) {
    std::string mode = as["mode"].get<std::string>();
    bool ok;
    if (mode == "equal") {
        double e = parse_num(as["expected"]);
        ok = std::isnan(e) ? std::isnan(actual) : (actual == e);
    } else if (mode == "abs") {
        ok = std::fabs(actual - as["expected"].get<double>()) <= as["tol"].get<double>();
    } else if (mode == "rel") {
        double e = as["expected"].get<double>();
        ok = std::fabs(actual - e) / std::fabs(e) <= as["tol"].get<double>();
    } else {
        throw std::runtime_error("unknown comparison mode: " + mode);
    }
    if (ok)
        bftest::report_pass();
    else
        bftest::report_fail(__FILE__, __LINE__, where + ": value mismatch");
}

static void check_bool(bool actual, const json& as, const std::string& where) {
    if (actual == as["expected"].get<bool>())
        bftest::report_pass();
    else
        bftest::report_fail(__FILE__, __LINE__, where + ": bool mismatch");
}

// --- GEV slice (bespoke) ---------------------------------------------------------------

static EstimationMethod parse_method(const std::string& m) {
    if (m == "mom") return EstimationMethod::MethodOfMoments;
    if (m == "lmom") return EstimationMethod::MethodOfLinearMoments;
    return EstimationMethod::MaximumLikelihood;
}

static GeneralizedExtremeValue build_gev(const json& construct, const json& datasets) {
    if (construct.contains("params")) {
        auto p = construct["params"];
        return GeneralizedExtremeValue(parse_num(p[0]), parse_num(p[1]), parse_num(p[2]));
    }
    const auto& fit = construct["fit"];
    std::vector<double> data;
    for (const auto& v : datasets[fit["dataset"].get<std::string>()]) data.push_back(v.get<double>());
    GeneralizedExtremeValue g;
    g.estimate(data, parse_method(fit["method"].get<std::string>()));
    return g;
}

static double dispatch_gev(const GeneralizedExtremeValue& g, const std::string& m, const json& a) {
    if (m == "mean") return g.mean();
    if (m == "median") return g.median();
    if (m == "mode") return g.mode();
    if (m == "sd") return g.standard_deviation();
    if (m == "skewness") return g.skewness();
    if (m == "kurtosis") return g.kurtosis();
    if (m == "minimum") return g.minimum();
    if (m == "maximum") return g.maximum();
    if (m == "pdf") return g.pdf(a[0].get<double>());
    if (m == "cdf") return g.cdf(a[0].get<double>());
    if (m == "quantile") return g.inverse_cdf(a[0].get<double>());
    if (m == "param") {
        std::string n = a[0].get<std::string>();
        return n == "location" ? g.xi() : n == "scale" ? g.alpha() : g.kappa();
    }
    if (m == "linear_moment")
        return g.linear_moments_from_parameters({g.xi(), g.alpha(), g.kappa()})[a[0].get<int>()];
    if (m == "quantile_gradient") return g.quantile_gradient(a[0].get<double>())[a[1].get<int>()];
    if (m == "parameter_covariance")
        return g.parameter_covariance(a[0].get<int>())[a[1].get<int>()][a[2].get<int>()];
    if (m == "quantile_variance") return g.quantile_variance(a[0].get<double>(), a[1].get<int>());
    if (m == "quantile_se")
        return std::sqrt(g.quantile_variance(a[0].get<double>(), a[1].get<int>()));
    throw std::runtime_error("unknown GEV fixture method: " + m);
}

static void run_gev(const json& spec) {
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        GeneralizedExtremeValue g = build_gev(c["construct"], datasets);
        std::string name = c["name"].get<std::string>();
        for (const auto& as : c["assertions"]) {
            std::string method = as["method"].get<std::string>();
            json args = as.contains("args") ? as["args"] : json::array();
            std::string where = name + "/" + method;
            if (as["mode"].get<std::string>() == "bool")
                check_bool(g.parameters_valid(), as, where);
            else
                check_value(dispatch_gev(g, method, args), as, where);
        }
    }
}

// --- Special-function path ------------------------------------------------------------

namespace sf = bestfit::numerics::math::special;

// Dispatch table: maps "Module.method" → a free function of (vector<double>) → double.
static const std::map<std::string, std::function<double(const std::vector<double>&)>>&
special_function_table() {
    static const std::map<std::string, std::function<double(const std::vector<double>&)>> t = {
        // Erf family
        {"Erf.function",      [](const std::vector<double>& a) { return sf::erf::function(a[0]); }},
        {"Erf.erfc",          [](const std::vector<double>& a) { return sf::erf::erfc(a[0]); }},
        {"Erf.inverse_erf",   [](const std::vector<double>& a) { return sf::erf::inverse_erf(a[0]); }},
        {"Erf.inverse_erfc",  [](const std::vector<double>& a) { return sf::erf::inverse_erfc(a[0]); }},
        // Gamma family
        {"Gamma.function",               [](const std::vector<double>& a) { return sf::function(a[0]); }},
        {"Gamma.log_gamma",              [](const std::vector<double>& a) { return sf::log_gamma(a[0]); }},
        {"Gamma.digamma",                [](const std::vector<double>& a) { return sf::digamma(a[0]); }},
        {"Gamma.trigamma",               [](const std::vector<double>& a) { return sf::trigamma(a[0]); }},
        {"Gamma.lower_incomplete",       [](const std::vector<double>& a) { return sf::lower_incomplete(a[0], a[1]); }},
        {"Gamma.upper_incomplete",       [](const std::vector<double>& a) { return sf::upper_incomplete(a[0], a[1]); }},
        {"Gamma.inverse_lower_incomplete", [](const std::vector<double>& a) { return sf::inverse_lower_incomplete(a[0], a[1]); }},
        {"Gamma.inverse_upper_incomplete", [](const std::vector<double>& a) { return sf::inverse_upper_incomplete(a[0], a[1]); }},
        // Beta family
        {"Beta.function",           [](const std::vector<double>& a) { return sf::beta::function(a[0], a[1]); }},
        {"Beta.incomplete",         [](const std::vector<double>& a) { return sf::beta::incomplete(a[0], a[1], a[2]); }},
        {"Beta.incbcf",             [](const std::vector<double>& a) { return sf::beta::detail::incbcf(a[0], a[1], a[2]); }},
        {"Beta.incbd",              [](const std::vector<double>& a) { return sf::beta::detail::incbd(a[0], a[1], a[2]); }},
        {"Beta.power_series",       [](const std::vector<double>& a) { return sf::beta::detail::power_series(a[0], a[1], a[2]); }},
        {"Beta.incomplete_inverse", [](const std::vector<double>& a) { return sf::beta::incomplete_inverse(a[0], a[1], a[2]); }},
        // Factorial family
        {"Factorial.function",             [](const std::vector<double>& a) { return sf::factorial::function(static_cast<int>(a[0])); }},
        {"Factorial.log_factorial",        [](const std::vector<double>& a) { return sf::factorial::log_factorial(static_cast<int>(a[0])); }},
        {"Factorial.binomial_coefficient", [](const std::vector<double>& a) { return sf::factorial::binomial_coefficient(static_cast<int>(a[0]), static_cast<int>(a[1])); }},
        // Bessel family
        {"Bessel.i0", [](const std::vector<double>& a) { return sf::bessel::i0(a[0]); }},
        {"Bessel.i1", [](const std::vector<double>& a) { return sf::bessel::i1(a[0]); }},
    };
    return t;
}

static void run_special_function(const json& spec) {
    std::string target = spec["target"].get<std::string>();
    const auto& table = special_function_table();
    auto it = table.find(target);
    if (it == table.end())
        throw std::runtime_error("unknown special-function target: " + target);
    const auto& fn = it->second;
    for (const auto& c : spec["cases"]) {
        std::string name = c["name"].get<std::string>();
        std::vector<double> args;
        for (const auto& v : c["args"]) args.push_back(parse_num(v));
        double actual = fn(args);
        for (const auto& as : c["assertions"]) {
            std::string where = target + "/" + name + "/" + as["method"].get<std::string>();
            check_value(actual, as, where);
        }
    }
}

// --- Generic polymorphic path ----------------------------------------------------------

static dist::ParameterEstimationMethod parse_pe_method(const std::string& m) {
    if (m == "mom") return dist::ParameterEstimationMethod::MethodOfMoments;
    if (m == "lmom") return dist::ParameterEstimationMethod::MethodOfLinearMoments;
    return dist::ParameterEstimationMethod::MaximumLikelihood;
}

static std::unique_ptr<dist::UnivariateDistributionBase> build_generic(const std::string& target,
                                                                       const json& construct,
                                                                       const json& datasets) {
    auto d = dist::create_distribution(target);
    if (construct.contains("params")) {
        std::vector<double> p;
        for (const auto& v : construct["params"]) p.push_back(parse_num(v));
        d->set_parameters(p);
        return d;
    }
    const auto& fit = construct["fit"];
    std::vector<double> data;
    for (const auto& v : datasets[fit["dataset"].get<std::string>()]) data.push_back(v.get<double>());
    auto* est = dynamic_cast<dist::IEstimation*>(d.get());
    if (est == nullptr) throw std::runtime_error(target + " does not support estimation");
    est->estimate(data, parse_pe_method(fit["method"].get<std::string>()));
    return d;
}

// --- Composite distribution path (TruncatedDistribution, and future Empirical/Kernel/Mixture/CR) ---

// build_component: create a sub-distribution from {"target": "...", "params": [...]} (or "fit").
// Recursive -- components can nest (Mixture inside CompetingRisks, etc.).
static std::unique_ptr<dist::UnivariateDistributionBase> build_component(const json& desc,
                                                                          const json& datasets) {
    std::string target = desc["target"].get<std::string>();
    auto d = dist::create_distribution(target);
    if (desc.contains("params")) {
        std::vector<double> p;
        for (const auto& v : desc["params"]) p.push_back(parse_num(v));
        d->set_parameters(p);
    } else if (desc.contains("fit")) {
        const auto& fit = desc["fit"];
        std::vector<double> data;
        for (const auto& v : datasets[fit["dataset"].get<std::string>()]) data.push_back(v.get<double>());
        auto* est = dynamic_cast<dist::IEstimation*>(d.get());
        if (est == nullptr) throw std::runtime_error(target + " does not support estimation");
        est->estimate(data, parse_pe_method(fit["method"].get<std::string>()));
    }
    return d;
}

// build_composite: switch on composite targets; returns a UnivariateDistributionBase* so
// dispatch_generic can be reused without modification for pdf/cdf/moments/etc.
// Future composites (KernelDensity, Mixture, CompetingRisks) add a case here.
static std::unique_ptr<dist::UnivariateDistributionBase> build_composite(const std::string& target,
                                                                          const json& construct,
                                                                          const json& datasets) {
    if (target == "TruncatedDistribution") {
        auto base = build_component(construct["base"], datasets);
        const auto& bounds = construct["bounds"];
        double lo = parse_num(bounds[0]);
        double hi = parse_num(bounds[1]);
        return std::make_unique<dist::TruncatedDistribution>(std::move(base), lo, hi);
    }
    if (target == "Empirical") {
        std::vector<double> xv, pv;
        for (const auto& v : construct["x"]) xv.push_back(parse_num(v));
        for (const auto& v : construct["p"]) pv.push_back(parse_num(v));
        auto pt = dist::EmpiricalTransform::NormalZ;
        if (construct.contains("p_transform")) {
            std::string t = construct["p_transform"].get<std::string>();
            if (t == "None") pt = dist::EmpiricalTransform::None;
            else if (t == "NormalZ") pt = dist::EmpiricalTransform::NormalZ;
            else throw std::runtime_error("unknown p_transform: " + t);
        }
        return std::make_unique<dist::EmpiricalDistribution>(std::move(xv), std::move(pv), pt);
    }
    if (target == "KernelDensity") {
        const auto& ds_name = construct["data"].get<std::string>();
        std::vector<double> data;
        for (const auto& v : datasets[ds_name]) data.push_back(v.get<double>());
        std::string kernel_str = "Gaussian";
        if (construct.contains("kernel")) kernel_str = construct["kernel"].get<std::string>();
        dist::KernelType kt = dist::KernelType::Gaussian;
        if      (kernel_str == "Epanechnikov") kt = dist::KernelType::Epanechnikov;
        else if (kernel_str == "Gaussian")     kt = dist::KernelType::Gaussian;
        else if (kernel_str == "Triangular")   kt = dist::KernelType::Triangular;
        else if (kernel_str == "Uniform")      kt = dist::KernelType::Uniform;
        else throw std::runtime_error("unknown kernel type: " + kernel_str);
        std::unique_ptr<dist::KernelDensity> kde;
        if (construct.contains("bandwidth"))
            kde = std::make_unique<dist::KernelDensity>(std::move(data), kt,
                                                        construct["bandwidth"].get<double>());
        else
            kde = std::make_unique<dist::KernelDensity>(std::move(data), kt);
        if (construct.contains("bounded_by_data"))
            kde->set_bounded_by_data(construct["bounded_by_data"].get<bool>());
        return kde;
    }
    if (target == "Mixture") {
        const auto& wts_json = construct["weights"];
        const auto& comps_json = construct["components"];
        std::vector<double> wts;
        std::vector<std::unique_ptr<dist::UnivariateDistributionBase>> comps;
        for (const auto& w : wts_json) wts.push_back(w.get<double>());
        for (const auto& c : comps_json) comps.push_back(build_component(c, datasets));
        return std::make_unique<dist::Mixture>(std::move(wts), std::move(comps));
    }
    if (target == "CompetingRisks") {
        const auto& comps_json = construct["components"];
        std::vector<std::unique_ptr<dist::UnivariateDistributionBase>> comps;
        for (const auto& c : comps_json) comps.push_back(build_component(c, datasets));
        auto cr = std::make_unique<dist::CompetingRisks>(std::move(comps));
        if (construct.contains("minimum_of_random_variables"))
            cr->minimum_of_random_variables = construct["minimum_of_random_variables"].get<bool>();
        return cr;
    }
    throw std::runtime_error("unknown composite target: " + target);
}

static bool is_composite_target(const std::string& target) {
    return target == "TruncatedDistribution" || target == "Empirical"
        || target == "KernelDensity" || target == "Mixture" || target == "CompetingRisks";
}

static double dispatch_generic(const dist::UnivariateDistributionBase& d, const std::string& m,
                               const json& a) {
    if (m == "mean") return d.mean();
    if (m == "median") return d.median();
    if (m == "mode") return d.mode();
    if (m == "sd") return d.standard_deviation();
    if (m == "skewness") return d.skewness();
    if (m == "kurtosis") return d.kurtosis();
    if (m == "minimum") return d.minimum();
    if (m == "maximum") return d.maximum();
    if (m == "pdf") return d.pdf(a[0].get<double>());
    if (m == "cdf") return d.cdf(a[0].get<double>());
    if (m == "quantile") return d.inverse_cdf(a[0].get<double>());
    if (m == "param") return d.get_parameters()[a[0].get<int>()];
    if (m == "linear_moment") {
        const auto* lm = dynamic_cast<const dist::ILinearMomentEstimation*>(&d);
        if (lm == nullptr) throw std::runtime_error("distribution has no L-moments");
        return lm->linear_moments_from_parameters(d.get_parameters())[a[0].get<int>()];
    }
    throw std::runtime_error("unknown fixture method: " + m);
}

static void run_generic(const json& spec) {
    std::string target = spec["target"].get<std::string>();
    bool composite = is_composite_target(target);
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        // Composite targets use build_composite; flat-param targets use build_generic.
        // dispatch_generic works for both since TruncatedDistribution is a UnivariateDistributionBase.
        std::unique_ptr<dist::UnivariateDistributionBase> d =
            composite ? build_composite(target, c["construct"], datasets)
                      : build_generic(target, c["construct"], datasets);
        std::string name = c["name"].get<std::string>();
        for (const auto& as : c["assertions"]) {
            std::string method = as["method"].get<std::string>();
            json args = as.contains("args") ? as["args"] : json::array();
            std::string where = target + "/" + name + "/" + method;
            if (as["mode"].get<std::string>() == "bool")
                check_bool(d->parameters_valid(), as, where);
            else
                check_value(dispatch_generic(*d, method, args), as, where);
        }
    }
}

// --- goodness_of_fit path -------------------------------------------------------------

namespace gof = bestfit::numerics::data;

static double dispatch_gof(const std::string& fn, const std::vector<double>& args,
                            const std::vector<double>& obs, const std::vector<double>& mod) {
    if (fn == "AIC")  return gof::GoodnessOfFit::aic(static_cast<int>(args[0]), args[1]);
    if (fn == "AICc") return gof::GoodnessOfFit::aicc(static_cast<int>(args[0]),
                                                       static_cast<int>(args[1]), args[2]);
    if (fn == "BIC")  return gof::GoodnessOfFit::bic(static_cast<int>(args[0]),
                                                      static_cast<int>(args[1]), args[2]);
    if (fn == "MSE")  return gof::GoodnessOfFit::mse(obs, mod);
    if (fn == "MAE")  return gof::GoodnessOfFit::mae(obs, mod);
    if (fn == "NashSutcliffeEfficiency")  return gof::GoodnessOfFit::nash_sutcliffe_efficiency(obs, mod);
    if (fn == "KlingGuptaEfficiency")     return gof::GoodnessOfFit::kling_gupta_efficiency(obs, mod);
    if (fn == "KlingGuptaEfficiencyMod")  return gof::GoodnessOfFit::kling_gupta_efficiency_mod(obs, mod);
    if (fn == "PBIAS")                    return gof::GoodnessOfFit::pbias(obs, mod);
    if (fn == "RSR")                      return gof::GoodnessOfFit::rsr(obs, mod);
    if (fn == "IndexOfAgreement")         return gof::GoodnessOfFit::index_of_agreement(obs, mod);
    if (fn == "ModifiedIndexOfAgreement") return gof::GoodnessOfFit::modified_index_of_agreement(obs, mod);
    if (fn == "RefinedIndexOfAgreement")  return gof::GoodnessOfFit::refined_index_of_agreement(obs, mod);
    if (fn == "VolumetricEfficiency")     return gof::GoodnessOfFit::volumetric_efficiency(obs, mod);
    throw std::runtime_error("unknown goodness_of_fit function: " + fn);
}

static void run_goodness_of_fit(const json& spec) {
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        std::string name = c["name"].get<std::string>();
        std::string fn = c["function"].get<std::string>();
        std::vector<double> args;
        if (c.contains("args"))
            for (const auto& v : c["args"]) args.push_back(parse_num(v));
        std::vector<double> obs, mod;
        if (c.contains("observed_dataset"))
            for (const auto& v : datasets[c["observed_dataset"].get<std::string>()])
                obs.push_back(v.get<double>());
        if (c.contains("modeled_dataset"))
            for (const auto& v : datasets[c["modeled_dataset"].get<std::string>()])
                mod.push_back(v.get<double>());
        double actual = dispatch_gof(fn, args, obs, mod);
        for (const auto& as : c["assertions"]) {
            std::string where = "gof/" + name;
            check_value(actual, as, where);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <fixtures-dir>\n", argv[0]);
        return 2;
    }
    int files = 0;
    for (const auto& entry : fs::recursive_directory_iterator(argv[1])) {
        if (entry.path().extension() != ".json") continue;
        ++files;
        std::ifstream in(entry.path());
        json spec = json::parse(in);
        std::string kind = spec.value("kind", "");
        if (kind == "special_function") {
            run_special_function(spec);
        } else if (kind == "goodness_of_fit") {
            run_goodness_of_fit(spec);
        } else if (kind == "univariate_distribution") {
            if (spec.value("target", "") == "GeneralizedExtremeValue")
                run_gev(spec);
            else
                run_generic(spec);
        }
    }
    if (files == 0) {
        std::fprintf(stderr, "no fixtures found under %s\n", argv[1]);
        return 2;
    }
    return bftest::summary("fixtures");
}
