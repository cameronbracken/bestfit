// Generic, fixture-driven validation for the C++ core.
//
// Reads the language-neutral oracle fixtures (the single source of truth shared with
// the R and Python packages) and checks every assertion. No oracle values live here --
// only the dispatch from fixture method names to the core API. The fixtures directory is
// passed as argv[1] (CMake points it at the repo's canonical fixtures/).
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "bestfit/numerics/distributions/generalized_extreme_value.hpp"
#include "check.hpp"
#include "third_party/json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;
using bestfit::numerics::distributions::EstimationMethod;
using bestfit::numerics::distributions::GeneralizedExtremeValue;

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

static EstimationMethod parse_method(const std::string& m) {
    if (m == "mom") return EstimationMethod::MethodOfMoments;
    if (m == "lmom") return EstimationMethod::MethodOfLinearMoments;
    return EstimationMethod::MaximumLikelihood;
}

static GeneralizedExtremeValue build(const json& construct, const json& datasets) {
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

static double dispatch(const GeneralizedExtremeValue& g, const std::string& m, const json& a) {
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
    throw std::runtime_error("unknown fixture method: " + m);
}

static void check(const GeneralizedExtremeValue& g, const json& as, const std::string& label) {
    std::string method = as["method"].get<std::string>();
    json args = as.contains("args") ? as["args"] : json::array();
    std::string mode = as["mode"].get<std::string>();
    std::string where = label + "/" + method;

    if (mode == "bool") {
        bool actual = (method == "parameters_valid") ? g.parameters_valid() : false;
        if (actual == as["expected"].get<bool>())
            bftest::report_pass();
        else
            bftest::report_fail(__FILE__, __LINE__, where + ": bool mismatch");
        return;
    }
    double actual = dispatch(g, method, args);
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
        if (spec.value("kind", "") != "univariate_distribution") continue;
        if (spec.value("target", "") != "GeneralizedExtremeValue") continue;  // only GEV so far
        json datasets = spec.value("datasets", json::object());
        for (const auto& c : spec["cases"]) {
            GeneralizedExtremeValue g = build(c["construct"], datasets);
            std::string name = c["name"].get<std::string>();
            for (const auto& a : c["assertions"]) check(g, a, name);
        }
    }
    if (files == 0) {
        std::fprintf(stderr, "no fixtures found under %s\n", argv[1]);
        return 2;
    }
    return bftest::summary("fixtures");
}
