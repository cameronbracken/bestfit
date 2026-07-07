// Standalone C++-only ctest for the X1 parameter-name surface.
//
// Oracle for behavior is the C# source itself: the column-0 display names of each
// distribution's `ParametersToString` (-> `ParameterNames`) and its
// `ParameterNamesShortForm`, in
// upstream/Numerics/Numerics/Distributions/Univariate/*.cs @ a2c4dbf. Per the validation
// model, these internal name lists get a C++-only ctest transcribing the upstream literals
// (public-API oracle values stay in fixtures/). The end-to-end consequence -- that a built
// Normal UnivariateDistributionModel now carries two distinctly-named ModelParameter owner
// names and produces two distinct "Parameter Prior: <name>" components (no name-keyed
// collapse) -- is asserted here too; the seeded PriorInfluenceDiagnostics numbers land in
// fixtures/analyses/diagnostics_smoke.json.
//
// Non-ASCII glyphs are written with \x escapes (the codebase convention, e.g.
// constant_trend.hpp) so the bytes match the C# UTF-8 exactly -- note µ (MICRO SIGN,
// \xC2\xB5) in Normal vs μ (GREEK SMALL MU, \xCE\xBC) in VonMises/NoncentralT.
#include <string>
#include <vector>

#include "bestfit/models/support/model_parameter.hpp"
#include "bestfit/models/support/prior_component.hpp"
#include "bestfit/models/univariate_distribution/univariate_distribution_model.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "bestfit/numerics/distributions/generalized_extreme_value.hpp"
#include "bestfit/numerics/distributions/gumbel.hpp"
#include "bestfit/numerics/distributions/log_pearson_type_iii.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "bestfit/numerics/distributions/poisson.hpp"
#include "bestfit/numerics/distributions/von_mises.hpp"
#include "bestfit/numerics/distributions/weibull.hpp"
#include "check.hpp"

using bestfit::models::PriorComponent;
using bestfit::models::UnivariateDistributionModel;
using bestfit::numerics::distributions::GeneralizedExtremeValue;
using bestfit::numerics::distributions::Gumbel;
using bestfit::numerics::distributions::LogPearsonTypeIII;
using bestfit::numerics::distributions::Normal;
using bestfit::numerics::distributions::Poisson;
using bestfit::numerics::distributions::UnivariateDistributionType;
using bestfit::numerics::distributions::VonMises;
using bestfit::numerics::distributions::Weibull;

namespace {

using SV = std::vector<std::string>;

void check_names(const SV& actual, const SV& expected) {
    CHECK_EQ(actual.size(), expected.size());
    if (actual.size() != expected.size()) return;
    for (std::size_t i = 0; i < actual.size(); ++i) CHECK_EQ(actual[i], expected[i]);
}

// --- Distribution-level name oracles (transcribed from the C# ParametersToString col0 /
// ParameterNamesShortForm) --------------------------------------------------------------
void test_normal_names() {
    Normal d;
    check_names(d.parameter_names(), SV{"Mean (\xC2\xB5)", "Std Dev (\xCF\x83)"});
    check_names(d.parameter_names_short_form(), SV{"\xC2\xB5", "\xCF\x83"});
}

void test_gev_names() {
    GeneralizedExtremeValue d;
    check_names(d.parameter_names(),
                SV{"Location (\xCE\xBE)", "Scale (\xCE\xB1)", "Shape (\xCE\xBA)"});
    check_names(d.parameter_names_short_form(), SV{"\xCE\xBE", "\xCE\xB1", "\xCE\xBA"});
}

void test_lp3_names() {
    LogPearsonTypeIII d;
    check_names(d.parameter_names(), SV{"Mean (of log) (\xC2\xB5)", "Std Dev (of log) (\xCF\x83)",
                                        "Skew (of log) (\xCE\xB3)"});
    check_names(d.parameter_names_short_form(), SV{"\xC2\xB5", "\xCF\x83", "\xCE\xB3"});
}

void test_gumbel_names() {
    Gumbel d;
    check_names(d.parameter_names(), SV{"Location (\xCE\xBE)", "Scale (\xCE\xB1)"});
    check_names(d.parameter_names_short_form(), SV{"\xCE\xBE", "\xCE\xB1"});
}

void test_weibull_names() {
    Weibull d;
    check_names(d.parameter_names(), SV{"Scale (\xCE\xBB)", "Shape (\xCE\xBA)"});
    check_names(d.parameter_names_short_form(), SV{"\xCE\xBB", "\xCE\xBA"});
}

void test_poisson_names() {
    Poisson d;
    check_names(d.parameter_names(), SV{"Rate (\xCE\xBB)"});
    check_names(d.parameter_names_short_form(), SV{"\xCE\xBB"});
}

// The MICRO-SIGN vs GREEK-MU distinction: VonMises uses the greek mu \xCE\xBC, unlike Normal.
void test_von_mises_uses_greek_mu() {
    VonMises d;
    check_names(d.parameter_names(),
                SV{"Mean Direction (\xCE\xBC)", "Concentration (\xCE\xBA)"});
    check_names(d.parameter_names_short_form(), SV{"\xCE\xBC", "\xCE\xBA"});
}

// --- Model wiring: two distinctly-named ModelParameter owner names + two distinct
// "Parameter Prior: <name>" components (no name-keyed collapse) --------------------------
void test_normal_model_distinct_owner_names_and_prior_components() {
    std::vector<double> data = {12500, 15300, 8900,  22100, 18700, 14200, 9800,
                                28500, 17400, 11600, 19200, 13800, 25600, 10500,
                                16900, 21300, 14700, 8200,  23800, 15900};
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, data);

    // Two ModelParameters, each carrying a distinct, non-empty owner name from ParameterNames.
    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(2));
    if (model.parameters().size() == 2) {
        CHECK_EQ(model.parameters()[0].owner_name(), std::string("Mean (\xC2\xB5)"));
        CHECK_EQ(model.parameters()[1].owner_name(), std::string("Std Dev (\xCF\x83)"));
        CHECK_TRUE(model.parameters()[0].owner_name() != model.parameters()[1].owner_name());
    }

    // The prior components must not collapse: two distinct "Parameter Prior: <name>" entries.
    std::vector<double> p;
    for (const auto& param : model.parameters()) p.push_back(param.value());
    std::vector<PriorComponent> comps = model.pointwise_prior_log_likelihood(p);

    int param_prior_count = 0;
    bool saw_mean = false;
    bool saw_stddev = false;
    for (const auto& c : comps) {
        if (c.name().rfind("Parameter Prior: ", 0) == 0) {
            ++param_prior_count;
            if (c.name() == std::string("Parameter Prior: Mean (\xC2\xB5)")) saw_mean = true;
            if (c.name() == std::string("Parameter Prior: Std Dev (\xCF\x83)")) saw_stddev = true;
        }
    }
    CHECK_EQ(param_prior_count, 2);  // was 1 (collapsed) before X1
    CHECK_TRUE(saw_mean);
    CHECK_TRUE(saw_stddev);
}

}  // namespace

int main() {
    test_normal_names();
    test_gev_names();
    test_lp3_names();
    test_gumbel_names();
    test_weibull_names();
    test_poisson_names();
    test_von_mises_uses_greek_mu();
    test_normal_model_distinct_owner_names_and_prior_components();
    return bftest::summary("parameter_names");
}
