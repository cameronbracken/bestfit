// Standalone tests for corehydro::numerics::sampling::StratificationOptions and Stratify::XValues.
//
// Oracle values are taken directly from the C# unit tests in
// upstream/Numerics/Test_Numerics/Sampling/Test_Stratification.cs
// (Test_XValues, Test_XValues_Multi, Test_XValues_Log10) and verified against the real
// Numerics library via tools/verify_oracles.py.
#include <cmath>
#include <vector>

#include "corehydro/numerics/sampling/stratification_options.hpp"
#include "corehydro/numerics/sampling/stratify.hpp"
#include "check.hpp"

using corehydro::numerics::sampling::Stratify;
using corehydro::numerics::sampling::StratificationOptions;

namespace {

void test_xvalues() {
    StratificationOptions options(100.0, 200.0, 100);
    auto bins = Stratify::XValues(options);
    std::vector<double> true_midpoint = {
        100.5,  101.5,  102.5,  103.5,  104.5,  105.5,  106.5,  107.5,  108.5,  109.5,
        110.5,  111.5,  112.5,  113.5,  114.5,  115.5,  116.5,  117.5,  118.5,  119.5,
        120.5,  121.5,  122.5,  123.5,  124.5,  125.5,  126.5,  127.5,  128.5,  129.5,
        130.5,  131.5,  132.5,  133.5,  134.5,  135.5,  136.5,  137.5,  138.5,  139.5,
        140.5,  141.5,  142.5,  143.5,  144.5,  145.5,  146.5,  147.5,  148.5,  149.5,
        150.5,  151.5,  152.5,  153.5,  154.5,  155.5,  156.5,  157.5,  158.5,  159.5,
        160.5,  161.5,  162.5,  163.5,  164.5,  165.5,  166.5,  167.5,  168.5,  169.5,
        170.5,  171.5,  172.5,  173.5,  174.5,  175.5,  176.5,  177.5,  178.5,  179.5,
        180.5,  181.5,  182.5,  183.5,  184.5,  185.5,  186.5,  187.5,  188.5,  189.5,
        190.5,  191.5,  192.5,  193.5,  194.5,  195.5,  196.5,  197.5,  198.5,  199.5,
    };
    CHECK_EQ(bins.size(), true_midpoint.size());
    for (std::size_t i = 0; i < bins.size(); ++i) CHECK_NEAR(bins[i].midpoint(), true_midpoint[i], 1e-6);
}

void test_xvalues_multi() {
    std::vector<StratificationOptions> options;
    options.emplace_back(100.0, 125.0, 10);
    options.emplace_back(125.0, 130.0, 20);
    options.emplace_back(130.0, 200.0, 70);
    auto bins = Stratify::XValues(options);
    std::vector<double> true_midpoint = {
        101.25,  103.75,  106.25,  108.75,  111.25,  113.75,  116.25,  118.75,  121.25,  123.75,
        125.125, 125.375, 125.625, 125.875, 126.125, 126.375, 126.625, 126.875, 127.125, 127.375,
        127.625, 127.875, 128.125, 128.375, 128.625, 128.875, 129.125, 129.375, 129.625, 129.875,
        130.5,   131.5,   132.5,   133.5,   134.5,   135.5,   136.5,   137.5,   138.5,   139.5,
        140.5,   141.5,   142.5,   143.5,   144.5,   145.5,   146.5,   147.5,   148.5,   149.5,
        150.5,   151.5,   152.5,   153.5,   154.5,   155.5,   156.5,   157.5,   158.5,   159.5,
        160.5,   161.5,   162.5,   163.5,   164.5,   165.5,   166.5,   167.5,   168.5,   169.5,
        170.5,   171.5,   172.5,   173.5,   174.5,   175.5,   176.5,   177.5,   178.5,   179.5,
        180.5,   181.5,   182.5,   183.5,   184.5,   185.5,   186.5,   187.5,   188.5,   189.5,
        190.5,   191.5,   192.5,   193.5,   194.5,   195.5,   196.5,   197.5,   198.5,   199.5,
    };
    CHECK_EQ(bins.size(), true_midpoint.size());
    for (std::size_t i = 0; i < bins.size(); ++i) CHECK_NEAR(bins[i].midpoint(), true_midpoint[i], 1e-6);
}

void test_xvalues_log10() {
    StratificationOptions options(100.0, 200.0, 100);
    auto bins = Stratify::XValues(options, true);
    std::vector<double> true_midpoint = {
        100.347777502836, 101.045751492337, 101.748580274861, 102.456297618163, 103.168937524872,
        103.886534234125, 104.60912222321,  105.336736209223, 106.069411150737, 106.807182249483,
        107.550084952036, 108.298154951525, 109.05142818934,  109.809940856868, 110.573729397223,
        111.342830507004, 112.117281138052, 112.897118499231, 113.682380058212, 114.473103543273,
        115.269326945117, 116.071088518688, 116.878426785017, 117.691380533071, 118.509988821613,
        119.334290981083, 120.164326615485, 121.000135604291, 121.841758104357, 122.68923455185,
        123.542605664196, 124.401912442031, 125.267196171173, 126.138498424606, 127.015861064478,
        127.899326244109, 128.788936410021, 129.684734303973, 130.586762965016, 131.495065731564,
        132.409686243472, 133.330668444133, 134.258056582592, 135.191895215669, 136.132229210105,
        137.079103744708, 138.032564312535, 138.992656723068, 139.959427104422, 140.932921905556,
        141.91318789851,  142.900272180647, 143.894222176918, 144.895085642142, 145.902910663299,
        146.917745661838, 147.93963939601,  148.968640963201, 150.004799802302, 151.048165696075,
        152.09878877355,  153.156719512431, 154.222008741522, 155.294707643172, 156.374867755728,
        157.462540976015, 158.557779561832, 159.660636134457, 160.771163681178, 161.889415557839,
        163.015445491405, 164.149307582538, 165.291056308204, 166.440746524284, 167.598433468212,
        168.764172761627, 169.938020413052, 171.120032820574, 172.310266774565, 173.508779460402,
        174.715628461219, 175.930871760673, 177.154567745726, 178.386775209458, 179.627553353884,
        180.876961792803, 182.13506055466,  183.401910085432, 184.677571251529, 185.962105342722,
        187.255574075086, 188.558039593965, 189.869564476958, 191.190211736925, 192.520044825016,
        193.859127633718, 195.207524499926, 196.565300208033, 197.932519993044, 199.309249543709,
    };
    CHECK_EQ(bins.size(), true_midpoint.size());
    for (std::size_t i = 0; i < bins.size(); ++i) CHECK_NEAR(bins[i].midpoint(), true_midpoint[i], 1e-6);
}

// Extra invariant: invalid options report IsValid == false and XValues returns empty,
// covering each of the three validation rules in turn.
void test_invalid_options() {
    StratificationOptions lower_ge_upper(200.0, 100.0, 100);
    CHECK_TRUE(!lower_ge_upper.is_valid());
    CHECK_TRUE(Stratify::XValues(lower_ge_upper).empty());

    StratificationOptions too_few_bins(100.0, 200.0, 1);
    CHECK_TRUE(!too_few_bins.is_valid());
    CHECK_TRUE(Stratify::XValues(too_few_bins).empty());

    StratificationOptions negative_prob(-0.1, 0.5, 10, true);
    CHECK_TRUE(!negative_prob.is_valid());
    CHECK_TRUE(Stratify::XValues(negative_prob).empty());

    StratificationOptions prob_over_one(0.0, 1.5, 10, true);
    CHECK_TRUE(!prob_over_one.is_valid());
    CHECK_TRUE(Stratify::XValues(prob_over_one).empty());
}

// Extra invariant: a valid probability options (isProbability == true) still makes
// XValues return empty per the `options.IsProbability` guard in Stratify::XValues.
void test_probability_options_guard() {
    StratificationOptions prob_options(0.0, 1.0, 100, true);
    CHECK_TRUE(prob_options.is_valid());
    CHECK_TRUE(Stratify::XValues(prob_options).empty());
}

}  // namespace

int main() {
    test_xvalues();
    test_xvalues_multi();
    test_xvalues_log10();
    test_invalid_options();
    test_probability_options_guard();

    return chtest::summary("stratification");
}
