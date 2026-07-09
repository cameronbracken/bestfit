// Structural + deterministic-math tests for
// bestfit::analyses::bivariate::CoincidentFrequencyAnalysis (X6).
//
// These transcribe the STRUCTURAL C# tests from
//   RMC.BestFit.Tests/Bivariate/CoincidentFrequencyAnalysisTests.cs @ fc28c0c
// and add a HAND-DERIVED analytic oracle for the conditional-frequency law
//   F_Z(z) = sum_j [ C(u_j, v_{j+1}) - C(u_j, v_j) ],  AEP = 1 - F_Z(z)
// computed by hand under an INDEPENDENCE copula (Normal copula with rho = 0, whose CDF is
// C(u,v) = u*v). Hardcoded oracles in this C++-only ctest are correct (public-API oracle
// values live in fixtures/*.json; this internal analysis math gets a C++-only ctest per the
// Phase-10 policy -- the exact numeric AEP curve under a real seeded fit is X12's job).
//
// HAND DERIVATION (auditable):
//   Grid: XValues = [-1, 1] (M=2), YValues = [-1, 1] (N=2), response Z[i][j] = X[i] + Y[j]:
//         Z = [[-2, 0], [0, 2]].
//   Marginals: X ~ Normal(0,1), Y ~ Normal(0,1). Copula: Normal(rho=0) => C(u,v)=u*v.
//   ComputeYBinEdges([-1,1]) = [-inf, 0, +inf].
//   BuildXZetas: u_i = Phi(x_i); zeta_i = Phi^-1(u_i) = x_i  => xZetas = [-1, 1].
//   BuildVEdges: v0=0, v2=1, v1 = Phi(midpoint=0) = 0.5      => vEdges = [0, 0.5, 1].
//   BuildZOutputBins(response, 5): zMin=-2, zMax=2, step=1   => Z = [-2,-1,0,1,2].
//
//   For each Z bin z, per column j in {0,1}:
//     u_j = Phi(zeta*),  zeta* from FindUStarInColumn (interp/extrap of (response[*,j], xZetas)).
//     column 0 (j=0):     C(u,v1) - C(u,v0=0) = u*0.5 - 0        = 0.5*u   [boundary C(u,0)=0]
//     column 1 (j=N-1):   C(u,v2=1) - C(u,v1) = u - u*0.5        = 0.5*u   [boundary C(u,1)=u]
//
//   z=-2: col0 u=Phi(-1);         col1 u=Phi(-3).
//         F_Z = 0.5*(Phi(-1)+Phi(-3)) = 0.5*(1-Phi(1)+1-Phi(3)) = 1 - 0.5*(Phi(1)+Phi(3)).
//         AEP = 0.5*(Phi(1)+Phi(3)) = 0.9199974240184564.
//   z=-1: col0 within-segment => zeta*=0, u=Phi(0)=0.5 -> 0.25;  col1 down-extrap zeta*=-2,
//         u=Phi(-2) -> 0.5*Phi(-2).  F_Z = 0.25 + 0.5*Phi(-2).  AEP = 0.75 - 0.5*(1-Phi(2))
//                                                                    = 0.25 + 0.5*Phi(2)
//                                                                    = 0.7386249340259104.
//   z= 0: col0 up-extrap zeta*=1, u=Phi(1) -> 0.5*Phi(1); col1 down-extrap zeta*=-1,
//         u=Phi(-1) -> 0.5*Phi(-1).  F_Z = 0.5*(Phi(1)+Phi(-1)) = 0.5.  AEP = 0.5.
//   z= 1: symmetric to z=-1: AEP = 0.2613750659740896.
//   z= 2: symmetric to z=-2: AEP = 0.08000257598154358.
//
// SKIPPED C# test methods (WPF/serialization/threading/notification -- no numerical content):
//   - Constructor_NullXValues_Throws / Constructor_NullXElement_Throws, the XElement ctor,
//     ToXElement_FromXElement_PreservesInputs, Constructor_WithLegacyZTableXml_StillReadsValues:
//     XML (de)serialization + null-array guards are a project-wide non-port; C++ takes the
//     ordinates by value (a std::vector cannot be null), so those guards are unreachable.
//   - XValues_Set_ClearsResults / BayesianAnalysis_CredibleIntervalWidthChange_ClearsResults /
//     BayesianAnalysis_PointEstimatorChange_DoesNotClearResults: INotifyPropertyChanged
//     cascades; no notification system in this port. The invalidation-on-mutate is exercised
//     directly (set_number_of_bins after a run clears results).
//   - Cancel_* / CancelAnalysis_* / RunAsync_WithUnestimatedBivariate (cancellation variant):
//     cancellation + Parallel.For dropped. The pre-flight validation throw is asserted here.
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "bestfit/analyses/bivariate/bivariate_analysis.hpp"
#include "bestfit/analyses/bivariate/coincident_frequency_analysis.hpp"
#include "bestfit/estimation/bayesian_analysis.hpp"
#include "bestfit/models/bivariate_distribution/bivariate_distribution.hpp"
#include "bestfit/models/data_frame/data_frame.hpp"
#include "bestfit/models/univariate_distribution/univariate_distribution_model.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "bestfit/numerics/distributions/copulas/base/copula_type.hpp"
#include "bestfit/numerics/distributions/copulas/normal_copula.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "check.hpp"

using bestfit::analyses::BivariateAnalysis;
using bestfit::analyses::bivariate::CoincidentFrequencyAnalysis;
using bestfit::estimation::PointEstimateType;
using bestfit::models::BivariateDistribution;
using bestfit::models::DataFrame;
using bestfit::models::ExactSeries;
using bestfit::models::UnivariateDistributionModel;
using bestfit::numerics::distributions::Normal;
using bestfit::numerics::distributions::UnivariateDistributionType;
using bestfit::numerics::distributions::copulas::CopulaType;
using bestfit::numerics::distributions::copulas::NormalCopula;

namespace {

using Matrix = std::vector<std::vector<double>>;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// 2x2 sum grid on X = Y = [-1, 1] (used by the hand-derived oracle).
Matrix sum_grid_2x2() { return {{-2.0, 0.0}, {0.0, 2.0}}; }

// 5x5 sum grid on X = Y = [-2,-1,0,1,2] (mirrors the C# BuildSumGrid; used for validation tests).
void sum_grid_5x5(std::vector<double>& x, std::vector<double>& y, Matrix& z) {
    x = {-2.0, -1.0, 0.0, 1.0, 2.0};
    y = {-2.0, -1.0, 0.0, 1.0, 2.0};
    z.assign(5, std::vector<double>(5, 0.0));
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j) z[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = x[static_cast<std::size_t>(i)] + y[static_cast<std::size_t>(j)];
}

// Owns the marginals for the analysis lifetime (BivariateDistribution holds non-owning marginal
// pointers; the analysis owns the distribution). Declared so the analysis is destroyed BEFORE
// the marginals.
struct BiFixture {
    std::unique_ptr<UnivariateDistributionModel> marginal_x;
    std::unique_ptr<UnivariateDistributionModel> marginal_y;
    std::unique_ptr<BivariateAnalysis> analysis;
};

std::unique_ptr<UnivariateDistributionModel> make_marginal(const std::vector<double>& data) {
    DataFrame df;
    df.set_exact_series(ExactSeries(data));
    return std::make_unique<UnivariateDistributionModel>(std::move(df), UnivariateDistributionType::Normal);
}

// A fresh (UNestimated) BivariateAnalysis over Normal marginals + Normal copula.
BiFixture make_fresh_bivariate() {
    BiFixture f;
    f.marginal_x = make_marginal(Normal(0.0, 1.0).generate_random_values(50, 111));
    f.marginal_y = make_marginal(Normal(0.0, 1.0).generate_random_values(50, 222));
    auto bd = std::make_unique<BivariateDistribution>(*f.marginal_x, *f.marginal_y, CopulaType::Normal);
    f.analysis = std::make_unique<BivariateAnalysis>(std::move(bd));
    return f;
}

// An ESTIMATED BivariateAnalysis: a tiny (deterministic, seeded) MCMC over the single copula
// parameter flips is_estimated() true so CFA's validation gate passes. The model has ONE
// parameter (Normal copula rho; marginals are held fixed), so the chain is cheap and robust.
// No numeric MCMC value is asserted -- this only opens CFA's estimated-upstream gate.
BiFixture make_estimated_bivariate() {
    BiFixture f = make_fresh_bivariate();
    auto& ba = f.analysis->bayesian_analysis();
    ba.set_iterations(500);
    ba.set_warmup_iterations(250);
    ba.set_output_length(500);
    ba.set_number_of_chains(4);
    f.analysis->run();
    return f;
}

// ---------------------------------------------------------------------------
// Constructor / defaults
// ---------------------------------------------------------------------------

// C# Constructor_Default_InitializesEmpty (C# 80-93).
void test_constructor_default_empty() {
    CoincidentFrequencyAnalysis cfa;
    CHECK_EQ(cfa.x_values().size(), static_cast<std::size_t>(0));
    CHECK_EQ(cfa.y_values().size(), static_cast<std::size_t>(0));
    CHECK_EQ(cfa.number_of_bins(), 50);
    CHECK_NEAR(cfa.bayesian_analysis().credible_interval_width, 0.90, 1e-12);
    CHECK_TRUE(!cfa.is_estimated());
    CHECK_TRUE(cfa.analysis_results() == nullptr);
    CHECK_TRUE(cfa.z_output_values() == nullptr);
}

// C# Constructor_WithInputs_StoresArguments (C# 98-108).
void test_constructor_with_inputs() {
    BiFixture f = make_fresh_bivariate();
    std::vector<double> x, y;
    Matrix z;
    sum_grid_5x5(x, y, z);
    CoincidentFrequencyAnalysis cfa(f.analysis.get(), x, y, z);
    CHECK_TRUE(&cfa.bivariate_analysis() == f.analysis.get());
    CHECK_EQ(cfa.x_values().size(), static_cast<std::size_t>(5));
    CHECK_EQ(cfa.y_values().size(), static_cast<std::size_t>(5));
    CHECK_EQ(cfa.bivariate_response().size(), static_cast<std::size_t>(5));
}

// C# Constructor_NullBivariateAnalysis_Throws (C# 114-118).
void test_constructor_null_bivariate_throws() {
    std::vector<double> x, y;
    Matrix z;
    sum_grid_5x5(x, y, z);
    CHECK_THROWS(CoincidentFrequencyAnalysis(nullptr, x, y, z));
}

// C# NumberOfBins_OutOfRange_DoesNotThrow_ValidateReportsError (C# 145-155).
void test_number_of_bins_out_of_range_no_throw() {
    CoincidentFrequencyAnalysis cfa;
    cfa.set_number_of_bins(2);
    CHECK_EQ(cfa.number_of_bins(), 2);
    cfa.set_number_of_bins(5000);
    CHECK_EQ(cfa.number_of_bins(), 5000);
}

// C# Constructor_OwnsBayesianAnalysis_ParameterlessInit / BayesianAnalysis_OwnedNotProxied (209-231).
// CFA owns its own presentation-settings object (CredibleIntervalWidth = 0.90), independent of the
// upstream BivariateAnalysis's estimation::BayesianAnalysis.
void test_owned_not_proxied_bayesian() {
    BiFixture f = make_fresh_bivariate();
    CoincidentFrequencyAnalysis cfa(f.analysis.get(), {-1.0, 1.0}, {-1.0, 1.0}, Matrix(2, std::vector<double>(2, 0.0)));
    CHECK_NEAR(cfa.bayesian_analysis().credible_interval_width, 0.90, 1e-12);
    // Distinct object from the upstream estimation::BayesianAnalysis (different type, different address).
    CHECK_TRUE(static_cast<const void*>(&cfa.bayesian_analysis()) !=
               static_cast<const void*>(&f.analysis->bayesian_analysis()));
}

// ---------------------------------------------------------------------------
// Validate (early-return gates -- no estimated upstream needed)
// ---------------------------------------------------------------------------

// C# Validate_MissingBivariateAnalysis_ReturnsError (C# 183-188).
void test_validate_missing_bivariate() {
    CoincidentFrequencyAnalysis cfa;
    auto v = cfa.validate();
    CHECK_TRUE(!v.is_valid);
    bool found = false;
    for (const auto& m : v.validation_messages)
        if (m.find("Bivariate analysis is required") != std::string::npos) found = true;
    CHECK_TRUE(found);
}

// C# Validate_UpstreamNotEstimated_ReturnsError (C# 193-206).
void test_validate_upstream_not_estimated() {
    BiFixture f = make_fresh_bivariate();  // NOT estimated
    std::vector<double> x, y;
    Matrix z;
    sum_grid_5x5(x, y, z);
    CoincidentFrequencyAnalysis cfa(f.analysis.get(), x, y, z);
    auto v = cfa.validate();
    CHECK_TRUE(!v.is_valid);
    bool found = false;
    for (const auto& m : v.validation_messages)
        if (m.find("not been estimated") != std::string::npos) found = true;
    CHECK_TRUE(found);
}

// C# RunAsync_WithUnestimatedBivariate_FailsValidation (C# 685-694): run() throws pre-loop.
void test_run_unestimated_throws() {
    BiFixture f = make_fresh_bivariate();  // NOT estimated
    std::vector<double> x, y;
    Matrix z;
    sum_grid_5x5(x, y, z);
    CoincidentFrequencyAnalysis cfa(f.analysis.get(), x, y, z);
    CHECK_THROWS(cfa.run());
}

// ---------------------------------------------------------------------------
// Validate (detail gates -- need an estimated upstream to pass the not-estimated early return)
// ---------------------------------------------------------------------------

bool has_message(const bestfit::models::ValidationResult& v, const char* needle) {
    for (const auto& m : v.validation_messages)
        if (m.find(needle) != std::string::npos) return true;
    return false;
}

void test_validate_details() {
    BiFixture est = make_estimated_bivariate();

    std::vector<double> x, y;
    Matrix z;
    sum_grid_5x5(x, y, z);

    // Good inputs -> valid, no messages (C# Validate_GoodInputs_IsValidWithNoMessages 276-286).
    {
        CoincidentFrequencyAnalysis cfa(est.analysis.get(), x, y, z);
        auto v = cfa.validate();
        CHECK_TRUE(v.is_valid);
        CHECK_EQ(v.validation_messages.size(), static_cast<std::size_t>(0));
    }
    // Non-ascending X (C# 290-301).
    {
        std::vector<double> bad_x = {-2.0, 0.0, -1.0, 1.0, 2.0};
        CoincidentFrequencyAnalysis cfa(est.analysis.get(), bad_x, y, z);
        auto v = cfa.validate();
        CHECK_TRUE(!v.is_valid);
        CHECK_TRUE(has_message(v, "X (primary) values must be strictly ascending"));
    }
    // Non-ascending Y (C# 305-316).
    {
        std::vector<double> bad_y = {-2.0, -1.0, 1.0, 0.0, 2.0};
        CoincidentFrequencyAnalysis cfa(est.analysis.get(), x, bad_y, z);
        auto v = cfa.validate();
        CHECK_TRUE(!v.is_valid);
        CHECK_TRUE(has_message(v, "Y (secondary) values must be strictly ascending"));
    }
    // Dimension mismatch (C# 320-332): 3x5 response, X length 5.
    {
        Matrix bad_z(3, std::vector<double>(5, 0.0));
        CoincidentFrequencyAnalysis cfa(est.analysis.get(), x, y, bad_z);
        auto v = cfa.validate();
        CHECK_TRUE(!v.is_valid);
        CHECK_TRUE(has_message(v, "must have 5 rows"));
    }
    // Non-monotonic response along X (C# 335-347).
    {
        Matrix bad_z = z;
        bad_z[2][2] = -10.0;
        CoincidentFrequencyAnalysis cfa(est.analysis.get(), x, y, bad_z);
        auto v = cfa.validate();
        CHECK_TRUE(!v.is_valid);
        CHECK_TRUE(has_message(v, "strictly increasing along X"));
    }
    // NaN in response (C# 350-362).
    {
        Matrix bad_z = z;
        bad_z[1][1] = std::numeric_limits<double>::quiet_NaN();
        CoincidentFrequencyAnalysis cfa(est.analysis.get(), x, y, bad_z);
        auto v = cfa.validate();
        CHECK_TRUE(!v.is_valid);
        CHECK_TRUE(has_message(v, "NaN"));
    }
    // NumberOfBins too small (C# 366-376).
    {
        CoincidentFrequencyAnalysis cfa(est.analysis.get(), x, y, z);
        cfa.set_number_of_bins(3);
        auto v = cfa.validate();
        CHECK_TRUE(!v.is_valid);
        CHECK_TRUE(has_message(v, "at least 5"));
    }
    // NumberOfBins too large (C# 380-390).
    {
        CoincidentFrequencyAnalysis cfa(est.analysis.get(), x, y, z);
        cfa.set_number_of_bins(1500);
        auto v = cfa.validate();
        CHECK_TRUE(!v.is_valid);
        CHECK_TRUE(has_message(v, "at most 1000"));
    }
    // NumberOfBins > 100 -> non-blocking warning (C# 393-405).
    {
        CoincidentFrequencyAnalysis cfa(est.analysis.get(), x, y, z);
        cfa.set_number_of_bins(200);
        auto v = cfa.validate();
        CHECK_TRUE(v.is_valid);
        CHECK_TRUE(has_message(v, "slow run times"));
    }
}

// ---------------------------------------------------------------------------
// Hand-derived analytic oracle (PRIMARY) -- static deterministic math, no MCMC.
// ---------------------------------------------------------------------------

// Literal expectations from the derivation in the file header.
constexpr double kAepM2 = 0.9199974240184564;  // z = -2
constexpr double kAepM1 = 0.7386249340259104;  // z = -1
constexpr double kAep0 = 0.5;                   // z =  0
constexpr double kAepP1 = 0.2613750659740896;  // z =  1
constexpr double kAepP2 = 0.08000257598154358;  // z =  2

void test_hand_oracle_static() {
    Matrix z = sum_grid_2x2();
    std::vector<double> x = {-1.0, 1.0};
    std::vector<double> y = {-1.0, 1.0};

    // BuildZOutputBins.
    std::vector<double> zbins = CoincidentFrequencyAnalysis::build_z_output_bins(z, 5);
    CHECK_EQ(zbins.size(), static_cast<std::size_t>(5));
    const double expected_bins[5] = {-2.0, -1.0, 0.0, 1.0, 2.0};
    for (int k = 0; k < 5; ++k) CHECK_NEAR(zbins[static_cast<std::size_t>(k)], expected_bins[k], 1e-12);

    // ComputeYBinEdges.
    std::vector<double> yedges = CoincidentFrequencyAnalysis::compute_y_bin_edges(y);
    CHECK_EQ(yedges.size(), static_cast<std::size_t>(3));
    CHECK_TRUE(std::isinf(yedges[0]) && yedges[0] < 0.0);
    CHECK_NEAR(yedges[1], 0.0, 1e-12);
    CHECK_TRUE(std::isinf(yedges[2]) && yedges[2] > 0.0);

    // BuildXZetas / BuildVEdges with Normal(0,1) marginals.
    Normal marg(0.0, 1.0);
    std::vector<double> xzetas = CoincidentFrequencyAnalysis::build_x_zetas(x, marg);
    CHECK_NEAR(xzetas[0], -1.0, 1e-9);
    CHECK_NEAR(xzetas[1], 1.0, 1e-9);
    std::vector<double> vedges = CoincidentFrequencyAnalysis::build_v_edges(y, yedges, marg);
    CHECK_NEAR(vedges[0], 0.0, 1e-12);
    CHECK_NEAR(vedges[1], 0.5, 1e-9);
    CHECK_NEAR(vedges[2], 1.0, 1e-12);

    // Independence copula: Normal copula with rho = 0 => C(u,v) = u*v.
    NormalCopula copula;
    copula.set_copula_parameters({0.0});

    const double expected_aep[5] = {kAepM2, kAepM1, kAep0, kAepP1, kAepP2};
    for (int k = 0; k < 5; ++k) {
        double fz = CoincidentFrequencyAnalysis::compute_fz_at_bin(zbins[static_cast<std::size_t>(k)], z, xzetas, vedges, copula);
        double aep = 1.0 - fz;
        CHECK_NEAR(aep, expected_aep[k], 1e-9);
    }
}

// ---------------------------------------------------------------------------
// Deterministic run() through the point-estimate-only path (realz <= 0).
// Reproduces the hand-derived oracle end-to-end via the real run() orchestration.
// ---------------------------------------------------------------------------

void test_run_point_estimate_only() {
    BiFixture est = make_estimated_bivariate();

    // Force exact independence + standard-normal marginals (overriding the tiny MCMC fit).
    est.marginal_x->distribution().set_parameters({0.0, 1.0});
    est.marginal_y->distribution().set_parameters({0.0, 1.0});
    est.analysis->bivariate_distribution().copula().set_copula_parameters({0.0});
    // Null the inner posterior so CFA takes the point-estimate-only (realz = 0) path. The outer
    // BivariateAnalysis::is_estimated() flag survives (independent of the inner results), so CFA's
    // validation gate still passes -- mirroring the C# XML-forced-estimated, no-MCMC-chain test.
    est.analysis->bayesian_analysis().clear_results();

    std::vector<double> x = {-1.0, 1.0};
    std::vector<double> y = {-1.0, 1.0};
    Matrix z = sum_grid_2x2();
    CoincidentFrequencyAnalysis cfa(est.analysis.get(), x, y, z);
    cfa.set_number_of_bins(5);
    cfa.run();

    CHECK_TRUE(cfa.is_estimated());
    const std::vector<double>* zo = cfa.z_output_values();
    CHECK_TRUE(zo != nullptr);
    CHECK_EQ(zo->size(), static_cast<std::size_t>(5));
    CHECK_NEAR((*zo)[0], -2.0, 1e-12);
    CHECK_NEAR((*zo)[4], 2.0, 1e-12);

    const auto* ar = cfa.analysis_results();
    CHECK_TRUE(ar != nullptr);
    const double expected_aep[5] = {kAepM2, kAepM1, kAep0, kAepP1, kAepP2};
    for (int k = 0; k < 5; ++k) {
        CHECK_NEAR(ar->mode_curve[static_cast<std::size_t>(k)], expected_aep[k], 1e-9);
        // Mean curve == mode curve on the point-estimate-only path.
        CHECK_NEAR(ar->mean_curve[static_cast<std::size_t>(k)], ar->mode_curve[static_cast<std::size_t>(k)], 0.0);
        // CIs are NaN bands (no posterior samples).
        CHECK_TRUE(std::isnan(ar->confidence_intervals[static_cast<std::size_t>(k)][0]));
        CHECK_TRUE(std::isnan(ar->confidence_intervals[static_cast<std::size_t>(k)][1]));
    }

    // ClearResults_AfterRun_DropsResultsAndZOutputs (C# 533-546).
    cfa.clear_results();
    CHECK_TRUE(cfa.analysis_results() == nullptr);
    CHECK_TRUE(cfa.z_output_values() == nullptr);
    CHECK_TRUE(!cfa.is_estimated());
}

// ---------------------------------------------------------------------------
// run() through the posterior loop (realz > 0): structural invariants only.
// ---------------------------------------------------------------------------

void test_run_posterior_loop_structure() {
    BiFixture est = make_estimated_bivariate();  // inner results retained -> realz > 0

    std::vector<double> x, y;
    Matrix z;
    sum_grid_5x5(x, y, z);
    // Standard-normal marginals for a well-behaved surface inversion.
    est.marginal_x->distribution().set_parameters({0.0, 1.0});
    est.marginal_y->distribution().set_parameters({0.0, 1.0});

    CoincidentFrequencyAnalysis cfa(est.analysis.get(), x, y, z);
    cfa.set_number_of_bins(10);
    cfa.run();

    CHECK_TRUE(cfa.is_estimated());
    const std::vector<double>* zo = cfa.z_output_values();
    CHECK_TRUE(zo != nullptr);
    CHECK_EQ(zo->size(), static_cast<std::size_t>(10));
    CHECK_NEAR((*zo)[0], -4.0, 1e-12);
    CHECK_NEAR((*zo)[9], 4.0, 1e-12);

    const auto* ar = cfa.analysis_results();
    CHECK_TRUE(ar != nullptr);
    for (int k = 0; k < 10; ++k) {
        double mode = ar->mode_curve[static_cast<std::size_t>(k)];
        double mean = ar->mean_curve[static_cast<std::size_t>(k)];
        double lo = ar->confidence_intervals[static_cast<std::size_t>(k)][0];
        double hi = ar->confidence_intervals[static_cast<std::size_t>(k)][1];
        CHECK_TRUE(std::isfinite(mode));
        CHECK_TRUE(std::isfinite(mean));
        // realz > 0 => CIs are real (not NaN) and ordered.
        CHECK_TRUE(std::isfinite(lo));
        CHECK_TRUE(std::isfinite(hi));
        CHECK_TRUE(lo <= hi + 1e-12);
        // AEP curve is (weakly) monotonically decreasing in z.
        if (k > 0) CHECK_TRUE(mode <= ar->mode_curve[static_cast<std::size_t>(k - 1)] + 1e-9);
    }

    // Mutating NumberOfBins after a run clears derived results (invalidation-on-mutate).
    cfa.set_number_of_bins(20);
    CHECK_TRUE(cfa.analysis_results() == nullptr);
    CHECK_TRUE(cfa.z_output_values() == nullptr);
    CHECK_TRUE(!cfa.is_estimated());
}

}  // namespace

int main() {
    test_constructor_default_empty();
    test_constructor_with_inputs();
    test_constructor_null_bivariate_throws();
    test_number_of_bins_out_of_range_no_throw();
    test_owned_not_proxied_bayesian();
    test_validate_missing_bivariate();
    test_validate_upstream_not_estimated();
    test_run_unestimated_throws();
    test_validate_details();
    test_hand_oracle_static();
    test_run_point_estimate_only();
    test_run_posterior_loop_structure();

    return bftest::summary("coincident_frequency_analysis");
}
