// Standalone tests for bestfit::estimation::MaximumAPosteriori (Phase 4, Task T8).
//
// Oracle for behavior is the C# source itself (upstream/RMC-BestFit/src/RMC.BestFit/Estimation/
// MaximumAPosteriori.cs @ fc28c0c) -- see maximum_a_posteriori.hpp's header for the exact
// method/line mapping and the sign-convention / adapter / gating notes. This ctest asserts:
//   - Estimate() with NelderMead succeeds on the Phase-4 dataset + a Normal model.
//   - The sign convention: maximum_log_likelihood() == model.log_likelihood(best values)
//     (the log-POSTERIOR, data + prior -- NOT data_log_likelihood).
//   - A LOOSE anchor to the MLE fit: the Phase-4 Normal model's set_default_parameters() gives
//     each parameter a Uniform(lower, upper) prior, so the posterior = data loglik + a flat-prior
//     constant, and the MAP fit ~ the MLE fit (mu ~ 16026, sigma ~ 5058). The exact MAP oracle
//     (with an informative prior) is T12's job; this is a sanity check only.
//   - GetCovarianceMatrix is symmetric and Cholesky-factorizable (positive-definite); standard
//     errors are all positive.
//   - GetAIC/GetBIC match GoodnessOfFit::aic/bic on maximum_log_likelihood directly.
//   - ParameterConfidenceIntervals brackets the fitted point estimate.
//   - Gating: BFGS/Powell/MultilevelSingleLinkage throw; Brent on a 2-parameter model throws;
//     compute_leverage_diagnostics() throws (Diagnostics layer deferred).
//   - get_observation_influence()/get_cooks_distance(): shape/finiteness/nonnegativity only
//     (exact oracles are T12's job).
//   - NO sandwich/robust methods exist on this class (MAP has no analogue -- see the brief).
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "bestfit/estimation/maximum_a_posteriori.hpp"
#include "bestfit/estimation/optimization_method.hpp"
#include "bestfit/models/univariate_distribution_model.hpp"
#include "bestfit/numerics/data/goodness_of_fit.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "bestfit/numerics/math/linalg/cholesky_decomposition.hpp"
#include "check.hpp"

using bestfit::estimation::MaximumAPosteriori;
using bestfit::estimation::OptimizationMethod;
using bestfit::models::UnivariateDistributionModel;
using bestfit::numerics::data::GoodnessOfFit;
using bestfit::numerics::distributions::UnivariateDistributionType;
using bestfit::numerics::math::linalg::CholeskyDecomposition;

namespace {

// Phase-4 dataset (brief's canonical sample; matches test_maximum_likelihood.cpp).
std::vector<double> sample_data() {
    return {12500, 15300, 9870, 21000, 18400, 11200, 26800, 14100, 19500, 11600};
}

void test_estimate_succeeds_with_nelder_mead() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumAPosteriori map(model, OptimizationMethod::NelderMead);

    bool ok = map.estimate();

    CHECK_TRUE(ok);
    CHECK_TRUE(map.is_estimated());
    CHECK_TRUE(map.status() == bestfit::numerics::math::optimization::OptimizationStatus::Success);
}

// Sign convention: maximum_log_likelihood() is the log-POSTERIOR (data + prior), i.e.
// model.log_likelihood(best values) -- NOT model.data_log_likelihood(best values).
void test_sign_convention_matches_log_likelihood() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumAPosteriori map(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(map.estimate());

    double from_model = model.log_likelihood(map.best_parameter_set().values);
    CHECK_NEAR(map.maximum_log_likelihood(), from_model, 1e-9);
}

// LOOSE anchor to the MLE fit: flat (Uniform) priors on both parameters mean the posterior
// differs from the data log-likelihood by only a parameter-independent constant, so the MAP
// point estimate should land close to the known MLE anchor (see test_maximum_likelihood.cpp).
// This is NOT a precise oracle -- exact MAP fixtures (with an informative prior) land with T12.
void test_loose_anchor_to_mle_fit() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumAPosteriori map(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(map.estimate());

    const auto& best = map.best_parameter_set().values;
    CHECK_TRUE(best.size() == 2);

    double rel_tol = 5e-2;
    CHECK_TRUE(std::fabs(best[0] - 16026.055) <= rel_tol * std::fabs(16026.055));
    CHECK_TRUE(std::fabs(best[1] - 5058.828) <= rel_tol * std::fabs(5058.828));
}

void test_covariance_matrix_is_symmetric_and_positive_definite() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumAPosteriori map(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(map.estimate());

    auto covariance = map.get_covariance_matrix();
    CHECK_TRUE(covariance.number_of_rows() == 2);
    CHECK_TRUE(covariance.number_of_columns() == 2);
    CHECK_NEAR(covariance(0, 1), covariance(1, 0), 1e-9);

    bool threw = false;
    try {
        CholeskyDecomposition chol(covariance);
        (void)chol;
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK_TRUE(!threw);

    auto standard_errors = map.get_standard_errors();
    CHECK_TRUE(standard_errors.size() == 2);
    for (double se : standard_errors) CHECK_TRUE(se > 0.0);
}

void test_aic_and_bic_match_goodness_of_fit() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumAPosteriori map(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(map.estimate());

    double mll = map.maximum_log_likelihood();
    CHECK_NEAR(map.get_aic(), GoodnessOfFit::aic(2, mll), 1e-12);
    CHECK_NEAR(map.get_bic(10), GoodnessOfFit::bic(10, 2, mll), 1e-12);
}

void test_parameter_confidence_intervals_bracket_the_estimate() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumAPosteriori map(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(map.estimate());

    auto cis = map.parameter_confidence_intervals(0.1);
    const auto& best = map.best_parameter_set().values;

    CHECK_TRUE(cis.number_of_rows() == 2);
    CHECK_TRUE(cis.number_of_columns() == 2);
    for (int i = 0; i < 2; ++i) {
        CHECK_TRUE(cis(i, 0) <= best[static_cast<std::size_t>(i)]);
        CHECK_TRUE(best[static_cast<std::size_t>(i)] <= cis(i, 1));
    }
}

void test_gated_optimization_methods_throw() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());

    CHECK_THROWS(MaximumAPosteriori(model, OptimizationMethod::BFGS));
    CHECK_THROWS(MaximumAPosteriori(model, OptimizationMethod::Powell));
    CHECK_THROWS(MaximumAPosteriori(model, OptimizationMethod::MultilevelSingleLinkage));
}

void test_brent_on_two_parameter_model_throws() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    CHECK_THROWS(MaximumAPosteriori(model, OptimizationMethod::Brent));
}

// compute_leverage_diagnostics(): Diagnostics layer deferred -- must throw (gate, per brief).
void test_leverage_diagnostics_gated() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumAPosteriori map(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(map.estimate());

    CHECK_THROWS(map.compute_leverage_diagnostics());
}

void test_profile_likelihood_shape_and_finiteness() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumAPosteriori map(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(map.estimate());

    const int bins = 25;
    auto profiles = map.profile_likelihood(bins);
    CHECK_TRUE(profiles.size() == static_cast<std::size_t>(model.number_of_parameters()));
    const auto& best = map.best_parameter_set().values;

    for (std::size_t p = 0; p < profiles.size(); ++p) {
        const auto& profile = profiles[p];
        CHECK_TRUE(profile.number_of_rows() == bins);
        CHECK_TRUE(profile.number_of_columns() == 2);

        const auto& parameter = model.parameters()[p];
        int closest_bin = 0;
        double closest_distance = std::numeric_limits<double>::infinity();
        for (int j = 0; j < bins; ++j) {
            double midpoint = profile(j, 0);
            CHECK_TRUE(midpoint >= parameter.lower_bound());
            CHECK_TRUE(midpoint <= parameter.upper_bound());
            CHECK_TRUE(!std::isnan(profile(j, 1)));

            double distance = std::fabs(midpoint - best[p]);
            if (distance < closest_distance) {
                closest_distance = distance;
                closest_bin = j;
            }
        }
        CHECK_TRUE(std::isfinite(profile(closest_bin, 1)));
    }
}

// get_observation_influence(): Nx(NumberOfParameters) Matrix, every entry finite.
void test_observation_influence_shape_and_finiteness() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumAPosteriori map(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(map.estimate());

    auto influence = map.get_observation_influence();
    CHECK_TRUE(influence.number_of_rows() == static_cast<int>(sample_data().size()));
    CHECK_TRUE(influence.number_of_columns() == model.number_of_parameters());

    for (int i = 0; i < influence.number_of_rows(); ++i)
        for (int j = 0; j < influence.number_of_columns(); ++j) CHECK_TRUE(std::isfinite(influence(i, j)));
}

// get_cooks_distance(): length n_obs, every entry >= 0 (it's a quadratic-form / p measure).
void test_cooks_distance_shape_and_nonnegative() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumAPosteriori map(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(map.estimate());

    auto cooks_d = map.get_cooks_distance();
    CHECK_TRUE(cooks_d.size() == sample_data().size());
    for (double d : cooks_d) CHECK_TRUE(std::isfinite(d) && d >= 0.0);
}

// set_optimizer_method()/clear_results(): switching methods (or calling clear_results directly)
// resets is_estimated()/status() to their pre-estimate state.
void test_set_optimizer_method_and_clear_results_reset_state() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumAPosteriori map(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(map.estimate());
    CHECK_TRUE(map.is_estimated());

    map.clear_results();
    CHECK_TRUE(!map.is_estimated());
    CHECK_TRUE(map.status() == bestfit::numerics::math::optimization::OptimizationStatus::None);

    CHECK_TRUE(map.estimate());
    CHECK_TRUE(map.is_estimated());

    // Switching to a different method clears results and rebuilds the optimizer.
    map.set_optimizer_method(OptimizationMethod::DifferentialEvolution);
    CHECK_TRUE(!map.is_estimated());
    CHECK_TRUE(map.optimizer_method() == OptimizationMethod::DifferentialEvolution);

    // Setting the SAME method is a no-op (C# guards on method != current method): prior
    // results should survive.
    CHECK_TRUE(map.estimate());
    map.set_optimizer_method(OptimizationMethod::DifferentialEvolution);
    CHECK_TRUE(map.is_estimated());
}

}  // namespace

int main() {
    test_estimate_succeeds_with_nelder_mead();
    test_sign_convention_matches_log_likelihood();
    test_loose_anchor_to_mle_fit();
    test_covariance_matrix_is_symmetric_and_positive_definite();
    test_aic_and_bic_match_goodness_of_fit();
    test_parameter_confidence_intervals_bracket_the_estimate();
    test_gated_optimization_methods_throw();
    test_brent_on_two_parameter_model_throws();
    test_leverage_diagnostics_gated();
    test_profile_likelihood_shape_and_finiteness();
    test_observation_influence_shape_and_finiteness();
    test_cooks_distance_shape_and_nonnegative();
    test_set_optimizer_method_and_clear_results_reset_state();

    return bftest::summary("maximum_a_posteriori");
}
