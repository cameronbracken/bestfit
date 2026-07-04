// Standalone tests for bestfit::estimation::MaximumLikelihood (Phase 4, Task T7).
//
// Oracle for behavior is the C# source itself (upstream/RMC-BestFit/src/RMC.BestFit/Estimation/
// MaximumLikelihood.cs @ fc28c0c) -- see maximum_likelihood.hpp's header for the exact
// method/line mapping and the sign-convention / adapter / gating notes. This ctest asserts:
//   - Estimate() with NelderMead succeeds on the Phase-4 dataset + a Normal model.
//   - The sign convention: maximum_log_likelihood() == model.data_log_likelihood(best values).
//   - A LOOSE real-oracle anchor (spike-proven against the REAL C# library, NelderMead): the
//     exact cross-language oracle + tolerance policy is T12's job -- C++ NelderMead may differ
//     from C# NelderMead by ULP-accumulated amounts, so this is a sanity check only (rel ~1e-3).
//   - GetCovarianceMatrix is symmetric and Cholesky-factorizable (positive-definite); standard
//     errors are all positive.
//   - GetAIC/GetBIC match GoodnessOfFit::aic/bic directly.
//   - ParameterConfidenceIntervals brackets the fitted point estimate.
//   - Gating: BFGS/Powell/MultilevelSingleLinkage throw; Brent on a 2-parameter model throws.
//   - profile_likelihood()/get_sandwich_covariance_matrix()/get_robust_standard_errors()/
//     get_observation_influence()/get_cooks_distance()/set_optimizer_method()/clear_results():
//     self-consistency + shape checks only (exact oracles are T12's job).
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "bestfit/estimation/maximum_likelihood.hpp"
#include "bestfit/estimation/optimization_method.hpp"
#include "bestfit/models/univariate_distribution/univariate_distribution_model.hpp"
#include "bestfit/numerics/data/goodness_of_fit.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "bestfit/numerics/math/linalg/cholesky_decomposition.hpp"
#include "check.hpp"

using bestfit::estimation::MaximumLikelihood;
using bestfit::estimation::OptimizationMethod;
using bestfit::models::UnivariateDistributionModel;
using bestfit::numerics::data::GoodnessOfFit;
using bestfit::numerics::distributions::UnivariateDistributionType;
using bestfit::numerics::math::linalg::CholeskyDecomposition;

namespace {

// Phase-4 dataset (brief's canonical sample; matches test_univariate_distribution_model.cpp).
std::vector<double> sample_data() {
    return {12500, 15300, 9870, 21000, 18400, 11200, 26800, 14100, 19500, 11600};
}

void test_estimate_succeeds_with_nelder_mead() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumLikelihood mle(model, OptimizationMethod::NelderMead);

    bool ok = mle.estimate();

    CHECK_TRUE(ok);
    CHECK_TRUE(mle.is_estimated());
    CHECK_TRUE(mle.status() == bestfit::numerics::math::optimization::OptimizationStatus::Success);
}

void test_sign_convention_matches_data_log_likelihood() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumLikelihood mle(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(mle.estimate());

    double from_model = model.data_log_likelihood(mle.best_parameter_set().values);
    CHECK_NEAR(mle.maximum_log_likelihood(), from_model, 1e-9);
}

// LOOSE real-oracle anchor (spike-proven against the real C# Numerics/RMC-BestFit library,
// NelderMead): fitted mu ~ 16026.055, sigma ~ 5058.828, maximum_log_likelihood ~ -99.4793.
// The exact oracle + tolerance policy lands with T12; this is a sanity check only.
void test_loose_oracle_anchor() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumLikelihood mle(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(mle.estimate());

    const auto& best = mle.best_parameter_set().values;
    CHECK_TRUE(best.size() == 2);

    double rel_tol = 1e-3;
    CHECK_TRUE(std::fabs(best[0] - 16026.055) <= rel_tol * std::fabs(16026.055));
    CHECK_TRUE(std::fabs(best[1] - 5058.828) <= rel_tol * std::fabs(5058.828));
    CHECK_TRUE(std::fabs(mle.maximum_log_likelihood() - (-99.4793)) <= rel_tol * std::fabs(99.4793));
}

void test_covariance_matrix_is_symmetric_and_positive_definite() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumLikelihood mle(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(mle.estimate());

    auto covariance = mle.get_covariance_matrix();
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

    auto standard_errors = mle.get_standard_errors();
    CHECK_TRUE(standard_errors.size() == 2);
    for (double se : standard_errors) CHECK_TRUE(se > 0.0);
}

void test_aic_and_bic_match_goodness_of_fit() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumLikelihood mle(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(mle.estimate());

    double mll = mle.maximum_log_likelihood();
    CHECK_NEAR(mle.get_aic(), GoodnessOfFit::aic(2, mll), 1e-12);
    CHECK_NEAR(mle.get_bic(10), GoodnessOfFit::bic(10, 2, mll), 1e-12);
}

void test_parameter_confidence_intervals_bracket_the_estimate() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumLikelihood mle(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(mle.estimate());

    auto cis = mle.parameter_confidence_intervals(0.1);
    const auto& best = mle.best_parameter_set().values;

    CHECK_TRUE(cis.number_of_rows() == 2);
    CHECK_TRUE(cis.number_of_columns() == 2);
    for (int i = 0; i < 2; ++i) {
        CHECK_TRUE(cis(i, 0) <= best[static_cast<std::size_t>(i)]);
        CHECK_TRUE(best[static_cast<std::size_t>(i)] <= cis(i, 1));
    }
}

void test_gated_optimization_methods_throw() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());

    CHECK_THROWS(MaximumLikelihood(model, OptimizationMethod::BFGS));
    CHECK_THROWS(MaximumLikelihood(model, OptimizationMethod::Powell));
    CHECK_THROWS(MaximumLikelihood(model, OptimizationMethod::MultilevelSingleLinkage));
}

void test_brent_on_two_parameter_model_throws() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    CHECK_THROWS(MaximumLikelihood(model, OptimizationMethod::Brent));
}

// profile_likelihood(): one Matrix(bins, 2) per parameter, midpoints within [lower, upper],
// and the loglik entry at the bin closest to the fitted value finite (self-consistency only --
// exact oracle is T12). NOTE: unlike the fitted point, the FAR ends of the default constraint
// ranges (get_parameter_constraints() sets very wide bounds, e.g. mu in +-1e6 for this sample)
// legitimately underflow the Normal density to exactly 0, so data_log_likelihood() returns -inf
// there -- a correct mathematical result, not a bug -- so this test only requires "not NaN"
// everywhere and finiteness specifically near the optimum.
void test_profile_likelihood_shape_and_finiteness() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumLikelihood mle(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(mle.estimate());

    const int bins = 25;
    auto profiles = mle.profile_likelihood(bins);
    CHECK_TRUE(profiles.size() == static_cast<std::size_t>(model.number_of_parameters()));
    const auto& best = mle.best_parameter_set().values;

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

// get_sandwich_covariance_matrix() / get_robust_standard_errors(): symmetric matrix, all
// robust standard errors strictly positive (self-consistency only -- exact oracle is T12).
void test_sandwich_covariance_and_robust_standard_errors() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumLikelihood mle(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(mle.estimate());

    auto sandwich = mle.get_sandwich_covariance_matrix();
    CHECK_TRUE(sandwich.number_of_rows() == 2);
    CHECK_TRUE(sandwich.number_of_columns() == 2);
    CHECK_NEAR(sandwich(0, 1), sandwich(1, 0), 1e-9);

    auto robust_se = mle.get_robust_standard_errors();
    CHECK_TRUE(robust_se.size() == 2);
    for (double se : robust_se) CHECK_TRUE(se > 0.0);
}

// get_observation_influence(): Nx(NumberOfParameters) Matrix, every entry finite.
void test_observation_influence_shape_and_finiteness() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumLikelihood mle(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(mle.estimate());

    auto influence = mle.get_observation_influence();
    CHECK_TRUE(influence.number_of_rows() == static_cast<int>(sample_data().size()));
    CHECK_TRUE(influence.number_of_columns() == model.number_of_parameters());

    for (int i = 0; i < influence.number_of_rows(); ++i)
        for (int j = 0; j < influence.number_of_columns(); ++j) CHECK_TRUE(std::isfinite(influence(i, j)));
}

// get_cooks_distance(): length n_obs, every entry >= 0 (it's a quadratic-form / p measure).
void test_cooks_distance_shape_and_nonnegative() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumLikelihood mle(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(mle.estimate());

    auto cooks_d = mle.get_cooks_distance();
    CHECK_TRUE(cooks_d.size() == sample_data().size());
    for (double d : cooks_d) CHECK_TRUE(std::isfinite(d) && d >= 0.0);
}

// set_optimizer_method()/clear_results(): switching methods (or calling clear_results directly)
// resets is_estimated()/status() to their pre-estimate state.
void test_set_optimizer_method_and_clear_results_reset_state() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumLikelihood mle(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(mle.estimate());
    CHECK_TRUE(mle.is_estimated());

    mle.clear_results();
    CHECK_TRUE(!mle.is_estimated());
    CHECK_TRUE(mle.status() == bestfit::numerics::math::optimization::OptimizationStatus::None);

    CHECK_TRUE(mle.estimate());
    CHECK_TRUE(mle.is_estimated());

    // Switching to a different method clears results and rebuilds the optimizer.
    mle.set_optimizer_method(OptimizationMethod::DifferentialEvolution);
    CHECK_TRUE(!mle.is_estimated());
    CHECK_TRUE(mle.optimizer_method() == OptimizationMethod::DifferentialEvolution);

    // Setting the SAME method is a no-op (C# 59-71 guards on method != current method): prior
    // results should survive.
    CHECK_TRUE(mle.estimate());
    mle.set_optimizer_method(OptimizationMethod::DifferentialEvolution);
    CHECK_TRUE(mle.is_estimated());
}

}  // namespace

int main() {
    test_estimate_succeeds_with_nelder_mead();
    test_sign_convention_matches_data_log_likelihood();
    test_loose_oracle_anchor();
    test_covariance_matrix_is_symmetric_and_positive_definite();
    test_aic_and_bic_match_goodness_of_fit();
    test_parameter_confidence_intervals_bracket_the_estimate();
    test_gated_optimization_methods_throw();
    test_brent_on_two_parameter_model_throws();
    test_profile_likelihood_shape_and_finiteness();
    test_sandwich_covariance_and_robust_standard_errors();
    test_observation_influence_shape_and_finiteness();
    test_cooks_distance_shape_and_nonnegative();
    test_set_optimizer_method_and_clear_results_reset_state();

    return bftest::summary("maximum_likelihood");
}
