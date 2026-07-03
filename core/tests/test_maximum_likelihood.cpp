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
#include <cmath>
#include <stdexcept>
#include <vector>

#include "bestfit/estimation/maximum_likelihood.hpp"
#include "bestfit/estimation/optimization_method.hpp"
#include "bestfit/models/univariate_distribution_model.hpp"
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

    return bftest::summary("maximum_likelihood");
}
