// ported from: Numerics/Data/Regression/LinearRegression.cs @ a2c4dbf
//
// Linear regression Y = alpha + beta*X + e, e ~ N(0, sigma), fitted by singular value
// decomposition (SVD). Ported with Task B9 because DataFrame.GetNonparametricMomentsROS
// fits its Regression-on-Order-Statistics line through this class.
//
// Ported surface: the (Matrix, Vector, hasIntercept) constructor with its three guards,
// FitSVD (coefficients, covariance, residuals, standard error, R^2 / adjusted R^2, degrees
// of freedom), HasIntercept / Y / X / Parameters / ParameterStandardErrors / Covariance /
// Residuals / StandardError / SampleSize / DegreesOfFreedom / RSquared / AdjRSquared,
// PrepareDesignMatrix / AddInterceptColumn, Predict, and PredictionIntervals.
//
// Deliberately NOT ported (display-only surface):
//   - Summary(): the R-style report table (its only consumers are humans; it needs the
//     unported HypothesisTests.FtestModels + Statistics.FiveNumberSummary formatting).
//   - ParameterNames and the Vector/Matrix `Header` handling that feeds it: Header is a UI
//     metadata property omitted from the ported linalg types (Phase 2 decision), and
//     ParameterNames exists solely for the Summary() report.
//   - ParameterTStats: declared upstream but never assigned by FitSVD (always null there),
//     so there is nothing to port.
//
// Exception mapping: C# ArgumentException -> std::invalid_argument; C# ArithmeticException
// -> std::domain_error (the repo's established mapping, see mcmc/hmc.hpp);
// C# ArgumentException in PrepareDesignMatrix -> std::invalid_argument.
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/distributions/student_t.hpp"
#include "bestfit/numerics/data/statistics.hpp"
#include "bestfit/numerics/math/linalg/matrix.hpp"
#include "bestfit/numerics/math/linalg/singular_value_decomposition.hpp"
#include "bestfit/numerics/math/linalg/vector.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::data::regression {

class LinearRegression {
   public:
    using Matrix = math::linalg::Matrix;
    using Vector = math::linalg::Vector;

    // Estimates the model Y = alpha + beta*X + e, where e ~ N(0, sigma).
    // `x` is the matrix of predictor values, `y` the response vector; `has_intercept`
    // determines if an intercept should be estimated (default true).
    LinearRegression(const Matrix& x, const Vector& y, bool has_intercept = true)
        : has_intercept_(has_intercept),
          y_(y),
          x_(has_intercept ? add_intercept_column(x) : x) {
        if (y.length() != x.number_of_rows())
            throw std::invalid_argument("X and Y must have the same number of rows.");
        if (y.length() <= 2) throw std::domain_error("There must be at least three data points.");
        if (x.number_of_columns() > y.length())
            throw std::domain_error(
                "A regression of the requested order requires at least " +
                std::to_string(x.number_of_columns()) + " data points. Only " +
                std::to_string(y.length()) + " data points have been provided.");

        // (The C# Y.Header / ParameterNames bookkeeping feeds only the skipped Summary()
        // report -- see the file header.)

        // Estimate the linear model.
        fit_svd();
    }

    // Determines if the linear model has an intercept.
    bool has_intercept() const { return has_intercept_; }

    // The vector of response values.
    const Vector& y() const { return y_; }

    // The matrix of predictor values (with the intercept column when HasIntercept).
    const Matrix& x() const { return x_; }

    // The list of estimated parameter values.
    const std::vector<double>& parameters() const { return parameters_; }

    // The list of the estimated parameter standard errors.
    const std::vector<double>& parameter_standard_errors() const {
        return parameter_standard_errors_;
    }

    // The estimated parameter covariance matrix (unscaled -- multiply by StandardError^2
    // for the coefficient covariance, exactly as the C# stores it).
    const Matrix& covariance() const { return covariance_; }

    // The residuals of the fitted linear model.
    const std::vector<double>& residuals() const { return residuals_; }

    // The model standard error.
    double standard_error() const { return standard_error_; }

    // The data sample size.
    int sample_size() const { return y_.length(); }

    // The model degrees of freedom.
    int degrees_of_freedom() const { return degrees_of_freedom_; }

    // The coefficient of determination (R-squared).
    double r_squared() const { return r_squared_; }

    // Adjusted R-squared.
    double adj_r_squared() const { return adj_r_squared_; }

    // Returns the predicted Y values given the X-values.
    std::vector<double> predict(const Matrix& x) const {
        Matrix xp = prepare_design_matrix(x);
        std::vector<double> result(static_cast<std::size_t>(xp.number_of_rows()));
        for (int i = 0; i < xp.number_of_rows(); i++) {
            // Tools.SumProduct(Parameters, xp.Row(i))
            std::vector<double> row(static_cast<std::size_t>(xp.number_of_columns()));
            for (int j = 0; j < xp.number_of_columns(); j++)
                row[static_cast<std::size_t>(j)] = xp(i, j);
            result[static_cast<std::size_t>(i)] = numerics::sum_product(parameters_, row);
        }
        return result;
    }

    // Returns the prediction intervals for Y in a matrix with columns: lower, upper, mean.
    // `alpha` is the confidence level; default 0.1 yields the 90 percent intervals.
    Matrix prediction_intervals(const Matrix& x, double alpha = 0.1) const {
        Matrix xp = prepare_design_matrix(x);
        const double percentiles[2] = {alpha / 2.0, 1.0 - alpha / 2.0};
        Matrix result(xp.number_of_rows(), 3);  // lower, upper, mean
        for (int i = 0; i < xp.number_of_rows(); i++) {
            std::vector<double> row(static_cast<std::size_t>(xp.number_of_columns()));
            for (int j = 0; j < xp.number_of_columns(); j++)
                row[static_cast<std::size_t>(j)] = xp(i, j);
            double mu = numerics::sum_product(parameters_, row);
            double s = standard_error_;
            double s2 = s * s;
            double n = static_cast<double>(degrees_of_freedom_);
            distributions::StudentT t(mu, std::sqrt(s2 / n + s2), n);
            result(i, 0) = t.inverse_cdf(percentiles[0]);
            result(i, 1) = t.inverse_cdf(percentiles[1]);
            result(i, 2) = mu;
        }
        return result;
    }

   private:
    // Estimate the model using Singular Value Decomposition (C# FitSVD).
    void fit_svd() {
        double mean_y = data::mean(y_.to_array());
        int i, j, k, n = x_.number_of_rows(), m = x_.number_of_columns();
        double sse = 0.0, sst = 0.0, sum = 0.0;
        degrees_of_freedom_ = n - m;
        residuals_.assign(static_cast<std::size_t>(n), 0.0);
        covariance_ = Matrix(m);

        // Estimate coefficients
        math::linalg::SingularValueDecomposition svd(x_);
        double tol = 1E-12;
        double thresh = (tol > 0.0 ? tol * svd.w()[0] : -1.0);
        Vector betas = svd.solve(y_, thresh);  // vector of fitted coefficients

        // Estimate uncertainty in the coefficients
        for (i = 0; i < n; i++) {
            sum = 0.0;
            for (j = 0; j < m; j++) sum += x_(i, j) * betas[j];
            residuals_[static_cast<std::size_t>(i)] = y_[i] - sum;
            sse += numerics::sqr(residuals_[static_cast<std::size_t>(i)]);
            sst += numerics::sqr(y_[i] - mean_y);
        }
        for (i = 0; i < m; i++) {
            for (j = 0; j < i + 1; j++) {
                sum = 0.0;
                for (k = 0; k < m; k++)
                    if (svd.w()[k] > svd.threshold())
                        sum += svd.v()(i, k) * svd.v()(j, k) / numerics::sqr(svd.w()[k]);
                covariance_(j, i) = covariance_(i, j) = sum;
            }
        }
        double se = std::sqrt(sse / degrees_of_freedom_);

        // Set the output
        parameters_ = betas.to_array();
        parameter_standard_errors_.clear();
        for (j = 0; j < m; j++)
            parameter_standard_errors_.push_back(std::sqrt(covariance_(j, j)) * se);
        standard_error_ = se;
        r_squared_ = 1 - sse / sst;
        adj_r_squared_ = 1 - (1 - r_squared_) * (n - 1) / degrees_of_freedom_;
    }

    // Prepares the design matrix for prediction by adding an intercept column if needed.
    // If the matrix already has the expected number of columns (e.g. internal X), it passes
    // through; if it has one fewer column and HasIntercept is true, the intercept column is
    // added.
    Matrix prepare_design_matrix(const Matrix& x) const {
        int expected = static_cast<int>(parameters_.size());
        if (x.number_of_columns() == expected) return x;
        if (has_intercept_ && x.number_of_columns() == expected - 1)
            return add_intercept_column(x);
        throw std::invalid_argument(
            "Expected " + std::to_string(expected) + " columns" +
            (has_intercept_ ? " (or " + std::to_string(expected - 1) + " without intercept)"
                            : "") +
            ", but got " + std::to_string(x.number_of_columns()) + ".");
    }

    // Helper method to add an intercept column to the covariate matrix.
    static Matrix add_intercept_column(const Matrix& x) {
        Matrix result(x.number_of_rows(), x.number_of_columns() + 1);
        for (int i = 0; i < x.number_of_rows(); i++) {
            result(i, 0) = 1.0;
            for (int j = 0; j < x.number_of_columns(); j++) result(i, j + 1) = x(i, j);
        }
        return result;
    }

    bool has_intercept_;
    Vector y_;
    Matrix x_;
    std::vector<double> parameters_;
    std::vector<double> parameter_standard_errors_;
    Matrix covariance_{0, 0};
    std::vector<double> residuals_;
    double standard_error_ = 0.0;
    int degrees_of_freedom_ = 0;
    double r_squared_ = 0.0;
    double adj_r_squared_ = 0.0;
};

}  // namespace bestfit::numerics::data::regression
