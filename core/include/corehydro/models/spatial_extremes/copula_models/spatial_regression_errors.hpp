// ported from: RMC-BestFit/src/RMC.BestFit/Models/SpatialExtremes/CopulaModels/SpatialRegressionErrors.cs @ c2e6192
//
// Spatial regression errors using the Gaussian-Process (GP) framework (Renard BHM). Models
// spatially correlated errors ε ~ MVN(0, Σ) with Σ_ij = σ²·ρ(h_ij). Parameter layout is
// [σ, correlation_params..., ε_1, ..., ε_n]:
//   - σ            : overall error scale (variance parameter)
//   - corr params  : the S1 correlation function's own parameters (Range for
//                    Exponential/Spherical, Range+Smoothness for PoweredExponential)
//   - ε_i          : one latent spatial error per site
//
// This class sits on top of both S1 (the correlation-model layer) and S2
// (CachedMultivariateNormal). It exposes TWO Cholesky paths, matching the C#:
//   (a) The CachedMVN path -- SetParameterValues rebuilds Σ and hands it to _mvn (S2), whose OWN
//       cached Cholesky powers LogPDF/PDF. That math is NOT duplicated here.
//   (b) This class's OWN escalating-jitter Cholesky (CholeskyDecomposition / TryCholesky /
//       ForwardSolve / BackwardSolve), used ONLY by GetKrigingPrediction. It tries the jitter
//       ladder {0, 1e-8, 1e-6, 1e-4, 1e-2}, adding the jitter UNIFORMLY to every diagonal so the
//       returned L is the true factor of A + jitter·I, and returns std::nullopt (C# null) when a
//       diagonal is non-positive. (See the C# remark carried on TryCholesky: the uniform-diagonal
//       jitter replaced an earlier single-entry perturbation that produced an L with L·Lᵀ ≠ A.)
//
// ErrorParameters aliasing: the C# stores the SAME ModelParameter references in both `Parameters`
// and `ErrorParameters`. C++ `std::vector<ModelParameter>` holds values, not references, so this
// port keeps a single owning `parameters_` vector and records the indices of the error entries in
// `error_indices_`. `error_parameters()` returns pointers INTO `parameters_`, so a write through
// either view is visible through the other -- the aliasing the C# relies on is preserved, and the
// object stays copy/move-safe (indices survive copies; raw ModelParameter pointers would not).
//
// DEVIATION (documented): the C# `Parameters.AddRange(_correlationFunction.Parameters)` shares the
// correlation function's ModelParameter references, so `Parameters[1..]` and the correlation
// function's parameter objects are identical. Here `parameters_` holds VALUE COPIES of those
// entries; SetParameterValues therefore writes the correlation values into BOTH `parameters_` and
// `_correlationFunction` (via SetParameterValues) to keep them consistent -- behavior identical to
// the C#, exactly as GaussianCopula (S2) already does for its delegated parameters.
//
// Ownership: `_correlationFunction` is a value-type held via unique_ptr (value-type + unique_ptr
// factory convention); `_mvn` is held by value and rebuilt (SetMean/SetCovariance) on every
// parameter change. The never-mutate rule is RELAXED for these model objects -- they mirror the
// C# mutable stateful API. The class is move-only (the unique_ptr member deletes implicit copy);
// deep copies go through Clone(), mirroring the C#.
//
// Deliberately NOT ported: XML / ToXElement / INPC -- none exist in this C# file (it carries no
// XML members at all). Tools.Distance is reused from corehydro::numerics::distance (added by S2),
// not re-added. The C# null-guards (coordinates, SetParameterValues values, kriging coordinates
// null) are vacuous here -- the ported signatures take std::vector<...> by const-ref, which cannot
// be null; only the dimension guards are meaningful and are ported.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/models/spatial_extremes/copula_models/cached_multivariate_normal.hpp"
#include "corehydro/models/spatial_extremes/spatial_correlation/basic_exponential.hpp"
#include "corehydro/models/spatial_extremes/spatial_correlation/correlation_function_type.hpp"
#include "corehydro/models/spatial_extremes/spatial_correlation/i_correlation_model.hpp"
#include "corehydro/models/spatial_extremes/spatial_correlation/powered_exponential.hpp"
#include "corehydro/models/spatial_extremes/spatial_correlation/spherical.hpp"
#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/models/support/subscript_formatter.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models::spatial_extremes {

class SpatialRegressionErrors {
   public:
    using Matrix2D = std::vector<std::vector<double>>;

    // Creates a new spatial regression errors model (C# ctor). The C# null guard on coordinates
    // is vacuous (vector cannot be null); the n×2 guard is ported. Selects the S1 correlation
    // impl by enum, sizes _mvn to Sites, precomputes the distance matrix, and sets defaults.
    SpatialRegressionErrors(const std::vector<std::vector<double>>& coordinates,
                            CorrelationFunctionType correlation_type, double max_error = 10)
        : coordinates_(coordinates),
          correlation_function_type_(correlation_type),
          mvn_(static_cast<int>(coordinates.size())) {
        std::size_t cols = coordinates.empty() ? 0 : coordinates[0].size();
        if (cols != 2) {
            throw std::invalid_argument("Coordinates must be n×2 array (X,Y) or (Lat,Lon).");
        }

        // Initialize correlation function.
        if (correlation_function_type_ == CorrelationFunctionType::Exponential)
            correlation_function_ = std::make_unique<BasicExponential>();
        else if (correlation_function_type_ == CorrelationFunctionType::PoweredExponential)
            correlation_function_ = std::make_unique<PoweredExponential>();
        else if (correlation_function_type_ == CorrelationFunctionType::Spherical)
            correlation_function_ = std::make_unique<Spherical>();

        // Precompute distance matrix.
        compute_distance_matrix();

        set_default_parameters(max_error);
    }

    // Gets the number of sites in the spatial model (C# Sites).
    int sites() const { return static_cast<int>(coordinates_.size()); }

    // Gets the spatial correlation function (C# CorrelationFunction).
    const ICorrelationModel& correlation_function() const { return *correlation_function_; }

    // Gets the list of all model parameters [σ, corr_params..., ε_1, ..., ε_n] (C# Parameters).
    // Mutable + const accessors (Clone writes through the mutable one).
    std::vector<ModelParameter>& parameters() { return parameters_; }
    const std::vector<ModelParameter>& parameters() const { return parameters_; }

    // Gets the error parameters [ε_1, ..., ε_n] as a view into Parameters (C# ErrorParameters).
    // Returns pointers into parameters_ so writes through either view alias each other (see
    // header). Two overloads by const-qualification.
    std::vector<ModelParameter*> error_parameters() {
        std::vector<ModelParameter*> result;
        result.reserve(error_indices_.size());
        for (std::size_t idx : error_indices_) result.push_back(&parameters_[idx]);
        return result;
    }
    std::vector<const ModelParameter*> error_parameters() const {
        std::vector<const ModelParameter*> result;
        result.reserve(error_indices_.size());
        for (std::size_t idx : error_indices_) result.push_back(&parameters_[idx]);
        return result;
    }

    // Gets the number of model parameters (C# NumberOfParameters => Parameters.Count).
    int number_of_parameters() const { return static_cast<int>(parameters_.size()); }

    // Gets the coordinates of all sites (C# Coordinates).
    const std::vector<std::vector<double>>& coordinates() const { return coordinates_; }

    // Gets the precomputed distance matrix (C# DistanceMatrix).
    const Matrix2D& distance_matrix() const { return distance_matrix_; }

    // Sets default parameters for the spatial error model (C# SetDefaultParameters). Rebuilds the
    // parameter list: σ, then a value-copy of each correlation-function parameter, then one latent
    // error per site (recorded in error_indices_ for the ErrorParameters alias).
    void set_default_parameters(double max_error) {
        parameters_.clear();
        error_indices_.clear();

        // Scale parameter (σ).
        parameters_.emplace_back(
            /*owner_name=*/"", /*name=*/"Error Scale (\xCF\x83)", /*value=*/max_error / 3.0,
            /*lower_bound=*/numerics::kDoubleMachineEpsilon, /*upper_bound=*/max_error,
            std::make_unique<numerics::distributions::Uniform>(numerics::kDoubleMachineEpsilon,
                                                               max_error),
            /*is_positive=*/true);

        // Correlation function parameters (value-copied; see header DEVIATION note).
        for (const ModelParameter& p : correlation_function_->parameters()) {
            parameters_.push_back(p);
        }

        // Latent error parameters (ε_i for each site).
        for (int i = 0; i < sites(); ++i) {
            parameters_.emplace_back(
                /*owner_name=*/"", /*name=*/std::string("\xCE\xB5") + to_subscript(i + 1),
                /*value=*/0.0, /*lower_bound=*/-max_error, /*upper_bound=*/max_error,
                std::make_unique<numerics::distributions::Uniform>(-max_error, max_error));
            error_indices_.push_back(parameters_.size() - 1);
        }
    }

    // Sets the model parameter values [σ, corr_params..., ε_1, ..., ε_n] (C# SetParameterValues).
    // The C# null guard is vacuous here; the count-mismatch guard mirrors the C# message.
    void set_parameter_values(const std::vector<double>& values) {
        if (values.size() != static_cast<std::size_t>(number_of_parameters())) {
            throw std::invalid_argument("Expected " + std::to_string(number_of_parameters()) +
                                        " parameters but got " + std::to_string(values.size()) +
                                        ".");
        }

        std::size_t index = 0;

        // Sigma parameter.
        parameters_[0].set_value(values[index]);
        double sigma = values[index];
        ++index;

        // Correlation parameters -- written into BOTH parameters_ and _correlationFunction (see
        // header DEVIATION note).
        std::vector<double> corr_params;
        for (int i = 0; i < correlation_function_->number_of_parameters(); ++i) {
            corr_params.push_back(values[index]);
            parameters_[index].set_value(values[index]);
            ++index;
        }
        correlation_function_->set_parameter_values(corr_params);

        // Error parameters. parameters_[index] IS error_parameters()[i] (aliased via
        // error_indices_), so the single write updates both views -- the C# writes both
        // Parameters[index] and ErrorParameters[i] to the same object.
        for (int i = 0; i < sites(); ++i) {
            parameters_[index].set_value(values[index]);
            ++index;
        }

        // Update covariance matrix: Σ_ij = σ² on the diagonal, else σ²·ρ(h_ij).
        int n = sites();
        Matrix2D cov_matrix(static_cast<std::size_t>(n),
                            std::vector<double>(static_cast<std::size_t>(n), 0.0));
        double sigma2 = sigma * sigma;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    cov_matrix[i][j] = sigma2;
                } else {
                    double h = distance_matrix_[i][j];
                    cov_matrix[i][j] = sigma2 * correlation_function_->evaluate(h);
                }
            }
        }

        std::vector<double> mean(static_cast<std::size_t>(n), 0.0);  // Zero mean.
        mvn_.set_mean(mean);
        mvn_.set_covariance(cov_matrix);
    }

    // Computes the probability density function of the spatial errors (C# PDF).
    double pdf() { return mvn_.pdf(current_errors()); }

    // Computes the log probability density function of the spatial errors (C# LogPDF).
    double log_pdf() { return mvn_.log_pdf(current_errors()); }

    // Gets the error value for a specific site (C# GetError). Range guard mirrors the C#
    // ArgumentOutOfRangeException.
    double get_error(int site_index) const {
        if (site_index < 0 || site_index >= sites()) {
            throw std::out_of_range("siteIndex");
        }
        return parameters_[error_indices_[static_cast<std::size_t>(site_index)]].value();
    }

    // Predicts the error at an ungauged location using simple kriging (conditional GP), C#
    // GetKrigingPrediction. Returns (Mean, Variance) as a std::pair mirroring the C# named tuple
    // field order. Falls back to IDW if the covariance Cholesky (path (b)) fails.
    std::pair<double, double> get_kriging_prediction(const std::vector<double>& new_coordinates) {
        if (new_coordinates.size() != 2) {
            throw std::invalid_argument("Coordinates must be a 2-element array [X, Y].");
        }

        double sigma = parameters_[0].value();
        double sigma2 = sigma * sigma;
        int n = sites();

        // Distances from the new location to all existing sites.
        std::vector<double> dist_to_new(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            dist_to_new[i] = numerics::distance(new_coordinates[0], new_coordinates[1],
                                                coordinates_[i][0], coordinates_[i][1]);
        }

        // k* (covariance between the new location and existing sites).
        std::vector<double> k_star(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            k_star[i] = sigma2 * correlation_function_->evaluate(dist_to_new[i]);
        }

        std::vector<double> errors = current_errors();

        // Build covariance matrix K.
        Matrix2D k_matrix(static_cast<std::size_t>(n),
                          std::vector<double>(static_cast<std::size_t>(n), 0.0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    k_matrix[i][j] = sigma2;
                } else {
                    k_matrix[i][j] = sigma2 * correlation_function_->evaluate(distance_matrix_[i][j]);
                }
            }
        }

        // Cholesky decomposition (path (b)) for numerical stability: K = L·Lᵀ.
        std::optional<Matrix2D> l = cholesky_decomposition(k_matrix);
        if (!l.has_value()) {
            // Fallback to IDW if Cholesky fails.
            return get_idw_prediction(new_coordinates);
        }

        // alpha = K⁻¹·errors via L·y = errors, Lᵀ·alpha = y.
        std::vector<double> y = forward_solve(*l, errors);
        std::vector<double> alpha = backward_solve(*l, y);

        // v = K⁻¹·k* via L·z = k*, Lᵀ·v = z.
        std::vector<double> z = forward_solve(*l, k_star);
        std::vector<double> v = backward_solve(*l, z);

        // Predicted mean: k*ᵀ·alpha.
        double pred_mean = 0.0;
        for (int i = 0; i < n; ++i) pred_mean += k_star[i] * alpha[i];

        // Predicted variance: k** - k*ᵀ·v = σ² - quadForm, clamped to non-negative.
        double k_star_star = sigma2;
        double quad_form = 0.0;
        for (int i = 0; i < n; ++i) quad_form += k_star[i] * v[i];
        double pred_var = std::max(k_star_star - quad_form, 0.0);

        return {pred_mean, pred_var};
    }

    // Predicts the error at an ungauged location using inverse-distance weighting (C#
    // GetIDWPrediction, power p=2). Returns (Mean, Variance) mirroring the C# tuple field order.
    std::pair<double, double> get_idw_prediction(const std::vector<double>& new_coordinates) {
        if (new_coordinates.size() != 2) {
            throw std::invalid_argument("Coordinates must be a 2-element array [X, Y].");
        }

        double sum_inv_dist = 0.0;
        double sum_weighted_error = 0.0;
        double sum_weighted_var = 0.0;
        double sigma2 = parameters_[0].value() * parameters_[0].value();

        for (int i = 0; i < sites(); ++i) {
            double dist = numerics::distance(new_coordinates[0], new_coordinates[1],
                                             coordinates_[i][0], coordinates_[i][1]);
            dist = std::max(dist, 1e-10);  // Avoid division by zero.

            double weight = 1.0 / (dist * dist);  // IDW with p=2.
            sum_inv_dist += weight;
            sum_weighted_error += weight * parameters_[error_indices_[static_cast<std::size_t>(i)]].value();
            sum_weighted_var += weight * weight * sigma2;
        }

        double pred_mean = sum_weighted_error / sum_inv_dist;
        double pred_var = sum_weighted_var / (sum_inv_dist * sum_inv_dist);

        return {pred_mean, pred_var};
    }

    // Returns a deep, independent copy of the spatial regression errors model (C# Clone).
    // Constructs a fresh model from the coordinates/type/upper-bound, copies each parameter's
    // Value/LowerBound/UpperBound and deep-clones its prior, then replays SetParameterValues so
    // the clone's MVN is rebuilt. Mutating the original afterward must not affect the clone.
    SpatialRegressionErrors clone() const {
        SpatialRegressionErrors result(coordinates_, correlation_function_type_,
                                       parameters_[0].upper_bound());

        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            result.parameters_[i].set_value(parameters_[i].value());
            result.parameters_[i].set_lower_bound(parameters_[i].lower_bound());
            result.parameters_[i].set_upper_bound(parameters_[i].upper_bound());
            // The C# `PriorDistribution is not null` guard is vacuous here -- every ModelParameter
            // ctor above populates a Uniform prior; cloned unconditionally.
            result.parameters_[i].set_prior_distribution(parameters_[i].prior_distribution().clone());
        }

        // Update the MVN with current values.
        std::vector<double> values;
        values.reserve(parameters_.size());
        for (const ModelParameter& p : parameters_) values.push_back(p.value());
        result.set_parameter_values(values);

        return result;
    }

   private:
    // Precomputes the distance matrix between all site pairs (C# ComputeDistanceMatrix): 0 on the
    // diagonal, else the Euclidean distance between coordinate rows i and j.
    void compute_distance_matrix() {
        int n = sites();
        distance_matrix_.assign(static_cast<std::size_t>(n),
                                std::vector<double>(static_cast<std::size_t>(n), 0.0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    distance_matrix_[i][j] = 0.0;
                } else {
                    distance_matrix_[i][j] = numerics::distance(
                        coordinates_[i][0], coordinates_[i][1], coordinates_[j][0],
                        coordinates_[j][1]);
                }
            }
        }
    }

    // Gathers the current ErrorParameters values into a vector (C# ErrorParameters.Select(...)).
    std::vector<double> current_errors() const {
        std::vector<double> errors;
        errors.reserve(error_indices_.size());
        for (std::size_t idx : error_indices_) errors.push_back(parameters_[idx].value());
        return errors;
    }

    // Performs Cholesky decomposition of a symmetric positive-definite matrix (C#
    // CholeskyDecomposition). If A is numerically indefinite, retries on A + δI for an increasing
    // uniform diagonal jitter δ ∈ {1e-8, 1e-6, 1e-4, 1e-2} so the returned L is the true factor of
    // the regularized matrix. Returns nullopt if all jitter levels fail.
    std::optional<Matrix2D> cholesky_decomposition(const Matrix2D& a) const {
        // Try unperturbed first, then escalating jitter on the diagonal.
        const double jitter_levels[] = {0.0, 1e-8, 1e-6, 1e-4, 1e-2};
        for (double jitter : jitter_levels) {
            std::optional<Matrix2D> l = try_cholesky(a, jitter);
            if (l.has_value()) return l;
        }
        return std::nullopt;
    }

    // Attempts a Cholesky factorization of A + jitter·I (C# TryCholesky). Returns nullopt if any
    // diagonal element is non-positive (matrix indefinite after the jitter).
    //
    // Remark carried from the C#: the original implementation perturbed only the failing diagonal
    // entry, producing an L for which L·Lᵀ ≠ A. The uniform-diagonal jitter here is the standard
    // "Cholesky with jitter" regularization and yields a valid factorization of A + δI at the cost
    // of a small bias proportional to δ.
    static std::optional<Matrix2D> try_cholesky(const Matrix2D& a, double jitter) {
        int n = static_cast<int>(a.size());
        Matrix2D l(static_cast<std::size_t>(n),
                   std::vector<double>(static_cast<std::size_t>(n), 0.0));

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                double sum = 0.0;
                for (int k = 0; k < j; ++k) sum += l[i][k] * l[j][k];

                if (i == j) {
                    // Add jitter uniformly to every diagonal so the resulting L is the true
                    // Cholesky factor of A + jitter·I (not a hybrid).
                    double diag = a[i][i] + jitter - sum;
                    if (diag <= 0) return std::nullopt;
                    l[i][j] = std::sqrt(diag);
                } else {
                    l[i][j] = (a[i][j] - sum) / l[j][j];
                }
            }
        }

        return l;
    }

    // Solves L·x = b for x using forward substitution (C# ForwardSolve).
    std::vector<double> forward_solve(const Matrix2D& l, const std::vector<double>& b) const {
        int n = static_cast<int>(b.size());
        std::vector<double> x(static_cast<std::size_t>(n), 0.0);
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < i; ++j) sum += l[i][j] * x[j];
            x[i] = (b[i] - sum) / l[i][i];
        }
        return x;
    }

    // Solves Lᵀ·x = b for x using backward substitution (C# BackwardSolve): Lᵀ[i,j] = L[j,i].
    std::vector<double> backward_solve(const Matrix2D& l, const std::vector<double>& b) const {
        int n = static_cast<int>(b.size());
        std::vector<double> x(static_cast<std::size_t>(n), 0.0);
        for (int i = n - 1; i >= 0; --i) {
            double sum = 0.0;
            for (int j = i + 1; j < n; ++j) sum += l[j][i] * x[j];  // Lᵀ[i,j] = L[j,i]
            x[i] = (b[i] - sum) / l[i][i];
        }
        return x;
    }

    std::vector<std::vector<double>> coordinates_;
    CorrelationFunctionType correlation_function_type_;
    std::unique_ptr<ICorrelationModel> correlation_function_;
    CachedMultivariateNormal mvn_;
    Matrix2D distance_matrix_;
    std::vector<ModelParameter> parameters_;
    std::vector<std::size_t> error_indices_;  // Indices of the ε_i entries within parameters_.
};

}  // namespace corehydro::models::spatial_extremes
