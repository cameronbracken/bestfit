// ported from: RMC-BestFit/src/RMC.BestFit/Models/SpatialExtremes/CopulaModels/GaussianCopula.cs @ fc28c0c
//
// Gaussian copula for modeling spatial dependence in extreme values. Transforms marginals to
// standard normal variables and applies a multivariate normal with correlation matrix R built
// from an S1 spatial correlation function (BasicExponential / PoweredExponential / Spherical).
//
// For observations with standard-normal transforms z_i, the copula density is
//   c(u) = phi_R(z) / prod phi(z_i)   (u_i = Phi(z_i)),
// evaluated through the CachedMultivariateNormal (S2) likelihood engine.
//
// Ownership: `_correlationFunction` is a value-type held via unique_ptr (value-type +
// unique_ptr factory convention); the enum selects the concrete S1 impl. `_mvn` is held by
// value and rebuilt (SetMean/SetCovariance) whenever the parameters change. The never-mutate
// rule is RELAXED for these model objects -- they mirror the C# mutable stateful API.
//
// Distance matrix: ComputeDistanceMatrix mirrors the C# 4-argument Tools.Distance call
// (Tools.cs:233) via the additive 4-arg numerics::distance(x1,y1,x2,y2) overload added to
// tools.hpp for this port (keeps the structural mirror). LogPDF uses Normal::standard_log_pdf,
// the additive helper added to normal.hpp for this port.
//
// Deliberately NOT ported (project-wide convention): XML / INPC plumbing. The C#
// ArgumentNullException null-guards (coordinates, SetParameterValues values, PDF/LogPDF input)
// are vacuous here -- the ported signatures take std::vector<...> by const-ref, which cannot be
// null; only the dimension guards are meaningful and are ported.
#pragma once
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "corehydro/models/spatial_extremes/copula_models/cached_multivariate_normal.hpp"
#include "corehydro/models/spatial_extremes/spatial_correlation/basic_exponential.hpp"
#include "corehydro/models/spatial_extremes/spatial_correlation/correlation_function_type.hpp"
#include "corehydro/models/spatial_extremes/spatial_correlation/i_correlation_model.hpp"
#include "corehydro/models/spatial_extremes/spatial_correlation/powered_exponential.hpp"
#include "corehydro/models/spatial_extremes/spatial_correlation/spherical.hpp"
#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models::spatial_extremes {

class GaussianCopula {
   public:
    // Creates a new Gaussian copula for spatial dependence modeling (C# ctor). The C# null
    // guard on coordinates is vacuous (vector cannot be null); the n×2 guard is ported. Selects
    // the S1 correlation impl by enum, sizes _mvn to Sites, and precomputes the distance matrix.
    GaussianCopula(const std::vector<std::vector<double>>& coordinates,
                   CorrelationFunctionType correlation_type)
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
    }

    // Gets the number of sites in the spatial model (C# Sites).
    int sites() const { return static_cast<int>(coordinates_.size()); }

    // Gets the spatial correlation function (C# CorrelationFunction).
    const ICorrelationModel& correlation_function() const { return *correlation_function_; }

    // Gets the list of correlation function parameters (C# Parameters) -- delegates to the
    // correlation function. Mutable + const accessors (Clone writes through the mutable one).
    std::vector<ModelParameter>& parameters() { return correlation_function_->parameters(); }
    const std::vector<ModelParameter>& parameters() const {
        return correlation_function_->parameters();
    }

    // Gets the number of model parameters (C# NumberOfParameters).
    int number_of_parameters() const { return static_cast<int>(parameters().size()); }

    // Sets the correlation function parameter values (C# SetParameterValues): pushes into the
    // correlation function, rebuilds R (R_ij = 1 on the diagonal, else rho(h_ij)), and pushes a
    // zero mean + R into _mvn.
    void set_parameter_values(const std::vector<double>& values) {
        correlation_function_->set_parameter_values(values);

        int n = sites();
        std::vector<std::vector<double>> corr_matrix(
            static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(n), 0.0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    corr_matrix[i][j] = 1.0;
                } else {
                    double h = distance_matrix_[i][j];
                    corr_matrix[i][j] = correlation_function_->evaluate(h);
                }
            }
        }

        std::vector<double> mean(static_cast<std::size_t>(n), 0.0);  // Zero mean.
        mvn_.set_mean(mean);
        mvn_.set_covariance(corr_matrix);
    }

    // Computes the copula probability density function (C# PDF): phi_R(z) / prod phi(z_i).
    double pdf(const std::vector<double>& z) {
        if (static_cast<int>(z.size()) != sites()) {
            throw std::invalid_argument("Input vector dimension mismatch.");
        }

        double numerator = mvn_.pdf(z);
        double denominator = 1.0;
        for (std::size_t i = 0; i < z.size(); ++i)
            denominator *= numerics::distributions::Normal::standard_pdf(z[i]);

        if (denominator <= 0) return 0.0;

        return numerator / denominator;
    }

    // Computes the log copula probability density function (C# LogPDF):
    // log phi_R(z) - sum log phi(z_i).
    double log_pdf(const std::vector<double>& z) {
        if (static_cast<int>(z.size()) != sites()) {
            throw std::invalid_argument("Input vector dimension mismatch.");
        }

        double numerator = mvn_.log_pdf(z);
        if (numerator == -std::numeric_limits<double>::infinity())
            return -std::numeric_limits<double>::infinity();

        double denominator = 0.0;
        for (std::size_t i = 0; i < z.size(); ++i)
            denominator += numerics::distributions::Normal::standard_log_pdf(z[i]);

        return numerator - denominator;
    }

    // Returns a deep, independent copy of the Gaussian copula (C# Clone). Constructs a fresh
    // copula, copies each parameter's Value/LowerBound/UpperBound and deep-clones its prior,
    // then pushes the current values through the clone's SetParameterValues so its _mvn is
    // rebuilt. Mutating the original afterward must not affect the clone.
    GaussianCopula clone() const {
        GaussianCopula result(coordinates_, correlation_function_type_);

        const std::vector<ModelParameter>& src = parameters();
        std::vector<ModelParameter>& dst = result.parameters();
        for (std::size_t i = 0; i < src.size(); ++i) {
            dst[i].set_value(src[i].value());
            dst[i].set_lower_bound(src[i].lower_bound());
            dst[i].set_upper_bound(src[i].upper_bound());
            // The C# `PriorDistribution is not null` guard is vacuous for the S1 correlation
            // models (their ctors always populate a Uniform prior); cloned unconditionally.
            dst[i].set_prior_distribution(src[i].prior_distribution().clone());
        }

        std::vector<double> values;
        values.reserve(src.size());
        for (const ModelParameter& p : src) values.push_back(p.value());
        result.set_parameter_values(values);

        return result;
    }

   private:
    // Precomputes the distance matrix between all site pairs (C# ComputeDistanceMatrix):
    // 0 on the diagonal, else the Euclidean distance between coordinate rows i and j.
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

    std::vector<std::vector<double>> coordinates_;
    CorrelationFunctionType correlation_function_type_;
    std::unique_ptr<ICorrelationModel> correlation_function_;
    CachedMultivariateNormal mvn_;
    std::vector<std::vector<double>> distance_matrix_;
};

}  // namespace corehydro::models::spatial_extremes
