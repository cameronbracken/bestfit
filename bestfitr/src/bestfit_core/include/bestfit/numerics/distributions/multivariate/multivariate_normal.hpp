// ported from: Numerics/Distributions/Multivariate/MultivariateNormal.cs @ a2c4dbf
//
// The Multivariate Normal (Gaussian) distribution, dimensions >= 1, mean vector mu and
// covariance matrix Sigma. This is the phase's heavyweight port (~2150 LOC of C#): the
// deterministic core (ctors, PDF/LogPDF/CDF for dim 1-2, Mahalanobis distance, sampling,
// the closed-form bivariate CDF machinery) lands first; the Genz-Bretz MVNDST
// quasi-Monte-Carlo integrator that backs the CDF for dim >= 3 lands in a second commit
// (this header's `mvndst()` is a throwing stub until then -- see the comment on that
// method). Member order mirrors the C# source throughout.
//
// Divergence notes (see also docs/upstream-csharp-issues.md for the running log):
//   - `Variance`/`StandardDeviation` are computed on every call rather than lazily cached
//     in nullable backing fields (the C# `_variance`/`_standardDeviation` pattern) --
//     same simplification and rationale as the Dirichlet port (dirichlet.hpp): the C#
//     caching is a micro-optimization orthogonal to the math, and this port's "never
//     mutate a value type without reason" convention prefers recomputation. Results are
//     identical.
//   - `CreateCorrelationMatrix`'s `invSqrtD = !D` (C# `Matrix.operator!` = a full
//     LU-decomposition-based inverse, not yet ported -- see matrix.hpp's header note) is
//     implemented here as an elementwise reciprocal of the diagonal instead. `D` is
//     provably diagonal at that call site (built via `Matrix::diagonal` + elementwise
//     `sqrt`), and the LU inverse of a diagonal matrix reduces exactly to elementwise
//     reciprocal (L=I, U=D, so Ainv=diag(1/d_ii) with no extra rounding) -- bit-identical
//     to the general algorithm for this diagonal input, without porting the general
//     LU-based `Matrix::inverse()` this port doesn't otherwise need.
//   - `clone()`'s `MVNUNI` copy is a VALUE copy (an independent RNG stream) rather than
//     the C# `_MVNUNI = this._MVNUNI` reference-sharing semantics (original and clone
//     alias the SAME `Random`/stream in C#, so later draws on either object consume from
//     one shared sequence). `Clone()` is not fixture-observable (mirrors the precedent
//     set for Dirichlet/Multinomial/BivariateEmpirical's Clone in fixtures/README.md),
//     and an independent stream is the safer, more intuitive choice for a value-oriented
//     C++ port.
//   - `ParametersValid`/`_parametersValid` and `_covSRTed` are ported verbatim as fields
//     but are DEAD STATE in the C# source: `_parametersValid` is initialized `true` and
//     never reassigned anywhere in the class (`SetParameters` throws on invalid input
//     rather than flipping the flag, so any live instance reports valid), and
//     `_covSRTed` is declared, copied by `Clone()`, and never read or written anywhere
//     else. Kept for structural fidelity; both are effectively constant for this class.
//   - The C# null-checks on the `mean`/`covariance` constructor arguments
//     (`ArgumentOutOfRangeException` on `null`) have no C++ equivalent (references/
//     vectors cannot be null) and are omitted from `validate_parameters`.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "bestfit/numerics/distributions/multivariate/base/multivariate_distribution.hpp"
#include "bestfit/numerics/distributions/multivariate/base/multivariate_distribution_type.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "bestfit/numerics/math/linalg/cholesky_decomposition.hpp"
#include "bestfit/numerics/math/linalg/matrix.hpp"
#include "bestfit/numerics/math/linalg/vector.hpp"
#include "bestfit/numerics/math/special/erf.hpp"
#include "bestfit/numerics/sampling/latin_hypercube.hpp"
#include "bestfit/numerics/sampling/mersenne_twister.hpp"
#include "bestfit/numerics/sampling/stratification_bin.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::distributions {

namespace la = bestfit::numerics::math::linalg;
namespace sf = bestfit::numerics::math::special;

class MultivariateNormal : public MultivariateDistribution {
   public:
    // Constructs a multivariate Gaussian with zero mean vector and identity covariance.
    explicit MultivariateNormal(int dimension) {
        std::vector<double> mean(static_cast<std::size_t>(dimension), 0.0);
        set_parameters(std::move(mean), la::Matrix::identity(dimension).to_array());
    }

    // Constructs a new Multivariate Normal distribution with an identity covariance matrix.
    explicit MultivariateNormal(std::vector<double> mean) {
        int dim = static_cast<int>(mean.size());
        set_parameters(std::move(mean), la::Matrix::identity(dim).to_array());
    }

    // Constructs a new Multivariate Normal distribution.
    MultivariateNormal(std::vector<double> mean, std::vector<std::vector<double>> covariance) {
        set_parameters(std::move(mean), std::move(covariance));
    }

    // --- MVNUNI: the uniform(0,1) generator required to compute the multivariate CDF for
    // dimensions greater than 2 (C# `public Random MVNUNI { get; set; }`). This port has
    // no generic RNG interface (every ported sampling routine takes a concrete
    // MersenneTwister -- see multinomial.hpp/dirichlet.hpp for the same choice), so the
    // property is exposed as a seed-setter mirroring how the C# tests assign it
    // (`MVNUNI = new MersenneTwister(12345)`), plus direct access for MVNDST's internals.
    void set_mvnuni_seed(int seed) { mvnuni_ = sampling::MersenneTwister(static_cast<std::uint32_t>(seed)); }
    sampling::MersenneTwister& mvnuni() const { return mvnuni_; }

    // The maximum number of function evaluations allowed when computing the multivariate
    // CDF. Default = 1,000 x D.
    int max_evaluations() const { return max_evaluations_; }
    void set_max_evaluations(int value) { max_evaluations_ = value; }

    // The absolute error tolerance when computing the multivariate CDF.
    double absolute_error() const { return absolute_error_; }
    void set_absolute_error(double value) { absolute_error_ = value; }

    // The relative error tolerance when computing the multivariate CDF.
    double relative_error() const { return relative_error_; }
    void set_relative_error(double value) { relative_error_ = value; }

    // --- Identity / parameters ---
    int dimension() const override { return dimension_; }
    MultivariateDistributionType type() const override {
        return MultivariateDistributionType::MultivariateNormal;
    }
    std::string display_name() const override { return "Multivariate Normal"; }
    std::string short_display_name() const override { return "Multi-N"; }
    bool parameters_valid() const override { return parameters_valid_; }

    // Gets the mean vector of the distribution.
    const std::vector<double>& mean() const { return mean_; }

    // Gets the median vector of the distribution (== mean).
    const std::vector<double>& median() const { return mean_; }

    // Gets the mode vector of the distribution (== mean).
    const std::vector<double>& mode() const { return mean_; }

    // Gets the variance vector of the distribution (computed each call -- see file header).
    std::vector<double> variance() const { return covariance_.diagonal(); }

    // Gets the standard deviation vector of the distribution (computed each call).
    std::vector<double> standard_deviation() const {
        std::vector<double> v = variance();
        for (double& x : v) x = std::sqrt(x);
        return v;
    }

    // Gets the Variance-Covariance matrix for the distribution.
    la::Matrix2D covariance() const { return covariance_.to_array(); }

    // Element accessor (not in the C# source -- added for fixture/dispatch convenience,
    // mirroring Dirichlet's/Multinomial's `covariance(i, j)` method).
    double covariance(int i, int j) const { return covariance_(i, j); }

    // Determines if the covariance matrix is positive definite.
    bool is_positive_definite() const { return cholesky_->is_positive_definite(); }

    // --- Parameter setting / validation ---

    // Set the distribution parameters.
    void set_parameters(std::vector<double> mean, std::vector<std::vector<double>> covariance) {
        // Validate parameters
        validate_parameters(mean, covariance, true);

        dimension_ = static_cast<int>(mean.size());
        mean_ = std::move(mean);
        covariance_ = la::Matrix(std::move(covariance));
        cholesky_.emplace(covariance_);
        double lndet = cholesky_->log_determinant();
        lnconstant_ = -(std::log(2.0 * kPi) * static_cast<double>(mean_.size()) + lndet) * 0.5;

        // Set up parameters for MVN CDF
        correlation_matrix_created_ = false;
        max_evaluations_ = 1000 * dimension_;
        lower_.assign(static_cast<std::size_t>(dimension_), -kInf);
        upper_.assign(static_cast<std::size_t>(dimension_), 0.0);
        infin_.assign(static_cast<std::size_t>(dimension_), 0);
        // NL reassignment quirk (MultivariateNormal.cs:254): NL defaults to 500 (see the
        // field declaration below, sized to match the C# Fortran-derived MVNDFN_A/B/INFI
        // working-storage arrays added in the second commit) but is REASSIGNED here, per
        // instance, to the number of below-diagonal correlation pairs. Only `_correl`'s
        // allocation and (in the second commit) `MVNDFN`'s local `Y` scratch array use
        // the reassigned value -- the fixed-500-sized working arrays are untouched by it.
        nl_ = static_cast<int>(dimension_ * (dimension_ - 1) / 2.0);
        // Create empty arrays
        correl_.assign(static_cast<std::size_t>(nl_), 0.0);
        correlation_ = la::Matrix(dimension_, dimension_);
    }

    // Validate the parameters. Returns nullopt if valid; if `throw_exception` is false and
    // invalid, returns the error message instead of throwing.
    std::optional<std::string> validate_parameters(const std::vector<double>& mean,
                                                     const std::vector<std::vector<double>>& covariance,
                                                     bool throw_exception) const {
        la::Matrix m(covariance);
        if (!m.is_square()) {
            if (throw_exception) throw std::out_of_range("Covariance matrix must be square.");
            return "Covariance matrix must be square.";
        }
        if (m.number_of_rows() != static_cast<int>(mean.size())) {
            if (throw_exception) throw std::out_of_range("Mean length must match covariance dimension.");
            return "Mean length must match covariance dimension.";
        }
        // Cholesky try/propagate quirk: the C# source constructs `new
        // CholeskyDecomposition(m)` directly here, with no try/catch. This port's
        // CholeskyDecomposition ctor (like the C# one) unconditionally THROWS on a
        // non-positive-definite or NaN-diagonal input rather than returning a status (see
        // cholesky_decomposition.hpp) -- so that exception always propagates out of
        // validate_parameters regardless of `throw_exception`, and the
        // `!chol.is_positive_definite()` check below never actually fires (construction
        // already succeeded by the time it runs). Ported verbatim anyway for structural
        // fidelity with the C# source.
        la::CholeskyDecomposition chol(m);
        if (!chol.is_positive_definite()) {
            if (throw_exception) throw std::out_of_range("Covariance matrix is not positive-definite.");
            return "Covariance matrix is not positive-definite.";
        }
        return std::nullopt;
    }

    // --- Distribution functions ---

    // The Probability Density Function (PDF) of the distribution evaluated at a point X.
    double pdf(const std::vector<double>& x) const override {
        return std::exp(-0.5 * mahalanobis(x) + lnconstant_);
    }

    // Returns the natural log of the PDF.
    double log_pdf(const std::vector<double>& x) const override {
        double f = -0.5 * mahalanobis(x) + lnconstant_;
        if (std::isnan(f) || std::isinf(f)) return -kInf;
        return f;
    }

    // Gets the Mahalanobis distance between a sample and this distribution.
    double mahalanobis(const std::vector<double>& x) const {
        if (x.size() != mean_.size())
            throw std::out_of_range("The vector must be the same dimension as the distribution.");
        std::vector<double> z(mean_.size());
        for (std::size_t i = 0; i < x.size(); ++i) z[i] = x[i] - mean_[i];
        la::Vector a = cholesky_->solve(la::Vector(z));
        double b = 0.0;
        for (std::size_t i = 0; i < z.size(); ++i) b += a[static_cast<int>(i)] * z[i];
        return b;
    }

    // The Cumulative Distribution Function (CDF) for the distribution evaluated at a point X.
    double cdf(const std::vector<double>& x) const override {
        if (dimension_ == 1) {
            double std_dev = std::sqrt(covariance_(0, 0));
            if (std_dev == 0.0) return x[0] >= mean_[0] ? 1.0 : 0.0;
            double z = (x[0] - mean_[0]) / std_dev;
            return Normal::standard_cdf(z);
        } else if (dimension_ == 2) {
            double sigma1 = std::sqrt(covariance_(0, 0));
            double sigma2 = std::sqrt(covariance_(1, 1));
            double rho = covariance_(0, 1) / (sigma1 * sigma2);
            if (std::isnan(rho)) {
                bool at_mean = x.size() == 2 && x[0] == mean_[0] && x[1] == mean_[1];
                return at_mean ? 1.0 : 0.0;
            }
            double z = (x[0] - mean_[0]) / sigma1;
            double w = (x[1] - mean_[1]) / sigma2;
            return bivariate_cdf(-z, -w, rho);
        } else {
            // Construct inputs for the GenzBretz method
            for (std::size_t i = 0; i < x.size(); ++i)
                upper_[i] = (x[i] - mean_[i]) / std::sqrt(covariance_(static_cast<int>(i), static_cast<int>(i)));

            double error = 0, val = 0;
            int inform = 0;

            if (!correlation_matrix_created_) create_correlation_matrix();
            mvndst(dimension_, lower_, upper_, infin_, correl_, max_evaluations_, absolute_error_,
                   relative_error_, error, val, inform);
            return val;
        }
    }

    // Computes the probability of a value occurring in the interval.
    double interval(std::vector<double> lower, std::vector<double> upper) const {
        std::vector<int> infin(upper.size(), 2);
        for (std::size_t i = 0; i < upper.size(); ++i) {
            lower[i] = (lower[i] - mean_[i]) / std::sqrt(covariance_(static_cast<int>(i), static_cast<int>(i)));
            upper[i] = (upper[i] - mean_[i]) / std::sqrt(covariance_(static_cast<int>(i), static_cast<int>(i)));
        }

        double error = 0, val = 0;
        int inform = 0;

        if (!correlation_matrix_created_) create_correlation_matrix();
        mvndst(dimension_, lower, upper, infin, correl_, max_evaluations_, absolute_error_, relative_error_,
               error, val, inform);
        return val;
    }

    // The inverse cumulative distribution function (InverseCDF).
    std::vector<double> inverse_cdf(const std::vector<double>& probabilities) const {
        std::vector<double> sample(static_cast<std::size_t>(dimension_));
        // Create vector z of standard normal variates for each dimension
        std::vector<double> z(static_cast<std::size_t>(dimension_));
        for (int j = 0; j < dimension_; ++j) z[static_cast<std::size_t>(j)] = Normal::standard_z(probabilities[static_cast<std::size_t>(j)]);
        // x = A*z + mu
        la::Vector Az = cholesky_->l() * la::Vector(z);
        for (int j = 0; j < dimension_; ++j) sample[static_cast<std::size_t>(j)] = Az[j] + mean_[static_cast<std::size_t>(j)];
        return sample;
    }

    // Returns an independent univariate Normal distribution for the given index.
    Normal independent_normal(int index) const {
        return Normal(mean_[static_cast<std::size_t>(index)], standard_deviation()[static_cast<std::size_t>(index)]);
    }

    // Returns a new univariate Normal distribution.
    static MultivariateNormal univariate(double mu, double sigma) {
        return MultivariateNormal(std::vector<double>{mu}, std::vector<std::vector<double>>{{sigma * sigma}});
    }

    // Returns a new bivariate Normal distribution.
    static MultivariateNormal bivariate(double mu1, double mu2, double sigma1, double sigma2, double rho) {
        std::vector<double> mu = {mu1, mu2};
        std::vector<std::vector<double>> covariance = {{sigma1 * sigma1, sigma1 * sigma2 * rho},
                                                         {sigma1 * sigma2 * rho, sigma2 * sigma2}};
        return MultivariateNormal(std::move(mu), std::move(covariance));
    }

    // Generate random values of a distribution given a sample size. seed<=0 seeds from a
    // clock (no reproducible draw; see make_master()).
    std::vector<std::vector<double>> generate_random_values(int sample_size, int seed = -1) const {
        sampling::MersenneTwister rng = make_master(seed);
        std::vector<std::vector<double>> sample(static_cast<std::size_t>(sample_size),
                                                  std::vector<double>(static_cast<std::size_t>(dimension_)));
        for (int i = 0; i < sample_size; ++i) {
            std::vector<double> z(static_cast<std::size_t>(dimension_));
            for (int j = 0; j < dimension_; ++j) z[static_cast<std::size_t>(j)] = Normal::standard_z(rng.next_double());
            la::Vector Az = cholesky_->l() * la::Vector(z);
            for (int j = 0; j < dimension_; ++j)
                sample[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = Az[j] + mean_[static_cast<std::size_t>(j)];
        }
        return sample;
    }

    // Use Latin hypercube method to generate random values of a distribution given a
    // sample size and a user-defined seed.
    std::vector<std::vector<double>> latin_hypercube_random_values(int sample_size, int seed) const {
        la::Matrix2D r = sampling::LatinHypercube::random(sample_size, dimension_, seed);
        std::vector<std::vector<double>> sample(static_cast<std::size_t>(sample_size),
                                                  std::vector<double>(static_cast<std::size_t>(dimension_)));
        for (int i = 0; i < sample_size; ++i) {
            std::vector<double> z(static_cast<std::size_t>(dimension_));
            for (int j = 0; j < dimension_; ++j)
                z[static_cast<std::size_t>(j)] = Normal::standard_z(r[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
            la::Vector Az = cholesky_->l() * la::Vector(z);
            for (int j = 0; j < dimension_; ++j)
                sample[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = Az[j] + mean_[static_cast<std::size_t>(j)];
        }
        return sample;
    }

    // Returns a 2D array of stratified z-variates. The first dimension is stratified, and
    // the remaining are sampled randomly.
    std::vector<std::vector<double>> stratified_random_values(
        const std::vector<sampling::StratificationBin>& stratification_bins, int seed) const {
        int sample_size = static_cast<int>(stratification_bins.size());
        // Matches the C# source exactly: always seeds directly from `seed` (no clock
        // fallback for seed<=0, unlike generate_random_values/latin_hypercube_random_values).
        sampling::MersenneTwister r(static_cast<std::uint32_t>(seed));
        std::vector<std::vector<double>> sample(static_cast<std::size_t>(sample_size),
                                                  std::vector<double>(static_cast<std::size_t>(dimension_)));
        for (int i = 0; i < sample_size; ++i) {
            std::vector<double> z(static_cast<std::size_t>(dimension_));
            for (int j = 0; j < dimension_; ++j) {
                if (j == 0) {
                    z[0] = Normal::standard_z(stratification_bins[static_cast<std::size_t>(i)].midpoint());
                } else {
                    z[static_cast<std::size_t>(j)] = Normal::standard_z(r.next_double());
                }
            }
            la::Vector Az = cholesky_->l() * la::Vector(z);
            for (int j = 0; j < dimension_; ++j)
                sample[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = Az[j] + mean_[static_cast<std::size_t>(j)];
        }
        return sample;
    }

    // ========================================================================================
    // Cumulative Distribution Support
    // ========================================================================================

    // Computes the bivariate normal CDF: the probability for two normal variates X and Y
    // whose correlation is R, that AH <= X and AK <= Y.
    //
    // Original FORTRAN77 by Thomas Donnelly; C++ (Numerics' C#, ported here) by John
    // Burkardt. NOTE: this method is declared `public static` in the C# source but is not
    // called from anywhere else in the class (or exercised by any upstream unit test) --
    // `bivariate_cdf`/`bvu`/`mvnphi` are what CDF()/MVNDST actually use. Ported verbatim
    // for API completeness per the task brief; not exercised by any fixture (no oracle
    // value exists to scrape).
    static double bivnor(double ah, double ak, double r) {
        double a2;
        double ap;
        double b;
        double cn;
        double con;
        double conex;
        double ex;
        double g2;
        double gh;
        double gk;
        double gw = 0;
        double h2;
        double h4;
        int i;
        int idig = 15;
        int i5 = 0;
        double rr;
        double s1;
        double s2;
        double sgn;
        double sn;
        double sp;
        double sqr;
        double t;
        double twopi = 6.283185307179587;
        double w2 = 0;
        double wh = 0;
        double wk = 0;

        b = 0.0;

        gh = gauss(-ah) / 2.0;
        gk = gauss(-ak) / 2.0;

        if (r == 0.0) {
            b = 4.00 * gh * gk;
            b = std::max(b, 0.0);
            b = std::min(b, 1.0);
            return b;
        }

        rr = (1.0 + r) * (1.0 - r);

        // (rr < 0.0 in the C# source is a fatal-error branch whose body is entirely
        // commented out -- nothing to port.)

        if (rr == 0.0) {
            if (r < 0.0) {
                if (ah + ak < 0.0) {
                    b = 2.0 * (gh + gk) - 1.0;
                }
            } else {
                if (ah - ak < 0.0) {
                    b = 2.0 * gk;
                } else {
                    b = 2.0 * gh;
                }
            }
            b = std::max(b, 0.0);
            b = std::min(b, 1.0);
            return b;
        }

        sqr = std::sqrt(rr);

        if (idig == 15) {
            con = twopi * 1.0E-15 / 2.0;
        } else {
            con = twopi / 2.0;
            for (i = 1; i <= idig; i++) {
                con = con / 10.0;
            }
        }
        // (0,0)
        if (ah == 0.0 && ak == 0.0) {
            b = 0.25 + std::asin(r) / twopi;
            b = std::max(b, 0.0);
            b = std::min(b, 1.0);
            return b;
        }
        // (0,nonzero)
        if (ah == 0.0 && ak != 0.0) {
            b = gk;
            wh = -ak;
            wk = (ah / ak - r) / sqr;
            gw = 2.0 * gk;
            i5 = 1;
        }
        // (nonzero,0)
        else if (ah != 0.0 && ak == 0.0) {
            b = gh;
            wh = -ah;
            wk = (ak / ah - r) / sqr;
            gw = 2.0 * gh;
            i5 = -1;
        }
        // (nonzero,nonzero)
        else if (ah != 0.0 && ak != 0.0) {
            b = gh + gk;
            if (ah * ak < 0.0) {
                b = b - 0.5;
            }
            wh = -ah;
            wk = (ak / ah - r) / sqr;
            gw = 2.0 * gh;
            i5 = -1;
        }

        for (;;) {
            sgn = -1.0;
            t = 0.0;

            if (wk != 0.0) {
                if (std::fabs(wk) == 1.0) {
                    t = wk * gw * (1.0 - gw) / 2.0;
                    b = b + sgn * t;
                } else {
                    if (1.0 < std::fabs(wk)) {
                        sgn = -sgn;
                        wh = wh * wk;
                        g2 = gauss(wh);
                        wk = 1.0 / wk;

                        if (wk < 0.0) {
                            b = b + 0.5;
                        }
                        b = b - (gw + g2) / 2.0 + gw * g2;
                    }
                    h2 = wh * wh;
                    a2 = wk * wk;
                    h4 = h2 / 2.0;
                    ex = std::exp(-h4);
                    w2 = h4 * ex;
                    ap = 1.0;
                    s2 = ap - ex;
                    sp = ap;
                    s1 = 0.0;
                    sn = s1;
                    conex = std::fabs(con / wk);

                    for (;;) {
                        cn = ap * s2 / (sn + sp);
                        s1 = s1 + cn;

                        if (std::fabs(cn) <= conex) {
                            break;
                        }
                        sn = sp;
                        sp = sp + 1.0;
                        s2 = s2 - w2;
                        w2 = w2 * h4 / sp;
                        ap = -ap * a2;
                    }
                    t = (std::atan(wk) - wk * s1) / twopi;
                    b = b + sgn * t;
                }
            }
            if (0 <= i5) {
                break;
            }
            if (ak == 0.0) {
                break;
            }
            wh = -ak;
            wk = (ah / ak - r) / sqr;
            gw = 2.0 * gk;
            i5 = 1;
        }

        b = std::max(b, 0.0);
        b = std::min(b, 1.0);

        return b;
    }

    // A function for computing the bivariate normal CDF.
    //
    // This code was copied from the Accord Math Library, http://accord-framework.net,
    // based on work by Alan Genz, Dept. of Mathematics, Washington State University
    // (3-clause BSD license; see the upstream C# source for the full license text). Based
    // on Drezner & Wesolowsky (1989), "On the computation of the bivariate normal
    // integral", with modifications for double precision and for |R| close to 1.
    static double bivariate_cdf(double z1, double z2, double r) {
        const double TWOPI = 2.0 * kPi;
        const double* x;
        const double* w;
        int n;
        if (std::fabs(r) < 0.3) {
            x = kBvndXn6;
            w = kBvndWn6;
            n = 3;
        } else if (std::fabs(r) < 0.75) {
            x = kBvndXn12;
            w = kBvndWn12;
            n = 6;
        } else {
            x = kBvndXn20;
            w = kBvndWn20;
            n = 10;
        }

        double h = z1;
        double k = z2;
        double hk = h * k;
        double bvn = 0.0;
        if (std::fabs(r) < 0.925) {
            if (std::fabs(r) > 0.0) {
                double sh = (h * h + k * k) / 2.0;
                double asr = std::asin(r);
                for (int i = 0; i < n; i++) {
                    for (int j = -1; j <= 1; j += 2) {
                        double sn = std::sin(asr * (j * x[i] + 1.0) / 2.0);
                        bvn = bvn + w[i] * std::exp((sn * hk - sh) / (1.0 - sn * sn));
                    }
                }

                bvn = bvn * asr / (2.0 * TWOPI);
            }

            return bvn + mvnphi(-h) * mvnphi(-k);
        }

        if (r < 0.0) {
            k = -k;
            hk = -hk;
        }

        if (std::fabs(r) < 1.0) {
            double sa = (1.0 - r) * (1.0 + r);
            double A = std::sqrt(sa);
            double sb = h - k;
            sb = sb * sb;
            double c = (4.0 - hk) / 8.0;
            double d = (12.0 - hk) / 16.0;
            double asr = -(sb / sa + hk) / 2.0;
            if (asr > -100)
                bvn = A * std::exp(asr) * (1.0 - c * (sb - sa) * (1.0 - d * sb / 5.0) / 3.0 + c * d * sa * sa / 5.0);
            if (-hk < 100.0) {
                double B = std::sqrt(sb);
                bvn = bvn - std::exp(-hk / 2.0) * std::sqrt(TWOPI) * mvnphi(-B / A) * B *
                                (1.0 - c * sb * (1.0 - d * sb / 5.0) / 3.0);
            }

            A = A / 2.0;
            for (int i = 0; i < n; i++) {
                for (int j = -1; j <= 1; j += 2) {
                    double xs = A * (j * x[i] + 1.0);
                    xs = xs * xs;
                    double rs = std::sqrt(1.0 - xs);
                    asr = -(sb / xs + hk) / 2.0;
                    if (asr > -100) {
                        bvn = bvn + A * w[i] * std::exp(asr) *
                                        (std::exp(-hk * xs / (2.0 * (1.0 + rs) * (1.0 + rs))) / rs -
                                         (1.0 + c * xs * (1.0 + d * xs)));
                    }
                }
            }

            bvn = -bvn / TWOPI;
        }

        if (r > 0.0) return bvn + mvnphi(-std::max(h, k));
        bvn = -bvn;
        if (k <= h) return bvn;
        if (h < 0.0) return bvn + mvnphi(k) - mvnphi(h);
        return bvn + mvnphi(-h) - mvnphi(-k);
    }

    // Computes the standard normal CDF, accurate to 1E-15.
    // Reference: J.L. Schonfelder, Math Comp 32(1978), pp 1232-1240.
    static double mvnphi(double Z) {
        static constexpr double kMvnphiA[44] = {
            6.10143081923200417926465815756E-1,   -4.34841272712577471828182820888E-1,
            1.76351193643605501125840298123E-1,   -6.0710795609249414860051215825E-2,
            1.7712068995694114486147141191E-2,    -4.321119385567293818599864968E-3,
            8.54216676887098678819832055E-4,      -1.27155090609162742628893940E-4,
            1.1248167243671189468847072E-5,       3.13063885421820972630152E-7,
            -2.70988068537762022009086E-7,        3.0737622701407688440959E-8,
            2.515620384817622937314E-9,           -1.028929921320319127590E-9,
            2.9944052119949939363E-11,            2.6051789687266936290E-11,
            -2.634839924171969386E-12,            -6.43404509890636443E-13,
            1.12457401801663447E-13,              1.7281533389986098E-14,
            -4.264101694942375E-15,               -5.45371977880191E-16,
            1.58697607761671E-16,                 2.0899837844334E-17,
            -5.900526869409E-18,                  -9.41893387554E-19,
            2.14977356470E-19,                    4.6660985008E-20,
            -7.243011862E-21,                     -2.387966824E-21,
            1.91177535E-22,                       1.20482568E-22,
            -6.72377E-25,                         -5.747997E-24,
            -4.28493E-25,                         2.44856E-25,
            4.3793E-26,                            -8.151E-27,
            -3.089E-27,                            9.3E-29,
            1.74E-28,                              1.6E-29,
            -8.0E-30,                              -2.0E-30,
        };

        double RTWO = 1.414213562373095048801688724209;
        int IM = 24;
        double BM, B, BP = 0, P, T, XA;

        XA = std::fabs(Z) / RTWO;
        if (XA > 100) {
            P = 0;
        } else {
            T = (8 * XA - 30) / (4 * XA + 15);
            BM = 0;
            B = 0;
            for (int i = IM; i >= 0; i--) {
                BP = B;
                B = BM;
                BM = T * B - BP + kMvnphiA[i];
            }

            P = std::exp(-XA * XA) * (BM - BP) / 4;
        }

        if (Z > 0) P = 1 - P;

        return P;
    }

    // ALGORITHM AS241  APPL.STATIST. (1988) VOL. 37, NO. 3
    //
    // Produces the normal deviate Z corresponding to a given lower tail area of p.
    static double phinvs(double P) {
        // Coefficients for P close to 0.5 (HASH SUM AB 55.88319 28806 14901 4439)
        static constexpr double A0 = 3.3871328727963666080;
        static constexpr double A1 = 1.3314166789178437745E+2;
        static constexpr double A2 = 1.9715909503065514427E+3;
        static constexpr double A3 = 1.3731693765509461125E+4;
        static constexpr double A4 = 4.5921953931549871457E+4;
        static constexpr double A5 = 6.7265770927008700853E+4;
        static constexpr double A6 = 3.3430575583588128105E+4;
        static constexpr double A7 = 2.5090809287301226727E+3;
        static constexpr double B1 = 4.2313330701600911252E+1;
        static constexpr double B2 = 6.8718700749205790830E+2;
        static constexpr double B3 = 5.3941960214247511077E+3;
        static constexpr double B4 = 2.1213794301586595867E+4;
        static constexpr double B5 = 3.9307895800092710610E+4;
        static constexpr double B6 = 2.8729085735721942674E+4;
        static constexpr double B7 = 5.2264952788528545610E+3;
        // Coefficients for P not close to 0, 0.5, or 1 (HASH SUM CD 49.33206 50330 16102 89036)
        static constexpr double C0 = 1.42343711074968357734;
        static constexpr double C1 = 4.63033784615654529590;
        static constexpr double C2 = 5.76949722146069140550;
        static constexpr double C3 = 3.64784832476320460504;
        static constexpr double C4 = 1.27045825245236838258;
        static constexpr double C5 = 2.41780725177450611770E-1;
        static constexpr double C6 = 2.27238449892691845833E-2;
        static constexpr double C7 = 7.74545014278341407640E-4;
        static constexpr double D1 = 2.05319162663775882187;
        static constexpr double D2 = 1.67638483018380384940;
        static constexpr double D3 = 6.89767334985100004550E-1;
        static constexpr double D4 = 1.48103976427480074590E-1;
        static constexpr double D5 = 1.51986665636164571966E-2;
        static constexpr double D6 = 5.47593808499534494600E-4;
        static constexpr double D7 = 1.05075007164441684324E-9;
        // Coefficients for P near 0 or 1 (HASH SUM EF 47.52583 31754 92896 71629)
        static constexpr double E0 = 6.65790464350110377720;
        static constexpr double E1 = 5.46378491116411436990;
        static constexpr double E2 = 1.78482653991729133580;
        static constexpr double E3 = 2.96560571828504891230E-1;
        static constexpr double E4 = 2.65321895265761230930E-2;
        static constexpr double E5 = 1.24266094738807843860E-3;
        static constexpr double E6 = 2.71155556874348757815E-5;
        static constexpr double E7 = 2.01033439929228813265E-7;
        static constexpr double F1 = 5.99832206555887937690E-1;
        static constexpr double F2 = 1.36929880922735805310E-1;
        static constexpr double F3 = 1.48753612908506148525E-2;
        static constexpr double F4 = 7.86869131145613259100E-4;
        static constexpr double F5 = 1.84631831751005468180E-5;
        static constexpr double F6 = 1.42151175831644588870E-7;
        static constexpr double F7 = 2.04426310338993978564E-15;

        double SPLIT1 = 0.425, SPLIT2 = 5, CONST1 = 0.180625, CONST2 = 1.6;

        double Q = (2 * P - 1) / 2;
        double R;
        if (std::fabs(Q) < SPLIT1) {
            R = CONST1 - Q * Q;
            return Q *
                   (((((((A7 * R + A6) * R + A5) * R + A4) * R + A3) * R + A2) * R + A1) * R + A0) /
                   (((((((B7 * R + B6) * R + B5) * R + B4) * R + B3) * R + B2) * R + B1) * R + 1);
        } else {
            double result;
            R = std::min(P, 1 - P);
            if (R > 0) {
                R = std::sqrt(-std::log(R));
                if (R < SPLIT2) {
                    R -= CONST2;
                    result = (((((((C7 * R + C6) * R + C5) * R + C4) * R + C3) * R + C2) * R + C1) * R + C0) /
                             (((((((D7 * R + D6) * R + D5) * R + D4) * R + D3) * R + D2) * R + D1) * R + 1);
                } else {
                    R -= SPLIT2;
                    result = (((((((E7 * R + E6) * R + E5) * R + E4) * R + E3) * R + E2) * R + E1) * R + E0) /
                             (((((((F7 * R + F6) * R + F5) * R + F4) * R + F3) * R + F2) * R + F1) * R + 1);
                }
            } else {
                result = 9;
            }
            if (Q < 0) {
                result = -result;
            }
            return result;
        }
    }

    // Adapted from a function for computing bivariate normal probabilities by Yihong Ge
    // and Alan Genz, Washington State University.
    //
    // BVU - calculate the probability that X is larger than SH and Y is larger than SK.
    static double bvu(double SH, double SK, double R) {
        // Gauss Legendre Points and Weights, N = 6, 12, and 20 (a SEPARATE literal copy of
        // the same underlying constants as kBvndX*/kBvndW* above -- the C# source keeps
        // two independently-rounded tables for BivariateCDF vs BVU; both are transcribed
        // verbatim rather than consolidated, to avoid changing either algorithm's exact
        // floating-point results).
        static constexpr double kX[10][3] = {
            {-0.932469514203152, -0.9815606342467192, -0.9931285991850949},
            {-0.6612093864662645, -0.9041172563704749, -0.9639719272779138},
            {-0.2386191860831969, -0.7699026741943047, -0.912234428251326},
            {0, -0.5873179542866175, -0.8391169718222188},
            {0, -0.3678314989981802, -0.7463319064601508},
            {0, -0.1252334085114689, -0.636053680726515},
            {0, 0, -0.5108670019508271},
            {0, 0, -0.37370608871541955},
            {0, 0, -0.22778585114164507},
            {0, 0, -0.07652652113349734},
        };
        static constexpr double kW[10][3] = {
            {0.17132449237917036, 0.04717533638651183, 0.0176140071391521},
            {0.3607615730481386, 0.10693932599531843, 0.0406014298003869},
            {0.46791393457269104, 0.16007832854334622, 0.062672048334109},
            {0, 0.20316742672306592, 0.0832767415767047},
            {0, 0.2334925365383548, 0.10193011981724},
            {0, 0.24914704581340277, 0.118194531961518},
            {0, 0, 0.131688638449176},
            {0, 0, 0.142096109318382},
            {0, 0, 0.149172986472603},
            {0, 0, 0.152753387130725},
        };

        double BVN;
        int LG, NG;
        double TWOPI = 6.283185307179586;
        double AS, A, B, C, D, RS, XS;
        double SN, ASR, H, K, BS, HS, HK;

        if (std::fabs(R) < .3) {
            NG = 0;
            LG = 3;
        } else if (std::fabs(R) < .75) {
            NG = 1;
            LG = 6;
        } else {
            NG = 2;
            LG = 10;
        }

        H = SH;
        K = SK;
        HK = H * K;
        BVN = 0;
        if (std::fabs(R) < .925) {
            HS = (H * H + K * K) / 2;
            ASR = std::asin(R);
            for (int i = 0; i < LG; i++) {
                SN = std::sin(ASR * (kX[i][NG] + 1) / 2);
                BVN += kW[i][NG] * std::exp((SN * HK - HS) / (1 - SN * SN));
                SN = std::sin(ASR * (-kX[i][NG] + 1) / 2);
                BVN += kW[i][NG] * std::exp((SN * HK - HS) / (1 - SN * SN));
            }

            BVN = BVN * ASR / (2 * TWOPI) + mvnphi(-H) * mvnphi(-K);
        } else {
            if (R < 0) {
                K = -K;
                HK = -HK;
            }
            if (std::fabs(R) < 1) {
                AS = (1 - R) * (1 + R);
                A = std::sqrt(AS);
                BS = (H - K) * (H - K);
                C = (4 - HK) / 8;
                D = (12 - HK) / 16;
                BVN = A * std::exp(-(BS / AS + HK) / 2) *
                      (1 - C * (BS - AS) * (1 - D * BS / 5) / 3 + C * D * AS * AS / 5);
                if (HK > -160) {
                    B = std::sqrt(BS);
                    BVN = BVN - std::exp(-HK / 2) * std::sqrt(TWOPI) * mvnphi(-B / A) * B *
                                    (1 - C * BS * (1 - D * BS / 5) / 3);
                }
                A = A / 2;
                for (int i = 0; i < LG; i++) {
                    XS = (A * (kX[i][NG] + 1)) * (A * (kX[i][NG] + 1));
                    RS = std::sqrt(1 - XS);
                    BVN = BVN + A * kW[i][NG] *
                                    (std::exp(-BS / (2 * XS) - HK / (1 + RS)) / RS -
                                     std::exp(-(BS / XS + HK) / 2) * (1 + C * XS * (1 + D * XS)));
                    XS = AS * (-kX[i][NG] + 1) * (-kX[i][NG] + 1) / 4;
                    RS = std::sqrt(1 - XS);
                    BVN = BVN + A * kW[i][NG] * std::exp(-(BS / XS + HK) / 2) *
                                    (std::exp(-HK * XS / (2 * (1 + RS) * (1 + RS))) / RS -
                                     (1 + C * XS * (1 + D * XS)));
                }

                BVN = -BVN / TWOPI;
            }

            if (R > 0) {
                BVN = BVN + mvnphi(-std::max(H, K));
            } else {
                BVN = -BVN;
                if (K > H) {
                    if (H < 0) {
                        BVN = BVN + mvnphi(K) - mvnphi(H);
                    } else {
                        BVN = BVN + mvnphi(-H) - mvnphi(-K);
                    }
                }
            }
        }

        // This can sometimes produce very small, but negative values (approx. -5E-73)
        if (BVN < 0.0) BVN = 0.0;
        return BVN;
    }

    // A subroutine for computing multivariate normal probabilities, using the algorithm
    // in Alan Genz, "Numerical Computation of Multivariate Normal Probabilities", J. of
    // Computational and Graphical Stat., 1(1992), pp. 141-149.
    //
    // STUB (this commit): the Genz-Bretz MVNDST quasi-Monte-Carlo integrator (`mvndfn`,
    // `mvndnt`, `mvnlms`, `covsrt`, `rcswp`, `dkbvrc`, `dksmrc` and their instance working
    // storage) lands in the second commit of this task. Until then this always throws --
    // the CDF dim>=3 dispatch arm and `interval()` above call it unconditionally (matching
    // the C# dispatch structure exactly), so both simply fail until MVNDST is completed.
    void mvndst(int N, const std::vector<double>& LOWER, const std::vector<double>& UPPER,
                const std::vector<int>& INFIN, const std::vector<double>& CORREL, int MAXPTS,
                double ABSEPS, double RELEPS, double& ERROR, double& VALUE, int& INFORM) const {
        (void)N;
        (void)LOWER;
        (void)UPPER;
        (void)INFIN;
        (void)CORREL;
        (void)MAXPTS;
        (void)ABSEPS;
        (void)RELEPS;
        (void)ERROR;
        (void)VALUE;
        (void)INFORM;
        throw std::logic_error(
            "MultivariateNormal::mvndst is not yet implemented (lands in Task 5 commit b: the "
            "Genz-Bretz MVNDST quasi-Monte-Carlo integrator).");
    }

    // Creates a copy of the distribution.
    std::unique_ptr<MultivariateDistribution> clone() const override {
        std::unique_ptr<MultivariateNormal> mvn(new MultivariateNormal());
        mvn->parameters_valid_ = parameters_valid_;
        mvn->dimension_ = dimension_;
        mvn->mean_ = mean_;
        mvn->covariance_ = covariance_.clone();
        mvn->cholesky_ = cholesky_;
        mvn->lnconstant_ = lnconstant_;
        mvn->correlation_ = correlation_.clone();
        mvn->correl_ = correl_;
        mvn->mvnuni_ = mvnuni_;  // value copy -- see file header divergence note
        mvn->max_evaluations_ = max_evaluations_;
        mvn->absolute_error_ = absolute_error_;
        mvn->relative_error_ = relative_error_;
        mvn->lower_ = lower_;
        mvn->upper_ = upper_;
        mvn->infin_ = infin_;
        mvn->correlation_matrix_created_ = correlation_matrix_created_;
        mvn->cov_srted_ = cov_srted_;
        mvn->nl_ = nl_;
        return mvn;
    }

   private:
    // Private parameterless ctor for clone() (matches the C# `private MultivariateNormal()`).
    MultivariateNormal() = default;

    static double gauss(double t) {
        // GAUSS returns the area of the lower tail of the normal curve.
        return (1.0 + sf::erf::function(t / std::sqrt(2.0))) / 2.0;
    }

    // Create the correlation arrays required for computing the CDF (see the file header's
    // divergence note on the `invSqrtD` elementwise-reciprocal simplification).
    void create_correlation_matrix() const {
        la::Matrix D = la::Matrix::diagonal(covariance_);
        D.sqrt();
        la::Matrix inv_sqrt_d(dimension_);
        for (int i = 0; i < dimension_; ++i) inv_sqrt_d(i, i) = 1.0 / D(i, i);
        correlation_ = (inv_sqrt_d * covariance_) * inv_sqrt_d;
        // Create collapsed matrix
        correl_.assign(static_cast<std::size_t>(nl_), 0.0);
        int t = 0;
        for (int i = 1; i < dimension_; ++i) {
            for (int j = 0; j < i; ++j) {
                correl_[static_cast<std::size_t>(t)] = correlation_(i, j);
                ++t;
            }
        }
        correlation_matrix_created_ = true;
    }

    static sampling::MersenneTwister make_clock_seeded() {
        auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
        return sampling::MersenneTwister(static_cast<std::uint32_t>(ticks));
    }

    static sampling::MersenneTwister make_master(int seed) {
        if (seed > 0) return sampling::MersenneTwister(static_cast<std::uint32_t>(seed));
        return make_clock_seeded();
    }

    // (`kInf` is inherited from MultivariateDistribution -- protected static constexpr,
    // usable unqualified here; not redeclared, matching the Dirichlet/Multinomial precedent.)

    static constexpr double kBvndWn20[10] = {
        0.017614007139152121, 0.040601429800386939, 0.062672048334109054, 0.083276741576704755,
        0.1019301198172404,   0.1181945319615184,   0.13168863844917661,  0.1420961093183821,
        0.14917298647260371,  0.15275338713072589,
    };
    static constexpr double kBvndXn20[10] = {
        -0.99312859918509488, -0.96397192727791381, -0.912234428251326,   -0.83911697182221878,
        -0.7463319064601508,  -0.636053680726515,   -0.51086700195082713, -0.3737060887154196,
        -0.2277858511416451,  -0.076526521133497324,
    };
    static constexpr double kBvndWn12[6] = {
        0.047175336386511772, 0.1069393259953183, 0.16007832854334639,
        0.2031674267230659,   0.2334925365383547, 0.24914704581340291,
    };
    static constexpr double kBvndXn12[6] = {
        -0.98156063424671913, -0.904117256370475, -0.769902674194305,
        -0.58731795428661715, -0.36783149899818018, -0.12523340851146919,
    };
    static constexpr double kBvndWn6[3] = {0.1713244923791705, 0.36076157304813838, 0.46791393457269043};
    static constexpr double kBvndXn6[3] = {-0.93246951420315216, -0.6612093864662647, -0.238619186083197};

    // --- Fields (mirrors the C# field declaration order) ---

    bool parameters_valid_ = true;  // dead state -- see file header divergence note
    int dimension_ = 0;
    std::vector<double> mean_;
    la::Matrix covariance_ = la::Matrix(0, 0);
    std::optional<la::CholeskyDecomposition> cholesky_;
    double lnconstant_ = 0.0;
    // (`_variance`/`_standardDeviation` lazy-cache fields omitted -- see file header note)

    // Variables required for the multivariate CDF. `mutable`: MultivariateDistribution's
    // `cdf()` is `const`, but MVNDST-backed evaluation genuinely mutates the correlation
    // cache, the per-call `upper_` scratch, and the MVNUNI RNG stream -- mirroring the C#
    // source's mutable instance state (this port's documented relaxed-mutation rule for
    // stateful binding/model objects; see bestfit/CLAUDE.md).
    mutable la::Matrix correlation_ = la::Matrix(0, 0);
    mutable std::vector<double> correl_;
    mutable sampling::MersenneTwister mvnuni_ = make_clock_seeded();
    int max_evaluations_ = 100000;
    double absolute_error_ = 1E-4;
    double relative_error_ = 1E-4;
    std::vector<double> lower_;
    mutable std::vector<double> upper_;
    std::vector<int> infin_;
    mutable bool correlation_matrix_created_ = false;
    bool cov_srted_ = false;  // dead field -- see file header divergence note

    // SAVE A, B, INFI, COV / NL = 500 default, reassigned per-instance in set_parameters
    // (see the NL-quirk comment there). The fixed-500/125250-sized MVNDST working-storage
    // arrays that this field originally sizes (MVNDFN_A/B/INFI/COV) and the DKBVRC lattice
    // state (SAMPLS/NP/VAREST) are added in the second commit of this task.
    int nl_ = 500;
};

}  // namespace bestfit::numerics::distributions
