// ported from: Numerics/Data/Statistics/Probability.cs @ a2c4dbf
//
// NARROW port. `Probability` is a large static utility class (basic two-event rules,
// joint-probability dispatch, unions, exclusive-combination enumeration, and three
// different correlated-joint-probability engines: HPCM, the original Pandey PCM, and an
// MVN-CDF-backed path). This file ports ONLY the members that CompetingRisks.cs's
// CDF/PDF/CumulativeIncidenceFunctions paths actually call, traced call-by-call from the
// C# source rather than from this class's full public surface.
//
// IMPORTANT finding (overload resolution, not a judgment call): CompetingRisks.cs's
// correlated-mode CDF calls read
//     Probability.UnionPCM(cdf, _mvn.Covariance)
//     Probability.JointProbability(cdf, ind, _mvn.Covariance)
// The second call's 3rd argument is a `double[,]` (`_mvn.Covariance`), which only matches
// the overload `JointProbability(IList<double>, int[], double[,]? correlationMatrix = null,
// DependencyType dependency = DependencyType.CorrelationMatrix)` -- there is no
// `JointProbability(IList<double>, int[], MultivariateNormal)` overload in the C# source.
// With a non-null correlationMatrix and the DEFAULT dependency (CorrelationMatrix), that
// overload unconditionally dispatches to `JointProbabilityHPCM` ("Haden Smith's
// modification of Pandey's Product of Conditional Marginals" method) -- never to
// `JointProbabilityMVN`. `UnionPCM` reaches the same 3-arg overload internally for each
// inclusion-exclusion term. CompetingRisks constructs a `MultivariateNormal` purely to
// hold its `.Covariance` matrix (mu/sigma bookkeeping for PerfectlyNegative's synthetic
// rho, or to pass through a user CorrelationMatrix) -- that MVN instance's `.CDF()` is
// NEVER called anywhere in CompetingRisks.cs. Consequence: the correlated CDF/PDF paths
// this file supports are fully DETERMINISTIC for any number of components D -- ONLY
// `MultivariateNormal.BivariateCDF` (Drezner/Genz closed-form bivariate normal CDF, no
// RNG) is used, never the seeded Genz-Bretz MVNDST quasi-Monte-Carlo integrator that
// backs `MultivariateNormal.CDF()` for dimension >= 3 (see multivariate_normal.hpp's
// `mvnuni_` note and Task 6's carry-forward note in .superpowers/sdd/progress.md, which
// both concern a DIFFERENT code path that CompetingRisks does not reach). See
// docs/upstream-csharp-issues.md for a short write-up of this call-path finding.
//
// Ported (the narrow subset CompetingRisks.cs's call sites reach):
//   DependencyType; JointProbability (2-arg dependency dispatch, and the 4-arg
//   indicators+correlationMatrix overload); IndependentJointProbability (1-arg + 2-arg);
//   PositiveJointProbability (1-arg + 2-arg); NegativeJointProbability (1-arg + 2-arg);
//   JointProbabilityHPCM; Union (2-arg dependency dispatch, renamed `union_probability`
//   -- `union` is a C++ keyword); IndependentUnion; PositivelyDependentUnion;
//   NegativelyDependentUnion; UnionPCM (2-arg + 6-arg).
//
// Explicitly OMITTED (unreachable from CompetingRisks.cs; later-phase item if some other
// caller needs them):
//   JointProbabilityMVN / JointProbabilitiesMVN / UnionMVN (the actual MVN-CDF-backed
//     joint-probability path -- see the finding above: CompetingRisks never calls these),
//   JointProbabilityPCM (Pandey's ORIGINAL, non-Haden-Smith-modified PCM -- HPCM is a
//     distinct, separately-implemented method in the C# source, not a generalization),
//   JointProbabilitiesPCM (array/Parallel.For variant of JointProbabilityPCM),
//   the `out List<...>` UnionPCM overload (diagnostic variant returning per-term detail),
//   the entire "Basic Probability Rules" region (AAndB/AOrB/ANotB/BNotA/AGivenB/BGivenA),
//   the entire "Exclusive Probability of all Combinations of Events" region
//   (IndependentExclusive/PositivelyDependentExclusive/NegativelyDependentExclusive/
//   ExclusivePCM/ExclusiveMVN and their array/out-list variants), CommonCauseAdjustment,
//   MutuallyExclusiveAdjustment.
//
// Small Tools.cs helpers this file needs (Clamp; Sum/Product/Min/Max with an indicators
// overload) are reimplemented narrowly in the `detail` namespace below rather than
// pulling in a full Tools.cs port (mirrors this project's existing "port only what's
// called" precedent for that class -- see tools.hpp's header note). `useComplement`
// (false in every call site reached here) is omitted from the indicator-overload helpers.
// Every site below that transcribes a C# `Math.Min`/`Math.Max`/`Tools.Max` call goes
// through `detail::nan_min`/`detail::nan_max`/`detail::max_value` rather than plain
// `std::min`/`std::max`/`std::max_element`: the BCL versions propagate NaN (return NaN if
// any operand is NaN), while the bare STL algorithms do not (a comparison against NaN is
// always false, so e.g. `std::min(a, b)` silently returns `a` when `b` is NaN).
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/special/factorial.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::data::probability {

namespace sf = corehydro::numerics::math::special;

// Same underlying type as MultivariateNormal::covariance()'s return type
// (corehydro::numerics::math::linalg::Matrix2D); aliased locally for a shorter, class-
// scoped name mirroring the C# `double[,]` parameters this file ports.
using Matrix2D = corehydro::numerics::math::linalg::Matrix2D;

// Mirrors Probability.DependencyType.
enum class DependencyType { Independent, PerfectlyPositive, PerfectlyNegative, CorrelationMatrix };

namespace detail {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Tools.Clamp narrow port.
inline double clamp(double x, double min_v, double max_v) {
    return x < min_v ? min_v : (x > max_v ? max_v : x);
}

// Math.Min(double, double) narrow port. The BCL's Math.Min propagates NaN (returns NaN if
// either argument is NaN); plain `std::min` does NOT (a comparison against NaN is always
// false, so `std::min(a, b)` degrades to "return a" when b is NaN). Used at every site that
// transcribes a C# `Math.Min` call so NaN inputs propagate identically to the C# source.
inline double nan_min(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return kNaN;
    return b < a ? b : a;
}

// Math.Max(double, double) narrow port; see nan_min above.
inline double nan_max(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return kNaN;
    return b > a ? b : a;
}

// Tools.Max(IList<double>) narrow port: returns NaN if any element is NaN. Plain
// `std::max_element` is NOT NaN-aware (it would silently return a position-dependent
// finite value unless NaN happens to be the very first element).
inline double max_value(const std::vector<double>& values) {
    double m = -std::numeric_limits<double>::infinity();
    for (double v : values) {
        if (std::isnan(v)) return kNaN;
        if (v > m) m = v;
    }
    return m;
}

// Tools.Sum(IList<double>, IList<int>, useComplement=false) narrow port.
inline double sum(const std::vector<double>& values, const std::vector<int>& indicators) {
    double s = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i)
        if (indicators[i] == 1) s += values[i];
    return s;
}

// Tools.Product(IList<double>, IList<int>, useComplement=false) narrow port. (Note: the
// 1-arg `Tools.Product(IList<double>)` overload is NOT ported here -- no member reached
// from CompetingRisks.cs calls it; `IndependentJointProbability(IList<double>)` below
// inlines its own product loop in the C# source rather than calling `Tools.Product`.)
inline double product(const std::vector<double>& values, const std::vector<int>& indicators) {
    double p = 1.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (indicators[i] == 1) {
            p *= values[i];
            if (p == 0.0) return 0.0;
        }
    }
    return p;
}

// Tools.Min(IList<double>) narrow port.
inline double min_value(const std::vector<double>& values) {
    double m = std::numeric_limits<double>::max();
    for (double v : values) {
        if (std::isnan(v)) return kNaN;
        if (v < m) m = v;
    }
    return m;
}

// Tools.Min(IList<double>, IList<int>, useComplement=false) narrow port.
inline double min_value(const std::vector<double>& values, const std::vector<int>& indicators) {
    double m = std::numeric_limits<double>::max();
    bool any = false;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (indicators[i] == 1) {
            if (std::isnan(values[i])) return kNaN;
            if (values[i] < m) m = values[i];
            any = true;
        }
    }
    return any ? m : kNaN;
}

}  // namespace detail

// --- Joint Probability (dispatch by DependencyType) ---

// Mirrors Probability.IndependentJointProbability(IList<double>).
inline double independent_joint_probability(const std::vector<double>& probabilities) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    double p = 1.0;
    for (double v : probabilities) {
        p *= v;
        if (p == 0.0) return 0.0;
    }
    return detail::clamp(p, 0.0, 1.0);
}

// Mirrors Probability.PositiveJointProbability(IList<double>).
inline double positive_joint_probability(const std::vector<double>& probabilities) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    return detail::clamp(detail::min_value(probabilities), 0.0, 1.0);
}

// Mirrors Probability.NegativeJointProbability(IList<double>).
inline double negative_joint_probability(const std::vector<double>& probabilities) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    double s = 0.0;
    for (double v : probabilities) s += v;
    return detail::nan_max(0.0, detail::nan_min(1.0, s) - 1.0);
}

// Mirrors Probability.JointProbability(IList<double>, DependencyType = Independent).
inline double joint_probability(const std::vector<double>& probabilities,
                                 DependencyType dependency = DependencyType::Independent) {
    if (dependency == DependencyType::Independent) return independent_joint_probability(probabilities);
    if (dependency == DependencyType::PerfectlyPositive) return positive_joint_probability(probabilities);
    if (dependency == DependencyType::PerfectlyNegative) return negative_joint_probability(probabilities);
    return detail::kNaN;
}

// --- Joint Probability with indicators + correlation matrix (correlated-mode CDF/PDF) ---

// Mirrors Probability.IndependentJointProbability(IList<double>, int[]).
inline double independent_joint_probability(const std::vector<double>& probabilities,
                                             const std::vector<int>& indicators) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    if (indicators.empty()) throw std::invalid_argument("indicators must have at least one row");
    if (probabilities.size() != indicators.size())
        throw std::invalid_argument("probabilities and indicators must have the same length");
    return detail::clamp(detail::product(probabilities, indicators), 0.0, 1.0);
}

// Mirrors Probability.PositiveJointProbability(IList<double>, int[]).
inline double positive_joint_probability(const std::vector<double>& probabilities,
                                          const std::vector<int>& indicators) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    if (indicators.empty()) throw std::invalid_argument("indicators must have at least one row");
    if (probabilities.size() != indicators.size())
        throw std::invalid_argument("probabilities and indicators must have the same length");
    return detail::clamp(detail::min_value(probabilities, indicators), 0.0, 1.0);
}

// Mirrors Probability.NegativeJointProbability(IList<double>, int[]).
inline double negative_joint_probability(const std::vector<double>& probabilities,
                                          const std::vector<int>& indicators) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    if (indicators.empty()) throw std::invalid_argument("indicators must have at least one row");
    if (probabilities.size() != indicators.size())
        throw std::invalid_argument("probabilities and indicators must have the same length");
    return detail::nan_max(0.0, detail::nan_min(1.0, detail::sum(probabilities, indicators)) - 1.0);
}

// Computes the joint probability of multiple events with dependency, using Haden Smith's
// modification of Pandey's method for the Product of Conditional Marginals (PCM).
// Mirrors Probability.JointProbabilityHPCM verbatim, INCLUDING the asymmetric guard: the
// "First cycle" below leaves `//if (cdf < 1e-300) cdf = 1e-300;` commented out (matching
// the C# source exactly), while the "Remaining cycles" loop (only reached for n >= 3)
// keeps that guard active. This looks like an oversight in the C# source (see
// docs/upstream-csharp-issues.md) but is preserved for bit-for-bit fidelity.
inline double joint_probability_hpcm(const std::vector<double>& probabilities,
                                      const std::vector<int>& indicators,
                                      const Matrix2D& correlation_matrix,
                                      std::vector<double>* conditional_probabilities = nullptr) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    if (indicators.empty()) throw std::invalid_argument("indicators must have at least one row");
    if (probabilities.size() != indicators.size())
        throw std::invalid_argument("probabilities and indicators must have the same length");
    int n = static_cast<int>(probabilities.size());
    if (static_cast<int>(correlation_matrix.size()) != n ||
        (n > 0 && static_cast<int>(correlation_matrix[0].size()) != n))
        throw std::invalid_argument(
            "correlation matrix must be square with dimensions equal to the length of probabilities");

    constexpr double zMin = -9.0, zMax = 9.0;
    Matrix2D R = correlation_matrix;  // mirrors `Array.Copy(correlationMatrix, R, ...)`

    for (int i = 0; i < n; ++i) {
        std::size_t ii = static_cast<std::size_t>(i);
        if (indicators[ii] == 0)
            R[ii][ii] = detail::clamp(distributions::Normal::standard_z(1.0), zMin, zMax);
        else
            R[ii][ii] = detail::clamp(distributions::Normal::standard_z(probabilities[ii]), zMin, zMax);
    }

    // First cycle
    double z1 = R[0][0];
    double pdf = distributions::Normal::standard_pdf(z1);
    double cdf = distributions::Normal::standard_cdf(z1);
    // (guard intentionally NOT applied here -- see header comment above)
    double A = pdf / cdf;
    double B = A * (z1 + A);
    for (int k = 1; k < n; ++k) {
        std::size_t kk = static_cast<std::size_t>(k);
        double z2 = R[kk][kk];
        double r12 = R[0][kk];
        r12 = std::fabs(r12) < 1e-3 ? 0.0 : r12;
        double p21 = distributions::MultivariateNormal::bivariate_cdf(-z1, -z2, r12) / cdf;
        p21 = detail::nan_max(0.0, detail::nan_min(1.0, p21));
        double z21 = detail::clamp(distributions::Normal::standard_z(p21), zMin, zMax);
        R[kk][0] = z21;
    }
    for (int ir = 1; ir < n - 1; ++ir) {
        std::size_t iir = static_cast<std::size_t>(ir);
        for (int ic = ir + 1; ic < n; ++ic) {
            std::size_t iic = static_cast<std::size_t>(ic);
            R[iir][iic] = (R[iir][iic] - R[0][iir] * R[0][iic] * B) /
                std::sqrt((1.0 - R[0][iir] * R[0][iir] * B) * (1.0 - R[0][iic] * R[0][iic] * B));
        }
    }

    // Remaining cycles (only reached when n >= 3)
    for (int j = 1; j < n - 1; ++j) {
        std::size_t jj = static_cast<std::size_t>(j);
        z1 = R[jj][jj - 1];
        pdf = distributions::Normal::standard_pdf(z1);
        cdf = distributions::Normal::standard_cdf(z1);
        if (cdf < 1e-300) cdf = 1e-300;
        A = pdf / cdf;
        B = A * (z1 + A);
        for (int k = j + 1; k < n; ++k) {
            std::size_t kk = static_cast<std::size_t>(k);
            double z2 = R[kk][jj - 1];
            double r12 = R[jj][kk];
            r12 = std::fabs(r12) < 1e-3 ? 0.0 : r12;
            double p21 = distributions::MultivariateNormal::bivariate_cdf(-z1, -z2, r12) / cdf;
            p21 = detail::nan_max(0.0, detail::nan_min(1.0, p21));
            double z21 = detail::clamp(distributions::Normal::standard_z(p21), zMin, zMax);
            R[kk][jj] = z21;
        }
        for (int ir = j + 1; ir < n - 1; ++ir) {
            std::size_t iir = static_cast<std::size_t>(ir);
            for (int ic = ir + 1; ic < n; ++ic) {
                std::size_t iic = static_cast<std::size_t>(ic);
                R[iir][iic] = (R[iir][iic] - R[jj][iir] * R[jj][iic] * B) /
                    std::sqrt((1.0 - R[jj][iir] * R[jj][iir] * B) * (1.0 - R[jj][iic] * R[jj][iic] * B));
            }
        }
    }

    double jp = std::log(distributions::Normal::standard_cdf(R[0][0]));
    if (conditional_probabilities != nullptr &&
        conditional_probabilities->size() == static_cast<std::size_t>(n))
        (*conditional_probabilities)[0] = distributions::Normal::standard_cdf(R[0][0]);
    for (int i = 1; i < n; ++i) {
        std::size_t ii = static_cast<std::size_t>(i);
        jp += std::log(distributions::Normal::standard_cdf(R[ii][ii - 1]));
        if (conditional_probabilities != nullptr &&
            conditional_probabilities->size() == static_cast<std::size_t>(n))
            (*conditional_probabilities)[ii] = distributions::Normal::standard_cdf(R[ii][ii - 1]);
    }
    jp = std::exp(jp);
    jp = std::min(1.0, std::max(0.0, jp));
    if (std::isnan(jp)) jp = 0.0;
    return jp;
}

// Mirrors Probability.JointProbability(IList<double>, int[], double[,]? = null,
// DependencyType = CorrelationMatrix). CompetingRisks.cs always calls this with a
// non-null correlationMatrix and the default (CorrelationMatrix) dependency, which
// dispatches to joint_probability_hpcm above -- see this file's header comment.
inline double joint_probability(const std::vector<double>& probabilities,
                                 const std::vector<int>& indicators,
                                 const Matrix2D* correlation_matrix = nullptr,
                                 DependencyType dependency = DependencyType::CorrelationMatrix) {
    if (dependency == DependencyType::CorrelationMatrix && correlation_matrix != nullptr)
        return joint_probability_hpcm(probabilities, indicators, *correlation_matrix);
    if (dependency == DependencyType::Independent)
        return independent_joint_probability(probabilities, indicators);
    if (dependency == DependencyType::PerfectlyPositive)
        return positive_joint_probability(probabilities, indicators);
    if (dependency == DependencyType::PerfectlyNegative)
        return negative_joint_probability(probabilities, indicators);
    return detail::kNaN;
}

// --- Probability of Union ---
// `Union` is renamed `union_probability` (`union` is a reserved C++ keyword).

// Mirrors Probability.IndependentUnion(IList<double>) (De Morgan's rule).
inline double independent_union(const std::vector<double>& probabilities) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    if (probabilities.size() == 1) return probabilities[0];
    double numerator = 1.0;
    for (double p : probabilities) {
        double q = 1.0 - p;
        if (q == 0.0) return 1.0;
        numerator *= q;
    }
    return 1.0 - numerator;
}

// Mirrors Probability.PositivelyDependentUnion(IList<double>).
inline double positively_dependent_union(const std::vector<double>& probabilities) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    if (probabilities.size() == 1) return probabilities[0];
    double m = detail::max_value(probabilities);
    return detail::clamp(m, 0.0, 1.0);
}

// Mirrors Probability.NegativelyDependentUnion(IList<double>).
inline double negatively_dependent_union(const std::vector<double>& probabilities) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities must have a length greater than 0");
    if (probabilities.size() == 1) return probabilities[0];
    double s = 0.0;
    for (double p : probabilities) s += p;
    return detail::clamp(s, 0.0, 1.0);
}

// Mirrors Probability.Union(IList<double>, DependencyType = Independent).
inline double union_probability(const std::vector<double>& probabilities,
                                 DependencyType dependency = DependencyType::Independent) {
    if (dependency == DependencyType::Independent) return independent_union(probabilities);
    if (dependency == DependencyType::PerfectlyPositive) return positively_dependent_union(probabilities);
    if (dependency == DependencyType::PerfectlyNegative) return negatively_dependent_union(probabilities);
    return detail::kNaN;
}

// Returns the probability of union using the inclusion-exclusion method, with dependence
// captured by joint_probability_hpcm via correlation_matrix. Mirrors the 6-arg
// Probability.UnionPCM(IList<double>, int[], int[,], double[,], double, double) verbatim.
inline double union_pcm(const std::vector<double>& probabilities,
                         const std::vector<int>& binomial_combinations,
                         const std::vector<std::vector<int>>& indicators,
                         const Matrix2D& correlation_matrix, double absolute_tolerance = 1e-4,
                         double relative_tolerance = 1e-4) {
    if (probabilities.empty() || binomial_combinations.empty() || indicators.empty())
        throw std::invalid_argument(
            "probabilities, binomial_combinations, and indicators must be non-empty");

    double result = 0.0;
    double s = 1.0;
    std::size_t j = 0;
    int c = binomial_combinations[j];
    double inc = detail::kNaN;
    double exc = detail::kNaN;
    std::size_t num_indicators = indicators.size();

    for (std::size_t i = 0; i < num_indicators; ++i) {
        if (static_cast<int>(i) == c) {
            if (j > 0) {
                if (s == 1.0) inc = result;
                else if (s == -1.0) exc = result;
            }
            double diff = std::fabs(inc - exc);
            if (j > 0 && j < binomial_combinations.size() && diff <= absolute_tolerance &&
                diff <= relative_tolerance * std::min(inc, exc)) {
                return result + 0.5 * diff;
            }
            s *= -1.0;
            ++j;
            if (j < binomial_combinations.size()) c += binomial_combinations[j];
        }
        if (i < probabilities.size()) {
            result += s * probabilities[i];
        } else {
            result += s * joint_probability(probabilities, indicators[i], &correlation_matrix);
        }
    }
    return result;
}

// Mirrors the 2-arg Probability.UnionPCM(IList<double>, double[,], double, double)
// convenience overload: builds the binomial-combination counts and all-subsets indicator
// table, then delegates to the 6-arg overload above.
inline double union_pcm(const std::vector<double>& probabilities, const Matrix2D& correlation_matrix,
                         double absolute_tolerance = 1e-4, double relative_tolerance = 1e-4) {
    if (probabilities.empty())
        throw std::invalid_argument("probabilities and correlation matrix must be non-empty");
    int N = static_cast<int>(probabilities.size());
    std::vector<int> binomial_combinations(static_cast<std::size_t>(N));
    for (int i = 1; i <= N; ++i)
        binomial_combinations[static_cast<std::size_t>(i - 1)] =
            static_cast<int>(sf::factorial::binomial_coefficient(N, i));
    auto indicators = sf::factorial::all_combinations(N);
    return union_pcm(probabilities, binomial_combinations, indicators, correlation_matrix,
                      absolute_tolerance, relative_tolerance);
}

}  // namespace corehydro::numerics::data::probability
