// ported from: RMC-BestFit/src/RMC.BestFit/Estimation/GeneralizedMethodOfMoments.cs @ fc28c0c
//
// The four file-scope delegate declarations from the top of GeneralizedMethodOfMoments.cs
// (C# lines 18-58), split into this small standalone header so B9's IGMMModel can consume
// the aliases WITHOUT including the full GMM class (the compile-order contract from the
// Phase 6 plan: IGMMModel exposes these delegate-typed members, and the GMM class's
// IGMMModel constructor consumes an IGMMModel -- keeping the aliases here breaks that cycle).
//
// C# `double[] parameters` arguments are `const std::vector<double>&` here: none of the
// upstream delegate implementations (the Bulletin17CDistribution moment machinery, the test
// stubs) writes back into the parameter array, so the M14 mutable-point convention that the
// optimizer Objective needs does not apply to these callbacks.
//
// The C# `(Vector G, Matrix S)` tuple returned by MomentConditionFunction is ported as the
// small `MomentConditionResult` struct below; its field names mirror the C# tuple item names
// (`result.G` / `result.S`) verbatim.
#pragma once
#include <functional>
#include <vector>

#include "bestfit/numerics/math/linalg/matrix.hpp"
#include "bestfit/numerics/math/linalg/vector.hpp"

namespace bestfit::estimation {

// The tuple returned by MomentConditionFunction (C# `(Vector G, Matrix S)`):
//   G: the mean error vector of the moment conditions (length q).
//   S: the covariance matrix of the moment conditions (q x q).
struct MomentConditionResult {
    bestfit::numerics::math::linalg::Vector G;
    bestfit::numerics::math::linalg::Matrix S;
};

// An optional function that computes the Jacobian matrix of the mean moment conditions with
// respect to the parameters (C# delegate `double[,] JacobianFunction(double[] parameters)`;
// the C# `double[,]` return is the port's row-major `Matrix2D`).
using JacobianFunction =
    std::function<bestfit::numerics::math::linalg::Matrix2D(const std::vector<double>&)>;

// An optional penalty function (C# delegate `double PenaltyFunction(double[] parameters)`).
// If there is no penalty, return 0.
using PenaltyFunction = std::function<double(const std::vector<double>&)>;

// The moment condition function (C# delegate
// `(Vector G, Matrix S) MomentConditionFunction(double[] parameters)`).
using MomentConditionFunction =
    std::function<MomentConditionResult(const std::vector<double>&)>;

// An optional function that returns per-observation moment condition vectors for influence
// diagnostics: an [n x q] matrix whose row i is observation i's contribution g_i(theta)
// (C# delegate `double[,] PointwiseMomentConditionFunction(double[] parameters)`).
using PointwiseMomentConditionFunction =
    std::function<bestfit::numerics::math::linalg::Matrix2D(const std::vector<double>&)>;

}  // namespace bestfit::estimation
