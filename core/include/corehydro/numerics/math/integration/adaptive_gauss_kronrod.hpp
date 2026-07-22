// ported from: Numerics/Mathematics/Integration/AdaptiveGuassKronrod.cs @ 2a0357a
//
// Adaptive Gauss-Kronrod integration (G10K21). Minimal port of the C# class used
// by VonMises::CDF. The C# class inherits Integrator which carries settings
// (AbsoluteTolerance, RelativeTolerance, MaxDepth, MaxFunctionEvaluations).
// This self-contained port hard-codes the same defaults and mirrors the recursive
// AdaptiveGK / EvaluateGaussKronrod structure method-for-method.
#pragma once
#include <cmath>
#include <functional>

#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::integration {

// G10K21 nodes (positive half only; mirror for negative half).
// Gauss-10 nodes are at indices 1, 3, 5, 7, 9 of the Kronrod array.
namespace detail_agk {
    inline constexpr double x_gauss[5] = {
        0.973906528517171720077964012084452,
        0.865063366688984510732096688423493,
        0.679409568299024406234327365114874,
        0.433395394129247190799265943165784,
        0.148874338981631210884826001129720
    };
    inline constexpr double w_gauss[5] = {
        0.066671344308688137593568809893332,
        0.149451349150580593145776339657697,
        0.219086362515982043995534934228163,
        0.269266719309996355091226921569469,
        0.295524224714752870173892994651338
    };
    inline constexpr double x_kronrod[11] = {
        0.995657163025808080735527280689003,
        0.973906528517171720077964012084452,
        0.930157491355708226001207180059508,
        0.865063366688984510732096688423493,
        0.780817726586416897063717578345042,
        0.679409568299024406234327365114874,
        0.562757134668604683339000099272694,
        0.433395394129247190799265943165784,
        0.294392862701460198131126603103866,
        0.148874338981631210884826001129720,
        0.000000000000000000000000000000000
    };
    inline constexpr double w_kronrod[11] = {
        0.011694638867371874278064396062192,
        0.032558162307964727478818972459390,
        0.054755896574351996031381300244580,
        0.075039674810919952767043140916190,
        0.093125454583697605535065465083366,
        0.109387158802297641899210590325805,
        0.123491976262065851077958109831074,
        0.134709217311473325928054001771707,
        0.142775938577060080797094273138717,
        0.147739104901338491374841515972068,
        0.149445554002916905664936468389821
    };
}  // namespace detail_agk

// Evaluates the G10K21 rule over [a, b].
// Returns {kronrod_result, gauss_result}.
inline std::pair<double, double> agk_evaluate(
        const std::function<double(double)>& f,
        double a, double b,
        int& func_evals) {
    using namespace detail_agk;
    double center      = 0.5 * (a + b);
    double half_length = 0.5 * (b - a);

    double result_gauss   = 0.0;
    double result_kronrod = 0.0;

    // Center point (x = 0 in the Kronrod array)
    double f0 = f(center);
    result_kronrod += w_kronrod[10] * f0;
    ++func_evals;

    for (int i = 0; i < 10; ++i) {
        double abscissa = half_length * x_kronrod[i];
        double fv1 = f(center - abscissa);
        double fv2 = f(center + abscissa);
        double fsum = fv1 + fv2;
        result_kronrod += w_kronrod[i] * fsum;
        // Gauss nodes are at odd indices of the Kronrod array: 1, 3, 5, 7, 9
        if (i % 2 == 1) {
            result_gauss += w_gauss[i / 2] * fsum;
        }
        func_evals += 2;
    }

    result_gauss   *= half_length;
    result_kronrod *= half_length;
    return {result_kronrod, result_gauss};
}

// Recursive adaptive subdivision. Mirrors the private AdaptiveGK method in C#.
inline double agk_recursive(
        const std::function<double(double)>& f,
        double a, double b,
        int depth,
        double kronrod_whole, double gauss_whole,
        double a0, double b0,
        double abs_tol, double rel_tol,
        int min_evals, int max_evals, int max_depth,
        int& func_evals) {
    double error = std::fabs(kronrod_whole - gauss_whole);
    double tol_scaled = rel_tol * std::fabs(b - a) / std::fabs(b0 - a0);
    bool abs_ok = error <= abs_tol;
    bool rel_ok = error <= tol_scaled * std::fabs(kronrod_whole);

    if (depth <= 0
        || std::fabs(a - b) <= kDoubleMachineEpsilon
        || func_evals >= max_evals
        || (func_evals >= min_evals
            && depth <= max_depth
            && (abs_ok || rel_ok))) {
        return kronrod_whole;
    }

    double m = 0.5 * (a + b);
    auto [kl, gl] = agk_evaluate(f, a, m, func_evals);
    auto [kr, gr] = agk_evaluate(f, m, b, func_evals);

    return agk_recursive(f, a, m, depth - 1, kl, gl, a0, b0,
                         abs_tol, rel_tol, min_evals, max_evals, max_depth, func_evals)
         + agk_recursive(f, m, b, depth - 1, kr, gr, a0, b0,
                         abs_tol, rel_tol, min_evals, max_evals, max_depth, func_evals);
}

/// Integrate f over [a, b] with the adaptive G10K21 rule.
/// Defaults match the C# Integrator base-class defaults.
inline double integrate(
        const std::function<double(double)>& f,
        double a, double b,
        double abs_tol     = 1e-8,
        double rel_tol     = 1e-8,
        int    max_depth   = 100,
        int    max_evals   = 10000000) {
    int func_evals = 0;
    auto [kronrod, gauss_init] = agk_evaluate(f, a, b, func_evals);
    return agk_recursive(f, a, b, max_depth, kronrod, gauss_init,
                         a, b, abs_tol, rel_tol,
                         /*min_evals=*/1, max_evals, max_depth, func_evals);
}

}  // namespace corehydro::numerics::math::integration
