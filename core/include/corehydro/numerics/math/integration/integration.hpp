// ported from: Numerics/Mathematics/Integration/Integration.cs @ a2c4dbf
//
// Static numerical-integration methods. Only `GaussLegendre20` (C# line 62) is ported in
// this slice (M8) -- it is what the uncertain-data (measurement-error) branch of the
// UnivariateDistribution stationary likelihood integrates with. The remaining methods of the
// C# static class (GaussLegendre 10-point, TrapezoidalRule, SimpsonsRule, Midpoint) are not
// yet needed by any ported caller and are deliberately not ported; add them here when a
// caller arrives. (The adaptive integrators live in their own files upstream and here --
// see adaptive_gauss_kronrod.hpp.)
//
// Nodes are roots of the Legendre polynomial P20(x); weights are the corresponding
// Christoffel numbers (Abramowitz and Stegun 1964, Table 25.4). Values are verbatim from the
// C# source.
#pragma once
#include <functional>

namespace corehydro::numerics::math::integration {

class Integration {
   public:
    // Returns the integral of `f` between `a` and `b` by twenty-point Gauss-Legendre
    // integration (C# `GaussLegendre20`, line 62). Exact for polynomials of degree 39 or
    // less; 10 symmetric node pairs (20 function evaluations total).
    static double gauss_legendre20(const std::function<double(double)>& f, double a, double b) {
        static constexpr double x[10] = {
            0.0765265211334973338, 0.2277858511416450781, 0.3737060887154195607,
            0.5108670019508270981, 0.6360536807265150254, 0.7463319064601507926,
            0.8391169718222188234, 0.9122344282513259059, 0.9639719272779137912,
            0.9931285991850949247};
        static constexpr double w[10] = {
            0.1527533871307258507, 0.1491729864726037467, 0.1420961093183820514,
            0.1316886384491766269, 0.1181945319615184174, 0.1019301198172404351,
            0.0832767415767047487, 0.0626720483341090636, 0.0406014298003869413,
            0.0176140071391521183};
        double xm = 0.5 * (b + a);
        double xr = 0.5 * (b - a);
        double s = 0;
        for (int j = 0; j < 10; j++) {
            double dx = xr * x[j];
            s += w[j] * (f(xm + dx) + f(xm - dx));
        }
        s *= xr;  // C# `return s *= xr;`
        return s;
    }
};

}  // namespace corehydro::numerics::math::integration
