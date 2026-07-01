// ported from: Numerics/Mathematics/Special Functions/Beta.cs @ a2c4dbf
#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>

#include "bestfit/numerics/tools.hpp"
#include "bestfit/numerics/math/special/gamma.hpp"

namespace bestfit::numerics::math::special {

// Alias for gamma-namespace functions used by beta
namespace gamma_sf = bestfit::numerics::math::special;

namespace beta::detail {

inline double incbcf(double a, double b, double x) {
    double xk, pk, pkm1, pkm2, qk, qkm1, qkm2;
    double k1, k2, k3, k4, k5, k6, k7, k8;
    double r, t, ans, thresh;
    constexpr double big    = 4503599627370496.0;
    constexpr double biginv = 2.2204460492503131e-16;
    k1 = a;
    k2 = a + b;
    k3 = a;
    k4 = a + 1.0;
    k5 = 1.0;
    k6 = b - 1.0;
    k7 = k4;
    k8 = a + 2.0;
    pkm2 = 0.0;
    qkm2 = 1.0;
    pkm1 = 1.0;
    qkm1 = 1.0;
    ans = 1.0;
    r = 1.0;
    thresh = 3.0 * bestfit::numerics::kDoubleMachineEpsilon;
    for (int i = 1; i <= 300; ++i) {
        xk = -(x * k1 * k2) / (k3 * k4);
        pk = pkm1 + pkm2 * xk;
        qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1; pkm1 = pk;
        qkm2 = qkm1; qkm1 = qk;
        xk = x * k5 * k6 / (k7 * k8);
        pk = pkm1 + pkm2 * xk;
        qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1; pkm1 = pk;
        qkm2 = qkm1; qkm1 = qk;
        if (qk != 0.0) r = pk / qk;
        if (r != 0.0) {
            t = std::fabs((ans - r) / r);
            ans = r;
        } else {
            t = 1.0;
        }
        if (t < thresh) return ans;
        k1 += 1.0; k2 += 1.0; k3 += 2.0; k4 += 2.0;
        k5 += 1.0; k6 -= 1.0; k7 += 2.0; k8 += 2.0;
        if (std::fabs(qk) + std::fabs(pk) > big) {
            pkm2 *= biginv; pkm1 *= biginv;
            qkm2 *= biginv; qkm1 *= biginv;
        }
        if (std::fabs(qk) < biginv || std::fabs(pk) < biginv) {
            pkm2 *= big; pkm1 *= big;
            qkm2 *= big; qkm1 *= big;
        }
    }
    return ans;
}

inline double incbd(double a, double b, double x) {
    double xk, pk, pkm1, pkm2, qk, qkm1, qkm2;
    double k1, k2, k3, k4, k5, k6, k7, k8;
    double r, t, ans, z, thresh;
    constexpr double big    = 4503599627370496.0;
    constexpr double biginv = 2.2204460492503131e-16;
    k1 = a;
    k2 = b - 1.0;
    k3 = a;
    k4 = a + 1.0;
    k5 = 1.0;
    k6 = a + b;
    k7 = a + 1.0;
    k8 = a + 2.0;
    pkm2 = 0.0;
    qkm2 = 1.0;
    pkm1 = 1.0;
    qkm1 = 1.0;
    z = x / (1.0 - x);
    ans = 1.0;
    r = 1.0;
    thresh = 3.0 * bestfit::numerics::kDoubleMachineEpsilon;
    for (int i = 1; i <= 300; ++i) {
        xk = -(z * k1 * k2) / (k3 * k4);
        pk = pkm1 + pkm2 * xk;
        qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1; pkm1 = pk;
        qkm2 = qkm1; qkm1 = qk;
        xk = z * k5 * k6 / (k7 * k8);
        pk = pkm1 + pkm2 * xk;
        qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1; pkm1 = pk;
        qkm2 = qkm1; qkm1 = qk;
        if (qk != 0.0) r = pk / qk;
        if (r != 0.0) {
            t = std::fabs((ans - r) / r);
            ans = r;
        } else {
            t = 1.0;
        }
        if (t < thresh) return ans;
        k1 += 1.0; k2 -= 1.0; k3 += 2.0; k4 += 2.0;
        k5 += 1.0; k6 += 1.0; k7 += 2.0; k8 += 2.0;
        if (std::fabs(qk) + std::fabs(pk) > big) {
            pkm2 *= biginv; pkm1 *= biginv;
            qkm2 *= biginv; qkm1 *= biginv;
        }
        if (std::fabs(qk) < biginv || std::fabs(pk) < biginv) {
            pkm2 *= big; pkm1 *= big;
            qkm2 *= big; qkm1 *= big;
        }
    }
    return ans;
}

inline double power_series(double a, double b, double x) {
    constexpr double LogMin    = -745.13321910194122;
    constexpr double LogMax    =  709.782712893384;
    constexpr double GammaMax  =  171.62437695630271;
    double s, t, u, v, n, t1, z, ai;
    ai = 1.0 / a;
    u  = (1.0 - b) * x;
    v  = u / (a + 1.0);
    t1 = v;
    t  = u;
    n  = 2.0;
    s  = 0.0;
    z  = bestfit::numerics::kDoubleMachineEpsilon * ai;
    while (std::fabs(v) > z) {
        u  = (n - b) * x / n;
        t *= u;
        v  = t / (a + n);
        s += v;
        n += 1.0;
    }
    s += t1;
    s += ai;
    u = a * std::log(x);
    if (a + b < GammaMax && std::fabs(u) < LogMax) {
        t  = gamma_sf::function(a + b) / (gamma_sf::function(a) * gamma_sf::function(b));
        s  = s * t * std::pow(x, a);
    } else {
        t = gamma_sf::log_gamma(a + b) - gamma_sf::log_gamma(a) - gamma_sf::log_gamma(b) + u + std::log(s);
        s = (t < LogMin) ? 0.0 : std::exp(t);
    }
    return s;
}

}  // namespace beta::detail

namespace beta {

inline double function(double a, double b) {
    return std::exp(gamma_sf::log_gamma(a) + gamma_sf::log_gamma(b) - gamma_sf::log_gamma(a + b));
}

inline double incomplete(double a, double b, double x) {
    constexpr double GammaMax = 171.62437695630271;
    constexpr double LogMin   = -745.13321910194122;
    constexpr double LogMax   =  709.782712893384;

    if (a <= 0.0) throw std::out_of_range("beta::incomplete: a must be > 0");
    if (b <= 0.0) throw std::out_of_range("beta::incomplete: b must be > 0");
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;
    if (x < 0.0 || x > 1.0) throw std::out_of_range("beta::incomplete: x must be in [0,1]");

    double aa, bb, t, xx, xc, w, y;
    bool flag = false;

    if (b * x <= 1.0 && x <= 0.95) {
        return detail::power_series(a, b, x);
    }

    w = 1.0 - x;
    if (x > a / (a + b)) {
        flag = true;
        aa = b; bb = a; xc = x; xx = w;
    } else {
        aa = a; bb = b; xc = w; xx = x;
    }

    if (flag && bb * xx <= 1.0 && xx <= 0.95) {
        t = detail::power_series(aa, bb, xx);
        t = (t <= bestfit::numerics::kDoubleMachineEpsilon)
            ? 1.0 - bestfit::numerics::kDoubleMachineEpsilon
            : 1.0 - t;
        return t;
    }

    y = xx * (aa + bb - 2.0) - (aa - 1.0);
    if (y < 0.0) {
        w = detail::incbcf(aa, bb, xx);
    } else {
        w = detail::incbd(aa, bb, xx) / xc;
    }

    y = aa * std::log(xx);
    t = bb * std::log(xc);
    if (aa + bb < GammaMax && std::fabs(y) < LogMax && std::fabs(t) < LogMax) {
        t  = std::pow(xc, bb);
        t *= std::pow(xx, aa);
        t /= aa;
        t *= w;
        t *= gamma_sf::function(aa + bb) / (gamma_sf::function(aa) * gamma_sf::function(bb));
        if (flag) {
            t = (t <= bestfit::numerics::kDoubleMachineEpsilon)
                ? 1.0 - bestfit::numerics::kDoubleMachineEpsilon
                : 1.0 - t;
        }
        return t;
    }

    y += t + gamma_sf::log_gamma(aa + bb) - gamma_sf::log_gamma(aa) - gamma_sf::log_gamma(bb);
    y += std::log(w / aa);
    t  = (y < LogMin) ? 0.0 : std::exp(y);
    if (flag) {
        t = (t <= bestfit::numerics::kDoubleMachineEpsilon)
            ? 1.0 - bestfit::numerics::kDoubleMachineEpsilon
            : 1.0 - t;
    }
    return t;
}

inline double incomplete_inverse(double aa, double bb, double yy0) {
    constexpr double LogMin = -745.13321910194122;

    if (yy0 <= 0.0) return 0.0;
    if (yy0 >= 1.0) return 1.0;

    double a, b, y0, d, y, x, x0, x1, lgm, yp, di, dithresh, yl, yh;
    int i, dir;
    bool nflg, rflg;

    if (aa <= 1.0 || bb <= 1.0) {
        nflg = true;
        dithresh = 4.0 * bestfit::numerics::kDoubleMachineEpsilon;
        rflg = false;
        a = aa; b = bb; y0 = yy0;
        x = a / (a + b);
        y = incomplete(a, b, x);
        goto ihalve;
    } else {
        nflg = false;
        dithresh = 0.0001;
    }

    yp = -bestfit::numerics::math::special::detail::normal_standard_z(yy0);
    if (yy0 > 0.5) {
        rflg = true;
        a = bb; b = aa; y0 = 1.0 - yy0;
        yp = -yp;
    } else {
        rflg = false;
        a = aa; b = bb; y0 = yy0;
    }

    lgm = (yp * yp - 3.0) / 6.0;
    x0  = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
    y   = yp * std::sqrt(x0 + lgm) / x0
          - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0))
          * (lgm + 5.0 / 6.0 - 2.0 / (3.0 * x0));
    y = 2.0 * y;
    if (y < LogMin) {
        x0 = 1.0;
        throw std::underflow_error("beta::incomplete_inverse: underflow");
    }

    x  = a / (a + b * std::exp(y));
    y  = incomplete(a, b, x);
    yp = (y - y0) / y0;
    if (std::fabs(yp) < 0.01) goto newt;

ihalve:
    x0 = 0.0; yl = 0.0;
    x1 = 1.0; yh = 1.0;
    di = 0.5; dir = 0;
    for (i = 0; i < 400; ++i) {
        if (i != 0) {
            x  = x0 + di * (x1 - x0);
            if (x == 1.0) x = 1.0 - bestfit::numerics::kDoubleMachineEpsilon;
            y  = incomplete(a, b, x);
            yp = (x1 - x0) / (x1 + x0);
            if (std::fabs(yp) < dithresh) { x0 = x; goto newt; }
        }
        if (y < y0) {
            x0 = x; yl = y;
            if (dir < 0) { dir = 0; di = 0.5; }
            else if (dir > 1) { di = 0.5 * di + 0.5; }
            else              { di = (y0 - y) / (yh - yl); }
            dir += 1;
            if (x0 > 0.75) {
                if (rflg) {
                    rflg = false; a = aa; b = bb; y0 = yy0;
                } else {
                    rflg = true;  a = bb; b = aa; y0 = 1.0 - yy0;
                }
                x = 1.0 - x;
                y = incomplete(a, b, x);
                goto ihalve;
            }
        } else {
            x1 = x;
            if (rflg && x1 < bestfit::numerics::kDoubleMachineEpsilon) {
                x0 = 0.0; goto done;
            }
            yh = y;
            if (dir > 0) { dir = 0; di = 0.5; }
            else if (dir < -1) { di = 0.5 * di; }
            else               { di = (y - y0) / (yh - yl); }
            dir -= 1;
        }
    }

    if (x0 >= 1.0) { x0 = 1.0 - bestfit::numerics::kDoubleMachineEpsilon; goto done; }
    if (x == 0.0) throw std::underflow_error("beta::incomplete_inverse: underflow");

newt:
    if (nflg) goto done;
    x0  = x;
    lgm = gamma_sf::log_gamma(a + b) - gamma_sf::log_gamma(a) - gamma_sf::log_gamma(b);
    for (i = 0; i < 10; ++i) {
        if (i != 0) y = incomplete(a, b, x0);
        d = (a - 1.0) * std::log(x0) + (b - 1.0) * std::log(1.0 - x0) + lgm;
        if (d < LogMin) throw std::underflow_error("beta::incomplete_inverse: underflow");
        d  = std::exp(d);
        d  = (y - y0) / d;
        x  = x0;
        x0 -= d;
        if (x0 <= 0.0) throw std::underflow_error("beta::incomplete_inverse: underflow");
        if (x0 >= 1.0) { x0 = 1.0 - bestfit::numerics::kDoubleMachineEpsilon; goto done; }
        if (std::fabs(d / x0) < 64.0 * bestfit::numerics::kDoubleMachineEpsilon) goto done;
    }

done:
    if (rflg) {
        if (x0 <= std::numeric_limits<double>::epsilon()) {
            x0 = 1.0 - std::numeric_limits<double>::epsilon();
        } else {
            x0 = 1.0 - x0;
        }
    }
    return x0;
}

}  // namespace beta

}  // namespace bestfit::numerics::math::special
