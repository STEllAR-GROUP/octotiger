// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define OCTOTIGER_QUARTIC_HD __host__ __device__
#else
#define OCTOTIGER_QUARTIC_HD
#endif

namespace radiation {

// Positive root of a*y + b*y^4 = 1. The caller supplies finite a,b in
// [0,1] with max(a,b) EXACTLY 1, as in RadiationM1::thermalExchange.
// No exceptions or data-dependent iteration; invalid inputs are outside this
// scalar kernel's contract. The root lies in [0.724491959...,1].
//
// Ferrari's resolvent for b*y^4+a*y-1 is S^3+4*b*S=a^2*b.
// Its usual radical formulas divide by small coefficients and subtract nearly
// equal roots. The two normalizations below keep every intermediate bounded.
// For U^3-V^3=q and U*V=t, use U-V=q/(U^2+t+V^2).
//
// b=1: w=S/a^2=1/(U^2+4/3+V^2), including its a=0 limit.
//      y=(sqrt(2/sqrt(w)-a^2*w)-a*sqrt(w))/2.
// a=1: S=cbrt(b)*z, z^3+4*cbrt(b)*z=1. Rationalizing both
//      subtractions gives y=4/[(1/sqrt(z)+z)*(sqrt(z)+sqrt(2/sqrt(z)-z))].
// The latter expression includes b=0 without dividing by b or cbrt(b).
// One fixed Newton correction removes the remaining radical roundoff.
OCTOTIGER_QUARTIC_HD inline double positiveQuarticRoot(double a, double b) {
    constexpr double fourThirds = 4.0 / 3.0;
    double y;
    if (b == 1.0) {
        const double q = a * a;
        const double u = ::cbrt(0.5 * q + ::sqrt(0.25 * q * q +
            fourThirds * fourThirds * fourThirds));
        const double v = fourThirds / u;
        const double w = 1.0 / (u * u + fourThirds + v * v);
        const double sqrtW = ::sqrt(w);
        y = 0.5 * (::sqrt(2.0 / sqrtW - q * w) - a * sqrtW);
    } else {
        const double t = fourThirds * ::cbrt(b);
        const double u = ::cbrt(0.5 + ::sqrt(0.25 + t * t * t));
        const double v = t / u;
        const double z = 1.0 / (u * u + t + v * v);
        const double sqrtZ = ::sqrt(z);
        const double inverseSqrtZ = 1.0 / sqrtZ;
        y = 4.0 / ((inverseSqrtZ + z) *
            (sqrtZ + ::sqrt(2.0 * inverseSqrtZ - z)));
    }
    const double y2 = y * y;
    const double residual = ::fma(a, y, ::fma(b, y2 * y2, -1.0));
    const double derivative = ::fma(4.0 * b, y2 * y, a);
    return ::fmin(1.0, y - residual / derivative);
}

// Uniform-work alternative with the same coefficient contract. Convexity
// keeps Newton iterates from y=1 above the positive root r in exact arithmetic.
// With e=x-r, e_next=b*e^2*(3*x^2+2*x*r+r^2)/(a+4*b*x^3)
// <=1.5*e^2/x. Therefore t=e/r satisfies t_next<=1.5*t^2/(1+t).
// Scaling gives r>=0.724491959... and t_initial<=0.38028...; six corrections
// bound t below 4e-22 in exact arithmetic. Floating-point accuracy is tested
// separately across the standalone endpoint/logarithmic sweep.
// This avoids the radicals as well as the variable iteration count. Keep the
// iteration count fixed: this is intentionally not a convergence-tolerance loop.
OCTOTIGER_QUARTIC_HD inline double positiveQuarticRootFixedNewton(double a, double b) {
    double y = 1.0;
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll
#endif
    for (int iteration = 0; iteration < 6; ++iteration) {
        const double y2 = y * y;
        const double residual = ::fma(a, y, ::fma(b, y2 * y2, -1.0));
        y -= residual / ::fma(4.0 * b, y2 * y, a);
    }
    return ::fmin(1.0, y);
}

} // namespace radiation

#undef OCTOTIGER_QUARTIC_HD
