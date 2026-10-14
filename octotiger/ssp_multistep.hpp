// Copyright (c) 2026 The STEllAR Group
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace ssp_multistep {

struct coefficients {
    double c1, c2, c3, c4, c5, v;
    // h*v5 stays finite even when previous_dt/h exceeds the double range.
    // Form that duration once before applying coefficients to field arrays.
    long double v5;
};

// Keep a margin from the singular alpha=1/2 boundary. Decreases have no
// artificial ratio limit; both forward-Euler bounds still apply.
constexpr double maximum_growth = 1.9;
constexpr double minimum_alpha = 1.0 / maximum_growth;

namespace detail {

inline long double c4_from_v(long double a, long double v) {
    const long double ratio = (a + v) / (a + 2 * v);
    const long double scale = (a + 1) / a;
    return scale * scale * ratio * ratio / (2 * v);
}

inline long double previous_derivative_weight(long double a, long double v) {
    return -(a + 1) * (2 * a * a * v - 2 * a * a + 5 * a * v * v -
        6 * a * v - 3 * v * v) / (2 * a * a * (a + 2 * v) * (a + 2 * v));
}

inline bool finite(const coefficients& c) {
    return std::isfinite(c.c1) && std::isfinite(c.c2) && std::isfinite(c.c3) &&
        std::isfinite(c.c4) && std::isfinite(c.c5) && std::isfinite(c.v) &&
        std::isfinite(c.v5);
}

} // namespace detail

// Evaluate the paper's one-parameter family without its cancellation-prone
// Cardano expression. This is an algebraic diagnostic: unlike make_coefficients,
// it can return negative weights (as in the paper's verification table).
inline bool from_c4(double alpha, double c4, coefficients& result) {
    if (!(alpha > 0) || !std::isfinite(alpha) || !(c4 > 0) || !std::isfinite(c4)) {
        return false;
    }
    const long double a = alpha;
    long double lower = 0;
    long double upper = std::max(1.0L, a);
    for (int i = 0; i < 128 && detail::c4_from_v(a, upper) > c4; ++i) {
        upper *= 2;
    }
    if (detail::c4_from_v(a, upper) > c4) {
        return false;
    }
    // c4(v) decreases strictly for v > 0, so the positive root is unique.
    for (int i = 0; i < 128; ++i) {
        const long double middle = lower + (upper - lower) / 2;
        if (detail::c4_from_v(a, middle) > c4) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    const long double v = lower + (upper - lower) / 2;
    const long double z = v / (a + v);
    coefficients c;
    c.c2 = static_cast<double>(z * z);
    c.c1 = 1 - c.c2;
    c.c3 = static_cast<double>((a + 1) * (a + 1) *
        ((4 * a - 2) * v * v + (2 * a * a - 4 * a) * v - a * a) /
        (2 * a * a * a * v * (a + 2 * v)));
    c.c4 = c4;
    c.c5 = 1 - c.c3 - c.c4;
    c.v = static_cast<double>(v);
    c.v5 = static_cast<double>(detail::previous_derivative_weight(a, v) / c.c5);
    if (!detail::finite(c)) {
        return false;
    }
    result = c;
    return true;
}

// Use z=v/(alpha+v), rather than subtracting c4 from one or evaluating
// powers of alpha. This retains the small historical weights for large alpha.
namespace detail {

inline bool make_coefficients_from_ratio(long double alpha, coefficients& result) {
    if (!std::isfinite(alpha) || alpha < minimum_alpha) {
        return false;
    }
    const long double s = 1 / alpha;
    const long double root = std::sqrt(1 + 2 * s * s);
    const long double z = s / (root + 1 - s);
    const long double d = 1 + z * (2 - z);
    const long double a = 1 + z * (4 + z);
    const long double b = 2 + z * (1 - z) * (1 - z);
    const long double p = 5 + z * (15 + z * (-6 + z * (-2 + z * (5 - z))));
    const long double denominator = (1 + z) * d * d * d;
    coefficients c;
    c.c2 = static_cast<double>(z * z);
    c.c1 = 1 - c.c2;
    c.c3 = 0;
    c.c4 = static_cast<double>((1 - z) * a * a / denominator);
    c.c5 = static_cast<double>(z * z * p / denominator);
    c.v = static_cast<double>(1 / (root + 1 - 2 * s));
    c.v5 = alpha * (a * b / p);
    // c2 and c5 may round to zero at extreme ratios. v5 is evaluated
    // independently, so this limiting rounding never divides by zero.
    if (!finite(c) || c.c1 < 0 || c.c2 < 0 || c.c4 < 0 || c.c5 < 0 ||
        c.v <= 0 || c.v5 < 0) {
        return false;
    }
    result = c;
    return true;
}

} // namespace detail

inline bool make_coefficients(double alpha, coefficients& result) {
    return detail::make_coefficients_from_ratio(alpha, result);
}

// Taking the ratio in extended precision also supports finite timesteps whose
// ratio exceeds the double range. The physical growth cap is checked before
// converting to alpha, including at subnormal timestep scales.
inline bool make_step_coefficients(double previous_dt, double dt, coefficients& result) {
    if (!(previous_dt > 0) || !std::isfinite(previous_dt) ||
        !(dt > 0) || !std::isfinite(dt) ||
        static_cast<long double>(dt) > static_cast<long double>(previous_dt) * maximum_growth) {
        return false;
    }
    return detail::make_coefficients_from_ratio(
        static_cast<long double>(previous_dt) / dt, result);
}

// For the production c3=0 family, c1*v-alpha*c2 = (1+2z-z*z)/(2*(1+z)).
// This avoids multiplying an overflowing double ratio by a tiny weight.
inline double stage_abscissa(const coefficients& c) {
    const double z = std::sqrt(c.c2);
    return (1 + z * (2 - z)) / (2 * (1 + z));
}

// Return the largest representable supported timestep no larger than requested.
// The old Euler duration h*v5 approaches 2*previous_dt/5 as h tends to zero;
// an insufficient historical FE bound can therefore still require RK3.
inline double select_step(double previous_dt, double requested_dt,
    double current_fe, double previous_fe) {
    if (!(previous_dt > 0) || !std::isfinite(previous_dt) ||
        !(requested_dt > 0) || !std::isfinite(requested_dt) ||
        !(current_fe > 0) || !(previous_fe > 0) ||
        static_cast<long double>(previous_fe) <= 0.4L * previous_dt) {
        return 0;
    }
    // v >= 1/2, so 2*current_fe is also an upper bound. This avoids a search
    // across the whole exponent range after a sudden reduction in the CFL limit.
    const long double cap = std::min(static_cast<long double>(requested_dt),
        std::min(static_cast<long double>(previous_dt) * maximum_growth,
            2 * static_cast<long double>(current_fe)));
    double upper = static_cast<double>(cap);
    if (static_cast<long double>(upper) > cap) {
        upper = std::nextafter(upper, 0.0);
    }
    if (!(upper > 0)) {
        return 0;
    }
    const auto admissible = [=](double dt) {
        coefficients c;
        return make_step_coefficients(previous_dt, dt, c) &&
            static_cast<long double>(dt) * c.v <= current_fe &&
            static_cast<long double>(dt) * c.v5 <= previous_fe;
    };
    if (admissible(upper)) {
        return upper;
    }
    // First find a positive admissible point, with no scale-dependent floor.
    // Each halving removes at least one binary exponent until subnormals end.
    double lower = upper / 2;
    while (lower > 0 && !admissible(lower)) {
        upper = lower;
        lower /= 2;
    }
    if (!(lower > 0)) {
        return 0;
    }
    // Both Euler durations increase with h on 0<h/previous_dt<=1.9. The
    // bracket now spans a factor of two, so 64 bisections resolve double ulps.
    for (int i = 0; i < 64; ++i) {
        const double middle = lower + (upper - lower) / 2;
        if (middle == lower || middle == upper) {
            break;
        }
        if (admissible(middle)) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    return lower;
}

} // namespace ssp_multistep
