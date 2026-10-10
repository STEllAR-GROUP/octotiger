// Copyright (c) 2026 The STEllAR Group
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace ssp_multistep {

struct coefficients {
    double c1, c2, c3, c4, c5, v, v5;
};

// Avoid the singular limit alpha -> 1/2 and excessive timestep decreases.
// Outside this interval the caller starts fresh with SSPRK3.
constexpr double minimum_alpha = 0.7;
constexpr double maximum_alpha = 4.0;

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

// The nonlinear third-order conditions give
//   c1 = 1 - (v/(alpha+v))^2,
//   c4 = (alpha+1)^2 (alpha+v)^2 /
//        (2 alpha^2 v (alpha+2v)^2).
// The SSP boundary c3=0 has the positive root used below. Zero is a valid
// convex weight. This evaluates the paper's family exactly instead of using
// its empirical c4 curve fit, which can produce negative weights.
inline bool make_coefficients(double alpha, coefficients& result) {
    if (!std::isfinite(alpha) || alpha < minimum_alpha || alpha > maximum_alpha) {
        return false;
    }
    const long double a = alpha;
    const long double v = a / (std::sqrt(a * a + 2) + a - 2);
    const long double z = v / (a + v);
    coefficients c;
    c.c2 = static_cast<double>(z * z);
    c.c1 = 1 - c.c2;
    c.c3 = 0;
    c.c4 = static_cast<double>(detail::c4_from_v(a, v));
    c.c5 = 1 - c.c4;
    c.v = static_cast<double>(v);
    c.v5 = static_cast<double>(detail::previous_derivative_weight(a, v) / c.c5);
    if (!detail::finite(c) || c.c1 < 0 || c.c2 < 0 || c.c4 < 0 || c.c5 <= 0 ||
        c.v <= 0 || c.v5 < 0) {
        return false;
    }
    result = c;
    return true;
}

// Return the largest supported timestep no larger than requested_dt, or zero
// when history cannot safely be used and the caller must take an SSPRK3 step.
// The FE limits belong to F(y_n) and the cached F(y_{n-1}), respectively.
// The stage derivative must satisfy the same current FE limit.
inline double select_step(double previous_dt, double requested_dt,
    double current_fe, double previous_fe) {
    if (!(previous_dt > 0) || !std::isfinite(previous_dt) ||
        !(requested_dt > 0) || !std::isfinite(requested_dt) ||
        !(current_fe > 0) || !(previous_fe > 0)) {
        return 0;
    }
    double lower = previous_dt / maximum_alpha;
    double upper = std::min(requested_dt, previous_dt / minimum_alpha);
    if (previous_dt / lower > maximum_alpha) {
        lower = std::nextafter(lower, std::numeric_limits<double>::infinity());
    }
    if (previous_dt / upper < minimum_alpha) {
        upper = std::nextafter(upper, 0.0);
    }
    if (!(lower > 0) || lower > upper) {
        return 0;
    }
    const auto admissible = [=](double dt) {
        coefficients c;
        return make_coefficients(previous_dt / dt, c) && dt * c.v <= current_fe &&
            dt * c.v5 <= previous_fe;
    };
    if (!admissible(lower)) {
        return 0;
    }
    if (admissible(upper)) {
        return upper;
    }
    // Both dt*v and dt*v5 increase with dt on the supported branch. For the
    // latter, put z=v/(alpha+v), which increases with dt. The numerator of
    // d(v5/alpha)/dz has positive Bernstein coefficients on [0,1]:
    // 15,23,226/7,282/7,1572/35,316/7,296/7,40,48.
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
