// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace radiation {

inline double limited_slope(double left, double right) {
    double const centered = 0.5 * (left + right);
    double const theta = 64.0 / 37.0;
    return (std::copysign(0.5, left) + std::copysign(0.5, right)) *
        std::min({theta * std::abs(left), theta * std::abs(right), std::abs(centered)});
}

// Conservative volume interpolation from hpxr2c_radiation: complete_rad_amr_boundary.
// u_child = u_parent + scale * Σ_d sign(child,d) slope_d.                 (1)
// slope_d includes the quarter-parent-cell displacement. One scale for all
// fields and children preserves the parent average; independently clipping
// children would not. The caller supplies model-specific admissibility.
template <size_t Fields, class Admissible>
std::array<std::array<double, Fields>, 8> interpolate_children(
    std::array<double, Fields> const& center,
    std::array<std::array<double, Fields>, 3> const& slope, Admissible admissible) {
    if (!admissible(center)) {
        throw std::invalid_argument("Invalid parent radiation state for AMR interpolation");
    }
    for (auto const& direction : slope) {
        for (double value : direction) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Nonfinite radiation AMR slope");
            }
        }
    }
    std::array<std::array<double, Fields>, 8> children;
    double scale = 1.0;
    for (;;) {
        bool accepted = true;
        for (size_t child = 0; child < children.size(); ++child) {
            auto& state = children[child];
            for (size_t field = 0; field < Fields; ++field) {
                state[field] = center[field];
                for (size_t dimension = 0; dimension < 3; ++dimension) {
                    state[field] += scale * ((child & (1 << (2 - dimension))) ? 1 : -1) *
                        slope[dimension][field];
                }
            }
            accepted = admissible(state) && accepted;
        }
        if (accepted) {
            return children;
        }
        scale *= 0.5;
        if (scale < std::numeric_limits<double>::epsilon()) {
            children.fill(center);
            return children;
        }
    }
}

// FLD and M1 both require finite moments and nonnegative energy. A flux-cone
// constraint belongs to the M1 caller, not generic transfer (lab-frame FLD
// flux can include advective radiation enthalpy).
template <size_t Fields>
bool finite_nonnegative_energy(std::array<double, Fields> const& state) {
    static_assert(Fields > 0);
    for (double value : state) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return state[0] >= 0.0;
}

}    // namespace radiation
