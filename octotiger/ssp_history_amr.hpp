// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace ssp_multistep {

struct prolonged_history_cell {
    std::array<double, 8> state;
    std::array<double, 8> rhs;
};

// Conservative piecewise-linear refinement using only the historical block.
// Freeze each minmod choice from the state (or the derivative where the state
// is locally constant), then apply the identical linear stencil to both. At block boundaries the three-point one-sided
// derivative is limited by twice the inward difference. Smooth reconstruction
// is at least second-order, including those boundaries. This common mapping
// alone does not establish a positivity or TVD theorem for the transferred pair.
template<class Values>
prolonged_history_cell prolong_history_cell(const Values& state, const Values& rhs,
        int nx, const std::array<int, 3>& cell) {
    if (nx < 3 || state.size() != size_t(nx) * nx * nx || rhs.size() != state.size()) {
        throw std::runtime_error("Invalid SSP history reconstruction shape");
    }
    for (int dim = 0; dim < 3; ++dim) {
        if (cell[dim] < 0 || cell[dim] >= nx) {
            throw std::runtime_error("Invalid SSP history reconstruction index");
        }
    }
    const std::array<int, 3> stride = {nx * nx, nx, 1};
    const int center = cell[0] * stride[0] + cell[1] * stride[1] + cell[2];
    std::array<double, 3> state_slope{}, rhs_slope{};
    for (int dim = 0; dim < 3; ++dim) {
        std::array<int, 3> index;
        std::array<double, 3> left, right;
        if (cell[dim] == 0) {
            index = {center, center + stride[dim], center + 2 * stride[dim]};
            left = {-1.5, 2.0, -0.5};
            right = {-2.0, 2.0, 0.0};
        } else if (cell[dim] == nx - 1) {
            index = {center, center - stride[dim], center - 2 * stride[dim]};
            left = {1.5, -2.0, 0.5};
            right = {2.0, -2.0, 0.0};
        } else {
            index = {center - stride[dim], center, center + stride[dim]};
            left = {-1.0, 1.0, 0.0};
            right = {0.0, -1.0, 1.0};
        }
        double a = 0.0, b = 0.0;
        for (int k = 0; k < 3; ++k) {
            a += left[k] * double(state[index[k]]);
            b += right[k] * double(state[index[k]]);
        }
        if (a == 0.0 && b == 0.0) {
            // Either candidate leaves this locally constant state unchanged.
            // Choose from the derivative instead of discarding its gradient.
            for (int k = 0; k < 3; ++k) {
                a += left[k] * double(rhs[index[k]]);
                b += right[k] * double(rhs[index[k]]);
            }
        }
        if ((a > 0.0 && b > 0.0) || (a < 0.0 && b < 0.0)) {
            const auto& weights = std::abs(a) <= std::abs(b) ? left : right;
            for (int k = 0; k < 3; ++k) {
                state_slope[dim] += weights[k] * double(state[index[k]]);
                rhs_slope[dim] += weights[k] * double(rhs[index[k]]);
            }
        }
    }
    prolonged_history_cell result;
    for (int child = 0; child < 8; ++child) {
        result.state[child] = double(state[center]);
        result.rhs[child] = double(rhs[center]);
        for (int dim = 0; dim < 3; ++dim) {
            const double offset = (child & (1 << dim)) ? 0.25 : -0.25;
            result.state[child] += offset * state_slope[dim];
            result.rhs[child] += offset * rhs_slope[dim];
        }
    }
    return result;
}

// Compare the ideal-gas thermal cone without overflowing E*rho or momentum
// squares. Divide momentum by the larger square-root scale first, avoiding an
// overflowing intermediate when density and energy have very different units.
inline bool ideal_gas_history_admissible(double rho, double energy,
        double sx, double sy, double sz) {
    if (!(rho > 0.0) || !(energy >= 0.0) || !std::isfinite(rho)
            || !std::isfinite(energy) || !std::isfinite(sx)
            || !std::isfinite(sy) || !std::isfinite(sz)) {
        return false;
    }
    if (energy == 0.0) {
        return sx == 0.0 && sy == 0.0 && sz == 0.0;
    }
    const double a = std::sqrt(rho), b = std::sqrt(energy);
    const double large = std::max(a, b), small = std::min(a, b);
    const double x = (sx / large) / small;
    const double y = (sy / large) / small;
    const double z = (sz / large) / small;
    return std::hypot(x, y, z) <= std::sqrt(2.0);
}

// Scale the entire reconstruction about its coarse mean. If the historical
// Euler endpoint is admissible in the coarse cell, protect that endpoint too.
// An inadmissible coarse endpoint cannot be repaired by conservative subcell
// slope scaling; in that case this helper protects only the past state.
inline double history_pair_positivity_factor(const prolonged_history_cell& values,
        double state, double rhs, double duration) {
    double theta = 1.0;
    const auto bound = [&theta](double mean, double fine) {
        if (mean >= 0.0 && fine < 0.0) {
            theta = std::min(theta, mean / (mean - fine));
        }
    };
    for (int child = 0; child < 8; ++child) {
        bound(state, values.state[child]);
        if (duration > 0.0 && std::isfinite(duration)) {
            bound(state + duration * rhs,
                values.state[child] + duration * values.rhs[child]);
        }
    }
    return theta < 1.0 ? std::nextafter(theta, 0.0) : theta;
}

} // namespace ssp_multistep
