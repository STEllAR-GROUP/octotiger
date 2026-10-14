// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include <cmath>
#include <limits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define OCTOTIGER_SOURCE_HD __host__ __device__ inline
#else
#define OCTOTIGER_SOURCE_HD inline
#endif

namespace radiation { namespace implicit_source {

template <class Real> struct State {
    Real radiation[4]; // E, Q=F/c
    Real momentum[3];
    Real gas_energy, internal_energy;
};
template <class Real> struct Parameters {
    Real density, absorption, extinction, log_alpha, c, reduced_c_factor;
    bool marshak = false; // Existing linear-emission verification problem.
    bool fixed_material = false; // Stationary Marshak medium: heating, without recoil.
    bool prescribed_scattering = false; // Prescribed rest medium: scattering without heating or recoil.
};
enum class Failure {
    none, invalid_input, no_admissible_seed, singular_jacobian, no_descent, iteration_limit,
    inadmissible_iterate
};
template <class Real> struct Diagnostics {
    int iterations = 0;
    Real residual = Real(0);
    Failure failure = Failure::none;
};

namespace detail {
template <class R> inline constexpr R machine_epsilon = std::numeric_limits<R>::epsilon();
template <class R> inline constexpr R largest = std::numeric_limits<R>::max();
template <class R> OCTOTIGER_SOURCE_HD bool finite(R x) {
    return x == x && x <= largest<R> && x >= -largest<R>;
}
template <class R> OCTOTIGER_SOURCE_HD R maximum(R a, R b) { return a > b ? a : b; }
template <class R> OCTOTIGER_SOURCE_HD R norm(const R* x) { return ::hypot(::hypot(x[0], x[1]), x[2]); }
template <class R> OCTOTIGER_SOURCE_HD bool finite_state(const State<R>& state) {
    if (!finite(state.radiation[0]) || !finite(state.internal_energy) || !finite(state.gas_energy)) {
        return false;
    }
    for (int d = 0; d < 3; ++d) {
        if (!finite(state.radiation[d + 1]) || !finite(state.momentum[d])) {
            return false;
        }
    }
    return true;
}
template <class R> OCTOTIGER_SOURCE_HD bool physical(const State<R>& state) {
    if (!finite_state(state) || state.radiation[0] < 0 || state.internal_energy < 0) {
        return false;
    }
    const R energy = state.radiation[0];
    if (energy == 0) {
        return state.radiation[1] == 0 && state.radiation[2] == 0 && state.radiation[3] == 0;
    }
    // Match M1::ConservedState::checkState's squared normalized-flux bound.
    // Reject a certainly nonphysical component before dividing, so even an
    // extremely small positive E cannot overflow Q/E with traps enabled.
    R flux_squared = 0;
    for (int d = 1; d < 4; ++d) {
        const R magnitude = ::fabs(state.radiation[d]);
        if (magnitude > energy && magnitude - energy > energy) {
            return false;
        }
        const R normalized = state.radiation[d] / energy;
        flux_squared += normalized * normalized;
    }
    return flux_squared <= R(1) + R(4096) * detail::machine_epsilon<R>;
}

// Compute the bounded backward-Euler weights without overflowing h*c*chi.
template <class R> OCTOTIGER_SOURCE_HD void weights(R h, R c, R chi, R eps, R& inverse, R& weight) {
    if (h == 0 || chi == 0) {
        inverse = 1;
        weight = 0;
        return;
    }
    const R logarithm = ::log(h) + ::log(c) + ::log(chi) + ::log(eps);
    if (logarithm >= 0) {
        const R reciprocal = ::exp(-logarithm);
        weight = R(1) / (R(1) + reciprocal);
        inverse = reciprocal * weight;
    } else {
        const R eta = ::exp(logarithm);
        inverse = R(1) / (R(1) + eta);
        weight = eta * inverse;
    }
}

// e + w*alpha*e^4 = rhs. Scaling bounds both coefficients in [0,1],
// with one exactly one, so the positive root stays in [0.72449...,1].
// emission returns w*alpha*e^4 without evaluating an unscaled fourth power.
template <class R> OCTOTIGER_SOURCE_HD bool thermal(R rhs, R w, R log_alpha, bool marshak,
    R& gas, R& emission, R& derivative_fraction) {
    if (!finite(rhs) || rhs < 0 || !finite(w) || w < 0) {
        return false;
    }
    if (w == 0 || rhs == 0) {
        gas = rhs;
        emission = 0;
        derivative_fraction = 0;
        return true;
    }
    if (marshak) {
        const R inverse = R(1) / (R(1) + w);
        gas = rhs * inverse;
        derivative_fraction = w * inverse;
        emission = rhs * derivative_fraction;
        return finite(gas) && finite(emission);
    }
    const R log_rhs = ::log(rhs);
    const R log_coefficient = ::log(w) + log_alpha;
    const R quartic_scale = (log_rhs - log_coefficient) / R(4);
    const bool linear_bound = log_rhs <= quartic_scale;
    const R scale = linear_bound ? rhs : ::exp(quartic_scale);
    const R a = linear_bound ? R(1) : scale / rhs;
    const R exponent = log_coefficient + R(3) * log_rhs;
    const R b = linear_bound ? ::exp(exponent < 0 ? exponent : R(0)) : R(1);
    if (!finite(scale) || scale <= 0 || !finite(a) || !finite(b) || a < 0 || a > 1 || b < 0 || b > 1) {
        return false;
    }
    R y = 1;
    bool converged = false;
    for (int iteration = 0; iteration < 16; ++iteration) {
        const R y2 = y * y;
        const R residual = a * y + b * y2 * y2 - R(1);
        if (::fabs(residual) <= R(8) * detail::machine_epsilon<R>) {
            converged = true;
            break;
        }
        y -= residual / (a + R(4) * b * y2 * y);
    }
    if (!converged || !(y > 0) || y > R(1)) {
        return false;
    }
    gas = scale * y;
    emission = rhs * b * (y * y) * (y * y);
    // This is (4*w*alpha*e^3)/(1+4*w*alpha*e^3), evaluated without
    // division by a tiny e or an overflowing emission derivative.
    if (emission > gas) {
        derivative_fraction = R(1) / (R(1) + R(.25) * gas / emission);
    } else {
        const R ratio = gas > 0 ? R(4) * (emission / gas) : R(0);
        derivative_fraction = ratio / (R(1) + ratio);
    }
    return finite(gas) && gas >= 0 && finite(emission) && emission >= 0;
}

template <class R> struct Evaluation {
    State<R> state;
    R residual[3], jacobian[3][3], relative_residual;
};
template <class R> struct Stage {
    State<R> rhs;
    Parameters<R> material;
    R h, chat, inverse_absorption, thermal_weight, inverse_extinction, extinction_weight;
    R initial_velocity[3];
};

// Eliminate gas momentum/energy by their exact RSLA-weighted invariants.
// The dual-energy offset g-e-|m|^2/(2rho), if present, is preserved.
template <class R> OCTOTIGER_SOURCE_HD bool evaluate(const Stage<R>& stage, const R* Q,
    Evaluation<R>& result, bool* inadmissible = nullptr) {
    if (inadmissible) {
        *inadmissible = false;
    }
    const auto& rhs = stage.rhs;
    const auto& p = stage.material;
    const R eps = p.reduced_c_factor;
    const R mu = (R(1) / p.density) / stage.chat;
    R velocity[3], kinetic_change = 0, velocity_Q = 0;
    for (int d = 0; d < 3; ++d) {
        if (!finite(Q[d])) {
            return false;
        }
        const R dm = (rhs.radiation[d + 1] - Q[d]) / stage.chat;
        result.state.momentum[d] = rhs.momentum[d] + dm;
        velocity[d] = result.state.momentum[d] / p.density;
        kinetic_change += dm * (stage.initial_velocity[d] + R(.5) * dm / p.density);
        velocity_Q += velocity[d] * Q[d];
        result.state.radiation[d + 1] = Q[d];
    }
    const R work_coefficient = R(2) * p.absorption - p.extinction;
    const R work = work_coefficient * velocity_Q;
    const R inverse_h = stage.inverse_absorption * stage.h;
    const R thermal_rhs = rhs.internal_energy - kinetic_change + stage.thermal_weight * rhs.radiation[0] - inverse_h * work;
    R gas, emission, fraction;
    if (!finite(mu) || !finite(kinetic_change) || !finite(velocity_Q) || !finite(work) ||
        !thermal(thermal_rhs, stage.thermal_weight, p.log_alpha, p.marshak, gas, emission, fraction)) {
        return false;
    }
    const R E = ::fma(eps * inverse_h, work,
        ::fma(eps, emission, stage.inverse_absorption * rhs.radiation[0]));
    result.state.radiation[0] = E;
    result.state.internal_energy = gas;
    result.state.gas_energy = rhs.gas_energy + (rhs.radiation[0] - E) / eps;
    if (!physical(result.state)) {
        if (inadmissible) {
            *inadmissible = finite_state(result.state);
        }
        return false;
    }
    // M1: H=(2E+sqrt(4E^2-3Q^2))/3 and E*I+P=H*I+Q Q^T/H.
    R f[3] = {0, 0, 0}, f2 = 0;
    if (E > 0) {
        for (int d = 0; d < 3; ++d) {
            f[d] = Q[d] / E;
            f2 += f[d] * f[d];
        }
    }
    const R root = ::sqrt(R(4) - R(3) * f2);
    const R H = E * ((R(2) + root) / R(3));
    const R H_E = (R(2) + R(4) / root) / R(3);
    const R Z_over_H = H > 0 ? velocity_Q / H : R(0);
    R dE[3], dH[3], dZ[3];
    for (int d = 0; d < 3; ++d) {
        dZ[d] = velocity[d] - mu * Q[d];
        dE[d] = eps * (fraction * velocity[d] / stage.chat +
            (R(1) - fraction) * inverse_h * work_coefficient * dZ[d]);
        dH[d] = H_E * dE[d] - f[d] / root;
    }
    const R drag_weight = stage.extinction_weight / p.c;
    R residual_norm = 0;
    for (int i = 0; i < 3; ++i) {
        const R beta = H > 0 ? Q[i] / H : R(0);
        const R advective_flux = H * velocity[i] + Q[i] * Z_over_H;
        result.residual[i] = Q[i] - stage.inverse_extinction * rhs.radiation[i + 1] - drag_weight * advective_flux;
        residual_norm = maximum(residual_norm, ::fabs(result.residual[i]));
        for (int j = 0; j < 3; ++j) {
            const R delta = i == j ? R(1) : R(0);
            const R derivative = dH[j] * (velocity[i] - beta * Z_over_H) + beta * dZ[j] +
                delta * (Z_over_H - H * mu);
            result.jacobian[i][j] = delta - drag_weight * derivative;
            if (!finite(result.jacobian[i][j])) {
                return false;
            }
        }
    }
    const R scale = maximum(E, stage.inverse_extinction * norm(rhs.radiation + 1));
    result.relative_residual = scale > 0 ? residual_norm / scale : residual_norm;
    return finite(result.relative_residual);
}

template <class R> OCTOTIGER_SOURCE_HD bool newton_direction(const Evaluation<R>& value, R* direction) {
    R matrix[3][4];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            matrix[i][j] = value.jacobian[i][j];
        }
        matrix[i][3] = -value.residual[i];
    }
    for (int j = 0; j < 3; ++j) {
        int pivot = j;
        for (int i = j + 1; i < 3; ++i) {
            if (::fabs(matrix[i][j]) > ::fabs(matrix[pivot][j])) {
                pivot = i;
            }
        }
        if (!finite(matrix[pivot][j]) || matrix[pivot][j] == 0) {
            return false;
        }
        if (pivot != j) {
            for (int k = j; k < 4; ++k) {
                const R swap = matrix[j][k];
                matrix[j][k] = matrix[pivot][k];
                matrix[pivot][k] = swap;
            }
        }
        const R diagonal = matrix[j][j];
        for (int k = j; k < 4; ++k) {
            matrix[j][k] /= diagonal;
        }
        for (int i = 0; i < 3; ++i) {
            if (i != j) {
                const R factor = matrix[i][j];
                for (int k = j; k < 4; ++k) {
                    matrix[i][k] -= factor * matrix[j][k];
                }
            }
        }
    }
    for (int i = 0; i < 3; ++i) {
        direction[i] = matrix[i][3];
        if (!finite(direction[i])) {
            return false;
        }
    }
    return true;
}
} // namespace detail

// Solve Y = rhs + h*S(Y), with supplied stage-frozen opacities. Retains the
// code's first-order-v/c near-LTE momentum model, not the full unequal-opacity
// mixed-frame four-force. A finite rhs need not be realizable (e.g. an explicit
// pressure-gradient contribution); the accepted solution must be realizable.
// Failure leaves output untouched. No global floating-point environment changes.
template <class R> OCTOTIGER_SOURCE_HD bool solve(const State<R>& rhs, const Parameters<R>& p,
    R h, State<R>& output, Diagnostics<R>* diagnostics = nullptr) {
    Diagnostics<R> status;
    const auto fail = [&](Failure failure) {
        status.failure = failure;
        if (diagnostics) {
            *diagnostics = status;
        }
        return false;
    };
    if (!detail::finite(h) || h < 0 || !detail::finite(p.density) || p.density <= 0 ||
        !detail::finite(p.c) || p.c <= 0 || !detail::finite(p.reduced_c_factor) ||
        p.reduced_c_factor <= 0 || p.reduced_c_factor > 1 ||
        !detail::finite(p.absorption) || p.absorption < 0 || !detail::finite(p.extinction) || p.extinction < 0 ||
        !detail::finite(p.log_alpha) || !detail::finite(rhs.gas_energy) || !detail::finite(rhs.internal_energy)) {
        return fail(Failure::invalid_input);
    }
    if (p.prescribed_scattering && (p.absorption != 0 || p.marshak || p.fixed_material)) {
        return fail(Failure::invalid_input);
    }
    for (int d = 0; d < 4; ++d) {
        if (!detail::finite(rhs.radiation[d]) || (d < 3 && (!detail::finite(rhs.momentum[d]) ||
                (p.prescribed_scattering && rhs.momentum[d] != 0)))) {
            return fail(Failure::invalid_input);
        }
    }
    if (h == 0 || (p.absorption == 0 && p.extinction == 0)) {
        if (!detail::physical(rhs)) {
            return fail(Failure::no_admissible_seed);
        }
        output = rhs;
        if (diagnostics) {
            *diagnostics = status;
        }
        return true;
    }
    detail::Stage<R> stage;
    stage.rhs = rhs;
    stage.material = p;
    stage.h = h;
    stage.chat = p.c * p.reduced_c_factor;
    if (!detail::finite(stage.chat) || stage.chat <= 0) {
        return fail(Failure::invalid_input);
    }
    R absorption_weight;
    detail::weights(h, p.c, p.absorption, p.reduced_c_factor, stage.inverse_absorption, absorption_weight);
    stage.thermal_weight = absorption_weight / p.reduced_c_factor;
    detail::weights(h, p.c, p.extinction, p.reduced_c_factor, stage.inverse_extinction, stage.extinction_weight);
    if (p.prescribed_scattering) {
        // This is the constrained source equation at the implicit stage, not
        // a coupled update followed by restoring the prescribed material.
        // The external support receives the absorbed radiation momentum.
        State<R> result = rhs;
        for (int d = 1; d < 4; ++d) {
            result.radiation[d] = stage.inverse_extinction * rhs.radiation[d];
        }
        if (!detail::physical(result)) {
            return fail(Failure::no_admissible_seed);
        }
        output = result;
        if (diagnostics) {
            *diagnostics = status;
        }
        return true;
    }
    if (p.fixed_material) {
        if (!p.marshak) {
            return fail(Failure::invalid_input);
        }
        // Solve the two linear energy equations together. Momentum absorbed
        // from the radiation is carried by the external support of the slab.
        State<R> result = rhs;
        R gas, emission, fraction;
        const R thermal_rhs = rhs.internal_energy + stage.thermal_weight * rhs.radiation[0];
        if (!detail::thermal(thermal_rhs, stage.thermal_weight, p.log_alpha, true,
                gas, emission, fraction)) {
            return fail(Failure::no_admissible_seed);
        }
        result.internal_energy = gas;
        result.gas_energy = rhs.gas_energy + (gas - rhs.internal_energy);
        result.radiation[0] = ::fma(p.reduced_c_factor, emission,
            stage.inverse_absorption * rhs.radiation[0]);
        for (int d = 1; d < 4; ++d) {
            result.radiation[d] = stage.inverse_extinction * rhs.radiation[d];
        }
        if (!detail::physical(result)) {
            return fail(Failure::no_admissible_seed);
        }
        output = result;
        if (diagnostics) {
            *diagnostics = status;
        }
        return true;
    }
    for (int d = 0; d < 3; ++d) {
        stage.initial_velocity[d] = rhs.momentum[d] / p.density;
    }
    detail::Evaluation<R> value;
    bool seeded = false;
    // Damped flux is important when the explicit stage rhs is far outside the
    // realizability cone. No candidate is clipped into an admissible state.
    const R seed_factors[4] = {stage.inverse_extinction, R(0), R(1), R(.5) * stage.inverse_extinction};
    for (R factor : seed_factors) {
        R Q[3];
        for (int d = 0; d < 3; ++d) {
            Q[d] = factor * rhs.radiation[d + 1];
        }
        detail::Evaluation<R> candidate;
        if (detail::evaluate(stage, Q, candidate) && (!seeded || candidate.relative_residual < value.relative_residual)) {
            value = candidate;
            seeded = true;
        }
    }
    if (!seeded) {
        return fail(Failure::no_admissible_seed);
    }
    const R tolerance = R(4096) * detail::machine_epsilon<R>;
    for (int iteration = 0; iteration < 40; ++iteration) {
        status.iterations = iteration;
        status.residual = value.relative_residual;
        if (value.relative_residual <= tolerance) {
            output = value.state;
            if (diagnostics) {
                *diagnostics = status;
            }
            return true;
        }
        R direction[3];
        if (!detail::newton_direction(value, direction)) {
            return fail(Failure::singular_jacobian);
        }
        R damping = 1;
        bool accepted = false;
        bool all_inadmissible = true;
        for (int backtrack = 0; backtrack < 32; ++backtrack) {
            R Q[3];
            for (int d = 0; d < 3; ++d) {
                Q[d] = value.state.radiation[d + 1] + damping * direction[d];
            }
            detail::Evaluation<R> candidate;
            bool inadmissible = false;
            const bool evaluated = detail::evaluate(stage, Q, candidate, &inadmissible);
            all_inadmissible = all_inadmissible && inadmissible;
            if (evaluated &&
                (candidate.relative_residual <= tolerance ||
                    candidate.relative_residual < (R(1) - R(.0001) * damping) * value.relative_residual)) {
                value = candidate;
                accepted = true;
                break;
            }
            damping *= R(.5);
        }
        if (!accepted) {
            // A physical seed can still lead Newton into the realizability
            // boundary. Distinguish that observed obstruction from an
            // admissible non-descent step or a numerical evaluation failure,
            // so the outer predictor can retry its stage iteration safely.
            return fail(all_inadmissible ? Failure::inadmissible_iterate : Failure::no_descent);
        }
    }
    status.iterations = 40;
    status.residual = value.relative_residual;
    return fail(Failure::iteration_limit);
}

} } // namespace radiation::implicit_source

#undef OCTOTIGER_SOURCE_HD
