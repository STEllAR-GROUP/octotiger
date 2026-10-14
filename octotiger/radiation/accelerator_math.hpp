// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include "octotiger/radiation/accelerator.hpp"

#include <cfloat>
#include <cmath>
#include <cstddef>

#ifndef RADIATION_HD
#if defined(__CUDACC__) || defined(__HIPCC__)
#define RADIATION_HD __host__ __device__ inline
#else
#define RADIATION_HD inline
#endif
#endif

// Device-callable counterpart of RadiationM1<double, 3>. Keep the equations and
// roundoff tolerance in step with m1.hpp; the host reference remains independent
// so accelerator parity tests detect changes in either implementation.
namespace radiation_accelerator {
namespace math {

constexpr double roundoff = 4096.0 * DBL_EPSILON;

RADIATION_HD bool finite(double x) {
    return x == x && x <= DBL_MAX && x >= -DBL_MAX;
}
RADIATION_HD double minimum(double x, double y) {
    return x < y ? x : y;
}
RADIATION_HD double maximum(double x, double y) {
    return x > y ? x : y;
}
RADIATION_HD bool finite4(const double* x) {
    for (int f = 0; f < 4; ++f) {
        if (!finite(x[f])) {
            return false;
        }
    }
    return true;
}
RADIATION_HD double norm2(const double* x) {
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
}
RADIATION_HD bool conserved_valid(const double* u) {
    if (!finite4(u) || u[0] < 0) {
        return false;
    }
    if (u[0] == 0) {
        return u[1] == 0 && u[2] == 0 && u[3] == 0;
    }
    double f[3];
    for (int d = 0; d < 3; ++d) {
        // Reject clearly inadmissible input before potentially overflowing Q/E.
        if (::fabs(u[d + 1]) > u[0] &&
            ::fabs(u[d + 1]) - u[0] > roundoff * u[0]) {
            return false;
        }
        f[d] = u[d + 1] / u[0];
    }
    return norm2(f) <= 1 + roundoff;
}
RADIATION_HD bool canonical(double* q) {
    if (!finite4(q) || q[0] < 0) {
        return false;
    }
    for (int d = 1; d < 4; ++d) {
        if (::fabs(q[d]) > 1 + roundoff) {
            return false;
        }
    }
    const double b2 = norm2(q + 1);
    if (!(b2 <= 1 + roundoff)) {
        return false;
    }
    if (q[0] == 0) {
        for (int f = 0; f < 4; ++f) {
            q[f] = 0;
        }
    } else if (b2 > 1) {
        const double scale = 1 / ::sqrt(b2);
        for (int d = 1; d < 4; ++d) {
            q[d] *= scale;
        }
    }
    return true;
}
RADIATION_HD bool to_primitives(const double* u, double* q) {
    if (!conserved_valid(u)) {
        return false;
    }
    if (u[0] == 0) {
        for (int f = 0; f < 4; ++f) {
            q[f] = 0;
        }
        return true;
    }
    double f[3];
    for (int d = 0; d < 3; ++d) {
        f[d] = u[d + 1] / u[0];
    }
    double f2 = norm2(f);
    if (f2 > 1) {
        const double scale = 1 / ::sqrt(f2);
        for (int d = 0; d < 3; ++d) {
            f[d] *= scale;
        }
        f2 = 1;
    }
    const double h_over_e = (2 + ::sqrt(4 - 3 * f2)) / 3;
    q[0] = u[0] * h_over_e;
    for (int d = 0; d < 3; ++d) {
        q[d + 1] = f[d] * (1 / h_over_e);
    }
    return finite4(q);
}
RADIATION_HD bool to_conserved(const double* q, double* u) {
    const double b2 = minimum(1, norm2(q + 1));
    u[0] = q[0] * (0.25 * (3 + b2));
    for (int d = 1; d < 4; ++d) {
        u[d] = q[0] * q[d];
    }
    return conserved_valid(u);
}
RADIATION_HD double minmod(double left, double right) {
    if ((left > 0 && right > 0) || (left < 0 && right < 0)) {
        return ::copysign(minimum(minimum(1.3 * ::fabs(left), 1.3 * ::fabs(right)),
            0.5 * ::fabs(left + right)), left);
    }
    return 0;
}
RADIATION_HD bool reconstruct(const double* qm, const double* q,
    const double* qp, double* minus, double* plus) {
    double slope[4];
    for (int f = 0; f < 4; ++f) {
        slope[f] = 0.5 * minmod(q[f] - qm[f], qp[f] - q[f]);
    }
    const double a = norm2(slope + 1);
    const double b = ::fabs(q[1] * slope[1] + q[2] * slope[2] + q[3] * slope[3]);
    const double room = maximum(0, 1 - norm2(q + 1));
    if (a + 2 * b > room) {
        const double denominator = ::sqrt(b * b + a * room) + b;
        const double scale = denominator > 0 ? room / denominator : 0;
        for (int d = 1; d < 4; ++d) {
            slope[d] *= scale;
        }
    }
    for (int f = 0; f < 4; ++f) {
        minus[f] = q[f] - slope[f];
        plus[f] = q[f] + slope[f];
    }
    return canonical(minus) && canonical(plus);
}

struct FluxState {
    double u[4], flux[4], minus, plus;
};
RADIATION_HD bool physical_flux(const double* q, int normal, double c,
    double grid_velocity, FluxState& out) {
    if (!to_conserved(q, out.u)) {
        return false;
    }
    const double b2 = minimum(1, norm2(q + 1));
    const double pressure = 0.25 * (1 - b2) * q[0];
    const double bn = q[normal + 1];
    double transverse2 = 0;
    for (int d = 0; d < 3; ++d) {
        if (d != normal) {
            transverse2 += q[d + 1] * q[d + 1];
        }
    }
    const double one_minus_b2 = 1 - b2;
    const double root = ::sqrt(one_minus_b2 * (3 * one_minus_b2 + 2 * transverse2));
    out.minus = c * (2 * bn - root) / (3 - b2) - grid_velocity;
    out.plus = c * (2 * bn + root) / (3 - b2) - grid_velocity;
    out.flux[0] = c * out.u[normal + 1] - grid_velocity * out.u[0];
    for (int d = 0; d < 3; ++d) {
        const double pnd = q[0] * bn * q[d + 1] + (d == normal ? pressure : 0);
        out.flux[d + 1] = c * pnd - grid_velocity * out.u[d + 1];
    }
    return finite4(out.flux) && finite(out.minus) && finite(out.plus);
}
RADIATION_HD bool hll(const double* left, const double* right, int normal,
    double c, double grid_velocity, double* flux) {
    FluxState l, r;
    if (!physical_flux(left, normal, c, grid_velocity, l) ||
        !physical_flux(right, normal, c, grid_velocity, r)) {
        return false;
    }
    const double sm = minimum(l.minus, r.minus);
    const double sp = maximum(l.plus, r.plus);
    if (sm >= 0 || sp <= 0) {
        for (int f = 0; f < 4; ++f) {
            flux[f] = sm >= 0 ? l.flux[f] : r.flux[f];
        }
    } else {
        const double wr = sp / (sp - sm);
        const double wl = -sm / (sp - sm);
        const double viscosity = sm * wr;
        for (int f = 0; f < 4; ++f) {
            flux[f] = wr * l.flux[f] + wl * r.flux[f] + viscosity * (r.u[f] - l.u[f]);
        }
    }
    return finite4(flux);
}

RADIATION_HD bool grid_valid(Grid g, int r) {
    // n^3 must fit the int cell index used by every backend.
    return g.n > 0 && g.n <= 1290 && g.bw >= 2 && g.bw <= (g.n - 1) / 2 &&
        r >= 0 && static_cast<size_t>(r) < size_t(g.n) * g.n * g.n &&
        finite(g.c) && g.c > 0;
}
RADIATION_HD bool primitives(Grid g, const double* u, double* primitive, int r) {
    if (!grid_valid(g, r)) {
        return false;
    }
    const size_t n3 = size_t(g.n) * g.n * g.n;
    double cell[4], q[4];
    cell[0] = u[r];
    for (int f = 1; f < 4; ++f) {
        cell[f] = u[size_t(f) * n3 + r] / g.c;
    }
    if (!to_primitives(cell, q)) {
        return false;
    }
    for (int f = 0; f < 4; ++f) {
        primitive[size_t(f) * n3 + r] = q[f];
    }
    return true;
}
RADIATION_HD bool face_flux(Grid g, const double* primitive,
    const double* coordinates, double* flux, int normal, int r) {
    if (!grid_valid(g, r) || normal < 0 || normal >= 3) {
        return false;
    }
    const int cell[3] = {r / (g.n * g.n), (r / g.n) % g.n, r % g.n};
    for (int d = 0; d < 3; ++d) {
        if (cell[d] < g.bw || cell[d] >= g.n - g.bw + (d == normal ? 1 : 0)) {
            return true;
        }
    }
    const double c = g.c * g.reduced_c_factor;
    if (!finite(c) || c <= 0 || !finite(g.reduced_c_factor) ||
        g.reduced_c_factor > 1 || !finite(g.omega)) {
        return false;
    }
    const size_t n3 = size_t(g.n) * g.n * g.n;
    const int stride = normal == 0 ? g.n * g.n : (normal == 1 ? g.n : 1);
    double qm[4], q[4], qp[4], left[4], right[4], unused[4], result[4];
    for (int f = 0; f < 4; ++f) {
        qm[f] = primitive[size_t(f) * n3 + r - 2 * stride];
        q[f] = primitive[size_t(f) * n3 + r - stride];
        qp[f] = primitive[size_t(f) * n3 + r];
    }
    if (!reconstruct(qm, q, qp, unused, left)) {
        return false;
    }
    for (int f = 0; f < 4; ++f) {
        qm[f] = q[f];
        q[f] = qp[f];
        qp[f] = primitive[size_t(f) * n3 + r + stride];
    }
    if (!reconstruct(qm, q, qp, right, unused)) {
        return false;
    }
    double vg = 0;
    if (g.omega != 0 && normal != 2) {
        const int transverse = normal == 0 ? 1 : 0;
        const double x = 0.5 * (coordinates[size_t(transverse) * n3 + r - stride] +
            coordinates[size_t(transverse) * n3 + r]);
        vg = (normal == 0 ? -g.omega : g.omega) * x;
    }
    if (!finite(vg) || !hll(left, right, normal, c, vg, result)) {
        return false;
    }
    for (int f = 1; f < 4; ++f) {
        result[f] *= g.c;
    }
    if (!finite4(result)) {
        return false;
    }
    for (int f = 0; f < 4; ++f) {
        flux[size_t(normal * 4 + f) * n3 + r] = result[f];
    }
    return true;
}
RADIATION_HD bool advance_cell(Grid g, double* state, const double* flux,
    double* before_rotation, int r) {
    if (!grid_valid(g, r)) {
        return false;
    }
    const int cell[3] = {r / (g.n * g.n), (r / g.n) % g.n, r % g.n};
    for (int d = 0; d < 3; ++d) {
        if (cell[d] < g.bw || cell[d] >= g.n - g.bw) {
            return true;
        }
    }
    if (!finite(g.dt) || g.dt < 0 || !finite(g.dx) || g.dx <= 0 || !finite(g.omega)) {
        return false;
    }
    const size_t n3 = size_t(g.n) * g.n * g.n;
    const double factor = g.dt / g.dx;
    double next[4], normalized[4];
    for (int f = 0; f < 4; ++f) {
        next[f] = state[size_t(f) * n3 + r] - factor * (
            (flux[size_t(f) * n3 + r + g.n * g.n] - flux[size_t(f) * n3 + r]) +
            (flux[size_t(4 + f) * n3 + r + g.n] - flux[size_t(4 + f) * n3 + r]) +
            (flux[size_t(8 + f) * n3 + r + 1] - flux[size_t(8 + f) * n3 + r]));
    }
    const double fx = next[1], fy = next[2];
    if (g.omega != 0) {
        const double angle = g.omega * g.dt;
        if (!finite(angle)) {
            return false;
        }
        const double cs = ::cos(angle), sn = ::sin(angle);
        next[1] = cs * fx + sn * fy;
        next[2] = cs * fy - sn * fx;
    }
    normalized[0] = next[0];
    for (int f = 1; f < 4; ++f) {
        normalized[f] = next[f] / g.c;
    }
    if (!finite4(next) || !conserved_valid(normalized)) {
        return false;
    }
    before_rotation[r] = fx;
    before_rotation[n3 + r] = fy;
    for (int f = 0; f < 4; ++f) {
        state[size_t(f) * n3 + r] = next[f];
    }
    return true;
}

// S&O backward Euler energy exchange, separately factored so its scalar
// positive-quartic solver can be replaced without changing transport/coupling.
RADIATION_HD bool thermal_exchange(double e, double E, double eta,
    double log_alpha, bool marshak, double eps, double& gas, double& radiation) {
    if (!finite(e) || !finite(E) || E < 0 || !finite(eta) || eta < 0 ||
        !finite(eps) || eps <= 0 || eps > 1 || !finite(log_alpha)) {
        return false;
    }
    if (eta == 0) {
        gas = e;
        radiation = E;
        return e >= 0;
    }
    const double inverse = 1 / (1 + eps * eta);
    const double weight = eta * inverse;
    const double rhs = e + weight * E;
    if (!finite(rhs) || rhs < 0) {
        return false;
    }
    if (marshak) {
        gas = rhs / (1 + weight);
        radiation = inverse * E + eps * weight * gas;
    } else if (rhs == 0) {
        gas = 0;
        radiation = inverse * E;
    } else {
        const double log_coefficient = ::log(weight) + log_alpha;
        const double log_rhs = ::log(rhs);
        const double quartic_log_scale = (log_rhs - log_coefficient) / 4;
        const bool linear_bound = log_rhs <= quartic_log_scale;
        const double scale = linear_bound ? rhs : ::exp(quartic_log_scale);
        const double linear = linear_bound ? 1 : scale / rhs;
        const double quartic = linear_bound ? ::exp(minimum(0, log_coefficient + 3 * log_rhs)) : 1;
        if (!finite(linear) || !finite(quartic) || linear < 0 || linear > 1 ||
            quartic < 0 || quartic > 1 || maximum(linear, quartic) != 1) {
            return false;
        }
        // Preserve the CPU solver: measured A100 batches favored this
        // safeguarded early-exit Newton iteration over both the analytic
        // quartic and fixed-step Newton alternatives retained in the tests.
        double lo = 0, hi = 1, y = 1;
        bool converged = false;
        for (int iteration = 0; iteration < 80; ++iteration) {
            const double y2 = y * y;
            const double residual = linear * y + quartic * y2 * y2 - 1;
            if (::fabs(residual) <= 16 * DBL_EPSILON) {
                gas = scale * y;
                radiation = inverse * E + eps * rhs * quartic * y2 * y2;
                converged = true;
                break;
            }
            if (residual > 0) {
                hi = y;
            } else {
                lo = y;
            }
            const double next = y - residual / (linear + 4 * quartic * y2 * y);
            y = next > lo && next < hi ? next : 0.5 * (lo + hi);
        }
        if (!converged) {
            return false;
        }
    }
    return finite(gas) && gas >= 0 && finite(radiation) && radiation >= 0;
}
RADIATION_HD bool couple_cell(CouplingInput const& input, double dt, double c,
    double eps, bool marshak, CouplingOutput& output) {
    if (!conserved_valid(input.radiation) || !finite(c) || c <= 0 ||
        !finite(eps) || eps <= 0 || eps > 1 || !finite(dt) || dt < 0 ||
        !finite(input.density) || input.density <= 0 ||
        !finite(input.gas_energy) || !finite(input.internal_energy) || input.internal_energy < 0 ||
        !finite(input.absorption) || input.absorption < 0 ||
        !finite(input.extinction) || input.extinction < 0 || !finite(input.log_alpha)) {
        return false;
    }
    for (int d = 0; d < 3; ++d) {
        if (!finite(input.momentum[d])) {
            return false;
        }
    }
    CouplingOutput next;
    if (dt == 0 || (input.absorption == 0 && input.extinction == 0)) {
        for (int f = 0; f < 4; ++f) {
            next.radiation[f] = input.radiation[f];
        }
        for (int d = 0; d < 3; ++d) {
            next.momentum[d] = input.momentum[d];
        }
        next.gas_energy = input.gas_energy;
        next.internal_energy = input.internal_energy;
        output = next;
        return true;
    }
    const double transport_c = eps * c;
    if (!finite(transport_c) || transport_c <= 0) {
        return false;
    }
    double q[4], velocity[3];
    if (!to_primitives(input.radiation, q)) {
        return false;
    }
    const double b2 = minimum(1, norm2(q + 1));
    const double pressure = 0.25 * (1 - b2) * q[0];
    double beta_v = 0, velocity_Q = 0;
    for (int d = 0; d < 3; ++d) {
        velocity[d] = input.momentum[d] * (1 / input.density);
        beta_v += q[d + 1] * velocity[d];
        velocity_Q += velocity[d] * input.radiation[d + 1];
    }
    const double eta = dt * transport_c * input.extinction;
    if (!finite(eta) || !finite(beta_v) || !finite(velocity_Q)) {
        return false;
    }
    const double inverse = 1 / (1 + eta);
    const double weight = eta * inverse;
    double kinetic_change = 0;
    for (int d = 0; d < 3; ++d) {
        const double advective_flux = ((input.radiation[0] + pressure) * velocity[d] +
            q[0] * q[d + 1] * beta_v) / c;
        next.radiation[d + 1] = inverse * input.radiation[d + 1] + weight * advective_flux;
        const double dm = (input.radiation[d + 1] - next.radiation[d + 1]) / transport_c;
        next.momentum[d] = input.momentum[d] + dm;
        kinetic_change += dm * (velocity[d] + 0.5 * dm / input.density);
    }
    const double explicit_work = dt * (2 * input.absorption - input.extinction) * velocity_Q;
    const double radiation_star = input.radiation[0] + eps * explicit_work;
    const double gas_star = input.internal_energy - explicit_work - kinetic_change;
    if (!thermal_exchange(gas_star, radiation_star, dt * c * input.absorption,
            input.log_alpha, marshak, eps, next.internal_energy, next.radiation[0])) {
        return false;
    }
    next.gas_energy = input.gas_energy + (input.radiation[0] - next.radiation[0]) / eps;
    if (!conserved_valid(next.radiation) || !finite(next.gas_energy)) {
        return false;
    }
    for (int d = 0; d < 3; ++d) {
        if (!finite(next.momentum[d])) {
            return false;
        }
    }
    output = next;
    return true;
}

} // namespace math
} // namespace radiation_accelerator
