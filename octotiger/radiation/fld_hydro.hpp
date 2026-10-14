// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

// First-order CPU reference hyperbolic part of mixed-frame grey FLD.
// The implicit part must use thermal_target(), with velocity held fixed during
// that solve. The whole explicit+implicit substep is accepted or retried in
// scratch by the caller. No independent radiation momentum is advanced.
//
// Krumholz et al. (2007), https://arxiv.org/abs/astro-ph/0611003,
// equations (34), (38)--(42): retain the full tensor enthalpy flux and its
// matching velocity-squared exchange. Momentum uses the leading force -lambda
// grad(E). This is a nonrelativistic, leading-regime mixed-frame model, not a
// second-order relativistic moment system. In particular, do not clip the
// derived lab flux to cE: the comoving diffusion flux alone is flux limited.

#include "octotiger/radiation/fld.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <stddef.h>
#include <string>
#include <utility>
#include <vector>

namespace radiation {
namespace fld {
namespace hydro {

using vector = std::array<double, 3>;
using tensor = std::array<vector, 3>;
using mesh_type = octotiger::elliptic::composite_mesh;
using face_type = octotiger::elliptic::face;

struct state {
    double density = 0;
    vector momentum{};
    double gas_energy = 0; // Internal plus kinetic energy density.
    double radiation_energy = 0; // Lab-frame energy density.
};

inline double dot(const vector& a, const vector& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double norm(const vector& a) {
    return std::hypot(a[0], a[1], a[2]);
}

inline vector velocity(const state& u) {
    return {{u.momentum[0] / u.density, u.momentum[1] / u.density,
        u.momentum[2] / u.density}};
}

inline double kinetic_energy(const state& u) {
    return .5 * dot(u.momentum, velocity(u));
}

inline double internal_energy(const state& u) {
    return u.gas_energy - kinetic_energy(u);
}

inline bool admissible(const state& u) {
    return u.density > 0 && std::isfinite(u.density) &&
        std::isfinite(u.gas_energy) && std::isfinite(norm(u.momentum)) &&
        u.radiation_energy >= 0 && std::isfinite(u.radiation_energy) &&
        internal_energy(u) > 0 && std::isfinite(internal_energy(u));
}

struct closure {
    double energy = 0;
    double chi_rosseland = 0;
    double lambda = 1. / 3.;
    double eddington_factor = 1. / 3.;
    vector gradient{};
    vector diffusive_flux{};
    tensor eddington{{{{1. / 3., 0, 0}}, {{0, 1. / 3., 0}}, {{0, 0, 1. / 3.}}}};

    vector enthalpy_flux(const vector& v) const {
        vector result{};
        for (int i = 0; i < 3; ++i) {
            result[i] = energy * (v[i] + dot(eddington[i], v));
        }
        return result;
    }

    vector lab_flux(const vector& v) const {
        vector result = enthalpy_flux(v);
        for (int i = 0; i < 3; ++i) {
            result[i] += diffusive_flux[i];
        }
        return result;
    }

    vector force() const {
        vector result{};
        for (int i = 0; i < 3; ++i) {
            result[i] = -lambda * gradient[i];
        }
        return result;
    }

    // Frozen-velocity thermal equilibrium is B(u)=h E-2 v.Fd/c^2.
    // The NEGATIVE v.H/c^2 term cancels the leading dynamic-diffusion boost.
    // In particular uniform moving LTE has E=B/(1-4 beta^2/3), not E=B.
    double thermal_factor(const vector& v, double c) const {
        const vector beta{{v[0] / c, v[1] / c, v[2] / c}};
        double value = 1 - dot(beta, beta);
        for (int i = 0; i < 3; ++i) {
            value -= beta[i] * dot(eddington[i], beta);
        }
        return value;
    }

    double thermal_target(const vector& v, double c) const {
        // Scale each factor before multiplication so physical c does not
        // overflow a square, and tiny code-unit light speeds remain usable.
        double correction = 0;
        for (int i = 0; i < 3; ++i) {
            correction += 2 * (v[i] / c) * (diffusive_flux[i] / c);
        }
        return thermal_factor(v, c) * energy - correction;
    }
};

inline bool make_closure(double E, double chi, const vector& gradient, double c,
    closure& output) {
    const double g = norm(gradient);
    if (!(E >= 0) || !(chi >= 0) || !(c > 0) || !std::isfinite(E) ||
        !std::isfinite(chi) || !std::isfinite(g) || !std::isfinite(c)) {
        return false;
    }
    output = closure{};
    output.energy = E;
    output.chi_rosseland = chi;
    output.gradient = gradient;
    if (g == 0) {
        return true;
    }
    const long double optical = static_cast<long double>(chi) * E;
    const double R = optical > 0 ? static_cast<double>(g / optical)
        : std::numeric_limits<double>::infinity();
    output.lambda = fld::limiter(R);
    const double fraction = fld::limiter_flux_factor(R);
    output.eddington_factor = output.lambda + fraction * fraction;
    const double isotropic = .5 * (1 - output.eddington_factor);
    const double directed = .5 * (3 * output.eddington_factor - 1);
    vector direction{};
    for (int i = 0; i < 3; ++i) {
        direction[i] = gradient[i] / g;
        output.diffusive_flux[i] = -c * E * fraction * direction[i];
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            output.eddington[i][j] = directed * direction[i] * direction[j] +
                (i == j ? isotropic : 0);
        }
    }
    return std::isfinite(norm(output.diffusive_flux));
}

struct boundary_state {
    state value;
    double chi_rosseland = 0;
    vector radiation_gradient{};
};

using boundary_callback = std::function<boundary_state(
    const face_type&, const state&, const closure&)>;

inline boundary_state outflow_boundary(const state& interior, const closure& pressure) {
    return {interior, pressure.chi_rosseland, pressure.gradient};
}

inline boundary_state reflecting_boundary(const face_type& face, const state& interior,
    const closure& pressure) {
    auto exterior = outflow_boundary(interior, pressure);
    exterior.value.momentum[face.axis] *= -1;
    exterior.radiation_gradient[face.axis] *= -1;
    return exterior;
}

struct result {
    bool success = false;
    std::string message;
    std::vector<state> state_values;
    double stable_dt = 0;
    std::vector<double> mass_boundary_by_cell; // Step-integrated, positive inward.
    std::vector<vector> momentum_boundary_by_cell;
    std::vector<double> gas_boundary_by_cell;
    std::vector<double> radiation_boundary_by_cell;
    std::vector<double> radiation_work_by_cell; // Opposite gas work, not external energy.
    double conservation_error = 0; // Combined gas+radiation energy minus boundary input.
};

namespace detail {

inline double signal_speed(const state& u, const closure& pressure, double gamma) {
    const double p = (gamma - 1) * internal_energy(u);
    // A conservative bound for frozen-closure hyperbolic characteristics.
    // ||I+f|| <= 2, 0 <= lambda <= 1/3; the 2|v| term also bounds
    // anisotropic enthalpy advection. Gas sound speed alone is insufficient.
    return 2 * norm(velocity(u)) +
        std::sqrt((gamma * p + 2 * pressure.lambda * u.radiation_energy) / u.density);
}

inline state physical_flux(const state& u, const closure& pressure, int axis, double gamma) {
    const vector v = velocity(u);
    const double p = (gamma - 1) * internal_energy(u);
    state flux;
    flux.density = u.momentum[axis];
    for (int d = 0; d < 3; ++d) {
        flux.momentum[d] = u.momentum[d] * v[axis] + (d == axis ? p : 0);
    }
    flux.gas_energy = (u.gas_energy + p) * v[axis];
    flux.radiation_energy = pressure.enthalpy_flux(v)[axis];
    return flux;
}

inline state rusanov(const state& left, const state& right, const closure& pl,
    const closure& pr, int axis, double gamma, double speed) {
    const auto fl = physical_flux(left, pl, axis, gamma);
    const auto fr = physical_flux(right, pr, axis, gamma);
    state flux;
    flux.density = .5 * (fl.density + fr.density - speed * (right.density - left.density));
    for (int d = 0; d < 3; ++d) {
        flux.momentum[d] = .5 * (fl.momentum[d] + fr.momentum[d] -
            speed * (right.momentum[d] - left.momentum[d]));
    }
    flux.gas_energy = .5 * (fl.gas_energy + fr.gas_energy -
        speed * (right.gas_energy - left.gas_energy));
    flux.radiation_energy = .5 * (fl.radiation_energy + fr.radiation_energy -
        speed * (right.radiation_energy - left.radiation_energy));
    return flux;
}

inline void add_flux(state& u, const state& flux, double scale) {
    u.density += scale * flux.density;
    for (int d = 0; d < 3; ++d) {
        u.momentum[d] += scale * flux.momentum[d];
    }
    u.gas_energy += scale * flux.gas_energy;
    u.radiation_energy += scale * flux.radiation_energy;
}

} // namespace detail

inline result hyperbolic_step(const mesh_type& mesh, const std::vector<state>& input,
    const std::vector<double>& chi_rosseland, double dt, double c, double gamma,
    const boundary_callback& boundary, double cfl = .4) {
    result output;
    output.state_values = input;
    const size_t n = input.size();
    auto fail = [&](const std::string& message) {
        output.success = false;
        output.message = message;
        output.state_values = input;
        output.mass_boundary_by_cell.assign(n, 0);
        output.momentum_boundary_by_cell.assign(n, vector{});
        output.gas_boundary_by_cell.assign(n, 0);
        output.radiation_boundary_by_cell.assign(n, 0);
        output.radiation_work_by_cell.assign(n, 0);
        output.conservation_error = 0;
        return output;
    };
    if (n != mesh.cells().size() || n != chi_rosseland.size() || n == 0 ||
        !(dt >= 0) || !std::isfinite(dt) || !(c > 0) || !std::isfinite(c) ||
        !(gamma > 1) || !std::isfinite(gamma) || !(cfl > 0 && cfl <= .5)) {
        return fail("Invalid FLD hydro input or CFL");
    }
    std::vector<double> energies(n);
    for (size_t i = 0; i < n; ++i) {
        if (!admissible(input[i]) || !(chi_rosseland[i] >= 0) ||
            !std::isfinite(chi_rosseland[i])) {
            return fail("Nonpositive or nonfinite FLD hydro state");
        }
        if (norm(velocity(input[i])) > .1 * c) {
            return fail("FLD hydro requires nonrelativistic velocity |v|/c <= 0.1");
        }
        energies[i] = input[i].radiation_energy;
    }
    std::vector<closure> pressure(n);
    for (size_t i = 0; i < n; ++i) {
        if (!make_closure(energies[i], chi_rosseland[i], mesh.cell_gradient(i, energies),
                c, pressure[i])) {
            return fail("Invalid FLD hydro closure");
        }
    }
    // Use boundary traces when forming the force gradient. Evaluating trace
    // differences keeps constant radiation fields exactly force free.
    std::vector<vector> gradients(n);
    std::vector<boundary_state> exterior(mesh.faces().size());
    for (size_t f = 0; f < mesh.faces().size(); ++f) {
        const auto& face = mesh.faces()[f];
        if (face.boundary_side == 0) {
            const double difference = .5 * (energies[face.upper] - energies[face.lower]);
            gradients[face.lower][face.axis] += difference * face.area / mesh.volume(face.lower);
            gradients[face.upper][face.axis] += difference * face.area / mesh.volume(face.upper);
        } else {
            const size_t i = face.lower == octotiger::elliptic::boundary_index ? face.upper : face.lower;
            exterior[f] = boundary ? boundary(face, input[i], pressure[i])
                : outflow_boundary(input[i], pressure[i]);
            if (!admissible(exterior[f].value) || !(exterior[f].chi_rosseland >= 0) ||
                !std::isfinite(exterior[f].chi_rosseland)) {
                return fail("Invalid FLD hydro exterior state");
            }
            const double difference = .5 * (exterior[f].value.radiation_energy - energies[i]);
            gradients[i][face.axis] += face.boundary_side * difference * face.area / mesh.volume(i);
        }
    }
    for (size_t i = 0; i < n; ++i) {
        if (!make_closure(energies[i], chi_rosseland[i], gradients[i], c, pressure[i])) {
            return fail("Invalid FLD hydro boundary closure");
        }
    }
    std::vector<state> fluxes(mesh.faces().size());
    std::vector<double> speeds(n, 0);
    for (size_t f = 0; f < mesh.faces().size(); ++f) {
        const auto& face = mesh.faces()[f];
        state left, right;
        closure pl, pr;
        if (face.boundary_side == 0) {
            left = input[face.lower];
            right = input[face.upper];
            pl = pressure[face.lower];
            pr = pressure[face.upper];
        } else {
            const size_t i = face.lower == octotiger::elliptic::boundary_index ? face.upper : face.lower;
            // Re-evaluate the callback with the physical boundary gradient.
            exterior[f] = boundary ? boundary(face, input[i], pressure[i])
                : outflow_boundary(input[i], pressure[i]);
            closure pe;
            if (!admissible(exterior[f].value) ||
                norm(velocity(exterior[f].value)) > .1 * c ||
                !make_closure(exterior[f].value.radiation_energy, exterior[f].chi_rosseland,
                    exterior[f].radiation_gradient, c, pe)) {
                return fail("Invalid FLD hydro boundary closure");
            }
            if (face.boundary_side < 0) {
                left = exterior[f].value;
                pl = pe;
                right = input[i];
                pr = pressure[i];
            } else {
                left = input[i];
                pl = pressure[i];
                right = exterior[f].value;
                pr = pe;
            }
        }
        const double a = std::max(detail::signal_speed(left, pl, gamma),
            detail::signal_speed(right, pr, gamma));
        if (!std::isfinite(a)) {
            return fail("Nonfinite FLD hydro signal speed");
        }
        fluxes[f] = detail::rusanov(left, right, pl, pr, face.axis, gamma, a);
        if (face.lower != octotiger::elliptic::boundary_index) {
            speeds[face.lower] += face.area * a;
        }
        if (face.upper != octotiger::elliptic::boundary_index) {
            speeds[face.upper] += face.area * a;
        }
    }
    output.stable_dt = std::numeric_limits<double>::max();
    for (size_t i = 0; i < n; ++i) {
        if (speeds[i] > 0) {
            output.stable_dt = std::min(output.stable_dt, cfl * mesh.volume(i) / speeds[i]);
        }
    }
    if (dt > output.stable_dt * (1 + 8 * std::numeric_limits<double>::epsilon())) {
        return fail("FLD hydro step exceeds radiation-modified CFL");
    }
    output.mass_boundary_by_cell.assign(n, 0);
    output.momentum_boundary_by_cell.assign(n, vector{});
    output.gas_boundary_by_cell.assign(n, 0);
    output.radiation_boundary_by_cell.assign(n, 0);
    output.radiation_work_by_cell.assign(n, 0);
    std::vector<state> next = input;
    for (size_t f = 0; f < mesh.faces().size(); ++f) {
        const auto& face = mesh.faces()[f];
        const double factor = dt * face.area;
        if (face.lower != octotiger::elliptic::boundary_index) {
            detail::add_flux(next[face.lower], fluxes[f], -factor / mesh.volume(face.lower));
        }
        if (face.upper != octotiger::elliptic::boundary_index) {
            detail::add_flux(next[face.upper], fluxes[f], factor / mesh.volume(face.upper));
        }
        if (face.boundary_side != 0) {
            const size_t i = face.lower == octotiger::elliptic::boundary_index ? face.upper : face.lower;
            const double inward = -face.boundary_side * factor;
            output.mass_boundary_by_cell[i] += inward * fluxes[f].density;
            for (int d = 0; d < 3; ++d) {
                output.momentum_boundary_by_cell[i][d] += inward * fluxes[f].momentum[d];
            }
            output.gas_boundary_by_cell[i] += inward * fluxes[f].gas_energy;
            output.radiation_boundary_by_cell[i] += inward * fluxes[f].radiation_energy;
        }
    }
    long double conservation = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!admissible(next[i])) {
            return fail("FLD hydro transport produced inadmissible state");
        }
        const vector force = pressure[i].force();
        const vector old_velocity = velocity(next[i]);
        for (int d = 0; d < 3; ++d) {
            next[i].momentum[d] += dt * force[d];
        }
        const vector new_velocity = velocity(next[i]);
        double work = 0;
        for (int d = 0; d < 3; ++d) {
            work += .5 * dt * force[d] * (old_velocity[d] + new_velocity[d]);
        }
        // This midpoint impulse work is exactly the kinetic-energy increment
        // at fixed density. The opposite radiation increment conserves total
        // energy without heating the gas by a discretization artifact.
        next[i].gas_energy += work;
        next[i].radiation_energy -= work;
        output.radiation_work_by_cell[i] = -work * mesh.volume(i);
        if (!admissible(next[i]) || norm(new_velocity) > .1 * c) {
            return fail("FLD hydro force/work produced inadmissible state");
        }
        conservation += (static_cast<long double>(next[i].gas_energy) - input[i].gas_energy +
            static_cast<long double>(next[i].radiation_energy) - input[i].radiation_energy) * mesh.volume(i)
            - output.gas_boundary_by_cell[i] - output.radiation_boundary_by_cell[i];
    }
    output.state_values = std::move(next);
    output.conservation_error = static_cast<double>(conservation);
    output.success = true;
    output.message = "accepted hyperbolic candidate; implicit solve still required";
    return output;
}

} // namespace hydro
} // namespace fld
} // namespace radiation
