//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/problem.hpp"
#include "octotiger/test_problems/exact_sod.hpp"

#include "octotiger/defs.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/options.hpp"
#include "octotiger/real.hpp"

#include <cmath>

OCTOTIGER_EXPORT std::vector<real> advection_test_init(real x, real y, real z, real dx) {
	return advection_test_analytic(x, y, z, 0.0);
}

OCTOTIGER_EXPORT std::vector<real> advection_test_analytic(real x, real y, real z, real t) {
	std::vector<real> U(opts().n_fields, 0.0);
	const real fgamma = grid::get_fgamma();
	const auto r0 = 1.0/3.0;
	constexpr auto x0 = 0.5;
	constexpr auto y0 = 0.0;
	constexpr auto z0 = 0.0;
	const auto dx = x - x0;
	const auto dy = y - y0;
	const auto dz = z - z0;
	const auto r = std::sqrt(dx * dx + dy * dy + dz * dz);
	U[rho_i] = r < r0 ? std::max(sin(M_PI * (r / r0)) / (M_PI * (r / r0)),1e-6) : 1.0e-6;
	U[egas_i] = 1.0e-6;
	U[tau_i] = std::pow(U[egas_i], 1.0 / fgamma);
	U[spc_i] = U[rho_i];
	return U;
}

namespace {

// Compact C3 bump with an elementary primitive, so initialization and the
// validation reference use exact finite-volume averages rather than samples.
real amr_advection_bump_primitive(real s) {
    s = std::max(real(-1.0), std::min(real(1.0), s));
    const real s2 = s * s;
    return s * (1.0 + s2 * (-4.0 / 3.0 + s2 * (6.0 / 5.0
        + s2 * (-4.0 / 7.0 + s2 / 9.0))));
}

real amr_advection_bump_average(real coordinate, real center, real dx) {
    constexpr real radius = 0.25;
    if (dx > 0.0) {
        return radius / dx * (amr_advection_bump_primitive((coordinate + dx / 2.0 - center) / radius)
            - amr_advection_bump_primitive((coordinate - dx / 2.0 - center) / radius));
    }
    const real s = (coordinate - center) / radius;
    const real value = std::max(real(0.0), 1.0 - s * s);
    return value * value * value * value;
}

std::vector<real> amr_advection_state(real x, real y, real z, real t, real dx) {
    std::vector<real> state(opts().n_fields, 0.0);
    const real density = 1.0 + 0.2 * amr_advection_bump_average(x, -0.4 + t, dx)
        * amr_advection_bump_average(y, 0.0, dx) * amr_advection_bump_average(z, 0.0, dx);
    const real thermal = 0.01 / (grid::get_fgamma() - 1.0);
    state[rho_i] = density;
    state[spc_i] = density;
    state[sx_i] = density; // Unit translation speed; the exact pressure is constant.
    state[egas_i] = thermal + 0.5 * density;
    state[tau_i] = std::pow(thermal, 1.0 / grid::get_fgamma());
    return state;
}

} // namespace

std::vector<real> amr_advection_init(real x, real y, real z, real dx) {
    return amr_advection_state(x, y, z, 0.0, dx);
}

std::vector<real> amr_advection_analytic(real x, real y, real z, real t) {
    return amr_advection_state(x, y, z, t, 0.0);
}

bool refine_amr_advection(integer level, integer max_level, real, real, real,
        std::vector<real> const& state, std::array<std::vector<real>, NDIM> const&) {
    return level < max_level && state[rho_i] > opts().refinement_floor;
}

std::vector<real> sod_shock_tube_init(real x, real y, real z, real dx) {
	return sod_shock_tube_analytic(x,y,z,-dx);
}

std::vector<real> sod_shock_tube_analytic(real x0, real y, real z, real t) {
	std::vector<real> U(opts().n_fields, 0.0);
        const real fgamma = grid::get_fgamma();

        const real theta = opts().sod_theta;
        real cos_theta = std::cos(theta * M_PI / 180.0);
        real sin_theta = std::sin(theta * M_PI / 180.0);
        const real phi = opts().sod_phi;
        real cos_phi = std::cos(phi * M_PI / 180.0);
        real sin_phi = std::sin(phi * M_PI / 180.0);

        real dx = 0.0;
        if (t < 0.0) {   // if t is negative initialize
                dx = -t; // dx is important for knowing which cells lay in both sides of the initial discontinuity.
                t = 0.0; // Those in-betweem cells are set to the average values, i.e., (q_l+q_r)/2
        }

        if (std::floor(phi) == phi) {     // handling special degrees that result in exactly zero
                if (std::abs((int)phi) % 180 == 90) {
                        cos_phi = 0.0;
                }
                else if (std::abs((int)phi) % 180 == 0) {
                        sin_phi = 0.0;
                }
        }
        if (std::floor(theta) == theta) {
                if (std::abs((int)theta) % 180 == 90) {
                        cos_theta = 0.0;
                }
                else if (std::abs((int)theta) % 180 == 0) {
                        sin_theta = 0.0;
                }

        }

        sod_state_t s;
        sod_init_t sod_initial;
        sod_initial.rhol = opts().sod_rhol;
        sod_initial.rhor = opts().sod_rhor;
        sod_initial.pl = opts().sod_pl;
        sod_initial.pr = opts().sod_pr;
        sod_initial.gamma = fgamma;
        real x = x0 * cos_theta * sin_phi + y * sin_theta * sin_phi + z * cos_phi;
        exact_sod(&s, &sod_initial, x, t, dx);
	U[rho_i] = s.rho;
	U[egas_i] = s.p / (fgamma - 1.0);
        U[sx_i] = s.rho * s.v * cos_theta * sin_phi;
        U[sy_i] = s.rho * s.v * sin_theta * sin_phi;
        U[sz_i] = s.rho * s.v * cos_phi;
	U[tau_i] = std::pow(U[egas_i], 1.0 / fgamma);
	U[egas_i] += s.rho * s.v * s.v / 2.0;
	U[spc_i] = s.rho;
	return U;
}
