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
	U[ein_i] = U[egas_i];
	U[spc_i] = U[rho_i];
	return U;
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
		if (std::abs((int) phi) % 180 == 90) {
			cos_phi = 0.0;
		} else if (std::abs((int) phi) % 180 == 0) {
			sin_phi = 0.0;
		}
	}
	if (std::floor(theta) == theta) {
		if (std::abs((int) theta) % 180 == 90) {
			cos_theta = 0.0;
		} else if (std::abs((int) theta) % 180 == 0) {
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
	U[ein_i] = U[egas_i];
	U[egas_i] += s.rho * s.v * s.v / 2.0;
	U[spc_i] = s.rho;
	return U;
}
