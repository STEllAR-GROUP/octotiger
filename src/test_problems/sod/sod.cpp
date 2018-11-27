/*
 * rotating_star.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: dmarce1
 */

#include "../../defs.hpp"
#include "sod.hpp"



std::vector<real> sod_shock_tube_init(real x0, real y, real z, real ) {
	return sod_shock_tube_analytic(x0,y0,z0,0.0);
}

std::vector<real> sod_shock_tube_analytic(real x0, real y, real z, real t) {
	std::vector<real> U(opts().n_fields, 0.0);
	const real fgamma = grid::get_fgamma();
	sod_state_t s;
	real x = (x0 + y + z) / std::sqrt(3.0);
	exact_sod(&s, &sod_init, x, t);
	U[rho_i] = s.rho;
	U[egas_i] = s.p / (fgamma - 1.0);
	U[sx_i] = s.rho * s.v / std::sqrt(3.0);
	U[sy_i] = s.rho * s.v / std::sqrt(3.0);
	U[sz_i] = s.rho * s.v / std::sqrt(3.0);
	U[tau_i] = std::pow(U[egas_i], 1.0 / fgamma);
	U[egas_i] += s.rho * s.v * s.v / 2.0;
	U[spc_ac_i] = s.rho;
	return U;
}
