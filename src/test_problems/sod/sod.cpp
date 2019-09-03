//  Copyright (c) 2019 Dominic C Marcello
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

std::vector<real> sod_shock_tube_init(real x, real y, real z, real ) {
	return sod_shock_tube_analytic(x,y,z,0.0);
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
	U[spc_i] = s.rho;
	return U;
}
