//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#pragma once

#include "octotiger/unitiger/physics.hpp"
#include "octotiger/unitiger/physics_impl.hpp"

template<int NDIM, int INX, class PHYS>
void hydro_computer<NDIM, INX, PHYS>::advance(const hydro::state_type &U0, hydro::state_type &U, const std::vector<std::vector<std::vector<safe_real>>>& q,const hydro::flux_type &F, const hydro::x_type &X,
		safe_real dx, safe_real dt, safe_real beta, safe_real omega) {
	static thread_local std::vector<std::vector<safe_real>> dudt(nf_, std::vector < safe_real > (geo::H_N3));
	for (int f = 0; f < nf_; f++) {
		for (const auto &i : geo::find_indices(geo::H_BW, geo::H_NX - geo::H_BW)) {
			dudt[f][i] = 0.0;
		}
	}
	for (int dim = 0; dim < NDIM; dim++) {
		for (int f = 0; f < nf_; f++) {
			for (const auto &i : geo::find_indices(geo::H_BW, geo::H_NX - geo::H_BW)) {
				const auto fr = F[dim][f][i + geo::H_DN[dim]];
				const auto fl = F[dim][f][i];
				dudt[f][i] -= (fr - fl) * INVERSE(dx);
			}
		}
	}
	PHYS::template derivative_source<INX>(dudt, U,q,X, omega, dx);
	PHYS ::template source<INX>(dudt, U, F, X, omega, dx);
	for (int f = 0; f < nf_; f++) {
		for (const auto &i : geo::find_indices(geo::H_BW, geo::H_NX - geo::H_BW)) {
			safe_real u0 = U0[f][i];
			safe_real u1 = U[f][i] + dudt[f][i] * dt;
			U[f][i] = u0 * (1.0 - beta) + u1 * beta;
		}
	}

}

