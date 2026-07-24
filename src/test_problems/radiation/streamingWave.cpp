/*
 * radiationWave.cpp
 *
 *  Created on: Jun 22, 2026
 *      Author: dmarce1
 */

#include <hpx/hpx_init.hpp>

#include "octotiger/defs.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/hpxfft/hpxfft.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/radiation/opacities.hpp"

std::vector<real> analyticStreamingWave(real x_, real y_, real z_, real t) {
	FpeGuard fpeGuard{};
	CgsToCode const convert;
	auto const gamma = opts().gas_gamma;
	auto const xScale = opts().xscale;
	auto const c = physcon().c;
	std::vector<real> G(opts().n_fields, 0_R);
	std::vector<real> R(NRF, 0_R);
	//	constexpr auto almost1 = 1.0 - sqrt(std::numeric_limits<double>::epsilon());

	real const eint = 1_R;

	G[rho_i] = 1_R;
	G[egas_i] = eint * G[rho_i];
	G[tau_i] = gasEnergy2Entropy(G[rho_i], G[egas_i]); 

	auto const rhoinv = INVERSE(G[rho_i]);
	G[egas_i] += G[sx_i] * G[sx_i] * rhoinv / 2_R;
	G[egas_i] += G[sy_i] * G[sy_i] * rhoinv / 2_R;
	G[egas_i] += G[sz_i] * G[sz_i] * rhoinv / 2_R;
	for (int i = 0; i < opts().n_species; i++) {
		G[spc_i] = G[rho_i];
	}
	auto const n = normalize(Vector<Real, NDIM>({2_R, -3_R, 1_R}));
	Vector<Real, NDIM> x({x_, y_, z_});
	auto const omega = (n.dot(x) - c * t) / (2_R * xScale);
	R[0] = eps_R + 0.5_R + 0.5_R * cos(2_R * M_PI * omega);
//	std::cout << "almostOne = " << almostOne << "n = " << n << std::endl;
	R[1] = n[0] * R[0] * almostOne;
	R[2] = n[1] * R[0] * almostOne;
	R[3] = n[2] * R[0] * almostOne;

	G.insert(G.end(), R.begin(), R.end());
	return G;
}

std::vector<real> testStreamingWave(Real x, Real y, Real z, Real dx) {
	return analyticStreamingWave(x, y, z, 0_R);
}
