/*
 * radiationFront.cpp
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

std::vector<real> analyticStreamingFront(real x_, real y_, real z_, real t) {
	FpeGuard fpeGuard{};
	CgsToCode const convert;
	auto const gamma = opts().gas_gamma;
	auto const xScale = opts().xscale;
	auto const c = physcon().c;
	std::vector<real> G(opts().n_fields, 0_R);
	std::vector<real> R(NRF, 0_R);

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

	auto xfront = x_ - c * t;
	while (xfront < -xScale) {
		xfront += 2_R * xScale;
	}
	while (xfront > +xScale) {
		xfront -= 2_R * xScale;
	}

	if (xfront < 0_R) {
		R[0] = 1_R - eps_R;
	} else {
		R[0] = 1e-10_R;
	}

	R[1] = R[0] * almostOne;
	R[2] = 0_R;
	R[3] = 0_R;

	G.insert(G.end(), R.begin(), R.end());
	return G;
}

std::vector<real> testStreamingFront(Real x, Real y, Real z, Real dx) {
	return analyticStreamingFront(x, y, z, 0_R);
}
