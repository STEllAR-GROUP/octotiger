/*
 * physics_impl.hpp
 *
 *  Created on: Sep 30, 2019
 *      Author: dmarce1
 */

//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef OCTOTIGER_UNITIGER_radiation_physics_HPP12443_
#define OCTOTIGER_UNITIGER_radiation_physics_HPP12443_

#include "octotiger/math/Real.hpp"
#include "octotiger/test_problems/blast.hpp"
#include "octotiger/test_problems/exact_sod.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/radiation/m1.hpp"



template<int NDIM>
int radiation_physics<NDIM>::field_count() {
	return nf_;
}

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::physical_flux(const std::vector<Real> &U, std::vector<Real> &F, int dim,
		Real &am, Real &ap, std::array<Real, NDIM> &x, std::array<Real, NDIM> &vg) {
    // Legacy adapter: physical units and Hanawa-Audit M1 closure, just as rad_grid.
    radiation_m1::state u{};
    for (int f = 0; f < 1 + NDIM; ++f) u[f] = U[f];
    const auto result = radiation_m1::physical_flux(u, dim, physcon().c, vg[dim]);
    am = std::min(Real(0), result.minus);
    ap = std::max(Real(0), result.plus);
    for (int f = 0; f < 1 + NDIM; ++f) F[f] = result.flux[f];
}

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::post_process(hydro::state_type &U, Real dx) {
	static const cell_geometry<NDIM, INX> geo;
}

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::source(hydro::state_type &dudt, const hydro::state_type &U, const hydro::flux_type &F,
		const hydro::x_type X, Real omega, Real dx) {
	static const cell_geometry<NDIM, INX> geo;

}

/*** Reconstruct uses this - GPUize****/

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::pre_angmom(const hydro::state_type &U, const hydro::recon_type<NDIM> &Q,
		std::array<Real, cell_geometry<NDIM, INX>::NANGMOM> &Z,
		std::array<std::array<Real, cell_geometry<NDIM, INX>::NDIR>, NDIM> &S, int i, Real dx) {
    // Radiation carries only E and F; no auxiliary angular-momentum fields.

}

/*** Reconstruct uses this - GPUize****/

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::post_angmom(const hydro::state_type &U, const hydro::recon_type<NDIM> &Q,
		std::array<Real, cell_geometry<NDIM, INX>::NANGMOM> &Z,
		std::array<std::array<Real, cell_geometry<NDIM, INX>::NDIR>, NDIM> &S, int i, Real dx) {
    // Radiation carries only E and F; no auxiliary angular-momentum fields.

}

/*** Reconstruct uses this - GPUize****/

template<int NDIM>
template<int INX>
const hydro::state_type& radiation_physics<NDIM>::pre_recon(const hydro::state_type &U, const hydro::x_type X,
		Real omega, bool angmom) {
    static thread_local hydro::state_type V;
    V = U;
    for (std::size_t i = 0; i < U[0].size(); ++i) {
        const Real E = U[er_i][i];
        for (int d = 0; d < NDIM; ++d) V[fx_i + d][i] = E > 0 ? U[fx_i + d][i] / physcon().c / E : 0;
    }
    return V;
}

/*** Reconstruct uses this - GPUize****/

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::post_recon(std::vector<std::vector<std::vector<Real>>> &Q, const hydro::x_type X,
		Real omega, bool angmom) {
    // Only used by the legacy unitiger driver; rad_grid reconstructs six faces itself.
    for (std::size_t direction = 0; direction < Q[0].size(); ++direction) {
        for (std::size_t i = 0; i < Q[0][direction].size(); ++i) {
            Real f2 = 0;
            for (int d = 0; d < NDIM; ++d) f2 += Q[fx_i + d][direction][i] * Q[fx_i + d][direction][i];
            const Real scale = physcon().c * Q[er_i][direction][i] / std::sqrt(std::max(Real(1), f2));
            for (int d = 0; d < NDIM; ++d) Q[fx_i + d][direction][i] *= scale;
        }
    }
}

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::analytic_solution(test_type test, hydro::state_type &U, const hydro::x_type &X,
		Real time) {
	static const cell_geometry<NDIM, INX> geo;
}

template<int NDIM>
template<int INX>
std::vector<typename hydro_computer<NDIM, INX, radiation_physics<NDIM>>::bc_type> radiation_physics<NDIM>::initialize(
		radiation_physics<NDIM>::test_type t, hydro::state_type &U, hydro::x_type &X) {
	static const cell_geometry<NDIM, INX> geo;

	std::vector<typename hydro_computer<NDIM, INX, radiation_physics<NDIM>>::bc_type> bc(2 * NDIM);

	for (int i = 0; i < 2 * NDIM; i++) {
		bc[i] = hydro_computer<NDIM, INX, radiation_physics<NDIM>>::OUTFLOW;
	}

	for (int dim = 0; dim < NDIM; dim++) {
		X[dim].resize(geo.H_N3);
	}
	for (int f = 0; f < nf_; f++) {
		U[f].resize(geo.H_N3, 0.0);
	}

	const Real dx = 1.0 / INX;

	for (int i = 0; i < geo.H_N3; i++) {
		int k = i;
		int j = 0;
		for (int dim = 0; dim < NDIM; dim++) {
			X[NDIM - 1 - dim][i] = (((k % geo.H_NX) - geo.H_BW) + 0.5) * dx - 0.5;
			k /= geo.H_NX;
			j++;
		}
	}
	for (int i = 0; i < geo.H_N3; i++) {
		double xsum = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			xsum += X[dim][i];
		}
		if (xsum < 0.000001) {
			U[er_i][i] = 1.0;
		} else {
			U[er_i][i] = 1.0e-1;
		}
		U[fx_i][i] = 0.0;
//		U[fx_i][i] = U[er_i][i] * clight;
	}

	return bc;
}

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::enforce_outflows(hydro::state_type &U, const hydro::x_type &X, int face) {
	std::array<int, 3> lb, ub;
	static const cell_geometry<NDIM, INX> geo;

	for (int dim = 0; dim < NDIM; dim++) {
		lb[dim] = 0;
		ub[dim] = geo.H_N3;
	}
	for (int dim = NDIM; dim < 3; dim++) {
		lb[dim] = 0;
		ub[dim] = 1;
	}
	if (face % 2 == 0) {
		lb[face / 2] = 0;
		ub[face / 2] = geo.H_BW;
	} else {
		lb[face / 2] = geo.H_NX - geo.H_BW;
		ub[face / 2] = geo.H_NX;
	}
	const double c = clight;
	for (int j = lb[0]; j < ub[0]; j++) {
		for (int k = lb[1]; k < ub[1]; k++) {
			for (int l = lb[2]; l < ub[2]; l++) {
				const int i = geo.to_index(j, k, l);
				constexpr auto er_floor = 1.0e-10;
				for (int dim = 0; dim < NDIM; dim++) {
//					U[fx_i + dim][i] = 0.0;
				}
//				U[er_i][i] = er_floor;
			}
		}
	}
}


#endif /* OCTOTIGER_UNITIGER_radiation_physics_IMPL_HPP_ */
