//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER____FLUX____HPP123
#define OCTOTIGER____FLUX____HPP123

#include "octotiger/unitiger/physics.hpp"
#include "octotiger/unitiger/physics_impl.hpp"

//#define FACE_ONLY_HYDRO

template<int NDIM, int INX, class PHYS>
timestep_t hydro_computer<NDIM, INX, PHYS>::flux(const hydro::state_type &U, const hydro::recon_type<NDIM> &Q, hydro::flux_type &F, hydro::x_type &X,
		safe_real omega) {

	PROFILE();

	timestep_t ts;
	ts.a = 0.0;
	static thread_local std::vector<safe_real> UR(nf_), UL(nf_), this_flux(nf_);

	static const cell_geometry<NDIM, INX> geo;

	static constexpr auto faces = geo.face_pts();
	static constexpr auto weights = geo.face_weight();
	static constexpr auto xloc = geo.xloc();
	static constexpr auto levi_civita = geo.levi_civita();

	const auto dx = X[0][geo.H_DNX] - X[0][0];

	for (int dim = 0; dim < NDIM; dim++) {

		const auto indices = geo.get_indexes(3, geo.face_pts()[dim][0]);

		// zero-initialize F
		for (int f = 0; f < nf_; f++) {
#pragma ivdep
			for (const auto &i : indices) {
				F[dim][f][i] = 0.0;
			}
		}

		for (const auto &i : indices) {
			safe_real ap = 0.0, am = 0.0;
			safe_real this_ap, this_am;
			for (int fi = 0; fi < geo.NFACEDIR; fi++) {
#ifdef FACE_ONLY_HYDRO
				if( fi != 0 ) {
					continue;
				}
#endif
				const auto d = faces[dim][fi];
				for (int f = 0; f < nf_; f++) {
					UR[f] = Q[f][d][i];
					UL[f] = Q[f][geo::flip_dim(d, dim)][i - geo.H_DN[dim]];
				}
				std::array < safe_real, NDIM > x;
				std::array < safe_real, NDIM > vg;
				for (int dim = 0; dim < NDIM; dim++) {
					x[dim] = X[dim][i] + 0.5 * xloc[d][dim] * dx;
				}
				if (NDIM > 1) {
					vg[0] = -omega * (X[1][i] + 0.5 * xloc[d][1] * dx);
					vg[1] = +omega * (X[0][i] + 0.5 * xloc[d][0] * dx);
					if (NDIM == 3) {
						vg[2] = 0.0;
					}
				} else {
					vg[0] = 0.0;
				}

				safe_real amr, apr, aml, apl;
				static thread_local std::vector<safe_real> FR(nf_), FL(nf_);

				PHYS::template physical_flux<INX>(UR, FR, dim, amr, apr, x, vg);
				PHYS::template physical_flux<INX>(UL, FL, dim, aml, apl, x, vg);
				this_ap = std::max(std::max(apr, apl), safe_real(0.0));
				this_am = std::min(std::min(amr, aml), safe_real(0.0));
#pragma ivdep
				for (int f = 0; f < nf_; f++) {
					if (this_ap - this_am != 0.0) {
						this_flux[f] = (this_ap * FL[f] - this_am * FR[f] + this_ap * this_am * (UR[f] - UL[f])) / (this_ap - this_am);
					} else {
						this_flux[f] = (FL[f] + FR[f]) / 2.0;
					}
				}
				am = std::min(am, this_am);
				ap = std::max(ap, this_ap);
#pragma ivdep
				for (int f = 0; f < nf_; f++) {
#ifdef FACE_ONLY_HYDRO
					F[dim][f][i] += this_flux[f];
#else
					F[dim][f][i] += weights[fi] * this_flux[f];
#endif
				}
			}
			const auto this_amax = std::max(ap, safe_real(-am));
			if (this_amax > ts.a) {
				ts.a = this_amax;
				ts.x = X[0][i];
				ts.y = X[1][i];
				ts.z = X[2][i];
				ts.ur = UL;
				ts.ul = UR;
				ts.dim = dim;
			}
		}
	}
	return ts;
}

#endif
