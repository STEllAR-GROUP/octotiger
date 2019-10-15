//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER____FLUX____HPP123
#define OCTOTIGER____FLUX____HPP123

#include "octotiger/unitiger/physics.hpp"
#include "octotiger/unitiger/physics_impl.hpp"



template<int NDIM, int INX>
safe_real hydro_computer<NDIM, INX>::flux(const hydro::state_type &U, const hydro::recon_type<NDIM> &Q, hydro::flux_type &F, hydro::x_type &X,
		safe_real omega) {

	static thread_local auto fluxes = std::vector < std::vector
			< std::vector<std::array<safe_real, geo::NFACEDIR>>
					>> (NDIM, std::vector < std::vector<std::array<safe_real, geo::NFACEDIR>>
							> (nf_, std::vector<std::array<safe_real, geo::NFACEDIR>>(geo::H_N3)));

	static const cell_geometry<NDIM, INX> geo;

	static constexpr auto faces = geo.face_pts();
	static constexpr auto weights = geo.face_weight();
	static constexpr auto xloc = geo.xloc();
	static constexpr auto kdelta = geo.kronecker_delta();

	const auto dx = X[0][geo.H_DNX] - X[0][0];

	safe_real amax = 0.0;
	for (int dim = 0; dim < NDIM; dim++) {
		std::vector<safe_real> UR(nf_), UL(nf_), this_flux(nf_);
		std::vector<safe_real> UR0(nf_), UL0(nf_);

		const auto indices = geo.get_indexes(3, geo.face_pts()[dim][0]);

		for (const auto &i : indices) {
			safe_real ap = 0.0, am = 0.0;
			safe_real this_ap, this_am;
#ifdef FACES_ONLY
#warning "Compiling with only face center fluxes"
			for (int fi = 0; fi < 1; fi++) {
#else
			for (int fi = 0; fi < geo.NFACEDIR; fi++) {
#endif
				const auto d = faces[dim][fi];
				for (int f = 0; f < nf_; f++) {
					UR0[f] = U[f][i];
					UL0[f] = U[f][i - geo.H_DN[dim]];
					UR[f] = Q[f][i][d];
					UL[f] = Q[f][i - geo.H_DN[dim]][geo::flip_dim(d, dim)];
				}
				std::array < safe_real, NDIM > x;
				std::array < safe_real, NDIM > vg;
				for( int dim = 0; dim < NDIM; dim++ ) {
					x[dim] = X[dim][i] + 0.5 * xloc[d][dim] * dx;
				}
				if constexpr (NDIM > 1) {
					vg[0] = -omega * (X[1][i] + 0.5 * xloc[d][1] * dx);
					vg[1] = +omega * (X[0][i] + 0.5 * xloc[d][0] * dx);
					if constexpr (NDIM == 3) {
						vg[2] = 0.0;
					}
				} else {
					vg[0] = 0.0;
				}
				physics < NDIM > ::flux(UL, UR, UL0, UR0, this_flux, dim, this_am, this_ap, x,vg);
				am = std::min(am, this_am);
				ap = std::max(ap, this_ap);
				for (int f = 0; f < nf_; f++) {
					fluxes[dim][f][i][fi] = this_flux[f];
				}
			}
			const auto this_amax = std::max(ap, safe_real(-am));
			if (this_amax > amax) {
				amax = this_amax;
			}
		}
		for (int f = 0; f < nf_; f++) {
			for (const auto &i : indices) {
				F[dim][f][i] = 0.0;
				for (int fi = 0; fi < geo.NFACEDIR; fi++) {
#ifdef FACES_ONLY
					constexpr auto w = 1.0;
#else
					const auto &w = weights[fi];
#endif
					F[dim][f][i] += w * fluxes[dim][f][i][fi];
				}
			}
		}
	}
	return amax;
}



#endif
