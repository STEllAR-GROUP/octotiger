//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER____FLUX____HPP123
#define OCTOTIGER____FLUX____HPP123

#include "octotiger/fld_hydro.hpp"
#include "octotiger/unitiger/physics.hpp"
#include "octotiger/unitiger/physics_impl.hpp"
#include "octotiger/common_kernel/struct_of_array_data.hpp"

//#define FACE_ONLY_HYDRO

template<int NDIM, int INX, class PHYS>
timestep_t hydro_computer<NDIM, INX, PHYS>::flux(hydro::state_type const &U,
    hydro::recon_type<NDIM> const &Q, hydro::flux_type &F, hydro::x_type &X, safe_real omega,
    std::vector<std::vector<safe_real>> const *radiation_recon,
    std::array<std::vector<safe_real>, NDIM> const *radiation_acoustic,
    std::array<std::vector<safe_real>, NDIM> *radiation_flux) {

	PROFILE();
	// input Q, X
	// output F

	timestep_t ts{};
	ts.a = 0.0;
	// bunch of tmp containers
	static thread_local std::vector<safe_real> UR(nf_), UL(nf_), this_flux(nf_);

    // bunch of small helpers
	static const cell_geometry<NDIM, INX> geo;
	static constexpr auto faces = geo.face_pts();
	static constexpr auto weights = geo.face_weight();
	static constexpr auto xloc = geo.xloc();
	static constexpr auto levi_civita = geo.levi_civita();

	const auto dx = X[0][geo.H_DNX] - X[0][0];

  int max_index = 0;
	for (int dim = 0; dim < NDIM; dim++) {

		const auto indices = geo.get_indexes(3, geo.face_pts()[dim][0]);

		// zero-initialize F
		for (int f = 0; f < nf_; f++) {
#pragma ivdep
			for (const auto &i : indices) {
				F[dim][f][i] = 0.0;
			}
		}

        if (radiation_flux != nullptr) {
            for (auto const i : indices) {
                (*radiation_flux)[dim][i] = 0.0;
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
				// why store this?
				for (int f = 0; f < nf_; f++) { 
					UR[f] = Q[f][d][i];
					UL[f] = Q[f][geo::flip_dim(d, dim)][i - geo.H_DN[dim]];
				}
				std::array < safe_real, NDIM > x;
				std::array < safe_real, NDIM > vg;
				for (int dim = 0; dim < NDIM; dim++) {
					x[dim] = X[dim][i] + 0.5 * xloc[d][dim] * dx;
				}
				if HOST_CONSTEXPR (NDIM > 1) {
					vg[0] = -omega * (X[1][i] + 0.5 * xloc[d][1] * dx);
					vg[1] = +omega * (X[0][i] + 0.5 * xloc[d][0] * dx);
					if HOST_CONSTEXPR (NDIM == 3) {
						vg[2] = 0.0;
					}
				} else {
					vg[0] = 0.0;
				}

				safe_real amr, apr, aml, apl;
				static thread_local std::vector<safe_real> FR(nf_), FL(nf_);

				PHYS::template physical_flux<INX>(UR, FR, dim, amr, apr, x, vg);
				PHYS::template physical_flux<INX>(UL, FL, dim, aml, apl, x, vg);
                double radiation_left = 0.0;
                double radiation_right = 0.0;
                if (radiation_recon != nullptr) {
                    radiation_right = std::max(0.0, double((*radiation_recon)[d][i]));
                    radiation_left = std::max(0.0, double((*radiation_recon)[geo::flip_dim(d, dim)][i - geo.H_DN[dim]]));
                    double const vr = 0.5 * (apr + amr);
                    double const vl = 0.5 * (apl + aml);
                    double const cr = 0.5 * (apr - amr);
                    double const cl = 0.5 * (apl - aml);
                    double const crad_r = (*radiation_acoustic)[dim][i] * radiation_right / UR[physics<NDIM>::rho_i];
                    double const crad_l = (*radiation_acoustic)[dim][i - geo.H_DN[dim]] * radiation_left / UL[physics<NDIM>::rho_i];
                    double const speed_r = std::sqrt(cr * cr + crad_r);
                    double const speed_l = std::sqrt(cl * cl + crad_l);
                    apr = vr + speed_r;
                    amr = vr - speed_r;
                    apl = vl + speed_l;
                    aml = vl - speed_l;
                }

				this_ap = std::max(std::max(apr, apl), safe_real(0.0));
				this_am = std::min(std::min(amr, aml), safe_real(0.0));
#pragma ivdep
				for (int f = 0; f < nf_; f++) {
                    this_flux[f] = octotiger::fld::hll_flux(UL[f], UR[f], FL[f], FR[f], this_am, this_ap);
				}
                if (radiation_flux != nullptr) {
                    double const vl = UL[physics<NDIM>::sx_i + dim] / UL[physics<NDIM>::rho_i] - vg[dim];
                    double const vr = UR[physics<NDIM>::sx_i + dim] / UR[physics<NDIM>::rho_i] - vg[dim];
                    double const flux = octotiger::fld::hll_flux(radiation_left, radiation_right,
                        vl * radiation_left, vr * radiation_right, this_am, this_ap);
#ifdef FACE_ONLY_HYDRO
                    (*radiation_flux)[dim][i] += flux;
#else
                    (*radiation_flux)[dim][i] += weights[fi] * flux;
#endif
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
        max_index = i;
			}
		}
	}

	return ts;
}
#endif
