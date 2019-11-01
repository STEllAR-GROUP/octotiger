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

#include "octotiger/unitiger/safe_real.hpp"
#include "octotiger/test_problems/blast.hpp"
#include "octotiger/test_problems/exact_sod.hpp"

constexpr double clight = 1.0;

template<int NDIM>
int radiation_physics<NDIM>::field_count() {
	return nf_;
}

template<int NDIM>
void radiation_physics<NDIM>::physical_flux(const std::vector<safe_real> &U, std::vector<safe_real> &F, int dim, safe_real &am, safe_real &ap,
		std::array<safe_real, NDIM> &x, std::array<safe_real, NDIM> &vg) {
	const double c = clight;
	double fmag;
	const auto &er = U[er_i];
	const auto *fr = &U[fx_i];
	std::array<double, NDIM> n;
	std::array<double, NDIM> T;
	for (int this_dim = 0; this_dim < NDIM; this_dim++) {
		fmag += fr[fx_i + this_dim] * fr[fx_i + this_dim];
	}
	fmag = std::sqrt(fmag);
	const auto fedd = fmag / (c * er);
	const auto fmaginv = 1.0 / fmag;
	for (int this_dim = 0; this_dim < NDIM; this_dim++) {
		n[this_dim] = fr[this_dim] * fmaginv;
	}

	const auto f2 = fedd * fedd;
	const auto f3 = f2 * fedd;
	const auto f4 = f2 * f2;
	const auto f6 = f3 * f3;
	const auto s4m3f2 = sqrt(4 - (3 * f2));
	const auto den = (pow(5 + 2 * s4m3f2, 2) * s4m3f2);
	const auto lam_s = (-12 * f3 + fedd * (41 + 20 * s4m3f2)) / den;
	const auto lam_d = (2 * sqrt(3) * sqrt(-48 * f6 + 8 * f4 * (61 + 16 * s4m3f2) - f2 * (787 + 328 * s4m3f2) + (365 + 182 * s4m3f2))) / den;
	ap = lam_s * n[dim] + lam_d;
	am = lam_s * n[dim] - lam_d;

	const auto chi = (3 + 4 * fedd * fedd) / (5 + 2 * s4m3f2);
	for (int this_dim = 0; this_dim < NDIM; this_dim++) {
		T[this_dim] = (3 * chi - 1) / 2. * n[dim] * n[this_dim];
	}
	T[dim] += (1 - chi) / 2.;

	F[er_i] = fr[fx_i + dim];
	for (int this_dim = 0; this_dim < NDIM; this_dim++) {
		F[fx_i + this_dim] = er * T[this_dim];
	}

}

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::post_process(hydro::state_type &U, safe_real dx) {
	static const cell_geometry<NDIM, INX> geo;
}

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::source(hydro::state_type &dudt, const hydro::state_type &U, const hydro::flux_type &F, const hydro::x_type X, safe_real omega,
		safe_real dx) {
	static const cell_geometry<NDIM, INX> geo;

}

/*** Reconstruct uses this - GPUize****/

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::pre_angmom(const hydro::state_type &U, const hydro::recon_type<NDIM> &Q,
		std::array<safe_real, cell_geometry<NDIM, INX>::NANGMOM> &Z, std::array<std::array<safe_real, cell_geometry<NDIM, INX>::NDIR>, NDIM> &S, int i,
		safe_real dx) {
	static const cell_geometry<NDIM, INX> geo;
	for (int d = 0; d < geo.NDIR; d++) {
		if (d != geo.NDIR / 2) {
			const auto er = Q[er_i][i][d];
			for (int f = 0; f < NDIM; f++) {
				S[f][d] *= er;
			}
		}
	}
	for (int f = 0; f < geo.NANGMOM; f++) {
		const auto er = U[er_i][i];
		Z[f] *= er;
	}

}

/*** Reconstruct uses this - GPUize****/

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::post_angmom(const hydro::state_type &U, const hydro::recon_type<NDIM> &Q,
		std::array<safe_real, cell_geometry<NDIM, INX>::NANGMOM> &Z, std::array<std::array<safe_real, cell_geometry<NDIM, INX>::NDIR>, NDIM> &S, int i,
		safe_real dx) {
	static const cell_geometry<NDIM, INX> geo;
	for (int d = 0; d < geo.NDIR; d++) {
		if (d != geo.NDIR / 2) {
			const auto er = Q[er_i][i][d];
			for (int f = 0; f < NDIM; f++) {
				S[f][d] /= er;
			}
		}
	}
	for (int f = 0; f < geo.NANGMOM; f++) {
		const auto er = U[er_i][i];
		Z[f] /= er;
	}

}

/*** Reconstruct uses this - GPUize****/

template<int NDIM>
template<int INX>
const hydro::state_type& radiation_physics<NDIM>::pre_recon(const hydro::state_type &U, const hydro::x_type X, safe_real omega, bool angmom) {
	static const cell_geometry<NDIM, INX> geo;
	static const auto indices = geo.find_indices(0, geo.H_NX);
	static thread_local hydro::state_type V;
	V = U;
	const auto dx = X[0][geo.H_DNX] - X[0][0];
	for (int j = 0; j < geo.H_NX_X; j++) {
		for (int k = 0; k < geo.H_NX_Y; k++) {
			for (int l = 0; l < geo.H_NX_Z; l++) {
				const int i = geo.to_index(j, k, l);
				const auto er = V[er_i][i];
				const auto erinv = 1.0 / er;
				for (int dim = 0; dim < NDIM; dim++) {
					V[fx_i + dim][i] *= erinv;
					V[wx_i + dim][i] *= erinv;
				}
				static constexpr auto kdelta = geo.kronecker_delta();
				for (int n = 0; n < geo.NANGMOM; n++) {
					for (int m = 0; m < NDIM; m++) {
						for (int l = 0; l < NDIM; l++) {
							V[wx_i + n][i] -= kdelta[n][m][l] * X[m][i] * V[fx_i + l][i];
						}
					}
				}
			}
		}
	}
	return V;
}

/*** Reconstruct uses this - GPUize****/

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::post_recon(std::vector<std::vector<std::vector<safe_real>>> &Q, const hydro::x_type X, safe_real omega, bool angmom) {
	static const cell_geometry<NDIM, INX> geo;
	static const auto indices = geo.find_indices(2, geo.H_NX - 2);
	const auto dx = X[0][geo.H_DNX] - X[0][0];
	const auto xloc = geo.xloc();
	for (int d = 0; d < geo.NDIR; d++) {
		if (d != geo.NDIR / 2) {
			for (int j = 0; j < geo.H_NX_XM6; j++) {
				for (int k = 0; k < geo.H_NX_YM6; k++) {
					for (int l = 0; l < geo.H_NX_ZM6; l++) {
						const int i = geo.to_index(j + 3, k + 3, l + 3);
						const auto er = Q[er_i][d][i];
						static constexpr auto kdelta = geo.kronecker_delta();
						for (int n = 0; n < geo.NANGMOM; n++) {
							for (int m = 0; m < NDIM; m++) {
								for (int l = 0; l < NDIM; l++) {
									Q[wx_i + n][d][i] += kdelta[n][m][l] * (X[m][i] + 0.5 * xloc[d][m] * dx) * Q[fx_i + l][d][i];
								}
							}
							Q[wx_i + n][d][i] *= er;
						}
						for (int dim = 0; dim < NDIM; dim++) {
							Q[fx_i + dim][d][i] *= er;
						}
					}
				}
			}
		}
	}
}

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::analytic_solution(test_type test, hydro::state_type &U, const hydro::x_type &X, safe_real time) {
	static const cell_geometry<NDIM, INX> geo;
}

template<int NDIM>
template<int INX>
std::vector<typename hydro_computer<NDIM, INX, radiation_physics<NDIM>>::bc_type> radiation_physics<NDIM>::initialize(radiation_physics<NDIM>::test_type t,
		hydro::state_type &U, hydro::x_type &X) {
	static const cell_geometry<NDIM, INX> geo;

	std::vector<typename hydro_computer<NDIM, INX, radiation_physics<NDIM>>::bc_type> bc(2 * NDIM);

	for (int i = 0; i < 2 * NDIM; i++) {
		bc[i] = hydro_computer<NDIM, INX, radiation_physics<NDIM>>::OUTFLOW;
	}

	return bc;
}

#endif /* OCTOTIGER_UNITIGER_radiation_physics_IMPL_HPP_ */
