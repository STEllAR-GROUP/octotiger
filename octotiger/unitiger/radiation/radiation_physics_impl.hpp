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
#include "octotiger/physcon.hpp"

#define CHECK_FLUX( er, fx, fy, fz) if( ((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c) > 1 ) {printf( "flux exceded %s %i %e\n", __FILE__, __LINE__, sqrt(((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c))); abort();}

template<int NDIM>
int radiation_physics<NDIM>::field_count() {
	return nf_;
}

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::physical_flux(const std::vector<safe_real> &U, std::vector<safe_real> &F, int dim,
		safe_real &am, safe_real &ap, std::array<safe_real, NDIM> &x, std::array<safe_real, NDIM> &vg) {
	static const cell_geometry<NDIM, INX> geo;
	static constexpr auto levi_civita = geo.levi_civita();
	const double c = clight;
	double fmag = 0.0;
	const auto &er = U[er_i];
	const auto *fr = &U[fx_i];
	std::array<double, NDIM> n;
	std::array<double, NDIM> T;
	for (int d = 0; d < NDIM; d++) {
		fmag += fr[d] * fr[d];
	}
	fmag = std::sqrt(fmag);
	const auto fedd = fmag / (c * er);
	if (fmag > 0.0) {
		const auto fmaginv = 1.0 / fmag;
		for (int d = 0; d < NDIM; d++) {
			n[d] = fr[d] * fmaginv;
		}
	} else {
		for (int d = 0; d < NDIM; d++) {
			n[d] = 0.0;
		}
	}

	const auto f2 = fedd * fedd;
	const auto tmp2 = 4 - 3 * f2;
	const auto tmp = sqrt(tmp2);
	const auto tmp3 = (2.0 / 3.0) * (tmp2 - tmp) + 2 * n[dim] * n[dim] * (2 - f2 - tmp);
//	if( tmp3 < 0.0 ) {
//		printf( "%e %e %e\n",f2, (2.0 / 3.0) * (tmp2 - tmp) , 2 * n[dim] * n[dim] * (2 - f2 - tmp) );
//	}
	ap = (n[dim] * fedd + sqrt(tmp3)) / tmp;
	am = (n[dim] * fedd - sqrt(tmp3)) / tmp;
	ap *= physcon().c;
	am *= physcon().c;
	ap = std::max(ap, 0.0);
	am = std::min(am, 0.0);
	if (fabs(ap) > 1 || fabs(am) > 1) {
	//	printf("Error %s %i\n", __FILE__, __LINE__);
	//	abort();
	}
//	ap = physcon().c;
//	am = -physcon().c;
	const auto chi = (3 + 4 * fedd * fedd) / (5 + 2 * tmp);
	for (int d = 0; d < NDIM; d++) {
		T[d] = (3 * chi - 1) / 2. * n[dim] * n[d];
	}
	T[dim] += (1 - chi) / 2.;

	F[er_i] = fr[dim] - vg[dim] * er;
	for (int d = 0; d < NDIM; d++) {
		if (fabs(T[dim] - .333) > 0.1) {
		//	printf("%i %i %e\n", dim, d, T[d]);
		}
		F[fx_i + d] = er * T[d] - vg[dim] * fr[d];
	}

}

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::post_process(hydro::state_type &U, safe_real dx) {
	static const cell_geometry<NDIM, INX> geo;
}

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::source(hydro::state_type &dudt, const hydro::state_type &U, const hydro::flux_type &F,
		const hydro::x_type X, safe_real omega, safe_real dx) {
	static const cell_geometry<NDIM, INX> geo;

}

/*** Reconstruct uses this - GPUize****/

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::pre_angmom(const hydro::state_type &U, const hydro::recon_type<NDIM> &Q,
		std::array<safe_real, cell_geometry<NDIM, INX>::NANGMOM> &Z,
		std::array<std::array<safe_real, cell_geometry<NDIM, INX>::NDIR>, NDIM> &S, int i, safe_real dx) {
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
		std::array<safe_real, cell_geometry<NDIM, INX>::NANGMOM> &Z,
		std::array<std::array<safe_real, cell_geometry<NDIM, INX>::NDIR>, NDIM> &S, int i, safe_real dx) {
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
const hydro::state_type& radiation_physics<NDIM>::pre_recon(const hydro::state_type &U, const hydro::x_type X,
		safe_real omega, bool angmom) {
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
				CHECK_FLUX(V[er_i][i], V[fx_i][i], V[fy_i][i], V[fz_i][i]);
				for (int dim = 0; dim < NDIM; dim++) {
					V[fx_i + dim][i] *= erinv;
				}
		//		V[wx_i][i] = sqrt(V[fx_i][i] * V[fx_i][i] + V[fy_i][i] * V[fy_i][i] + V[fz_i][i] * V[fz_i][i]);
			}
		}
	}
	return V;
}

/*** Reconstruct uses this - GPUize****/

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::post_recon(std::vector<std::vector<std::vector<safe_real>>> &Q, const hydro::x_type X,
		safe_real omega, bool angmom) {
	static const cell_geometry<NDIM, INX> geo;
	const auto dx = X[0][geo.H_DNX] - X[0][0];
	const auto xloc = geo.xloc();
	for (int d = 0; d < geo.NDIR; d++) {
		if (d != geo.NDIR / 2) {
			for (int j = 0; j < geo.H_NX_XM4; j++) {
				for (int k = 0; k < geo.H_NX_YM4; k++) {
					for (int l = 0; l < geo.H_NX_ZM4; l++) {
						const int i = geo.to_index(j + 2, k + 2, l + 2);
						const auto er = Q[er_i][d][i];
						static constexpr auto lc = geo.levi_civita();
						Q[fx_i][d][i] *= er;
						Q[fy_i][d][i] *= er;
						Q[fz_i][d][i] *= er;
						CHECK_FLUX(Q[er_i][d][i], Q[fx_i][d][i], Q[fy_i][d][i], Q[fz_i][d][i]);
					}
				}
			}
		}
	}
}

template<int NDIM>
template<int INX>
void radiation_physics<NDIM>::analytic_solution(test_type test, hydro::state_type &U, const hydro::x_type &X,
		safe_real time) {
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

	const safe_real dx = 1.0 / INX;

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

#define CHECK_FLUX( er, fx, fy, fz) if( ((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c) > 1 ) {printf( "flux exceded %s %i %e fx %e fy %e fz %e er %e\n", __FILE__, __LINE__, sqrt(((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c)), fx, fy, fz, er*physcon().c); abort();}

#endif /* OCTOTIGER_UNITIGER_radiation_physics_IMPL_HPP_ */
