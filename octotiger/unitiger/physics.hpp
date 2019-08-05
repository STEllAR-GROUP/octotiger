/*
 * physics.hpp
 *
 *  Created on: Aug 5, 2019
 *      Author: dmarce1
 */

#include "./safe_real.hpp"

#ifndef OCTOTIGER_UNITIGER_PHYSICS_HPP_
#define OCTOTIGER_UNITIGER_PHYSICS_HPP_

template<int NDIM>
struct physics {

	static constexpr int rho_i = 0;
	static constexpr int egas_i = 1;
	static constexpr int tau_i = 2;
	static constexpr int pot_i = 3;
	static constexpr int sx_i = 4;
	static constexpr int sy_i = 5;
	static constexpr int sz_i = 6;
	static constexpr int zx_i = 4 + NDIM;
	static constexpr int zy_i = 5 + NDIM;
	static constexpr int zz_i = 6 + NDIM;
	static constexpr int nf = 4 + NDIM + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2));

	static int field_count() {
		return nf;
	}

	static void to_prim(std::vector<safe_real> u, safe_real &p, safe_real &v, int dim) {
		const auto rho = u[rho_i];
		const auto rhoinv = safe_real(1.) / rho;
		safe_real ek = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			ek += pow(u[sx_i + dim], 2) * rhoinv * safe_real(0.5);
		}
		auto ein = max(u[egas_i] - ek, 0.0);
		if (ein < safe_real(0.001) * u[egas_i]) {
			ein = pow(std::max(u[tau_i], safe_real(0.0)), FGAMMA);
		}
		v = u[sx_i + dim] * rhoinv;
		p = (FGAMMA - 1.0) * ein;
	}

	static void flux(const std::vector<safe_real> &UL, const std::vector<safe_real> &UR, std::vector<safe_real> &F, int dim, safe_real &a,
			std::array<safe_real, NDIM> &vg) {

		safe_real pr, vr, pl, vl, vr0, vl0;

		to_prim(UR, pr, vr0, dim);
		to_prim(UL, pl, vl0, dim);
		vr = vr0 - vg[dim];
		vl = vl0 - vg[dim];
		if (a < 0.0) {
			safe_real ar, al;
			ar = abs(vr) + sqrt(FGAMMA * pr / (UR[rho_i]));
			al = abs(vl) + sqrt(FGAMMA * pl / (UL[rho_i]));
			a = std::max(al, ar);
		}
		for (int f = 0; f < nf; f++) {
			F[f] = safe_real(0.5) * ((vr - a) * UR[f] + (vl + a) * UL[f]);
		}
		F[sx_i + dim] += safe_real(0.5) * (pr + pl);
		F[egas_i] += safe_real(0.5) * (pr * vr0 + pl * vl0);
	}

	template<int INX>
	static void post_process(std::vector<std::vector<safe_real>> &U, const cell_geometry<NDIM,INX> &geo) {
		constexpr auto dir = geo.directions[NDIM - 1];
		for (const auto &i : find_indices<NDIM, geo.H_NX>(geo.H_BW, geo.H_NX - geo.H_BW)) {
			safe_real ek = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				ek += U[sx_i + dim][i] * U[sx_i + dim][i];
			}
			ek *= 0.5 * INVERSE(U[rho_i][i]);
			auto egas_max = U[egas_i][i];
			for (int d = 0; d < geo.NDIR; d++) {
				egas_max = std::max(egas_max, U[egas_i][i + dir[d]]);
			}
			safe_real ein = U[egas_i][i] - ek;
			if (ein > 0.1 * egas_max) {
				U[tau_i][i] = POWER(ein, 1.0 / FGAMMA);
			}
		}

	}

};

#endif /* OCTOTIGER_UNITIGER_PHYSICS_HPP_ */
