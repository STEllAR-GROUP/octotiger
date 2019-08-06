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

	static void to_prim(std::vector<safe_real> u, safe_real &p, safe_real &v, int dim, safe_real dx) {
		const auto rho = u[rho_i];
		const auto rhoinv = safe_real(1.) / rho;
		safe_real ek = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			ek += pow(u[sx_i + dim], 2) * rhoinv * safe_real(0.5);
		}
		if constexpr (NDIM > 1) {
			for (int n = 0; n < pow<int, int>(3, NDIM - 2); n++) {
				ek += pow(u[zx_i + n], 2) * safe_real(0.5) * rhoinv / (dx * dx);
			}
		}
		auto ein = max(u[egas_i] - ek, 0.0);
		if (ein < safe_real(0.001) * u[egas_i]) {
			ein = pow(std::max(u[tau_i], safe_real(0.0)), FGAMMA);
		}
		v = u[sx_i + dim] * rhoinv;
		p = (FGAMMA - 1.0) * ein;
	}

	static void flux(const std::vector<safe_real> &UL, const std::vector<safe_real> &UR, std::vector<safe_real> &F, int dim, safe_real &a,
			std::array<safe_real, NDIM> &vg, safe_real dx) {

		safe_real pr, vr, pl, vl, vr0, vl0;

		to_prim(UR, pr, vr0, dim, dx);
		to_prim(UL, pl, vl0, dim, dx);
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
	void post_process(hydro::state_type &U, safe_real dx) {
		static const cell_geometry<NDIM, INX> geo;
		constexpr auto dir = geo.directions[NDIM - 1];
		const static auto is = geo.find_indices(geo.H_BW, geo.H_NX - geo.H_BW);
		for (auto i : is) {
			safe_real ek = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				ek += U[sx_i + dim][i] * U[sx_i + dim][i];
			}
			for (int n = 0; n < geo.NANGMOM; n++) {
				ek += pow(U[zx_i + n][i], 2) / (dx * dx);
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

	template<int INX>
	static void source(hydro::state_type &dudt, const hydro::state_type &U, const hydro::flux_type &F, const hydro::x_type<NDIM> X, safe_real omega,
			safe_real dx) {
		static constexpr cell_geometry<NDIM, INX> geo;
		for (int dim = 0; dim < NDIM; dim++) {
			static constexpr auto kdelta = geo.kdeltas[NDIM - 1];
			for (int n = 0; n < geo.NANGMOM; n++) {
				const auto m = dim;
				for (int l = 0; l < NDIM; l++) {
					for (const auto &i : geo.find_indices(geo.H_BW, geo.H_NX - geo.H_BW)) {
						const auto fr = F[dim][sx_i + l][i + geo.H_DN[dim]];
						const auto fl = F[dim][sx_i + l][i];
						dudt[zx_i + n][i] -= kdelta[n][m][l] * 0.5 * (fr + fl);
					}
				}
			}
		}
		for (const auto &i : geo.find_indices(geo.H_BW, geo.H_NX - geo.H_BW)) {
			if constexpr (NDIM == 2) {
				dudt[zx_i][i] += omega * (X[i][0] * U[sx_i][i] + X[i][1] * U[sy_i][i]);
			} else if constexpr (NDIM == 3) {
				dudt[zx_i][i] -= omega * X[i][2] * U[sx_i][i];
				dudt[zy_i][i] -= omega * X[i][2] * U[sy_i][i];
				dudt[zz_i][i] += omega * (X[i][0] * U[sx_i][i] + X[i][1] * U[sy_i][i]);
			}

		}
		for (const auto &i : geo.find_indices(geo.H_BW, geo.H_NX - geo.H_BW)) {
			dudt[sx_i][i] += U[sy_i][i] * omega;
			dudt[sy_i][i] -= U[sx_i][i] * omega;
		}

	}

};

#endif /* OCTOTIGER_UNITIGER_PHYSICS_HPP_ */
