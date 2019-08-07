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
	static constexpr int spc_i = 4 + NDIM + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2));
	static int nf;
	static int n_species;

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
			ein = pow(u[tau_i], FGAMMA);
		}
		v = u[sx_i + dim] * rhoinv;
		p = (FGAMMA - 1.0) * ein;
	}

	static void flux(const std::vector<safe_real> &UL, const std::vector<safe_real> &UR, const std::vector<safe_real> &UL0, const std::vector<safe_real> &UR0,
			std::vector<safe_real> &F, int dim, safe_real &am, safe_real &ap, std::array<safe_real, NDIM> &vg, safe_real dx) {

		safe_real pr, vr, pl, vl, vr0, vl0;

		static thread_local std::vector<safe_real> FR(nf), FL(nf);
		to_prim(UR, pr, vr0, dim, dx);
		to_prim(UL, pl, vl0, dim, dx);
		vr = vr0 - vg[dim];
		vl = vl0 - vg[dim];
		if (ap < 0.0) {
			safe_real cr, cl;
			cr = sqrt(FGAMMA * pr / (UR[rho_i]));
			cl = sqrt(FGAMMA * pl / (UL[rho_i]));
			ap = std::max(vr + cr, vl + cl);
			am = std::min(vr - cr, vl - cl);
			ap = std::max(ap, safe_real(0.0));
			am = std::min(am, safe_real(0.0));
		}
		for (int f = 0; f < nf; f++) {
			FR[f] = vr * UR[f];
			FL[f] = vl * UL[f];
		}
		FR[sx_i + dim] += pr;
		FL[sx_i + dim] += pl;
		FR[egas_i] += (UR[pot_i] + pr) * vr0;
		FL[egas_i] += (UL[pot_i] + pl) * vl0;
		for (int f = 0; f < nf; f++) {
			F[f] = (ap * FL[f] - am * FR[f] + ap * am * (UR[f] - UL[f])) / (ap - am);
		}
		constexpr static int npos = 3;
		constexpr static std::array<int, npos> pos_fields = { rho_i, tau_i, egas_i };

		constexpr safe_real max_change = 1.0e-6;
		safe_real theta = 1.0;
		for (int f = 0; f < npos; f++) {
			safe_real thetaR = 1.0, thetaL = 1.0;
			const auto umin = max_change * std::max(UR0[f], UL0[f]);
			if (UR[f] < umin && UR0[f] != UR[f]) {
				thetaR = (UR0[f] - umin) / (UR0[f] - UR[f]);
			}
			if (UL[f] < umin && UL0[f] != UL[f]) {
				thetaL = (UL0[f] - umin) / (UL0[f] - UL[f]);
			}
			theta = std::min(theta, std::max(std::min(std::min(thetaL, thetaR), safe_real(1.0)), safe_real(0.0)));
		}
		if (theta < 1.0) {
			for (int f = 0; f < nf; f++) {
				F[f] *= theta;
			}
			const auto c0 = 1.0 - theta;
			to_prim(UR0, pr, vr0, dim, dx);
			to_prim(UL0, pl, vl0, dim, dx);
			vr = vr0 - vg[dim];
			vl = vl0 - vg[dim];
			for (int f = 0; f < nf; f++) {
				F[f] += c0 * safe_real(0.5) * ((vr - ap) * UR0[f] + (vl + ap) * UL0[f]);
			}
			F[sx_i + dim] += c0 * safe_real(0.5) * (pr + pl);
			F[egas_i] += c0 * safe_real(0.5) * (pr * vr0 + pl * vl0);
			F[egas_i] += c0 * safe_real(0.5) * (UR[pot_i] * vr0 + UL[pot_i] * vl0);
		}

	}

	template<int INX>
	static void post_process(hydro::state_type &U, safe_real dx) {
		static const cell_geometry<NDIM, INX> geo;
		constexpr auto dir = geo.direction();
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
			static constexpr auto kdelta = geo.kronecker_delta();
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
			if constexpr (NDIM == 3) {
				dudt[zx_i][i] -= omega * X[2][i] * U[sx_i][i];
				dudt[zy_i][i] -= omega * X[2][i] * U[sy_i][i];
			}
			if constexpr (NDIM >= 2) {
				dudt[zx_i][i] += omega * (X[0][i] * U[sx_i][i] + X[1][i] * U[sy_i][i]);
			}
		}
		for (const auto &i : geo.find_indices(geo.H_BW, geo.H_NX - geo.H_BW)) {
			dudt[sx_i][i] += U[sy_i][i] * omega;
			dudt[sy_i][i] -= U[sx_i][i] * omega;
		}

	}

	template<int INX>
	static const hydro::state_type pre_recon(const hydro::state_type &U, safe_real dx) {
		static constexpr cell_geometry<NDIM, INX> geo;
		static const auto indices = geo.find_indices(0, geo.H_NX);
		auto V = U;
		for (const auto &i : indices) {
			const auto rho = V[rho_i][i];
			const auto rhoinv = 1.0 / rho;
			for (int dim = 0; dim < NDIM; dim++) {
				auto &s = V[sx_i + dim][i];
				V[egas_i][i] -= 0.5 * s * s * rhoinv;
				s *= rhoinv;
			}
			for (int n = 0; n < geo.NANGMOM; n++) {
				auto &z = V[zx_i + n][i];
				V[egas_i][i] -= 0.5 * z * z * rhoinv / (dx * dx);
				z *= rhoinv;
			}
			for (int si = 0; si < n_species; si++) {
				V[spc_i + si][i] *= rhoinv;
			}
			V[pot_i][i] *= rhoinv;
		}
		return V;
	}

	template<int INX>
	static hydro::recon_type<NDIM> post_recon(const hydro::recon_type<NDIM> &P, safe_real dx) {
		static constexpr cell_geometry<NDIM, INX> geo;
		static const auto indices = geo.find_indices(2, geo.H_NX - 2);
		auto Q = P;
		for (const auto &i : indices) {
			for (int d = 0; d < geo.NDIR; d++) {
				if (d != geo.NDIR / 2) {
					const auto rho = Q[rho_i][i][d];
					for (int dim = 0; dim < NDIM; dim++) {
						auto &v = Q[sx_i + dim][i][d];
						Q[egas_i][i][d] += 0.5 * v * v * rho;
						v *= rho;
					}
					for (int n = 0; n < geo.NANGMOM; n++) {
						auto &z = Q[zx_i + n][i][d];
						Q[egas_i][i][d] += 0.5 * z * z * rho / (dx * dx);
						z *= rho;
					}
					Q[pot_i][i][d] *= rho;
					for (int si = 0; si < n_species; si++) {
						Q[spc_i + si][i][d] *= rho;
					}
				}
			}
		}
		return Q;
	}

};

template<int NDIM>
int physics<NDIM>::nf = (4 + NDIM + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2))) + 2;

template<int NDIM>
int physics<NDIM>::n_species = 2;

#endif /* OCTOTIGER_UNITIGER_PHYSICS_HPP_ */
