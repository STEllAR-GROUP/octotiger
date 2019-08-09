/*
 * physics.hpp
 *
 *  Created on: Aug 5, 2019
 *      Author: dmarce1
 */

#include "./safe_real.hpp"
#include "../test_problems/blast.hpp"

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

	enum test_type {
		SOD, BLAST, KH
	};

	static int field_count() {
		return nf_;
	}

	static void set_fgamma( safe_real fg ) {
		fgamma_ = fg;
	}

	static void to_prim(std::vector<safe_real> u, safe_real &p, safe_real &v, int dim, safe_real dx) {
		const auto rho = u[rho_i];
		const auto rhoinv = safe_real(1.) / rho;
		safe_real ek = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			ek += pow(u[sx_i + dim], 2) * rhoinv * safe_real(0.5);
		}
		if constexpr (NDIM > 1) {
			for (int n = 0; n < std::pow<int, int>(3, NDIM - 2); n++) {
				ek += pow(u[zx_i + n], 2) * safe_real(0.5) * rhoinv / (dx * dx);
			}
		}
		auto ein = std::max(u[egas_i] - ek, 0.0);
		if (ein < safe_real(0.001) * u[egas_i]) {
			ein = pow(u[tau_i], fgamma_);
		}
		v = u[sx_i + dim] * rhoinv;
		p = (fgamma_ - 1.0) * ein;
	}

	static void flux(const std::vector<safe_real> &UL, const std::vector<safe_real> &UR, const std::vector<safe_real> &UL0, const std::vector<safe_real> &UR0,
			std::vector<safe_real> &F, int dim, safe_real &am, safe_real &ap, std::array<safe_real, NDIM> &vg, safe_real dx) {

		safe_real pr, vr, pl, vl, vr0, vl0;

		static thread_local std::vector<safe_real> FR(nf_), FL(nf_);
		to_prim(UR, pr, vr0, dim, dx);
		to_prim(UL, pl, vl0, dim, dx);
		vr = vr0 - vg[dim];
		vl = vl0 - vg[dim];
		if (ap < 0.0) {
			safe_real cr, cl;
			cr = sqrt(fgamma_ * pr / (UR[rho_i]));
			cl = sqrt(fgamma_ * pl / (UL[rho_i]));
			ap = std::max(vr + cr, vl + cl);
			am = std::min(vr - cr, vl - cl);
			ap = std::max(ap, safe_real(0.0));
			am = std::min(am, safe_real(0.0));
		}
		for (int f = 0; f < nf_; f++) {
			FR[f] = vr * UR[f];
			FL[f] = vl * UL[f];
		}
		FR[sx_i + dim] += pr;
		FL[sx_i + dim] += pl;
		FR[egas_i] += (UR[pot_i] + pr) * vr0;
		FL[egas_i] += (UL[pot_i] + pl) * vl0;
		for (int f = 0; f < nf_; f++) {
			F[f] = (ap * FL[f] - am * FR[f] + ap * am * (UR[f] - UL[f])) / (ap - am);
		}
		constexpr static int npos = 3;
		constexpr static std::array<int, npos> pos_fields = { rho_i, tau_i, egas_i };

		constexpr safe_real max_change = 1.0e-6;
		safe_real theta = 1.0;
		for (int fi = 0; fi < npos; fi++) {
			const int f = pos_fields[fi];
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
			for (int f = 0; f < nf_; f++) {
				F[f] *= theta;
			}
			const auto c0 = 1.0 - theta;
			to_prim(UR0, pr, vr0, dim, dx);
			to_prim(UL0, pl, vl0, dim, dx);
			vr = vr0 - vg[dim];
			vl = vl0 - vg[dim];
			for (int f = 0; f < nf_; f++) {
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
				U[tau_i][i] = POWER(ein, 1.0 / fgamma_);
			}
		}
	}

	template<int INX>
	static void source(hydro::state_type &dudt, const hydro::state_type &U, const hydro::flux_type &F, const hydro::x_type X, safe_real omega,
			safe_real dx) {
		static const cell_geometry<NDIM, INX> geo;
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
	static const hydro::state_type pre_recon(const hydro::state_type &U, const hydro::x_type X, safe_real omega, bool angmom) {
		static const cell_geometry<NDIM, INX> geo;
		static const auto indices = geo.find_indices(0, geo.H_NX);
		auto V = U;
		const auto dx = X[0][geo.H_DNX] - X[0][0];
		for (const auto &i : indices) {
			const auto rho = V[rho_i][i];
			const auto rhoinv = 1.0 / rho;
			for (int dim = 0; dim < NDIM; dim++) {
				auto &s = V[sx_i + dim][i];
				V[egas_i][i] -= 0.5 * s * s * rhoinv;
				s *= rhoinv;
			}
			if (angmom) {
				for (int n = 0; n < geo.NANGMOM; n++) {
					auto &z = V[zx_i + n][i];
					V[egas_i][i] -= 0.5 * z * z * rhoinv / (dx * dx);
					z *= rhoinv;
				}
			}
			for (int si = 0; si < n_species_; si++) {
				V[spc_i + si][i] *= rhoinv;
			}
			V[pot_i][i] *= rhoinv;
		}
		return V;
	}

	template<int INX>
	static hydro::recon_type<NDIM> post_recon(const hydro::recon_type<NDIM> &P, const hydro::x_type X, safe_real omega, bool angmom) {
		static const cell_geometry<NDIM, INX> geo;
		static const auto indices = geo.find_indices(2, geo.H_NX - 2);
		auto Q = P;
		const auto dx = X[0][geo.H_DNX] - X[0][0];
		for (const auto &i : indices) {
			for (int d = 0; d < geo.NDIR; d++) {
				if (d != geo.NDIR / 2) {
					const auto rho = Q[rho_i][i][d];
					for (int dim = 0; dim < NDIM; dim++) {
						auto &v = Q[sx_i + dim][i][d];
						Q[egas_i][i][d] += 0.5 * v * v * rho;
						v *= rho;
					}
					if (angmom) {
						for (int n = 0; n < geo.NANGMOM; n++) {
							auto &z = Q[zx_i + n][i][d];
							Q[egas_i][i][d] += 0.5 * z * z * rho / (dx * dx);
							z *= rho;
						}
					}
					Q[pot_i][i][d] *= rho;
					safe_real w = 0.0;
					for (int si = 0; si < n_species_; si++) {
						w += Q[spc_i + si][i][d];
						Q[spc_i + si][i][d] *= rho;
					}
					w = 1.0 / w;
					for (int si = 0; si < n_species_; si++) {
						Q[spc_i + si][i][d] *= w;
					}
				}
			}
		}
		return Q;
	}

	template<int INX>
	using comp_type = hydro_computer<NDIM, INX>;

	template<int INX>
	std::vector<typename comp_type<INX>::bc_type> initialize(test_type t, hydro::state_type &U, hydro::x_type &X);

	template<int INX>
	static void analytic_solution(test_type test, hydro::state_type &U, const hydro::x_type &X, safe_real time) {
		static const cell_geometry<NDIM, INX> geo;
		static safe_real rmax = 0.0;
		static std::once_flag one;

		std::call_once(one, [&X]() {
			const auto dx = X[0][geo.H_DNX] - X[0][0];
			for (int i = 0; i < geo.H_N3; i++) {
				safe_real r = 0.0;
				for (int dim = 0; dim < NDIM; dim++) {
					r += X[dim][i] * X[dim][i];
				}
				r = sqrt(r);
				rmax = std::max(r, rmax);
			}
			rmax *= 2.0;
		});

		for (int f = 0; f < nf_; f++) {
			for (auto &u : U[f]) {
				u = 0.0;
			}
		}

		for (int i = 0; i < geo.H_N3; i++) {
			safe_real r = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				r += X[dim][i] * X[dim][i];
			}
			r = sqrt(r);
			double den, vel, pre;
			sedov::solution(time + 7e-4, r, rmax, den, vel, pre, NDIM);
			U[rho_i][i] = den;
			U[tau_i][i] = pow(pre / (fgamma_ - 1.0), 1.0 / fgamma_);
			U[egas_i][i] = pre / (fgamma_ - 1.0) + 0.5 * den * vel * vel;
			for (int dim = 0; dim < NDIM; dim++) {
				U[sx_i + dim][i] = den * vel * X[dim][i] / r;
			}
			U[spc_i][i] = den;
		}
	}

private:
	static int nf_;
	static int n_species_;
	static safe_real fgamma_;

};

template<int NDIM>
template<int INX>
std::vector<typename hydro_computer<NDIM, INX>::bc_type> physics<NDIM>::initialize(physics<NDIM>::test_type t, hydro::state_type &U, hydro::x_type &X) {
	static const cell_geometry<NDIM, INX> geo;

	std::vector<typename hydro_computer<NDIM, INX>::bc_type> bc(2 * NDIM);

	for (int i = 0; i < 2 * NDIM; i++) {
		bc[i] = hydro_computer<NDIM, INX>::OUTFLOW;
	}

	switch (t) {
	case SOD:
		break;
	case BLAST:
		break;
	case KH:
		for (int i = 0; i < 2 * NDIM; i++) {
			bc[i] = hydro_computer<NDIM, INX>::PERIODIC;
		}
		break;
	}
	U.resize(nf_);
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
			X[j][i] = (((k % geo.H_NX) - geo.H_BW) + 0.5) * dx - 0.5;
			k /= geo.H_NX;
			j++;
		}
	}

	for (int i = 0; i < geo.H_N3; i++) {
		double rho = 0, vx = 0, vy = 0, vz = 0, p = 0, r;
		safe_real x2, xsum;
		switch (t) {
		case SOD:
			xsum = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				xsum += X[dim][i];
			}
			if (xsum < 0) {
				rho = 1.0;
				p = 1.0;
			} else {
				rho = 0.125;
				p = 0.1;
			}
			break;
		case BLAST:

			x2 = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				x2 += X[dim][i] * X[dim][i];
			}
			r = sqrt(x2);
			double v;
			sedov::solution(7e-4, r, std::sqrt(3) + dx, rho, v, p, NDIM);
			p = std::max((fgamma_ - 1.0) * 1.0e-20, p);
			vx = v * X[0][i] / r;
			if constexpr (NDIM >= 2) {
				vy = v * X[1][i] / r;
			}
			if constexpr (NDIM == 3) {
				vz = v * X[2][i] / r;
			}
			break;
		case KH:
			const auto eps = []() {
				return (rand() + 0.5) / RAND_MAX * 1.0e-3;
			};

			U[physics<NDIM>::tau_i][i] = 1.0;
			p = 1.0;
			if (X[1][i] < 0.0) {
				rho = 1.0 + eps();
				vx = -0.5;
			} else {
				rho = 2.0 + eps();
				vx = +0.5;
			}
			break;
		}
		U[rho_i][i] = rho;
		U[spc_i][i] = rho;
		U[sx_i][i] = rho * vx;
		U[egas_i][i] = p / (fgamma_ - 1.0) + 0.5 * rho * vx * vx;
		U[tau_i][i] = std::pow(p / (fgamma_ - 1.0), 1.0 / fgamma_);
		if constexpr (NDIM >= 2) {
			U[sy_i][i] = rho * vy;
			U[egas_i][i] += 0.5 * rho * vy * vy;
		}
		if constexpr (NDIM >= 3) {
			U[sz_i][i] = rho * vz;
			U[egas_i][i] += 0.5 * rho * vz * vz;
		}

	}

	return bc;
}

template<int NDIM>
int physics<NDIM>::nf_ = (4 + NDIM + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2))) + 2;

template<int NDIM>
int physics<NDIM>::n_species_ = 2;

template<int NDIM>
safe_real physics<NDIM>::fgamma_ = 7./4.;




#endif /* OCTOTIGER_UNITIGER_PHYSICS_HPP_ */
