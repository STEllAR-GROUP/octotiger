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
#ifndef OCTOTIGER_UNITIGER_PHYSICS_HPP12443_
#define OCTOTIGER_UNITIGER_PHYSICS_HPP12443_

#include "octotiger/unitiger/safe_real.hpp"
#include "octotiger/test_problems/blast.hpp"
#include "octotiger/test_problems/exact_sod.hpp"
#include "octotiger/profiler.hpp"

template<int NDIM>
int physics<NDIM>::field_count() {
	return nf_;
}

template<int NDIM>
safe_real physics<NDIM>::deg_pres(safe_real x) {
	safe_real p;
	if (x < 0.001) {
		p = 1.6 * A_ * std::pow(x, 5);
	} else {
		p = A_ * (x * (2 * x * x - 3) * std::sqrt(x * x + 1) + 3 * asinh(x));
	}
	return p;
}

template<int NDIM>
void physics<NDIM>::to_prim(std::vector<safe_real> u, safe_real &p, safe_real &v, safe_real &cs, int dim) {
	const auto rho = u[rho_i];
	const auto rhoinv = INVERSE(rho);
	double hdeg = 0.0, pdeg = 0.0, edeg = 0.0, dpdeg_drho = 0.0;
	if (A_ != 0.0) {
		const auto x = std::pow(rho / B_, 1.0 / 3.0);
		hdeg = 8.0 * A_ / B_ * (std::sqrt(x * x + 1.0) - 1.0);
		pdeg = deg_pres(x);
		if (x > 0.001) {
			edeg = rho * hdeg - pdeg;
		} else {
			edeg = 2.4 * A_ * std::pow(x, 5);
		}
		dpdeg_drho = 8.0 / 3.0 * A_ / B_ * x * x / std::sqrt(x * x + 1.0);
	}
	safe_real ek = 0.0;
	for (int dim = 0; dim < NDIM; dim++) {
		ek += pow(u[sx_i + dim], 2) * rhoinv * safe_real(0.5);
	}
	auto ein = u[egas_i] - ek - edeg;
	if (ein <= de_switch_1 * u[egas_i]) {
		ein = POWER(u[tau_i], fgamma_);
	}
	const double dp_drho = dpdeg_drho + (fgamma_ - 1.0) * ein * rhoinv;
	const double dp_deps = (fgamma_ - 1.0) * rho;
	v = u[sx_i + dim] * rhoinv;
	p = (fgamma_ - 1.0) * ein + pdeg;
	const auto z = p * rhoinv * rhoinv * dp_deps + dp_drho;
	if( z < 0.0 ) {
		print( "%e %e %e %e %e %e %e %e %e\n", p, rhoinv, dpdeg_drho, dp_deps, ein, dp_drho, u[tau_i], ek, edeg);
	}
	cs = SQRT(z);
}

template<int NDIM>
void physics<NDIM>::to_prim_experimental(const std::vector<double> &u, double &p, double &v, double &cs, const int dim) noexcept {
	const auto rho = u[rho_i];
	const auto rhoinv = (1.) / rho;
	double hdeg = 0.0, pdeg = 0.0, edeg = 0.0, dpdeg_drho = 0.0;

	// all workitems choose the same path
	if (A_ != 0.0) {
		const auto Binv = 1.0 / B_;
		const auto x = std::pow(rho * Binv, 1.0 / 3.0);
		const auto x_sqr = x * x;
		const auto x_sqr_sqrt = std::sqrt(x_sqr + 1.0);
		const auto x_pow_5 = x_sqr * x_sqr * x;
		hdeg = 8.0 * A_ * Binv * (x_sqr_sqrt - 1.0);
        if (x < 0.001) {
            pdeg = 1.6 * A_ * x_pow_5;
        } else {
            pdeg = A_ * (x * (2 * x_sqr - 3) * x_sqr_sqrt + 3 * asinh(x));
        }
        if (x > 0.001) {
			edeg = rho * hdeg - pdeg;
		} else {
			edeg = 2.4 * A_ * x_pow_5 ;
		}
		dpdeg_drho = 8.0 / 3.0 * A_ * Binv * x_sqr / x_sqr_sqrt;
	}
	double ek = 0.0;
	for (int dim = 0; dim < NDIM; dim++) {
		ek += u[sx_i + dim] * u[sx_i + dim] * rhoinv * 0.5;
	}
	auto ein = u[egas_i] - ek - edeg;
	if (ein < de_switch_1 * u[egas_i]) {
		ein = pow(u[tau_i], fgamma_);
	}
	const double dp_drho = dpdeg_drho + (fgamma_ - 1.0) * ein * rhoinv;
	const double dp_deps = (fgamma_ - 1.0) * rho;
	v = u[sx_i + dim] * rhoinv;
	p = (fgamma_ - 1.0) * ein + pdeg;
	cs = std::sqrt(p * rhoinv * rhoinv * dp_deps + dp_drho);
}

template<int NDIM>
template<int INX>
void physics<NDIM>::physical_flux(const std::vector<safe_real> &U, std::vector<safe_real> &F, int dim, safe_real &am, safe_real &ap,
		std::array<safe_real, NDIM> &x, std::array<safe_real, NDIM> &vg) {
	static const cell_geometry<NDIM, INX> geo;
	static constexpr auto levi_civita = geo.levi_civita();
	safe_real p, v, v0, c;
	to_prim(U, p, v0, c, dim);
	v = v0 - vg[dim];
	am = v - c;
	ap = v + c;
#pragma ivdep
	for (int f = 0; f < nf_; f++) {
		F[f] = v * U[f];
	}
	F[sx_i + dim] += p;
	F[egas_i] += v0 * p;
	for (int n = 0; n < geo.NANGMOM; n++) {
#pragma ivdep
		for (int m = 0; m < NDIM; m++) {
			F[lx_i + n] += levi_civita[n][m][dim] * x[m] * p;
		}
	}
}
template<int NDIM>
template<int INX>
void physics<NDIM>::physical_flux_experimental(const std::vector<safe_real> &U, std::vector<safe_real> &F, int dim, safe_real &am, safe_real &ap,
		std::array<safe_real, NDIM> &x, std::array<safe_real, NDIM> &vg) {
	static const cell_geometry<NDIM, INX> geo;
	static constexpr auto levi_civita = geo.levi_civita();
	safe_real p, v, v0, c;
	to_prim_experimental(U, p, v0, c, dim);
	// to_prim(U, p, v0, c, dim);
	v = v0 - vg[dim];
	am = v - c;
	ap = v + c;
#pragma ivdep
	for (int f = 0; f < nf_; f++) {
		F[f] = v * U[f];
	}
	F[sx_i + dim] += p;
	F[egas_i] += v0 * p;
	for (int n = 0; n < geo.NANGMOM; n++) {
#pragma ivdep
		for (int m = 0; m < NDIM; m++) {
			F[lx_i + n] += levi_civita[n][m][dim] * x[m] * p;
		}
	}
}

template<int NDIM>
template<int INX>
void physics<NDIM>::post_process(hydro::state_type &U, const hydro::x_type &X, safe_real dx) {
	static const cell_geometry<NDIM, INX> geo;
	constexpr
	auto dir = geo.direction();
	const static auto is = geo.find_indices(geo.H_BW, geo.H_NX - geo.H_BW);
	for (auto i : is) {
		double hdeg = 0.0, pdeg = 0.0, edeg = 0.0;
		if (A_ != 0.0) {
			const auto x = std::pow(U[rho_i][i] / B_, 1.0 / 3.0);
			hdeg = 8.0 * A_ / B_ * (std::sqrt(x * x + 1.0) - 1.0);
			pdeg = deg_pres(x);
			edeg = U[rho_i][i] * hdeg - pdeg;
		}

		safe_real ek = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			ek += U[sx_i + dim][i] * U[sx_i + dim][i];
		}
		ek *= 0.5 * INVERSE(U[rho_i][i]);
		auto egas_max = U[egas_i][i];
		for (int d = 0; d < geo.NDIR; d++) {
			egas_max = std::max(egas_max, U[egas_i][i + dir[d]]);
		}
		safe_real ein = U[egas_i][i] - ek - edeg;
		if (ein > de_switch_2 * egas_max) {
			U[tau_i][i] = POWER(ein, 1.0 / fgamma_);
		}
		if (rho_sink_radius_ > 0.0) {
			double r = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				r += X[dim][i] * X[dim][i];
			}
			r = std::sqrt(r);
			if (r < rho_sink_radius_) {
				for (int s = 0; s < spc_i; s++) {
					U[spc_i + s][i] = rho_sink_floor_ / n_species_;
				}
				U[rho_i][i] = rho_sink_floor_;
				U[tau_i][i] = std::pow(rho_sink_floor_, 1.0 / fgamma_);
				U[egas_i][i] = rho_sink_floor_;
				for (int dim = 0; dim < NDIM; dim++) {
					U[sx_i + dim][i] = 0.0;
				}

			}
		}
	}
}

template<int NDIM>
template<int INX>
void physics<NDIM>::source(hydro::state_type &dudt, const hydro::state_type &U, const hydro::flux_type &F, const hydro::x_type X, safe_real omega,
		safe_real dx) {
	static const cell_geometry<NDIM, INX> geo;
	static constexpr auto levi_civita = geo.levi_civita();
	for (const auto &i : geo.find_indices(geo.H_BW, geo.H_NX - geo.H_BW)) {
		if HOST_CONSTEXPR (NDIM == 3) {
			dudt[lx_i][i] += U[ly_i][i] * omega;
			dudt[ly_i][i] -= U[lx_i][i] * omega;
		}
		if HOST_CONSTEXPR (NDIM >= 2) {
			dudt[sx_i][i] += U[sy_i][i] * omega;
			dudt[sy_i][i] -= U[sx_i][i] * omega;
		}
		safe_real r = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			r += X[dim][i] * X[dim][i];
		}
		r = std::sqrt(r);
		for (int dim = 0; dim < NDIM; dim++) {
			const auto f = sx_i + dim;
			const auto x = X[dim][i];
			double a;
			double r0 = 0.00;
			const auto c0 = std::max(U[rho_i][i] - 1.0e-5, 0.0) / U[rho_i][i];
			if (r > r0) {
				const auto r3inv = 1.0 / (r * r * r) * c0;
				a = x * r3inv * GM_;
			} else {
				const auto r30inv = 1.0 / (r0 * r0 * r0) * c0;
				a = x * r30inv * GM_;
			}
			dudt[f][i] -= a;
			dudt[egas_i][i] -= U[f][i] * a;
		}
	}

}

/*** Reconstruct uses this - GPUize****/

template<int NDIM>
template<int INX>
const hydro::state_type& physics<NDIM>::pre_recon(const hydro::state_type &U, const hydro::x_type X, safe_real omega, bool angmom) {
	PROFILE();
	static const cell_geometry<NDIM, INX> geo;
	static const auto indices = geo.find_indices(0, geo.H_NX);
	static thread_local hydro::state_type V;
	V = U;
	for (int j = 0; j < geo.H_NX_X; j++) {
		for (int k = 0; k < geo.H_NX_Y; k++) {
#pragma ivdep
			for (int l = 0; l < geo.H_NX_Z; l++) {
				const int i = geo.to_index(j, k, l);
				const auto rho = V[rho_i][i];
				const auto rhoinv = 1.0 / rho;
				for (int dim = 0; dim < NDIM; dim++) {
					auto &s = V[sx_i + dim][i];
					V[egas_i][i] -= 0.5 * s * s * rhoinv;
					s *= rhoinv;
				}
				for (int si = 0; si < n_species_; si++) {
					V[spc_i + si][i] *= rhoinv;
				}
				V[pot_i][i] *= rhoinv;
			}
		}
	}
	for (int n = 0; n < geo.NANGMOM; n++) {
		for (int j = 0; j < geo.H_NX_X; j++) {
			for (int k = 0; k < geo.H_NX_Y; k++) {
#pragma ivdep
				for (int l = 0; l < geo.H_NX_Z; l++) {
					const int i = geo.to_index(j, k, l);
					const auto rho = V[rho_i][i];
					const auto rhoinv = 1.0 / rho;
					V[lx_i + n][i] *= rhoinv;
				}
			}
		}
		static constexpr auto levi_civita = geo.levi_civita();
		for (int m = 0; m < NDIM; m++) {
			for (int q = 0; q < NDIM; q++) {
				const auto lc = levi_civita[n][m][q];
				if (lc != 0) {
					for (int j = 0; j < geo.H_NX_X; j++) {
						for (int k = 0; k < geo.H_NX_Y; k++) {
#pragma ivdep
							for (int l = 0; l < geo.H_NX_Z; l++) {
								const int i = geo.to_index(j, k, l);
								V[lx_i + n][i] -= lc * X[m][i] * V[sx_i + q][i];
							}
						}
					}
				}
			}
		}
	}
	if (NDIM >= 2) {
		for (int j = 0; j < geo.H_NX_X; j++) {
			for (int k = 0; k < geo.H_NX_Y; k++) {
#pragma ivdep
				for (int l = 0; l < geo.H_NX_Z; l++) {
					const int i = geo.to_index(j, k, l);
					V[sx_i][i] += omega * X[1][i];
					V[sy_i][i] -= omega * X[0][i];
				}
			}
		}
	}
	return V;
}

template<int NDIM>
void physics<NDIM>::set_degenerate_eos(safe_real a, safe_real b) {
	A_ = a;
	B_ = b;
}

template<int NDIM>
void physics<NDIM>::set_dual_energy_switches(safe_real one, safe_real two) {
	de_switch_1 = one;
	de_switch_2 = two;
}

template<int NDIM>
void physics<NDIM>::set_fgamma(safe_real fg) {
	fgamma_ = fg;
}

template<int NDIM>
template<int INX>
const std::vector<std::vector<safe_real>>& physics<NDIM>::find_contact_discs(const hydro::state_type &U) {
	PROFILE();
	static const cell_geometry<NDIM, INX> geo;
	auto dir = geo.direction();
	static thread_local std::vector<std::vector<safe_real>> disc(geo.NDIR / 2, std::vector<double>(geo.H_N3));
	static thread_local std::vector<safe_real> P(H_N3);
	for (int j = 0; j < geo.H_NX_XM2; j++) {
		for (int k = 0; k < geo.H_NX_YM2; k++) {
#pragma ivdep
			for (int l = 0; l < geo.H_NX_ZM2; l++) {
				const int i = geo.to_index(j + 1, k + 1, l + 1);
				const auto rho = U[rho_i][i];
				const auto rhoinv = 1.0 / U[rho_i][i];
				double hdeg = 0.0, pdeg = 0.0, edeg = 0.0;
				if (A_ != 0.0) {
					const auto x = std::pow(rho / B_, 1.0 / 3.0);
					hdeg = 8.0 * A_ / B_ * (std::sqrt(x * x + 1.0) - 1.0);
					pdeg = deg_pres(x);
					edeg = rho * hdeg - pdeg;
				}
				safe_real ek = 0.0;
				for (int dim = 0; dim < NDIM; dim++) {
					ek += pow(U[sx_i + dim][i], 2) * rhoinv * safe_real(0.5);
				}
				auto ein = U[egas_i][i] - ek - edeg;
				if (ein < de_switch_1 * U[egas_i][i]) {
					//	print( "%e\n", U[tau_i][i]);
					ein = pow(U[tau_i][i], fgamma_);
				}
				P[i] = (fgamma_ - 1.0) * ein + pdeg;
			}
		}
	}
	for (int d = 0; d < geo.NDIR / 2; d++) {
		const auto di = dir[d];
		for (int j = 0; j < geo.H_NX_XM4; j++) {
			for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
				for (int l = 0; l < geo.H_NX_ZM4; l++) {
					constexpr auto K0 = 0.1;
					const int i = geo.to_index(j + 2, k + 2, l + 2);
					const auto P_r = P[i + di];
					const auto P_l = P[i - di];
					const auto tmp1 = fgamma_ * K0;
					const auto tmp2 = std::abs(P_r - P_l) / std::min(std::abs(P_r), std::abs(P_l));
					disc[d][i] = tmp2 / tmp1;
				}
			}
		}
	}
	return disc;
}
/*** Reconstruct uses this - GPUize****/

template<int NDIM>
template<int INX>
void physics<NDIM>::post_recon(std::vector<std::vector<std::vector<safe_real>>> &Q, const hydro::x_type X, safe_real omega, bool angmom) {
	PROFILE();
	static const cell_geometry<NDIM, INX> geo;
	static const auto indices = geo.find_indices(2, geo.H_NX - 2);
	const auto dx = X[0][geo.H_DNX] - X[0][0];
	const auto xloc = geo.xloc();
	static constexpr auto levi_civita = geo.levi_civita();
	auto dir = geo.direction();

	for (int d = 0; d < geo.NDIR; d++) {
		if (d != geo.NDIR / 2) {
			if (NDIM >= 2) {
				for (int j = 0; j < geo.H_NX_XM4; j++) {
					for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
						for (int l = 0; l < geo.H_NX_ZM4; l++) {
							const int i = geo.to_index(j + 2, k + 2, l + 2);
							Q[sx_i][d][i] -= omega * (X[1][i] + 0.5 * xloc[d][1] * dx);
							Q[sy_i][d][i] += omega * (X[0][i] + 0.5 * xloc[d][0] * dx);
						}
					}
				}
			}

			for (int n = 0; n < geo.NANGMOM; n++) {
				for (int q = 0; q < NDIM; q++) {
					for (int m = 0; m < NDIM; m++) {
						const auto lc = levi_civita[n][m][q];
						if (lc != 0) {
							for (int j = 0; j < geo.H_NX_XM4; j++) {
								for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
									for (int l = 0; l < geo.H_NX_ZM4; l++) {
										const int i = geo.to_index(j + 2, k + 2, l + 2);
										const auto rho = Q[rho_i][d][i];
										Q[lx_i + n][d][i] += lc * (X[m][i] + 0.5 * xloc[d][m] * dx) * Q[sx_i + q][d][i];
									}
								}
							}
						}
					}
				}
				for (int j = 0; j < geo.H_NX_XM4; j++) {
					for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
						for (int l = 0; l < geo.H_NX_ZM4; l++) {
							const int i = geo.to_index(j + 2, k + 2, l + 2);
							const auto rho = Q[rho_i][d][i];
							Q[lx_i + n][d][i] *= rho;
						}
					}
				}
			}
			for (int dim = 0; dim < NDIM; dim++) {
				for (int j = 0; j < geo.H_NX_XM4; j++) {
					for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
						for (int l = 0; l < geo.H_NX_ZM4; l++) {
							const int i = geo.to_index(j + 2, k + 2, l + 2);
							const auto rho = Q[rho_i][d][i];
							auto &v = Q[sx_i + dim][d][i];
							Q[egas_i][d][i] += 0.5 * v * v * rho;
							v *= rho;
						}
					}
				}
			}
			for (int j = 0; j < geo.H_NX_XM4; j++) {
				for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
					for (int l = 0; l < geo.H_NX_ZM4; l++) {
						const int i = geo.to_index(j + 2, k + 2, l + 2);
						const auto rho = Q[rho_i][d][i];
						Q[pot_i][d][i] *= rho;
					}
				}
			}
			for (int j = 0; j < geo.H_NX_XM4; j++) {
				for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
					for (int l = 0; l < geo.H_NX_ZM4; l++) {
						const int i = geo.to_index(j + 2, k + 2, l + 2);
						const auto rho = Q[rho_i][d][i];
						safe_real w = 0.0;
						for (int si = 0; si < n_species_; si++) {
							w += Q[spc_i + si][d][i];
							Q[spc_i + si][d][i] *= rho;
						}
						if (w <= 0.0) {
							print("NO SPECIES %i\n", i);
							abort();
						}
						w = 1.0 / w;
						for (int si = 0; si < n_species_; si++) {
							Q[spc_i + si][d][i] *= w;
						}
					}
				}
			}
		}
	}

}

template<int NDIM>
template<int INX>
void physics<NDIM>::analytic_solution(test_type test, hydro::state_type &U, const hydro::x_type &X, safe_real time) {
	static const cell_geometry<NDIM, INX> geo;
	static safe_real rmax = 0.0;
	static std::once_flag one;

	std::call_once(one, [&X]() {
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
		safe_real rsum = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			r += X[dim][i] * X[dim][i];
			rsum += X[dim][i];
		}
		r = sqrt(r);
		double den = 0, vel = 0, pre = 0;

		if (test == BLAST) {
#if defined(OCTOTIGER_HAVE_BLAT_TEST)
			sedov::solution(time + 7e-4, r, rmax, den, vel, pre, NDIM);
			for (int dim = 0; dim < NDIM; dim++) {
				U[sx_i + dim][i] = den * vel * X[dim][i] / r;
			}
#else
			std::cout << "ERROR! Octo-Tiger was not compiled with BLAST test support!" << std::endl;
			exit(EXIT_FAILURE);
#endif
		} else if (test == SOD) {
			sod_state_t sod_state;
			exact_sod(&sod_state, &sod_init, rsum / std::sqrt(NDIM), time, 1.0 / INX);
			den = sod_state.rho;
			vel = sod_state.v;
			for (int dim = 0; dim < NDIM; dim++) {
				U[sx_i + dim][i] = den * vel / std::sqrt(NDIM);
			}
		} else if (test == CONTACT) {
			pre = 1.0;
			vel = 10.0;
			den = 1.0 + 1.0e-6 * sin(2.0 * M_PI * (X[0][i] - vel * time));
		} else if (test == KEPLER) {
			vel = 1.0 / std::sqrt(r);
			pre = 1.0e-6;
			if (r > 0.25 && r < 0.75) {
				den = 1.0;
			} else {
				den = 1.0e-6;
			}
			U[sx_i][i] = -den * vel * X[1][i] / r;
			U[sy_i][i] = +den * vel * X[0][i] / r;
		}

		U[rho_i][i] = den;
		U[tau_i][i] = pow(pre / (fgamma_ - 1.0), 1.0 / fgamma_);
		U[egas_i][i] = pre / (fgamma_ - 1.0) + 0.5 * den * vel * vel;
		U[spc_i][i] = den;
	}
}

template<int NDIM>
void physics<NDIM>::set_n_species(int n) {
	n_species_ = n;
}

template<int NDIM>
void physics<NDIM>::update_n_field() {
	// nf_ = (4 + NDIM + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2))) + n_species_;
	// ;
  nf_ = (4 + NDIM + (NDIM == 1 ? 0 : (NDIM == 3 ? 3 : (NDIM == 2 ? 1 : 0)) )) + n_species_;
}

template<int NDIM>
template<int INX>
std::vector<typename hydro_computer<NDIM, INX, physics<NDIM>>::bc_type> physics<NDIM>::initialize(physics<NDIM>::test_type t, hydro::state_type &U,
		hydro::x_type &X) {
	static const cell_geometry<NDIM, INX> geo;

	std::vector<typename hydro_computer<NDIM, INX, physics<NDIM>>::bc_type> bc(2 * NDIM);

	for (int i = 0; i < 2 * NDIM; i++) {
		bc[i] = hydro_computer<NDIM, INX, physics<NDIM>>::OUTFLOW;
	}

	switch (t) {
	case SOD:
		break;
	case BLAST:
		break;
	case KEPLER:
		rho_sink_radius_ = 0.05;
		rho_sink_floor_ = 1.0e-10;
		set_central_force(1);
		break;
	case KH:
	case CONTACT:
		for (int i = 0; i < 2 * NDIM; i++) {
			bc[i] = hydro_computer<NDIM, INX, physics<NDIM>>::PERIODIC;
		}
		break;
	}
	const auto xlocs = geo.xloc();
	const auto weights = geo.volume_weight();
	std::array < safe_real, NDIM > x;
	safe_real rho = 0, vx = 0, vy = 0, vz = 0, p = 0, r;
	safe_real x2, xsum;
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
		for (int f = 0; f < nf_; f++) {
			U[f][i] = 0.0;
		}
		const auto xlocs = geo.xloc();
		const auto weights = geo.volume_weight();
		std::array < safe_real, NDIM > x;
		double rho = 0, vx = 0, vy = 0, vz = 0, p = 0, r;
		safe_real x2, xsum, xhalf;
		xhalf = -X[0][geo.to_index(geo.H_BW, geo.H_BW, geo.H_BW)] / 2;
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = X[dim][i];
		}
		xsum = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			xsum += x[dim];
		}
		const auto eps = []() {
			return (rand() + 0.5) / RAND_MAX * 1.0e-3;
		};

		x2 = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			x2 += x[dim] * x[dim];
		}
		r = sqrt(x2);

		switch (t) {
		case CONTACT:
			p = 1.0;
			vx = 10.0;
			rho = 1.0 + 1.0e-6 * sin(2.0 * M_PI * x[0]);
			U[rho_i][i] += rho;
			U[spc_i][i] += rho;
			break;
		case SOD:
			if (xsum < -xhalf / 2.0) {
				rho = 1.0;
				p = 1.0;
			} else {
				rho = 0.125;
				p = 0.1;
			}
			U[rho_i][i] += rho;
			if (xsum < -xhalf) {
				U[spc_i + 0][i] += rho;
			} else if (xsum < 0.0) {
				U[spc_i + 1][i] += rho;
			} else if (xsum < xhalf) {
				U[spc_i + 2][i] += rho;
			} else {
				U[spc_i + 3][i] += rho;
			}
			break;
		case BLAST:
#if defined(OCTOTIGER_HAVE_BLAT_TEST)

			double v;
			sedov::solution(7e-4, r, std::sqrt(3) + 5.0 * dx, rho, v, p, NDIM);
			p = std::max((fgamma_ - 1.0) * 1.0e-20, p);
//				/***************/
//				if( r < 1.5*dx ) {
//					p = 1.0e+3;
//				} else {
//					p = 1.0e-3;
//				}
//				rho = 1.0;
//				/**************/
			vx = v * X[0][i] / r;
			if constexpr (NDIM >= 2) {
				vy = v * X[1][i] / r;
			}
			if constexpr (NDIM == 3) {
				vz = v * X[2][i] / r;
			}
			U[rho_i][i] += rho;
			U[spc_i][i] += rho;
			break;
#else
			std::cout << "ERROR! Octo-Tiger was not compiled with BLAST test support!" << std::endl;
			exit(EXIT_FAILURE);
#endif
		case KH:

			U[physics < NDIM > ::tau_i][i] = 1.0;
			p = 1.0;
			if (x[1] < 0.0) {
				rho = 1.0 + eps();
				vx = -0.5;
			} else {
				rho = 2.0 + eps();
				vx = +0.5;
			}
			U[rho_i][i] += rho;
			U[spc_i][i] += rho;
			break;
		case KEPLER:
			p = 1.0e-10 / 100.0;
			vx = -X[1][i] * std::pow(r, -1.5);
			vy = +X[0][i] * std::pow(r, -1.5);
			if (r > 0.1 && r < 0.4) {
				rho = 1.0;
			} else {
				rho = 1.0e-10;
			}
			vz = 0.0;
			U[rho_i][i] += rho;
			U[spc_i][i] += rho;
			break;

		}
		U[sx_i][i] += (rho * vx);
		U[egas_i][i] += (p / (fgamma_ - 1.0) + 0.5 * rho * vx * vx);
		U[tau_i][i] += (std::pow(p / (fgamma_ - 1.0), 1.0 / fgamma_));
		if HOST_CONSTEXPR (NDIM >= 2) {
			U[sy_i][i] += rho * vy;
			U[egas_i][i] += 0.5 * rho * vy * vy;
		}
		if HOST_CONSTEXPR (NDIM >= 3) {
			U[sz_i][i] += rho * vz;
			U[egas_i][i] += 0.5 * rho * vz * vz;
		}
		static constexpr auto levi_civita = geo.levi_civita();
		for (int n = 0; n < geo.NANGMOM; n++) {
			U[lx_i + n][i] = 0.0;
			for (int m = 0; m < NDIM; m++) {
				for (int l = 0; l < NDIM; l++) {
					U[lx_i + n][i] += levi_civita[n][m][l] * X[m][i] * U[sx_i + l][i];
				}
			}
		}
	}

	return bc;
}

template<int NDIM>
template<int INX>
void physics<NDIM>::enforce_outflow(hydro::state_type &U, int dim, int dir) {
	static const cell_geometry<NDIM, INX> geo;
	int lb, ub;
	if (dir == 1) {
		lb = geo.H_NX - geo.H_BW;
		ub = geo.H_NX;
	} else {
		lb = 0;
		ub = geo.H_BW;
	}
	for (int j = 0; j < geo.H_NX_Y; j++) {
		for (int k = 0; k < geo.H_NX_Z; k++) {
			for (int l = lb; l < ub; l++) {
				int i;
				if (dim == 0) {
					i = geo.to_index(l, j, k);
				} else if (dim == 1) {
					i = geo.to_index(j, l, k);
				} else if (dim == 2) {
					i = geo.to_index(j, k, l);
				}
				if (dir == +1) {
					U[sx_i + dim][i] = std::max(U[sx_i + dim][i], 0.0);
				} else if (dir == -1) {
					U[sx_i + dim][i] = std::min(U[sx_i + dim][i], 0.0);
				}
			}
		}
	}
}

#endif /* OCTOTIGER_UNITIGER_PHYSICS_IMPL_HPP_ */
