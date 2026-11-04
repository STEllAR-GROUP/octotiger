//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <array>
#include <cmath>
#include <fenv.h>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <hpx/include/future.hpp>

#include "octotiger/defs.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/radiation/implicit.hpp"
#include "octotiger/radiation/kernel_interface.hpp"
#include "octotiger/radiation/opacities.hpp"
#include "octotiger/radiation/rad_grid.hpp"
#include "octotiger/radiation/subcycle.hpp"
#include "octotiger/real.hpp"
#include "octotiger/roe.hpp"
#include "octotiger/space_vector.hpp"
#include "octotiger/unitiger/radiation/radiation_physics_impl.hpp"

#if !defined(HPX_COMPUTE_DEVICE_CODE)

using real = double;

#define CHECK_FLUX( er, fx, fy, fz) if( ((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c) > 1 ) {printf( "flux exceded %s %i %e fx %e fy %e fz %e er %e\n", __FILE__, __LINE__, sqrt(((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c)), fx, fy, fz, er*physcon().c); abort();}

std::unordered_map<std::string, int> rad_grid::str_to_index;
std::unordered_map<int, std::string> rad_grid::index_to_str;

void rad_grid::static_init() {
	str_to_index["er"] = er_i;
	str_to_index["fx"] = fx_i;
	str_to_index["fy"] = fy_i;
	str_to_index["fz"] = fz_i;
	for (auto const &s : str_to_index) {
		index_to_str[s.second] = s.first;
	}
}

std::vector<std::string> rad_grid::get_field_names() {
	std::vector < std::string > rc;
	for (auto i : str_to_index) {
		rc.push_back(i.first);
	}
	return rc;
}

void rad_grid::set(const std::string name, real *data) {
	auto iter = str_to_index.find(name);
	real eunit = opts().problem == MARSHAK ? 1 : opts().code_to_g / std::pow(opts().code_to_s, 2) / opts().code_to_cm;
	real funit = opts().problem == MARSHAK ? 1 : eunit * opts().code_to_cm / opts().code_to_s;
	if (iter != str_to_index.end()) {
		int f = iter->second;
		int jjj = 0;
		for (int i = 0; i < INX; i++) {
			for (int j = 0; j < INX; j++) {
				for (int k = 0; k < INX; k++) {
					int const iii = rindex(k + RAD_BW, j + RAD_BW, i + RAD_BW);
					data[jjj] /= f == er_i ? eunit : funit;
					U[f][iii] = data[jjj];
					jjj++;
				}
			}
		}
	}

}

std::vector<silo_var_t> rad_grid::var_data() const {
	std::vector < silo_var_t > s;
	real eunit = opts().problem == MARSHAK ? 1 : opts().code_to_g / std::pow(opts().code_to_s, 2) / opts().code_to_cm;
	real funit = opts().problem == MARSHAK ? 1 : eunit * opts().code_to_cm / opts().code_to_s;
	for (auto l : str_to_index) {
		int const f = l.second;
		std::string this_name = l.first;
		int jjj = 0;
		silo_var_t this_s(this_name);
		for (int i = 0; i < INX; i++) {
			for (int j = 0; j < INX; j++) {
				for (int k = 0; k < INX; k++) {
					int const iii = rindex(k + RAD_BW, j + RAD_BW, i + RAD_BW);
					this_s(jjj) = U[f][iii];
					this_s(jjj) *= f == er_i ? eunit : funit;
					this_s.set_range(this_s(jjj));
					jjj++;
				}
			}
		}
		s.push_back(std::move(this_s));
	}
	return std::move(s);
}

constexpr auto _0 = real(0);
constexpr auto _1 = real(1);
constexpr auto _2 = real(2);
constexpr auto _3 = real(3);
constexpr auto _4 = real(4);
constexpr auto _5 = real(5);


void rad_grid::rad_imp(std::vector<real> &egas, std::vector<real> &tau, std::vector<real> &sx, std::vector<real> &sy,
		std::vector<real> &sz, std::vector<real> const &rho, real dt) {
	PROFILE()
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);

#define Sqrt(a) sqrt(a)
#define Power(a,b) pow(a,b)

	auto const find4root =
			[](double A, double B, double G) {

				return -0.5
						* Sqrt(
								(4*Power(0.66666666666666666666,0.333333333333333333333333)*G)/ Power(9*A*Power(B,2) + Sqrt(3)* Sqrt(27*Power(A,2)*Power(B,4) - 256*Power(A,3)*Power(G,3)), 0.333333333333333333333333) + Power(9*A*Power(B,2) + Sqrt(3)*Sqrt(27*Power(A,2)*Power(B,4) - 256*Power(A,3)*Power(G,3)), 0.333333333333333333333333)/ (Power(2,0.333333333333333333333333)*Power(3,0.66666666666666666666)*A))
						+
						Sqrt(
								(-4*Power(0.66666666666666666666,0.333333333333333333333333)*G)/ Power(9*A*Power(B,2) + Sqrt(3)* Sqrt(27*Power(A,2)*Power(B,4) - 256*Power(A,3)*Power(G,3)), 0.333333333333333333333333) - Power(9*A*Power(B,2) + Sqrt(3)*Sqrt(27*Power(A,2)*Power(B,4) - 256*Power(A,3)*Power(G,3)), 0.333333333333333333333333)/ (Power(2,0.333333333333333333333333)*Power(3,0.66666666666666666666)*A) + (2*B)/(A*Sqrt((4*Power(0.66666666666666666666,0.3333333333333333333333333333)*G)/ Power(9*A*Power(B,2) + Sqrt(3)*Sqrt(27*Power(A,2)*Power(B,4) - 256*Power(A,3)*Power(G,3)), 0.3333333333333333333333333333) + Power(9*A*Power(B,2) + Sqrt(3)*Sqrt(27*Power(A,2)*Power(B,4) - 256*Power(A,3)*Power(G,3)), 0.333333333333333333333333)/ (Power(2,0.3333333333333333)*Power(3,0.66666666666666666666)*A))))
								/ 2.;

			};
	integer const d = H_BW - RAD_BW;
	real const clight = physcon().c; // / opts().clight_retard;
	real const sigma = physcon().sigma;
	real const kb = physcon().kb;
	real const mh = physcon().mh;
	real const clightinv = INVERSE(clight);
	real const fgamma = grid::get_fgamma();
	auto const chat = physcon().c / opts().clight_retard;
	for (integer xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
		for (integer yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
			for (integer zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
				integer const D = H_BW - RAD_BW;
				integer const iiir = rindex(xi, yi, zi);
				integer const iiih = hindex(xi + D, yi + D, zi + D);
				CHECK_FLUX(U[er_i][iiir], U[fx_i][iiir], U[fy_i][iiir], U[fz_i][iiir]);
				real const rhoinv = INVERSE(rho[iiih]);
				real vx = sx[iiih] * rhoinv;
				real vy = sy[iiih] * rhoinv;
				real vz = sz[iiih] * rhoinv;
				real e0 = egas[iiih];
				e0 -= 0.5 * vx * vx * rho[iiih];
				e0 -= 0.5 * vy * vy * rho[iiih];
				e0 -= 0.5 * vz * vz * rho[iiih];
				if (opts().eos == WD) {
					e0 -= ztwd_energy(rho[iiih]);
				}
				if (e0 < egas[iiih] * 0.001) {
					e0 = std::pow(tau[iiih], fgamma);
				}
				real kap_p = kappa_p(rho[iiih], e0, mmw[iiih], X_spc[iiih], Z_spc[iiih]);
				real kap_r = kappa_R(rho[iiih], e0, mmw[iiih], X_spc[iiih], Z_spc[iiih]);

				auto dFx = kap_r * (4.0 / 3.0) * (sx[iiih] / rho[iiih]) * U[er_i][iiir];
				auto dFy = kap_r * (4.0 / 3.0) * (sy[iiih] / rho[iiih]) * U[er_i][iiir];
				auto dFz = kap_r * (4.0 / 3.0) * (sz[iiih] / rho[iiih]) * U[er_i][iiir];

				auto dEx = kap_r * (sx[iiih] / rho[iiih] / clight) * U[fx_i][iiir];
				auto dEy = kap_r * (sy[iiih] / rho[iiih] / clight) * U[fy_i][iiir];
				auto dEz = kap_r * (sz[iiih] / rho[iiih] / clight) * U[fz_i][iiir];

				U[fx_i][iiir] += dFx * dt;
				U[fy_i][iiir] += dFy * dt;
				U[fz_i][iiir] += dFz * dt;
				sx[iiih] -= dFx * dt;
				sy[iiih] -= dFy * dt;
				sz[iiih] -= dFz * dt;
				egas[iiih] -= dEx * dt;
				egas[iiih] -= dEy * dt;
				egas[iiih] -= dEz * dt;
				U[er_i][iiir] += dEx * dt;
				U[er_i][iiir] += dEy * dt;
				U[er_i][iiir] += dEz * dt;

				kap_p = kappa_p(rho[iiih], e0, mmw[iiih], X_spc[iiih], Z_spc[iiih]);
				kap_r = kappa_R(rho[iiih], e0, mmw[iiih], X_spc[iiih], Z_spc[iiih]);
				vx = sx[iiih] * rhoinv;
				vy = sy[iiih] * rhoinv;
				vz = sz[iiih] * rhoinv;
				e0 = egas[iiih];
				e0 -= 0.5 * vx * vx * rho[iiih];
				e0 -= 0.5 * vy * vy * rho[iiih];
				e0 -= 0.5 * vz * vz * rho[iiih];
				if (opts().eos == WD) {
					e0 -= ztwd_energy(rho[iiih]);
				}
				if (e0 < egas[iiih] * 0.001) {
					e0 = std::pow(tau[iiih], fgamma);
				}
		//		printf( "%e\n", mmw[iiih]); cm * g / s^2  (g /  (1/ s^2 / cm, 4)
				real const A = 4.0 * dt * kap_p * sigma * pow(mmw[iiih] * mh * (fgamma - 1.) / (kb * rho[iiih]), 4.0);
				real const B = (1.0 + clight * dt * kap_p);
				real const C = -(1.0 + clight * dt * kap_p) * e0 - U[er_i][iiir] * dt * clight * kap_p;
		//		printf( "2. %e %e %e\n", A, B, C );
				real newE = find4root(A, B, C);
				real dE = e0 - newE;
				egas[iiih] -= dE;
				U[er_i][iiir] += dE;
				tau[iiih] = pow(e0 - dE, 1.0 / fgamma);

				auto fx0 = U[fx_i][iiir];
				auto fy0 = U[fy_i][iiir];
				auto fz0 = U[fz_i][iiir];
				U[fx_i][iiir] = fx0 / (1.0 + clight * dt * kap_r);
				U[fy_i][iiir] = fy0 / (1.0 + clight * dt * kap_r);
				U[fz_i][iiir] = fz0 / (1.0 + clight * dt * kap_r);
				U[sx_i][iiir] += fx0 - U[fx_i][iiir];
				U[sy_i][iiir] += fy0 - U[fy_i][iiir];
				U[sz_i][iiir] += fz0 - U[fz_i][iiir];

				/*	if (((U[fx_i][iiir]) * (U[fx_i][iiir]) + (U[fy_i][iiir]) * (U[fy_i][iiir])
				 + (U[fz_i][iiir]) * (U[fz_i][iiir])) / (U[er_i][iiir] * U[er_i][iiir] * physcon().c * physcon().c)
				 > 1) {
				 printf("%e %e %e %e %i %i %i\n", U[fx_i][iiir], U[fy_i][iiir], U[fz_i][iiir],
				 U[er_i][iiir] * (physcon().c), xi, yi, zi);
				 }*/
				CHECK_FLUX(U[er_i][iiir], U[fx_i][iiir], U[fy_i][iiir], U[fz_i][iiir]);
			}
		}
	}
}

void rad_grid::set_dx(real _dx) {
	dx = _dx;
}

void rad_grid::set_X(std::vector<std::vector<real>> const &x) {
	X.resize(NDIM);
	for (integer d = 0; d != NDIM; ++d) {
		X[d].resize(RAD_N3);
		for (integer xi = 0; xi != RAD_NX; ++xi) {
			for (integer yi = 0; yi != RAD_NX; ++yi) {
				for (integer zi = 0; zi != RAD_NX; ++zi) {
					auto const D = H_BW - RAD_BW;
					integer const iiir = rindex(xi, yi, zi);
					integer const iiih = hindex(xi + D, yi + D, zi + D);
					//		printf( "%i %i %i %i %i %i \n", d, iiir, xi, yi, zi, iiih);
					X[d][iiir] = x[d][iiih];
				}
			}
		}
	}
}

real rad_grid::hydro_signal_speed(std::vector<real> const &egas, std::vector<real> const &tau,
		std::vector<real> const &sx, std::vector<real> const &sy, std::vector<real> const &sz,
		std::vector<real> const &rho) {
	real a = 0.0;
	real const fgamma = grid::get_fgamma();
	for (integer xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
		for (integer yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
			for (integer zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
				integer const D = H_BW - RAD_BW;
				integer const iiir = rindex(xi, yi, zi);
				integer const iiih = hindex(xi + D, yi + D, zi + D);
				real const rhoinv = INVERSE(rho[iiih]);
				real vx = sx[iiih] * rhoinv;
				real vy = sy[iiih] * rhoinv;
				real vz = sz[iiih] * rhoinv;
				real e0 = egas[iiih];
				e0 -= 0.5 * vx * vx * rho[iiih];
				e0 -= 0.5 * vy * vy * rho[iiih];
				e0 -= 0.5 * vz * vz * rho[iiih];
				if (opts().eos == WD) {
					e0 -= ztwd_energy(rho[iiih]);
				}
				if (e0 < egas[iiih] * 0.001) {
					e0 = std::pow(tau[iiih], fgamma);
				}

				real this_a = (4.0 / 9.0) * U[er_i][iiir] * rhoinv;
				//		printf( "%e %e %e %e\n",rho[iiih], e0, mmw[iiir],dx );
				real const cons = kappa_R(rho[iiih], e0, mmw[iiir], X_spc[iiir], Z_spc[iiir]) * dx;
				if (cons < 32.0) {
					this_a *= std::max(1.0 - std::exp(-cons), 0.0);
				}
				a = std::max(this_a, a);
			}
		}
	}
	return SQRT(a);
}

void rad_grid::compute_mmw(std::vector<std::vector<safe_real>> const &U) {
	mmw.resize(RAD_N3);
	X_spc.resize(RAD_N3);
	Z_spc.resize(RAD_N3);
	for (integer i = 0; i != RAD_NX; ++i) {
		for (integer j = 0; j != RAD_NX; ++j) {
			for (integer k = 0; k != RAD_NX; ++k) {
				integer const d = H_BW - RAD_BW;
				integer const iiir = rindex(i, j, k);
				integer const iiih = hindex(i + d, j + d, k + d);
				specie_state_t < real > spc;
				for (integer si = 0; si != opts().n_species; ++si) {
					spc[si] = U[spc_i + si][iiih];
				}
				mean_ion_weight(spc, mmw[iiir], X_spc[iiir], Z_spc[iiir]);
			}
		}
	}

}

void node_server::compute_radiation(real dt, real omega) {
    // Retain the legacy transport/source method during the staged migration.
    // All nodes participate in exchanges; only physical leaves evolve state.
    auto &rgrid = *rad_grid_ptr;
    rgrid.set_dx(grid_ptr->get_dx());
    rgrid.set_X(grid_ptr->get_X());
    real local_ratio = 0;
    real const c = physcon().c;
    real const retard = opts().clight_retard;
    real const dx = grid_ptr->get_dx();
    if (!(dt >= 0) || !std::isfinite(dt) || !(c > 0) || !std::isfinite(c) ||
        !(retard > 0) || !std::isfinite(retard) || !(dx > 0) || !std::isfinite(dx) ||
        !std::isfinite(current_time) ||
        static_cast<long double>(current_time) + dt > std::numeric_limits<real>::max()) {
        local_ratio = std::numeric_limits<real>::max();
    } else if (!is_refined) {
        // Compute in extended precision before narrowing, avoiding an overflow
        // or divide-by-zero trap before the whole tree can reject the step.
        long double const ratio = static_cast<long double>(dt) * c * 10 / dx / retard;
        local_ratio = static_cast<real>(std::min(ratio,
            static_cast<long double>(std::numeric_limits<real>::max())));
    }
    // A single tree-wide count keeps all cycle-numbered exchange channels aligned.
    int const substeps = radiation::subcycle_count(radiation_iteration_max(local_ratio));
    if (dt == 0) {
        all_rad_bounds(current_time);
        return;
    }
    if (!is_refined) {
        rgrid.compute_mmw(grid_ptr->U);
    }
    for (int index = 0; index < substeps; ++index) {
        auto const interval = radiation::substep(dt, substeps, index);
        real const step = interval.duration();
        if (!is_refined && index == 0 && opts().rad_implicit) {
            rgrid.rad_imp(grid_ptr->get_field(egas_i), grid_ptr->get_field(tau_i),
                grid_ptr->get_field(sx_i), grid_ptr->get_field(sy_i), grid_ptr->get_field(sz_i),
                grid_ptr->get_field(rho_i), 0.5 * step);
        }
        rgrid.store();
        constexpr std::array<double, 2> beta{1.0, 0.5};
        for (int stage = 0; stage < 2; ++stage) {
            // Legacy RK2 stages evaluate physical boundaries at t_n and t_n + dt.
            all_rad_bounds(current_time + (stage == 0 ? interval.begin : interval.end));
            rgrid.compute_flux(omega);
            GET(exchange_rad_flux_corrections());
            if (!is_refined) {
                rgrid.advance(step, beta[stage]);
            }
        }
        if (!is_refined && opts().rad_implicit) {
            real const source_dt = index + 1 == substeps ? 0.5 * step : step;
            rgrid.rad_imp(grid_ptr->get_field(egas_i), grid_ptr->get_field(tau_i),
                grid_ptr->get_field(sx_i), grid_ptr->get_field(sy_i), grid_ptr->get_field(sz_i),
                grid_ptr->get_field(rho_i), source_dt);
        }
    }
    // Refresh parents before regrid/coarsening, and ghosts at the common endpoint.
    all_rad_bounds(current_time + dt);
}

template<class T>
T minmod(T a, T b) {
	return (std::copysign(0.5, a) + std::copysign(0.5, b)) * std::min(std::abs(a), std::abs(b));
}

void rad_grid::allocate() {
	U.resize(NRF);
	Ushad.resize(NRF);
	flux.resize(NDIM);
	for (int d = 0; d < NDIM; d++) {
		flux[d].resize(NRF);
	}
	for (integer f = 0; f != NRF; ++f) {
		U0[f].resize(RAD_N3);
		U[f].resize(RAD_N3);
		Ushad[f].resize(HS_N3);
		for (integer d = 0; d != NDIM; ++d) {
			flux[d][f].resize(RAD_N3);
		}
	}
}

void rad_grid::store() {
	for (integer f = 0; f != NRF; ++f) {
		for (integer i = 0; i != RAD_N3; ++i) {
			U0[f][i] = U[f][i];
		}
	}
}

void rad_grid::restore() {
	for (integer f = 0; f != NRF; ++f) {
		for (integer i = 0; i != RAD_N3; ++i) {
			U[f][i] = U0[f][i];
		}
	}
}

void rad_grid::sanity_check() {
	for (integer xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
		for (integer yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
			for (integer zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
				integer const iiir = rindex(xi, yi, zi);
				if (U[er_i][iiir] <= 0.0) {
					printf("INSANE\n");
					//		printf("%e %i %i %i\n", U[er_i][iiir], xi, yi, zi);
					abort();
				}
			}
		}
	}
}

void rad_grid::compute_flux(real omega) {
	PROFILE()
	;
	radiation_physics < NDIM > ::set_clight(physcon().c / opts().clight_retard);
	auto const &q = hydro.reconstruct(U, X, omega);
	hydro.flux(U, q, flux, X, omega);
}

void rad_grid::change_units(real m, real l, real t, real k) {
	real const l2 = l * l;
	real const t2 = t * t;
	real const t2inv = 1.0 * INVERSE(t2);
	real const tinv = 1.0 * INVERSE(t);
	real const l3 = l2 * l;
	real const l3inv = 1.0 * INVERSE(l3);
	for (integer i = 0; i != RAD_N3; ++i) {
		U[er_i][i] *= (m * l2 * t2inv) * l3inv;
		U[fx_i][i] *= tinv * (m * t2inv);
		U[fy_i][i] *= tinv * (m * t2inv);
		U[fz_i][i] *= tinv * (m * t2inv);
	}
}

void rad_grid::advance(real dt, real beta) {
	real const l = dt * INVERSE(dx);
	integer const D[3] = { DX, DY, DZ };
	for (integer f = 0; f != NRF; ++f) {
		for (integer xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
			for (integer yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
				for (integer zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
					integer const iii = rindex(xi, yi, zi);
					CHECK_FLUX(U[er_i][iii], U[fx_i][iii], U[fy_i][iii], U[fz_i][iii]);
					real const &u0 = U0[f][iii];
					real u1 = U[f][iii];
					for (integer d = 0; d != NDIM; ++d) {
						u1 -= l * (flux[d][f][iii + D[d]] - flux[d][f][iii]);
					}
					U[f][iii] = u0 * (1.0 - beta) + beta * u1;
					CHECK_FLUX(U[er_i][iii], U[fx_i][iii], U[fy_i][iii], U[fz_i][iii]);
				}
			}
		}
	}
}

void rad_grid::set_physical_boundaries(geo::face face, real t) {
	for (integer i = 0; i != RAD_NX; ++i) {
		for (integer j = 0; j != RAD_NX; ++j) {
			for (integer k = 0; k != RAD_BW; ++k) {
				integer iii1, iii0;
				switch (face) {
				case 0:
					iii1 = rindex(k, i, j);
					iii0 = rindex(RAD_BW, i, j);
					break;
				case 1:
					iii1 = rindex(RAD_NX - 1 - k, i, j);
					iii0 = rindex(RAD_NX - 1 - RAD_BW, i, j);
					break;
				case 2:
					iii1 = rindex(i, k, j);
					iii0 = rindex(i, RAD_BW, j);
					break;
				case 3:
					iii1 = rindex(i, RAD_NX - 1 - k, j);
					iii0 = rindex(i, RAD_NX - 1 - RAD_BW, j);
					break;
				case 4:
					iii1 = rindex(i, j, k);
					iii0 = rindex(i, j, RAD_BW);
					break;
				case 5:
				default:
					iii1 = rindex(i, j, RAD_NX - 1 - k);
					iii0 = rindex(i, j, RAD_NX - 1 - RAD_BW);
				}
				for (integer f = 0; f != NRF; ++f) {
					U[f][iii1] = U[f][iii0];
				}
				switch (face) {
				case 0:
					if (opts().problem == MARSHAK) {
						if (t > 0) {
							auto u = marshak_wave_analytic(-opts().xscale, 0, 0, t);
							U[fx_i][iii1] = u[opts().n_fields + fx_i];
							U[er_i][iii1] = std::max(u[opts().n_fields + er_i], 1.0e-10);
						} else {
							U[fx_i][iii1] = 0.0;
							U[er_i][iii1] = 1.0e-10;
						}
					} else {
						U[fx_i][iii1] = std::min(U[fx_i][iii1], 0.0);
					}
					break;
				case 1:
					U[fx_i][iii1] = std::max(U[fx_i][iii1], 0.0);
					break;
				case 2:
					U[fy_i][iii1] = std::min(U[fy_i][iii1], 0.0);
					break;
				case 3:
					U[fy_i][iii1] = std::max(U[fy_i][iii1], 0.0);
					break;
				case 4:
					U[fz_i][iii1] = std::min(U[fz_i][iii1], 0.0);
					break;
				case 5:
					U[fz_i][iii1] = std::max(U[fz_i][iii1], 0.0);
					break;
				}
			}
		}
	}
}

void rad_grid::initialize_erad(const std::vector<safe_real> rho, const std::vector<safe_real> tau) {
	real const fgamma = grid::get_fgamma();
	for (integer xi = 0; xi != RAD_NX; ++xi) {
		for (integer yi = 0; yi != RAD_NX; ++yi) {
			for (integer zi = 0; zi != RAD_NX; ++zi) {
				auto const D = H_BW - RAD_BW;
				integer const iiir = rindex(xi, yi, zi);
				integer const iiih = hindex(xi + D, yi + D, zi + D);
				real const ei = POWER(tau[iiih], fgamma);
				//	U[er_i][iiir] = B_p((double) rho[iiih], (double) ei, (double) mmw[iiir]) * (4.0 * M_PI / physcon().c);
				//	U[fx_i][iiir] = U[fy_i][iiir] = U[fz_i][iiir] = 0.0;
			}
		}
	}
}

rad_grid::rad_grid(real _dx) :
		dx(_dx), is_coarse(HS_N3), has_coarse(HS_N3) {
	allocate();
}

rad_grid::rad_grid() :
		dx(0), is_coarse(HS_N3), has_coarse(HS_N3) {
	allocate();
}

void rad_grid::set_field(real v, integer f, integer i, integer j, integer k) {
	U[f][rindex(i, j, k)] = v;
}

real rad_grid::get_field(integer f, integer i, integer j, integer k) const {
	return U[f][rindex(i, j, k)];
}

;

using erad_init_action_type = node_server::erad_init_action;
HPX_REGISTER_ACTION(erad_init_action_type);

hpx::future<void> node_client::erad_init() const {
	return hpx::async<typename node_server::erad_init_action>(get_unmanaged_gid());
}

void node_server::erad_init() {
	std::array<hpx::future<void>, NCHILD> futs;
	int index = 0;
	if (is_refined) {
		for (auto &child : children) {
			futs[index++] = child.erad_init();
		}
	}
	grid_ptr->rad_init();
	if (is_refined) {
		hpx::wait_all(futs);
	}
}

#endif
