/*
 * problem.cpp
 *
 *  Created on: May 29, 2015
 *      Author: dmarce1
 */

#include "defs.hpp"
#include "problem.hpp"
#include "grid.hpp"
#include "lane_emden.hpp"
#include <cmath>
#include "exact_sod.hpp"

init_func_type problem = nullptr;
refine_test_type refine_test_function = refine_test;

#ifdef RADIATION
bool refine_test(integer level, integer max_level, real x, real y, real z, std::vector<real> U) {
	bool rc = false;
	real den_floor = 1.0e-4;
	integer test_level = max_level;
	for (integer this_test_level = test_level; this_test_level >= 1; --this_test_level) {
		if (U[rho_i] > den_floor) {
			rc = rc || (level < this_test_level);
		}
		if (rc) {
			break;
		}
		den_floor /= 8.0;
	}
	return rc;

}

bool radiation_test_refine(integer level, integer max_level, real x, real y, real z, std::vector<real> U, std::array<std::vector<real>, NDIM> const& dudx) {

	return level < max_level;

	bool rc = false;
	real den_floor = 1.0e-1;
	integer test_level = max_level;
	for (integer this_test_level = test_level; this_test_level >= 1; --this_test_level) {
		if (U[rho_i] > den_floor) {
			rc = rc || (level < this_test_level);
		}
		if (rc) {
			break;
		}
		den_floor /= 8.0;
	}
	return rc;

}
#endif


std::vector<real> radiation_test_problem(real x, real y, real z, real dx) {
	std::vector<real> u(NF + NRF, real(0));
	x -= 0.0;
	y -= 0.0;
	z -= 0.0;
	real r = std::max(dx, 0.50);
	const real eint = 1.0e-1;
	if (std::sqrt(x * x + y * y + z * z) < r) {
		u[rho_i] = 1.0;
	} else {
		u[rho_i] = 1.0e-10;
	}
	u[tau_i] = std::pow( eint * u[rho_i], 1.0 / grid::get_fgamma() );
//	u[sx_i] = 0.0; //u[rho_i] / 10.0;
	const real fgamma = grid::get_fgamma();
	u[egas_i] = std::pow(u[tau_i], fgamma);
	u[egas_i] += u[sx_i] * u[sx_i] / u[rho_i] / 2.0;
	u[egas_i] += u[sy_i] * u[sy_i] / u[rho_i] / 2.0;
	u[egas_i] += u[sz_i] * u[sz_i] / u[rho_i] / 2.0;
	u[spc_ac_i] = u[rho_i];
	return u;
}

bool refine_sod(integer level, integer max_level, real x, real y, real z, std::vector<real> const& U, std::array<std::vector<real>, NDIM> const& dudx) {
	for (integer i = 0; i != NDIM; ++i) {
		if (std::abs(dudx[i][rho_i] / U[rho_i]) > 0.1) {
			return level < max_level;
		}
	}
	return false;
}

bool refine_blast(integer level, integer max_level, real x, real y, real z, std::vector<real> const& U, std::array<std::vector<real>, NDIM> const& dudx) {
	for (integer i = 0; i != NDIM; ++i) {
		if (std::abs(dudx[i][rho_i] / U[rho_i]) > 0.1) {
			return level < max_level;
		}
		if (std::abs(dudx[i][tau_i]) > 0.01) {
			return level < max_level;
		}
	}
	return false;
}

bool refine_test(integer level, integer max_level, real x, real y, real z, std::vector<real> const& U, std::array<std::vector<real>, NDIM> const& dudx) {
	bool rc = false;
	real den_floor = 1.0e-4;
	integer test_level = max_level;
	for (integer this_test_level = test_level; this_test_level >= 1; --this_test_level) {
		if (U[rho_i] > den_floor) {
			rc = rc || (level < this_test_level);
		}
		if (rc) {
			break;
		}
		den_floor /= 8.0;
	}
	return rc;

}

bool refine_test_bibi(integer level, integer max_level, real x, real y, real z, std::vector<real> const& U, std::array<std::vector<real>, NDIM> const& dudx) {
	bool rc = false;
	real den_floor = 1.0e-2;
	//integer test_level = ((U[spc_de_i]+U[spc_dc_i]) < 0.5*U[rho_i] ? max_level  - 1 : max_level);
//	integer test_level = ((U[spc_ae_i]+U[spc_de_i]) > 0.5*U[rho_i] ? max_level  - 1 : max_level);
	integer test_level = ((U[spc_dc_i] + U[spc_de_i]) > 0.5 * U[rho_i] ? max_level - 1 : max_level);
//	integer test_level = max_level;
	for (integer this_test_level = test_level; this_test_level >= 1; --this_test_level) {
		if (U[rho_i] > den_floor) {
			rc = rc || (level < this_test_level);
		}
		if (rc) {
			break;
		}
		den_floor /= 8.0;
	}
	return rc;

}

void set_refine_test(const refine_test_type& rt) {
	refine_test_function = rt;
}

refine_test_type get_refine_test() {
	return refine_test_function;
}

void set_problem(const init_func_type& p) {
	problem = p;
}

init_func_type get_problem() {
	return problem;
}

/*
std::vector<real> null_problem(real x, real y, real z, real dx) {
	std::vector<real> u(NF, real(0));
	return u;
}*/

std::vector<real> blast_wave(real x, real y, real z, real dx) {
	const real fgamma = grid::get_fgamma();
	x -= 0.453;
	y -= 0.043;
	std::vector<real> u(NF, real(0));
	u[spc_dc_i] = u[rho_i] = 1.0;
	const real a = std::sqrt(2.0) * dx;
	real r = std::sqrt(x * x + y * y + z * z);
	u[egas_i] = std::max(1.0e-10, exp(-r * r / a / a));
	u[tau_i] = std::pow(u[egas_i], ONE / fgamma);
	return u;
}

std::vector<real> sod_shock_tube_init(real x0, real y, real z, real t) {
	std::vector<real> U(NF, 0.0);
	const real fgamma = grid::get_fgamma();
	sod_state_t s;
	real x = (x0 + y + z) / std::sqrt(3.0);
	exact_sod(&s, &sod_init, x, 0.0);
	U[rho_i] = s.rho;
	U[egas_i] = s.p / (fgamma - 1.0);
	U[sx_i] = s.rho * s.v / std::sqrt(3.0);
	U[sy_i] = s.rho * s.v / std::sqrt(3.0);
	U[sz_i] = s.rho * s.v / std::sqrt(3.0);
	U[tau_i] = std::pow(U[egas_i], 1.0 / fgamma);
	U[egas_i] += s.rho * s.v * s.v / 2.0;
	U[spc_ac_i] = s.rho;
	//printf( "%e %e\n", t, s.v);
	return U;
}

std::vector<real> sod_shock_tube_analytic(real x0, real y, real z, real t) {
	std::vector<real> U(NF, 0.0);
	const real fgamma = grid::get_fgamma();
	sod_state_t s;
	real x = (x0 + y + z) / std::sqrt(3.0);
	exact_sod(&s, &sod_init, x, t);
	U[rho_i] = s.rho;
	U[egas_i] = s.p / (fgamma - 1.0);
	U[sx_i] = s.rho * s.v / std::sqrt(3.0);
	U[sy_i] = s.rho * s.v / std::sqrt(3.0);
	U[sz_i] = s.rho * s.v / std::sqrt(3.0);
	U[tau_i] = std::pow(U[egas_i], 1.0 / fgamma);
	U[egas_i] += s.rho * s.v * s.v / 2.0;
	U[spc_ac_i] = s.rho;
	return U;
}

const real dxs = 0.0;
const real dys = -0.0;

std::vector<real> double_solid_sphere_analytic_phi(real x0, real y0, real z0) {
	std::vector<real> u(4, real(0));
	auto u1 = solid_sphere_analytic_phi(x0, y0, z0, dxs);
	auto u2 = solid_sphere_analytic_phi(x0, y0, z0, dys);
	for (integer f = 0; f != 4; ++f) {
		u[f] = u1[f] + u2[f];
	}
	return u;
}

const real ssr0 = 1.0 / 3.0;
std::vector<real> solid_sphere_analytic_phi(real x, real y, real z, real xshift) {
	const real r0 = ssr0;
	const real M = 1.0;
	std::vector<real> g(4);
	x -= xshift;
//	x0 -= -0.0444;
//	y0 -= +0.345;
//	z0 -= -.2565;
	const real r = std::sqrt(x * x + y * y + z * z);
	const real r3 = r * r * r;
	const real Menc = M * std::pow(std::min(r / r0, 1.0), 3);
	if (r < r0) {
		g[phi_i] = -M * (3.0 * r0 * r0 - r * r) / (2.0 * r0 * r0 * r0);
	} else {
		g[phi_i] = -M / r;
	}
	g[gx_i] = -Menc * x / r3;
	g[gy_i] = -Menc * y / r3;
	g[gz_i] = -Menc * z / r3;
	return g;
}

std::vector<real> double_solid_sphere(real x0, real y0, real z0, real dx) {
	std::vector<real> u(NF, real(0));
	auto u1 = solid_sphere(x0, y0, z0, dx, dxs);
	auto u2 = solid_sphere(x0, y0, z0, dx, dys);
	for (integer f = 0; f != NF; ++f) {
		u[f] = u1[f] + u2[f];
	}
	return u;
}

std::vector<real> solid_sphere(real x0, real y0, real z0, real dx, real xshift) {
	const integer N = 25;
	const real r0 = ssr0;
	const real rho_floor = 1.0e-50;
	const real V = 4.0 / 3.0 * M_PI * r0 * r0 * r0;
	const real drho = 1.0 / real(N * N * N) / V;
	std::vector<real> u(NF, real(0));
	x0 -= xshift;
//	x0 -= -0.0444;
//	y0 -= +0.345;
//	z0 -= -.2565;
	const auto mm = [](real a, real b) {
		if( a * b < ZERO ) {
			return ZERO;
		} else if( a > ZERO ) {
			return std::min(a,b);
		} else {
			return std::max(a,b);
		}
	};
	const real xmax = std::max(std::abs(x0 + dx / 2.0), std::abs(x0 - dx / 2.0));
	const real ymax = std::max(std::abs(y0 + dx / 2.0), std::abs(y0 - dx / 2.0));
	const real zmax = std::max(std::abs(z0 + dx / 2.0), std::abs(z0 - dx / 2.0));
	const real xmin = mm(x0 + dx / 2.0, x0 - dx / 2.0);
	const real ymin = mm(y0 + dx / 2.0, y0 - dx / 2.0);
	const real zmin = mm(z0 + dx / 2.0, z0 - dx / 2.0);
	if (xmax * xmax + ymax * ymax + zmax * zmax <= r0 * r0) {
		u[rho_i] += drho * N * N * N;
	} else if (xmin * xmin + ymin * ymin + zmin * zmin <= r0 * r0) {
		x0 -= dx / 2.0;
		y0 -= dx / 2.0;
		z0 -= dx / 2.0;
		const real d = dx / N;
		x0 += d / 2.0;
		y0 += d / 2.0;
		z0 += d / 2.0;
		const real r2 = r0 * r0;
		for (integer i = 0; i < N; ++i) {
			const real x2 = std::pow(x0 + real(i) * d, 2);
			for (integer j = 0; j < N; ++j) {
				const real y2 = std::pow(y0 + real(j) * d, 2);
#pragma GCC ivdep
				for (integer k = 0; k < N; ++k) {
					const real z2 = std::pow(z0 + real(k) * d, 2);
					if (x2 + y2 + z2 < r2) {
						u[rho_i] += drho;
					}
				}
			}
		}
	}
	u[rho_i] = std::max(u[rho_i], rho_floor);
	return u;
}

const real x0 = 0.0;
const real y0_ = 0.0;
const real z0 = 0.0;
const real rmax = 3.7;
const real dr = rmax / 128.0;
const real alpha = 1.0;

std::vector<real> star(real x, real y, real z, real) {
	const real fgamma = grid::get_fgamma();

	x -= 0.0;
	y -= 0.0;
	z -= 0.0;
//	real menc;
	const real r = std::sqrt(x * x + y * y + z * z);
	std::vector<real> u(NF, real(0));
	real theta;
	const real n = real(1) / (fgamma - real(1));
	const real rho_min = 1.0e-10;
	const real theta_min = std::pow(rho_min, real(1) / n);
	const auto c0 = real(4) * real(M_PI) / (n + real(1));
	if (r <= rmax) {
		theta = lane_emden(r, dr);
		theta = std::max(theta, theta_min);
	} else {
		theta = theta_min;
	}
	u[rho_i] = std::max(std::pow(theta, n), 1.0e-10);
	u[spc_i] = u[rho_i];
	u[egas_i] = std::pow(theta, fgamma * n) * c0 / (fgamma - real(1));
	if (theta <= theta_min) {
		u[egas_i] *= real(100);
	}
	u[tau_i] = std::pow(u[egas_i], (real(1) / real(fgamma)));
	return u;
}

std::vector<real> moving_star(real x, real y, real z, real dx) {
	real vx = 1.0;
	real vy = 1.0;
	real vz = 0.0;
//	x += x0 + vx * t;
//	y += y0 + vy * t;
//	z += z0 + vz * t;
	auto u = star(x, y, z, dx);
	u[sx_i] = u[rho_i] * vx;
	u[sy_i] = u[rho_i] * vy;
	u[sz_i] = u[rho_i] * vz;
	u[egas_i] += (u[sx_i] * u[sx_i] + u[sy_i] * u[sy_i] + u[sz_i] * u[sz_i]) / u[rho_i] / 2.0;
	return u;
}

std::vector<real> moving_star_analytic(real x, real y, real z, real t) {
	real vx = 1.0;
	real vy = 1.0;
	real vz = 0.0;
	const real omega = grid::get_omega();
	const real x0 = x;
	const real y0 = y;
	x = x0 * cos(omega * t) - y0 * sin(omega * t);
	y = y0 * cos(omega * t) + x0 * sin(omega * t);
	x -= vx * t;
	y -= vy * t;
	z -= vz * t;
	auto u = star(x, y, z, 0.0);
	u[sx_i] = u[rho_i] * vx;
	u[sy_i] = u[rho_i] * vy;
	u[sz_i] = u[rho_i] * vz;
	u[egas_i] += (u[sx_i] * u[sx_i] + u[sy_i] * u[sy_i] + u[sz_i] * u[sz_i]) / u[rho_i] / 2.0;
	return u;
}

std::vector<real> equal_mass_binary(real x, real y, real z, real) {
	const integer don_i = spc_ac_i;
	const integer acc_i = spc_dc_i;
	const real fgamma = grid::get_fgamma();

	real theta;
	real alpha = 1.0 / 15.0;
	const real n = real(1) / (fgamma - real(1));
	const real rho_min = 1.0e-12;
	std::vector<real> u(NF, real(0));
	const real d = 1.0 / 2.0;
	real x1 = x - d;
	real x2 = x + d;
	real y1 = y;
	real y2 = y;
	real z1 = z;
	real z2 = z;

	const real r1 = std::sqrt(x1 * x1 + y1 * y1 + z1 * z1) / alpha;
	const real r2 = std::sqrt(x2 * x2 + y2 * y2 + z2 * z2) / alpha;

	const real theta_min = std::pow(rho_min, real(1) / n);
	const auto c0 = real(4) * real(M_PI) * alpha * alpha / (n + real(1));

	if (r1 <= rmax || r2 <= rmax) {
		real r = std::min(r1, r2);
		theta = lane_emden(r, dr);
		theta = std::max(theta, theta_min);
	} else {
		theta = theta_min;
	}
	u[rho_i] = std::pow(theta, n);
	u[egas_i] = std::pow(theta, fgamma * n) * c0 / (fgamma - real(1));
	u[egas_i] = std::max(u[egas_i], ei_floor);
	u[tau_i] = std::pow(u[egas_i], (real(1) / real(fgamma)));
	u[sx_i] = -DEFAULT_OMEGA * y * u[rho_i];
	u[sy_i] = +DEFAULT_OMEGA * x * u[rho_i];
	u[egas_i] += HALF * DEFAULT_OMEGA * DEFAULT_OMEGA * (x * x + y * y) * u[rho_i];
	if (x < ZERO) {
		u[acc_i] = u[rho_i];
		u[don_i] = ZERO;
	} else {
		u[don_i] = u[rho_i];
		u[acc_i] = ZERO;
	}
	return u;
}
