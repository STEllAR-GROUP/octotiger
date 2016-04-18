/*
 * problem.cpp
 *
 *  Created on: May 29, 2015
 *      Author: dmarce1
 */

#include "problem.hpp"
#include "lane_emden.hpp"
#include <cmath>
#include "defs.hpp"

init_func_type problem = nullptr;
refine_test_type refine_test_function = refine_test;

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

bool refine_test_bibi(integer level, integer max_level, real x, real y, real z, std::vector<real> U) {
	bool rc = false;
	real den_floor = 1.0e-4;
	integer test_level = (U[frac0_i] > U[frac1_i] ? max_level : max_level - 1);
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

real interp_scf_pre(real x, real y, real z);
real interp_scf_rho(real x, real y, real z);

std::vector<real> old_scf(real x, real y, real z, real omega, real core1, real core2, real) {
	std::vector<real> u(NF, real(0));
	const real rho_floor = 1.0e-11;
	const real rho = std::max(interp_scf_rho(x, y, z), rho_floor);
	const real pre = std::max(interp_scf_pre(x, y, z), 1.0e-13);
	u[rho_i] = rho;
	if (((x > 0.0) && (rho >= core2)) || ((x < 0.0) && (rho >= core1))) {
		u[frac0_i] = rho - rho_floor / TWO;
		u[frac1_i] = rho_floor / TWO;
	} else {
		u[frac1_i] = rho - rho_floor / TWO;
		u[frac0_i] = rho_floor / TWO;
	}
	u[sx_i] = -y * omega * rho;
	u[sy_i] = +x * omega * rho;
	u[sz_i] = 0.0;
	u[egas_i] = pre / (fgamma - 1.0);
	u[tau_i] = std::pow(u[egas_i], ONE / fgamma);
	u[egas_i] += 0.5 * omega * omega * (x * x + y * y) * rho;
	return u;
}

std::vector<real> null_problem(real x, real y, real z, real dx) {
	std::vector<real> u(NF, real(0));
	return u;
}

std::vector<real> sod_shock_tube(real x, real y, real z, real) {
	std::vector<real> u(NF, real(0));
	if (x + y + z > real(0)) {
		u[rho_i] = 1.0;
		u[egas_i] = 2.5;
	} else {
		u[rho_i] = 0.125;
		u[egas_i] = 0.1;
	}
	u[tau_i] = std::pow(u[egas_i], ONE / fgamma);
	return u;
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
//	y0 -= 0.5;
//	z -= 0.2565;
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
	x0 -= -0.0444;
	y0 -= +0.345;
	z0 -= -.2565;
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
const real alpha = real(1) / real(5);

#include "bibi.hpp"

std::vector<real> star(real x, real y, real z, real) {

	static bibi_polytrope bibi(5.0e-3/0.9251, 0.25, 3.0, 1.5, 0.75, 0.5);
	x -= 0.125;
	y -= 0.0;
	z -= 0.0;
	real menc;
	const real r = std::sqrt(x * x + y * y + z * z);
	std::vector<real> u(NF, real(0));
	real rho, pre;
	const bool in_core = bibi.solve_at(r, rho, pre, menc);
	u[rho_i] = std::max(rho, 1.0e-10);
	u[frac0_i] = u[rho_i] * (in_core ? 1.0 : 0.0);
	u[frac1_i] = u[rho_i] * (in_core ? 0.0 : 1.0);
	u[egas_i] = std::max(pre, 1.0e-10) / (fgamma - 1.0);
	u[tau_i] = std::pow(u[egas_i], (real(1) / real(fgamma)));
	if (r < 0.25) {
//		printf("%e %e\n", rho, r);
	}
	/*	real theta;
	 const real n = real(1) / (fgamma - real(1));
	 const real rho_min = 1.0e-3;
	 const real theta_min = std::pow(rho_min, real(1) / n);
	 const auto c0 = real(4) * real(M_PI) * alpha * alpha / (n + real(1));
	 if (r <= rmax) {
	 theta = lane_emden(r, dr);
	 theta = std::max(theta, theta_min);
	 } else {
	 theta = theta_min;
	 }
	 u[rho_i] = std::pow(theta, n);
	 u[egas_i] = std::pow(theta, fgamma * n) * c0 / (fgamma - real(1));
	 if (theta <= theta_min) {
	 u[egas_i] *= real(100);
	 }
	 u[tau_i] = std::pow(u[egas_i], (real(1) / real(fgamma)));
	 u[sx_i] = -DEFAULT_OMEGA * y * u[rho_i];
	 u[sy_i] = +DEFAULT_OMEGA * x * u[rho_i];*/
	return u;
}

std::vector<real> equal_mass_binary(real x, real y, real z, real) {
	const integer don_i = frac1_i;
	const integer acc_i = frac0_i;

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
