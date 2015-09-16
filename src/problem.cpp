/*
 * problem.cpp
 *
 *  Created on: May 29, 2015
 *      Author: dmarce1
 */

#include "problem.hpp"
#include "lane_emden.hpp"
#include <cmath>

std::vector<real> sod_shock_tube(real x, real y, real z) {
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

std::vector<real> solid_sphere(real x, real y, real z) {
	const real r0 = 0.5;
	std::vector<real> u(NF, real(0));
//	x -= 0.5;
//	y -= 0.5;
//	z -= 0.5;
	if (x*x + y*y + z*z < r0*r0) {
		u[rho_i] = 1.0;
	} else {
		u[rho_i] = 1.0e-10;
	}
	return u;
}

const real x0 = 0.0;
const real y0_ = 0.0;
const real z0 = 0.0;
const real rmax = 3.7;
const real dr = rmax / 128.0;
const real alpha = real(1) / real(5);

std::vector<real> star(real x, real y, real z) {
	x -= x0;
	y -= y0_;
	z -= z0;
	real theta;
	const real n = real(1) / (fgamma - real(1));
	const real rho_min = 1.0e-3;
	std::vector<real> u(NF, real(0));
	const real r = std::sqrt(x * x + y * y + z * z) / alpha;
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
	u[sy_i] = +DEFAULT_OMEGA * x * u[rho_i];
	return u;
}

std::vector<real> equal_mass_binary( real x, real y, real z) {
	real theta;
	real alpha = 1.0 / 15.0;
	const real n = real(1) / (fgamma - real(1));
	const real rho_min = 1.0e-12;
	std::vector<real> u(NF, real(0));
	const real d = 1.0/2.0;
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
		real r = std::min(r1,r2);
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
	u[egas_i] += HALF*DEFAULT_OMEGA*DEFAULT_OMEGA*(x*x+y*y)*u[rho_i];
	return u;
}
