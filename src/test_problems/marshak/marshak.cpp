//  Copyright (c) 2019 AUTHORS
//                2010 James R. Craig
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef TESTME
#include "octotiger/defs.hpp"
#include "octotiger/grid.hpp"
#endif

#include <cmath>
#include <complex>
#include <fstream>
#include <functional>
#include <iostream>
#include <unordered_map>

#include <functional>

constexpr int MAX_DEPTH = 30;
constexpr int MIN_DEPTH = 10;

static constexpr double eps = 16.0;
static constexpr double kappa = MARSHAK_OPAC;
static constexpr double c = 1.0;
static constexpr double toler = 1.0e-10;


double integrate(const std::function<double(double)>& f, double a, double c, double toler = 1.0e-4, int depth = 0) {
	const double b = 0.5 * (a + c);
	const double fa = f(a);
	const double fb = f(b);
	const double fc = f(c);
	const double I1 = fb;
	double I2 = 0.5 * (fa + fc);
	if ((std::abs(I2 - I1) < toler && depth >= MIN_DEPTH) || depth == MAX_DEPTH) {
		return (2.0 * I1 + I2) / 3.0 * (c - a);
	}
	return integrate(f, a, b, toler, depth + 1) + integrate(f, b, c, toler, depth + 1);
}

double marshak_u(double x, double t, double eps) {
	const auto integrand = [x,t,eps](double eta) {
		if( eta == 0.0 ) {
			return 0.0;
		} else if( eta == 1.0 ) {
			return 0.0;
		} else {
			const double g2_1 = eta * eta * (eps + 1.0 / (1.0 - eta * eta));
			const double g2_2 = (1.0 - eta) * (eps + 1.0 / eta);
			const double t1 = std::acos(std::sqrt(3.0 / (3.0 + 4.0 * g2_1)));
			const double t2 = std::acos(std::sqrt(3.0 / (3.0 + 4.0 * g2_2)));
			const double num1 = std::sin(x * std::sqrt(g2_1) + t1);
			const double den1 = eta * std::sqrt(3.0 + 4.0 * g2_1);
			const double num2 = std::sin(x * std::sqrt(g2_2) + t2);
			const double den2 = eta * std::sqrt(3.0 + 4.0 * g2_2)
			* (1.0 + eps * eta);
			const double exp1 = std::exp(-t * eta * eta);
			const double exp2 = std::exp(-t * (1.0 + 1.0 / (eta * eps)));
			double I1 = exp1 * num1 / den1 * std::sqrt(3.0) / M_PI * 2.0;
			double I2 = exp2 * num2 / den2 * std::sqrt(3.0) / M_PI;
			return 1.0 - I1 - I2;
		}
	};
	return std::max(0.0,integrate(integrand, 0.0, 1.0));
}

double marshak_du(double x, double t, double eps) {
	const auto integrand = [x,t,eps](double eta) {
		if( eta == 0.0 ) {
			return 0.0;
		} else if( eta == 1.0 ) {
			return 0.0;
		} else {
			const double g2_1 = eta * eta * (eps + 1.0 / (1.0 - eta * eta));
			const double g2_2 = (1.0 - eta) * (eps + 1.0 / eta);
			const double t1 = std::acos(std::sqrt(3.0 / (3.0 + 4.0 * g2_1)));
			const double t2 = std::acos(std::sqrt(3.0 / (3.0 + 4.0 * g2_2)));
			const double num1 = std::cos(x * std::sqrt(g2_1) + t1)* std::sqrt(g2_1);
			const double den1 = eta * std::sqrt(3.0 + 4.0 * g2_1);
			const double num2 = std::cos(x * std::sqrt(g2_2) + t2) * std::sqrt(g2_2);
			const double den2 = eta * std::sqrt(3.0 + 4.0 * g2_2)
			* (1.0 + eps * eta);
			const double exp1 = std::exp(-t * eta * eta);
			const double exp2 = std::exp(-t * (1.0 + 1.0 / (eta * eps)));
			double I1 = exp1 * num1 / den1 * std::sqrt(3.0) / M_PI * 2.0;
			double I2 = exp2 * num2 / den2 * std::sqrt(3.0) / M_PI;
			return -(I1+I2);
		}
	};
	return integrate(integrand, 0.0, 1.0);
}

double marshak_v(double x, double t, double eps) {
	const auto integrand = [x,t,eps](double eta) {
		if( eta == 0.0 ) {
			return 0.0;
		} else if( eta == 1.0 ) {
			return 0.0;
		} else {
			const double g2_1 = eta * eta * (eps + 1.0 / (1.0 - eta * eta));
			const double g2_2 = (1.0 - eta) * (eps + 1.0 / eta);
			const double t1 = std::acos(std::sqrt(3.0 / (3.0 + 4.0 * g2_1)));
			const double t2 = std::acos(std::sqrt(3.0 / (3.0 + 4.0 * g2_2)));
			const double num1 = std::sin(x * std::sqrt(g2_1) + t1);
			const double den1 = eta * std::sqrt(3.0 + 4.0 * g2_1);
			const double num2 = std::sin(x * std::sqrt(g2_2) + t2);
			const double den2 = eta * std::sqrt(3.0 + 4.0 * g2_2)
			* (1.0 + eps * eta);
			const double exp1 = std::exp(-t * eta * eta);
			const double exp2 = std::exp(-t * (1.0 + 1.0 / (eta * eps)));
			double I1 = exp1 * num1 / den1 * std::sqrt(3.0) / M_PI * 2.0;
			double I2 = exp2 * num2 / den2 * std::sqrt(3.0) / M_PI;
			I1 *= (1.0 + g2_1 - eps * eta * eta);
			I2 *= (1.0 + g2_2 - eps * (1.0 + 1.0 / (eta * eps)));
			return 1.0 - I1 - I2;
		}
	};
	return std::max(0.0,integrate(integrand, 0.0, 1.0));
}

struct solution {
	double u, v, du;
};

struct pair_hash {
	std::size_t operator()( const std::pair<double,double>& v ) const {
		return std::hash<double>()(v.first) ^std::hash<double>()(v.second);
	}
};

static std::unordered_map<std::pair<double,double>, solution, pair_hash > sol_dir;
static hpx::lcos::local::mutex mtx_;

// NOTE: Why are y0 and z0 are unused?
std::vector<double> marshak_wave_analytic(double x0, double y0, double z0, double t) {
	std::vector<double> U(opts().n_fields + NRF, 0.0);
	double z = x0 + opts().xscale;
	z *= std::sqrt(3)*kappa;
	t *= eps * kappa * c;
	double u;
	double v;
	double du;
	{
		std::lock_guard < hpx::lcos::local::mutex > lock(mtx_);
		auto iter = sol_dir.find(std::make_pair(z, t));
		if (iter != sol_dir.end()) {
			u = iter->second.u;
			v = iter->second.v;
			du = iter->second.du;
		} else {
			solution s;
			s.du = du = marshak_du(z, t, eps);
			s.u = u = marshak_u(z, t, eps);
			s.v = v = marshak_v(z, t, eps);
			sol_dir.insert(std::make_pair(std::make_pair(z, t), s));
		}
	}

	double rho = 1.0;
	double erad = 4.0 * u / c;
	double e = 4.0 * v / c;
	double fx = -4.0 / std::sqrt(3) * du * c;
	U[rho_i] = U[spc_i] = rho;
	const double fy = 0;
	const double fz = 0;
	U[egas_i] = e;
	U[ein_i] = e;
	assert(!std::isnan(erad));
	assert(!std::isnan(fx));
	assert(!std::isnan(fy));
	assert(!std::isnan(fz));
	U[opts().n_fields + rad_grid::er_i] = erad;
	U[opts().n_fields + rad_grid::fx_i] = fx;
	U[opts().n_fields + rad_grid::fy_i] = fy;
	U[opts().n_fields + rad_grid::fz_i] = fz;
	return std::move(U);
}


// NOTE: Why are x, y, and z unused?
std::vector<double> marshak_wave(double x, double y, double z, double dx) {
	std::vector<double> u(opts().n_fields);
	double e = 1.0e-20;
	u[rho_i] = u[spc_i] = 1.0;
	u[egas_i] = e;
	u[ein_i] = e;
	return u;

}
