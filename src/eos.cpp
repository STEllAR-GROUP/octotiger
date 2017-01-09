/*
 * struct_eos.cpp
 *
 *  Created on: Feb 22, 2016
 *      Author: dmarce1
 */

#include "eos.hpp"
#include "util.hpp"
#include "grid.hpp"
#include "options.hpp"

const real wdcons = (2.216281751e32 / 1.989e+33);

extern options opts;

real struct_eos::energy(real d) const {
	return std::max(d * density_to_enthalpy(d) - pressure(d), 0.0);
}

void struct_eos::conversion_factors(real& m, real& l, real& t) const {
	real b = B();
	real a = A;
	real g = G;
	real g_fact = pow(a / g, 1.5) / (b * b);
	real cm_fact = pow(a / g, 0.5) / (b);
	real s_fact = 1.0 / sqrt(b * g);
	m = 2.21628175088037E+032 / g_fact;
	l = 483400430.180755 / cm_fact;
	t = 2.7637632869 / s_fact;
}

real struct_eos::enthalpy_to_density(real h) const {
	if (opts.eos == WD) {
		const real c = 8.0 * A / B();
		const real x = sqrt(pow((h / c) + 1.0, 2) - 1.0);
		return B() * x * x * x;
	} else {
		real res;
		ASSERT_NONAN(dC());
		ASSERT_NONAN(dE());
		ASSERT_NONAN(HE());
		ASSERT_NONAN(HC());
		if (h < HE()) {
			res = dE() * std::pow(h / HE(), n_E);
		} else {
			ASSERT_POSITIVE(h - HE() + HC());
			res = dC() * std::pow((h - HE() + HC()) / HC(), n_C);
		}
		ASSERT_NONAN(res);
		return res;
	}
}

real struct_eos::dE() const {
	return f_E * d0();
}
real struct_eos::s0() const {
	const real fgamma = grid::get_fgamma();
	return std::pow(P0() / (fgamma - 1.0), 1.0 / fgamma) / dE();
}
real struct_eos::P0() const {
	real den;
	if (d0() > dC()) {
		den = ((1.0 + n_C) / dC()) * std::pow(d0() / dC(), 1.0 / n_C) - (1.0 + n_C) / dC() + (1.0 + n_E) / dE();
	} else {
		den = (1.0 + n_E) / dE();
	}
	return h0() / den;
}
void struct_eos::set_frac(real f) {
	real mu = f_E / f_C;
	f_C = f;
	f_E = mu * f_C;
}
real struct_eos::get_frac() const {
	return f_C;
}
real struct_eos::HC() const {
	return P0() / dC() * (1.0 + n_C);
}
real struct_eos::HE() const {
	return P0() / dE() * (1.0 + n_E);
}
real struct_eos::density_to_enthalpy(real d) const {
	if (opts.eos == WD) {
		const real x = pow(d / B(), 1.0 / 3.0);
		if (x >= 0.01) {
			return (8.0 * A / B()) * (sqrt(x * x + 1.0) - 1.0);
		} else {
			return (4.0 * A / B()) * x * x;
		}
	} else {
		if (d >= dC()) {
			return P0() * (1.0 / dC() * (1.0 + n_C) * (std::pow(d / dC(), 1.0 / n_C) - 1.0) + 1.0 / dE() * (1.0 + n_E));
		} else if (d <= dE()) {
			return P0() / dE() * (1.0 + n_E) * std::pow(d / dE(), 1.0 / n_E);
		} else {
			return P0() / dE() * (1.0 + n_E);
		}
	}
}

real struct_eos::pressure(real d) const {
	if (opts.eos == WD) {
		const real x = pow(d / B(), 1.0 / 3.0);
		if (x < 0.01) {
			return 1.6 * A * pow(x, 5);
		} else {
			return A * (x * (2.0 * x * x - 3.0) * sqrt(x * x + 1.0) + 3.0 * asinh(x));
		}

	} else {
		if (d >= dC()) {
			return P0() * std::pow(d / dC(), 1.0 + 1.0 / n_C);
		} else if (d <= dE() && d > 0.0) {
			ASSERT_POSITIVE(d);
			ASSERT_POSITIVE(dE());
			//	printf( "n_E %e %e\n", n_E, P0());
			return P0() * std::pow(d / dE(), 1.0 + 1.0 / n_E);
		} else if (d > 0.0) {
			return P0();
		} else {
			return 0.0;
		}
	}
}

real struct_eos::dC() const {
	return f_C * d0();
}

void struct_eos::set_d0_using_struct_eos(real newd, const struct_eos& other) {
	if (opts.eos == WD) {
		d0_ = newd;
		A = other.A;
	} else {
		std::function<double(double)> fff = [&](real h) {
			set_d0(newd);
			set_h0(h);
			return s0() - other.s0();
		};
		real new_h;
		real min_h = 1.0e-10;
		real max_h = h0() * 100.0;
		if (!find_root(fff, min_h, max_h, new_h)) {
			printf("Error in struct_eos line %i\n", __LINE__);
			abort();
		}
	}
}

struct_eos::struct_eos(real M, real R) {
//B = 1.0;
	real m, r;
	d0_ = M / (R * R * R);
	A = M / R;
	while (true) {
		initialize(m, r);
		//	printf("%e %e  %e  %e %e  %e \n", d0, A, m, M, r, R);
		const real m0 = M / m;
		const real r0 = R / r;
		d0_ *= m0 / (r0 * r0 * r0);
		A /= m0 / r0;
		if (std::abs(1.0 - M / m) < 1.0e-10) {
			break;
		}
	}
}

struct_eos::struct_eos(real M, const struct_eos& other) {
	d0_ = other.d0_;
//B = 1.0;
//	printf("!!!!!!!!!!!!!!!!!!!\n");
	*this = other;
	std::function<double(double)> fff = [&](real newd) {
		real m, r;
		set_d0_using_struct_eos(newd, other);
		initialize(m, r);
//		printf("%e %e %e %e %e\n", M, m, d0, newd, other.d0);
			return M - m;
		};
//	printf("!!!!!!!!!!!!!!!!!!!\n");
	real new_d0;
	find_root(fff, 1.0e-20 * other.d0_, 1.0e+20 * other.d0_, new_d0);
}

struct_eos::struct_eos(real M, real _n_C, const struct_eos& other) {
	*this = other;
	n_C = _n_C;
	M0 = M;
	std::function<double(double)> fff = [&](real radius) {
		real m, r;
		initialize(m, r);
		M0 *= M / m;
		R0 = radius;
		return s0() - other.s0();
	};
	real new_radius;
	find_root(fff, 1.0e-10, opts.xscale / 2.0, new_radius);
}

struct_eos::struct_eos(real M, real R, real _n_C, real _n_E, real mu, const struct_eos& other) :
		n_C(_n_C), n_E(_n_E), M0(1.0), R0(1.0) {
	std::function<double(double)> fff = [&](real frac) {
		f_C = frac;
		f_E = frac / mu;
		real m, r;
		initialize(m, r);
		M0 *= M / m;
		R0 *= R / r;
		//	printf( "%e %e %e\n", s0(), s0() - other.s0(), frac);
			return s0() - other.s0();
		};
	real new_frac;
	find_root(fff, 1.0e-10, 1.0 - 1.0e-10, new_frac);
	set_frac(new_frac);
}

void struct_eos::set_entropy(real other_s0) {
	std::function<double(double)> fff = [&](real frac) {
		set_frac(frac);
		return s0() - other_s0;
	};
	real new_frac;
	find_root(fff, 0.0, 1.0, new_frac);
	set_frac(new_frac);
}

real struct_eos::dhdot_dr(real h, real hdot, real r) const {
	real a, b, d;
	d = this->enthalpy_to_density(h);
	if (r != 0.0) {
		a = 2.0 * hdot / r;
	} else {
		a = 0.0;
	}
	b = 4.0 * M_PI * G * d;
	return -(a + b);
}

real struct_eos::dh_dr(real h, real hdot, real r) const {
	return hdot;
}

struct_eos::struct_eos(real M, real R, real _n_C, real _n_E, real core_frac, real mu) :
		M0(1.0), R0(1.0), n_C(_n_C), n_E(_n_E) {
	real m, r, cm;
	real interface_core_density;
	const auto func = [&](real icd) {
		f_C = icd;
		f_E = icd / mu;
		initialize(m, r, cm);
//		printf( "%e %e %e\n", icd, cm/m, core_frac);
			return cm - core_frac*m;
		};
	auto _func = std::function < real(real) > (func);
	if (!find_root(_func, 0.0, 1.0, interface_core_density, 1.0e-3)) {
		printf("UNable to produce core_Frac\n");
		abort();
	}
//	printf( "--------------- %e\n", s0());
	f_C = interface_core_density;
	f_E = interface_core_density / mu;
	M0 *= M / m;
	R0 *= R / r;
//	printf( "--------------- %e\n", s0());
}

void struct_eos::initialize(real& mass, real& radius) {
	if (opts.eos == WD) {

		const real dr0 = (1.0 / B()) * sqrt(A / G) / 100.0;

		real h, hdot, r, m;
		h = density_to_enthalpy(d0_);
		hdot = 0.0;
		r = 0.0;
		m = 0.0;
		real dr = dr0;
		real d;
		integer i = 0;
		do {
			if (hdot != 0.0) {
				dr = std::max(std::min(dr0, std::abs(h / hdot) / 2.0), dr0 * 1.0e-6);
			}
			d = this->enthalpy_to_density(h);
			//	printf("%e %e %e\n", r, d, h);
			//	printf("%e %e %e %e %e\n", r, m, h, d, dr);
			const real dh1 = dh_dr(h, hdot, r) * dr;
			const real dhdot1 = dhdot_dr(h, hdot, r) * dr;
			const real dm1 = 4.0 * M_PI * d * sqr(r) * dr;
			if (h + dh1 <= ZERO) {
				break;
			}
			d = this->enthalpy_to_density(h + dh1);
			const real dh2 = dh_dr(h + dh1, hdot + dhdot1, r + dr) * dr;
			const real dhdot2 = dhdot_dr(h + dh1, hdot + dhdot1, r + dr) * dr;
			const real dm2 = 4.0 * M_PI * d * sqr(r + dr) * dr;
			h += (dh1 + dh2) / 2.0;
			hdot += (dhdot1 + dhdot2) / 2.0;
			r += dr;
			m += (dm1 + dm2) / 2.0;
			++i;
		} while (h > 0.0);
		mass = m;
		radius = r;
	} else {
		const real dr0 = R0 / 10000.0;

		real h, hdot, r, m;
		h = h0();
		hdot = 0.0;
		r = 0.0;
		m = 0.0;
		real dr = dr0;
		real d;
		integer i = 0;
		do {
			if (hdot != 0.0) {
				dr = std::max(std::min(dr0, std::abs(h / hdot) / 2.0), dr0 * 1.0e-6);
			}
			d = this->enthalpy_to_density(h);
			//	printf("%e %e %e\n", r, d, h);
			//	printf("%e %e %e %e %e\n", r, m, h, d, dr);
			const real dh1 = dh_dr(h, hdot, r) * dr;
			const real dhdot1 = dhdot_dr(h, hdot, r) * dr;
			const real dm1 = 4.0 * M_PI * d * sqr(r) * dr;
			if (h + dh1 <= ZERO) {
				break;
			}
			d = this->enthalpy_to_density(h + dh1);
			const real dh2 = dh_dr(h + dh1, hdot + dhdot1, r + dr) * dr;
			const real dhdot2 = dhdot_dr(h + dh1, hdot + dhdot1, r + dr) * dr;
			const real dm2 = 4.0 * M_PI * d * sqr(r + dr) * dr;
			h += (dh1 + dh2) / 2.0;
			hdot += (dhdot1 + dhdot2) / 2.0;
			r += dr;
			m += (dm1 + dm2) / 2.0;
			++i;
		} while (h > 0.0);
		mass = m;
		radius = r;
	}
}

void struct_eos::initialize(real& mass, real& radius, real& core_mass) {

	const real dr0 = R0 / 2000.0;
	core_mass = 0.0;
	real h, hdot, r, m;
	h = h0();
	hdot = 0.0;
	r = 0.0;
	m = 0.0;
	real dr = dr0;
	real d;
	do {
		if (hdot != 0.0) {
			dr = std::max(std::min(dr0, std::abs(h / hdot) / 2.0), dr0 * 1.0e-6);
		}
		d = this->enthalpy_to_density(h);
//		printf("%e %e %e %e %e\n", r, m, h, d, dr);
		const real dh1 = dh_dr(h, hdot, r) * dr;
		const real dhdot1 = dhdot_dr(h, hdot, r) * dr;
		const real dm1 = 4.0 * M_PI * d * sqr(r) * dr;
		if (d >= dC()) {
			core_mass += dm1 / 2.0;
		}
		if (h + dh1 <= ZERO) {
			break;
		}
		d = this->enthalpy_to_density(h + dh1);
		const real dh2 = dh_dr(h + dh1, hdot + dhdot1, r + dr) * dr;
		const real dhdot2 = dhdot_dr(h + dh1, hdot + dhdot1, r + dr) * dr;
		const real dm2 = 4.0 * M_PI * d * sqr(r + dr) * dr;
		if (enthalpy_to_density(h + dh1) >= dC()) {
			core_mass += dm1 / 2.0;
		}
		h += (dh1 + dh2) / 2.0;
		hdot += (dhdot1 + dhdot2) / 2.0;
		r += dr;
		m += (dm1 + dm2) / 2.0;
	} while (h > 0.0);
	mass = m;
	radius = r;
}

real struct_eos::d0() const {
	return M0 / (R0 * R0 * R0);
}

real struct_eos::h0() const {
	if (opts.eos == WD) {
		return density_to_enthalpy(d0_);
	} else {
		return G * M0 / R0;
	}
}

void struct_eos::set_h0(real h) {
	if (opts.eos == WD) {
		std::function<double(double)> fff = [&](real a) {
			A = a;
			//	printf( "%e %e %e\n", h0(), h, A);
				return h0() - h;
			};
		real new_a;
		if (!find_root(fff, A * 1.0e-6, A * 1.0e+6, new_a)) {
			printf("Error in struct_eos line %i\n", __LINE__);
			abort();
		}
	} else {
		const real d = d0();
		R0 = std::sqrt(h / (G * d));
		M0 = h * R0 / G;
	}
}

void struct_eos::set_d0(real d) {
	if (opts.eos == WD) {
		d0_ = d;
	} else {
		R0 = std::sqrt(h0() / d / G);
		M0 = R0 * R0 * R0 * d;
	}
}

real struct_eos::B() const {
	return sqrt(pow(A / G, 1.5) / wdcons);
}

real struct_eos::get_R0() const {
	if( opts.eos == WD ) {
		real m, r;
		struct_eos tmp = *this;
		tmp.initialize(m, r);
		return r;
	} else {
		real m, r;
		struct_eos tmp = *this;
		tmp.initialize(m, r);
		return r;
	}
}


real struct_eos::density_at(real R, real dr) const {

	real h, hdot, r;
	h = h0();
	hdot = 0.0;
	const int N = std::max(int(R / dr + 1.0), 16);
	dr = R / real(N);
	for (integer i = 0; i < N; ++i) {
		//	printf("%e %e %e\n", r, h, dr);
		r = i * dr;
		const real dh1 = dh_dr(h, hdot, r) * dr;
		const real dhdot1 = dhdot_dr(h, hdot, r) * dr;
		if (h + dh1 <= ZERO) {
			//	printf( "\n");
			return 0.0;
		}
		const real dh2 = dh_dr(h + dh1, hdot + dhdot1, r + dr) * dr;
		const real dhdot2 = dhdot_dr(h + dh1, hdot + dhdot1, r + dr) * dr;
		h += (dh1 + dh2) / 2.0;
		hdot += (dhdot1 + dhdot2) / 2.0;
		if (h <= ZERO) {
			//		printf( "\n");
			return 0.0;
		}
	}
	real d = this->enthalpy_to_density(h);
//	printf( "%d\n", d);
	return d;
}
