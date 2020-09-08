//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/eos.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/real.hpp"
#include "octotiger/util.hpp"

#include <cmath>
#include <cstdio>
#include <functional>

const real wdcons = (2.216281751e32 / 1.989e+33);

constexpr real struct_eos::G;

#define N53

void struct_eos::conversion_factors(real &m, real &l, real &t) const {
	real b = B();
	real a = A;
	real g = G;
	real g_fact = pow(a * INVERSE(g), 1.5) * INVERSE(b * b);
	real cm_fact = pow(a * INVERSE(g), 0.5) * INVERSE(b);
	real s_fact = 1.0 * INVERSE(SQRT(b * g));
	m = 2.21628175088037E+032 * INVERSE(g_fact);
	l = 483400430.180755 * INVERSE(cm_fact);
	t = 2.7637632869 * INVERSE(s_fact);
}

real struct_eos::energy(real d) const {
	const auto b = physcon().B;
	const auto edeg = ztwd_energy(d);
	const auto pdeg = ztwd_pressure(d);
	const auto x = std::pow(d / b, 1.0 / 3.0);
#ifndef N53
	const auto K = wd_eps * ztwd_pressure(d0()) / std::pow(d0(), 4.0 / 3.0);
	const auto egas = 1.5 * K * std::pow(d, 4.0 / 3.0);
#else
	const auto K = wd_eps * ztwd_pressure(d0()) / std::pow(d0(), 5.0 / 3.0);
	const auto egas = 1.5 * K * std::pow(d, 5.0 / 3.0);
#endif
	return edeg + egas;
//	return std::max(d * density_to_enthalpy(d) - pressure(d), 0.0);
}

void struct_eos::set_wd_T0(double t, double abar, double zbar) {
	wd_T0 = t;
	wd_eps = physcon().kb * d0() * t * (zbar + 1) / abar / physcon().mh / ztwd_pressure(d0(), A, B());

}

real struct_eos::enthalpy_to_density(real h) const {
	if (opts().eos == WD) {
#ifndef N53
		const real b = B();
		const auto K = wd_eps * ztwd_pressure(d0()) / std::pow(d0(), 4.0 / 3.0);
		const auto h0 = 8.0 * A / b;
		const auto a0 = h0 * h0 - 16 * K * K * std::pow(b, 2.0 / 3.0);
		const auto b0 = 4.0 * (h + h0) * K * std::pow(b, 1.0 / 3.0);
		const auto c0 = -(h * h + 2.0 * h * h0);
		const auto x = (-b0 + sqrt(b0 * b0 - 4.0 * a0 * c0)) / (2.0 * a0);
#else
		const real b = B();
		const auto K = wd_eps * ztwd_pressure(d0(), A, b) / std::pow(d0(), 5.0 / 3.0);
		double x2;
		const auto h0 = 8.0 * A / b;
		if (K != 0.0) {
			const auto a0 = 25. / 4. * K * K * pow(b, 4. / 3.);
			const auto b0 = -((h + h0) * 2.5 * K * pow(b, 2. / 3.) + h0 * h0);
			const auto c0 = h * h + 2 * h0 * h;
			x2 = (-b0 - std::sqrt(b0 * b0 - 4.0 * a0 * c0)) / (2.0 * a0);
//			if( c0 != 0.0 ) {
//				printf( "%e %e %e %e\n", x2, b0 * b0 ,4.0 * a0 * c0, a0);
//			}
		} else {
			x2 = std::pow((h + h0) / h0, 2) - 1;
		}
		const auto x = sqrt(x2);
#endif
		const real rho = b * x * x * x;
		return rho;
	} else {
		real res;
		ASSERT_NONAN(dC());
		ASSERT_NONAN(dE());
		ASSERT_NONAN(HE());
		ASSERT_NONAN(HC());
		const real _h0 = density_to_enthalpy(rho_cut);
		//	const real _h1 = h0() * POWER(rho_cut / d0(), 1.0 / 1.5);
		//	h += hfloor();
		if (h > _h0 || !opts().v1309) {
			if (h < HE()) {
				res = dE() * POWER(h * INVERSE( HE() ), n_E);
			} else {
				ASSERT_POSITIVE(h - HE() + HC());
				res = dC() * POWER((h - HE() + HC()) * INVERSE( HC() ), n_C);
			}
		} else {
			h -= hfloor();
			res = POWER((1.0 + n_E) / 2.5 * (std::max(h, 0.0) * INVERSE(_h0)), 1.5) * rho_cut;
			//	return 0.0;
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
	return POWER(P0() / (fgamma - 1.0), 1.0 / fgamma) * INVERSE(dE());
}
real struct_eos::P0() const {
	real den;
	if (d0() > dC()) {
		den = ((1.0 + n_C) * INVERSE(dC())) * POWER(d0() * INVERSE( dC()), 1.0 * INVERSE( n_C)) - (1.0 + n_C) * INVERSE(dC()) + (1.0 + n_E) * INVERSE(dE());
	} else {
		den = (1.0 + n_E) * INVERSE(dE());
	}
	return h0() * INVERSE(den);
}

void struct_eos::set_frac(real f) {
	real mu = f_E * INVERSE(f_C);
	f_C = f;
	f_E = mu * f_C;
}
real struct_eos::get_frac() const {
	return f_C;
}
real struct_eos::HC() const {
	return P0() * INVERSE(dC()) * (1.0 + n_C);
}
real struct_eos::HE() const {
	return P0() * INVERSE(dE()) * (1.0 + n_E);
}

real struct_eos::density_to_enthalpy(real d) const {
	if (opts().eos == WD) {
		const real b = B();
#ifndef N53
		const auto K = wd_eps * ztwd_pressure(d0()) / std::pow(d0(), 4.0 / 3.0);
		const auto hideal = 4 * K * pow(d,1.0/3.0);
#else
		const auto K = wd_eps * ztwd_pressure(d0(), A, b) / std::pow(d0(), 5.0 / 3.0);
		const auto hideal = 2.5 * K * pow(d, 2.0 / 3.0);
#endif
		const real x = POWER(d * INVERSE(b), 1.0 / 3.0);
		real h;
		if (x > 0.01) {
			h = ((8.0 * A) * INVERSE(b)) * (SQRT(x * x + 1.0) - 1.0);
		} else {
			h = (8.0 * A * INVERSE(b)) * (0.5 * x * x);
		}
		return h + hideal;
	} else {
		if (d >= dC()) {
			return P0() * (1.0 * INVERSE(dC()) * (1.0 + n_C) * (POWER(d * INVERSE( dC()), 1.0 * INVERSE( n_C)) - 1.0) + 1.0 * INVERSE(dE()) * (1.0 + n_E));
		} else if (d <= dE()) {
			return P0() * INVERSE(dE()) * (1.0 + n_E) * POWER(d * INVERSE( dE()), 1.0 * INVERSE( n_E));
		} else {
			return P0() * INVERSE(dE()) * (1.0 + n_E);
		}
	}
}

real struct_eos::pressure(real d) const {
	if (opts().eos == WD) {
		const real b = B();
		const real x = pow(d * INVERSE(b), 1.0 / 3.0);
		real pd;
		if (x < 0.01) {
			pd = 1.6 * A * pow(x, 5);
		} else {
			pd = A * (x * (2.0 * x * x - 3.0) * sqrt(x * x + 1.0) + 3.0 * asinh(x));
		}
		return pd;

	} else {
		if (d >= dC()) {
			return P0() * POWER(d * INVERSE( dC() ), 1.0 + 1.0 * INVERSE( n_C));
		} else if (d <= dE() && d > 0.0) {
			ASSERT_POSITIVE(d);
			ASSERT_POSITIVE(dE());
			//	printf( "n_E %e %e\n", n_E, P0());
			if (opts().v1309) {
				if (d < rho_cut) {
					const real h0 = density_to_enthalpy(rho_cut);
					return rho_cut * h0 * INVERSE(1.0 + n_E) * POWER(d * INVERSE( rho_cut ), 5. / 3.);
				}
			}
			return P0() * POWER(d * INVERSE( dE() ), 1.0 + 1.0 * INVERSE( n_E));
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

void struct_eos::set_d0_using_struct_eos(real newd, const struct_eos &other) {
	if (opts().eos == WD) {
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

struct_eos::struct_eos(real M, real R) :
		rho_cut(0.0), wd_eps(0), wd_core_cut(0.5) {
//B = 1.0;
	real m, r;
	d0_ = M * INVERSE(R * R * R);
	A = M * INVERSE(R);
	while (true) {
		initialize(m, r);
		//	printf("%e %e  %e  %e %e  %e \n", d0, A, m, M, r, R);
		const real m0 = M * INVERSE(m);
		const real r0 = R * INVERSE(r);
		d0_ *= m0 * INVERSE(r0 * r0 * r0);
		A /= m0 * INVERSE(r0);
		physcon().A = A;
		physcon().B = B();
		normalize_constants();
		if (std::abs(1.0 - M * INVERSE(m)) < 1.0e-10) {
			break;
		}
	}
}

struct_eos::struct_eos(real M, const struct_eos &other) :
		rho_cut(0.0), wd_eps(0), wd_core_cut(0.5) {
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

struct_eos::struct_eos(real M, real _n_C, const struct_eos &other) :
		rho_cut(0.0) {
	*this = other;
	n_C = _n_C;
	M0 = M;
	std::function<double(double)> fff = [&](real radius) {
		real m, r;
		initialize(m, r);
		M0 *= M * INVERSE(m);
		R0 = radius;
		return s0() - other.s0();
	};
	real new_radius;
	find_root(fff, 1.0e-10, opts().xscale / 2.0, new_radius);
}

struct_eos::struct_eos(real M, real R, real _n_C, real _n_E, real mu, const struct_eos &other) :
		M0(1.0), R0(1.0), n_C(_n_C), n_E(_n_E), rho_cut(0.0) {
	std::function<double(double)> fff = [&](real frac) {
		f_C = frac;
		f_E = frac * INVERSE(mu);
		real m, r;
		initialize(m, r);
		M0 *= M * INVERSE(m);
		R0 *= R * INVERSE(r);
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
	real a;
	if (r != 0.0) {
		a = 2.0 * hdot * INVERSE(r);
	} else {
		a = 0.0;
	}
	real d = this->enthalpy_to_density(h);
	real b = 4.0 * M_PI * G * d;
	return -(a + b);
}

real struct_eos::dh_dr(real h, real hdot, real r) const {
	return hdot;
}

struct_eos::struct_eos(real M, real R, real _n_C, real _n_E, real core_frac, real mu) :
		M0(1.0), R0(1.0), n_C(_n_C), n_E(_n_E), rho_cut(0.0), wd_eps(0), wd_core_cut(0.5) {
	real m, r, cm;
	real interface_core_density;
	const auto func = [&](real icd) {
		f_C = icd;
		f_E = icd * INVERSE(mu);
		initialize(m, r, cm);
//		printf( "%e %e %e\n", icd, cm/m, core_frac);
		return cm - core_frac * m;
	};
	auto _func = std::function<real(real)>(func);
	if (!find_root(_func, 0.0, 1.0, interface_core_density, 1.0e-3)) {
		printf("UNable to produce core_Frac\n");
		abort();
	}
//	printf( "--------------- %e\n", s0());
	f_C = interface_core_density;
	f_E = interface_core_density * INVERSE(mu);
	M0 *= M * INVERSE(m);
	R0 *= R * INVERSE(r);
//	printf( "--------------- %e\n", s0());
}

void struct_eos::initialize(real &mass, real &radius) {
	if (opts().eos == WD) {

		const real dr0 = (1.0 * INVERSE(B())) * SQRT(A * INVERSE (G)) / 100.0;

		real h = density_to_enthalpy(d0_);
		real hdot = 0.0;
		real r = 0.0;
		real m = 0.0;
		real dr = dr0;
		integer i = 0;
		do {
			if (hdot != 0.0) {
				dr = std::max(std::min(dr0, std::abs(h * INVERSE(hdot)) / 2.0), dr0 * 1.0e-6);
			}
			real d = this->enthalpy_to_density(h);
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
		const real dr0 = R0 / 10.0;

		real h = h0();
		real hdot = 0.0;
		real r = 0.0;
		real m = 0.0;
		real dr = dr0;
		real d;
		integer i = 0;
		do {
			if (hdot != 0.0) {
				dr = std::max(std::min(dr0, std::abs(h * INVERSE(hdot)) / 2.0), dr0 * 1.0e-6);
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
	my_radius = radius;
//	printf( "Radius = %e\n", my_radius);
}

void struct_eos::initialize(real &mass, real &radius, real &core_mass) {

	const real dr0 = R0 / 100.0;
	core_mass = 0.0;
	real h = h0();
	real hdot = 0.0;
	real r = 0.0;
	real m = 0.0;
	real dr = dr0;
	real d;
	do {
		if (hdot != 0.0) {
			dr = std::max(std::min(dr0, std::abs(h * INVERSE(hdot)) / 2.0), dr0 * 1.0e-6);
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
	my_radius = radius;
	//printf( "Radius = %e\n", my_radius);
}

real struct_eos::d0() const {
	if (opts().eos == WD) {
		return d0_;
	} else {
		return M0 * INVERSE(R0 * R0 * R0);
	}
}

real struct_eos::h0() const {
	if (opts().eos == WD) {
		return density_to_enthalpy(d0_);
	} else {
		return G * M0 * INVERSE(R0);
	}
}

void struct_eos::set_h0(real h) {
	if (opts().eos == WD) {
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
		R0 = SQRT(h * INVERSE (G * d));
		M0 = h * R0 * INVERSE(G);
	}
}

void struct_eos::set_d0(real d) {
	if (opts().eos == WD) {
		d0_ = d;
	} else {
		R0 = SQRT(h0() * INVERSE( d * G ));
		M0 = R0 * R0 * R0 * d;
	}
}

real struct_eos::B() const {
	return SQRT(POWER(A * INVERSE( G ), 1.5) * INVERSE( wdcons));
}

real struct_eos::get_R0() const {
//	return my_radius;
	if (opts().eos == WD) {
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

real struct_eos::density_at(real R, real dr) {

	real r;
	real h = h0();
	real hdot = 0.0;
	const int N = std::max(int(R / dr + 1.0), 32);
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
