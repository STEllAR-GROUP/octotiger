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
//#include "octotiger/lane_emden.hpp"
#include "octotiger/mesa/mesa.hpp"

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

real struct_eos::enthalpy_to_density(real h) {
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
				if (filename == "") {
					res = dE() * POWER(h * INVERSE( HE() ), n_E);
				} else {
					if (h == 0.0) {
						res = 0.0;
					} else {
						if (enthalpy_to_density_prof == nullptr) {
							if (rho_vec.size() == 0) {
								real he, pe, re;
								real de = 1.001 * dE();
								auto pair_of_funcs = build_rho_of_h_from_mesa(filename, he, de, pe, re, P_vec, rho_vec, h_vec);
								enthalpy_to_density_prof = pair_of_funcs.first;
								density_to_pressure_prof = pair_of_funcs.second;
								printf("built from file, rho(h=0): %e\n", enthalpy_to_density_prof(0.0));
							} else {
                                	                        auto pair_of_funcs = build_rho_of_h_from_relation(P_vec, rho_vec, h_vec);
                                        	                enthalpy_to_density_prof = pair_of_funcs.first;
                                                	        density_to_pressure_prof = pair_of_funcs.second;
							}
						}
						res = enthalpy_to_density_prof(h);
					}
				}
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
        if (filename != "") {
	        return PE_prof;
	}
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
	if (filename != "") {
		return HE_prof;
	}
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
			if ((filename == "") || (d == 0.0)) {
				return P0() * INVERSE(dE()) * (1.0 + n_E) * POWER(d * INVERSE( dE()), 1.0 * INVERSE( n_E));
                        } else {
				printf("enthalpy at the envelope needs to be decided by the profile!! d: %e, dE: %e\n", d, dE());
				abort();
                        }
		} else {
			return P0() * INVERSE(dE()) * (1.0 + n_E);
		}
	}
}

real struct_eos::pressure(real d) {
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
			if (filename != "") {
				if (density_to_pressure_prof == nullptr) {
					if (rho_vec.size() == 0) {
                                		real he, pe, re;
	                                	real de = 1.001 * dE();
        	                                auto pair_of_funcs = build_rho_of_h_from_mesa(filename, he, de, pe, re, P_vec, rho_vec, h_vec);
                	                        enthalpy_to_density_prof = pair_of_funcs.first;
                        	                density_to_pressure_prof = pair_of_funcs.second;
					} else {
	                                        auto pair_of_funcs = build_rho_of_h_from_relation(P_vec, rho_vec, h_vec);
        	                                enthalpy_to_density_prof = pair_of_funcs.first;
                                                density_to_pressure_prof = pair_of_funcs.second;
					}
				}
				return std::max(0.0, density_to_pressure_prof(d));
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
		rho_cut(0.0), wd_eps(0), wd_core_cut(0.5), p_mass(0.0), p_smooth_l(0.0), my_radius(0.0) {
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
		rho_cut(0.0), wd_eps(0), wd_core_cut(0.5), p_mass(0.0), p_smooth_l(0.0), my_radius(0.0) {
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
		rho_cut(0.0), p_mass(0.0), p_smooth_l(0.0), my_radius(0.0) {
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
		M0(1.0), R0(1.0), n_C(_n_C), n_E(_n_E), rho_cut(0.0), p_mass(0.0), p_smooth_l(0.0), my_radius(0.0) {
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

real struct_eos::dhdot_dr(real h, real hdot, real r) {
	real a;
	if (r != 0.0) {
		a = 2.0 * hdot * INVERSE(r);
	} else {
		a = 0.0;
	}
	real d = this->enthalpy_to_density(h);
	real b = 4.0 * M_PI * G * d;
	real c;
        if ((p_mass > 0.0) && (p_smooth_l > 0.0)) {
		const auto khi_tilde = [](real x) {
			if ((0.0 <= x) && (x <= 1.0)) {
	        		return 4.0/3.0 - 2.0*x*x + x*x*x;
			} else if ((1.0 < x) && (x <= 2.0)) {
		        	return 8.0/3.0 - 4.0*x + 2.0*x*x - x*x*x/3.0;
			} else {
		        	return 0.0;
			}
		};
		real const d_av = p_mass * INVERSE(POWER(p_smooth_l, 3));
        	c = 3.0 * G * d_av * khi_tilde( r / p_smooth_l );
        } else {
		c = 0.0;
	}
	return -(a + b + c);
}

real struct_eos::dh_dr(real h, real hdot, real r) const {
	return hdot;
}

struct_eos::struct_eos(real M, real R, real _n_C, real _n_E, real core_frac, real mu) :
		M0(1.0), R0(1.0), n_C(_n_C), n_E(_n_E), rho_cut(0.0), wd_eps(0), wd_core_cut(0.5), p_mass(0.0), p_smooth_l(0.0), my_radius(0.0) {
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
	initialize(m, r, cm);
	//printf("r: %e, m: %e, cm %e!!!\n", r, m, cm);
	my_radius = r;
//	printf( "--------------- %e\n", s0());
}

struct_eos::struct_eos(real M, real R, real _n_C, real _n_E, real core_frac, real mu, real _p_mass) :
                M0(1.0), R0(1.0), n_C(_n_C), n_E(_n_E), rho_cut(0.0), wd_eps(0), wd_core_cut(0.5), my_radius(0.0) {
        real m, r, cm, cr;
        real interface_core_density, smooth_frac, rho_frac;
	real const p_smooth = 0.5 * core_frac * R;
	while (true) {  // the following is experimental and probably not working properly
		M0 = 1.0;
		R0 = 1.0;
	        const auto func = [&](real icd) {
			f_C = icd;
                	f_E = icd * INVERSE(mu);
	                initialize(m, r);	
                	return (r - cr) * INVERSE(r * (1 - core_frac)) - 1.0;
        	};
	        auto _func = std::function<real(real)>(func);
	        if (!find_root(_func, 0.0, 1.0, interface_core_density, 1.0e-6)) {
        	        printf("UNable to produce envelope radial frac\n");
                	abort();
	        }
        	f_C = interface_core_density;
	        f_E = interface_core_density * INVERSE(mu);
		printf("found f_C: %e\n", f_C);
        	const auto func2 = [&](real fl) {
                	p_smooth_l = fl;
	        	p_mass = rho_frac * POWER(p_smooth_l, 3);
                	initialize(m, r);
        	        return cr * INVERSE(2 * R0 * fl) - 1.0;
	        };
        	auto _func2 = std::function<real(real)>(func2);
	        if (!find_root(_func2, 1e-10,  1e10, smooth_frac, 1.0e-5)) {
        	        printf("UNable to produce radius R\n");
                	abort();
	        }
		printf("found p_smooth_l: %e\n", smooth_frac);
		p_smooth_l = smooth_frac;
		R0 = p_smooth_l;
                const auto func3 = [&](real frho) {
                        p_mass = frho * POWER(p_smooth_l, 3);
                        initialize(m, r);
                        return _p_mass * (m - cm) * INVERSE((M - _p_mass) * M0 * p_mass) - 1.0;
                };
                auto _func3 = std::function<real(real)>(func3);
                if (!find_root(_func3, 1e-12, 1e12, rho_frac, 1.0e-6)) {
                	printf("UNable to produce mass M\n");
                	abort();
                }
                printf("found rho_frac: %e, p_mass_f: %e\n", rho_frac, rho_frac * POWER(p_smooth_l, 3));
		p_mass = rho_frac * POWER(p_smooth_l, 3);
		initialize(m, r);
		printf("M0: %e, R0: %e, rho0: %e\n", M0, R0, M0/(R0 * R0 * R0));
		printf("R0: %e, Rcore: %e, M0: %e, Mcore: %e, d0: %e, rho_c: %e, h0: %e, hE/h0: %e\n", r, cr, m, cm, d0(), dC(), h0(), HE()/h0());
		printf("r: %e, R0: %e, Rcore: %e, m: %e, M0: %e, Mcore: %e, d0: %e, rho_c: %e, h0: %e, hE/h0: %e, dev: %e\n", r, R0, cr, m, M0, cm, d0(), dC(), h0(), HE()/h0(), cr * INVERSE(core_frac * R) - 1.0);
		printf("\n----------------\nthe convergence: %e\n", SQRT(POWER((r - cr) * INVERSE(r * (1 - core_frac)) - 1.0, 2))); 
		if (SQRT(POWER(r * INVERSE(R) - 1.0, 2)) < 0.03) {
			break;
		}
	}
}

struct_eos::struct_eos(real M, real core_frac, const std::string& _filename) :
                M0(1.0), R0(1.0), n_C(3.0), n_E(3.0), rho_cut(0.0), wd_eps(0), wd_core_cut(0.5) {
        real m, r, mc, rc, h_c, hdot_c, h0_found, p_mass_found;
	filename = _filename;
	mesa_profiles mesa_p(filename);
	real const R_st = mesa_p.get_R0();
	real r_c;
	real dE_prof, omega_prof, RE_prof;
	mesa_p.state_at(dE_prof, PE_prof, omega_prof, core_frac * R_st);
	printf("loaded from file the following core-enevelope values\n");
        printf("HE: %e, PE: %e, DE: %e, Rstar: %e, Rcore %e\n", HE_prof, PE_prof, dE_prof, R_st, core_frac * R_st);
	auto rho_of_h_relation = build_rho_of_h_from_mesa(filename, HE_prof, dE_prof, PE_prof, RE_prof, P_vec, rho_vec, h_vec); // gets the enthalpy pressure updated density and radius of the interface based on the given density
	p_smooth_l = 0.5 * RE_prof;
	enthalpy_to_density_prof = rho_of_h_relation.first;
	density_to_pressure_prof = rho_of_h_relation.second;
        set_d0(dE_prof); // d0 in this case is the boundary density so f_E = f_C = 1.0
        f_C = 1.0;
        f_E = 1.0;
	printf("updated interface values - HE: %e, PE: %e, DE: %e, Rstar: %e, Rcore %e, rE: %e, rho(h=0.0) = %e\n", HE_prof, PE_prof, dE_prof, R_st, core_frac * R_st, RE_prof, enthalpy_to_density_prof(0.0));
        const auto func = [&](real cur_h0) {  // finding h0 that statisfies the enthalpy boundary at the interface radii, h(RE) = HE 
		real const p_mass0 = p_mass;
		set_h0(cur_h0*HE_prof);
                initialize(m, r, mc, rc, h_c, hdot_c, RE_prof); // retrieves h at RE_prof into h_c
		return  h_c * INVERSE(HE_prof) - 1.0;
        };
        auto _func = std::function<real(real)>(func);
	const auto func_mass = [&](real cur_m_p) { // adjust the particle mass to get the correct total mass
		printf("setting particle mass to: %e\n", cur_m_p);
		p_mass = cur_m_p;
		real const rho_av = p_mass / (4.0 * M_PI / 3.0) / POWER(p_smooth_l, 3);
		printf("m_p: %e, p_smooth: %e, rho av: %e\n", p_mass, p_smooth_l, 3.0 * p_mass * INVERSE(4.0 * M_PI * POWER(p_smooth_l, 3)));
		if(rho_av < dE_prof) {
			printf("The particle 'average density' is smaller than the boundary density! cannot proceed\n");
			abort();
		}
		real const Hmax = density_to_enthalpy(rho_av);
        	if (!find_root(_func, 1.0, Hmax/HE_prof, h0_found, 1.0e-5)) {
                	printf("UNable to produce h0\n");
                	abort();
		}
		printf("found h0: %e, rho_max: %e\n", h0_found*HE_prof, enthalpy_to_density(h0_found*HE_prof));
		set_h0(h0_found*HE_prof);
		initialize(m, r, mc, rc, h_c, hdot_c, R_st); // getting the mass (including the particle) into m
		printf("r: %e, rc: %e, m: %e, mc: %e, d0: %e, rho_c: %e, h0: %e, hE/h0: %e, h_c: %e, hdot_c: %e, r_c: %e, cur_m_p: %e, m(dhdr): %e\n", r, rc, m, mc, d0(), dC(), h0(), HE()/h0(), h_c, hdot_c, RE_prof, cur_m_p, -hdot_c * RE_prof * RE_prof);
		return m * INVERSE(M) - 1.0;
        };
	auto _func_mass = std::function<real(real)>(func_mass);
        if (!find_root(_func_mass, 0.05*M, 0.6*M, p_mass_found, 1.0e-5)) {
        	printf("UNable to produce particle mass\n");
                abort();
        }
	p_mass = p_mass_found;
	real const rho_av = p_mass / (4.0 * M_PI / 3.0) / POWER(p_smooth_l, 3);
	real const Hmax = density_to_enthalpy(rho_av);
        if (!find_root(_func, 1.0, Hmax/HE_prof, h0_found, 1.0e-5)) {
                printf("UNable to produce h0\n");
                abort();
        }
        printf("found h0: %e, rho_max: %e\n", h0_found*HE_prof, enthalpy_to_density(h0_found*HE_prof));
        set_h0(h0_found*HE_prof);
        initialize(m, r, mc, rc, h_c, hdot_c, RE_prof);
        printf("r: %e, rc: %e, m: %e, mc: %e, d0: %e, rho_c: %e, h0: %e, hE/h0: %e, h_c: %e, hdot_c: %e, r_c: %e, m(dhdr): %e\n", r, rc, m, mc, d0(), dC(), h0(), HE()/h0(), h_c, hdot_c, r_c, -hdot_c * RE_prof * RE_prof);  // hdot shuold be equal to the inner mass according to the profile
        initialize(m, r, mc, rc, h_c, hdot_c, R_st);
	printf("r: %e, rc: %e, m: %e, mc: %e, d0: %e, rho_c: %e, h0: %e, hE/h0: %e, h_c: %e, hdot_c: %e, r_c: %e, d(r_c): %e, p_mass: %e, p_smooth: %e\n", r, rc, m, mc, d0(), dC(), h0(), HE()/h0(), h_c, hdot_c, r_c, enthalpy_to_density(h_c), p_mass, p_smooth_l); // getting the mass
	my_radius = r;
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
//	my_radius = radius;
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
	/*	if (R0 != 1.0) {
			printf("%e %e %e %e %e %e %e, %e\n", r, m, h, d, dr, f_C, M0, R0);
		}*/
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
			core_mass += dm2 / 2.0;
		}
		h += (dh1 + dh2) / 2.0;
		hdot += (dhdot1 + dhdot2) / 2.0;
		r += dr;
		m += (dm1 + dm2) / 2.0;
	} while (h > 0.0);
	mass = m;
	radius = r;
//	my_radius = radius;
//	printf( "Radius = %e\n", my_radius);
}

void struct_eos::initialize(real &mass, real &radius, real &core_mass, real &core_radius, real &h_at_rc, real &hdot_at_rc, real rc) {

        const real dr0 = R0 * 1.0e-4;
        core_mass = 0.0;
	core_radius = 0.0;
        real h = h0();
        real hdot = 0.0;
	h_at_rc = 0.0;
	hdot_at_rc = 0.0;
        real r = 0.0;
	real m = 0.0;
/*
	h = HE();
	hdot = -opts().star_rho_out / (rc * rc);
	core_mass = 0.0;
	r = rc;
	m += opts().star_rho_out;
	printf("starting at r_c: %e, h: %e, hdot: %e, p_mass: %e, p_smooth: %e\n", rc, h, hdot, p_mass, p_smooth_l);
 */
	m += p_mass;
	core_mass += p_mass;
        real dr = dr0;
        real d;
        do {
                if (hdot != 0.0) {
                        dr = std::max(std::min(dr0, std::abs(h * INVERSE(hdot)) / 2.0), dr0 * 1.0e-6);
                }
                d = this->enthalpy_to_density(h);
  //              printf("%e %e %e\n", r, d, h);
       //         printf("%e %e %e %e %e\n", r, m, h, d, dr);
                const real dh1 = dh_dr(h, hdot, r) * dr;
                const real dhdot1 = dhdot_dr(h, hdot, r) * dr;
                const real dm1 = 4.0 * M_PI * d * sqr(r) * dr;
                if (d >= dC()) {
                        core_mass += dm1 / 2.0;
			core_radius += dr / 2.0;
                }
                if (h + dh1 <= ZERO) {
                        break;
                }
                d = this->enthalpy_to_density(h + dh1);
                const real dh2 = dh_dr(h + dh1, hdot + dhdot1, r + dr) * dr;
                const real dhdot2 = dhdot_dr(h + dh1, hdot + dhdot1, r + dr) * dr;
                const real dm2 = 4.0 * M_PI * d * sqr(r + dr) * dr;
                if (enthalpy_to_density(h + dh1) >= dC()) {
                        core_mass += dm2 / 2.0;
			core_radius += dr / 2.0;
                }
                h += (dh1 + dh2) / 2.0;
                hdot += (dhdot1 + dhdot2) / 2.0;
		if (((r + dr) > rc) && (h_at_rc == 0.0)) {
                        h_at_rc = h;
                        hdot_at_rc = hdot;			
			break;
		}
                r += dr;
                m += (dm1 + dm2) / 2.0;
        } while (h > 0.0);
        mass = m;
        radius = r;
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

void struct_eos::set_p_mass(real pmass) {
	p_mass = pmass;
}

real struct_eos::B() const {
	return SQRT(POWER(A * INVERSE( G ), 1.5) * INVERSE( wdcons));
}

real struct_eos::get_R0() const {
	if (my_radius > 0.0) {
		return my_radius;
	}	
	if (opts().eos == WD) {
		real m, r;
		struct_eos tmp = *this;
		tmp.initialize(m, r);
		return r;
	} else {
		real m, r;
		struct_eos tmp = *this;
		if (p_mass == 0.0) {
			tmp.initialize(m, r);
		} else {
			real cr, cm, h, hdot;
			tmp.initialize(m, r, cm, cr, h, hdot, p_smooth_l*1e6);
		}
		return r;
	}
}

void struct_eos::update_R0() {
	real r;
	struct_eos tmp = *this;
        if (opts().eos == WD) {
                real m;
                tmp.initialize(m, r);
        } else {
                real m, cm;
                if (p_mass == 0.0) {
                        tmp.initialize(m, r, cm);
                } else {
                        real cr, cm, h, hdot;
                        tmp.initialize(m, r, cm, cr, h, hdot, p_smooth_l*1e6);
                }
        }
	printf("updating my radius with: %e\n", r);
	my_radius = r;
}

real struct_eos::get_RC() const {
                real m, r, cr, cm, h, hdot;
		struct_eos tmp = *this;
                tmp.initialize(m, r, cm, cr, h, hdot, p_smooth_l*1e6);
                return cr;
}

real struct_eos::get_p_mass() const {
	return (*this).p_mass;
}

real struct_eos::density_at(real R, real dr) {
	real r;
	real h = h0();
	real hdot = 0.0;
	int N = std::min(std::max(int(R / dr + 1.0), 2),10);
	if (p_mass >= 0.0) {
		N *= 60;
	}
	dr = R / real(N);
	for (integer i = 0; i < N; ++i) {
		r = i * dr;
		const real dh1 = dh_dr(h, hdot, r) * dr;
		const real dhdot1 = dhdot_dr(h, hdot, r) * dr;
		if (h + dh1 <= ZERO) {
			return 0.0;
		}
		const real dh2 = dh_dr(h + dh1, hdot + dhdot1, r + dr) * dr;
		const real dhdot2 = dhdot_dr(h + dh1, hdot + dhdot1, r + dr) * dr;
		h += (dh1 + dh2) / 2.0;
		hdot += (dhdot1 + dhdot2) / 2.0;
		if (h <= ZERO) {
			return 0.0;
		}
	}
	real d = this->enthalpy_to_density(h);
	return d;
}
