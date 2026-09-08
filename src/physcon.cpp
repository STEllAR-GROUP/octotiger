//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define __NPHYSCON__
#include "octotiger/future.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/math/Real.hpp"
#include "octotiger/math/Debug.hpp"
#include "octotiger/util.hpp"

#include <hpx/hpx.hpp>
#include <hpx/collectives/broadcast_direct.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <functional>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

physcon_t& physcon() {
	static physcon_t physcon_ = { 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0 };
	return physcon_;
}

//physcon_t physcon() = { 6.00228e+22, 6.67259e-8, 2 * 9.81011e+5, 1.380658e-16, 1.0, 1.0, 1.0, 1.0, { 4.0, 4.0, 4.0, 4.0, 4.0 }, { 2.0, 2.0, 2.0, 2.0, 2.0 } };

Real find_T_rad_gas(Real p, Real rho, Real mu) {
	const Real cg = physcon().kb / (mu * physcon().mh);
	const Real cr = (4.0 * physcon().sigma) / (3.0 * physcon().c);
	Real T = std::min(p / (cg * rho), std::pow(p / cr, 0.25));
	Real dfdT, f;
	for (int i = 0; i != opts().ipr_nr_maxiter; ++i) {
		f = cg * rho * T / p + cr * std::pow(T, 4) / p - 1.0;
		if (std::abs(f) < opts().ipr_nr_tol) {
			return T;
		}
		dfdT = cg * rho / p + 4.0 * cr * std::pow(T, 3) / p;
		T -= f / dfdT;
	}
        std::cout << "Error: exceeded number of iterations in Newton-Rahpson method, could not find accurate temperature.\n";
        std::cout << "Maximum number of iteration " << opts().ipr_nr_maxiter << ". Current iteration find: " << -1.0 << "+" << cg << "*" << T << "+" << cr << "*" << T << "^4=" << f << "\n";
	abort();
	return T;
}

Real find_ei_rad_gas(Real p, Real rho, Real mu, Real gamma, Real &T) {
	T = find_T_rad_gas(p, rho, mu);
        const Real cg_e = physcon().kb / (mu * physcon().mh) / (gamma - 1.0);
        const Real cr_e = (4.0 * physcon().sigma) / physcon().c;
	return cg_e * rho * T + cr_e * std::pow(T, 4); 
}

void these_units(Real &m, Real &l, Real &t, Real &k) {
	const Real Acgs = 6.00228e+22;
	const Real Bcgs = 2 * 9.81011e+5;
	const Real Gcgs = 6.67259e-8;
	const Real kbcgs = 1.380658e-16;
	Real m1, l1, t1, k1;
	Real kb = kbcgs;
	Real A, B, G;
	if ((opts().radiation || opts().eos == WD) && opts().gravity) {
		A = physcon().A;
		B = physcon().B;
		G = physcon().G;
		m1 = std::pow(A / G, 1.5) / (B * B);
		l1 = std::sqrt(A / G) / B;
		t1 = 1.0 / std::sqrt(B * G);
		k1 = (m1 * l1 * l1) / (t1 * t1) / kb;
		A = Acgs;
		B = Bcgs;
		G = Gcgs;
		Real m2 = std::pow(A / G, 1.5) / (B * B);
		Real l2 = std::sqrt(A / G) / B;
		Real t2 = 1.0 / std::sqrt(B * G);
		Real k2 = (m2 * l2 * l2) / (t2 * t2) / kb;
		m = m2 / m1;
		l = l2 / l1;
		t = t2 / t1;
		k = 1.0;
		;
	} else if (!opts().radiation && !opts().gravity) {
		m = opts().code_to_g;
		l = opts().code_to_cm;
		t = opts().code_to_s;
		k = 1.0;
	} else if (opts().radiation) {
		m = opts().code_to_g;
		l = opts().code_to_cm;
		t = opts().code_to_cm / 2.99792458e+10;
		k = 1.0;
	} else {
		G = 1.0;
		m = opts().code_to_g;
		l = opts().code_to_cm;
		t = std::sqrt(l * l * l / m) / std::sqrt(Gcgs / G);
		k = 1.0;
	}

//	printf("%e %e %e %e\n", l, m, t, k);
	if (opts().problem == MARSHAK) {
		opts().code_to_g = 1.0;
		opts().code_to_s = 1.0;
		opts().code_to_cm = 1.0;
	} else {
		opts().code_to_cm = l;
		opts().code_to_g = m;
		opts().code_to_s = t;
	}
}

void normalize_constants() {
	Real m, l, t, k;
	these_units(m, l, t, k);
	m = 1.0 / m;
	l = 1.0 / l;
	t = 1.0 / t;
	k = 1.0 / k;
	physcon().kb = 1.380658e-16 * (m * l * l) / (t * t) / k;
	physcon().c = 2.99792458e+10 * (l / t);
	physcon().mh = 1.6733e-24 * m;
	physcon().sigma = 5.67051e-5 * m / (t * t * t) / (k * k * k * k);
	physcon().h = 6.6260755e-27 * m * l * l / t;
	if (hpx::get_locality_id() == 0) {
		printf("Normalized constants 222\n");
		printf("%e %e %e %e\n", 1.0 / m, 1.0 / l, 1.0 / t, 1.0 / k);
		printf("A = %e | B = %e | G = %e | kb = %e | c = %e | mh = %e | sigma = %e | h = %e\n", physcon().A, physcon().B, physcon().G, physcon().kb,
				physcon().c, physcon().mh, physcon().sigma, physcon().h);
	}
	if (opts().problem == MARSHAK) {
		opts().code_to_g = 1.0;
		opts().code_to_s = 1.0;
		opts().code_to_cm = 1.0;
	} else {
		opts().code_to_g = 1.0 / m;
		opts().code_to_s = 1.0 / t;
		opts().code_to_cm = 1.0 / l;
	}
}

void set_units(Real m, Real l, Real t, Real k) {
//	m = 1.0 / m;
//	l = 1.0 / l;
//	t = 1.0 / t;
//	k = 1.0 / k;
	const Real Acgs = 6.00228e+22;
	physcon().A = 6.00233345657677e+22 * m / l / (t * t);
	physcon().B = 2 * 9.81011e+05 * m
			/ (l * l * l);
	physcon().kb = 1.380658e-16 * (m * l * l) / (t * t) / k;
	physcon().c = 2.99792458e+10 * (l / t);
	physcon().mh = 1.6733e-24 * m;
	physcon().sigma = 5.67051e-5 * m / (t * t * t) / (k * k * k * k);
	physcon().h = 6.6260755e-27 * m * l * l / t;
//	if (hpx::get_locality_id() == 0) {
		printf("normalized constants\n");
		printf("%e %e %e %e\n", 1.0 / m, 1.0 / l, 1.0 / t, 1.0 / k);
		printf("A = %e | B = %e | G = %e | kb = %e | c = %e | mh = %e | sigma = %e | h = %e\n", physcon().A, physcon().B, physcon().G, physcon().kb,
				physcon().c, physcon().mh, physcon().sigma, physcon().h);
//	}
}

struct call_normalize_constants {
	call_normalize_constants() {
		normalize_constants();
	}
};

//call_normalize_constants init;

hpx::future<void> set_physcon(const physcon_t &p);

HPX_PLAIN_ACTION(set_physcon, set_physcon_action);

HPX_REGISTER_BROADCAST_ACTION_DECLARATION (set_physcon_action);
HPX_REGISTER_BROADCAST_ACTION (set_physcon_action);

hpx::future<void> set_physcon(const physcon_t &p) {
	hpx::future<void> f;
	if (hpx::get_locality_id() == 0 && options::all_localities.size() > 1) {
		std::vector<hpx::id_type> remotes;
		remotes.reserve(options::all_localities.size() - 1);
		for (hpx::id_type const &id : options::all_localities) {
			if (id != hpx::find_here())
				remotes.push_back(id);
		}
		f = hpx::lcos::broadcast < set_physcon_action > (remotes, p);
	} else {
		f = hpx::make_ready_future();
	}
	physcon() = p;
	return f;
}

void node_server::set_cgs(bool change) {

	Real m, l, t, k;
	if (change) {
		these_units(m, l, t, k);
	}
	physcon().A = 6.00228e+22;
	physcon().B = 2 * 9.81011e+5;
	physcon().G = 6.67259e-8;
	physcon().kb = 1.380658e-16;
	physcon().c = 2.99792458e+10;
	physcon().mh = 1.6733e-24;
	physcon().sigma = 5.67051e-5;
	physcon().h = 6.6260755e-27;
	physcon_t tmp = physcon();
	auto f1 = set_physcon(tmp);
	if (change) {
//		printf("%e %e %e %e\n", m, l, t, k);
		change_units(m, l, t, k);
		auto f3 = grid::static_change_units(m, l, t, k);
		f3.get();
	}
	f1.get();

}

HPX_PLAIN_ACTION(set_AB, set_AB_action);

HPX_REGISTER_BROADCAST_ACTION_DECLARATION (set_AB_action);
HPX_REGISTER_BROADCAST_ACTION (set_AB_action);

void set_AB(Real a, Real b) {
	if (hpx::get_locality_id() == 0) {
		std::vector<hpx::id_type> remotes;
		remotes.reserve(options::all_localities.size() - 1);
		for (hpx::id_type const &id : options::all_localities) {
			if (id != hpx::find_here())
				remotes.push_back(id);
		}
		if (remotes.size() > 0) {
			hpx::lcos::broadcast < set_AB_action > (remotes, a, b).get();
		}
	}
	physcon().A = a;
	physcon().B = b;
	normalize_constants();
}

HPX_PLAIN_ACTION(grid::static_change_units, static_change_units_action);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION (static_change_units_action);
HPX_REGISTER_BROADCAST_ACTION (static_change_units_action);

hpx::future<void> grid::static_change_units(Real m, Real l, Real t, Real k) {
//	printf("%e %e %e %e\n", m, l, t, k);
	hpx::future<void> f;
	if (hpx::get_locality_id() == 0 && options::all_localities.size() > 1) {
		std::vector<hpx::id_type> remotes;
		remotes.reserve(options::all_localities.size() - 1);
		for (hpx::id_type const &id : options::all_localities) {
			if (id != hpx::find_here())
				remotes.push_back(id);
		}
		f = hpx::lcos::broadcast < static_change_units_action > (remotes, m, l, t, k);
	} else {
		f = hpx::make_ready_future();
	}
	grid::omega /= t;
	scaling_factor *= l;
	return f;
}

void mean_ion_weight(const specie_state_t<> species, Real &mmw, Real &X, Real &Z) {
//	Real N;
	Real mtot = 0.0;
	Real ntot = 0.0;
	X = Z = 0;
	for (integer i = 0; i != opts().n_species; ++i) {
		const Real m = species[i];
		ntot += m * (opts().atomic_number[i] + 1.0) / opts().atomic_mass[i];
		X += m * opts().X[i];
		Z += m * opts().Z[i];
		mtot += m;
	}
	mmw = mtot / ntot;
	X /= mtot;
	Z /= mtot;
}

using change_units_action_type = node_server::change_units_action;
HPX_REGISTER_ACTION (change_units_action_type);

hpx::future<void> node_client::change_units(Real a, Real b, Real c, Real d) const {
	return hpx::async<typename node_server::change_units_action>(get_unmanaged_gid(), a, b, c, d);
}

void node_server::change_units(Real a, Real b, Real c, Real d) {
	dx *= b;
	hpx::future<void> f;
	std::array<hpx::future<void>, NCHILD> futs;
	if (is_refined) {
		integer index = 0;
		for (auto i = children.begin(); i != children.end(); ++i) {
			futs[index++] = i->change_units(a, b, c, d);
		}
	}
	grid_ptr->change_units(a, b, c, d);
	if (is_refined) {
		f = hpx::when_all(futs.begin(), futs.end());
	} else {
		f = hpx::make_ready_future();
	}
	f.get();
}

Real stellar_temp_from_rho_mu_s(Real rho, Real mu, Real s) {
	const Real m = physcon().mh;
	const Real k = physcon().kb;
	const Real pi = M_PI;
	const Real c = physcon().c;
	const Real sigma = physcon().sigma;
	const Real num1 = rho * 27.0 * std::pow(k, 4) * std::pow(pi, 7) * std::exp((2.0 * m * s * mu) / k) - 5.0;
	const Real den1 = 400.0 * std::pow(c, 5) * std::pow(m, 4) * mu * sigma;
	const Real num2 = 3.0 * c * k * rho * LambertW(num1 / den1);
	const Real den2 = 8.0 * m * mu * sigma;
	const Real T = std::pow(num2 / den2, 1.0 / 3.0);
	return T;
}

Real stellar_enthalpy_from_rho_mu_s(Real rho, Real mu, Real s) {
	const Real m = physcon().mh;
	const Real k = physcon().kb;
	const Real B = physcon().B;
	const Real A = physcon().A;
	const Real x = std::pow(rho / B, 1.0 / 3.0);
	const Real sigma = physcon().sigma;
	const Real c = physcon().c;
	const Real T = stellar_temp_from_rho_mu_s(rho, mu, s);
	const Real hd = (8.0 * A / B) * (std::sqrt(x * x + 1.0) - 1.0);
	const Real hg = (2.5 * k * T) / (mu * m);
	const Real hr = (16.0 / 3.0) * (sigma / c) * std::pow(T, 4) / rho;
	return hd + hg + hr;
}

Real stellar_rho_from_enthalpy_mu_s(Real h, Real mu, Real s) {
	Real rho;
	std::function<Real(Real)> f = [=](Real h) {
		return stellar_enthalpy_from_rho_mu_s(h, mu, s);
	};
	bool rc = find_root(f, 1.0e-20, 1.0e+20, rho);
	if (!rc) {
		abort_error()
		;
	}
	return rho;
}

void rad_coupling_vars(Real rho, Real e, Real mmw, Real &bp, Real &kp, Real &dkpde, Real &dbde) {
	constexpr const Real gm1 = 2.0 / 3.0;
	constexpr Real pi_inv = 1.0 / M_PI;
	constexpr Real coeff = 30.262 * 4.0e+25;
	const Real Z = 0.0;
	const Real einv = 1.0 / e;
	const Real T = (gm1 * mmw * physcon().mh / physcon().kb) * (e / rho);
	kp = coeff * (Z + 0.0001) * rho * rho * std::pow(T, -3.5);
	dkpde = -3.5 * kp * einv;
	bp = physcon().sigma * pi_inv * std::pow(T, 4.0);
	dbde = 4.0 * bp * einv;
}

#endif
