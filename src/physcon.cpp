/*
 * physcon.cpp
 *
 *  Created on: Mar 10, 2017
 *      Author: dminacore
 */

#define __NPHYSCON__
#include "physcon.hpp"
#include <hpx/hpx.hpp>
#include "options.hpp"
#include "future.hpp"
#include <hpx/lcos/broadcast.hpp>
#include "util.hpp"

physcon_t physcon = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, { 4.0, 4.0, 4.0, 4.0, 4.0 }, { 2.0, 2.0, 2.0, 2.0, 2.0 } };

#include "node_server.hpp"

void normalize_constants();

real find_T_rad_gas(real p, real rho, real mu) {
	const real cg = physcon.kb / (mu * physcon.mh);
	const real cr = (4.0 * physcon.sigma) / (3.0 * physcon.c);
	real T = std::min(p / (cg * rho), std::pow(p / cr, 0.25));
	real dfdT, f;
	for (int i = 0; i != 6; ++i) {
		f = cg * rho * T + cr * std::pow(T, 4) - p;
		dfdT = cg * rho + 4.0 * cr * std::pow(T, 3);
		T -= f / dfdT;
	}
//	printf("%e\n", f / (T * dfdT));
	return T;
}

static void these_units(real& m, real& l, real& t, real& k) {
	real A = physcon.A;
	real B = physcon.B;
	real G = physcon.G;
	real kb = physcon.kb;
	real m1 = std::pow(A / G, 1.5) / (B * B);
	real l1 = std::sqrt(A / G) / B;
	real t1 = 1.0 / std::sqrt(B * G);
	real k1 = (m1 * l1 * l1) / (t1 * t1) / kb;
	A = 6.00228e+22;
	B = 2 * 9.81011e+5;
	G = 6.67259e-8;
	kb = 1.380658e-16;
	real m2 = std::pow(A / G, 1.5) / (B * B);
	real l2 = std::sqrt(A / G) / B;
	real t2 = 1.0 / std::sqrt(B * G);
	real k2 = (m2 * l2 * l2) / (t2 * t2) / kb;
//	printf("%e %e %e %e\n", m2, l2, t2, k2);
	m = m2 / m1;
	l = l2 / l1;
	t = t2 / t1;
	k = k2 / k1;
}

void normalize_constants() {
	real m, l, t, k;
//	physcon.A = 6.00228e+22;
//	physcon.B = 2 * 9.81011e+5;
//	physcon.G = 6.67259e-8;
//	physcon.kb = 1.380658e-16;
	these_units(m, l, t, k);
	m = 1.0 / m;
	l = 1.0 / l;
	t = 1.0 / t;
//	k = 1.0 / k;
	//k = 1.0 / 1000000.0;
	k = 1.0;
	physcon.kb = 1.380658e-16 * (m * l * l) / (t*t) / k;
	physcon.c = 2.99792458e+10 * (l / t);
	physcon.mh = 1.6733e-24 * m;
	physcon.sigma = 5.67051e-5 * m / (t * t * t) / (k * k * k * k);
	physcon.h = 6.6260755e-27 * m * l * l / t;
//	printf("Noralized constants\n");
//	printf("%e %e %e %e\n", 1.0/m, 1.0/l, 1.0/t, 1.0/k);
//	printf("A = %e | B = %e | G = %e | kb = %e | c = %e | mh = %e | sigma = %e | h = %e\n", physcon.A, physcon.B, physcon.G, physcon.kb, physcon.c, physcon.mh,
///			physcon.sigma, physcon.h);
}

struct call_normalize_constants
{
    call_normalize_constants()
    {
        normalize_constants();
    }
};
call_normalize_constants init;

hpx::future<void> set_physcon(const physcon_t& p);

HPX_PLAIN_ACTION(set_physcon, set_physcon_action);

HPX_REGISTER_BROADCAST_ACTION_DECLARATION (set_physcon_action);
HPX_REGISTER_BROADCAST_ACTION (set_physcon_action);

hpx::future<void> set_physcon(const physcon_t& p) {
	hpx::future<void> f;
	if (hpx::get_locality_id() == 0 && options::all_localities.size() > 1) {
		std::vector<hpx::id_type> remotes;
		remotes.reserve(options::all_localities.size() - 1);
		for (hpx::id_type const& id : options::all_localities) {
			if (id != hpx::find_here())
				remotes.push_back(id);
		}
		f = hpx::lcos::broadcast < set_physcon_action > (remotes, p);
	} else {
		f = hpx::make_ready_future();
	}
	physcon = p;
	return f;
}

void node_server::set_cgs(bool change) {

	real m, l, t, k;
	if (change) {
		these_units(m, l, t, k);
	}
	physcon.A = 6.00228e+22;
	physcon.B = 2 * 9.81011e+5;
	physcon.G = 6.67259e-8;
	physcon.kb = 1.380658e-16;
	physcon.c = 2.99792458e+10;
	physcon.mh = 1.6733e-24;
	physcon.sigma = 5.67051e-5;
	physcon.h = 6.6260755e-27;
	auto f1 = set_physcon(physcon);
	if (change) {
		printf("%e %e %e %e\n", m, l, t, k);
		auto f2 = change_units(m, l, t, k);
		auto f3 = grid::static_change_units(m, l, t, k);
		f2.get();
		f3.get();
	}
	f1.get();

}

HPX_PLAIN_ACTION(set_AB, set_AB_action);

HPX_REGISTER_BROADCAST_ACTION_DECLARATION (set_AB_action);
HPX_REGISTER_BROADCAST_ACTION (set_AB_action);

void set_AB(real a, real b) {
	if (hpx::get_locality_id() == 0) {
        std::vector<hpx::id_type> remotes;
        remotes.reserve(options::all_localities.size()-1);
        for (hpx::id_type const& id: options::all_localities)
        {
            if(id != hpx::find_here())
                remotes.push_back(id);
        }
        if (remotes.size() > 0) {
            hpx::lcos::broadcast<set_AB_action>(remotes, a, b).get();
        }
	}
	physcon.A = a;
	physcon.B = b;
	normalize_constants();
}

HPX_PLAIN_ACTION(grid::static_change_units, static_change_units_action);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION (static_change_units_action);
HPX_REGISTER_BROADCAST_ACTION (static_change_units_action);

hpx::future<void> grid::static_change_units(real m, real l, real t, real k) {
	printf("%e %e %e %e\n", m, l, t, k);
	hpx::future<void> f;
	if (hpx::get_locality_id() == 0 && options::all_localities.size() > 1) {
		std::vector<hpx::id_type> remotes;
		remotes.reserve(options::all_localities.size() - 1);
		for (hpx::id_type const& id : options::all_localities) {
			if (id != hpx::find_here())
				remotes.push_back(id);
		}
		f = hpx::lcos::broadcast < static_change_units_action > (remotes, m, l, t, k);
	} else {
		f = hpx::make_ready_future();
	}
	grid::omega /= t;
	pivot *= l;
	scaling_factor *= l;
	return f;
}

real mean_ion_weight(const std::array<real, NSPECIES> species) {
//	real N;
	real mtot = 0.0;
	real ntot = 0.0;
	for (integer i = 0; i != NSPECIES; ++i) {
		const real m = species[i];
		ntot += m * (physcon._Z[i] + 1.0) / physcon._A[i];
		mtot += m;
	}
	if (ntot > 0.0) {
		return mtot / ntot;
	} else {
		return 0.0;
	}
}

typedef node_server::change_units_action change_units_action_type;
HPX_REGISTER_ACTION (change_units_action_type);

hpx::future<void> node_client::change_units(real a, real b, real c, real d) const {
	return hpx::async<typename node_server::change_units_action>(get_unmanaged_gid(), a, b, c, d);
}

hpx::future<void> node_server::change_units(real a, real b, real c, real d) {
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
	return f;
}

real stellar_temp_from_rho_mu_s(real rho, real mu, real s) {
	const real m = physcon.mh;
	const real k = physcon.kb;
	const real pi = M_PI;
	const real c = physcon.c;
	const real sigma = physcon.sigma;
	const real num1 = rho * 27.0 * std::pow(k, 4) * std::pow(pi, 7) * std::exp((2.0 * m * s * mu) / k) - 5.0;
	const real den1 = 400.0 * std::pow(c, 5) * std::pow(m, 4) * mu * sigma;
	const real num2 = 3.0 * c * k * rho * LambertW(num1 / den1);
	const real den2 = 8.0 * m * mu * sigma;
	const real T = std::pow(num2 / den2, 1.0 / 3.0);
	return T;
}

real stellar_enthalpy_from_rho_mu_s(real rho, real mu, real s) {
	const real m = physcon.mh;
	const real k = physcon.kb;
	const real B = physcon.B;
	const real A = physcon.A;
	const real x = std::pow(rho / B, 1.0 / 3.0);
	const real sigma = physcon.sigma;
	const real c = physcon.c;
	const real T = stellar_temp_from_rho_mu_s(rho, mu, s);
	const real hd = (8.0 * A / B) * (std::sqrt(x * x + 1.0) - 1.0);
	const real hg = (2.5 * k * T) / (mu * m);
	const real hr = (16.0 / 3.0) * (sigma / c) * std::pow(T, 4) / rho;
	return hd + hg + hr;
}

real stellar_rho_from_enthalpy_mu_s(real h, real mu, real s) {
	real rho;
	std::function<real(real)> f = [=](real h) {
		return stellar_enthalpy_from_rho_mu_s(h,mu,s);
	};
	bool rc = find_root(f, 1.0e-20, 1.0e+20, rho);
	if (!rc) {
		abort_error()
		;
	}
	return rho;
}


void rad_coupling_vars( real rho, real e, real mmw, real& bp, real& kp, real& dkpde, real& dbde) {
	constexpr  const real gm1 = 2.0 / 3.0;
	constexpr real pi_inv = 1.0 / M_PI;
	constexpr real coeff = 30.262 * 4.0e+25;
	const real Z = 0.0;
	const real einv = 1.0 / e;
	const real T = (gm1 * mmw * physcon.mh / physcon.kb) * (e / rho);
	kp =  coeff * (Z + 0.0001) * rho * rho * std::pow(T, -3.5);
	dkpde = -3.5 * kp * einv;
	bp = physcon.sigma * pi_inv * std::pow(T, 4.0);
	dbde = 4.0 * bp * einv;
}



real kappa_R(real rho, real e, real mmw) {
	const real Z = 0.0;
	const real zbar = 2.0; //GENERALIZE
	const real T = temperature(rho, e, mmw);
	const real f1 = (T * T + 2.7e+11 * rho);
	const real f2 = (1.0 + std::pow(T / (4.5e+8), 0.86));
	const real k_ff_bf = 4.0e+25 * (Z + 0.0001) * rho * std::pow(T, -3.5);
	const real k_T = 0.2 * T * T / (f1 * f2);
	const real k_cond = 2.6e-7 * zbar * (T * T) / (rho * rho) * (1.0 + std::pow(rho / 2.0e+6, 2.0 / 3.0));
	const real k_rad = (k_ff_bf + k_T);
	const real k_tot = k_rad * k_cond / (k_rad + k_cond);
	//const real k_tot = k_rad;;
	return rho * k_tot;
}


real temperature(real rho, real e, real mmw) {
	const real gm1 = 2.0 / 3.0;
	return (gm1 * mmw * physcon.mh / physcon.kb) * (e / rho);
}

real kappa_p(real rho, real e, real mmw) {
	const real T = temperature(rho, e, mmw);
	const real Z = 0.0;
	const real k = 30.262 * 4.0e+25 * (Z + 0.0001) * rho * std::pow(T, -3.5);
	return rho * k;
}

real dkappa_p_de(real rho, real e, real mmw) {
	const real T = temperature(rho, e, mmw);
	return -3.5 * kappa_p(rho, e, mmw) / e;
}

real B_p(real rho, real e, real mmw) {
	const real T = temperature(rho, e, mmw);
	return (physcon.sigma / M_PI) * std::pow(T, 4.0);
}

real dB_p_de(real rho, real e, real mmw) {
	return 4.0 * B_p(rho, e, mmw) / e;
}

