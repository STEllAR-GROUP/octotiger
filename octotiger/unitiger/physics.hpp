//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER_UNITIGER_PHYSICS_HPP_
#define OCTOTIGER_UNITIGER_PHYSICS_HPP_

#include "octotiger/unitiger/safe_real.hpp"
#include "octotiger/test_problems/blast.hpp"
#include "octotiger/test_problems/exact_sod.hpp"
#include "octotiger/stellar_eos/stellar_eos.hpp"
#include "octotiger/stellar_eos/ideal_eos.hpp"
#include <functional>
#include <vector>

template<int NDIM>
struct physics {
	static constexpr char const *field_names3[] =
			{ "rho", "egas", "ein", "pot", "sx", "sy", "sz", "zx", "zy", "zz", "spc_1", "spc_2", "spc_3", "spc_4", "spc_5" };
	static constexpr char const *field_names2[] = { "rho", "egas", "ein", "pot", "sx", "sy", "zz", "spc_1", "spc_2", "spc_3", "spc_4", "spc_5" };
	static constexpr char const *field_names1[] = { "rho", "egas", "ein", "pot", "sx", "spc_1", "spc_2", "spc_3", "spc_4", "spc_5" };
	static constexpr int rho_i = 0;
	static constexpr int egas_i = 1;
	static constexpr int ein_i = 2;
	static constexpr int pot_i = 3;
	static constexpr int sx_i = 4;
	static constexpr int sy_i = 5;
	static constexpr int sz_i = 6;
	static constexpr int lx_i = 4 + NDIM;
	static constexpr int ly_i = 5 + NDIM;
	static constexpr int lz_i = 6 + NDIM;
	static constexpr int spc_i = 4 + NDIM + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2));
	static safe_real de_switch_1;
	static safe_real de_switch_2;
	static stellar_eos* eos;

	enum test_type {
		SOD, BLAST, KH, CONTACT, KEPLER
	};

	static std::string get_test_type_string(test_type t) {
		switch (t) {
		case SOD:
			return "SOD";
		case BLAST:
			return "BLAST";
		case KH:
			return "KH";
		case CONTACT:
			return "CONTACT";
		default:
			return "OCTOTIGER";
		}
	}

	static int field_count();

	static bool contact_field(int f) {
		return (f == rho_i || (f >= spc_i && f < spc_i + n_species_));
	}

	static void set_fgamma(safe_real fg);

	static void to_prim(std::vector<safe_real> u, safe_real &p, safe_real &v, safe_real &c, int dim);

	static void enforce_outflows(hydro::state_type &U, const hydro::x_type &X, int face) {

	}

	template<int INX>
	static void physical_flux(const std::vector<safe_real> &U, std::vector<safe_real> &F, int dim, safe_real &am, safe_real &ap, std::array<safe_real, NDIM> &x,
			std::array<safe_real, NDIM> &vg);

	template<int INX>
	static void post_process(hydro::state_type &U, const hydro::x_type &X, safe_real dx);

	template<int INX>
	static void source(hydro::state_type &dudt, const hydro::state_type &U, const hydro::flux_type &F, const hydro::x_type X, safe_real omega, safe_real dx);

	template<int INX>
	static void derivative_source(hydro::state_type &dudt, const hydro::state_type &U, const std::vector<std::vector<std::vector<safe_real>>> &Q,
			const hydro::x_type X, safe_real omega, safe_real dx);

	/*** Reconstruct uses this - GPUize****/
	template<int INX>
	static const hydro::state_type& pre_recon(const hydro::state_type &U, const hydro::x_type X, safe_real omega, bool angmom);
	/*** Reconstruct uses this - GPUize****/
	template<int INX>
	static void post_recon(std::vector<std::vector<std::vector<safe_real>>> &Q, const hydro::x_type X, safe_real omega, bool angmom);
	template<int INX>
	using comp_type = hydro_computer<NDIM, INX, physics<NDIM>>;

	template<int INX>
	std::vector<typename comp_type<INX>::bc_type> initialize(test_type t, hydro::state_type &U, hydro::x_type &X);

	template<int INX>
	static void analytic_solution(test_type test, hydro::state_type &U, const hydro::x_type &X, safe_real time);

	template<int INX>
	static const std::vector<std::vector<double>>& find_contact_discs(const hydro::state_type &U);

	static void set_n_species(int n);

	static void update_n_field();

	static void set_dual_energy_switches(safe_real one, safe_real two);

	static void set_central_force(safe_real GM) {
		GM_ = GM;
	}
	static int get_angmom_index() {
		return sx_i;
	}

	template<int INX>
	static void enforce_outflow(hydro::state_type &U, int dim, int dir);

	using eos_func = std::function<double(double,double,double, double)>;
	using eos_func2 = std::function<std::pair<double,double>(double,double,double, double)>;


	static void set_atomic_data(const std::vector<double> a, const std::vector<double> z) {
		for (int s = 0; s < n_species_; s++) {
			A[s] = a[s];
			Z[s] = z[s];
		}
	}

	static void set_stellar_eos(stellar_eos* e) {
		delete eos;
		eos = e;
	}

private:
	static safe_real rho_sink_radius_;
	static safe_real rho_sink_floor_;
	static int nf_;
	static int n_species_;
	static safe_real fgamma_;
	static safe_real GM_;
	static std::vector<safe_real> A;
	static std::vector<safe_real> Z;

};

template<int NDIM>
std::vector<safe_real> physics<NDIM>::A;

template<int NDIM>
std::vector<safe_real> physics<NDIM>::Z;

template<int NDIM>
stellar_eos* physics<NDIM>::eos = new ideal_eos;

//definitions of the declarations (and initializations) of the static constexpr variables
template<int NDIM>
constexpr char const *physics<NDIM>::field_names1[];
template<int NDIM>
constexpr char const *physics<NDIM>::field_names2[];
template<int NDIM>
constexpr char const *physics<NDIM>::field_names3[];

template<int NDIM>
safe_real physics<NDIM>::rho_sink_radius_ = 0.0;

template<int NDIM>
safe_real physics<NDIM>::rho_sink_floor_ = 0.0;

template<int NDIM>
safe_real physics<NDIM>::GM_ = 0.0;

template<int NDIM>
safe_real physics<NDIM>::de_switch_1 = 1e-3;

template<int NDIM>
safe_real physics<NDIM>::de_switch_2 = 1e-1;

template<int NDIM>
int physics<NDIM>::nf_ = (4 + NDIM + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2))) + physics<NDIM>::n_species_;

template<int NDIM>
int physics<NDIM>::n_species_ = -1;

template<int NDIM>
safe_real physics<NDIM>::fgamma_ = 7. / 5.;

#endif /* OCTOTIGER_UNITIGER_PHYSICS_HPP_ */
