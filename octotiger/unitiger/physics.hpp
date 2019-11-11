//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER_UNITIGER_PHYSICS_HPP_
#define OCTOTIGER_UNITIGER_PHYSICS_HPP_

#include "octotiger/unitiger/safe_real.hpp"
#include "octotiger/test_problems/blast.hpp"
#include "octotiger/test_problems/exact_sod.hpp"

template<int NDIM>
struct physics {

	static constexpr int rho_i = 0;
	static constexpr int egas_i = 1;
	static constexpr int tau_i = 2;
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

	enum test_type {
		SOD, BLAST, KH, CONTACT
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

	static void set_fgamma(safe_real fg);

	static void to_prim(std::vector<safe_real> u, safe_real &p, safe_real &v, int dim);

	static void physical_flux(const std::vector<safe_real> &U, std::vector<safe_real> &F, int dim, safe_real &am, safe_real &ap, std::array<safe_real, NDIM> &x,
			std::array<safe_real, NDIM> &vg);

	template<int INX>
	static void post_process(hydro::state_type &U, safe_real dx);

	template<int INX>
	static void source(hydro::state_type &dudt, const hydro::state_type &U, const hydro::flux_type &F, const hydro::x_type X, safe_real omega, safe_real dx);

	/*** Reconstruct uses this - GPUize****/
	template<int INX>
	static const hydro::state_type& pre_recon(const hydro::state_type &U, const hydro::x_type X, safe_real omega, bool angmom);
	/*** Reconstruct uses this - GPUize****/
	template<int INX>
	static void post_recon( std::vector<std::vector<std::vector<safe_real>>> &Q, const hydro::x_type X,
			safe_real omega, bool angmom);
	template<int INX>
	using comp_type = hydro_computer<NDIM, INX, physics<NDIM>>;

	template<int INX>
	std::vector<typename comp_type<INX>::bc_type> initialize(test_type t, hydro::state_type &U, hydro::x_type &X);

	template<int INX>
	static void analytic_solution(test_type test, hydro::state_type &U, const hydro::x_type &X, safe_real time);

	template<int INX>
	static const std::vector<std::vector<bool>>& find_contact_discs( const hydro::state_type &U);

	static void set_n_species(int n);

	static void set_dual_energy_switches(safe_real one, safe_real two) {
		de_switch_1 = one;
		de_switch_2 = two;
	}

	static int get_angmom_index() {
		return sx_i;
	}

private:
	static int nf_;
	static int n_species_;
	static safe_real fgamma_;

};

template<int NDIM>
safe_real physics<NDIM>::de_switch_1 = 1e-3;

template<int NDIM>
safe_real physics<NDIM>::de_switch_2 = 1e-1;

template<int NDIM>
int physics<NDIM>::nf_ = (4 + NDIM + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2))) + physics<NDIM>::n_species_;

template<int NDIM>
int physics<NDIM>::n_species_ = 5;

template<int NDIM>
safe_real physics<NDIM>::fgamma_ = 7. / 5.;

#endif /* OCTOTIGER_UNITIGER_PHYSICS_HPP_ */
