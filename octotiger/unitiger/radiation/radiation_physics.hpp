//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER_UNITIGER_radiation_physics_HPP_
#define OCTOTIGER_UNITIGER_radiation_physics_HPP_

#include "octotiger/unitiger/safe_real.hpp"
#include "octotiger/test_problems/blast.hpp"
#include "octotiger/test_problems/exact_sod.hpp"

template<int NDIM>
struct radiation_physics {

	static constexpr char const *field_names3[] = { "er", "fx", "fy", "fz", "wx", "wy", "wz" };
	static constexpr char const *field_names2[] = { "er", "fx", "fy", "wz" };
	static constexpr char const *field_names1[] = { "er", "fx" };
	static constexpr int er_i = 0;
	static constexpr int fx_i = 1;
	static constexpr int fy_i = 2;
	static constexpr int fz_i = 3;
	static constexpr int wx_i = 1 + NDIM;
	static constexpr int wy_i = 1 + NDIM;
	static constexpr int wz_i = 1 + NDIM;
	static bool angmom_;

	enum test_type {
		CONTACT
	};

	static std::string get_test_type_string(test_type t) {
		switch (t) {
		case CONTACT:
			return "CONTACT";
		default:
			return "OCTOTIGER";
		}
	}

	static int field_count();

	static bool contact_field(int f) {
		return false;
	}

	template<int INX>
	static const std::vector<std::vector<double>>& find_contact_discs(const hydro::state_type &U) {
		static std::vector<std::vector<double>> a;
		return a;
	}

	template<int INX>
	static void physical_flux(const std::vector<safe_real> &U, std::vector<safe_real> &F, int dim, safe_real &am, safe_real &ap, std::array<safe_real, NDIM> &x,
			std::array<safe_real, NDIM> &vg);

	template<int INX>
	static void post_process(hydro::state_type &U, safe_real dx);

	template<int INX>
	static void source(hydro::state_type &dudt, const hydro::state_type &U, const hydro::flux_type &F, const hydro::x_type X, safe_real omega, safe_real dx);

	/*** Reconstruct uses this - GPUize****/
	template<int INX>
	static void pre_angmom(const hydro::state_type &U, const hydro::recon_type<NDIM> &Q, std::array<safe_real, cell_geometry<NDIM, INX>::NANGMOM> &Z,
			std::array<std::array<safe_real, cell_geometry<NDIM, INX>::NDIR>, NDIM> &S, int i, safe_real dx);

	template<int INX>
	static void enforce_outflows(hydro::state_type &U, const hydro::x_type &X, int face);

	/*** Reconstruct uses this - GPUize****/
	template<int INX>
	static void post_angmom(const hydro::state_type &U, const hydro::recon_type<NDIM> &Q, std::array<safe_real, cell_geometry<NDIM, INX>::NANGMOM> &Z,
			std::array<std::array<safe_real, cell_geometry<NDIM, INX>::NDIR>, NDIM> &S, int i, safe_real dx);

	/*** Reconstruct uses this - GPUize****/
	template<int INX>
	static const hydro::state_type& pre_recon(const hydro::state_type &U, const hydro::x_type X, safe_real omega, bool angmom);
	/*** Reconstruct uses this - GPUize****/
	template<int INX>
	static void post_recon(std::vector<std::vector<std::vector<safe_real>>> &Q, const hydro::x_type X, safe_real omega, bool angmom);
	template<int INX>
	using comp_type = hydro_computer<NDIM, INX, radiation_physics<NDIM>>;

	template<int INX>
	std::vector<typename comp_type<INX>::bc_type> initialize(test_type t, hydro::state_type &U, hydro::x_type &X);

	template<int INX>
	static void analytic_solution(test_type test, hydro::state_type &U, const hydro::x_type &X, safe_real time);

	static int get_angmom_index() {
		return sx_i;
	}

	static void set_clight(safe_real r) {
		clight = r;
	}

private:
	static int nf_;
	static safe_real clight;

};

template<int NDIM>
safe_real radiation_physics<NDIM>::clight = 1.0;

template<int NDIM>
int radiation_physics<NDIM>::nf_ = (1 + NDIM + (NDIM == 1 ? 0 : (NDIM == 3 ? 3 : (NDIM == 2 ? 1 : 0)) ));
//int radiation_physics<NDIM>::nf_ = (1 + NDIM + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2)));

#endif /* OCTOTIGER_UNITIGER_radiation_physics_HPP_ */
