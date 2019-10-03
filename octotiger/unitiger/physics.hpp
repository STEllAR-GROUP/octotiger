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
	static constexpr int zx_i = 4 + NDIM;
	static constexpr int zy_i = 5 + NDIM;
	static constexpr int zz_i = 6 + NDIM;
	static constexpr int spc_i = 4 + NDIM + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2));
	static bool angmom_;

	enum test_type {
		SOD, BLAST, KH, CONTACT
	};

        static std::string get_test_type_string(test_type t){
                switch (t)
                {
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

	static void to_prim(std::vector<safe_real> u, safe_real &p, safe_real &v, int dim, safe_real dx);

	static void physical_flux(const std::vector<safe_real> &U, std::vector<safe_real> &F, int dim, safe_real &am, safe_real &ap,
			std::array<safe_real, NDIM> &vg, safe_real dx);

	static void flux(const std::vector<safe_real> &UL, const std::vector<safe_real> &UR, const std::vector<safe_real> &UL0, const std::vector<safe_real> &UR0,
			std::vector<safe_real> &F, int dim, safe_real &am, safe_real &ap, std::array<safe_real, NDIM> &vg, safe_real dx);


	template<int INX>
	static void post_process(hydro::state_type &U, safe_real dx);

	template<int INX>
	static void source(hydro::state_type &dudt, const hydro::state_type &U, const hydro::flux_type &F, const hydro::x_type X, safe_real omega, safe_real dx);

	/*** Reconstruct uses this - GPUize****/
	template<int INX>
	static void pre_angmom(const hydro::state_type &U, const hydro::recon_type<NDIM> &Q, std::array<safe_real, cell_geometry<NDIM, INX>::NANGMOM> &Z,
			std::array<std::array<safe_real, cell_geometry<NDIM, INX>::NDIR>, NDIM> &S, int i, double dx);

	/*** Reconstruct uses this - GPUize****/
	template<int INX>
	static void post_angmom(const hydro::state_type &U, const hydro::recon_type<NDIM> &Q, std::array<safe_real, cell_geometry<NDIM, INX>::NANGMOM> &Z,
			std::array<std::array<safe_real, cell_geometry<NDIM, INX>::NDIR>, NDIM> &S, int i, double dx);

	/*** Reconstruct uses this - GPUize****/
	template<int INX>
	static const hydro::state_type pre_recon(const hydro::state_type &U, const hydro::x_type X, safe_real omega, bool angmom);
	/*** Reconstruct uses this - GPUize****/
	template<int INX>
	static std::vector<std::vector<std::vector<safe_real>>> post_recon(const std::vector<std::vector<std::vector<safe_real>>> &P, const hydro::x_type X, safe_real omega, bool angmom);
	template<int INX>
	using comp_type = hydro_computer<NDIM, INX>;

	template<int INX>
	std::vector<typename comp_type<INX>::bc_type> initialize(test_type t, hydro::state_type &U, hydro::x_type &X);

	template<int INX>
	static void analytic_solution(test_type test, hydro::state_type &U, const hydro::x_type &X, safe_real time);

	static void set_n_species(int n);

	static void set_angmom();

private:
	static int nf_;
	static int n_species_;
	static safe_real fgamma_;

};

template<int NDIM>
bool physics<NDIM>::angmom_ = false;

template<int NDIM>
int physics<NDIM>::nf_ = (4 + NDIM + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2))) + physics<NDIM>::n_species_;

template<int NDIM>
int physics<NDIM>::n_species_ = 5;

template<int NDIM>
safe_real physics<NDIM>::fgamma_ = 7. / 5.;

#endif /* OCTOTIGER_UNITIGER_PHYSICS_HPP_ */
