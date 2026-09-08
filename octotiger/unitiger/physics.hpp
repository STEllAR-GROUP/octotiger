//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER_UNITIGER_PHYSICS_HPP_
#define OCTOTIGER_UNITIGER_PHYSICS_HPP_

#include "octotiger/math/Real.hpp"
#include "octotiger/test_problems/blast.hpp"
#include "octotiger/test_problems/exact_sod.hpp"

template<int NDIM>
struct physics {
	static constexpr char const *field_names3[] = { "rho", "egas", "tau", "pot", "sx", "sy", "sz", "zx", "zy", "zz", "spc_1", "spc_2", "spc_3", "spc_4", "spc_5" };
	static constexpr char const *field_names2[] = { "rho", "egas", "tau", "pot", "sx", "sy", "zz", "spc_1", "spc_2", "spc_3", "spc_4", "spc_5" };
	static constexpr char const *field_names1[] = { "rho", "egas", "tau", "pot", "sx", "spc_1", "spc_2", "spc_3", "spc_4", "spc_5" };
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
    // std::pow is not constexpr in device code! Workaround with ternary operator:
	//static constexpr int spc_i = 4 + NDIM + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2));
    static constexpr int spc_i = 4 + NDIM + (NDIM == 3 ? 3 : (NDIM == 2 ? 1 : 0));
	static Real de_switch_1;
	static Real de_switch_2;

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

	static void set_fgamma(Real fg);

	static void to_prim(std::vector<Real> u, Real &p, Real &v, Real& c, int dim);
	// static void to_prim_experimental(const double rho, const double sx, const double tau, const double egas, Real &p, Real &v, Real& c, int dim);
	static void to_prim_experimental(const std::vector<double> &u, double &p, double &v, double &cs, const int dim) noexcept;

	static void enforce_outflows(hydro::state_type &U, const hydro::x_type &X, int face) {

	}

	template<int INX>
	static void physical_flux(const std::vector<Real> &U, std::vector<Real> &F, int dim, Real &am, Real &ap, std::array<Real, NDIM> &x,
			std::array<Real, NDIM> &vg);
	template<int INX>
	static void physical_flux_experimental(const std::vector<Real> &U, std::vector<Real> &F, int dim, Real &am, Real &ap, std::array<Real, NDIM> &x,
			std::array<Real, NDIM> &vg);

	template<int INX>
	static void post_process(hydro::state_type &U, const hydro::x_type& X, Real dx);

	static void set_degenerate_eos(Real, Real);
        static void set_ideal_plus_rad_eos(Real, Real, Real, int, bool, Real);

	template<int INX>
	static void source(hydro::state_type &dudt, const hydro::state_type &U, const hydro::flux_type &F, const hydro::x_type X, Real omega, Real dx);

	/*** Reconstruct uses this - GPUize****/
	template<int INX>
	static const hydro::state_type& pre_recon(const hydro::state_type &U, const hydro::x_type X, Real omega, bool angmom);
	/*** Reconstruct uses this - GPUize****/
	template<int INX>
	static void post_recon(std::vector<std::vector<std::vector<Real>>> &Q, const hydro::x_type X, Real omega, bool angmom);
	template<int INX>
	using comp_type = hydro_computer<NDIM, INX, physics<NDIM>>;

	template<int INX>
	std::vector<typename comp_type<INX>::bc_type> initialize(test_type t, hydro::state_type &U, hydro::x_type &X);

	template<int INX>
	static void analytic_solution(test_type test, hydro::state_type &U, const hydro::x_type &X, Real time);

	template<int INX>
	static const std::vector<std::vector<double>>& find_contact_discs(const hydro::state_type &U);

	static void set_n_species(int n);
	static int get_n_species() {
    return n_species_;
  }

	static void update_n_field();

	static Real get_mu_average(std::vector<Real> u);

	static void set_mu(std::vector<Real>, std::vector<Real>);

	static void set_dual_energy_switches(Real one, Real two);

	static void set_central_force(Real GM) {
		GM_ = GM;
	}
	static int get_angmom_index() {
		return sx_i;
	}

	template<int INX>
	static void enforce_outflow(hydro::state_type &U, int dim, int dir);

public:
	static Real rho_sink_radius_;
	static Real rho_sink_floor_;
	static int nf_;
	static int n_species_;
	static Real fgamma_;
	static Real A_;
	static Real B_;
	static Real IPR_IC_;
	static Real IPR_RC_;
	static Real IPR_NR_tol;
	static int IPR_NR_maxiter;
	static bool IPR_test;
	static Real IPR_eint_floor;
	static std::vector<Real> mu_;
	static Real GM_;
	static Real deg_pres(Real x);
	static Real pres_IPR(Real t, const Real a0, const Real a1, const Real a2, int &iter_num, const Real tol = 1.48e-08, const int max_iter = 50);
	static Real pres_IPR_ft(Real t, const Real a0, const Real a1, const Real a2);
	static Real pres_IPR_dft(Real t, const Real a0, const Real a1, const Real a2);

};

//definitions of the declarations (and initializations) of the static constexpr variables
template<int NDIM>
constexpr char const * physics<NDIM>::field_names1[];
template<int NDIM>
constexpr char const * physics<NDIM>::field_names2[];
template<int NDIM>
constexpr char const * physics<NDIM>::field_names3[];

template<int NDIM>
Real physics<NDIM>::rho_sink_radius_ = 0.0;

template<int NDIM>
Real physics<NDIM>::rho_sink_floor_ = 0.0;

template<int NDIM>
Real physics<NDIM>::GM_ = 0.0;

template<int NDIM>
Real physics<NDIM>::A_ = 0.0;

template<int NDIM>
Real physics<NDIM>::B_ = 1.0;

// IPR eos definitions
template<int NDIM>
Real physics<NDIM>::IPR_IC_ = 0.0;

template<int NDIM>
Real physics<NDIM>::IPR_RC_ = 0.0;

template<int NDIM>
Real physics<NDIM>::IPR_NR_tol = 1.48e-08;

template<int NDIM>
int physics<NDIM>::IPR_NR_maxiter = 50.0;

template<int NDIM>
Real physics<NDIM>::IPR_eint_floor = 0.0;

template<int NDIM>
bool physics<NDIM>::IPR_test = false;
//

template<int NDIM>
std::vector<Real> physics<NDIM>::mu_;

template<int NDIM>
Real physics<NDIM>::de_switch_1 = 1e-3;

template<int NDIM>
Real physics<NDIM>::de_switch_2 = 1e-1;

template<int NDIM>
//int physics<NDIM>::nf_ = (4 + NDIM + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2))) + physics<NDIM>::n_species_;
int physics<NDIM>::nf_ = (4 + NDIM + (NDIM == 1 ? 0 : (NDIM == 3 ? 3 : (NDIM == 2 ? 1 : 0)) )) + physics<NDIM>::n_species_;

template<int NDIM>
int physics<NDIM>::n_species_ = 5;

template<int NDIM>
Real physics<NDIM>::fgamma_ = 7. / 5.;

#endif /* OCTOTIGER_UNITIGER_PHYSICS_HPP_ */
