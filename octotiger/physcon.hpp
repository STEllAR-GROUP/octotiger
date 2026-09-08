//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef SRC_PHYSCON444_HPP_
#define SRC_PHYSCON444_HPP_

#include "octotiger/options.hpp"
#include "octotiger/math/Real.hpp"

#include <hpx/serialization/traits/is_bitwise_serializable.hpp>

#include <initializer_list>
#include <vector>

template<class T = Real>
struct specie_state_t: public std::vector<T> {
	specie_state_t() :
			std::vector<T>(opts().n_species,0.0) {
	}
	specie_state_t(std::initializer_list<T> list ) : std::vector<T>(list) {

	}
};

struct physcon_t {
	Real A;
	Real G;
	Real B;
	Real kb;
	Real sigma;
	Real c;
	Real mh;
	Real h;
	template<class Arc>
	void serialize(Arc& arc, unsigned) {
		arc & A;
		arc & G;
		arc & B;
		arc & c;
		arc & sigma;
		arc & kb;
		arc & mh;
		arc & h;
	}
};

HPX_IS_BITWISE_SERIALIZABLE(physcon_t);

physcon_t& physcon();

void mean_ion_weight(const specie_state_t<> species, Real& mmw, Real& X, Real& Z);
void set_AB(Real, Real);

void set_units(Real m, Real l, Real t, Real k);
Real stellar_temp_from_rho_mu_s(Real rho, Real mu, Real s);
Real stellar_enthalpy_from_rho_mu_s(Real rho, Real mu, Real s);
Real stellar_rho_from_enthalpy_mu_s(Real h, Real mu, Real s);
Real find_T_rad_gas(Real p, Real rho, Real mu);
Real find_ei_rad_gas(Real p, Real rho, Real mu, Real gamma, Real &T);

OCTOTIGER_EXPORT void normalize_constants();

void these_units(Real& m, Real& l, Real& t, Real& k);

void rad_coupling_vars(Real rho, Real e, Real mmw, Real& bp, Real& kp, Real& dkpde, Real& dbde);

#endif /* SRC_PHYSCON_HPP_ */
