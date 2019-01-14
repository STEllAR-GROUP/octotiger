/*
 * physcon.hpp
 *
 *  Created on: Mar 10, 2017
 *      Author: dminacore
 */

#ifndef SRC_PHYSCON444_HPP_
#define SRC_PHYSCON444_HPP_

#include "octotiger/real.hpp"
#include "octotiger/options.hpp"

#include <hpx/traits/is_bitwise_serializable.hpp>

#include <initializer_list>
#include <vector>

template<class T = real>
struct specie_state_t: public std::vector<T> {
	specie_state_t() :
			std::vector<T>(opts().n_species) {
	}
	specie_state_t(std::initializer_list<T> list ) : std::vector<T>(list) {

	}
};

struct physcon_t {
	real A;
	real G;
	real B;
	real kb;
	real sigma;
	real c;
	real mh;
	real h;
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

void mean_ion_weight(const specie_state_t<> species, real& mmw, real& X, real& Z);
void set_AB(real, real);

void set_units(real m, real l, real t, real k);
real stellar_temp_from_rho_mu_s(real rho, real mu, real s);
real stellar_enthalpy_from_rho_mu_s(real rho, real mu, real s);
real stellar_rho_from_enthalpy_mu_s(real h, real mu, real s);
real find_T_rad_gas(real p, real rho, real mu);


void these_units(real& m, real& l, real& t, real& k);

void rad_coupling_vars(real rho, real e, real mmw, real& bp, real& kp, real& dkpde, real& dbde);

#endif /* SRC_PHYSCON_HPP_ */
