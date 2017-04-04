/*
 * physcon.hpp
 *
 *  Created on: Mar 10, 2017
 *      Author: dminacore
 */

#ifndef SRC_PHYSCON_HPP_
#define SRC_PHYSCON_HPP_

#include "defs.hpp"
#include <hpx/traits/is_bitwise_serializable.hpp>

struct physcon_t {
	real A;
	real G;
	real B;
	real kb;
	real sigma;
	real c;
	real mh;
	real h;
	std::array<real, NSPECIES> _A;
	std::array<real, NSPECIES> _Z;
	template<class Arc>
	void serialize( Arc& arc, unsigned ) {
		arc & A;
		arc & G;
		arc & B;
		arc & c;
		arc & sigma;
		arc & kb;
		arc & mh;
		arc & h;
		arc & _A;
		arc & _Z;
	}
};

HPX_IS_BITWISE_SERIALIZABLE(physcon_t);

#ifndef __NPHYSCON__
extern physcon_t physcon;
#endif

real mean_ion_weight(const std::array<real,NSPECIES> species);
void set_AB(real, real);

real stellar_temp_from_rho_mu_s(real rho, real mu, real s);
real stellar_enthalpy_from_rho_mu_s(real rho, real mu, real s);
real stellar_rho_from_enthalpy_mu_s(real h, real mu, real s);
real temperature(real rho, real e, real mmw);
real kappa_p(real rho, real e, real mmw);
real dkappa_p_de(real rho, real e, real mmw);
real kappa_R(real rho, real e, real mmw);
real B_p(real rho, real e, real mmw);
real dB_p_de(real rho, real e, real mmw);
real find_T_rad_gas(real p, real rho, real mu);

void rad_coupling_vars( real rho, real e, real mmw, real& bp, real& kp, real& dkpde, real& dbde);


#endif /* SRC_PHYSCON_HPP_ */
