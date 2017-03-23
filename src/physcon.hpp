/*
 * physcon.hpp
 *
 *  Created on: Mar 10, 2017
 *      Author: dminacore
 */

#ifndef SRC_PHYSCON_HPP_
#define SRC_PHYSCON_HPP_

#include "defs.hpp"

struct physcon_t {
	real A;
	real G;
	real B;
	real c;
	real sigma;
	real kb;
	real mh;
	std::array<real, NSPECIES> _A;
	std::array<real, NSPECIES> _Z;
};

#ifndef __NPHYSCON__
extern physcon_t physcon;
#endif

real mean_ion_weight(const std::array<real,NSPECIES> species);
void set_AB(real, real);

#endif /* SRC_PHYSCON_HPP_ */
