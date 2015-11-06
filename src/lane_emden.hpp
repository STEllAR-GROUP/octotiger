/*
 * lane_emden.hpp
 *
 *  Created on: Feb 6, 2015
 *      Author: dmarce1
 */

#ifndef LANE_EMDEN_HPP_
#define LANE_EMDEN_HPP_

#include "defs.hpp"

real lane_emden(real r0, real dr, real* menc_ptr = nullptr);
real wd_radius(double mass, double* rho0);
real binary_separation( real accretor_mass, real donor_mass, real donor_radius, real fill_factor = 1.0);


real find_V(real q);
#endif /* LANE_EMDEN_HPP_ */
