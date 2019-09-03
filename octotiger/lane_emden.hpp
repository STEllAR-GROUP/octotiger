//  Copyright (c) 2019
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef LANE_EMDEN_HPP_
#define LANE_EMDEN_HPP_

#include "octotiger/real.hpp"

real lane_emden(real r0, real dr, real* m_enc = nullptr);
real wd_radius(double mass, double* rho0);
real binary_separation( real accretor_mass, real donor_mass, real donor_radius, real fill_factor = 1.0);


real find_V(real q);
#endif /* LANE_EMDEN_HPP_ */
