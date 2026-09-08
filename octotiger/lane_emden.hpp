//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef LANE_EMDEN_HPP_
#define LANE_EMDEN_HPP_

#include "octotiger/math/Real.hpp"

Real lane_emden(Real r0, Real dr, Real n, Real* m_enc = nullptr);
Real wd_radius(double mass, double* rho0);
Real binary_separation( Real accretor_mass, Real donor_mass, Real donor_radius, Real fill_factor = 1.0);


Real find_V(Real q);
#endif /* LANE_EMDEN_HPP_ */
