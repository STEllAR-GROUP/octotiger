//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef ROE_HPP_
#define ROE_HPP_

#include "octotiger/defs.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/real.hpp"
#include "octotiger/safe_math.hpp"
#include "octotiger/space_vector.hpp"

#include <algorithm>
#include <vector>


#define de_switch1 (opts().dual_energy_sw1)
#define de_switch2 (opts().dual_energy_sw2)

real roe_fluxes(hydro_state_t<std::vector<real>>& F, hydro_state_t<std::vector<real>>& UL,
		hydro_state_t<std::vector<real>>& UR,  const std::vector<space_vector>& X, real omega, integer dimension, real dx);


inline real ztwd_pressure(real d, real A = physcon().A, real B = physcon().B) {
    const real x = POWER(d / B, 1.0 / 3.0);
    real p;
    if (x < 0.01) {
        p = 1.6 * A * SQRT(x) * cube(x);
    } else {
        p = A * (x * (2.0 * SQRT(x) - 3.0) * SQRT(SQRT(x) + 1.0) + 3.0 * asinh(x));
    }
    return p;
}

inline real ztwd_enthalpy(real d, real A = physcon().A, real B = physcon().B) {
#ifndef NDEBUG
	if( d < 0.0 ) {
		printf( "d = %e in ztwd_enthalpy\n", d);
		abort();
	}
#endif
    const real x = pow(d / B, 1.0 / 3.0);
    real h;
    if (x < 0.01) {
        h = 4.0 * A / B * sqr(x);
    } else {
        h = 8.0 * A / B * (sqrt(sqr(x) + 1.0) - 1.0);
    }
    return h;
}

OCTOTIGER_FORCEINLINE real ztwd_energy(real d, real A = physcon().A, real B = physcon().B) {
    return std::max(ztwd_enthalpy(d) * d - ztwd_pressure(d), real(0));
}

real ztwd_sound_speed(real d, real ei);

#endif /* ROE_HPP_ */
