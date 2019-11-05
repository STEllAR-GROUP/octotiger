/*
 * ztwd.hpp
 *
 *  Created on: Nov 4, 2019
 *      Author: dmarce1
 */

#ifndef OCTOTIGER_ZTWD_HPP_
#define OCTOTIGER_ZTWD_HPP_

#include <octotiger/safe_math.hpp>

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



#endif /* OCTOTIGER_ZTWD_HPP_ */
