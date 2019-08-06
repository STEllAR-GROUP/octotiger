/*
 * util.hpp
 *
 *  Created on: Aug 5, 2019
 *      Author: dmarce1
 */

#ifndef OCTOTIGER_UNITIGER_UTI1L_HPP_
#define OCTOTIGER_UNITIGER_UTI1L_HPP_

template<int a, int b>
static constexpr int int_pow() {
	if constexpr (b == 0) {
		return 1;
	} else {
		return a * int_pow<a, b - 1>();
	}
}


static inline safe_real minmod(safe_real a, safe_real b) {
	return (std::copysign(0.5, a) + std::copysign(0.5, b)) * std::min(std::abs(a), std::abs(b));
}

static inline safe_real minmod_theta(safe_real a, safe_real b, safe_real c) {
	return minmod(c * minmod(a, b), 0.5 * (a + b));
}

#endif /* OCTOTIGER_UNITIGER_UTIL_HPP_ */
