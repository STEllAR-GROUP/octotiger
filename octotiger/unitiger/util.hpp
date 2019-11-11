//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER_UNITIGER_UTI1L_HPP_
#define OCTOTIGER_UNITIGER_UTI1L_HPP_

template<int NDIM, int NX>
std::array<int, NDIM> index_to_dims(int i) {
	std::array<int, NDIM> dims;
	for (int j = 0; j < NDIM; j++) {
		dims[NDIM - 1 - j] = i % NX;
		i /= NX;
	}
	return dims;
}

template<int a, int b>
static constexpr int int_pow() {
	int c = 1;
	for (int i = 0; i < b; i++) {
		c *= a;
	}
	return a;
}

static inline safe_real minmod(safe_real a, safe_real b) {
	return (std::copysign(0.5, a) + std::copysign(0.5, b)) * std::min(std::abs(a), std::abs(b));
}

static inline safe_real minmod_theta(safe_real a, safe_real b, safe_real c) {
	return minmod(c * minmod(a, b), 0.5 * (a + b));
}

#endif /* OCTOTIGER_UNITIGER_UTIL_HPP_ */
