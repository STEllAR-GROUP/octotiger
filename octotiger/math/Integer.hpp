/*
 * Integer.hpp
 *
 *  Created on: Feb 20, 2026
 *      Author: dmarce1
 */

#ifndef INCLUDE_INTEGER_HPP_
#define INCLUDE_INTEGER_HPP_

#include <cstdlib>
#include <type_traits>

using Integer = std::make_signed_t<size_t>;

inline constexpr Integer operator"" _I(long long unsigned i) {
	return Integer(i);
}

inline constexpr Integer pow(Integer x, Integer n) {
	if (n == 0_I) return 1_I;
	Integer z = 1_I;
	Integer y = x;
	while (n != 1_I) {
		if ((n & 1_I) != 0_I) {
			z *= y;
		}
		n >>= 1_I;
		y *= y;
	}
	return z * y;
}

inline constexpr Integer round(Integer x, Integer n) {
	return n * (1 + (x - 1) / n);
}

#endif /* INCLUDE_INTEGER_HPP_ */
