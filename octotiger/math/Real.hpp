/*
 * Real.hpp
 *
 *  Created on: Feb 20, 2026
 *      Author: dmarce1
 */

#ifndef INCLUDE_REAL_HPP_
#define INCLUDE_REAL_HPP_

#include <cmath>
#include <limits>

#include "Integer.hpp"

using Real = double;

inline constexpr double operator"" _R(long double x) {
	return static_cast<double>(x);
}

inline constexpr Real operator"" _R(long long unsigned x) {
	return static_cast<double>(x);
}

inline constexpr double eps_R = std::numeric_limits<double>::epsilon();
inline constexpr double huge_R = std::numeric_limits<double>::max();
inline constexpr double inf_R = std::numeric_limits<double>::infinity();
inline constexpr double NaN_R = std::numeric_limits<double>::signaling_NaN();
inline constexpr double tiny_R = std::numeric_limits<double>::min();


using HiPrec = long double;

inline constexpr long double operator"" _Hi(long double x) {
	return x;
}

inline constexpr HiPrec operator"" _Hi(long long unsigned x) {
	return static_cast<long double>(x);
}

inline constexpr long double eps_Hi = std::numeric_limits<long double>::epsilon();
inline constexpr long double huge_Hi = std::numeric_limits<long double>::max();
inline constexpr long double inf_Hi = std::numeric_limits<long double>::infinity();
inline constexpr long double NaN_Hi = std::numeric_limits<long double>::signaling_NaN();
inline constexpr long double tiny_Hi = std::numeric_limits<long double>::min();

 // namespace HighPrecisionReals

#endif /* INCLUDE_REAL_HPP_ */
