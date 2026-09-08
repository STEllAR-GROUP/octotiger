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
#include <numbers>

#include "Integer.hpp"


using Real = double;

inline constexpr double operator""_R(long double x) {
	return static_cast<double>(x);
}

inline constexpr Real operator""_R(long long unsigned x) {
	return static_cast<Real>(x);
}

inline constexpr Real eps_R = std::numeric_limits<Real>::epsilon();
inline constexpr Real huge_R = std::numeric_limits<Real>::max();
//inline constexpr Real inf_R = std::numeric_limits<Real>::infinity();
inline constexpr Real NaN_R = std::numeric_limits<Real>::signaling_NaN();
inline constexpr Real tiny_R = std::numeric_limits<Real>::min();
inline constexpr Real pi_R = std::numbers::pi_v<Real>;


#endif /* INCLUDE_REAL_HPP_ */
