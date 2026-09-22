/*
 * Real.hpp
 *
 *  Created on: Feb 20, 2026
 *      Author: dmarce1
 */

#pragma once

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

inline constexpr Real epsilonR = std::numeric_limits<Real>::epsilon();
inline constexpr Real hugeR = std::numeric_limits<Real>::max();
// inline constexpr Real infinityR = std::numeric_limits<Real>::infinity();
inline constexpr Real nanR = std::numeric_limits<Real>::signaling_NaN();
inline constexpr Real tinyR = std::numeric_limits<Real>::min();
inline constexpr Real piR = std::numbers::pi_v<Real>;
