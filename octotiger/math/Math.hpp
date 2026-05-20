/*
 * Math.hpp
 *
 *  Created on: Feb 4, 2026
 *      Author: dmarce1
 */

#ifndef MATH12_HPP_
#define MATH12_HPP_

#include "./Debug.hpp"
#include "./Integer.hpp"
#include "./Real.hpp"

#include <cmath>

//template <typename T>
//constexpr auto sqr(T const &v) {
//	return v * v;
//}

template <typename T>
constexpr auto inv(T const &v) {
	ASSERT_NONZERO(v);
	return 1_R / v;
}

template <typename T>
constexpr T pow(T x, Integer n) {
	if (n == 0_I) return T(1_R);
	if (n < 0_I) return pow(inv(x), -n);
	x = (n < 0_I) ? inv(x) : x;
	n = (n < 0_I) ? -n : +n;
	T z = T(1_I);
	T y = x;
	while (true) {
		z = bool(n & 1_I) ? z * y : z;
		n >>= 1_I;
		if (!n) return z;
		y *= y;
	}
	return z;
}

constexpr Integer fact(Integer n) {
	if (n == 0_I) return 1_I;
	return n * fact(n - 1_I);
}

constexpr Integer fallingPochhammer(auto x, Integer n) {
	if (n == 0) return 1;
	return (x + 1 - n) * fallingPochhammer(x, n - 1);
}

constexpr Integer dfact(Integer n) {
	if (n <= 0_I) return 1_I;
	return n * dfact(n - 2_I);
}

constexpr Integer binco(Integer n, Integer k) {
	if (k == 0_I || k == n) return 1_I;
	if (k < 0_I || k > n) return 0_I;
	if (n < 2_I * k) return binco(n, n - k);
	Integer y = 1_I;
	Integer z = 1_I;
	for (Integer l = 0_I; l < k; l++) {
		z *= l + 1_I;
		y *= n - l;
	}
	return y / z;
}

template <Integer N, Integer M>
consteval auto roundUp() {
	return ((N + M - 1_I) / M) * M;
}

constexpr auto roundUp(Integer n, Integer m) {
	return ((n + m - 1_I) / m) * m;
}

template <Integer N, Integer M>
consteval auto roundDown() {
	return (N / M) * M;
}

constexpr auto roundDown(Integer n, Integer m) {
	return (n / m) * m;
}

template <Integer W>
constexpr auto ilog2() {
	if constexpr (W) {
		return 1_I + ilog2<(W >> 1_I)>();
	} else {
		return 0_I;
	}
}

constexpr Real root(Real x, Integer n) {
	int exp;
	Real y, y0, a;
	std::frexp(x, &exp);
	y = 1_I << Integer(std::abs(exp) / Real(n) + 0.5_R);
	if (x < 1_R) y = inv(y);
	a = inv(n);
	do {
		y0 = y;
		y *= 1_I + a * (x * pow(y, -n) - 1_I);
	} while (std::abs(y - y0) > n * std::abs(y * eps_R));
	return y;
}

template <typename T>
constexpr signed char sign(T const &v) {
	if (v > T(0)) return +1;
	if (v < T(0)) return -1;
	return 0;
}
#endif /* MATH_HPP_ */
