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
#include "./Sign.hpp"

#include <cmath>

template <typename T>
constexpr auto inv(T const &v) {
	ASSERT_NONZERO(v);
	return T(1) / v;
}

using std::pow;

template <typename T>
constexpr T pow(T x, std::integral auto n) {
	if (n == 0_I) return T(1);
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

inline constexpr Integer nonepow(Integer n) {
	return 1 - ((n & 1) << 1);
}

inline constexpr Integer twopow(Integer n) {
	return (1_I << n);
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

template <typename T>
constexpr T root(T x, Integer n) {
	using std::abs;
	constexpr auto zero = T(0);
	constexpr auto one = T(1);
	constexpr auto two = T(2);
	constexpr auto half = T(0.5);
	constexpr auto eps = std::numeric_limits<T>::epsilon();
	if ((n < 0) && ((-n) & 1)) return -root(x, (-n));
	T y = one;
	T a = inv(T(n));
	if (x == zero) return zero;
	while (pow(y, n) < x) {
		y *= two;
	}
	while (pow(y, n) > x && y > one) {
		y *= half;
	}
	T y0;
	do {
		y0 = y;
		y *= one + a * (x * pow(y, -n) - one);
	} while (abs(y - y0) > T(n) * abs(y * eps));
	return y;
}

constexpr auto sqr(auto v) {
	return v * v;
}

constexpr auto cube(auto v) {
	return v * sqr(v);
}

template <std::integral Dividend, std::integral Divisor>
constexpr auto divEuclidean(Dividend dividend, Divisor _divisor) {
	using std::abs;
	using std::div;
	auto const divisor = static_cast<Dividend>(_divisor);
	auto qr = div(dividend, divisor);
	if (qr.rem < 0) {
		qr.rem += abs(divisor);
		qr.quot -= sign(divisor);
	}
	return qr;
}

// template <class D, class B, class S = Real>
// struct inheritArithmetic {
//	friend D operator+(D const &a)
//		requires requires(B const &x) { +x; }
//	{
//		return D(+static_cast<B const &>(a));
//	}
//
//	friend D operator-(D const &a)
//		requires requires(B const &x) { -x; }
//	{
//		return D(-static_cast<B const &>(a));
//	}
//
//	friend D operator+(D const &a, D const &b)
//		requires requires(B const &x, B const &y) { x + y; }
//	{
//		return D(static_cast<B const &>(a) + static_cast<B const &>(b));
//	}
//
//	friend D operator-(D const &a, D const &b)
//		requires requires(B const &x, B const &y) { x - y; }
//	{
//		return D(static_cast<B const &>(a) - static_cast<B const &>(b));
//	}
//
//	friend D operator*(D const &a, D const &b)
//		requires requires(B const &x, B const &y) { x *y; }
//	{
//		return D(static_cast<B const &>(a) * static_cast<B const &>(b));
//	}
//
//	friend D operator/(D const &a, D const &b)
//		requires requires(B const &x, B const &y) { x / y; }
//	{
//		return D(static_cast<B const &>(a) / static_cast<B const &>(b));
//	}
//
//
//	friend D operator+(D const &a, S const &s)
//		requires requires(B const &x, S const &y) { x + y; }
//	{
//		return D(static_cast<B const &>(a) + s);
//	}
//
//
//	friend D operator-(D const &a, S const &s)
//		requires requires(B const &x, S const &y) { x - y; }
//	{
//		return D(static_cast<B const &>(a) - s);
//	}
//
//
//	friend D operator*(D const &a, S const &s)
//		requires requires(B const &x, S const &y) { x *y; }
//	{
//		return D(static_cast<B const &>(a) * s);
//	}
//
//
//	friend D operator/(D const &a, S const &s)
//		requires requires(B const &x, S const &y) { x / y; }
//	{
//		return D(static_cast<B const &>(a) / s);
//	}
//
//
//	friend D operator+(S const &s, D const &a)
//		requires requires(S const &x, B const &y) { x + y; }
//	{
//		return D(s + static_cast<B const &>(a));
//	}
//
//
//	friend D operator-(S const &s, D const &a)
//		requires requires(S const &x, B const &y) { x - y; }
//	{
//		return D(s - static_cast<B const &>(a));
//	}
//
//
//	friend D operator*(S const &s, D const &a)
//		requires requires(S const &x, B const &y) { x *y; }
//	{
//		return D(s * static_cast<B const &>(a));
//	}
//
//
//	friend D operator/(S const &s, D const &a)
//		requires requires(S const &x, B const &y) { x / y; }
//	{
//		return D(s / static_cast<B const &>(a));
//	}
// };

#endif /* MATH_HPP_ */
