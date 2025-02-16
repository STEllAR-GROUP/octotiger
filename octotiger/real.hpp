//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef REAL_HPP_
#define REAL_HPP_

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

#ifndef NDEBUG
#define CHECK true
#else
#define CHECK false
#endif

#ifndef __CUDA_ARCH__
struct Real {
	using Type = double;
	Real() {
		if constexpr (CHECK) {
			value = std::numeric_limits<Type>::signaling_NaN();
		}
	}
	constexpr explicit Real(double a) :
			value(Type(a)) {
	}
	Real& operator=(Real const &a) {
		value = a.value;
		return *this;
	}
	constexpr operator Type() const {
		return value;
	}
	constexpr Real operator+() const {
		debug_check(*this);
		return *this;
	}
	constexpr Real operator-() const {
		debug_check(*this);
		return Real(-value);
	}
	constexpr Real operator+(Real const &a) const {
		debug_check(a);
		debug_check(*this);
		return Real(value + a.value);
	}
	constexpr Real operator-(Real const &a) const {
		debug_check(a);
		debug_check(*this);
		return Real(value - a.value);
	}
	constexpr Real operator*(Real const &a) const {
		debug_check(a);
		debug_check(*this);
		return Real(value * a.value);
	}
	const Real operator/(Real const &a) const {
		Real result;
		zero_check(a);
		debug_check(a);
		debug_check(*this);
		result.value = value / a.value;
		return result;
	}
	Real& operator+=(Real const &a) {
		debug_check(a);
		debug_check(*this);
		*this = *this + a;
		return *this;
	}
	Real& operator-=(Real const &a) {
		debug_check(a);
		debug_check(*this);
		*this = *this - a;
		return *this;
	}
	Real& operator*=(Real const &a) {
		debug_check(a);
		debug_check(*this);
		*this = *this * a;
		return *this;
	}
	Real& operator/=(Real const &a) {
		zero_check(a);
		debug_check(a);
		*this = *this / a;
		return *this;
	}
	bool operator==(Real const &a) const {
		debug_check(a);
		debug_check(*this);
		return value == a.value;
	}
	bool operator!=(Real const &a) const {
		debug_check(a);
		debug_check(*this);
		return value != a.value;
	}
	bool operator<=(Real const &a) const {
		debug_check(a);
		debug_check(*this);
		return value <= a.value;
	}
	bool operator>=(Real const &a) const {
		debug_check(a);
		debug_check(*this);
		return value >= a.value;
	}
	bool operator<(Real const &a) const {
		debug_check(a);
		debug_check(*this);
		return value < a.value;
	}
	bool operator>(Real const &a) const {
		debug_check(a);
		debug_check(*this);
		return value > a.value;
	}
	static Real zero() {
		Real z;
		z.value = Type(0);
		return z;
	}
	static constexpr Real tiny() {
		return Real(std::numeric_limits<double>::min());
	}
	static constexpr Real epsilon() {
		return Real(std::numeric_limits<double>::epsilon());
	}
	friend Real abs(Real a) {
		debug_check(a);
		a.value = std::fabs(a.value);
		return a;
	}
	friend Real expm1(Real a) {
		debug_check(a);
		a.value = std::expm1(a.value);
		return a;
	}
	friend Real exp(Real a) {
		debug_check(a);
		a.value = std::exp(a.value);
		return a;
	}
	friend Real erf(Real a) {
		debug_check(a);
		a.value = std::erf(a.value);
		return a;
	}
	friend Real log(Real a) {
		debug_check(a);
		a.value = std::log(a.value);
		return a;
	}
	friend Real cos(Real a) {
		debug_check(a);
		a.value = std::cos(a.value);
		return a;
	}
	friend Real sin(Real a) {
		debug_check(a);
		a.value = std::sin(a.value);
		return a;
	}
	friend Real acos(Real a) {
		debug_check(a);
		range_check(-Real(1), a, Real(1));
		a.value = std::acos(a.value);
		return a;
	}
	friend Real asin(Real a) {
		debug_check(a);
		range_check(-Real(1), a, Real(1));
		a.value = std::asin(a.value);
		return a;
	}
	friend Real sqrt(Real a) {
		nonneg_check(a);
		debug_check(a);
		a.value = std::sqrt(a.value);
		return a;
	}
	friend Real pow(Real a, Real b) {
		nonneg_check(a);
		debug_check(a);
		debug_check(b);
		a.value = std::pow(a.value, b.value);
		return a;
	}
	friend Real pow(Real x, int n) {
		nonneg_check(x);
		debug_check(x);
		if (n < 0) {
			return Real(1) / pow(x, -n);
		} else {
			Real y = Real(1);
			Real xn = x;
			while (n) {
				if (n & 1) {
					y *= xn;
				}
				xn *= xn;
				n >>= 1;
			}
			return y;
		}
	}
	friend Real max(Real a, Real b) {
		debug_check(a);
		debug_check(b);
		a.value = std::max(a.value, b.value);
		return a;
	}
	friend Real min(Real a, Real b) {
		debug_check(a);
		debug_check(b);
		a.value = std::min(a.value, b.value);
		return a;
	}
	friend Real copysign(Real a, Real b) {
		debug_check(a);
		debug_check(b);
		a.value = std::copysign(a.value, b.value);
		return a;
	}
	friend std::string to_string(Real r) {
		return std::to_string(r.value);
	}
	template<class A>
	void serialize(A &&arc, unsigned) {
		arc & value;
	}
private:
	Type value;
	static void nonneg_check(Real a) {
		if constexpr (CHECK) {
			if (a.value < 0.0) {
				std::string errorString = "FATAL ERROR: Illegal operation on negative number.\n";
				std::cout << errorString;
				abort();
			}
		}
	}
	static void zero_check(Real a) {
		if constexpr (CHECK) {
			if (a.value == 0.0) {
				std::string errorString = "FATAL ERROR: Divide by zero.\n";
				std::cout << errorString;
				abort();
			}
		}
	}
	static void debug_check(Real a) {
		if constexpr (CHECK) {
			if (!std::isfinite(a.value)) {
				std::string errorString = "FATAL ERROR: Operation on NaN\n";
				std::cout << errorString;
				abort();
			}
		}
	}
	static void range_check(Real a, Real b, Real c) {
		if constexpr (CHECK) {
			if ((b < a) || (b > c)) {
				std::string errorString = "FATAL ERROR: Operation on NaN\n";
				std::cout << errorString;
				abort();
			}
		}
	}
};
#endif

using real_type = double;
using real = double;

#endif /* REAL_HPP_ */
