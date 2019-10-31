//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef __SAFE_REAL_HPPP
#define __SAFE_REAL_HPPP

#include <cmath>
////
//struct safe_real {
//	inline constexpr safe_real() :
//			r_(std::numeric_limits<double>::signaling_NaN()) {
//	}
//	inline constexpr safe_real(double r) :
//			r_(r) {
//	}
//	inline safe_real& operator +=( const safe_real& other ) {
//		r_ += other.r_;
//		return *this;
//	}
//	inline safe_real& operator -=( const safe_real& other ) {
//		r_ -= other.r_;
//		return *this;
//	}
//	inline safe_real& operator *=( const safe_real& other ) {
//		r_ *= other.r_;
//		return *this;
//	}
//	inline safe_real& operator /=( const safe_real& other ) {
//		r_ /= other.r_;
//		return *this;
//	}
//	inline safe_real operator-() const {
//		return safe_real(-r_);
//	}
//	inline safe_real operator+() const {
//		return *this;
//	}
//	inline constexpr operator double() const {
//		return r_;
//	}
//
//	template<class Arc>
//	void serialize(Arc& a, unsigned i) {
//		a & r_;
//	}
//private:
//	double r_;
//};

using safe_real = double;

#endif
