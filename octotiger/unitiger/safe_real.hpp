#ifndef __SAFE_REAL_HPPP
#define __SAFE_REAL_HPPP

#include <cmath>

struct safe_real {
	inline constexpr safe_real() :
			r_(SNAN) {
	}
	inline constexpr safe_real(double r) :
			r_(r) {
	}
	inline safe_real& operator +=( const safe_real& other ) {
		r_ += other.r_;
		return *this;
	}
	inline safe_real& operator -=( const safe_real& other ) {
		r_ -= other.r_;
		return *this;
	}
	inline safe_real& operator *=( const safe_real& other ) {
		r_ *= other.r_;
		return *this;
	}
	inline safe_real& operator /=( const safe_real& other ) {
		r_ /= other.r_;
		return *this;
	}
	inline operator double() const {
		return r_;
	}
private:
	double r_;
};

#endif
