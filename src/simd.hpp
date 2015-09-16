/*
 * simd_vector.hpp
 *
 *  Created on: Jun 7, 2015
 *      Author: dmarce1
 */

#ifndef SIMD_VECTOR_HPP_
#define SIMD_VECTOR_HPP_


#include <cstdlib>
#include "immintrin.h"

#define SIMD_SIZE 2
#define USE_SIMD
const std::size_t simd_len = 4 * SIMD_SIZE;

#ifdef USE_SIMD

class simd_vector {
private:
	__m256d v[SIMD_SIZE];
public:
	simd_vector() = default;
	inline ~simd_vector() = default;
	simd_vector(const simd_vector&) = default;
	inline simd_vector(double d) {
		for( integer i = 0; i!= SIMD_SIZE; ++i) {
	//		_mm_empty();
			v[i] =_mm256_set_pd(d,d,d,d);
		}
	}
	inline double sum() const {
		double r = ZERO;
		for( integer i = 0; i != simd_len; ++i) {
			r += (*this)[i];
		}
		return r;
	}
	simd_vector(simd_vector&& other) {
		*this = std::move(other);
	}
	inline simd_vector& operator=(const simd_vector& other) = default;
	simd_vector& operator=(simd_vector&& other) {
		for( integer i = 0; i != SIMD_SIZE; ++i ) {
			v[i] = std::move(other.v[i]);
		}	
		return *this;
	}
	inline simd_vector operator+( const simd_vector& other) const {
		simd_vector r;
		for( integer i = 0; i != SIMD_SIZE; ++i) {
			//_mm_empty();
			r.v[i] = _mm256_add_pd(v[i], other.v[i]);
		}
		return r;
	}
	inline simd_vector operator-( const simd_vector& other) const {
		simd_vector r;
		for( integer i = 0; i != SIMD_SIZE; ++i) {
			//_mm_empty();
			r.v[i] = _mm256_sub_pd(v[i], other.v[i]);
		}
		return r;
	}
	inline simd_vector operator*( const simd_vector& other) const {
		simd_vector r;
		for( integer i = 0; i != SIMD_SIZE; ++i) {
			//_mm_empty();
			r.v[i] = _mm256_mul_pd(v[i], other.v[i]);
		}
		return r;
	}
	inline simd_vector operator/( const simd_vector& other) const {
		simd_vector r;
		for( integer i = 0; i != SIMD_SIZE; ++i) {
			//_mm_empty();
			r.v[i] = _mm256_div_pd(v[i], other.v[i]);
		}
		return r;
	}
	inline simd_vector operator+() const {
		return *this;
	}
	inline simd_vector operator-() const {
		return simd_vector(ZERO) - *this;
	}
	inline simd_vector& operator+=( const simd_vector& other) {
		*this = *this + other;
		return *this;
	}
	inline simd_vector& operator-=( const simd_vector& other) {
		*this = *this - other;
		return *this;
	}
	inline simd_vector& operator*=( const simd_vector& other) {
		*this = *this * other;
		return *this;
	}
	inline simd_vector& operator/=( const simd_vector& other) {
		*this = *this / other;
		return *this;
	}

	inline simd_vector operator*( double d) const {
		const simd_vector other = d;
		return other * *this;
	}
	inline simd_vector operator/(double d) const {
		const simd_vector other = ONE / d;
		return *this * other;
	}

	inline simd_vector operator*=( double d) {
		*this = *this * d;
		return *this;
	}
	inline simd_vector operator/=(double d) {
		*this = *this * (ONE/d);
		return *this;
	}
	inline double& operator[](std::size_t i) {
		double* a = reinterpret_cast<double*>(&v);
		return a[i];
	}
	inline double operator[](std::size_t i) const {
		const double* a = reinterpret_cast<const double*>(&v);
		return a[i];
	}

	real max() const {
		const real a = std::max((*this)[0],(*this)[1]);
		const real b = std::max((*this)[2],(*this)[3]);
		const real c = std::max((*this)[4],(*this)[5]);
		const real d = std::max((*this)[6],(*this)[7]);
		const real e = std::max(a,b);
		const real f = std::max(c,d);
		return std::max(e,f);
	}
	real min() const {
		const real a = std::min((*this)[0],(*this)[1]);
		const real b = std::min((*this)[2],(*this)[3]);
		const real c = std::min((*this)[4],(*this)[5]);
		const real d = std::min((*this)[6],(*this)[7]);
		const real e = std::min(a,b);
		const real f = std::min(c,d);
		return std::min(e,f);
	}
	friend simd_vector sqrt(const simd_vector&);
	friend simd_vector operator*(double, const simd_vector& other);
	friend simd_vector operator/(double, const simd_vector& other);
	friend simd_vector max(const simd_vector& a, const simd_vector& b);

};

inline simd_vector sqrt(const simd_vector& vec) {
	simd_vector r;
	for (integer i = 0; i != SIMD_SIZE; ++i) {
		//_mm_empty();
		r.v[i] = _mm256_sqrt_pd(vec.v[i]);
	}
	return r;
}

inline simd_vector operator*(double d, const simd_vector& other) {
	const simd_vector a = d;
	return a * other;
}

inline simd_vector operator/(double d, const simd_vector& other) {
	const simd_vector a = d;
	return a / other;
}

inline void simd_pack(simd_vector* dest, real* src, integer src_len, integer pos) {
	for (integer i = 0; i != src_len; ++i) {
		dest[i][pos] = src[i];
	}
}

inline void simd_unpack(real* dest, simd_vector* src, integer src_len, integer pos) {
	for (integer i = 0; i != src_len; ++i) {
		dest[i] = src[i][pos];
	}
}

inline simd_vector max(const simd_vector& a, const simd_vector& b) {
	simd_vector r;
	for (integer i = 0; i != SIMD_SIZE; ++i) {
		//_mm_empty();
		r.v[i] = _mm256_max_pd(a.v[i], b.v[i]);
	}
	return r;
}

inline simd_vector abs(const simd_vector& a) {
	return max(a, -a);
}

#else

class simd_vector {
private:
	real v[simd_len];
public:
	simd_vector() = default;
	inline ~simd_vector() = default;
	simd_vector(const simd_vector&) = default;
	inline simd_vector(double d) {
		for( integer i = 0; i != simd_len; ++i) {
			v[i] = d;
		}
	}
	inline double sum() const {
		double r = ZERO;
		for( integer i = 0; i != simd_len; ++i) {
			r += (*this)[i];
		}
		return r;
	}
	simd_vector(simd_vector&& other) {
		*this = std::move(other);
	}
	inline simd_vector& operator=(const simd_vector& other) = default;
	simd_vector& operator=(simd_vector&& other) {
		for( integer i = 0; i != simd_len; ++i) {
			v[i] = std::move(other.v[i]);
		}
		return *this;
	}

	inline simd_vector operator+( const simd_vector& other) const {
		simd_vector r;
		for( integer i = 0; i != simd_len; ++i) {
			r.v[i] = v[i] + other.v[i];
		}
		return r;
	}
	inline simd_vector operator-( const simd_vector& other) const {
		simd_vector r;
		for( integer i = 0; i != simd_len; ++i) {
			r.v[i] = v[i] - other.v[i];
		}
		return r;
	}
	inline simd_vector operator*( const simd_vector& other) const {
		simd_vector r;
		for( integer i = 0; i != simd_len; ++i) {
			r.v[i] = v[i] * other.v[i];
		}
		return r;
	}
	inline simd_vector operator/( const simd_vector& other) const {
		simd_vector r;
		for( integer i = 0; i != simd_len; ++i) {
			r.v[i] = v[i] / other.v[i];
		}
		return r;
	}
	inline simd_vector operator+() const {
		return *this;
	}
	inline simd_vector operator-() const {
		return simd_vector(ZERO) - *this;
	}
	inline simd_vector& operator+=( const simd_vector& other) {
		*this = *this + other;
		return *this;
	}
	inline simd_vector& operator-=( const simd_vector& other) {
		*this = *this - other;
		return *this;
	}
	inline simd_vector& operator*=( const simd_vector& other) {
		*this = *this * other;
		return *this;
	}
	inline simd_vector& operator/=( const simd_vector& other) {
		*this = *this / other;
		return *this;
	}

	inline simd_vector operator*( double d) const {
		const simd_vector other = d;
		return other * *this;
	}
	inline simd_vector operator/(double d) const {
		const simd_vector other = ONE / d;
		return *this * other;
	}

	inline simd_vector operator*=( double d) {
		*this = *this * d;
		return *this;
	}
	inline simd_vector operator/=(double d) {
		*this = *this * (ONE/d);
		return *this;
	}
	inline double& operator[](std::size_t i) {
		double* a = reinterpret_cast<double*>(&v);
		return a[i];
	}
	inline double operator[](std::size_t i) const {
		const double* a = reinterpret_cast<const double*>(&v);
		return a[i];
	}

	real max() const {
		const real a = std::max((*this)[0],(*this)[1]);
		const real b = std::max((*this)[2],(*this)[3]);
		const real c = std::max((*this)[4],(*this)[5]);
		const real d = std::max((*this)[6],(*this)[7]);
		const real e = std::max(a,b);
		const real f = std::max(c,d);
		return std::max(e,f);
	}
	real min() const {
		const real a = std::min((*this)[0],(*this)[1]);
		const real b = std::min((*this)[2],(*this)[3]);
		const real c = std::min((*this)[4],(*this)[5]);
		const real d = std::min((*this)[6],(*this)[7]);
		const real e = std::min(a,b);
		const real f = std::min(c,d);
		return std::min(e,f);
	}
	friend simd_vector sqrt(const simd_vector&);
	friend simd_vector operator*(double, const simd_vector& other);
	friend simd_vector operator/(double, const simd_vector& other);
	friend simd_vector max(const simd_vector& a, const simd_vector& b);

};

inline simd_vector sqrt(const simd_vector& vec) {
	simd_vector r;
	for (integer i = 0; i != simd_len; ++i) {
		r.v[i] =std::sqrt(vec.v[i]);
	}
	return r;
}

inline simd_vector operator*(double d, const simd_vector& other) {
	const simd_vector a = d;
	return a * other;
}

inline simd_vector operator/(double d, const simd_vector& other) {
	const simd_vector a = d;
	return a / other;
}

inline void simd_pack(simd_vector* dest, real* src, integer src_len, integer pos) {
	for (integer i = 0; i != src_len; ++i) {
		dest[i][pos] = src[i];
	}
}

inline void simd_unpack(real* dest, simd_vector* src, integer src_len, integer pos) {
	for (integer i = 0; i != src_len; ++i) {
		dest[i] = src[i][pos];
	}
}

inline simd_vector max(const simd_vector& a, const simd_vector& b) {
	simd_vector r;
	for (integer i = 0; i != simd_len; ++i) {
		r.v[i] = std::max(a.v[i], b.v[i]);
	}
	return r;
}

inline simd_vector abs(const simd_vector& a) {
	return max(a, -a);
}

#endif

#endif /* SIMD_VECTOR_HPP_ */
