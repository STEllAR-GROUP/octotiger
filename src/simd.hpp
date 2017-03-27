/*
 * counting_double_array.hpp
 *
 *  Created on: Jun 7, 2015
 *      Author: dmarce1
 */

#ifndef SIMD_VECTOR_HPP_
#define SIMD_VECTOR_HPP_

#include <cstddef>
#include "defs.hpp"

constexpr std::size_t simd_len = 8;

#if !defined(HPX_HAVE_DATAPAR)

#include "immintrin.h"
#include <cstdlib>
#include <cstdio>

#ifdef USE_SIMD
#if !defined(__MIC__) && !defined(__AVX512F__)
#define SIMD_SIZE 2
#define __mxd __m256d
#define _mmx_set_pd(d)    _mm256_set_pd((d),(d),(d),(d))
#define _mmx_add_pd(a,b)  _mm256_add_pd((a),(b))
#define _mmx_sub_pd(a,b)  _mm256_sub_pd((a),(b))
#define _mmx_mul_pd(a,b)  _mm256_mul_pd((a),(b))
#define _mmx_div_pd(a,b)  _mm256_div_pd((a),(b))
#define _mmx_sqrt_pd(a)   _mm256_sqrt_pd(a)
#define _mmx_max_pd(a, b) _mm256_max_pd((a),(b))
#else
//#warning "Compiling for Intel MIC"
#define SIMD_SIZE 1
#define __mxd __m512d
#define _mmx_set_pd(d)    _mm512_set_pd((d),(d),(d),(d),(d),(d),(d),(d))
#define _mmx_add_pd(a,b)  _mm512_add_pd((a),(b))
#define _mmx_sub_pd(a,b)  _mm512_sub_pd((a),(b))
#define _mmx_mul_pd(a,b)  _mm512_mul_pd((a),(b))
#define _mmx_div_pd(a,b)  _mm512_div_pd((a),(b))
#define _mmx_sqrt_pd(a)   _mm512_sqrt_pd(a)
#define _mmx_max_pd(a, b) _mm512_max_pd((a),(b))
#endif
#else
#define SIMD_SIZE 8
#define __mxd double
#define _mmx_set_pd(d)    (d)
#define _mmx_add_pd(a,b)  ((a)+(b))
#define _mmx_sub_pd(a,b)  ((a)-(b))
#define _mmx_mul_pd(a,b)  ((a)*(b))
#define _mmx_div_pd(a,b)  ((a)/(b))
#define _mmx_sqrt_pd(a)   sqrt(a)
#define _mmx_max_pd(a, b) std::max((a),(b))
#endif

class simd_vector {
private:
	__mxd v[SIMD_SIZE];
public:
	simd_vector() {
		*this = 0;
	}
	inline ~simd_vector() = default;
	simd_vector(const simd_vector&) = default;
	inline simd_vector(double d) {
		for( integer i = 0; i!= SIMD_SIZE; ++i) {
			v[i] =_mmx_set_pd(d);
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
			r.v[i] = _mmx_add_pd(v[i], other.v[i]);
		}
		return r;
	}
	inline simd_vector operator-( const simd_vector& other) const {
		simd_vector r;
		for( integer i = 0; i != SIMD_SIZE; ++i) {
			//_mm_empty();
			r.v[i] = _mmx_sub_pd(v[i], other.v[i]);
		}
		return r;
	}
	inline simd_vector operator*( const simd_vector& other) const {
		simd_vector r;
		for( integer i = 0; i != SIMD_SIZE; ++i) {
			//_mm_empty();
			r.v[i] = _mmx_mul_pd(v[i], other.v[i]);
		}
		return r;
	}
	inline simd_vector operator/( const simd_vector& other) const {
		simd_vector r;
		for( integer i = 0; i != SIMD_SIZE; ++i) {
			//_mm_empty();
			r.v[i] = _mmx_div_pd(v[i], other.v[i]);
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
            const simd_vector other = simd_vector(ONE / d);
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

	double max() const {
		const double a = std::max((*this)[0],(*this)[1]);
		const double b = std::max((*this)[2],(*this)[3]);
		const double c = std::max((*this)[4],(*this)[5]);
		const double d = std::max((*this)[6],(*this)[7]);
		const double e = std::max(a,b);
		const double f = std::max(c,d);
		return std::max(e,f);
	}
	double min() const {
		const double a = std::min((*this)[0],(*this)[1]);
		const double b = std::min((*this)[2],(*this)[3]);
		const double c = std::min((*this)[4],(*this)[5]);
		const double d = std::min((*this)[6],(*this)[7]);
		const double e = std::min(a,b);
		const double f = std::min(c,d);
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
		r.v[i] = _mmx_sqrt_pd(vec.v[i]);
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

inline void simd_pack(simd_vector* dest, double* src, integer src_len, integer pos) {
	for (integer i = 0; i != src_len; ++i) {
		dest[i][pos] = src[i];
	}
}

inline void simd_unpack(double* dest, simd_vector* src, integer src_len, integer pos) {
	for (integer i = 0; i != src_len; ++i) {
		dest[i] = src[i][pos];
	}
}

inline simd_vector max(const simd_vector& a, const simd_vector& b) {
	simd_vector r;
	for (integer i = 0; i != SIMD_SIZE; ++i) {
		//_mm_empty();
		r.v[i] = _mmx_max_pd(a.v[i], b.v[i]);
	}
	return r;
}

inline simd_vector abs(const simd_vector& a) {
	return max(a, -a);
}


/*
typedef double v4sd __attribute__ ((vector_size (32)));
*/
class v4sd {
private:
	__m256d x;
public:
	inline v4sd& operator=(const v4sd& other) {
		x = other.x;
		return *this;
	}
	inline v4sd(const v4sd& other) {
		x = other.x;
	}
	inline v4sd() {
	}
	inline v4sd(const std::initializer_list<double>& other) {
		if( other.size() != 4 ) {
			printf( "Error file %s line %i\n", __FILE__, __LINE__);
			abort();
		}
//		std::array<double, 4> n;
		auto j = other.begin();
		for (int i = 0; i != 4; ++i) {
			(*this)[i] = *j;
			++j;
		}
	}
	inline v4sd& operator=(double other) {
		x = _mm256_set1_pd(other);
		return *this;
	}
	inline v4sd operator+(const v4sd& other) const {
		v4sd r;
		r.x = _mm256_add_pd(x, other.x);
		return r;
	}
	inline v4sd operator-(const v4sd& other) const {
		v4sd r;
		r.x = _mm256_sub_pd(x, other.x);
		return r;
	}
	inline v4sd operator*(const v4sd& other) const {
		v4sd r;
		r.x = _mm256_mul_pd(x, other.x);
		return r;
	}
	inline v4sd operator/(const v4sd& other) const {
		v4sd r;
		r.x = _mm256_div_pd(x, other.x);
		return r;
	}
	inline v4sd& operator+=(const v4sd& other) {
		*this = *this + other;
		return *this;
	}
	inline v4sd& operator-=(const v4sd& other) {
		*this = *this - other;
		return *this;
	}
	inline v4sd& operator*=(const v4sd& other) {
		*this = *this * other;
		return *this;
	}
	inline v4sd& operator/=(const v4sd& other) {
		*this = *this / other;
		return *this;
	}
	inline const double& operator[](int i) const {
		const double* a = reinterpret_cast<const double*>(&x);
		return a[i];
	}
	inline double& operator[](int i) {
		double* a = reinterpret_cast<double*>(&x);
		return a[i];
	}
};

/*
class v4sd {
private:
	double x[4];
public:
	inline v4sd& operator=(const v4sd& other) {
		for( int i = 0; i != 4; i++) {
			x[i] = other.x[i];
		}
		return *this;
	}
	inline v4sd(const v4sd& other) {
		for( int i = 0; i != 4; i++) {
			x[i] = other.x[i];
		}
	}
	inline v4sd() {
	}
	inline v4sd(const std::initializer_list<double>& other) {
		std::array<double, 4> n;
		auto j = other.begin();
		for (int i = 0; i != 4; ++i) {
			(*this)[i] = *j;
			++j;
		}
	}
	inline v4sd& operator=(double other) {
		for( int i = 0; i != 4; i++) {
			x[i] = other;
		}
		return *this;
		return *this;
	}
	inline v4sd operator+(const v4sd& other) const {
		v4sd r;
		for( int i = 0; i != 4; i++) {
			r.x[i] = x[i] + other.x[i];
		}
		return r;
	}
	inline v4sd operator-(const v4sd& other) const {
		v4sd r;
		for( int i = 0; i != 4; i++) {
			r.x[i] = x[i] - other.x[i];
		}
		return r;
	}
	inline v4sd operator*(const v4sd& other) const {
		v4sd r;
		for( int i = 0; i != 4; i++) {
			r.x[i] = x[i] * other.x[i];
		}
		return r;
	}
	inline v4sd operator/(const v4sd& other) const {
		v4sd r;
		for( int i = 0; i != 4; i++) {
			r.x[i] = x[i] / other.x[i];
		}
		return r;
	}
	inline v4sd& operator+=(const v4sd& other) {
		*this = *this + other;
		return *this;
	}
	inline v4sd& operator-=(const v4sd& other) {
		*this = *this - other;
		return *this;
	}
	inline v4sd& operator*=(const v4sd& other) {
		*this = *this * other;
		return *this;
	}
	inline v4sd& operator/=(const v4sd& other) {
		*this = *this / other;
		return *this;
	}
	inline const double& operator[](int i) const {
		return x[i];
	}
	inline double& operator[](int i) {
		return x[i];
	}
};
*/

#else

#if !defined(OCTOTIGER_HAVE_OPERATIONS_COUNT)

#include <hpx/parallel/traits/vector_pack_type.hpp>
#include <hpx/runtime/serialization/datapar.hpp>

#if defined(Vc_HAVE_AVX512F)
using simd_vector = Vc::datapar<double, Vc::datapar_abi::avx512>;
using v4sd = Vc::datapar<double, Vc::datapar_abi::avx>;
#elif defined(Vc_HAVE_AVX)
using simd_vector = typename hpx::parallel::traits::vector_pack_type<double, 8>::type;
using v4sd = Vc::datapar<double, Vc::datapar_abi::avx>;
#else
using simd_vector = typename hpx::parallel::traits::vector_pack_type<double, 8>::type;
using v4sd = typename hpx::parallel::traits::vector_pack_type<double, 4>::type;
#endif

#else // OCTOTIGER_HAVE_OPERATIONS_COUNT

///////////////////////////////////////////////////////////////////////////////
#include <array>

#include "counting_double.hpp"

template <int N>
class counting_double_array;

template <int N>
counting_double_array<N> sqrt(const counting_double_array<N>&);
template <int N>
counting_double_array<N> operator*(double, const counting_double_array<N>& other);
template <int N>
counting_double_array<N> operator/(double, const counting_double_array<N>& other);
template <int N>
counting_double_array<N> max(const counting_double_array<N>& a, const counting_double_array<N>& b);
template <int N>
counting_double_array<N> min(const counting_double_array<N>& a, const counting_double_array<N>& b);

template <int N>
class counting_double_array
{
private:
    std::array<counting_double, N> v;

public:
    counting_double_array() {
        *this = counting_double(0.0);
    }
    inline ~counting_double_array() = default;
    counting_double_array(const counting_double_array&) = default;
    inline counting_double_array(double d) {
        for (integer i = 0; i != N; ++i) {
            v[i] = d;
        }
    }
    inline double sum() const {
        double r = ZERO;
        for (integer i = 0; i != N; ++i) {
            r += (*this)[i];
        }
        return r;
    }
    counting_double_array(counting_double_array&& other) {
        *this = std::move(other);
    }
    counting_double_array(counting_double const* d) {
        for (integer i = 0; i != N; ++i)
            v[i] = d[i];
    }

    inline counting_double_array& operator=(const counting_double_array& other) = default;
    counting_double_array& operator=(counting_double_array&& other) {
        for (integer i = 0; i != N; ++i) {
            v[i] = std::move(other.v[i]);
        }
        return *this;
    }
    counting_double_array& operator=(counting_double other) {
        for (integer i = 0; i != N; ++i) {
            v[i] = other;
        }
        return *this;
    }
    counting_double_array& operator=(counting_double_base_type other) {
        for (integer i = 0; i != N; ++i) {
            v[i] = other;
        }
        return *this;
    }
    inline counting_double_array operator+(const counting_double_array& other) const {
        counting_double_array r;
        for (integer i = 0; i != N; ++i) {
            r.v[i] = v[i] + other.v[i];
        }
        return r;
    }
    inline counting_double_array operator+(const counting_double& other) const {
        counting_double_array r;
        for (integer i = 0; i != N; ++i) {
            r.v[i] = v[i] + other;
        }
        return r;
    }
    inline counting_double_array operator-(const counting_double_array& other) const {
        counting_double_array r;
        for (integer i = 0; i != N; ++i) {
            r.v[i] = v[i] - other.v[i];
        }
        return r;
    }
    inline counting_double_array operator-(const counting_double& other) const {
        counting_double_array r;
        for (integer i = 0; i != N; ++i) {
            r.v[i] = v[i] - other;
        }
        return r;
    }
    inline counting_double_array operator*(const counting_double_array& other) const {
        counting_double_array r;
        for (integer i = 0; i != N; ++i) {
            r.v[i] = v[i] * other.v[i];
        }
        return r;
    }
    inline counting_double_array operator*(const counting_double& other) const {
        counting_double_array r;
        for (integer i = 0; i != N; ++i) {
            r.v[i] = v[i] * other;
        }
        return r;
    }
    inline counting_double_array operator/(const counting_double_array& other) const {
        counting_double_array r;
        for (integer i = 0; i != N; ++i) {
            r.v[i] = v[i] / other.v[i];
        }
        return r;
    }
    inline counting_double_array operator/(const counting_double& other) const {
        counting_double_array r;
        for (integer i = 0; i != N; ++i) {
            r.v[i] = v[i] / other;
        }
        return r;
    }
    inline counting_double_array operator+() const {
        return *this;
    }
    inline counting_double_array operator-() const {
        return counting_double_array(ZERO) - *this;
    }
    inline counting_double_array& operator+=(const counting_double_array& other) {
        *this = *this + other;
        return *this;
    }
    inline counting_double_array& operator-=(const counting_double_array& other) {
        *this = *this - other;
        return *this;
    }
    inline counting_double_array& operator*=(const counting_double_array& other) {
        *this = *this * other;
        return *this;
    }
    inline counting_double_array& operator/=(const counting_double_array& other) {
        *this = *this / other;
        return *this;
    }

    inline counting_double_array operator*(double d) const {
        const counting_double_array other = d;
        return other * *this;
    }
    inline counting_double_array operator/(double d) const {
        const counting_double_array other = ONE / d;
        return *this * other;
    }

    inline counting_double_array operator*=(double d) {
        *this = *this * d;
        return *this;
    }
    inline counting_double_array operator/=(double d) {
        *this = *this * (ONE / d);
        return *this;
    }
    inline counting_double& operator[](std::size_t i) {
        counting_double* a = reinterpret_cast<counting_double*>(&v);
        return a[i];
    }
    inline counting_double operator[](std::size_t i) const {
        const counting_double* a = reinterpret_cast<const counting_double*>(&v);
        return a[i];
    }

    double max() const {
        counting_double t = (*this)[0];
        for (integer i = 1; i != N; ++i) {
            t = std::max(t.value(), double((*this)[i]));
        }
        return t;
    }
    double min() const {
        counting_double t = (*this)[0];
        for (integer i = 1; i != N; ++i) {
            t = std::min(t.value(), double((*this)[i]));
        }
        return t;
    }
    template <int M> friend counting_double_array<M> sqrt(const counting_double_array<M>&);
    template <int M> friend counting_double_array<M> operator*(double, const counting_double_array<M>& other);
    template <int M> friend counting_double_array<M> operator/(double, const counting_double_array<M>& other);
    template <int M> friend counting_double_array<M> max(const counting_double_array<M>& a, const counting_double_array<M>& b);
    template <int M> friend counting_double_array<M> min(const counting_double_array<M>& a, const counting_double_array<M>& b);

    template <typename Archive>
    void serialize(Archive & ar, const unsigned) {
        ar & v;
    }
};

template <int N>
counting_double_array<N> sqrt(const counting_double_array<N>& vec) {
    counting_double_array<N> r;
    for (integer i = 0; i != N; ++i) {
        r.v[i] = std::sqrt(double(vec.v[i]));
    }
    return r;
}

template <int N>
counting_double_array<N> operator*(double d, const counting_double_array<N>& other) {
    const counting_double_array<N> a = d;
    return a * other;
}

template <int N>
counting_double_array<N> operator/(double d, const counting_double_array<N>& other) {
    const counting_double_array<N> a = d;
    return a / other;
}

template <int N>
void simd_pack(counting_double_array<N>* dest, double* src, integer src_len, integer pos) {
    for (integer i = 0; i != src_len; ++i) {
        dest[i][pos] = src[i];
    }
}

template <int N>
void simd_unpack(double* dest, counting_double_array<N>* src, integer src_len, integer pos) {
    for (integer i = 0; i != src_len; ++i) {
        dest[i] = src[i][pos];
    }
}

template <int N>
counting_double_array<N> max(const counting_double_array<N>& a, const counting_double_array<N>& b) {
    counting_double_array<N> r;
    for (integer i = 0; i != N; ++i) {
        r.v[i] = std::max(double(a.v[i]), double(b.v[i]));
    }
    return r;
}

template <int N>
counting_double_array<N> min(const counting_double_array<N>& a, const counting_double_array<N>& b) {
    counting_double_array<N> r;
    for (integer i = 0; i != N; ++i) {
        r.v[i] = std::min(double(a.v[i]), double(b.v[i]));
    }
    return r;
}

template <int N>
counting_double_array<N> abs(const counting_double_array<N>& a) {
    return max(a, -a);
}

///////////////////////////////////////////////////////////////////////////////
using simd_vector = counting_double_array<8>;
using v4sd = counting_double_array<4>;

#endif // OCTOTIGER_HAVE_OPERATIONS_COUNT

#endif // HPX_HAVE_DATAPAR

#endif // SIMD_VECTOR_HPP_
