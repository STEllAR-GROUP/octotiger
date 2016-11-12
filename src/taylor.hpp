/*
 * taylor.hpp
 *
 *  Created on: Jun 9, 2015
 *      Author: dmarce1
 */

#ifndef TAYLOR_HPP_
#define TAYLOR_HPP_

#include "defs.hpp"
//#include "simd.hpp"
#include <array>
#include <cmath>
#include "profiler.hpp"

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/include/parallel_fill.hpp>
#include <hpx/include/parallel_transform.hpp>
#endif

//class simd_vector;

#define MAX_ORDER 5

struct taylor_consts {
	static const real delta[3][3];
	static integer map2[3][3];
	static integer map3[3][3][3];
	static integer map4[3][3][3][3];
};

template<int N, class T = real>
class taylor {
private:
	static constexpr integer sizes[MAX_ORDER] = {1, 4, 10, 20, 35}; //
	static constexpr integer my_size = sizes[N - 1];
	static taylor_consts tc;
	std::array<T, my_size> data;
public:
	T& operator[](integer i) {
		return data[i];
	}
	const T& operator[](integer i) const {
		return data[i];
	}
	static constexpr decltype(my_size) size() {
		return my_size;
	}
	taylor() = default;
	~taylor() = default;
	taylor(const taylor<N, T>&) = default;
	taylor(taylor<N, T> && other) {
		data = std::move(other.data);
	}
	taylor<N, T>& operator=(const taylor<N, T>&) = default;
	taylor<N, T>& operator=(taylor<N, T> && other) {
		data = std::move(other.data);
		return *this;
	}

	taylor<N, T>& operator=(T d) {
#if !defined(HPX_HAVE_DATAPAR)
#pragma GCC ivdep
		for (integer i = 0; i != my_size; ++i) {
			data[i] = d;
		}
#else
        hpx::parallel::fill(
            hpx::parallel::datapar_execution,
            data.begin(), data.end(), d);
#endif
		return *this;
	}

	taylor<N, T>& operator*=(T d) {
#if !defined(HPX_HAVE_DATAPAR)
#pragma GCC ivdep
		for (integer i = 0; i != my_size; ++i) {
			data[i] *= d;
		}
#else
        hpx::parallel::transform(
            hpx::parallel::datapar_execution,
            data.begin(), data.end(),
            [d](T const& val)
            {
                return val * d;
            });
#endif
		return *this;
	}

	taylor<N, T>& operator/=(T d) {
#if !defined(HPX_HAVE_DATAPAR)
#pragma GCC ivdep
		for (integer i = 0; i != my_size; ++i) {
			data[i] /= d;
		}
#else
        hpx::parallel::transform(
            hpx::parallel::datapar_execution,
            data.begin(), data.end(),
            [d](T const& val)
            {
                return val / d;
            });
#endif
		return *this;
	}

	taylor<N, T>& operator+=(const taylor<N, T>& other) {
#if !defined(HPX_HAVE_DATAPAR)
#pragma GCC ivdep
		for (integer i = 0; i != my_size; ++i) {
			data[i] += other.data[i];
		}
#else
        hpx::parallel::transform(
            hpx::parallel::datapar_execution,
            data.begin(), data.end(),
            other.data.begin(), other.data.end(),
            [](T const& t1, T const& t2)
            {
                return t1 + t2;
            });
#endif
		return *this;
	}

	taylor<N, T>& operator-=(const taylor<N, T>& other) {
#if !defined(HPX_HAVE_DATAPAR)
#pragma GCC ivdep
		for (integer i = 0; i != my_size; ++i) {
			data[i] -= other.data[i];
		}
#else
        hpx::parallel::transform(
            hpx::parallel::datapar_execution,
            data.begin(), data.end(),
            other.data.begin(), other.data.end(),
            [](T const& t1, T const& t2)
            {
                return t1 - t2;
            });
#endif
		return *this;
	}

	taylor<N, T> operator+(const taylor<N, T>& other) const {
		taylor<N, T> r = *this;
		r += other;
		return r;
	}

	taylor<N, T> operator-(const taylor<N, T>& other) const {
		taylor<N, T> r = *this;
		r -= other;
		return r;
	}

	taylor<N, T> operator*(const T& d) const {
		taylor<N, T> r = *this;
		r *= d;
		return r;
	}

	taylor<N, T> operator/(const T& d) const {
		taylor<N, T> r = *this;
		r /= d;
		return r;
	}

	taylor<N, T> operator+() const {
		return *this;
	}

	taylor<N, T> operator-() const {
		taylor<N, T> r = *this;
#if !defined(HPX_HAVE_DATAPAR)
#pragma GCC ivdep
		for (integer i = 0; i != my_size; ++i) {
			r.data[i] = -r.data[i];
		}
#else
        hpx::parallel::transform(
            hpx::parallel::datapar_execution,
            other.data.begin(), other.data.end(),
            [](T const& val)
            {
                return -val;
            });
#endif
		return r;
	}

	T operator()() const {
		return data[0];
	}

	T operator()(integer i) const {
		return data[1 + i];
	}

	T operator()(integer i, integer j) const {
		return data[tc.map2[i][j]];
	}

	T operator()(integer i, integer j, integer k) const {
		return data[tc.map3[i][j][k]];
	}

	T operator()(integer i, integer j, integer k, integer l) const {
		return data[tc.map4[i][j][k][l]];
	}

	T& operator()() {
		return data[0];
	}

	T& operator()(integer i) {
		return data[1 + i];
	}

	T& operator()(integer i, integer j) {
		return data[tc.map2[i][j]];
	}

	T& operator()(integer i, integer j, integer k) {
		return data[tc.map3[i][j][k]];
	}

	T& operator()(integer i, integer j, integer k, integer l) {
		return data[tc.map4[i][j][k][l]];
	}

	taylor<N, T>& operator>>=(const std::array<T, NDIM>& X) {
		//PROF_BEGIN;
		const taylor<N, T>& A = *this;
		taylor<N, T> B = A;

		if (N > 1) {
			for (integer a = 0; a < NDIM; ++a) {
				B(a) += A() * X[a];
			}
			if (N > 2) {
				for (integer a = 0; a < NDIM; ++a) {
					for (integer b = a; b < NDIM; ++b) {
						B(a, b) += A(a) * X[b] + X[a] * A(b);
						B(a, b) += A() * X[a] * X[b];
					}
				}
				if (N > 3) {
					for (integer a = 0; a < NDIM; ++a) {
						for (integer b = a; b < NDIM; ++b) {
							for (integer c = b; c < NDIM; ++c) {
								B(a, b, c) += A(a, b) * X[c] + A(b, c) * X[a] + A(c, a) * X[b];
								B(a, b, c) += A(a) * X[b] * X[c] + A(b) * X[c] * X[a] + A(c) * X[a] * X[b];
								B(a, b, c) += A() * X[a] * X[b] * X[c];
							}
						}
					}
				}
			}
		}
		*this = B;
		//PROF_END;
		return *this;
	}

	taylor<N, T> operator>>(const std::array<T, NDIM>& X) const {
		taylor<N, T> r = *this;
		r >>= X;
		return r;
	}

	taylor<N, T>& operator<<=(const std::array<T, NDIM>& X) {
		//PROF_BEGIN;
		const taylor<N, T>& A = *this;
		taylor<N, T> B = A;

		if (N > 1) {
			for (integer a = 0; a != NDIM; a++) {
				B() += A(a) * X[a];
			}
			if (N > 2) {
				for (integer a = 0; a != NDIM; a++) {
					for (integer b = 0; b != NDIM; b++) {
						B() += A(a, b) * X[a] * X[b] * T(real(1) / real(2));
					}
				}
				for (integer a = 0; a != NDIM; a++) {
					for (integer b = 0; b != NDIM; b++) {
						B(a) += A(a, b) * X[b];
					}
				}
				if (N > 3) {
					for (integer a = 0; a != NDIM; a++) {
						for (integer b = 0; b != NDIM; b++) {
							for (integer c = 0; c != NDIM; c++) {
								B() += A(a, b, c) * X[a] * X[b] * X[c] * T(real(1) / real(6));
							}
						}
					}
					for (integer a = 0; a != NDIM; a++) {
						for (integer b = 0; b != NDIM; b++) {
							for (integer c = 0; c != NDIM; c++) {
								B(a) += A(a, b, c) * X[b] * X[c] * T(real(1) / real(2));
							}
						}
					}
					for (integer a = 0; a != NDIM; a++) {
						for (integer b = 0; b != NDIM; b++) {
							for (integer c = a; c != NDIM; c++) {
								B(a, c) += A(a, b, c) * X[b];
							}
						}
					}
				}
			}
		}
		*this = B;
		//PROF_END;
		return *this;
	}

	taylor<N, T> operator<<(const std::array<T, NDIM>& X) const {
		taylor<N, T> r = *this;
		r <<= X;
		return r;
	}

	void set_basis(const std::array<T, NDIM>& X);

	T* ptr() {
		return data.data();
	}

	const T* ptr() const {
		return data.data();
	}

	template<class Arc>
	void serialize(Arc& arc, const unsigned) {
		arc & data;
	}

};

#include "space_vector.hpp"

template<int N, class T>
taylor_consts taylor<N, T>::tc;

#endif /* TAYLOR_HPP_ */
