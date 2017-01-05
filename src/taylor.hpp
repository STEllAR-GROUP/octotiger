/*
 * taylor.hpp
 *
 *  Created on: Jun 9, 2015
 *      Author: dmarce1
 */

#ifndef TAYLOR_HPP_
#define TAYLOR_HPP_

#include "defs.hpp"
#include "simd.hpp"
#include "profiler.hpp"
#include "simd.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/include/parallel_equal.hpp>
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

constexpr integer taylor_sizes[MAX_ORDER] = {1, 4, 10, 20, 35}; //

template<int N, class T = real>
class taylor {
private:
	static constexpr integer my_size = taylor_sizes[N - 1];
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
            hpx::parallel::dataseq_execution,
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
            hpx::parallel::dataseq_execution,
            data.begin(), data.end(), data.begin(),
            [d](auto const& val)
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
            hpx::parallel::dataseq_execution,
            data.begin(), data.end(), data.begin(),
            [d](auto const& val)
            {
                return val / d;
            });
#endif
		return *this;
	}

	taylor<N, T>& operator+=(const taylor<N, T>& other) {
// #if !defined(HPX_HAVE_DATAPAR)
#pragma GCC ivdep
		for (integer i = 0; i != my_size; ++i) {
			data[i] += other.data[i];
		}
// #else
//         hpx::parallel::transform(
//             hpx::parallel::dataseq_execution,
//             data.begin(), data.end(),
//             other.data.begin(), other.data.end(), data.begin(),
//             [](auto const& t1, auto const& t2)
//             {
//                 return t1 + t2;
//             });
// #endif
		return *this;
	}

	taylor<N, T>& operator-=(const taylor<N, T>& other) {
// #if !defined(HPX_HAVE_DATAPAR)
#pragma GCC ivdep
		for (integer i = 0; i != my_size; ++i) {
			data[i] -= other.data[i];
		}
// #else
//         hpx::parallel::transform(
//             hpx::parallel::dataseq_execution,
//             data.begin(), data.end(),
//             other.data.begin(), other.data.end(), data.begin(),
//             [](auto const& t1, auto const& t2)
//             {
//                 return t1 - t2;
//             });
// #endif
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
            hpx::parallel::dataseq_execution,
            r.data.begin(), r.data.end(), r.data.begin(),
            [](T const& val)
            {
                return -val;
            });
#endif
		return r;
	}

#if defined(HPX_HAVE_DATAPAR)
    friend bool operator==(taylor<N, T> const& lhs, taylor<N, T> const& rhs)
    {
        return hpx::parallel::equal(
            hpx::parallel::dataseq_execution,
            lhs.data.begin(), lhs.data.end(), rhs.data.begin(),
            [](T const& t1, T const& t2)
            {
                return all_of(t1 == t2);
            });
    }
#endif

    static constexpr integer index() {
        return 0;
    }

    static constexpr integer index(integer i) {
        return 1 + i;
    }

    static integer index(integer i, integer j) {
        return tc.map2[i][j];
    }

    static integer index(integer i, integer j, integer k) {
        return tc.map3[i][j][k];
    }

    static integer index(integer i, integer j, integer k, integer l) {
        return tc.map4[i][j][k][l];
    }


	T const& operator()() const {
		return data[index()];
	}

	T const& operator()(integer i) const {
		return data[index(i)];
	}

	T const& operator()(integer i, integer j) const {
		return data[index(i, j)];
	}

	T const& operator()(integer i, integer j, integer k) const {
		return data[index(i, j, k)];
	}

	T const& operator()(integer i, integer j, integer k, integer l) const {
		return data[index(i, j, k, l)];
	}

	T& operator()() {
		return data[index()];
	}

	T& operator()(integer i) {
		return data[index(i)];
	}

	T& operator()(integer i, integer j) {
		return data[index(i, j)];
	}

	T& operator()(integer i, integer j, integer k) {
		return data[index(i, j, k)];
	}

	T& operator()(integer i, integer j, integer k, integer l) {
		return data[index(i, j, k, l)];
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

template<>
inline void taylor<5, simd_vector>::set_basis(const std::array<simd_vector, NDIM>& X) {
    constexpr integer N = 5;
    using T = simd_vector;
    //PROF_BEGIN;
    taylor<N, T>& A = *this;

	const T r2 = sqr(X[0]) + sqr(X[1]) + sqr(X[2]);
	T r2inv = 0.0;
// #if !defined(HPX_HAVE_DATAPAR)
	for (integer i = 0; i != simd_len; ++i) {
		if (r2[i] > 0.0) {
			r2inv[i] = ONE / r2[i];
		}
	}
// #else
//     where(r2 > 0.0) | r2inv = ONE / r2;
// #endif

    const T d0 = -sqrt(r2inv);

    // 1 MUL
    const T d1 = -d0 * r2inv;

    // 2 MULS
    const T d2 = T(-3) * d1 * r2inv;

    // 2 MULS
    const T d3 = T(-5) * d2 * r2inv;
//     const T d4 = -T(7) * d3 * r2inv;

    // Previously we've had this code. In my measurements the old code was
    // about 15% faster than the current code. However, I have measured it on
    // non-KNL platforms only.
//     taylor<N - 1, T> XX;
//     for (integer a = 0; a != NDIM; a++) {
//         auto const tmpa = X[a];
//         XX(a) = tmpa;
//
//         for (integer b = a; b != NDIM; b++) {
//             auto const tmpab = tmpa * X[b];
//             XX(a, b) = tmpab;
//
//             for (integer c = b; c != NDIM; c++) {
//                 XX(a, b, c) = tmpab * X[c];
//             }
//         }
//     }
//     A[0] = d0;
//     for (integer i = taylor_sizes[0]; i != taylor_sizes[1]; ++i) {
//         A[i] = XX[i] * d1;
//     }
//     for (integer i = taylor_sizes[1]; i != taylor_sizes[2]; ++i) {
//         A[i] = XX[i] * d2;
//     }
//     for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
//         A[i] = XX[i] * d3;
//     }
//     for (integer i = taylor_sizes[3]; i != taylor_sizes[4]; ++i) {
//         A[i] = ZERO;
//     }
//     for (integer a = 0; a != NDIM; a++) {
//         auto const XXa = XX(a) * d2;
//         A(a, a) += d1;
//         A(a, a, a) += XXa;
//         A(a, a, a, a) += XX(a, a) * d3;
//         A(a, a, a, a) += 2.0 * d2;
//         for (integer b = a; b != NDIM; b++) {
//             auto const XXab = XX(a, b) * d3;
//             A(a, a, b) += XX(b) * d2;
//             A(a, b, b) += XXa;
//             A(a, a, a, b) += XXab;
//             A(a, b, b, b) += XXab;
//             A(a, a, b, b) += d2;
//             for (integer c = b; c != NDIM; c++) {
//                 A(a, a, b, c) += XX(b, c) * d3;
//                 A(a, b, b, c) += XX(a, c) * d3;
//                 A(a, b, c, c) += XXab;
//             }
//         }
//     }

    ///////////////////////////////////////////////////////////////////////////
    A[0] = d0;
    for (integer i = taylor_sizes[0], a = 0; a != NDIM; ++a, ++i) {
        A[i] = X[a] * d1;
    }
    for (integer i = taylor_sizes[1], a = 0; a != NDIM; ++a) {
        T const Xad2 = X[a] * d2;
        for (integer b = a; b != NDIM; ++b, ++i) {
            A[i] = Xad2 * X[b];
        }
    }
    for (integer i = taylor_sizes[2], a = 0; a != NDIM; ++a) {
        T const Xad3 = X[a] * d3;
        for (integer b = a; b != NDIM; ++b) {
            T const Xabd3 = Xad3 * X[b];
            for (integer c = b; c != NDIM; ++c, ++i) {
                A[i] = Xabd3 * X[c];
            }
        }
    }
    for (integer i = taylor_sizes[3]; i != taylor_sizes[4]; ++i) {
        A[i] = ZERO;
    }

    ///////////////////////////////////////////////////////////////////////////
    auto const d22 = 2.0 * d2;
    for (integer a = 0; a != NDIM; a++) {
        auto const Xad2 = X[a] * d2;
        auto const Xad3 = X[a] * d3;
        A(a, a) += d1;
        A(a, a, a) += Xad2;
        A(a, a, a, a) += Xad3 * X[a] + d22;
        for (integer b = a; b != NDIM; b++) {
            auto const Xabd3 = Xad3 * X[b];
            auto const Xbd3 = X[b] * d3;
            A(a, a, b) += X[b] * d2;
            A(a, b, b) += Xad2;
            A(a, a, a, b) += Xabd3;
            A(a, b, b, b) += Xabd3;
            A(a, a, b, b) += d2;
            for (integer c = b; c != NDIM; c++) {
                A(a, a, b, c) += Xbd3 * X[c];
                A(a, b, b, c) += Xad3 * X[c];
                A(a, b, c, c) += Xabd3;
            }
        }
    }

    //PROF_END;
}

#endif /* TAYLOR_HPP_ */
