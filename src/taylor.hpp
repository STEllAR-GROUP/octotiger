/*
 * taylor.hpp
 *
 *  Created on: Jun 9, 2015
 *      Author: dmarce1
 */

#ifndef TAYLOR_HPP_
#define TAYLOR_HPP_

#include "defs.hpp"
#include "profiler.hpp"
#include "simd.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/include/parallel_equal.hpp>
#include <hpx/include/parallel_fill.hpp>
#include <hpx/include/parallel_transform.hpp>
#endif
#include <hpx/traits/is_bitwise_serializable.hpp>

// class simd_vector;

#define MAX_ORDER 5

struct taylor_consts
{
    static const real delta[3][3];
    static integer map2[3][3];
    static integer map3[3][3][3];
    static integer map4[3][3][3][3];
};

constexpr integer taylor_sizes[MAX_ORDER] = {1, 4, 10, 20, 35};    //

///////////////////////////////////////////////////////////////////////////////
template <int N, class T = real>
class taylor
{
private:
    static constexpr integer my_size = taylor_sizes[N - 1];
    static taylor_consts tc;
    std::array<T, my_size> data;

public:
    OCTOTIGER_FORCEINLINE T& operator[](integer i) {
        return data[i];
    }
    OCTOTIGER_FORCEINLINE const T& operator[](integer i) const {
        return data[i];
    }
    OCTOTIGER_FORCEINLINE static constexpr integer size() {
        return my_size;
    }
    taylor() = default;
    ~taylor() = default;
    taylor(const taylor<N, T>&) = default;
    OCTOTIGER_FORCEINLINE taylor(taylor<N, T>&& other) {
        data = std::move(other.data);
    }
    taylor<N, T>& operator=(const taylor<N, T>&) = default;
    OCTOTIGER_FORCEINLINE taylor<N, T>& operator=(taylor<N, T>&& other) {
        data = std::move(other.data);
        return *this;
    }

    OCTOTIGER_FORCEINLINE taylor<N, T>& operator=(T d) {
#pragma GCC ivdep
        for (integer i = 0; i != my_size; ++i) {
            data[i] = d;
        }
        return *this;
    }

    OCTOTIGER_FORCEINLINE taylor<N, T>& operator*=(T d) {
#pragma GCC ivdep
        for (integer i = 0; i != my_size; ++i) {
            data[i] *= d;
        }
        return *this;
    }

    OCTOTIGER_FORCEINLINE taylor<N, T>& operator/=(T d) {
#pragma GCC ivdep
        for (integer i = 0; i != my_size; ++i) {
            data[i] /= d;
        }
        return *this;
    }

    OCTOTIGER_FORCEINLINE taylor<N, T>& operator+=(const taylor<N, T>& other) {
#pragma GCC ivdep
        for (integer i = 0; i != my_size; ++i) {
            data[i] += other.data[i];
        }
        return *this;
    }

    OCTOTIGER_FORCEINLINE taylor<N, T>& operator-=(const taylor<N, T>& other) {
#pragma GCC ivdep
        for (integer i = 0; i != my_size; ++i) {
            data[i] -= other.data[i];
        }
        return *this;
    }

    OCTOTIGER_FORCEINLINE taylor<N, T> operator+(const taylor<N, T>& other) const {
        taylor<N, T> r = *this;
        r += other;
        return r;
    }

    OCTOTIGER_FORCEINLINE taylor<N, T> operator-(const taylor<N, T>& other) const {
        taylor<N, T> r = *this;
        r -= other;
        return r;
    }

    OCTOTIGER_FORCEINLINE taylor<N, T> operator*(const T& d) const {
        taylor<N, T> r = *this;
        r *= d;
        return r;
    }

    OCTOTIGER_FORCEINLINE taylor<N, T> operator/(const T& d) const {
        taylor<N, T> r = *this;
        r /= d;
        return r;
    }

    OCTOTIGER_FORCEINLINE taylor<N, T> operator+() const {
        return *this;
    }

    OCTOTIGER_FORCEINLINE taylor<N, T>& operator+=(v4sd const& other) {
#pragma GCC ivdep
        for (integer i = 0; i != 4; ++i) {
            data[i] += other[i];
        }
        return *this;
    }

    OCTOTIGER_FORCEINLINE taylor<N, T> operator-() const {
        taylor<N, T> r = *this;
#pragma GCC ivdep
        for (integer i = 0; i != my_size; ++i) {
            r.data[i] = -r.data[i];
        }
        return r;
    }

#if defined(HPX_HAVE_DATAPAR)
    OCTOTIGER_FORCEINLINE friend bool operator==(taylor<N, T> const& lhs, taylor<N, T> const& rhs) {
        return std::equal(lhs.data.begin(), lhs.data.end(), rhs.data.begin(),
            [](T const& t1, T const& t2) { return all_of(t1 == t2); });
    }
#endif

    OCTOTIGER_FORCEINLINE static constexpr integer index() {
        return 0;
    }

    OCTOTIGER_FORCEINLINE static constexpr integer index(integer i) {
        return 1 + i;
    }

    OCTOTIGER_FORCEINLINE static integer index(integer i, integer j) {
        return tc.map2[i][j];
    }

    OCTOTIGER_FORCEINLINE static integer index(integer i, integer j, integer k) {
        return tc.map3[i][j][k];
    }

    OCTOTIGER_FORCEINLINE static integer index(integer i, integer j, integer k, integer l) {
        return tc.map4[i][j][k][l];
    }

    OCTOTIGER_FORCEINLINE T const& operator()() const {
        return data[index()];
    }

    OCTOTIGER_FORCEINLINE T const& operator()(integer i) const {
        return data[index(i)];
    }

    OCTOTIGER_FORCEINLINE T const& operator()(integer i, integer j) const {
        return data[index(i, j)];
    }

    OCTOTIGER_FORCEINLINE T const& operator()(integer i, integer j, integer k) const {
        return data[index(i, j, k)];
    }

    OCTOTIGER_FORCEINLINE T const& operator()(integer i, integer j, integer k, integer l) const {
        return data[index(i, j, k, l)];
    }

    OCTOTIGER_FORCEINLINE T& operator()() {
        return data[index()];
    }

    OCTOTIGER_FORCEINLINE T& operator()(integer i) {
        return data[index(i)];
    }

    OCTOTIGER_FORCEINLINE T& operator()(integer i, integer j) {
        return data[index(i, j)];
    }

    OCTOTIGER_FORCEINLINE T& operator()(integer i, integer j, integer k) {
        return data[index(i, j, k)];
    }

    OCTOTIGER_FORCEINLINE T& operator()(integer i, integer j, integer k, integer l) {
        return data[index(i, j, k, l)];
    }

    taylor<N, T>& operator>>=(const std::array<T, NDIM>& X) {
        // PROF_BEGIN;
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
                                B(a, b, c) +=
                                    A(a) * X[b] * X[c] + A(b) * X[c] * X[a] + A(c) * X[a] * X[b];
                                B(a, b, c) += A() * X[a] * X[b] * X[c];
                            }
                        }
                    }
                }
            }
        }
        *this = B;
        // PROF_END;
        return *this;
    }

    OCTOTIGER_FORCEINLINE taylor<N, T> operator>>(const std::array<T, NDIM>& X) const {
        taylor<N, T> r = *this;
        r >>= X;
        return r;
    }

    taylor<N, T>& operator<<=(const std::array<T, NDIM>& X) {
        // PROF_BEGIN;
        const taylor<N, T>& A = *this;
        taylor<N, T> B = A;

        if (N > 1) {
            for (integer a = 0; a != NDIM; a++) {
                B() += A(a) * X[a];
            }
            if (N > 2) {
                for (integer a = 0; a != NDIM; a++) {
                    for (integer b = 0; b != NDIM; b++) {
                        B() += A(a, b) * X[a] * X[b] * T(HALF);
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
                                B() += A(a, b, c) * X[a] * X[b] * X[c] * T(SIXTH);
                            }
                        }
                    }
                    for (integer a = 0; a != NDIM; a++) {
                        for (integer b = 0; b != NDIM; b++) {
                            for (integer c = 0; c != NDIM; c++) {
                                B(a) += A(a, b, c) * X[b] * X[c] * T(HALF);
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
        // PROF_END;
        return *this;
    }

    OCTOTIGER_FORCEINLINE taylor<N, T> operator<<(const std::array<T, NDIM>& X) const {
        taylor<N, T> r = *this;
        r <<= X;
        return r;
    }

    void set_basis(const std::array<T, NDIM>& X);

    OCTOTIGER_FORCEINLINE T* ptr() {
        return data.data();
    }

    OCTOTIGER_FORCEINLINE const T* ptr() const {
        return data.data();
    }

    template <class Arc>
    void serialize(Arc& arc, const unsigned) {
        arc & data;
    }
};

namespace hpx { namespace traits
{
    template <int N, class T>
    struct is_bitwise_serializable<taylor<N, T> >
      : is_bitwise_serializable<typename std::remove_const<T>::type>
    {};
}}

#include "space_vector.hpp"

template <int N, class T>
taylor_consts taylor<N, T>::tc;

constexpr integer to_aa[] = {
    -1,
     4,  7,  9
};
constexpr integer to_aaa[] = {
    -1,
    10, 16, 19
};
constexpr integer to_aaaa[] = {
    -1,
    20, 30, 34
};

constexpr integer to_aab[] = {
    -1,
    -1, -1, -1,
    10, 11, 12, 16, 17, 19
};
constexpr integer to_abb[] = {
    -1,
    -1, -1, -1,
    10, 13, 15, 16, 18, 19
};
constexpr integer to_aaab[] = {
    -1,
    -1, -1, -1,
    20, 21, 22, 30, 31, 34
};
constexpr integer to_abbb[] = {
    -1,
    -1, -1, -1,
    20, 26, 29, 30, 33, 34
};
constexpr integer to_aabb[] = {
    -1,
    -1, -1, -1,
    20, 23, 25, 30, 32, 34
};

constexpr integer to_aabc[] = {
    -1,
    -1, -1, -1,
    -1, -1, -1, -1, -1, -1,
    20, 21, 22, 23, 24, 25, 30, 31, 32, 34
};
constexpr integer to_abbc[] = {
    -1,
    -1, -1, -1,
    -1, -1, -1, -1, -1, -1,
    20, 21, 22, 26, 27, 29, 30, 31, 33, 34
};
constexpr integer to_abcc[] = {
    -1,
    -1, -1, -1,
    -1, -1, -1, -1, -1, -1,
    20, 23, 25, 26, 28, 29, 30, 32, 33, 34
};

constexpr integer to_a[] = {
   -1,
    0,  1,  2,
    0,  0,  0,  1,  1,  2,
    0,  0,  0,  0,  0,  0,  1,  1,  1,  2
};
constexpr integer to_b[] = {
   -1,
   -1, -1, -1,
    0,  1,  2,  1,  2,  2,
    0,  0,  0,  1,  1,  2,  1,  1,  2,  2
};
constexpr integer to_c[] = {
   -1,
   -1, -1, -1,
   -1, -1, -1, -1, -1, -1,
    0,  1,  2,  1,  2,  2,  1,  2,  2,  2
};

template <>
inline void taylor<5, simd_vector>::set_basis(const std::array<simd_vector, NDIM>& X) {
    constexpr integer N = 5;
    using T = simd_vector;
    // PROF_BEGIN;

    // also highly optimized

    // A is D in the paper in formula (6)
    taylor<N, T>& A = *this;

    const T r2 = sqr(X[0]) + sqr(X[1]) + sqr(X[2]);
    T r2inv = 0.0;
    for (volatile integer i = 0; i != simd_len; ++i) {
        if (r2[i] > 0.0) {
            r2inv[i] = ONE / std::max(r2[i], 1.0e-20);
        }
    }

    // parts of formula (6)
    const T d0 = -sqrt(r2inv);
    // parts of formula (7)
    const T d1 = -d0 * r2inv;
    // parts of formula (8)
    const T d2 = T(-3) * d1 * r2inv;
    // parts of  formula (9)
    const T d3 = T(-5) * d2 * r2inv;
    //     const T d4 = -T(7) * d3 * r2inv;

    // formula (6)
    A[0] = d0;

    // formula (7)
    for (integer i = taylor_sizes[0], a = 0; a != NDIM; ++a, ++i) {
        A[i] = X[a] * d1;
    }
    // formula (8)
    for (integer i = taylor_sizes[1], a = 0; a != NDIM; ++a) {
        T const Xad2 = X[a] * d2;
        for (integer b = a; b != NDIM; ++b, ++i) {
            A[i] = Xad2 * X[b];
        }
    }
    // formula (9)
    for (integer i = taylor_sizes[2], a = 0; a != NDIM; ++a) {
        T const Xad3 = X[a] * d3;
        for (integer b = a; b != NDIM; ++b) {
            T const Xabd3 = Xad3 * X[b];
            for (integer c = b; c != NDIM; ++c, ++i) {
                A[i] = Xabd3 * X[c];
            }
        }
    }

    // formula (19)

    // set the coefficients to zero that are calculated next
    for (integer i = taylor_sizes[3]; i != taylor_sizes[4]; ++i) {
        A[i] = ZERO;
    }

    auto const d22 = 2.0 * d2;
//     for (integer a = 0; a != NDIM; a++) {
//         auto const Xad2 = X[a] * d2;
//         auto const Xad3 = X[a] * d3;
//         A(a, a) += d1;
//         A(a, a, a) += Xad2;
//         A(a, a, a, a) += Xad3 * X[a] + d22;
//         for (integer b = a; b != NDIM; b++) {
//             auto const Xabd3 = Xad3 * X[b];
//             auto const Xbd3 = X[b] * d3;
//             A(a, a, b) += X[b] * d2;
//             A(a, b, b) += Xad2;
//             A(a, a, a, b) += Xabd3;
//             A(a, b, b, b) += Xabd3;
//             A(a, a, b, b) += d2;
//             for (integer c = b; c != NDIM; c++) {
//                 A(a, a, b, c) += Xbd3 * X[c];
//                 A(a, b, b, c) += Xad3 * X[c];
//                 A(a, b, c, c) += Xabd3;
//             }
//         }
//     }
    for (integer i = taylor_sizes[0]; i != taylor_sizes[1]; ++i) {
        A[to_aa[i]] += d1;
        integer const to_a_idx = to_a[i];
        A[to_aaa[i]] += X[to_a_idx] * d2;
        A[to_aaaa[i]] += sqr(X[to_a_idx]) * d3 + d22;
    }
    for (integer i = taylor_sizes[1]; i != taylor_sizes[2]; ++i) {
        integer const to_a_idx = to_a[i];
        integer const to_b_idx = to_b[i];
        auto const Xabd3 = X[to_a_idx] * X[to_b_idx] * d3;
        A[to_aab[i]] += X[to_b_idx] * d2;
        A[to_abb[i]] += X[to_a_idx] * d2;
        A[to_aaab[i]] += Xabd3;
        A[to_abbb[i]] += Xabd3;
        A[to_aabb[i]] += d2;
    }
    for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
        integer const to_a_idx = to_a[i];
        integer const to_c_idx = to_c[i];
        auto const Xbd3 = X[to_b[i]] * d3;
        A[to_aabc[i]] += Xbd3 * X[to_c_idx];
        A[to_abbc[i]] += X[to_a_idx] * X[to_c_idx] * d3;
        A[to_abcc[i]] += X[to_a_idx] * Xbd3;
    }

    // PROF_END;
}

#endif /* TAYLOR_HPP_ */
