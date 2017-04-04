///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2017 Louisiana State University 
// Copyright (c) 2017 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

// Convention: i is the dimension with the least stride.
// Left/Fortran/Column-Major Layout : L[i][j]
// Right/C++/Row-Major Layout       : L[j][i] 

// TODO: Remove interactions list (significant code rewrite)

// TODO: Replace scalar loops annotated for vectorization with Vc

// TODO: Fusion/fission
// TODO: Loop invariant code motion 
// TODO: Post-vectorization unrolling 
// TODO: Split into two phases
// TODO: Fix SoA in grid
// TODO: Why don't we just access m0 and m1 directly?
// TODO: Don't use auto everywhere.
// TODO: Ilist indices should be SoA not AoS
// TODO: Looks like only 10 elements of n0 and n1 are used (taylor_sizes[2] to taylor_sizes[3])?

#include "vectorization_and_assumption_hints.hpp"

#if !defined(OCTOTIGER_82BA7119_59FB_4018_88F8_AF18F969220C)
#define OCTOTIGER_82BA7119_59FB_4018_88F8_AF18F969220C

///////////////////////////////////////////////////////////////////////////////

inline void grid::compute_interactions_initialize_L_c(std::true_type) noexcept
{
    #if defined(HPX_HAVE_DATAPAR)
        hpx::parallel::fill(hpx::parallel::execution::dataseq,
                            L_c.begin(), L_c.end(), ZERO);
    #else
        std::fill(std::begin(L_c), std::end(L_c), ZERO);
    #endif

    //s.add_fp_memstores(L_c.size());
}

inline void grid::compute_interactions_initialize_L_c(std::false_type) noexcept {}

///////////////////////////////////////////////////////////////////////////////
        
template <
    std::vector<interaction_type> const* __restrict__ IList 
  , std::size_t TileWidth
    >
inline void grid::compute_interactions_initialize_n_ang_mom(
    integer i_begin
  , integer i_end
  , compute_interactions_tile<TileWidth>& t
  , compute_interactions_stats_t& s
  , std::true_type
  , vector_function_tag 
    ) noexcept
{
    // TODO: Switch to Vc.

    BOOST_ASSUME((i_end - i_begin) == TileWidth);

    auto& M = *M_ptr;

    for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j)                // TRIP COUNT: 10
    {
        real* __restrict__    n0j  = t.n0[j].data();
        real* __restrict__    n1j  = t.n1[j].data();
        __m512d* __restrict__ Vn0j = reinterpret_cast<__m512d* __restrict__>(t.n0[j].data());
        __m512d* __restrict__ Vn1j = reinterpret_cast<__m512d* __restrict__>(t.n1[j].data());
        BOOST_ASSUME_ALIGNED(Vn0j, 64);
        BOOST_ASSUME_ALIGNED(Vn1j, 64);
        BOOST_ASSUME_ALIGNED(Vn0j, 64);
        BOOST_ASSUME_ALIGNED(Vn1j, 64);

        real const* __restrict__    M_  = M().data();
        real const* __restrict__    Mj  = M[j].data();
        __m512d const* __restrict__ VM_ = reinterpret_cast<__m512d const* __restrict__>(M().data());
        __m512d const* __restrict__ VMj = reinterpret_cast<__m512d const* __restrict__>(M[j].data());
        BOOST_ASSUME_ALIGNED(M_,  64);
        BOOST_ASSUME_ALIGNED(Mj,  64);
        BOOST_ASSUME_ALIGNED(VM_, 64);
        BOOST_ASSUME_ALIGNED(VMj, 64);

        for (integer i = i_begin; i < i_end; i += 8)
        {
            integer const ti = i - i_begin;

            __m512i const iii0 = {
                (*IList)[i  ].first
              , (*IList)[i+1].first
              , (*IList)[i+2].first
              , (*IList)[i+3].first
              , (*IList)[i+4].first
              , (*IList)[i+5].first
              , (*IList)[i+6].first
              , (*IList)[i+7].first
            };
            __m512i const iii1 = {
                (*IList)[i  ].second
              , (*IList)[i+1].second
              , (*IList)[i+2].second
              , (*IList)[i+3].second
              , (*IList)[i+4].second
              , (*IList)[i+5].second
              , (*IList)[i+6].second
              , (*IList)[i+7].second
            };

            __m512d const M_0 = _mm512_i64gather_pd(iii0, M_, 8);
            __m512d const M_1 = _mm512_i64gather_pd(iii1, M_, 8);

            s.add_fp_memloads(2*8);

            // NOTE: Due to the order of iteration, these computations are
            // performed redundantly.
            __m512d const M_1divM_0 = _mm512_div_pd(M_1, M_0);
            __m512d const M_0divM_1 = _mm512_div_pd(M_0, M_1);

            s.add_fp_divs(2*8);

            __m512d const Mj0 = _mm512_i64gather_pd(iii0, Mj, 8);
            __m512d const Mj1 = _mm512_i64gather_pd(iii1, Mj, 8);

            s.add_fp_memloads(2*8); 

            __m512d const n0jrhs = _mm512_fnmadd_pd(Mj0, M_1divM_0, Mj1); // -(Mj0 * M_1divM_0) + Mj1
                                                                          // Mj1 - Mj0 * M_1divM_0
            __m512d const n1jrhs = _mm512_fnmadd_pd(Mj1, M_0divM_1, Mj0); // -(Mj1 * M_0divM_1) + Mj0
                                                                          // Mj0 - Mj1 * M_0divM_1

            _mm512_store_pd(n0j + ti, n0jrhs);
            _mm512_store_pd(n1j + ti, n1jrhs);

            s.add_fp_fmas(      2*8);
            s.add_fp_tilestores(2*8);
        }
    }
}

template <
    std::vector<interaction_type> const* __restrict__ IList 
  , std::size_t TileWidth
    >
inline void grid::compute_interactions_initialize_n_ang_mom(
    integer i_begin
  , integer i_end
  , compute_interactions_tile<TileWidth>& t
  , compute_interactions_stats_t& s
  , std::true_type
  , scalar_function_tag 
    ) noexcept
{
    BOOST_ASSUME((i_end - i_begin) == TileWidth);

    auto& M = *M_ptr;

    for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j)                // TRIP COUNT: 10
    {
        real* __restrict__ n0j = t.n0[j].data();
        real* __restrict__ n1j = t.n1[j].data();
        BOOST_ASSUME_ALIGNED(n0j, 64);
        BOOST_ASSUME_ALIGNED(n1j, 64);

        real const* __restrict__ M_ = M().data();
        real const* __restrict__ Mj = M[j].data();
        BOOST_ASSUME_ALIGNED(M_, 64);
        BOOST_ASSUME_ALIGNED(Mj, 64);

        #pragma omp simd
        for (integer i = i_begin; i < i_end; ++i)                               // TRIP COUNT: 10 * TileWidth; UNIT STRIDE
        {
            integer const ti = i - i_begin;

            integer const iii0 = (*IList)[i].first;                             // 1 INT LOAD FROM MEM (indirect addressing)
            integer const iii1 = (*IList)[i].second;                            // 1 INT LOAD FROM MEM (indirect addressing)

            auto const M_0 = M_[iii0];                                          // 1 FP LOAD FROM MEM
            auto const M_1 = M_[iii1];                                          // 1 FP LOAD FROM MEM

            s.add_fp_memloads(2);

            // NOTE: Due to the order of iteration, these computations are
            // performed redundantly.
            auto const M_1divM_0 = M_1 / M_0;                                   // 1 FP DIV
            auto const M_0divM_1 = M_0 / M_1;                                   // 1 FP DIV

            s.add_fp_divs(2);

            auto const Mj0 = Mj[iii0];                                          // 1 FP LOAD FROM MEM
            auto const Mj1 = Mj[iii1];                                          // 1 FP LOAD FROM MEM

            s.add_fp_memloads(2);

            auto const n0jrhs = Mj1 - Mj0 * M_1divM_0;                          // 1 FP FMA
            auto const n1jrhs = Mj0 - Mj1 * M_0divM_1;                          // 1 FP FMA

            n0j[ti] = n0jrhs;                                                   // 1 FP STORE TO TILE
            n1j[ti] = n1jrhs;                                                   // 1 FP STORE TO TILE

            s.add_fp_fmas(      2);
            s.add_fp_tilestores(2);
        }
    }
}

template <
    std::vector<interaction_type> const* __restrict__ IList
  , std::size_t TileWidth
    >
inline void grid::compute_interactions_initialize_n_ang_mom(
    integer i_begin
  , integer i_end
  , compute_interactions_tile<TileWidth>& t
  , compute_interactions_stats_t& s
  , std::false_type
  , vector_function_tag
    ) noexcept
{
    BOOST_ASSUME((i_end - i_begin) == TileWidth);

    for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j)
    {
        __m512d* __restrict__ Vn0j = reinterpret_cast<__m512d* __restrict__>(t.n0[j].data());
        __m512d* __restrict__ Vn1j = reinterpret_cast<__m512d* __restrict__>(t.n1[j].data());
        BOOST_ASSUME_ALIGNED(Vn0j, 64);
        BOOST_ASSUME_ALIGNED(Vn1j, 64);

        for (integer i = i_begin; i < i_end; i += 8)
        {
            integer const ti = i - i_begin;

            *(Vn0j + ti) = _mm512_set1_pd(ZERO);
            *(Vn1j + ti) = _mm512_set1_pd(ZERO);

            s.add_fp_tilestores(2);
        }
    }
}

template <
    std::vector<interaction_type> const* __restrict__ IList
  , std::size_t TileWidth
    >
inline void grid::compute_interactions_initialize_n_ang_mom(
    integer i_begin
  , integer i_end
  , compute_interactions_tile<TileWidth>& t
  , compute_interactions_stats_t& s
  , std::false_type
  , scalar_function_tag
    ) noexcept
{
    BOOST_ASSUME((i_end - i_begin) == TileWidth);

    for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j)                // TRIP COUNT: 10
    {
        real* __restrict__ n0j = t.n0[j].data();
        real* __restrict__ n1j = t.n1[j].data();
        BOOST_ASSUME_ALIGNED(n0j, 64);
        BOOST_ASSUME_ALIGNED(n1j, 64);

        #pragma omp simd 
        for (integer i = i_begin; i < i_end; ++i)                               // TRIP COUNT: 10 * TileWidth; UNIT STRIDE
        {
            integer const ti = i - i_begin;

            n0j[ti] = ZERO;                                                     // 1 FP STORE TO TILE
            n1j[ti] = ZERO;                                                     // 1 FP STORE TO TILE

            s.add_fp_tilestores(2);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

template <
    std::vector<interaction_type> const* __restrict__ IList
  , std::size_t TileWidth
    >
inline void grid::compute_interactions_A0_A1_0(
    compute_interactions_tile<TileWidth>& t
  , compute_interactions_stats_t& s
  , std::true_type
    ) noexcept
{}

template <
    std::vector<interaction_type> const* __restrict__ IList 
  , std::size_t TileWidth
    >
inline void grid::compute_interactions_A0_A1_0(
    compute_interactions_tile<TileWidth>& t
  , compute_interactions_stats_t& s
  , std::false_type
    ) noexcept
{
    real* __restrict__ A00 = t.A0[0].data();
    real* __restrict__ A10 = t.A1[0].data();
    BOOST_ASSUME_ALIGNED(A00, 64);
    BOOST_ASSUME_ALIGNED(A10, 64);

    for (integer j = taylor_sizes[0]; j != taylor_sizes[1]; ++j)                // TRIP COUNT: 3
    {
        real const* __restrict__ m0j = t.m0[j].data();
        real const* __restrict__ m1j = t.m1[j].data();
        BOOST_ASSUME_ALIGNED(m0j, 64);
        BOOST_ASSUME_ALIGNED(m1j, 64);

        real const* __restrict__ Dj = t.D[j].data();
        BOOST_ASSUME_ALIGNED(Dj, 64);

        #pragma omp simd 
        for (integer ti = 0; ti < TileWidth; ++ti)                              // TRIP COUNT: 3 * TileWidth; UNIT STRIDE
        {
            A00[ti] -= m0j[ti] * Dj[ti];                                        // 3 FP LOADS FROM TILE, 1 FP FMA, 1 FP STORE TO TILE
            A10[ti] += m1j[ti] * Dj[ti];                                        // 3 FP LOADS FROM TILE, 1 FP FMA, 1 FP STORE TO TILE

            s.add_fp_tileloads( 6);
            s.add_fp_fmas(      2);
            s.add_fp_tilestores(2);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

template <
    std::vector<interaction_type> const* __restrict__ IList
  , std::size_t TileWidth
    >
inline void grid::compute_interactions_A0_A1(
    compute_interactions_tile<TileWidth>& t
  , compute_interactions_stats_t& s
  , std::true_type
    ) noexcept
{}

template <
    std::vector<interaction_type> const* __restrict__ IList 
  , std::size_t TileWidth
    >
inline void grid::compute_interactions_A0_A1(
    compute_interactions_tile<TileWidth>& t
  , compute_interactions_stats_t& s
  , std::false_type
    ) noexcept
{
    for (integer a = 0; a != NDIM; ++a)                                         // TRIP COUNT: 3
    {
        real* __restrict__ A0a = t.A0(a).data();
        real* __restrict__ A1a = t.A1(a).data();
        BOOST_ASSUME_ALIGNED(A0a, 64);
        BOOST_ASSUME_ALIGNED(A1a, 64);

        for (integer b = 0; b != NDIM; ++b)                                     // TRIP COUNT: 9
        {
            real const* __restrict__ m0a = t.m0(a).data();
            real const* __restrict__ m1a = t.m1(a).data();
            BOOST_ASSUME_ALIGNED(m0a, 64);
            BOOST_ASSUME_ALIGNED(m1a, 64);

            real const* __restrict__ Dab = t.D(a, b).data();
            BOOST_ASSUME_ALIGNED(Dab, 64);

            #pragma omp simd
            for (integer ti = 0; ti < TileWidth; ++ti)                          // TRIP COUNT: 9 * TileWidth; UNIT STRIDE
            {
                auto const tmp = Dab[ti];                                       // 1 FP LOAD FROM TILE
                A0a[ti] -= m0a[ti] * tmp;                                       // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE
                A1a[ti] -= m1a[ti] * tmp;                                       // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE

                s.add_fp_tileloads( 5);
                s.add_fp_fmas(      2);
                s.add_fp_tilestores(2);
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

template <
    std::vector<interaction_type> const* __restrict__ IList 
  , std::size_t TileWidth
    >
inline void grid::compute_interactions_B0_B1(
    compute_interactions_tile<TileWidth>& t
  , compute_interactions_stats_t& s
  , std::true_type
    ) noexcept
{
    for (integer a = 0; a != NDIM; ++a)                                         // TRIP COUNT: 3
    {
        real* __restrict__ B0a = t.B0[a].data();
        real* __restrict__ B1a = t.B1[a].data();
        BOOST_ASSUME_ALIGNED(B0a, 64);
        BOOST_ASSUME_ALIGNED(B1a, 64);

        for (integer b = 0; b != NDIM; ++b)                                     // TRIP COUNT: 9
            for (integer c = b; c != NDIM; ++c)                                 // TRIP COUNT: 18
                for (integer d = c; d != NDIM; ++d)                             // TRIP COUNT: 30
                {
                    real const sixth_factorbcd = sixth_factor(b, c, d);

                    real const* __restrict__ n0bcd = t.n0(b, c, d).data();
                    real const* __restrict__ n1bcd = t.n1(b, c, d).data();
                    BOOST_ASSUME_ALIGNED(n0bcd, 64);
                    BOOST_ASSUME_ALIGNED(n1bcd, 64);

                    real const* __restrict__ Dabcd = t.D(a, b, c, d).data();
                    BOOST_ASSUME_ALIGNED(Dabcd, 64);

                    #pragma omp simd 
                    for (integer ti = 0; ti < TileWidth; ++ti)                  // TRIP COUNT: 30 * TileWidth; UNIT STRIDE
                    {
                        auto const tmp = Dabcd[ti] * sixth_factorbcd;           // 1 FP LOAD FROM TILE, 1 FP MUL
                        B0a[ti] -= n0bcd[ti] * tmp;                             // 2 FP LOADS FROM TILE, 1 FP FMA, 1 FP STORE TO TILE
                        B1a[ti] -= n1bcd[ti] * tmp;                             // 2 FP LOADS FROM TILE, 1 FP FMA, 1 FP STORE TO TILE

                        s.add_fp_tileloads( 5);
                        s.add_fp_muls(      1);
                        s.add_fp_fmas(      2);
                        s.add_fp_tilestores(2);
                    }
                }
    }
}

template <
    std::vector<interaction_type> const* __restrict__ IList 
  , std::size_t TileWidth
    >
inline void grid::compute_interactions_B0_B1(
    compute_interactions_tile<TileWidth>& t
  , compute_interactions_stats_t& s
  , std::false_type
    ) noexcept
{}

///////////////////////////////////////////////////////////////////////////////

template <
    std::vector<interaction_type> const* __restrict__ IList
  , std::size_t TileWidth
    >
inline void grid::store_to_L_c(
    integer i_begin
  , integer i_end
  , compute_interactions_tile<TileWidth>& t
  , compute_interactions_stats_t& s
  , std::true_type
    ) noexcept
{
    BOOST_ASSUME((i_end - i_begin) == TileWidth);

    for (integer j = 0; j != NDIM; ++j)                                         // TRIP COUNT: 3
    {
        real* __restrict__ B0j = t.B0[j].data();
        real* __restrict__ B1j = t.B1[j].data();
        BOOST_ASSUME_ALIGNED(B0j, 64);
        BOOST_ASSUME_ALIGNED(B1j, 64);

        #pragma omp simd 
        for (integer i = i_begin; i < i_end; ++i)                               // TRIP COUNT: 3 * TileWidth; UNIT STRIDE
        {
            integer const ti = i - i_begin;

            integer const iii0 = (*IList)[i].first;                             // 1 INT LOAD FROM MEM (indirect addressing)
            integer const iii1 = (*IList)[i].second;                            // 1 INT LOAD FROM MEM (indirect addressing)

            L_c[iii0][j] += B0j[ti];                                            // 1 FP LOAD FROM TILE, 1 FP LOAD FROM CACHE, 1 FP ADD, 1 FP STORE TO MEM; FIXME INEFFICIENT ACCESS
            L_c[iii1][j] += B1j[ti];                                            // 1 FP LOAD FROM TILE, 1 FP LOAD FROM CACHE, 1 FP ADD, 1 FP STORE TO MEM; FIXME INEFFICIENT ACCESS

            s.add_fp_tileloads( 2);
            s.add_fp_cacheloads(2);
            s.add_fp_adds(      2);
            s.add_fp_memstores( 2);
        }
    }
}

template <
    std::vector<interaction_type> const* __restrict__ IList 
  , std::size_t TileWidth
    >
inline void grid::store_to_L_c(
    integer i_begin
  , integer i_end
  , compute_interactions_tile<TileWidth>& t
  , compute_interactions_stats_t& s
  , std::false_type
    ) noexcept
{}

///////////////////////////////////////////////////////////////////////////////

template <std::size_t TileWidth>
inline void 
set_basis(
    taylor<5, std::array<real, TileWidth>>& A
  , std::array<std::array<real, TileWidth>, NDIM> const& X
  , compute_interactions_stats_t& s
    ) noexcept
{
    alignas(128) std::array<real, TileWidth> d0_array;
    alignas(128) std::array<real, TileWidth> d1_array;
    alignas(128) std::array<real, TileWidth> d2_array;
    alignas(128) std::array<real, TileWidth> d3_array;

    real* __restrict__ d0 = d0_array.data();
    real* __restrict__ d1 = d1_array.data();
    real* __restrict__ d2 = d2_array.data();
    real* __restrict__ d3 = d3_array.data();
    BOOST_ASSUME_ALIGNED(d0, 64);
    BOOST_ASSUME_ALIGNED(d1, 64);
    BOOST_ASSUME_ALIGNED(d2, 64);
    BOOST_ASSUME_ALIGNED(d3, 64);

    real* __restrict__       A0 = A[0].data();
    real const* __restrict__ X0 = X[0].data();
    real const* __restrict__ X1 = X[1].data();
    real const* __restrict__ X2 = X[2].data();
    BOOST_ASSUME_ALIGNED(A0, 64);
    BOOST_ASSUME_ALIGNED(X0, 64);
    BOOST_ASSUME_ALIGNED(X1, 64);
    BOOST_ASSUME_ALIGNED(X2, 64);

    #pragma omp simd 
    for (integer ti = 0; ti < TileWidth; ++ti)                                  // TRIP COUNT: TileWidth; UNIT STRIDE
    {
        real const r2 = sqr(X0[ti]) + sqr(X1[ti]) + sqr(X2[ti]);                // 3 FP LOADS FROM TILE, 1 FP MUL, 2 FP FMA

        s.add_fp_tileloads(3);
        s.add_fp_muls(     1);
        s.add_fp_fmas(     2);

        real const r2inv = (r2 > 0.0) ? ONE / std::max(r2, 1.0e-20) : ZERO;     // 1 FP DIV

        s.add_fp_divs(1);

        real const d0ti = -sqrt(r2inv);                                         // 1 FP SQRT
        real const d1ti = -d0ti * r2inv;                                        // 1 FP MUL
        real const d2ti = real(-3) * d1ti * r2inv;                              // 2 FP MULS
        real const d3ti = real(-5) * d2ti * r2inv;                              // 2 FP MULS

        s.add_fp_sqrts(1);
        s.add_fp_muls( 5);

        A0[ti] = d0ti;                                                          // 1 FP STORE TO TILE
        d0[ti] = d0ti;                                                          // 1 FP STORE TO CACHE
        d1[ti] = d1ti;                                                          // 1 FP STORE TO CACHE
        d2[ti] = d2ti;                                                          // 1 FP STORE TO CACHE
        d3[ti] = d3ti;                                                          // 1 FP STORE TO CACHE

        s.add_fp_tilestores( 1);
        s.add_fp_cachestores(4);
    }

    for (integer j = taylor_sizes[0], a = 0; a != NDIM; ++a, ++j)               // TRIP COUNT: 3
    {
        real const* __restrict__ Xa = X[a].data();
        BOOST_ASSUME_ALIGNED(Xa, 64);

        real* __restrict__ Aj = A[j].data();
        BOOST_ASSUME_ALIGNED(Aj, 64);

        #pragma omp simd 
        for (integer ti = 0; ti < TileWidth; ++ti)                              // TRIP COUNT: 3 * TileWidth; UNIT STRIDE
        {
            Aj[ti] = Xa[ti] * d1[ti];                                           // 1 FP LOAD FROM CACHE, 2 FP LOADS FROM TILE, 1 FP MUL, 1 FP STORE TO TILE

            s.add_fp_cacheloads(1);
            s.add_fp_tileloads( 2);
            s.add_fp_muls(      1);
            s.add_fp_tilestores(1);
        }
    }

    for (integer j = taylor_sizes[1], a = 0; a != NDIM; ++a)                    // TRIP COUNT: 3
    {
        real const* __restrict__ Xa = X[a].data();
        BOOST_ASSUME_ALIGNED(Xa, 64);

        for (integer b = a; b != NDIM; ++b, ++j)                                // TRIP COUNT: 6
        {
            real const* __restrict__ Xb = X[b].data();
            BOOST_ASSUME_ALIGNED(Xb, 64);

            real* __restrict__ Aj = A[j].data();
            BOOST_ASSUME_ALIGNED(Aj, 64);

            #pragma omp simd
            for (integer ti = 0; ti < TileWidth; ++ti)                          // TRIP COUNT: TileWidth; UNIT STRIDE
            {
                Aj[ti] = Xa[ti] * Xb[ti] * d2[ti];                              // 1 FP LOAD FROM CACHE, 3 FP LOADS FROM TILE, 2 FP MUL, 1 FP STORE TO TILE

                s.add_fp_cacheloads(1);
                s.add_fp_tileloads( 3);
                s.add_fp_muls(      2);
                s.add_fp_tilestores(1);
            }
        }
    }

    for (integer j = taylor_sizes[2], a = 0; a != NDIM; ++a)                    // TRIP COUNT: 3
    {
        real const* __restrict__ Xa = X[a].data();
        BOOST_ASSUME_ALIGNED(Xa, 64);

        for (integer b = a; b != NDIM; ++b)                                     // TRIP COUNT: 6
        {
            real const* __restrict__ Xb = X[b].data();
            BOOST_ASSUME_ALIGNED(Xb, 64);

            for (integer c = b; c != NDIM; ++c, ++j)                            // TRIP COUNT: 10
            {
                real const* __restrict__ Xc = X[c].data();
                BOOST_ASSUME_ALIGNED(Xc, 64);

                real* __restrict__ Aj = A[j].data();
                BOOST_ASSUME_ALIGNED(Aj, 64);

                #pragma omp simd
                for (integer ti = 0; ti < TileWidth; ++ti)                      // TRIP COUNT: 10 * TileWidth; UNIT STRIDE
                {
                    Aj[ti] = Xa[ti] * Xb[ti] * Xc[ti] * d3[ti];                 // 1 FP LOAD FROM CACHE, 4 FP LOADS FROM TILE, 3 FP MUL, 1 FP STORE TO TILE
                    
                    s.add_fp_cacheloads(1);
                    s.add_fp_tileloads( 4);
                    s.add_fp_muls(      3);
                    s.add_fp_tilestores(1);
                }
            }
        }
    }

    for (integer j = taylor_sizes[3]; j != taylor_sizes[4]; ++j)                // TRIP COUNT: 15
    {
        real* __restrict__ Aj = A[j].data();
        BOOST_ASSUME_ALIGNED(Aj, 64);

        #pragma omp simd
        for (integer ti = 0; ti < TileWidth; ++ti)                              // TRIP COUNT: 15 * TileWidth; UNIT STRIDE
        {
            Aj[ti] = ZERO;                                                      // 1 FP STORE TO TILE

            s.add_fp_tilestores(1);
        }
    }

    for (integer a = 0; a != NDIM; ++a)                                         // TRIP COUNT: 3
    {
        real const* __restrict__ Xa = X[a].data();
        BOOST_ASSUME_ALIGNED(Xa, 64);

        real* __restrict__ Aaa   = A(a, a).data();
        real* __restrict__ Aaaa  = A(a, a, a).data();
        real* __restrict__ Aaaaa = A(a, a, a, a).data();
        BOOST_ASSUME_ALIGNED(Aaa,   64);
        BOOST_ASSUME_ALIGNED(Aaaa,  64);
        BOOST_ASSUME_ALIGNED(Aaaaa, 64);

        #pragma omp simd 
        for (integer ti = 0; ti < TileWidth; ++ti)                              // TRIP COUNT: 3 * TileWidth; UNIT STRIDE
        {
            auto const d1ti = d1[ti];                                           // 1 FP LOAD FROM CACHE
            auto const d2ti = d2[ti];                                           // 1 FP LOAD FROM CACHE
            auto const d3ti = d3[ti];                                           // 1 FP LOAD FROM CACHE
            auto const Xati = Xa[ti];                                           // 1 FP LOAD FROM TILE

            Aaa[ti]   += d1ti;                                                  // 1 FP LOAD FROM TILE, 1 FP ADD, 1 FP STORE TO TILE
            Aaaa[ti]  += Xati * d2ti;                                           // 1 FP LOAD FROM TILE, 1 FP FMA, 1 FP STORE TO TILE
            Aaaaa[ti] += Xati * Xati * d3ti + 2.0 * d2ti;                       // 1 FP LOAD FROM TILE, 2 FP FMA, 1 FP STORE TO TILE 

            s.add_fp_cacheloads(3);
            s.add_fp_tileloads( 4);
            s.add_fp_adds(      1);
            s.add_fp_muls(      1);
            s.add_fp_fmas(      3);
            s.add_fp_tilestores(3);
        }
    }

    for (integer a = 0; a != NDIM; ++a)                                         // TRIP COUNT: 3
    {
        real const* __restrict__ Xa = X[a].data();
        BOOST_ASSUME_ALIGNED(Xa, 64);

        for (integer b = a; b != NDIM; ++b)                                     // TRIP COUNT: 6
        {
            real const* __restrict__ Xb = X[b].data();
            BOOST_ASSUME_ALIGNED(Xb, 64);

            real* __restrict__ Aaab  = A(a, a, b).data();
            real* __restrict__ Aabb  = A(a, b, b).data();
            real* __restrict__ Aaaab = A(a, a, a, b).data();
            real* __restrict__ Aabbb = A(a, b, b, b).data();
            real* __restrict__ Aaabb = A(a, a, b, b).data();
            BOOST_ASSUME_ALIGNED(Aaab,  64);
            BOOST_ASSUME_ALIGNED(Aabb,  64);
            BOOST_ASSUME_ALIGNED(Aaaab, 64);
            BOOST_ASSUME_ALIGNED(Aabbb, 64);
            BOOST_ASSUME_ALIGNED(Aaabb, 64);

            #pragma omp simd 
            for (integer ti = 0; ti < TileWidth; ++ti)                          // TRIP COUNT: 6 * TileWidth; UNIT STRIDE
            {
                auto const d2ti = d2[ti];                                       // 1 FP LOAD FROM CACHE
                auto const d3ti = d3[ti];                                       // 1 FP LOAD FROM CACHE
                auto const Xati = Xa[ti];                                       // 1 FP LOAD FROM TILE
                auto const Xbti = Xb[ti];                                       // 1 FP LOAD FROM TILE

                auto const Xabd3 = Xati * Xbti * d3ti;                          // 2 FP MULS

                Aaab[ti]  += Xbti * d2ti;                                       // 1 FP LOAD FROM TILE, 1 FP FMA, 1 FP STORE TO TILE
                Aabb[ti]  += Xati * d2ti;                                       // 1 FP LOAD FROM TILE, 1 FP FMA, 1 FP STORE TO TILE
                Aaaab[ti] += Xabd3;                                             // 1 FP LOAD FROM TILE, 1 FP ADD, 1 FP STORE TO TILE
                Aabbb[ti] += Xabd3;                                             // 1 FP LOAD FROM TILE, 1 FP ADD, 1 FP STORE TO TILE
                Aaabb[ti] += d2ti;                                              // 1 FP LOAD FROM TILE, 1 FP ADD, 1 FP STORE TO TILE

                s.add_fp_cacheloads(2);
                s.add_fp_tileloads( 7);
                s.add_fp_adds(      3);
                s.add_fp_muls(      2);
                s.add_fp_fmas(      2);
                s.add_fp_tilestores(5);
            }
        }
    }

    for (integer a = 0; a != NDIM; ++a)                                         // TRIP COUNT: 3
    {
        real const* __restrict__ Xa = X[a].data();
        BOOST_ASSUME_ALIGNED(Xa, 64);

        for (integer b = a; b != NDIM; ++b)                                     // TRIP COUNT: 6
        {
            real const* __restrict__ Xb = X[b].data();
            BOOST_ASSUME_ALIGNED(Xb, 64);

            for (integer c = b; c != NDIM; ++c)                                 // TRIP COUNT: 10
            {
                real const* __restrict__ Xc = X[c].data();
                BOOST_ASSUME_ALIGNED(Xc, 64);

                real* __restrict__ Aaabc = A(a, a, b, c).data();
                real* __restrict__ Aabbc = A(a, b, b, c).data();
                real* __restrict__ Aabcc = A(a, b, c, c).data();
                BOOST_ASSUME_ALIGNED(Aaabc, 64);
                BOOST_ASSUME_ALIGNED(Aabbc, 64);
                BOOST_ASSUME_ALIGNED(Aabcc, 64);

                #pragma omp simd 
                for (integer ti = 0; ti < TileWidth; ++ti)                      // TRIP COUNT: 10 * TileWidth; UNIT STRIDE
                {
                    auto const d3ti = d3[ti];                                   // 1 FP LOAD FROM CACHE
                    auto const Xati = Xa[ti];                                   // 1 FP LOAD FROM TILE
                    auto const Xbti = Xb[ti];                                   // 1 FP LOAD FROM TILE
                    auto const Xcti = Xc[ti];                                   // 1 FP LOAD FROM TILE

                    auto const Xbd3 = Xbti * d3ti;                              // 1 FP MUL

                    Aaabc[ti] += Xbd3 * Xcti;                                   // 1 FP LOAD FROM TILE, 1 FP FMA, 1 FP STORE TO TILE
                    Aabbc[ti] += Xati * Xcti * d3ti;                            // 1 FP LOAD FROM TILE, 1 FP FMA, 1 FP MUL, 1 FP STORE TO TILE
                    Aabcc[ti] += Xati * Xbd3;                                   // 1 FP LOAD FROM TILE, 1 FP FMA, 1 FP STORE TO TILE

                    s.add_fp_cacheloads(1);
                    s.add_fp_tileloads( 6);
                    s.add_fp_muls(      2);
                    s.add_fp_fmas(      3);
                    s.add_fp_tilestores(3);
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

template <
    std::vector<interaction_type> const* __restrict__ IList
  , std::size_t TileWidth
  , ang_con_type AngConKind
  , gsolve_type SolveKind
    >
inline void grid::compute_interactions_non_leaf_tiled(
    integer i_begin
  , integer i_end
  , compute_interactions_tile<TileWidth>& t
  , compute_interactions_stats_t& s
    )
{
    auto& M = *M_ptr;
    auto const& com0 = *com0_ptr;

    BOOST_ASSUME((i_end - i_begin) == TileWidth);

    typename std::conditional<
        1 == TileWidth, scalar_function_tag, vector_function_tag
    >::type constexpr function_type{};
    std::integral_constant<
        bool, ANG_CON_ON == AngConKind
    > constexpr ang_con_is_on{};
    std::integral_constant<
        bool, RHO == SolveKind
    > constexpr is_rho_type{};
    std::integral_constant<
        bool, ANG_CON_ON == AngConKind && RHO == SolveKind
    > constexpr ang_con_is_on_and_is_rho_type{};

    ///////////////////////////////////////////////////////////////////////
    // GATHER 

    for (integer j = 0; j != NDIM; ++j)                                         // TRIP COUNT: 3
    {
        real* __restrict__ dXj = t.dX[j].data();
        BOOST_ASSUME_ALIGNED(dXj, 64);

        #pragma omp simd
        for (integer i = i_begin; i < i_end; ++i)                               // TRIP COUNT: 3 * TileWidth; UNIT STRIDE
        {
            integer const ti = i - i_begin;

            integer const iii0 = (*IList)[i].first;                             // 1 INT LOAD FROM MEM (indirect addressing)
            integer const iii1 = (*IList)[i].second;                            // 1 INT LOAD FROM MEM (indirect addressing)

            real const x = com0[j][iii0];                                       // 1 FP LOAD FROM MEM; FIXME INEFFICIENT ACCESS
            real const y = com0[j][iii1];                                       // 1 FP LOAD FROM MEM; FIXME INEFFICIENT ACCESS
            dXj[ti] = x - y;                                                    // 1 FP ADD, 1 FP STORE TO TILE

            s.add_fp_memloads(  2);
            s.add_fp_adds(      1);
            s.add_fp_tilestores(1);
        }
    }

    for (integer j = 0; j != taylor_sizes[3]; ++j)                              // TRIP COUNT: 20
    {
        real* __restrict__ m0j = t.m0[j].data();
        real* __restrict__ m1j = t.m1[j].data();
        BOOST_ASSUME_ALIGNED(m0j, 64);
        BOOST_ASSUME_ALIGNED(m1j, 64);

        real const* __restrict__ Mj = M[j].data();
        BOOST_ASSUME_ALIGNED(Mj, 64);

        #pragma omp simd
        for (integer i = i_begin; i < i_end; ++i)                               // TRIP COUNT: 20 * TileWidth; UNIT STRIDE
        {
            integer const ti = i - i_begin;

            integer const iii0 = (*IList)[i].first;                             // 1 INT LOAD FROM MEM (indirect addressing)
            integer const iii1 = (*IList)[i].second;                            // 1 INT LOAD FROM MEM (indirect addressing)

            m0j[ti] = Mj[iii1];                                                 // 1 FP LOAD FROM MEM, 1 FP STORE TO TILE
            m1j[ti] = Mj[iii0];                                                 // 1 FP LOAD FROM MEM, 1 FP STORE TO TILE

            s.add_fp_memloads(  2);
            s.add_fp_tilestores(2);
        }
    }

    for (integer a = 0; a != NDIM; ++a)                                         // TRIP COUNT: 10
    {
        real* __restrict__ B0a = t.B0[a].data();
        real* __restrict__ B1a = t.B1[a].data();
        BOOST_ASSUME_ALIGNED(B0a, 64);
        BOOST_ASSUME_ALIGNED(B1a, 64);

        #pragma omp simd 
        for (integer i = i_begin; i < i_end; ++i)                               // TRIP COUNT: 10 * TileWidth; UNIT STRIDE
        {
            integer const ti = i - i_begin;

            B0a[ti] = ZERO;                                                     // 1 FP STORE TO TILE
            B1a[ti] = ZERO;                                                     // 1 FP STORE TO TILE

            s.add_fp_tilestores(2);
        }
    }

    compute_interactions_initialize_n_ang_mom<IList, TileWidth>(i_begin, i_end, t, s, ang_con_is_on_and_is_rho_type, function_type);

    ///////////////////////////////////////////////////////////////////////
    // COMPUTE

    set_basis(t.D, t.dX, s);

    real* __restrict__ m00 = t.m0[0].data();
    real* __restrict__ m10 = t.m1[0].data();
    BOOST_ASSUME_ALIGNED(m00, 64);
    BOOST_ASSUME_ALIGNED(m10, 64);

    real const* __restrict__ m0_ = t.m0().data();
    real const* __restrict__ m1_ = t.m1().data();
    BOOST_ASSUME_ALIGNED(m0_, 64);
    BOOST_ASSUME_ALIGNED(m1_, 64);

    real* __restrict__ D0 = t.D[0].data();
    BOOST_ASSUME_ALIGNED(D0, 64);

    real* __restrict__ A00 = t.A0[0].data();
    real* __restrict__ A10 = t.A1[0].data();
    BOOST_ASSUME_ALIGNED(A00, 64);
    BOOST_ASSUME_ALIGNED(A10, 64);

    // FIXME: We're going to have problems here with the three back-to-back
    // loops writing to A00/A10. We need to fuse them if possible; use
    // masking if we must. This probably has to be hand coded.

    #pragma omp simd
    for (integer ti = 0; ti < TileWidth; ++ti)                                  // TRIP COUNT: TileWidth; UNIT STRIDE
    {
        A00[ti] = m00[ti] * D0[ti];                                             // 3 FP LOADS FROM TILE, 1 FP MUL, 1 FP STORE TO TILE
        A10[ti] = m10[ti] * D0[ti];                                             // 3 FP LOADS FROM TILE, 1 FP MUL, 1 FP STORE TO TILE

        s.add_fp_tileloads( 6);
        s.add_fp_muls(      2);
        s.add_fp_tilestores(2);
    }

    compute_interactions_A0_A1_0<IList, TileWidth>(t, s, is_rho_type); 

    for (integer j = taylor_sizes[1]; j != taylor_sizes[2]; ++j)                // TRIP COUNT: 6
    {
        real const half_factorj = half_factor[j];

        real const* __restrict__ m0j = t.m0[j].data();
        real const* __restrict__ m1j = t.m1[j].data();
        BOOST_ASSUME_ALIGNED(m0j, 64);
        BOOST_ASSUME_ALIGNED(m1j, 64);

        real const* __restrict__ Dj = t.D[j].data();
        BOOST_ASSUME_ALIGNED(Dj, 64);
        
        #pragma omp simd
        for (integer ti = 0; ti < TileWidth; ++ti)                              // TRIP COUNT: 6 * TileWidth; UNIT STRIDE
        {
            auto const tmp = Dj[ti] * half_factorj;                             // 1 FP LOAD FROM TILE, 1 FP MUL
            A00[ti] += m0j[ti] * tmp;                                           // 2 FP LOADS FROM TILE, 1 FP FMA, 1 FP STORE TO TILE
            A10[ti] += m1j[ti] * tmp;                                           // 2 FP LOADS FROM TILE, 1 FP FMA, 1 FP STORE TO TILE

            s.add_fp_tileloads( 5);
            s.add_fp_muls(      1);
            s.add_fp_fmas(      2);
            s.add_fp_tilestores(2);
        }
    }

    for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j)                // TRIP COUNT: 10
    {
        real const sixth_factorj = sixth_factor[j];
 
        real const* __restrict__ m0j = t.m0[j].data();
        real const* __restrict__ m1j = t.m1[j].data();
        BOOST_ASSUME_ALIGNED(m0j, 64);
        BOOST_ASSUME_ALIGNED(m1j, 64);

        real const* __restrict__ Dj = t.D[j].data();
        BOOST_ASSUME_ALIGNED(Dj, 64);

        #pragma omp simd 
        for (integer ti = 0; ti < TileWidth; ++ti)                              // TRIP COUNT: 10 * TileWidth; UNIT STRIDE
        {
            auto const tmp = Dj[ti] * sixth_factorj;                            // 1 FP LOAD FROM TILE, 1 FP MUL
            A00[ti] -= m0j[ti] * tmp;                                           // 2 FP LOADS FROM TILE, 1 FP FMA, 1 FP STORE TO TILE
            A10[ti] += m1j[ti] * tmp;                                           // 2 FP LOADS FROM TILE, 1 FP FMA, 1 FP STORE TO TILE

            s.add_fp_tileloads( 5);
            s.add_fp_muls(      1);
            s.add_fp_fmas(      2);
            s.add_fp_tilestores(2);
        }
    }

    for (integer a = 0; a != NDIM; ++a)                                         // TRIP COUNT: 3
    {
        real const* __restrict__ Da = t.D(a).data();
        BOOST_ASSUME_ALIGNED(Da, 64);

        real* __restrict__ A0a = t.A0(a).data();
        real* __restrict__ A1a = t.A1(a).data();
        BOOST_ASSUME_ALIGNED(A0a, 64);
        BOOST_ASSUME_ALIGNED(A1a, 64);

        #pragma omp simd 
        for (integer ti = 0; ti < TileWidth; ++ti)                              // 3 TRIP COUNT: TileWidth; UNIT STRIDE
        {
            auto const tmp = Da[ti];                                            // 1 FP LOAD FROM TILE
            A0a[ti] =  m0_[ti] * tmp;                                           // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE
            A1a[ti] = -m1_[ti] * tmp;                                           // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE

            s.add_fp_tileloads( 5);
            s.add_fp_fmas(      2);
            s.add_fp_tilestores(2);
        }
    }

    compute_interactions_A0_A1<IList, TileWidth>(t, s, is_rho_type); 

    for (integer a = 0; a != NDIM; ++a)                                         // TRIP COUNT: 3
    {
        real* __restrict__ A0a = t.A0(a).data();
        real* __restrict__ A1a = t.A1(a).data();
        BOOST_ASSUME_ALIGNED(A0a, 64);
        BOOST_ASSUME_ALIGNED(A1a, 64);

        for (integer b = 0; b != NDIM; ++b)                                     // TRIP COUNT: 9
            for (integer c = b; c != NDIM; ++c)                                 // TRIP COUNT: 18
            {
                real const half_factorcb = half_factor(c, b);

                real const* __restrict__ m0cb = t.m0(c, b).data();
                real const* __restrict__ m1cb = t.m1(c, b).data();
                BOOST_ASSUME_ALIGNED(m0cb, 64);
                BOOST_ASSUME_ALIGNED(m1cb, 64);

                real const* __restrict__ Dabc = t.D(a, b, c).data();
                BOOST_ASSUME_ALIGNED(Dabc, 64);

                #pragma omp simd
                for (integer ti = 0; ti < TileWidth; ++ti)                      // TRIP COUNT: 18 * TileWidth; UNIT STRIDE
                {
                    auto const tmp = Dabc[ti] * half_factorcb;                  // 1 FP LOAD FROM TILE, 1 FP MUL
                    A0a[ti] += m0cb[ti] * tmp;                                  // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE
                    A1a[ti] -= m1cb[ti] * tmp;                                  // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE

                    s.add_fp_tileloads( 5);
                    s.add_fp_muls(      1);
                    s.add_fp_fmas(      2);
                    s.add_fp_tilestores(2);
                }
            }
    }

    compute_interactions_B0_B1<IList, TileWidth>(t, s, ang_con_is_on_and_is_rho_type);

    for (integer a = 0; a != NDIM; ++a)                                         // TRIP COUNT: 3
        for (integer b = a; b != NDIM; ++b)                                     // TRIP COUNT: 6
        {
            real const* __restrict__ Dab = t.D(a, b).data();
            BOOST_ASSUME_ALIGNED(Dab, 64);

            real* __restrict__ A0ab = t.A0(a, b).data();
            real* __restrict__ A1ab = t.A1(a, b).data();
            BOOST_ASSUME_ALIGNED(A0ab, 64);
            BOOST_ASSUME_ALIGNED(A1ab, 64);

            #pragma omp simd
            for (integer ti = 0; ti < TileWidth; ++ti)                          // TRIP COUNT: 6 * TileWidth; UNIT STRIDE
            {
                auto const tmp = Dab[ti];                                       // 1 FP LOAD FROM TILE
                A0ab[ti] = m0_[ti] * tmp;                                       // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE
                A1ab[ti] = m1_[ti] * tmp;                                       // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE

                s.add_fp_tileloads( 5);
                s.add_fp_fmas(      2);
                s.add_fp_tilestores(2);
            }
        }

    for (integer a = 0; a != NDIM; ++a)                                         // TRIP COUNT: 3
        for (integer b = a; b != NDIM; ++b)                                     // TRIP COUNT: 6
        {
            real* __restrict__ A0ab = t.A0(a, b).data();
            real* __restrict__ A1ab = t.A1(a, b).data();
            BOOST_ASSUME_ALIGNED(A0ab, 64);
            BOOST_ASSUME_ALIGNED(A1ab, 64);

            for (integer c = 0; c != NDIM; ++c)                                 // TRIP COUNT: 18
            {
                real const* __restrict__ m0c = t.m0(c).data();
                real const* __restrict__ m1c = t.m1(c).data();
                BOOST_ASSUME_ALIGNED(m0c, 64);
                BOOST_ASSUME_ALIGNED(m1c, 64);

                real const* __restrict__ Dabc = t.D(a, b, c).data();
                BOOST_ASSUME_ALIGNED(Dabc, 64);

                #pragma omp simd 
                for (integer ti = 0; ti < TileWidth; ++ti)                      // TRIP COUNT: 18 * TileWidth; UNIT STRIDE
                {
                    auto const tmp = Dabc[ti];                                  // 1 FP LOAD FROM TILE
                    A0ab[ti] -= m0c[ti] * tmp;                                  // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE
                    A1ab[ti] += m1c[ti] * tmp;                                  // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE

                    s.add_fp_tileloads( 5);
                    s.add_fp_fmas(      2);
                    s.add_fp_tilestores(2);
                }
            }
        }

    for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j)                // TRIP COUNT: 10
    {
        real const* __restrict__ Dj = t.D[j].data();
        BOOST_ASSUME_ALIGNED(Dj, 64);

        real* __restrict__ A0j = t.A0[j].data();
        real* __restrict__ A1j = t.A1[j].data();
        BOOST_ASSUME_ALIGNED(A0j, 64);
        BOOST_ASSUME_ALIGNED(A1j, 64);

        #pragma omp simd 
        for (integer ti = 0; ti < TileWidth; ++ti)                              // TRIP COUNT: 10 * TileWidth; UNIT STRIDE
        {
            auto const tmp = Dj[ti];                                            // 1 FP LOAD FROM TILE
            A0j[ti] =  m00[ti] * tmp;                                           // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE
            A1j[ti] = -m10[ti] * tmp;                                           // 2 FP LOADS FROM TILE, 1 FMA, 1 FP STORE TO TILE

            s.add_fp_tileloads( 5);
            s.add_fp_fmas(      2);
            s.add_fp_tilestores(2);
        }
    }

    ///////////////////////////////////////////////////////////////////////
    // SCATTER: Store to L and L_c.

    for (integer j = 0; j != taylor_sizes[3]; ++j)                              // TRIP COUNT: 20 
    {
        real* __restrict__ A0j = t.A0[j].data();
        real* __restrict__ A1j = t.A1[j].data();
        BOOST_ASSUME_ALIGNED(A0j, 64);
        BOOST_ASSUME_ALIGNED(A1j, 64);

        #pragma omp simd 
        for (integer i = i_begin; i < i_end; ++i)                               // TRIP COUNT: 20 * TileWidth; UNIT STRIDE
        {
            integer const ti = i - i_begin;

            integer const iii0 = (*IList)[i].first;                             // 1 INT LOAD FROM MEM (indirect addressing)
            integer const iii1 = (*IList)[i].second;                            // 1 INT LOAD FROM MEM (indirect addressing)

            L[iii0][j] += A0j[ti];                                              // 1 FP LOAD FROM TILE, 1 FP LOAD FROM CACHE, 1 FP ADD, 1 FP STORE TO MEM; FIXME INEFFICIENT ACCESS
            L[iii1][j] += A1j[ti];                                              // 1 FP LOAD FROM TILE, 1 FP LOAD FROM CACHE, 1 FP ADD, 1 FP STORE TO MEM; FIXME INEFFICIENT ACCESS

            s.add_fp_tileloads( 2);
            s.add_fp_cacheloads(2);
            s.add_fp_adds(      2);
            s.add_fp_memstores( 2);
        }
    }

    store_to_L_c<IList, TileWidth>(i_begin, i_end, t, s, ang_con_is_on_and_is_rho_type);
}

template <
    std::vector<interaction_type> const* __restrict__ IList /* lol C# */
  , std::size_t TileWidth
  , ang_con_type AngConKind
  , gsolve_type SolveKind
    >
inline compute_interactions_stats_t grid::compute_interactions_non_leaf()
{
    static_assert(0 < TileWidth, "TileWidth cannot be negative.");
    static_assert(0 == (TileWidth % 16), "TileWidth must be a multiple of 16.");

    std::integral_constant<
        bool, ANG_CON_ON == AngConKind
    > constexpr ang_con_is_on{};

    // L stores the gravitational potential.
    // #10 in the paper ([Bryce] FIXME: Link to the paper) [Dominic].
    #if defined(HPX_HAVE_DATAPAR)
        hpx::parallel::fill(hpx::parallel::execution::dataseq,
                            L.begin(), L.end(), ZERO);
    #else
        std::fill(std::begin(L), std::end(L), ZERO);
    #endif

    // L_c stores the correction for angular momentum.
    // #20 in the paper ([Bryce] FIXME: Link to the paper) [Dominic].
    compute_interactions_initialize_L_c(ang_con_is_on);

    auto const ilist_primary_loop_size = (IList->size() / TileWidth) * TileWidth;

    compute_interactions_stats_t s;

    { // Vector primary loop.
        auto tile = std::make_unique<compute_interactions_tile<TileWidth>>();

        hpx::util::high_resolution_timer timer;

        for (integer i_begin = 0; i_begin != ilist_primary_loop_size; i_begin += TileWidth)
        {
            integer const i_end = i_begin + TileWidth;

            compute_interactions_non_leaf_tiled<IList, TileWidth, AngConKind, SolveKind>(i_begin, i_end, *tile, s);
        }

        s.add_time(timer.elapsed());
    }

    { // Scalar remainder loop.
        compute_interactions_stats_t dummy;

        auto tile = std::make_unique<compute_interactions_tile<1>>();

        for (integer i_begin = ilist_primary_loop_size; i_begin != IList->size(); ++i_begin)
        {
            integer const i_end = i_begin + 1;

            compute_interactions_non_leaf_tiled<IList, 1, AngConKind, SolveKind>(i_begin, i_end, *tile, dummy);
        }
    }

    return s;
}

#endif // OCTOTIGER_82BA7119_59FB_4018_88F8_AF18F969220C

