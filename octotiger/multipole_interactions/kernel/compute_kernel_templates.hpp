//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/cuda_util/cuda_global_def.hpp"

#if defined(KOKKOS_ENABLE_SYCL) && defined( __SYCL_DEVICE_ONLY__)
// Workaround to use sycl::sqrt with double types on GPU
// when using sycl.
#include "simd_common.hpp"
#include <simd.hpp>
#endif

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

#if defined(__clang__) // Clang can handle cuda device constexpr better
        constexpr double factor[20] = {1.000000, 1.000000, 1.000000, 1.000000,
            1.000000, 2.000000, 2.000000, 1.000000, 2.000000, 1.000000, 1.000000, 3.000000,
            3.000000, 3.000000, 6.000000, 3.000000, 1.000000, 3.000000, 3.000000, 1.00000};
        constexpr double factor_half[20] = {factor[0] / 2.0, factor[1] / 2.0,
            factor[2] / 2.0, factor[3] / 2.0, factor[4] / 2.0, factor[5] / 2.0, factor[6] / 2.0,
            factor[7] / 2.0, factor[8] / 2.0, factor[9] / 2.0, factor[10] / 2.0, factor[11] / 2.0,
            factor[12] / 2.0, factor[13] / 2.0, factor[14] / 2.0, factor[15] / 2.0,
            factor[16] / 2.0, factor[17] / 2.0, factor[18] / 2.0, factor[19] / 2.0};
        constexpr double factor_sixth[20] = {factor[0] / 6.0, factor[1] / 6.0,
            factor[2] / 6.0, factor[3] / 6.0, factor[4] / 6.0, factor[5] / 6.0, factor[6] / 6.0,
            factor[7] / 6.0, factor[8] / 6.0, factor[9] / 6.0, factor[10] / 6.0, factor[11] / 6.0,
            factor[12] / 6.0, factor[13] / 6.0, factor[14] / 6.0, factor[15] / 6.0,
            factor[16] / 6.0, factor[17] / 6.0, factor[18] / 6.0, factor[19] / 6.0};
#else
        CUDA_CALLABLE_METHOD const double factor[20] = {1.000000, 1.000000, 1.000000, 1.000000,
            1.000000, 2.000000, 2.000000, 1.000000, 2.000000, 1.000000, 1.000000, 3.000000,
            3.000000, 3.000000, 6.000000, 3.000000, 1.000000, 3.000000, 3.000000, 1.00000};
	#ifdef __NVCC__
        CUDA_CALLABLE_METHOD const double factor_half[20] = {1.000000 / 2.0, 1.000000 / 2.0, 1.000000 / 2.0, 1.000000 /2.0 ,
            1.000000 / 2.0 , 2.000000 / 2.0, 2.000000 / 2.0 , 1.000000 / 2.0 , 2.000000 / 2.0 , 1.000000 / 2.0, 1.000000 / 2.0, 3.000000 / 2.0,
            3.000000 / 2.0, 3.000000 / 2.0, 6.000000 / 2.0, 3.000000 / 2.0, 1.000000 / 2.0, 3.000000 / 2.0, 3.000000 / 2.0, 1.00000 / 2.0};

        CUDA_CALLABLE_METHOD const double factor_sixth[20] = {1.000000 / 6.0, 1.000000 / 6.0, 1.000000 / 6.0, 1.000000 / 6.0,
            1.000000 / 6.0, 2.000000 / 6.0, 2.000000 / 6.0, 1.000000 / 6.0, 2.000000 / 6.0, 1.000000 / 6.0, 1.000000 / 6.0, 3.000000 / 6.0,
            3.000000 / 6.0, 3.000000 / 6.0, 6.000000 / 6.0, 3.000000 / 6.0, 1.000000 / 6.0, 3.000000 / 6.0, 3.000000 / 6.0, 1.00000 / 6.0};
	#else
        CUDA_CALLABLE_METHOD const double factor_half[20] = {factor[0] / 2.0, factor[1] / 2.0,
            factor[2] / 2.0, factor[3] / 2.0, factor[4] / 2.0, factor[5] / 2.0, factor[6] / 2.0,
            factor[7] / 2.0, factor[8] / 2.0, factor[9] / 2.0, factor[10] / 2.0, factor[11] / 2.0,
            factor[12] / 2.0, factor[13] / 2.0, factor[14] / 2.0, factor[15] / 2.0,
            factor[16] / 2.0, factor[17] / 2.0, factor[18] / 2.0, factor[19] / 2.0};
        CUDA_CALLABLE_METHOD const double factor_sixth[20] = {factor[0] / 6.0, factor[1] / 6.0,
            factor[2] / 6.0, factor[3] / 6.0, factor[4] / 6.0, factor[5] / 6.0, factor[6] / 6.0,
            factor[7] / 6.0, factor[8] / 6.0, factor[9] / 6.0, factor[10] / 6.0, factor[11] / 6.0,
            factor[12] / 6.0, factor[13] / 6.0, factor[14] / 6.0, factor[15] / 6.0,
            factor[16] / 6.0, factor[17] / 6.0, factor[18] / 6.0, factor[19] / 6.0};
	#endif
#endif

        template <typename T>
        CUDA_GLOBAL_METHOD const inline T sqr(T const& val) noexcept {
            return val * val;
        }

        template <typename T, typename func>
        CUDA_GLOBAL_METHOD inline void compute_d_factors(T& d2, T& d3, T& X_00, T& X_11, T& X_22,
            T (&D_lower)[20], const T (&dX)[NDIM], func&& max) noexcept {
            X_00 = dX[0] * dX[0];
            X_11 = dX[1] * dX[1];
            X_22 = dX[2] * dX[2];

            T r2 = X_00 + X_11 + X_22;
            T r2inv = T(1.0) / max(r2, T(1.0e-20));

#if defined(KOKKOS_ENABLE_SYCL) && defined( __SYCL_DEVICE_ONLY__)
            // Workaround to use sycl::sqrt with double types on GPU
            // when using sycl. 
            double r2inv_tmp; 
            r2inv.copy_to(&r2inv_tmp, SIMD_NAMESPACE::element_aligned_tag{});
            T d0 = -sycl::sqrt(r2inv_tmp);
#else
            T d0 = -sqrt(r2inv);
#endif
            T d1 = -d0 * r2inv;
            d2 = -3.0 * d1 * r2inv;
            d3 = -5.0 * d2 * r2inv;

            D_lower[0] = d0;
            D_lower[1] = dX[0] * d1;
            D_lower[2] = dX[1] * d1;
            D_lower[3] = dX[2] * d1;

            const T X_12 = dX[1] * dX[2];
            const T X_01 = dX[0] * dX[1];
            const T X_02 = dX[0] * dX[2];

            D_lower[4] = d2 * X_00;
            D_lower[4] += d1;
            D_lower[5] = d2 * X_01;
            D_lower[6] = d2 * X_02;

            D_lower[7] = d2 * X_11;
            D_lower[7] += d1;
            D_lower[8] = d2 * X_12;

            D_lower[9] = d2 * X_22;
            D_lower[9] += d1;

            D_lower[10] = d3 * X_00 * dX[0];
            const T d2_X0 = d2 * dX[0];
            D_lower[10] += 3.0 * d2_X0;
            D_lower[11] = d3 * X_00 * dX[1];
            D_lower[11] += d2 * dX[1];
            D_lower[12] = d3 * X_00 * dX[2];
            D_lower[12] += d2 * dX[2];

            D_lower[13] = d3 * dX[0] * X_11;
            D_lower[13] += d2 * dX[0];
            D_lower[14] = d3 * dX[0] * X_12;

            D_lower[15] = d3 * dX[0] * X_22;
            D_lower[15] += d2_X0;

            D_lower[16] = d3 * X_11 * dX[1];
            const T d2_X1 = d2 * dX[1];
            D_lower[16] += 3.0 * d2_X1;

            D_lower[17] = d3 * X_11 * dX[2];
            D_lower[17] += d2 * dX[2];

            D_lower[18] = d3 * dX[1] * X_22;
            D_lower[18] += d2 * dX[1];

            D_lower[19] = d3 * X_22 * dX[2];
            const T d2_X2 = d2 * dX[2];
            D_lower[19] += 3.0 * d2_X2;
        }

        template <typename T>
        CUDA_GLOBAL_METHOD inline void compute_interaction_multipole_non_rho(
            const T (&m_partner)[20], T (&tmpstore)[20], const T (&D_lower)[20]) noexcept {
            tmpstore[0] += m_partner[4] * (D_lower[4] * factor_half[4]);
            tmpstore[1] += m_partner[4] * (D_lower[10] * factor_half[4]);
            tmpstore[2] += m_partner[4] * (D_lower[11] * factor_half[4]);
            tmpstore[3] += m_partner[4] * (D_lower[12] * factor_half[4]);

            tmpstore[0] += m_partner[5] * (D_lower[5] * factor_half[5]);
            tmpstore[1] += m_partner[5] * (D_lower[11] * factor_half[5]);
            tmpstore[2] += m_partner[5] * (D_lower[13] * factor_half[5]);
            tmpstore[3] += m_partner[5] * (D_lower[14] * factor_half[5]);

            tmpstore[0] += m_partner[6] * (D_lower[6] * factor_half[6]);
            tmpstore[1] += m_partner[6] * (D_lower[12] * factor_half[6]);
            tmpstore[2] += m_partner[6] * (D_lower[14] * factor_half[6]);
            tmpstore[3] += m_partner[6] * (D_lower[15] * factor_half[6]);

            tmpstore[0] += m_partner[7] * (D_lower[7] * factor_half[7]);
            tmpstore[1] += m_partner[7] * (D_lower[13] * factor_half[7]);
            tmpstore[2] += m_partner[7] * (D_lower[16] * factor_half[7]);
            tmpstore[3] += m_partner[7] * (D_lower[17] * factor_half[7]);

            tmpstore[0] += m_partner[8] * (D_lower[8] * factor_half[8]);
            tmpstore[1] += m_partner[8] * (D_lower[14] * factor_half[8]);
            tmpstore[2] += m_partner[8] * (D_lower[17] * factor_half[8]);
            tmpstore[3] += m_partner[8] * (D_lower[18] * factor_half[8]);

            tmpstore[0] += m_partner[9] * (D_lower[9] * factor_half[9]);
            tmpstore[1] += m_partner[9] * (D_lower[15] * factor_half[9]);
            tmpstore[2] += m_partner[9] * (D_lower[18] * factor_half[9]);
            tmpstore[3] += m_partner[9] * (D_lower[19] * factor_half[9]);

            tmpstore[0] -= m_partner[10] * (D_lower[10] * factor_sixth[10]);
            tmpstore[0] -= m_partner[11] * (D_lower[11] * factor_sixth[11]);
            tmpstore[0] -= m_partner[12] * (D_lower[12] * factor_sixth[12]);
            tmpstore[0] -= m_partner[13] * (D_lower[13] * factor_sixth[13]);
            tmpstore[0] -= m_partner[14] * (D_lower[14] * factor_sixth[14]);
            tmpstore[0] -= m_partner[15] * (D_lower[15] * factor_sixth[15]);
            tmpstore[0] -= m_partner[16] * (D_lower[16] * factor_sixth[16]);
            tmpstore[0] -= m_partner[17] * (D_lower[17] * factor_sixth[17]);
            tmpstore[0] -= m_partner[18] * (D_lower[18] * factor_sixth[18]);
            tmpstore[0] -= m_partner[19] * (D_lower[19] * factor_sixth[19]);

            tmpstore[4] += m_partner[0] * D_lower[4];
            tmpstore[5] += m_partner[0] * D_lower[5];
            tmpstore[6] += m_partner[0] * D_lower[6];
            tmpstore[7] += m_partner[0] * D_lower[7];
            tmpstore[8] += m_partner[0] * D_lower[8];
            tmpstore[9] += m_partner[0] * D_lower[9];

            tmpstore[4] -= m_partner[1] * D_lower[10];
            tmpstore[5] -= m_partner[1] * D_lower[11];
            tmpstore[6] -= m_partner[1] * D_lower[12];
            tmpstore[7] -= m_partner[1] * D_lower[13];
            tmpstore[8] -= m_partner[1] * D_lower[14];
            tmpstore[9] -= m_partner[1] * D_lower[15];

            tmpstore[4] -= m_partner[2] * D_lower[11];
            tmpstore[5] -= m_partner[2] * D_lower[13];
            tmpstore[6] -= m_partner[2] * D_lower[14];
            tmpstore[7] -= m_partner[2] * D_lower[16];
            tmpstore[8] -= m_partner[2] * D_lower[17];
            tmpstore[9] -= m_partner[2] * D_lower[18];

            tmpstore[4] -= m_partner[3] * D_lower[12];
            tmpstore[5] -= m_partner[3] * D_lower[14];
            tmpstore[6] -= m_partner[3] * D_lower[15];
            tmpstore[7] -= m_partner[3] * D_lower[17];
            tmpstore[8] -= m_partner[3] * D_lower[18];
            tmpstore[9] -= m_partner[3] * D_lower[19];
            tmpstore[10] += m_partner[0] * D_lower[10];
            tmpstore[11] += m_partner[0] * D_lower[11];
            tmpstore[12] += m_partner[0] * D_lower[12];
            tmpstore[13] += m_partner[0] * D_lower[13];
            tmpstore[14] += m_partner[0] * D_lower[14];
            tmpstore[15] += m_partner[0] * D_lower[15];
            tmpstore[16] += m_partner[0] * D_lower[16];
            tmpstore[17] += m_partner[0] * D_lower[17];
            tmpstore[18] += m_partner[0] * D_lower[18];
            tmpstore[19] += m_partner[0] * D_lower[19];
        }

        template <typename T>
        CUDA_GLOBAL_METHOD inline void compute_interaction_multipole_rho(const T& d2, const T& d3,
            const T& X_00, const T& X_11, const T& X_22, const T (&m_partner)[20],
            const T (&m_cell)[20], const T (&dX)[NDIM], T (&tmp_corrections)[NDIM]) noexcept {
            T n0_constant = m_partner[0] / m_cell[0];

            T D_upper[15];

            D_upper[0] = dX[0] * dX[0] * d3 + 2.0 * d2;
            const T d3_X00 = d3 * X_00;
            D_upper[0] += d2;
            D_upper[0] += 5.0 * d3_X00;
            const T d3_X01 = d3 * dX[0] * dX[1];
            D_upper[1] = 3.0 * d3_X01;
            const T d3_X02 = d3 * dX[0] * dX[2];
            D_upper[2] = 3.0 * d3_X02;
            T n0_tmp = m_partner[10] - m_cell[10] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[0] * factor_sixth[10]);
            tmp_corrections[1] -= n0_tmp * (D_upper[1] * factor_sixth[10]);
            tmp_corrections[2] -= n0_tmp * (D_upper[2] * factor_sixth[10]);

            D_upper[3] = d2;
            const T d3_X11 = d3 * X_11;
            D_upper[3] += d3_X11;
            D_upper[3] += d3 * X_00;
            const T d3_X12 = d3 * dX[1] * dX[2];
            D_upper[4] = d3_X12;

            n0_tmp = m_partner[11] - m_cell[11] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[1] * factor_sixth[11]);
            tmp_corrections[1] -= n0_tmp * (D_upper[3] * factor_sixth[11]);
            tmp_corrections[2] -= n0_tmp * (D_upper[4] * factor_sixth[11]);

            D_upper[5] = d2;
            const T d3_X22 = d3 * X_22;
            D_upper[5] += d3_X22;
            D_upper[5] += d3_X00;

            n0_tmp = m_partner[12] - m_cell[12] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[2] * factor_sixth[12]);
            tmp_corrections[1] -= n0_tmp * (D_upper[4] * factor_sixth[12]);
            tmp_corrections[2] -= n0_tmp * (D_upper[5] * factor_sixth[12]);

            D_upper[6] = 3.0 * d3_X01;
            D_upper[7] = d3 * dX[0] * dX[2];

            n0_tmp = m_partner[13] - m_cell[13] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[3] * factor_sixth[13]);
            tmp_corrections[1] -= n0_tmp * (D_upper[6] * factor_sixth[13]);
            tmp_corrections[2] -= n0_tmp * (D_upper[7] * factor_sixth[13]);

            D_upper[8] = d3 * dX[0] * dX[1];

            n0_tmp = m_partner[14] - m_cell[14] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[4] * factor_sixth[14]);
            tmp_corrections[1] -= n0_tmp * (D_upper[7] * factor_sixth[14]);
            tmp_corrections[2] -= n0_tmp * (D_upper[8] * factor_sixth[14]);

            D_upper[9] = 3.0 * d3_X02;

            n0_tmp = m_partner[15] - m_cell[15] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[5] * factor_sixth[15]);
            tmp_corrections[1] -= n0_tmp * (D_upper[8] * factor_sixth[15]);
            tmp_corrections[2] -= n0_tmp * (D_upper[9] * factor_sixth[15]);

            D_upper[10] = dX[1] * dX[1] * d3 + 2.0 * d2;
            D_upper[10] += d2;
            D_upper[10] += 5.0 * d3_X11;

            D_upper[11] = 3.0 * d3_X12;

            n0_tmp = m_partner[16] - m_cell[16] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[6] * factor_sixth[16]);
            tmp_corrections[1] -= n0_tmp * (D_upper[10] * factor_sixth[16]);
            tmp_corrections[2] -= n0_tmp * (D_upper[11] * factor_sixth[16]);

            D_upper[12] = d2;
            D_upper[12] += d3_X22;
            D_upper[12] += d3_X11;

            n0_tmp = m_partner[17] - m_cell[17] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[7] * factor_sixth[17]);
            tmp_corrections[1] -= n0_tmp * (D_upper[11] * factor_sixth[17]);
            tmp_corrections[2] -= n0_tmp * (D_upper[12] * factor_sixth[17]);

            D_upper[13] = 3.0 * d3_X12;

            n0_tmp = m_partner[18] - m_cell[18] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[8] * factor_sixth[18]);
            tmp_corrections[1] -= n0_tmp * (D_upper[12] * factor_sixth[18]);
            tmp_corrections[2] -= n0_tmp * (D_upper[13] * factor_sixth[18]);

            D_upper[14] = dX[2] * dX[2] * d3 + 2.0 * d2;
            D_upper[14] += d2;
            D_upper[14] += 5.0 * d3_X22;

            n0_tmp = m_partner[19] - m_cell[19] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[9] * factor_sixth[19]);
            tmp_corrections[1] -= n0_tmp * (D_upper[13] * factor_sixth[19]);
            tmp_corrections[2] -= n0_tmp * (D_upper[14] * factor_sixth[19]);
        }

        template <typename T, typename func>
        CUDA_GLOBAL_METHOD inline void compute_kernel_rho(T (&X)[NDIM], T (&Y)[NDIM],
            T (&m_partner)[20], T (&tmpstore)[20], T (&tmp_corrections)[3], T (&m_cell)[20],
            func&& max) noexcept {
            T dX[NDIM];
            dX[0] = X[0] - Y[0];
            dX[1] = X[1] - Y[1];
            dX[2] = X[2] - Y[2];

            T X_00, X_11, X_22;
            T d2, d3;
            T D_lower[20];

            compute_d_factors(d2, d3, X_00, X_11, X_22, D_lower, dX, max);

            tmpstore[0] += m_partner[0] * D_lower[0];
            tmpstore[1] += m_partner[0] * D_lower[1];
            tmpstore[2] += m_partner[0] * D_lower[2];
            tmpstore[3] += m_partner[0] * D_lower[3];
            compute_interaction_multipole_non_rho(m_partner, tmpstore, D_lower);

            compute_interaction_multipole_rho(
                d2, d3, X_00, X_11, X_22, m_partner, m_cell, dX, tmp_corrections);
        }

        template <typename T, typename func>
        CUDA_GLOBAL_METHOD inline void compute_kernel_non_rho(T (&X)[NDIM], T (&Y)[NDIM],
            T (&m_partner)[20], T (&tmpstore)[20], func&& max) noexcept {
            T dX[NDIM];
            dX[0] = X[0] - Y[0];
            dX[1] = X[1] - Y[1];
            dX[2] = X[2] - Y[2];

            T X_00, X_11, X_22;
            T d2, d3;
            T D_lower[20];
            compute_d_factors(d2, d3, X_00, X_11, X_22, D_lower, dX, max);

            tmpstore[0] += m_partner[0] * D_lower[0];
            tmpstore[1] += m_partner[0] * D_lower[1];
            tmpstore[2] += m_partner[0] * D_lower[2];
            tmpstore[3] += m_partner[0] * D_lower[3];
            tmpstore[0] -= m_partner[1] * D_lower[1];
            tmpstore[1] -= m_partner[1] * D_lower[4];
            tmpstore[1] -= m_partner[1] * D_lower[5];
            tmpstore[1] -= m_partner[1] * D_lower[6];
            tmpstore[0] -= m_partner[2] * D_lower[2];
            tmpstore[2] -= m_partner[2] * D_lower[5];
            tmpstore[2] -= m_partner[2] * D_lower[7];
            tmpstore[2] -= m_partner[2] * D_lower[8];
            tmpstore[0] -= m_partner[3] * D_lower[3];
            tmpstore[3] -= m_partner[3] * D_lower[6];
            tmpstore[3] -= m_partner[3] * D_lower[8];
            tmpstore[3] -= m_partner[3] * D_lower[9];
            compute_interaction_multipole_non_rho(m_partner, tmpstore, D_lower);
        }
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
