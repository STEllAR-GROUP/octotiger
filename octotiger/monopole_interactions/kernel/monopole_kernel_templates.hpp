//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

//#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/cuda_util/cuda_global_def.hpp"
#include "octotiger/multipole_interactions/kernel/compute_kernel_templates.hpp"

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        template <typename T>
        CUDA_GLOBAL_METHOD inline void compute_monopole_interaction(const T& monopole,
            T (&tmpstore)[4], const T (&four)[4], const T (&d_components)[2]) noexcept {
            tmpstore[0] = tmpstore[0] + four[0] * monopole * d_components[0];
            tmpstore[1] = tmpstore[1] + four[1] * monopole * d_components[1];
            tmpstore[2] = tmpstore[2] + four[2] * monopole * d_components[1];
            tmpstore[3] = tmpstore[3] + four[3] * monopole * d_components[1];
        }
        template <typename T>
        CUDA_GLOBAL_METHOD inline void compute_interaction_p2m_non_rho(
            const T (&m_partner)[20], T (&tmpstore)[4], const T (&D_lower)[20]) noexcept {
            tmpstore[0] += m_partner[4] * (D_lower[4] * multipole_interactions::factor_half[4]);
            tmpstore[1] += m_partner[4] * (D_lower[10] * multipole_interactions::factor_half[4]);
            tmpstore[2] += m_partner[4] * (D_lower[11] * multipole_interactions::factor_half[4]);
            tmpstore[3] += m_partner[4] * (D_lower[12] * multipole_interactions::factor_half[4]);

            tmpstore[0] += m_partner[5] * (D_lower[5] * multipole_interactions::factor_half[5]);
            tmpstore[1] += m_partner[5] * (D_lower[11] * multipole_interactions::factor_half[5]);
            tmpstore[2] += m_partner[5] * (D_lower[13] * multipole_interactions::factor_half[5]);
            tmpstore[3] += m_partner[5] * (D_lower[14] * multipole_interactions::factor_half[5]);

            tmpstore[0] += m_partner[6] * (D_lower[6] * multipole_interactions::factor_half[6]);
            tmpstore[1] += m_partner[6] * (D_lower[12] * multipole_interactions::factor_half[6]);
            tmpstore[2] += m_partner[6] * (D_lower[14] * multipole_interactions::factor_half[6]);
            tmpstore[3] += m_partner[6] * (D_lower[15] * multipole_interactions::factor_half[6]);

            tmpstore[0] += m_partner[7] * (D_lower[7] * multipole_interactions::factor_half[7]);
            tmpstore[1] += m_partner[7] * (D_lower[13] * multipole_interactions::factor_half[7]);
            tmpstore[2] += m_partner[7] * (D_lower[16] * multipole_interactions::factor_half[7]);
            tmpstore[3] += m_partner[7] * (D_lower[17] * multipole_interactions::factor_half[7]);

            tmpstore[0] += m_partner[8] * (D_lower[8] * multipole_interactions::factor_half[8]);
            tmpstore[1] += m_partner[8] * (D_lower[14] * multipole_interactions::factor_half[8]);
            tmpstore[2] += m_partner[8] * (D_lower[17] * multipole_interactions::factor_half[8]);
            tmpstore[3] += m_partner[8] * (D_lower[18] * multipole_interactions::factor_half[8]);

            tmpstore[0] += m_partner[9] * (D_lower[9] * multipole_interactions::factor_half[9]);
            tmpstore[1] += m_partner[9] * (D_lower[15] * multipole_interactions::factor_half[9]);
            tmpstore[2] += m_partner[9] * (D_lower[18] * multipole_interactions::factor_half[9]);
            tmpstore[3] += m_partner[9] * (D_lower[19] * multipole_interactions::factor_half[9]);

            tmpstore[0] -= m_partner[10] * (D_lower[10] * multipole_interactions::factor_sixth[10]);
            tmpstore[0] -= m_partner[11] * (D_lower[11] * multipole_interactions::factor_sixth[11]);
            tmpstore[0] -= m_partner[12] * (D_lower[12] * multipole_interactions::factor_sixth[12]);
            tmpstore[0] -= m_partner[13] * (D_lower[13] * multipole_interactions::factor_sixth[13]);
            tmpstore[0] -= m_partner[14] * (D_lower[14] * multipole_interactions::factor_sixth[14]);
            tmpstore[0] -= m_partner[15] * (D_lower[15] * multipole_interactions::factor_sixth[15]);
            tmpstore[0] -= m_partner[16] * (D_lower[16] * multipole_interactions::factor_sixth[16]);
            tmpstore[0] -= m_partner[17] * (D_lower[17] * multipole_interactions::factor_sixth[17]);
            tmpstore[0] -= m_partner[18] * (D_lower[18] * multipole_interactions::factor_sixth[18]);
            tmpstore[0] -= m_partner[19] * (D_lower[19] * multipole_interactions::factor_sixth[19]);
        }

        template <typename T>
        CUDA_GLOBAL_METHOD inline void compute_interaction_p2m_rho(const T& d2, const T& d3,
            const T& X_00, const T& X_11, const T& X_22, const T (&m_partner)[20],
            const T (&dX)[NDIM], T (&tmp_corrections)[NDIM]) noexcept {
            T D_upper[15];

            D_upper[0] = dX[0] * dX[0] * d3 + 2.0 * d2;
            const T d3_X00 = d3 * X_00;
            D_upper[0] += d2;
            D_upper[0] += 5.0 * d3_X00;
            const T d3_X01 = d3 * dX[0] * dX[1];
            D_upper[1] = 3.0 * d3_X01;
            const T d3_X02 = d3 * dX[0] * dX[2];
            D_upper[2] = 3.0 * d3_X02;

            tmp_corrections[0] -=
                m_partner[10] * (D_upper[0] * multipole_interactions::factor_sixth[10]);
            tmp_corrections[1] -=
                m_partner[10] * (D_upper[1] * multipole_interactions::factor_sixth[10]);
            tmp_corrections[2] -=
                m_partner[10] * (D_upper[2] * multipole_interactions::factor_sixth[10]);

            D_upper[3] = d2;
            const T d3_X11 = d3 * X_11;
            D_upper[3] += d3_X11;
            D_upper[3] += d3 * X_00;
            const T d3_X12 = d3 * dX[1] * dX[2];
            D_upper[4] = d3_X12;

            tmp_corrections[0] -=
                m_partner[11] * (D_upper[1] * multipole_interactions::factor_sixth[11]);
            tmp_corrections[1] -=
                m_partner[11] * (D_upper[3] * multipole_interactions::factor_sixth[11]);
            tmp_corrections[2] -=
                m_partner[11] * (D_upper[4] * multipole_interactions::factor_sixth[11]);

            D_upper[5] = d2;
            const T d3_X22 = d3 * X_22;
            D_upper[5] += d3_X22;
            D_upper[5] += d3_X00;

            tmp_corrections[0] -=
                m_partner[12] * (D_upper[2] * multipole_interactions::factor_sixth[12]);
            tmp_corrections[1] -=
                m_partner[12] * (D_upper[4] * multipole_interactions::factor_sixth[12]);
            tmp_corrections[2] -=
                m_partner[12] * (D_upper[5] * multipole_interactions::factor_sixth[12]);

            D_upper[6] = 3.0 * d3_X01;
            D_upper[7] = d3 * dX[0] * dX[2];

            tmp_corrections[0] -=
                m_partner[13] * (D_upper[3] * multipole_interactions::factor_sixth[13]);
            tmp_corrections[1] -=
                m_partner[13] * (D_upper[6] * multipole_interactions::factor_sixth[13]);
            tmp_corrections[2] -=
                m_partner[13] * (D_upper[7] * multipole_interactions::factor_sixth[13]);

            D_upper[8] = d3 * dX[0] * dX[1];

            tmp_corrections[0] -=
                m_partner[14] * (D_upper[4] * multipole_interactions::factor_sixth[14]);
            tmp_corrections[1] -=
                m_partner[14] * (D_upper[7] * multipole_interactions::factor_sixth[14]);
            tmp_corrections[2] -=
                m_partner[14] * (D_upper[8] * multipole_interactions::factor_sixth[14]);

            D_upper[9] = 3.0 * d3_X02;

            tmp_corrections[0] -=
                m_partner[15] * (D_upper[5] * multipole_interactions::factor_sixth[15]);
            tmp_corrections[1] -=
                m_partner[15] * (D_upper[8] * multipole_interactions::factor_sixth[15]);
            tmp_corrections[2] -=
                m_partner[15] * (D_upper[9] * multipole_interactions::factor_sixth[15]);

            D_upper[10] = dX[1] * dX[1] * d3 + 2.0 * d2;
            D_upper[10] += d2;
            D_upper[10] += 5.0 * d3_X11;
            D_upper[11] = 3.0 * d3_X12;

            tmp_corrections[0] -=
                m_partner[16] * (D_upper[6] * multipole_interactions::factor_sixth[16]);
            tmp_corrections[1] -=
                m_partner[16] * (D_upper[10] * multipole_interactions::factor_sixth[16]);
            tmp_corrections[2] -=
                m_partner[16] * (D_upper[11] * multipole_interactions::factor_sixth[16]);

            D_upper[12] = d2;
            D_upper[12] += d3_X22;
            D_upper[12] += d3_X11;

            tmp_corrections[0] -=
                m_partner[17] * (D_upper[7] * multipole_interactions::factor_sixth[17]);
            tmp_corrections[1] -=
                m_partner[17] * (D_upper[11] * multipole_interactions::factor_sixth[17]);
            tmp_corrections[2] -=
                m_partner[17] * (D_upper[12] * multipole_interactions::factor_sixth[17]);

            D_upper[13] = 3.0 * d3_X12;

            tmp_corrections[0] -=
                m_partner[18] * (D_upper[8] * multipole_interactions::factor_sixth[18]);
            tmp_corrections[1] -=
                m_partner[18] * (D_upper[12] * multipole_interactions::factor_sixth[18]);
            tmp_corrections[2] -=
                m_partner[18] * (D_upper[13] * multipole_interactions::factor_sixth[18]);

            D_upper[14] = dX[2] * dX[2] * d3 + 2.0 * d2;
            D_upper[14] += d2;
            D_upper[14] += 5.0 * d3_X22;

            tmp_corrections[0] -=
                m_partner[19] * (D_upper[9] * multipole_interactions::factor_sixth[19]);
            tmp_corrections[1] -=
                m_partner[19] * (D_upper[13] * multipole_interactions::factor_sixth[19]);
            tmp_corrections[2] -=
                m_partner[19] * (D_upper[14] * multipole_interactions::factor_sixth[19]);
        }
        template <typename T, typename func>
        CUDA_GLOBAL_METHOD inline void compute_kernel_p2m_non_rho(
            T (&X)[NDIM], T (&Y)[NDIM], T (&m_partner)[20], T (&tmpstore)[4], func&& max) noexcept {
            T dX[NDIM];
            dX[0] = X[0] - Y[0];
            dX[1] = X[1] - Y[1];
            dX[2] = X[2] - Y[2];

            T X_00, X_11, X_22;
            T d2, d3;
            T D_lower[20];
            multipole_interactions::compute_d_factors(d2, d3, X_00, X_11, X_22, D_lower, dX, max);

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
            compute_interaction_p2m_non_rho(m_partner, tmpstore, D_lower);
        }
        template <typename T, typename func>
        CUDA_GLOBAL_METHOD inline void compute_kernel_p2m_rho(T (&X)[NDIM], T (&Y)[NDIM],
            T (&m_partner)[20], T (&tmpstore)[4], T (&tmp_corrections)[3], func&& max) noexcept {
            T dX[NDIM];
            dX[0] = X[0] - Y[0];
            dX[1] = X[1] - Y[1];
            dX[2] = X[2] - Y[2];

            T X_00, X_11, X_22;
            T d2, d3;
            T D_lower[20];
            multipole_interactions::compute_d_factors(d2, d3, X_00, X_11, X_22, D_lower, dX, max);

            tmpstore[0] += m_partner[0] * D_lower[0];
            tmpstore[1] += m_partner[0] * D_lower[1];
            tmpstore[2] += m_partner[0] * D_lower[2];
            tmpstore[3] += m_partner[0] * D_lower[3];
            compute_interaction_p2m_non_rho(m_partner, tmpstore, D_lower);

            compute_interaction_p2m_rho(d2, d3, X_00, X_11, X_22, m_partner, dX, tmp_corrections);
        }
    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
