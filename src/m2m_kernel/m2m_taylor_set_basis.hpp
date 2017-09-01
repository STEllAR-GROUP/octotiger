#pragma once

#include "m2m_simd_types.hpp"

namespace octotiger {
namespace fmm {

    // overload for kernel-specific simd type
    static inline void set_basis(
        std::array<m2m_vector, 35>& A, const std::array<m2m_vector, NDIM>& X) {
        // A is D in the paper in formula (6)
        // taylor<N, T>& A = *this;

        const m2m_vector r2 = X[0] * X[0] + X[1] * X[1] + X[2] * X[2];
        // m2m_vector::mask_type mask = r2 > 0.0;
        // m2m_vector r2inv = 0.0;
        // // initialized to 1 to fix div by zero after masking
        // m2m_vector tmp_max = 1.0;
        // Vc::where(mask, tmp_max) = Vc::max(r2, m2m_vector(1.0e-20));
        // m2m_vector tmp_one = 1.0;
        // Vc::where(mask, r2inv) = ONE / tmp_max;
        m2m_vector tmp_max = Vc::max(r2, m2m_vector(1.0e-20));
        m2m_vector r2inv = ONE / tmp_max;

        // parts of formula (6)
        const m2m_vector d0 = -sqrt(r2inv);
        // parts of formula (7)
        const m2m_vector d1 = -d0 * r2inv;
        // parts of formula (8)
        const m2m_vector d2 = m2m_vector(-3) * d1 * r2inv;
        // parts of  formula (9)
        const m2m_vector d3 = m2m_vector(-5) * d2 * r2inv;
        //     const m2m_vector d4 = -m2m_vector(7) * d3 * r2inv;

        // formula (6)
        A[0] = d0;

        A[1] = X[0] * d1;
        A[2] = X[1] * d1;
        A[3] = X[2] * d1;

        m2m_vector X_00 = X[0] * X[0];
        m2m_vector X_11 = X[1] * X[1];
        m2m_vector X_22 = X[2] * X[2];

        m2m_vector X_12 = X[1] * X[2];
        m2m_vector X_01 = X[0] * X[1];
        m2m_vector X_02 = X[0] * X[2];

        A[4] = d2 * X_00;
        A[4] += d1;
        A[5] = d2 * X_01;
        A[6] = d2 * X_02;

        A[7] = d2 * X_11;
        A[7] += d1;
        A[8] = d2 * X_12;

        A[9] = d2 * X_22;
        A[9] += d1;

        A[10] = d3 * X_00 * X[0];
        m2m_vector d2_X0 = d2 * X[0];
        // A[10] += d2 * X[0];
        // A[10] += d2 * X[0];
        // A[10] += d2 * X[0];
        A[10] += 3.0 * d2_X0;
        A[11] = d3 * X_00 * X[1];
        A[11] += d2 * X[1];
        A[12] = d3 * X_00 * X[2];
        A[12] += d2 * X[2];

        A[13] = d3 * X[0] * X_11;
        A[13] += d2 * X[0];
        A[14] = d3 * X[0] * X_12;

        A[15] = d3 * X[0] * X_22;
        // A[15] += d2 * X[0];
        A[15] += d2_X0;

        A[16] = d3 * X_11 * X[1];
        m2m_vector d2_X1 = d2 * X[1];
        // A[16] += d2 * X[1];
        // A[16] += d2 * X[1];
        // A[16] += d2 * X[1];
        A[16] += 3.0 * d2_X1;

        A[17] = d3 * X_11 * X[2];
        A[17] += d2 * X[2];

        A[18] = d3 * X[1] * X_22;
        A[18] += d2 * X[1];

        A[19] = d3 * X_22 * X[2];
        m2m_vector d2_X2 = d2 * X[2];
        // A[19] += d2 * X[2];
        // A[19] += d2 * X[2];
        // A[19] += d2 * X[2];
        A[19] += 3.0 * d2_X2;

        A[20] = X[0] * X[0] * d3 + 2.0 * d2;
        m2m_vector d3_X00 = d3 * X_00;
        // A[20] += d3 * X_00;
        // A[20] += d3 * X_00;
        A[20] += d2;
        // A[20] += d3 * X_00;
        // A[20] += d3 * X_00;
        // A[20] += d3 * X_00;
        A[20] += 5.0 * d3_X00;
        m2m_vector d3_X01 = d3 * X_01;
        // A[21] = d3 * X_01;
        // A[21] += d3 * X_01;
        // A[21] += d3 * X_01;
        A[21] = 3.0 * d3_X01;
        m2m_vector d3_X02 = d3 * X_02;
        A[22] = 3.0 * d3_X02;
        A[23] = d2;
        m2m_vector d3_X11 = d3 * X_11;
        A[23] += d3_X11;
        A[23] += d3 * X_00;
        m2m_vector d3_X12 = d3 * X_12;
        A[24] = d3_X12;
        A[25] = d2;
        m2m_vector d3_X22 = d3 * X_22;
        A[25] += d3_X22;
        A[25] += d3_X00;
        A[26] = 3.0 * d3_X01;
        A[27] = d3 * X_02;
        A[28] = d3 * X_01;
        A[29] = 3.0 * d3_X02;
        A[30] = X[1] * X[1] * d3 + 2.0 * d2;
        A[30] += d2;
        A[30] += 5.0 * d3_X11;

        A[31] = 3.0 * d3_X12;
        A[32] = d2;
        A[32] += d3_X22;
        A[32] += d3_X11;
        A[33] = 3.0 * d3_X12;
        A[34] = X[2] * X[2] * d3 + 2.0 * d2;
        A[34] += d2;
        A[34] += 5.0 * d3_X22;
    }

    class D_split
    {
    public:
        const std::array<m2m_vector, NDIM>& X;

        m2m_vector X_00;
        m2m_vector X_11;
        m2m_vector X_22;

        m2m_vector r2;
        m2m_vector r2inv;

        m2m_vector d0;
        m2m_vector d1;
        m2m_vector d2;
        m2m_vector d3;

    public:
        D_split(const std::array<m2m_vector, NDIM>& X)
          : X(X) {
            X_00 = X[0] * X[0];
            X_11 = X[1] * X[1];
            X_22 = X[2] * X[2];

            r2 = X_00 + X_11 + X_22;
            // m2m_vector::mask_type mask = r2 > 0.0;
            // m2m_vector r2inv = 0.0;
            // // initialized to 1 to fix div by zero after masking
            // m2m_vector tmp_max = 1.0;
            // Vc::where(mask, tmp_max) = Vc::max(r2, m2m_vector(1.0e-20));
            // m2m_vector tmp_one = 1.0;
            // Vc::where(mask, r2inv) = ONE / tmp_max;
            r2inv = ONE / Vc::max(r2, m2m_vector(1.0e-20));

            // parts of formula (6)
            d0 = -sqrt(r2inv);
            // parts of formula (7)
            d1 = -d0 * r2inv;
            // parts of formula (8)
            d2 = m2m_vector(-3) * d1 * r2inv;
            // parts of  formula (9)
            d3 = m2m_vector(-5) * d2 * r2inv;
        }

        // overload for kernel-specific simd type
        inline void calculate_D_lower(std::array<m2m_vector, 20>& A) {
            // formula (6)
            A[0] = d0;

            A[1] = X[0] * d1;
            A[2] = X[1] * d1;
            A[3] = X[2] * d1;

            m2m_vector X_12 = X[1] * X[2];
            m2m_vector X_01 = X[0] * X[1];
            m2m_vector X_02 = X[0] * X[2];

            A[4] = d2 * X_00;
            A[4] += d1;
            A[5] = d2 * X_01;
            A[6] = d2 * X_02;

            A[7] = d2 * X_11;
            A[7] += d1;
            A[8] = d2 * X_12;

            A[9] = d2 * X_22;
            A[9] += d1;

            A[10] = d3 * X_00 * X[0];
            m2m_vector d2_X0 = d2 * X[0];
            A[10] += 3.0 * d2_X0;
            A[11] = d3 * X_00 * X[1];
            A[11] += d2 * X[1];
            A[12] = d3 * X_00 * X[2];
            A[12] += d2 * X[2];

            A[13] = d3 * X[0] * X_11;
            A[13] += d2 * X[0];
            A[14] = d3 * X[0] * X_12;

            A[15] = d3 * X[0] * X_22;
            A[15] += d2_X0;

            A[16] = d3 * X_11 * X[1];
            m2m_vector d2_X1 = d2 * X[1];
            A[16] += 3.0 * d2_X1;

            A[17] = d3 * X_11 * X[2];
            A[17] += d2 * X[2];

            A[18] = d3 * X[1] * X_22;
            A[18] += d2 * X[1];

            A[19] = d3 * X_22 * X[2];
            m2m_vector d2_X2 = d2 * X[2];
            A[19] += 3.0 * d2_X2;
        }

        // // overload for kernel-specific simd type
        // inline void calculate_D_upper(std::array<m2m_vector, 15>& A) {
        //     A[0] = X[0] * X[0] * d3 + 2.0 * d2;
        //     m2m_vector d3_X00 = d3 * X_00;
        //     A[0] += d2;
        //     A[0] += 5.0 * d3_X00;
        //     m2m_vector d3_X01 = d3 * X[0] * X[1];
        //     A[1] = 3.0 * d3_X01;
        //     m2m_vector X_02 = X[0] * X[2];
        //     m2m_vector d3_X02 = d3 * X_02;
        //     A[2] = 3.0 * d3_X02;
        //     A[3] = d2;
        //     m2m_vector d3_X11 = d3 * X_11;
        //     A[3] += d3_X11;
        //     A[3] += d3 * X_00;
        //     m2m_vector d3_X12 = d3 * X[1] * X[2];
        //     A[4] = d3_X12;
        //     A[5] = d2;
        //     m2m_vector d3_X22 = d3 * X_22;
        //     A[5] += d3_X22;
        //     A[5] += d3_X00;
        //     A[6] = 3.0 * d3_X01;
        //     A[7] = d3 * X_02;
        //     A[8] = d3 * X[0] * X[1];
        //     A[9] = 3.0 * d3_X02;
        //     A[10] = X[1] * X[1] * d3 + 2.0 * d2;
        //     A[10] += d2;
        //     A[10] += 5.0 * d3_X11;

        //     A[11] = 3.0 * d3_X12;
        //     A[12] = d2;
        //     A[12] += d3_X22;
        //     A[12] += d3_X11;
        //     A[13] = 3.0 * d3_X12;
        //     A[14] = X[2] * X[2] * d3 + 2.0 * d2;
        //     A[14] += d2;
        //     A[14] += 5.0 * d3_X22;
        // }
    };

}    // namespace fmm
}    // namespace octotiger
