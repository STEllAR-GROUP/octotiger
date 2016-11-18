/*
 * taylor.cpp
 *
 *  Created on: Jun 9, 2015
 *      Author: dmarce1
 */

#include "taylor.hpp"
#include "simd.hpp"

integer taylor_consts::map2[3][3] = {
    { 0, 1, 2 },
    { 1, 3, 4 },
    { 2, 4, 5 }
};
integer taylor_consts::map3[3][3][3] = {
    { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } },
    { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } },
    { { 2, 4, 5 }, { 4, 7, 8 }, { 5, 8, 9 } }
};
integer taylor_consts::map4[3][3][3][3] = {
    { { { 0,  1,  2 }, { 1,  3,  4 }, { 2,  4,  5 } },
      { { 1,  3,  4 }, { 3,  6,  7 }, { 4,  7,  8 } },
      { { 2,  4,  5 }, { 4,  7,  8 }, { 5,  8,  9 } } },
    { { { 1,  3,  4 }, { 3,  6,  7 }, { 4,  7,  8 } },
      { { 3,  6,  7 }, { 6, 10, 11 }, { 7, 11, 12 } },
      { { 4,  7,  8 }, { 7, 11, 12 }, { 8, 12, 13 } } },
    { { { 2,  4,  5 }, { 4,  7,  8 }, { 5,  8,  9 } },
      { { 4,  7,  8 }, { 7, 11, 12 }, { 8, 12, 13 } },
      { { 5,  8,  9 }, { 8, 12, 13 }, { 9, 13, 14 } } }
};

const real taylor_consts::delta[3][3] = {
    { ONE, ZERO, ZERO },
    { ZERO, ONE, ZERO },
    { ZERO, ZERO, ONE }
};

static /*__attribute__((constructor))*/ void init() {
    // The taylor coefficients for the various dimensions are all stored in an
    // one-dimensional array:
    //     dim  indices
    //      0     0
    //      1     1 ... 3
    //      2     4 ... 9
    //      3    10 ... 19
    //      4    20 ... 34
    for (integer i = 0; i != NDIM; ++i) {
        for (integer j = 0; j != NDIM; ++j) {
            taylor_consts::map2[i][j] += 4;
            for (integer k = 0; k != NDIM; ++k) {
                taylor_consts::map3[i][j][k] += 10;
                for (integer l = 0; l != NDIM; ++l) {
                    taylor_consts::map4[i][j][k][l] += 20;
                }
            }
        }
    }
}

//  integer m = 20;
// 	for (integer i = 0; i != NDIM; ++i) {
// 		for (integer j = i; j != NDIM; ++j) {
// 			for (integer k = j; k != NDIM; ++k) {
// 				for (integer l = k; l != NDIM; ++l) {
// 					taylor_consts::map4[i][j][k][l] = m;
// 					taylor_consts::map4[i][j][l][k] = m;
// 					taylor_consts::map4[i][l][k][j] = m;
// 					taylor_consts::map4[i][k][l][j] = m;
// 					taylor_consts::map4[i][k][j][l] = m;
// 					taylor_consts::map4[i][l][j][k] = m;
// 					taylor_consts::map4[l][j][k][i] = m;
// 					taylor_consts::map4[k][j][l][i] = m;
// 					taylor_consts::map4[j][l][k][i] = m;
// 					taylor_consts::map4[j][k][l][i] = m;
// 					taylor_consts::map4[l][k][j][i] = m;
// 					taylor_consts::map4[k][l][j][i] = m;
// 					taylor_consts::map4[k][j][i][l] = m;
// 					taylor_consts::map4[l][j][i][k] = m;
// 					taylor_consts::map4[k][l][i][j] = m;
// 					taylor_consts::map4[l][k][i][j] = m;
// 					taylor_consts::map4[j][k][i][l] = m;
// 					taylor_consts::map4[j][l][i][k] = m;
// 					taylor_consts::map4[j][i][k][l] = m;
// 					taylor_consts::map4[j][i][l][k] = m;
// 					taylor_consts::map4[l][i][k][j] = m;
// 					taylor_consts::map4[k][i][l][j] = m;
// 					taylor_consts::map4[k][i][j][l] = m;
// 					taylor_consts::map4[l][i][j][k] = m;
// 					++m;
// 				}
// 			}
// 		}
// 	}
// }

struct init_taylor_data
{
    init_taylor_data()
    {
        init();
    }
};
init_taylor_data init_taylor;

template<>
void taylor<5, simd_vector>::set_basis(const std::array<simd_vector, NDIM>& X) {
    constexpr integer N = 5;
    using T = simd_vector;
    //PROF_BEGIN;

    // Assume *this will be in L1 (we only call set_basis on taylor<>s that are
    // local temporaries on the stack).
    taylor<N, T>& A = *this;

    // Assume this will be in L1 (this is local and lives on the stack).
    taylor<N - 1, T> XX;

    ///////////////////////////////////////////////////////////////////////////

    // Assume tmps stay in registers (NOTE: CURRENTLY SPILLING)

    // XX[1] = X[0];
    //     XX[4] = X[0] * X[0];
    //         XX[10] = X[0] * X[0] * X[0];
    //         XX[11] = X[0] * X[0] * X[1];
    //         XX[12] = X[0] * X[0] * X[2];
    //     XX[5] = X[0] * X[1];
    //         XX[13] = X[0] * X[1] * X[1];
    //         XX[14] = X[0] * X[1] * X[2];
    //     XX[6] = X[0] * X[2];
    //         XX[15] = X[0] * X[2] * X[2];
    //
    // XX[2] = X[1];
    //     XX[7] = X[1] * X[1];
    //         XX[16] = X[1] * X[1] * X[1];
    //         XX[17] = X[1] * X[1] * X[2];
    //     XX[8] = X[1] * X[2];
    //         XX[18] = X[1] * X[2] * X[2];
    //
    // XX[3] = X[2];
    //     XX[9] = X[2] * X[2];
    //         XX[19] = X[2] * X[2] * X[2];

    // Remember, XX is a local temporaries on the stack.  Assume its in the L1
    // and all stores are between L1 and registers.
    //
    // LOADS  X[0:2] * SIMD_LEN * SIZEOF(REAL)   = 192  BYTES
    // STORES XX[0:19] * SIMD_LEN * SIZEOF(REAL) = 1280 BYTES
    //                                           = 1472 BYTES
    //
    // 3 ITERATIONS * 0 FLOP * SIMD_LEN
    //  6 ITERATIONS * 1 FLOP * SIMD_LEN  =  48 FLOP
    //   10 ITERATIONS * 1 FLOP * SIMD_LEN = 80 FLOP
    //                                    = 128 FLOP
    for (integer a = 0; a != NDIM; a++) {
        auto const tmpa = X[a];                   // 1 LOAD (NOTE: CURRENTLY SPILLING)
        XX(a) = tmpa;                             // 1 STORE

        for (integer b = a; b != NDIM; b++) {
            auto const tmpab = tmpa * X[b];       // 1 LOAD, 1 MUL
            XX(a, b) = tmpab;                     // 1 STORE

            for (integer c = b; c != NDIM; c++) {
                XX(a, b, c) = tmpab * X[c];       // 1 LOAD, 1 STORE, 1 MUL
            }
        }
    }

    // 3 LOADS, 3 MULS, 2 ADDS, 1 DIV (AKA 1 RCP + 1 MUL)
    const T r2inv = ONE / (X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);

    // 1 SQRT
    const T d0 = -sqrt(r2inv);

    // 1 MUL
    const T d1 = -d0 * r2inv;

    // 2 MULS
    const T d2 = -T(3) * d1 * r2inv;

    // 2 MULS
    const T d3 = -T(5) * d2 * r2inv;

//     // 2 MULS
//     const T d4 = -T(7) * d3 * r2inv;

    // 1 STORE
    A() = d0;

    ///////////////////////////////////////////////////////////////////////////

    // Remember, XX and A (aka *this) are local temporaries on the stack.
    // Assume they're in the L1 and all loads and stores are between L1 and
    // registers.

    // A[1] = XX[1] * d1;
    //     A[4] = XX[4] * d2;
    //         A[10] = XX[10] * d3;
    //             A[20] = 0.0;
    //             A[21] = 0.0;
    //             A[22] = 0.0;
    //         A[11] = XX[11] * d3;
    //             A[23] = 0.0;
    //             A[24] = 0.0;
    //         A[12] = XX[12] * d3;
    //             A[25] = 0.0;
    //     A[5] = XX[5] * d2;
    //         A[13] = XX[13] * d3;
    //             A[26] = 0.0;
    //             A[27] = 0.0;
    //         A[14] = XX[14] * d3;
    //             A[28] = 0.0;
    //     A[6] = XX[6] * d2;
    //         A[15] = XX[15] * d3;
    //             A[29] = 0.0;
    //
    // A[2] = XX[2] * d1;
    //     A[7] = XX[7] * d2;
    //         A[16] = XX[16] * d3;
    //             A[30] = 0.0;
    //             A[31] = 0.0;
    //         A[17] = XX[17] * d3;
    //             A[32] = 0.0;
    //     A[8] = XX[8] * d2;
    //         A[18] = XX[18] * d3;
    //             A[33] = 0.0;
    //
    // A[3] = XX[3] * d1;
    //     A[9] = XX[9] * d2;
    //         A[19] = XX[19] * d3;
    //             A[34] = 0.0;

    // LOADS  XX[0:19] * SIMD_LEN * SIZEOF(REAL) = 1280 BYTES
    // STORES A[0:34] * SIMD_LEN * SIZEOF(REAL)  = 2240 BYTES
    //                                           = 3520 BYTES
    //
    // 3  ITERATIONS * 1 FLOP * SIMD_LEN     =  24 FLOP
    //  6  ITERATIONS * 1 FLOP * SIMD_LEN    =  48 FLOP
    //   10  ITERATIONS * 1 FLOP * SIMD_LEN  =  80 FLOP
    //    15 ITERATIONS * 0 FLOP * SIMD_LEN  =
    //                                       = 152 FLOP
//     for (integer a = 0; a != NDIM; a++) {
//         A(a) = XX(a) * d1;                            // 2 LOADS, 1 STORE, 1 MUL
//         for (integer b = a; b != NDIM; b++) {
//             A(a, b) = XX(a, b) * d2;                  // 2 LOADS, 1 STORE, 1 MUL
//             for (integer c = b; c != NDIM; c++) {
//                 A(a, b, c) = XX(a, b, c) * d3;        // 2 LOADS, 1 STORE, 1 MUL
//                 for (integer d = c; d != NDIM; ++d) {
//                     A(a, b, c, d) = 0.0;              // 1 STORE
//                 }
//             }
//         }
//     }
    for (integer i = taylor_sizes[0]; i != taylor_sizes[1]; ++i) {
        A[i] = XX[i] * d1;
    }
    for (integer i = taylor_sizes[1]; i != taylor_sizes[2]; ++i) {
        A[i] = XX[i] * d2;
    }
    for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
        A[i] = XX[i] * d3;
    }
    for (integer i = taylor_sizes[3]; i != taylor_sizes[4]; ++i) {
        A[i] = ZERO;
    }

    ///////////////////////////////////////////////////////////////////////////

    // ...
    //     A[4] = A[4] + d1;
    //     A[10] = A[10] + XX[1] * d2;
    //     A[20] = A[20] + XX[4] * d3 + 2.0 * d2;
    //         A[10] = A[10] + XX[1] * d2;
    //         A[10] = A[10] + XX[1] * d2;
    //         A[20] = A[20] + XX[4] * d3
    //         A[20] = A[20] + XX[4] * d3
    //         A[20] = A[20] + d2;
    //             A[20] = A[20] + XX[4] * d3;
    //             A[20] = A[20] + XX[4] * d3;
    //             A[20] = A[20] + XX[4] * d3;
    //             A[21] = A[21] + XX[5] * d3;
    //             A[21] = A[21] + XX[5] * d3;
    //             A[23] = A[23] + XX[4] * d3;
    //             A[22] = A[22] + XX[5] * d3;
    //             A[22] = A[22] + XX[5] * d3;
    //             A[25] = A[25] + XX[4] * d3;
    //         A[11] = A[11] + XX[2] * d2;
    //         A[13] = A[13] + XX[1] * d2;
    //         A[21] = A[21] + XX[5] * d3;
    //         A[26] = A[26] + XX[5] * d3;
    //         A[23] = A[23] + d2;
    //             A[22] = A[22] + XX[7] * d3;
    //             A[26] = A[26] + XX[5] * d3;
    //             A[26] = A[26] + XX[4] * d3;
    //             A[24] = A[24] + XX[8] * d3;
    //             A[27] = A[27] + XX[6] * d3;
    //             A[28] = A[28] + XX[4] * d3;
    //         A[12] = A[12] + XX[3] * d2;
    //         A[15] = A[13] + XX[1] * d2;
    //         A[21] = A[21] + XX[6] * d3;
    //         A[29] = A[29] + XX[6] * d3;
    //         A[25] = A[25] + d2;
    //             A[25] = A[25] + XX[9] * d3;
    //             A[29] = A[29] + XX[6] * d3;
    //             A[29] = A[29] + XX[4] * d3;
    //
    // ...
    //     A[7] = A[7] + d1;
    //     A[16] = A[16] + XX[2] * d2;
    //     A[30] = A[30] + XX[7] * d3 + 2.0 * d2;
    //         A[16] = A[16] + XX[2] * d2;
    //         A[16] = A[16] + XX[2] * d2;
    //         A[30] = A[30] + XX[7] * d3
    //         A[30] = A[30] + XX[7] * d3
    //         A[30] = A[30] + d2;
    //             A[30] = A[30] + XX[7] * d3;
    //             A[30] = A[30] + XX[7] * d3;
    //             A[30] = A[30] + XX[7] * d3;
    //             A[31] = A[31] + XX[8] * d3;
    //             A[31] = A[31] + XX[8] * d3;
    //             A[32] = A[32] + XX[7] * d3;
    //         A[17] = A[16] + XX[3] * d2;
    //         A[18] = A[16] + XX[2] * d2;
    //         A[31] = A[31] + XX[8] * d3
    //         A[33] = A[33] + XX[8] * d3
    //         A[32] = A[32] + d2;
    //             A[32] = A[32] + XX[9] * d3;
    //             A[33] = A[33] + XX[9] * d3;
    //             A[33] = A[33] + XX[8] * d3;
    //
    // ...
    //     A[9] = A[9] + d1;
    //     A[19] = A[19] + XX[3] * d2;
    //     A[34] = A[34] + XX[9] * d3 + 2.0 * d2;
    //         A[19] = A[19] + XX[3] * d2;
    //         A[19] = A[19] + XX[3] * d2;
    //         A[34] = A[34] + XX[9] * d3
    //         A[34] = A[34] + XX[9] * d3
    //         A[34] = A[34] + d2;
    //             A[34] = A[34] + XX[9] * d3;
    //             A[34] = A[34] + XX[9] * d3;
    //             A[34] = A[34] + XX[9] * d3;

    // LOADS  XX[0:19] * SIMD_LEN * SIZEOF(REAL) = 1280 BYTES
    // STORES A[0:34] * SIMD_LEN * SIZEOF(REAL)  = 2240 BYTES
    //                                           = 3520 BYTES
    //
    // FIXME: counts FLOP
    // 3  ITERATIONS * 0 FLOP * SIMD_LEN     =  0 FLOP
    //  6  ITERATIONS * . FLOP * SIMD_LEN    =    FLOP
    //   10 ITERATIONS * . FLOP * SIMD_LEN   =    FLOP
    //    15 ITERATIONS * . FLOP * SIMD_LEN =
    //                                       =    FLOP
    for (integer a = 0; a != NDIM; a++) {
        auto const XXa = XX(a) * d2;
        A(a, a) += d1;
        A(a, a, a) += XXa;
        A(a, a, a, a) += XX(a, a) * d3;
        A(a, a, a, a) += 2.0 * d2;
        for (integer b = a; b != NDIM; b++) {
            auto const XXab = XX(a, b) * d3;
            A(a, a, b) += XX(b) * d2;
            A(a, b, b) += XXa;
            A(a, a, a, b) += XXab;
            A(a, b, b, b) += XXab;
            A(a, a, b, b) += d2;
            for (integer c = b; c != NDIM; c++) {
                A(a, a, b, c) += XX(b, c) * d3;
                A(a, b, b, c) += XX(a, c) * d3;
                A(a, b, c, c) += XXab;
            }
        }
    }

    //PROF_END;
}
