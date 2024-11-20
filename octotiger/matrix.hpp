//  Copyright (c) 2024 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER_MATRIX_HPP_
#define OCTOTIGER_MATRIX_HPP_


#include <iostream>
#include <array>
#include <stdexcept>
#include <cmath>

template <typename T, int R>
using matrix_t = std::array<std::array<T, R>, R>;

template <typename T, int R>
matrix_t<T, R> inverse(const matrix_t<T, R>& A) {
    matrix_t<T, R> B;
    matrix_t<T, R> C;
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < R; ++j) {
            C[i][j] = A[i][j];
            B[i][j] = (i == j) ? T(1) : T(0);
        }
    }
    for (int i = 0; i < R; ++i) {
        T pivot = C[i][i];
        if (std::abs(pivot) < 1e-30) {
            throw std::runtime_error("Matrix is singular and cannot be inverted.");
        }
        for (int j = 0; j < R; ++j) {
            C[i][j] /= pivot;
            B[i][j] /= pivot;
        }
        for (int k = 0; k < R; ++k) {
            if (k != i) {
                T factor = C[k][i];
                for (int j = 0; j < R; ++j) {
                    C[k][j] -= factor * C[i][j];
                    B[k][j] -= factor * B[i][j];
                }
            }
        }
    }
    return B;
}




#endif /* OCTOTIGER_MATRIX_HPP_ */
