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
#include <functional>

template<typename T, int R>
using matrix_t = std::array<std::array<T, R>, R>;

template<typename T, int R>
matrix_t<T, R> inverse(const matrix_t<T, R> &A) {
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

template<typename T, int R>
matrix_t<T, R> Jacobian(const std::function<std::array<T, R>(std::array<T, R>)> &f, const std::array<T, R> &x) {
	matrix_t<T, R> J;
	const T h = T(1e-9) * x[0];
	const T half_h = T(0.5) * h;
	const T inv_h = 1.0 / h;
	std::array<T, R> xhi;
	std::array<T, R> xlo;
	T fp;
	T fm;
	T df_dx;
	for (int n = 0; n < R; n++) {
		for (int m = 0; m < R; m++) {
			xhi = xlo = x;
			xhi[m] += half_h;
			xlo[m] -= half_h;
			const auto jhi = f(xhi);
			const auto jlo = f(xlo);
			for (int n = 0; n < R; n++) {
				J[n][m] = (jhi[n] - jlo[n]) * inv_h;
			}
		}
	}
	return J;
}

template<typename T, int R>
matrix_t<T, R> operator*(const std::array<T, R> &A, const std::array<T, R> &B) {
	std::array<T, R> C;
	for (int n = 0; n < R; n++) {
		for (int m = 0; m < R; m++) {
			C[n][m] = T(0);
			for (int p = 0; p < R; p++) {
				C[n][m] += A[n][p] * B[p][m];
			}
		}
	}
	return C;
}

template<typename T, int R>
std::array<T, R> solve(const std::function<std::array<T, R>(std::array<T, R>)> &ftest, std::array<T, R> &guess) {
	std::array<T, R> f;
	matrix_t<T, R> J;
	matrix_t<T, R> inv_J;
	T err;
	int iter = 0;
	std::array<T, R> x = guess;
	static hpx::mutex mutex;
	do {
		f = ftest(x);
		J = Jacobian<T, R>(ftest, x);
		inv_J = inverse<T, R>(J);
		for (int n = 0; n < R; n++) {
			for (int m = 0; m < R; m++) {
				x[m] -= inv_J[m][n] * f[n];
			}
		}
		err = 0.0;
		for (int n = 0; n < R; n++) {
			err += f[n] * f[n];
		}
		err = std::sqrt(err) / x[0];
		iter++;
		if (iter > 1000) {
			printf("Iterations exceeded %e\n", err);
		}
	} while (err > 1.0e-6);
	return x;
}

#endif /* OCTOTIGER_MATRIX_HPP_ */
