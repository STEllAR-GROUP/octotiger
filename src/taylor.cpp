/*
 * taylor.cpp
 *
 *  Created on: Jun 9, 2015
 *      Author: dmarce1
 */

#include "taylor.hpp"
#include "simd.hpp"

integer taylor_consts::map4[3][3][3][3];
integer taylor_consts::map2[3][3] = { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } };
integer taylor_consts::map3[3][3][3] = { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 2, 4, 5 }, { 4, 7, 8 }, { 5,
	8, 9 } } };

const real taylor_consts::delta[3][3] = { { ONE, ZERO, ZERO }, { ZERO, ONE, ZERO }, { ZERO, ZERO, ONE } };

static /*__attribute__((constructor))*/ void init() {
	integer m = 20;
	for (integer i = 0; i != NDIM; ++i) {
		for (integer j = 0; j != NDIM; ++j) {
			taylor_consts::map2[i][j] += 4;
			for (integer k = 0; k != NDIM; ++k) {
				taylor_consts::map3[i][j][k] += 10;
			}
		}
	}
	for (integer i = 0; i != NDIM; ++i) {
		for (integer j = i; j != NDIM; ++j) {
			for (integer k = j; k != NDIM; ++k) {
				for (integer l = k; l != NDIM; ++l) {
					taylor_consts::map4[i][j][k][l] = m;
					taylor_consts::map4[i][j][l][k] = m;
					taylor_consts::map4[i][l][k][j] = m;
					taylor_consts::map4[i][k][l][j] = m;
					taylor_consts::map4[i][k][j][l] = m;
					taylor_consts::map4[i][l][j][k] = m;
					taylor_consts::map4[l][j][k][i] = m;
					taylor_consts::map4[k][j][l][i] = m;
					taylor_consts::map4[j][l][k][i] = m;
					taylor_consts::map4[j][k][l][i] = m;
					taylor_consts::map4[l][k][j][i] = m;
					taylor_consts::map4[k][l][j][i] = m;
					taylor_consts::map4[k][j][i][l] = m;
					taylor_consts::map4[l][j][i][k] = m;
					taylor_consts::map4[k][l][i][j] = m;
					taylor_consts::map4[l][k][i][j] = m;
					taylor_consts::map4[j][k][i][l] = m;
					taylor_consts::map4[j][l][i][k] = m;
					taylor_consts::map4[j][i][k][l] = m;
					taylor_consts::map4[j][i][l][k] = m;
					taylor_consts::map4[l][i][k][j] = m;
					taylor_consts::map4[k][i][l][j] = m;
					taylor_consts::map4[k][i][j][l] = m;
					taylor_consts::map4[l][i][j][k] = m;
					++m;
				}
			}
		}
	}
}

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
	taylor<N, T>& A = *this;
	taylor<N - 1, T> XX;
	const T r2inv = ONE / (X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);

	for (integer a = 0; a != NDIM; a++) {
		XX(a) = X[a];
		for (integer b = a; b != NDIM; b++) {
            auto const tmp = X[a] * X[b];
			XX(a, b) = tmp;
			for (integer c = b; c != NDIM; c++) {
				XX(a, b, c) = tmp * X[c];
			}
		}
	}

	const T d0 = -sqrt(r2inv);
	const T d1 = -d0 * r2inv;
	const T d2 = -T(3) * d1 * r2inv;
	const T d3 = -T(5) * d2 * r2inv;
	const T d4 = -T(7) * d3 * r2inv;

	A() = d0;
	for (integer a = 0; a != NDIM; a++) {
		A(a) = XX(a) * d1;
		for (integer b = a; b != NDIM; b++) {
			A(a, b) = XX(a, b) * d2;
			for (integer c = b; c != NDIM; c++) {
				A(a, b, c) = XX(a, b, c) * d3;
				for (integer d = c; d != NDIM && c != NDIM; ++d) {
					A(a, b, c, d) = 0.0;
				}
			}
		}
	}

	for (integer a = 0; a != NDIM; a++) {
		A(a, a) += d1;
        auto const XXa = XX(a) * d2;
		A(a, a, a) += XXa;
		A(a, a, a, a) += XX(a, a) * d3;
		A(a, a, a, a) += 2.0 * d2;
		for (integer b = a; b != NDIM; b++) {
			A(a, a, b) += XX(b) * d2;
			A(a, b, b) += XXa;
            auto const XXab = XX(a, b) * d3;
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
