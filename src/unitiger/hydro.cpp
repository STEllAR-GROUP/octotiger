//============================================================================
// Name        : hydro.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#ifdef NOHPX
#include "../../octotiger/octotiger/unitiger/unitiger.hpp"
#include "../../octotiger/octotiger/unitiger/hydro.hpp"
#else
#include "../../octotiger/unitiger/unitiger.hpp"
#include "../../octotiger/unitiger/hydro.hpp"
#endif

#include <functional>

namespace hydro {

void filter_cell1d(std::array<safe_real, 3> &C, safe_real C0) {
	constexpr int center = 1;
	constexpr int ndir = 3;
	std::array<safe_real, ndir> Q;
	safe_real total;
	total = 0.0;
	for (int i = 0; i < ndir; i++) {
		if (i == center) {
			continue;
		}
		total += vol_weight1d[i];
	}
	C[center] = (C0 - total) * INVERSE(vol_weight1d[center]);
	for (int i = 0; i < ndir; i++) {
		Q[i] = 0.0;
		for (int j = 0; j < ndir; j++) {
			Q[i] += filter1d[i][j] * C[j];
		}
	}
	for (int i = 2; i < 3; i++) {
		attenuate(Q[i], Q[i - 1]);
	}
	for (int i = 0; i < ndir; i++) {
		C[i] = 0.0;
		for (int j = 0; j < ndir; j++) {
			C[i] += inv_filter1d[i][j] * Q[j];
		}
	}
	total = 0.0;
	for (int i = 0; i < ndir; i++) {
		total += vol_weight1d[i];
	}
	const auto dif = C0 - total;
	for (int i = 0; i < ndir; i++) {
		C[i] += dif;
	}
}

void filter_cell2d(std::array<safe_real, 9> &C, safe_real C0) {
	constexpr int center = 4;
	constexpr int ndir = 9;
	std::array<safe_real, ndir> Q;
	safe_real total;
	total = 0.0;
	for (int i = 0; i < ndir; i++) {
		if (i == center) {
			continue;
		}
		total += vol_weight2d[i] * C[i];
	}
	C[center] = (C0 - total) * INVERSE(vol_weight2d[center]);
	for (int i = 0; i < ndir; i++) {
		Q[i] = 0.0;
		for (int j = 0; j < ndir; j++) {
			Q[i] += filter2d[i][j] * C[j];
		}
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (i == 2) {
				attenuate(Q[3 * i + j], Q[3 * (i - 1) + j]);
			}
			if (j == 2) {
				attenuate(Q[3 * i + j], Q[3 * i + j - 1]);
			}
		}
	}
	for (int i = 0; i < ndir; i++) {
		C[i] = 0.0;
		for (int j = 0; j < ndir; j++) {
			C[i] += inv_filter2d[i][j] * Q[j];
		}
	}
	total = 0.0;
	for (int i = 0; i < ndir; i++) {
		total += vol_weight2d[i] * C[i];
	}
	const auto dif = C0 - total;
	for (int i = 0; i < ndir; i++) {
		C[i] += dif;
	}
}

void filter_cell3d(std::array<safe_real, 27> &C, safe_real C0) {
	constexpr int center = 13;
	constexpr int ndir = 27;
	std::array<safe_real, ndir> Q;
	safe_real total;
	total = 0.0;
	for (int i = 0; i < ndir; i++) {
		if (i == center) {
			continue;
		}
		total += vol_weight3d[i];
	}
	C[center] = (C0 - total) * INVERSE(vol_weight3d[center]);
	for (int i = 0; i < ndir; i++) {
		Q[i] = 0.0;
		for (int j = 0; j < ndir; j++) {
			Q[i] += filter3d[i][j] * C[j];
		}
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				if (i == 2) {
					attenuate(Q[9 * k + 3 * i + j], Q[9 * k + 3 * (i - 1) + j]);
				}
				if (j == 2) {
					attenuate(Q[9 * k + 3 * i + j], Q[9 * k + 3 * i + j - 1]);
				}
				if (k == 2) {
					attenuate(Q[9 * (k - 1) + 3 * i + j], Q[9 * k + 3 * i + j]);
				}
			}
		}
	}
	for (int i = 0; i < ndir; i++) {
		C[i] = 0.0;
		for (int j = 0; j < ndir; j++) {
			C[i] += inv_filter3d[i][j] * Q[j];
		}
	}
	total = 0.0;
	for (int i = 0; i < ndir; i++) {
		total += vol_weight3d[i];
	}
	const auto dif = C0 - total;
	for (int i = 0; i < ndir; i++) {
		C[i] += dif;
	}
}

void output_cell2d(FILE *fp, const std::array<safe_real, 9> &C, int joff, int ioff) {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (i < 2)
				fprintf(fp, "%i %i %e %i %i %e \n", i + ioff, j + joff, double(C[3 * i + j]), 1, 0, double(C[3 * (i + 1) + j] - C[3 * i + j]));
			if (j < 2)
				fprintf(fp, "%i %i %e %i %i %e \n", i + ioff, j + joff, double(C[3 * i + j]), 0, 1, double(C[3 * i + j + 1] - C[3 * i + j]));
		}
	}
}


}
