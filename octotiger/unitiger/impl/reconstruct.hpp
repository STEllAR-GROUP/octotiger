#ifdef NOHPX
#include "../basis.hpp"
#else
#include "octotiger/unitiger/basis.hpp"
#endif

template<int NDIM, int INX, int ORDER>
const std::vector<std::vector<std::array<safe_real, hydro_computer<NDIM, INX, ORDER>::NDIR>>> hydro_computer<NDIM, INX, ORDER>::reconstruct(
		std::vector<std::vector<safe_real>> U) {

	static constexpr auto faces = lower_face_members[NDIM - 1];
	static constexpr auto weights = quad_weights[NDIM - 1];
	static constexpr auto face_loc = face_locs[NDIM - 1];
	static constexpr auto kdelta = kdeltas[NDIM - 1];
	static constexpr auto dx = 1.0 / INX;

	static constexpr auto dir = directions[NDIM - 1];

	int bw = bound_width();

	const auto measure_angmom = [](const std::array<std::array<safe_real, NDIR>, NDIM> &C) {
		std::array < safe_real, NANGMOM > L;
		for (int n = 0; n < NANGMOM; n++) {
			L[n] = 0.0;
			for (int m = 0; m < NDIM; m++) {
				for (int l = 0; l < NDIM; l++) {
					for (int d = 0; d < NDIR; d++) {
						L[n] += vol_weights[NDIM - 1][d] * kdelta[n][m][l] * 0.5 * face_loc[d][m] * C[l][d] * dx;
					}
				}
			}
		}
		return L;
	};

	const auto add_angmom = [](std::array<std::array<safe_real, NDIR>, NDIM> &C, std::array<safe_real, NANGMOM> &Z) {
		for (int d = 0; d < NDIR; d++) {
			for (int n = 0; n < NANGMOM; n++) {
				for (int m = 0; m < NDIM; m++) {
					for (int l = 0; l < NDIM; l++) {
						const auto tmp = 6.0 * Z[n] / dx;
						C[l][d] += kdelta[n][m][l] * 0.5 * face_loc[d][m] * tmp;
					}
				}
			}
		}
	};

	const auto reconstruct = [this](std::vector<std::array<safe_real, NDIR>> &q, const std::vector<safe_real> &u, int order) {
		if (order == 1) {
			for (const auto &i : find_indices(1, H_NX - 1)) {
				for (int d = 0; d < NDIR; d++) {
					q[i][d] = u[i];
				}
			}
		} else if (order == 2) {
			for (const auto &i : find_indices(1, H_NX - 1)) {
				for (int d = 0; d < NDIR / 2; d++) {
					const auto di = dir[d];
					const auto slp = minmod(u[i + di] - u[i], u[i] - u[i - di]);
					q[i][d] = u[i] + 0.5 * slp;
					q[i][flip(d)] = u[i] - 0.5 * slp;
				}
			}
		} else if (order == 3) {
			for (const auto &i : find_indices(1, H_NX - 1)) {
				for (int d = 0; d < NDIR / 2; d++) {
					const auto di = dir[d];
					D1[i][d] = minmod_theta(u[i + di] - u[i], u[i] - u[i - di], 2.0);
				}
			}
			for (const auto &i : find_indices(1, H_NX - 1)) {
				for (int d = 0; d < NDIR / 2; d++) {
					const auto di = dir[d];
					q[i][d] = 0.5 * (u[i] + u[i + di]);
					q[i][d] += (1.0 / 6.0) * (D1[i][d] - D1[i + di][d]);
					q[i + di][flip(d)] = q[i][d];
				}
			}
			for (const auto &i : find_indices(2, H_NX - 2)) {
				for (int d = 0; d < NDIR / 2; d++) {
					limit_slope(q[i][d], u[i], q[i][flip(d)]);
				}
				filter_cell(q[i], u[i]);
			}
		}
	};

	for (int f = 0; f < sx_i + NDIM; f++) {
		reconstruct(Q[f], U[f], ORDER);
	}

	std::array<std::vector<safe_real>, NANGMOM> storeZ;
	for (int n = 0; n < NANGMOM; n++) {
		storeZ[n] = U[zx_i + n];
	}

	safe_real z1 = z_error(U);

	if (ORDER == 3) {
		for (int dim = 0; dim < NDIM; dim++) {
			for (const auto &i : find_indices(1, H_NX - 1)) {
				for (int d = 0; d < NDIR / 2; d++) {
					const auto &u = U[sx_i + dim];
					const auto di = dir[d];
					const auto slp = minmod_theta(u[i + di] - u[i], u[i] - u[i - di], 2.0);
					L[dim][i][d] = u[i] + 0.5 * slp;
					L[dim][i][flip(d)] = u[i] - 0.5 * slp;
				}
			}
		}
		for (const auto &i : find_indices(2, H_NX - 2)) {
			std::array < safe_real, NANGMOM > Z;
			for (int dim = 0; dim < NANGMOM; dim++) {
				Z[dim] = U[zx_i + dim][i];
			}
//			if (std::abs(Z[0]) > 1.0e-3) {
//				//			printf("\n");
//			}
			std::array<std::array<safe_real, NDIR>, NDIM> S;
			for (int dim = 0; dim < NDIM; dim++) {
				for (int d = 0; d < NDIR; d++) {
					S[dim][d] = Q[sx_i + dim][i][d];
				}
			}
			auto am1 = measure_angmom(S);
			decltype(Z) am2;
			for (int dim = 0; dim < NANGMOM; dim++) {
				am2[dim] = U[zx_i + dim][i] - am1[dim];
			}
			const auto S0 = S;
			add_angmom(S, am2);
			for (int dim = 0; dim < NDIM; dim++) {
				for (int d = 0; d < NDIR; d++) {
					if (d != NDIR / 2) {
						auto &s = S[dim][d];
						const auto &q = Q[sx_i + dim][i][d];
						const auto &s0 = S0[dim][d];
						const auto &u0 = U[sx_i + dim][i];
						const safe_real l = 0.5 * (L[dim][i][d] + L[dim][i + dir[d]][flip(d)]);
						const safe_real up = U[sx_i + dim][i + dir[d]];
						const auto M = std::max(u0, up);
						const auto m = std::min(u0, up);
						s = std::min(s, M);
						s = std::max(s, m);
					}
				}
			}
			for (int f = sx_i; f < sx_i + NDIM; f++) {
				const auto dim = f - sx_i;
				for (int d = 0; d < NDIR / 2; d++) {
					const auto di = dir[d];
					limit_slope(S[dim][d], U[f][i], S[dim][flip(d)]);
				}
				filter_cell(S[dim], U[f][i]);
			}
			am2 = measure_angmom(S);
			for (int n = 0; n < NANGMOM; n++) {
				U[zx_i + n][i] = Z[n] - am2[n];
			}
			for (int dim = 0; dim < NDIM; dim++) {
				for (int d = 0; d < NDIR; d++) {
					Q[sx_i + dim][i][d] = S[dim][d];
				}
			}
//			if (std::abs(Z[0]) > 1.0e-3) {
//				printf("%e %e %e\n", (double) Z[0], (double) am1[0], (double) am2[0]);
//			}
		}
	}

	if (z1 != 0.0) {
		FILE *fp = fopen("z.txt", "at");
		auto z2 = z_error(U);
		fprintf(fp, "%e %e \n ", (double) z2, (double) z2 / z1);
		fclose(fp);
	}
	for (int f = sx_i + NDIM; f < nf; f++) {
		reconstruct(Q[f], U[f], ORDER);
	}

	for (int n = 0; n < NANGMOM; n++) {
		U[zx_i + n] = storeZ[n];
	}
	return Q;
}


