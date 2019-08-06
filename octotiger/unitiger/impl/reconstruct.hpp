#ifdef NOHPX
#include "../basis.hpp"
#else
#include "octotiger/unitiger/basis.hpp"
#endif

#include "../util.hpp"

template<int NDIM, int INX>
const hydro::recon_type<NDIM> hydro_computer<NDIM, INX>::reconstruct(hydro::state_type &U, safe_real dx) {

	static constexpr auto face_loc = geo::face_locs[NDIM - 1];
	static constexpr auto kdelta = geo::kdeltas[NDIM - 1];

	static constexpr auto dir = geo::directions[NDIM - 1];

	static const auto indices1 = find_interior_indices<1>();
	static const auto indices2 = find_interior_indices<2>();

	const auto measure_angmom = [dx](const std::array<std::array<safe_real, geo::NDIR>, NDIM> &C) {
		std::array < safe_real, geo::NANGMOM > L;
		for (int n = 0; n < geo::NANGMOM; n++) {
			L[n] = 0.0;
			for (int m = 0; m < NDIM; m++) {
				for (int l = 0; l < NDIM; l++) {
					for (int d = 0; d < geo::NDIR; d++) {
						L[n] += geo::vol_weights[NDIM - 1][d] * kdelta[n][m][l] * 0.5 * face_loc[d][m] * C[l][d] * dx;
					}
				}
			}
		}
		return L;
	};

	const auto add_angmom = [dx](std::array<std::array<safe_real, geo::NDIR>, NDIM> &C, std::array<safe_real, geo::NANGMOM> &Z) {
		for (int d = 0; d < geo::NDIR; d++) {
			for (int n = 0; n < geo::NANGMOM; n++) {
				for (int m = 0; m < NDIM; m++) {
					for (int l = 0; l < NDIM; l++) {
						const auto tmp = 6.0 * Z[n] / dx;
						C[l][d] += kdelta[n][m][l] * 0.5 * face_loc[d][m] * tmp;
					}
				}
			}
		}
	};

	const auto reconstruct = [this](std::vector<std::array<safe_real, geo::NDIR>> &q, const std::vector<safe_real> &u) {
		for (const auto &i : indices1) {
			for (int d = 0; d < geo::NDIR / 2; d++) {
				const auto di = dir[d];
				D1[i][d] = minmod_theta(u[i + di] - u[i], u[i] - u[i - di], 2.0);
			}
		}
		for (const auto &i : indices1) {
			for (int d = 0; d < geo::NDIR / 2; d++) {
				const auto di = dir[d];
				q[i][d] = 0.5 * (u[i] + u[i + di]);
				q[i][d] += (1.0 / 6.0) * (D1[i][d] - D1[i + di][d]);
				q[i + di][geo::flip(d)] = q[i][d];
			}
		}
		for (const auto i : indices2) {
			for (int d = 0; d < geo::NDIR / 2; d++) {
				limit_slope(q[i][d], u[i], q[i][geo::flip(d)]);
			}
			filter_cell(q[i], u[i]);
		}
	};

	if (angmom_count_ == 0) {
		for (int f = 0; f < nf; f++) {
			reconstruct(Q[f], U[f]);
		}

	} else {
		for (int f = 0; f < angmom_index_; f++) {
			reconstruct(Q[f], U[f]);
		}

		int sx_i = angmom_index_;
		int zx_i = sx_i + NDIM;

		for (int angmom_pair = 0; angmom_pair < angmom_count_; angmom_pair++) {
			for (int f = sx_i; f < sx_i + NDIM; f++) {
				reconstruct(Q[f], U[f]);
			}

			std::array<std::vector<safe_real>, geo::NANGMOM> storeZ;
			for (int n = 0; n < geo::NANGMOM; n++) {
				storeZ[n] = U[zx_i + n];
			}

//			safe_real z1 = z_error(U);

			for (int dim = 0; dim < NDIM; dim++) {
				for (const auto &i : indices1) {
					for (int d = 0; d < geo::NDIR / 2; d++) {
						const auto &u = U[sx_i + dim];
						const auto di = dir[d];
						const auto slp = minmod_theta(u[i + di] - u[i], u[i] - u[i - di], 2.0);
						L[dim][i][d] = u[i] + 0.5 * slp;
						L[dim][i][geo::flip(d)] = u[i] - 0.5 * slp;
					}
				}
			}
			for (const auto &i : indices2) {
				std::array < safe_real, geo::NANGMOM > Z;
				for (int dim = 0; dim < geo::NANGMOM; dim++) {
					Z[dim] = U[zx_i + dim][i];
				}
//			if (std::abs(Z[0]) > 1.0e-3) {
//				//			printf("\n");
//			}
				std::array<std::array<safe_real, geo::NDIR>, NDIM> S;
				for (int dim = 0; dim < NDIM; dim++) {
					for (int d = 0; d < geo::NDIR; d++) {
						S[dim][d] = Q[sx_i + dim][i][d];
					}
				}
				auto am1 = measure_angmom(S);
				decltype(Z) am2;
				for (int dim = 0; dim < geo::NANGMOM; dim++) {
					am2[dim] = U[zx_i + dim][i] - am1[dim];
				}
				const auto S0 = S;
				add_angmom(S, am2);
				for (int dim = 0; dim < NDIM; dim++) {
					for (int d = 0; d < geo::NDIR; d++) {
						if (d != geo::NDIR / 2) {
							auto &s = S[dim][d];
							const auto &q = Q[sx_i + dim][i][d];
							const auto &s0 = S0[dim][d];
							const auto &u0 = U[sx_i + dim][i];
							const safe_real l = 0.5 * (L[dim][i][d] + L[dim][i + dir[d]][geo::flip(d)]);
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
					for (int d = 0; d < geo::NDIR / 2; d++) {
						const auto di = dir[d];
						limit_slope(S[dim][d], U[f][i], S[dim][geo::flip(d)]);
					}
					filter_cell(S[dim], U[f][i]);
				}
				am2 = measure_angmom(S);
				for (int n = 0; n < geo::NANGMOM; n++) {
					U[zx_i + n][i] = Z[n] - am2[n];
				}
				for (int dim = 0; dim < NDIM; dim++) {
					for (int d = 0; d < geo::NDIR; d++) {
						Q[sx_i + dim][i][d] = S[dim][d];
					}
				}
//			if (std::abs(Z[0]) > 1.0e-3) {
//				printf("%e %e %e\n", (double) Z[0], (double) am1[0], (double) am2[0]);
//			}
			}

//			if (z1 != 0.0) {
//				FILE *fp = fopen("z.txt", "at");
//				auto z2 = z_error(U);
//				fprintf(fp, "%e %e \n ", (double) z2, (double) z2 / z1);
//				fclose(fp);
//			}
			for (int f = zx_i; f < zx_i + geo::NANGMOM; f++) {
				reconstruct(Q[f], U[f]);
			}
			for (int n = 0; n < geo::NANGMOM; n++) {
				U[zx_i + n] = storeZ[n];
			}
			sx_i += geo::NANGMOM + NDIM;
			zx_i += geo::NANGMOM + NDIM;
		}
		for (int f = angmom_index_ + (geo::NFACEDIR + NDIM) * angmom_count_; f < nf; f++) {
			reconstruct(Q[f], U[f]);
		}

	}
	return Q;
}

