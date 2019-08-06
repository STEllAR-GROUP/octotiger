/*
 * impl.hpp
 *
 *  Created on: Aug 3, 2019
 *      Author: dmarce1
 */

#ifndef HYDRO_IMPL_HPP_
#define HYDRO_IMPL_HPP_

#ifndef NOHPX
//#include <hpx/include/async.hpp>
//#include <hpx/include/future.hpp>
//#include <hpx/lcos/when_all.hpp>

//using namespace hpx;
#endif

inline void limit_slope(safe_real &ql, safe_real q0, safe_real &qr) {
	const safe_real tmp1 = qr - ql;
	const safe_real tmp2 = qr + ql;

	if (bool(qr < q0) != bool(q0 < ql)) {
		qr = ql = q0;
		return;
	}
	const safe_real tmp3 = tmp1 * tmp1 / 6.0;
	const safe_real tmp4 = tmp1 * (q0 - 0.5 * tmp2);
	if (tmp4 > tmp3) {
		ql = 3.0 * q0 - 2.0 * qr;
	} else if (-tmp3 > tmp4) {
		qr = 3.0 * q0 - 2.0 * ql;
	}
}

#include "./flux.hpp"
#include "./reconstruct.hpp"

template<int NDIM, int INX, int ORDER>
std::vector<int> hydro_computer<NDIM, INX, ORDER>::find_indices(int lb, int ub) {
	std::vector<int> I;
	for (int i = 0; i < geo::H_N3; i++) {
		int k = i;
		bool interior = true;
		for (int dim = 0; dim < NDIM; dim++) {
			int this_i = k % geo::H_NX;
			if (this_i < lb || this_i >= ub) {
				interior = false;
				break;
			} else {
				k /= geo::H_NX;
			}
		}
		if (interior) {
			I.push_back(i);
		}
	}
	return I;
}

template<int NDIM, int INX, int ORDER>
hydro_computer<NDIM, INX, ORDER>::hydro_computer() {
	nf = 4 + NDIM + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2));
	angmom_count_ = 0;
	D1 = decltype(D1)(geo::H_N3);
	Q = decltype(Q)(nf, std::vector<std::array<safe_real, geo::NDIR>>(geo::H_N3));
	fluxes = decltype(fluxes)(NDIM, std::vector < std::vector<std::array<safe_real, geo::NFACEDIR>> > (nf, std::vector<std::array<safe_real, geo::NFACEDIR>>(geo::H_N3)));
	L = decltype(L)(NDIM, std::vector<std::array<safe_real, geo::NDIR>>(geo::H_N3));

	for (const auto &i : find_indices(0, geo::H_NX)) {
		for (int d = 0; d < geo::NDIR / 2; d++) {
			D1[i][d] = NAN;
		}
	}

}



template<int NDIM, int INX, int ORDER>
inline safe_real hydro_computer<NDIM, INX, ORDER>::minmod(safe_real a, safe_real b) {
	return (std::copysign(0.5, a) + std::copysign(0.5, b)) * std::min(std::abs(a), std::abs(b));
}

template<int NDIM, int INX, int ORDER>
inline safe_real hydro_computer<NDIM, INX, ORDER>::minmod_theta(safe_real a, safe_real b, safe_real c) {
	return minmod(c * minmod(a, b), 0.5 * (a + b));
}

template<int NDIM, int INX, int ORDER>
inline safe_real hydro_computer<NDIM, INX, ORDER>::bound_width() {
	int bw = 1;
	int next_bw = 1;
	for (int dim = 1; dim < NDIM; dim++) {
		next_bw *= geo::H_NX;
		bw += next_bw;
	}
	return bw;
}

template<int NDIM, int INX, int ORDER>
void hydro_computer<NDIM, INX, ORDER>::update_tau(std::vector<std::vector<safe_real>> &U, safe_real dx) {
	constexpr auto dir = geo::directions[NDIM - 1];
	int bw = bound_width();
	for (int i = bw; i < geo::H_N3 - bw; i++) {
		safe_real ek = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			ek += U[sx_i + dim][i] * U[sx_i + dim][i];
		}
		for (int n = 0; n < geo::NANGMOM; n++) {
			ek += pow(U[zx_i + n][i], 2) / (dx * dx);
		}
		ek *= 0.5 * INVERSE(U[rho_i][i]);
		auto egas_max = U[egas_i][i];
		for (int d = 0; d < geo::NDIR; d++) {
			egas_max = std::max(egas_max, U[egas_i][i + dir[d]]);
		}
		safe_real ein = U[egas_i][i] - ek;
		if (ein > 0.1 * egas_max) {
			U[tau_i][i] = POWER(ein, 1.0 / FGAMMA);
		}
	}
}

template<int NDIM, int INX, int ORDER>
void hydro_computer<NDIM, INX, ORDER>::boundaries(std::vector<std::vector<safe_real>> &U) {

	for (int f = 0; f < nf; f++) {
		if (NDIM == 1) {
			for (int i = 0; i < geo::H_BW + 20; i++) {
				U[f][i] = U[f][geo::H_BW];
				U[f][geo::H_NX - 1 - i] = U[f][geo::H_NX - geo::H_BW - 1];
			}
		} else if (NDIM == 2) {

			const auto index = [](int i, int j) {
				return i + geo::H_NX * j;
			};

			for (int i = 0; i < geo::H_BW; i++) {
				for (int j = 0; j < geo::H_NX; j++) {
					int j0 = j;
					j0 = std::max(j0, geo::H_BW);
					j0 = std::min(j0, geo::H_NX - 1 - geo::H_BW);
					U[f][index(i, j)] = U[f][index(geo::H_BW, j0)];
					U[f][index(j, i)] = U[f][index(j0, geo::H_BW)];
					U[f][index(geo::H_NX - 1 - i, j)] = U[f][index(geo::H_NX - 1 - geo::H_BW, j0)];
					U[f][index(j, geo::H_NX - 1 - i)] = U[f][index(j0, geo::H_NX - 1 - geo::H_BW)];
				}
			}
		} else {
			const auto index = [](int i, int j, int k) {
				return i + geo::H_NX * j + k * geo::H_NX * geo::H_NX;
			};

			for (int i = 0; i < geo::H_BW; i++) {
				for (int j = 0; j < geo::H_NX; j++) {
					for (int k = 0; k < geo::H_NX; k++) {
						int j0 = j;
						j0 = std::max(j0, geo::H_BW);
						j0 = std::min(j0, geo::H_NX - 1 - geo::H_BW);
						int k0 = k;
						k0 = std::max(k0, geo::H_BW);
						k0 = std::min(k0, geo::H_NX - 1 - geo::H_BW);
						U[f][index(i, j, k)] = U[f][index(geo::H_BW, j0, k0)];
						U[f][index(j, i, k)] = U[f][index(j0, geo::H_BW, k0)];
						U[f][index(j, k, i)] = U[f][index(j0, k0, geo::H_BW)];
						U[f][index(geo::H_NX - 1 - i, j, k)] = U[f][index(geo::H_NX - 1 - geo::H_BW, j0, k0)];
						U[f][index(j, geo::H_NX - 1 - i, k)] = U[f][index(j0, geo::H_NX - 1 - geo::H_BW, k0)];
						U[f][index(j, k, geo::H_NX - 1 - i)] = U[f][index(j0, k0, geo::H_NX - 1 - geo::H_BW)];
					}
				}
			}
		}
	}
}

template<int NDIM, int INX, int ORDER>
void hydro_computer<NDIM, INX, ORDER>::advance(const std::vector<std::vector<safe_real>> &U0, std::vector<std::vector<safe_real>> &U,
		const std::vector<std::vector<std::vector<safe_real>>> &F, const std::vector<std::array<safe_real, NDIM>> &X, safe_real dx, safe_real dt,
		safe_real beta, safe_real omega) {
	int stride = 1;
	int bw = bound_width();
	std::vector < std::vector < safe_real >> dudt(nf, std::vector < safe_real > (geo::H_N3, 0.0));
	for (int dim = 0; dim < NDIM; dim++) {
		for (int f = 0; f < nf; f++) {
			for (const auto &i : find_indices(geo::H_BW, geo::H_NX - geo::H_BW)) {
				const auto fr = F[dim][f][i + stride];
				const auto fl = F[dim][f][i];
				dudt[f][i] -= (fr - fl) * INVERSE(dx);
			}
		}
		static constexpr auto kdelta = geo::kdeltas[NDIM - 1];
		for (int n = 0; n < geo::NANGMOM; n++) {
			const auto m = dim;
			for (int l = 0; l < NDIM; l++) {
				for (const auto &i : find_indices(geo::H_BW, geo::H_NX - geo::H_BW)) {
					const auto fr = F[dim][sx_i + l][i + stride];
					const auto fl = F[dim][sx_i + l][i];
					dudt[zx_i + n][i] -= kdelta[n][m][l] * 0.5 * (fr + fl);
				}
			}
		}
		stride *= geo::H_NX;
	}
	for (const auto &i : find_indices(geo::H_BW, geo::H_NX - geo::H_BW)) {
		if constexpr (NDIM == 2) {
			dudt[zx_i][i] += omega * (X[i][0] * U[sx_i][i] + X[i][1] * U[sy_i][i]);
		} else if constexpr (NDIM == 3) {
			dudt[zx_i][i] -= omega * X[i][2] * U[sx_i][i];
			dudt[zy_i][i] -= omega * X[i][2] * U[sy_i][i];
			dudt[zz_i][i] += omega * (X[i][0] * U[sx_i][i] + X[i][1] * U[sy_i][i]);
		}

	}
	for (const auto &i : find_indices(geo::H_BW, geo::H_NX - geo::H_BW)) {
		dudt[sx_i][i] += U[sy_i][i] * omega;
		dudt[sy_i][i] -= U[sx_i][i] * omega;
	}
	for (int f = 0; f < nf; f++) {
		for (const auto &i : find_indices(geo::H_BW, geo::H_NX - geo::H_BW)) {
			safe_real u0 = U0[f][i];
			safe_real u1 = U[f][i] + dudt[f][i] * dt;
			U[f][i] = u0 * (1.0 - beta) + u1 * beta;
		}
	}

}

template<int NDIM, int INX, int ORDER>
void hydro_computer<NDIM, INX, ORDER>::output(const std::vector<std::vector<safe_real>> &U, const std::vector<std::array<safe_real, NDIM>> &X, int num) {
	std::string filename = "Y." + std::to_string(num);
	if (NDIM == 1) {
		filename += ".txt";
		FILE *fp = fopen(filename.c_str(), "wt");
		for (int i = 0; i < geo::H_NX; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				fprintf(fp, "%13.6e ", double(X[i][dim]));
			}
			for (int f = 0; f < nf; f++) {
				fprintf(fp, "%13.6e ", double(U[f][i]));
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
	} else {
		filename += ".silo";
		auto db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Uni-tiger", DB_PDB);
		const char *coord_names[] = { "x", "y", "z" };
		safe_real coords[NDIM][geo::H_NX + 1];
		for (int i = 0; i < geo::H_NX + 1; i++) {
			const auto x = safe_real(i - geo::H_BW) / INX - safe_real(0.5);
			for (int dim = 0; dim < NDIM; dim++) {
				coords[dim][i] = x;
			}
		}
		void *coords_[] = { coords, coords + 1, coords + 2 };
		int dims1[] = { geo::H_NX + 1, geo::H_NX + 1, geo::H_NX + 1 };
		int dims2[] = { geo::H_NX, geo::H_NX, geo::H_NX };
		const auto &field_names = NDIM == 2 ? field_names2 : field_names3;
		DBPutQuadmesh(db, "quadmesh", coord_names, coords_, dims1, NDIM, DB_DOUBLE, DB_COLLINEAR, NULL);
		for (int f = 0; f < nf; f++) {
			DBPutQuadvar1(db, field_names[f], "quadmesh", U[f].data(), dims2, NDIM, NULL, 0, DB_DOUBLE, DB_ZONECENT,
			NULL);
		}
		DBClose(db);
	}

}

#endif /* IMPL_HPP_ */
