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

template<int NDIM, int INX>
hydro_computer<NDIM, INX>::hydro_computer() {
	nf = 4 + NDIM + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2));
	angmom_count_ = 0;
	D1 = decltype(D1)(geo::H_N3);
	Q = decltype(Q)(nf, std::vector<std::array<safe_real, geo::NDIR>>(geo::H_N3));
	fluxes = decltype(fluxes)(NDIM,
			std::vector < std::vector<std::array<safe_real, geo::NFACEDIR>> > (nf, std::vector<std::array<safe_real, geo::NFACEDIR>>(geo::H_N3)));
	L = decltype(L)(NDIM, std::vector<std::array<safe_real, geo::NDIR>>(geo::H_N3));

	for (const auto &i : find_indices<NDIM, INX>(0, geo::H_NX)) {
		for (int d = 0; d < geo::NDIR / 2; d++) {
			D1[i][d] = NAN;
		}
	}

}

template<int NDIM, int INX>
void hydro_computer<NDIM, INX>::post_process(hydro::state_type &U, safe_real dx) {
	physics < NDIM > p;
	p.template post_process < NDIM > (U, dx);
}

template<int NDIM, int INX>
void hydro_computer<NDIM, INX>::boundaries(hydro::state_type &U) {

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

template<int NDIM, int INX>
void hydro_computer<NDIM, INX>::advance(const hydro::state_type &U0, hydro::state_type &U, const hydro::flux_type &F, const hydro::x_type<NDIM> &X,
		safe_real dx, safe_real dt, safe_real beta, safe_real omega) {
	static thread_local std::vector<std::vector<safe_real>> dudt(nf, std::vector < safe_real > (geo::H_N3));
	for (int f = 0; f < nf; f++) {
		for (const auto &i : find_indices<NDIM, INX>(geo::H_BW, geo::H_NX - geo::H_BW)) {
			dudt[f][i] = 0.0;
		}
	}
	for (int dim = 0; dim < NDIM; dim++) {
		for (int f = 0; f < nf; f++) {
			for (const auto &i : find_indices<NDIM, INX>(geo::H_BW, geo::H_NX - geo::H_BW)) {
				const auto fr = F[dim][f][i + geo::H_DN[dim]];
				const auto fl = F[dim][f][i];
				dudt[f][i] -= (fr - fl) * INVERSE(dx);
			}
		}
	}
	physics < NDIM > ::template source<INX>(dudt, U, F, X, omega, dx);
	for (int f = 0; f < nf; f++) {
		for (const auto &i : find_indices<NDIM, INX>(geo::H_BW, geo::H_NX - geo::H_BW)) {
			safe_real u0 = U0[f][i];
			safe_real u1 = U[f][i] + dudt[f][i] * dt;
			U[f][i] = u0 * (1.0 - beta) + u1 * beta;
		}
	}

}

template<int NDIM, int INX>
void hydro_computer<NDIM, INX>::output(const hydro::state_type &U, const hydro::x_type<NDIM> &X, int num) {
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
