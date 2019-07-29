//============================================================================
// Name        : hydro.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <limits>
#include <silo.h>

#define SNAN std::numeric_limits<double>::signaling_NaN()

using namespace std;

double tmax = 1.0e-1;


#ifdef OCTOTIGER_GRIDDIM
#include "octotiger/hydro_defs.hpp"
#else
constexpr int NDIM = 2;
constexpr int H_BW = 3;
constexpr int H_NX = (2 * H_BW + 100);
constexpr int H_DNX = 1;
constexpr int H_DNY = H_NX;
constexpr int H_DNZ = (H_NX * H_NX);
constexpr int H_N3 = std::pow(H_NX, NDIM);
constexpr auto rho_i = 0;
constexpr auto egas_i = 1;
constexpr auto sx_i = 2;
constexpr auto sy_i = 3;
constexpr auto sz_i = 4;
#endif
constexpr int H_DN0 = 0;
constexpr int NDIR = std::pow(3, NDIM);
constexpr int NFACEDIR = std::pow(3, NDIM - 1);

constexpr char *field_names[] = { "rho", "egas", "sx", "sy", "sz" };

constexpr int NF = (2 + NDIM);

constexpr int directions[3][27] = { {
/**/-H_DNX, +H_DN0, +H_DNX /**/
}, {
/**/-H_DNX - H_DNY, +H_DN0 - H_DNY, +H_DNX - H_DNY,/**/
/**/-H_DNX + H_DN0, +H_DN0 + H_DN0, +H_DNX + H_DN0,/**/
/**/-H_DNX + H_DNY, +H_DN0 + H_DNY, +H_DNX + H_DNY, /**/
}, {
/**/-H_DNX - H_DNY - H_DNZ, +H_DN0 - H_DNY - H_DNZ, +H_DNX - H_DNY - H_DNZ,/**/
/**/-H_DNX + H_DN0 - H_DNZ, +H_DN0 + H_DN0 - H_DNZ, +H_DNX + H_DN0 - H_DNZ,/**/
/**/-H_DNX + H_DNY - H_DNZ, +H_DN0 + H_DNY - H_DNZ, +H_DNX + H_DNY - H_DNZ,/**/
/**/-H_DNX - H_DNY + H_DN0, +H_DN0 - H_DNY + H_DN0, +H_DNX - H_DNY + H_DN0,/**/
/**/-H_DNX + H_DN0 + H_DN0, +H_DN0 + H_DN0 + H_DN0, +H_DNX + H_DN0 + H_DN0,/**/
/**/-H_DNX + H_DNY + H_DN0, +H_DN0 + H_DNY + H_DN0, +H_DNX + H_DNY + H_DN0,/**/
/**/-H_DNX - H_DNY + H_DNZ, +H_DN0 - H_DNY + H_DNZ, +H_DNX - H_DNY + H_DNZ,/**/
/**/-H_DNX + H_DN0 + H_DNZ, +H_DN0 + H_DN0 + H_DNZ, +H_DNX + H_DN0 + H_DNZ,/**/
/**/-H_DNX + H_DNY + H_DNZ, +H_DN0 + H_DNY + H_DNZ, +H_DNX + H_DNY + H_DNZ/**/

} };
constexpr int lower_face_members[3][3][9] = { { { 0 } }, { { 3, 0, 6 }, { 1, 0, 2 } }, { { 12, 0, 3, 6, 9, 15, 18 }, {
		10, 0, 1, 2, 9, 11, 18, 19, 20 }, { 4, 0, 1, 2, 3, 5, 6, 7, 8 } } };

constexpr double quad_weights[3][9] = { { 1.0 }, { 2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0 }, { 16. / 36., 1. / 36., 4. / 36.,
		1. / 36., 4. / 36., 4. / 36., 1. / 36., 4. / 36., 1. / 36. } };

#define FGAMMA (7.0/5.0)

#define flip( d ) (NDIR - 1 - (d))

template<class VECTOR>
void to_prim(VECTOR u, double &p, double &v, int dim) {
	const auto rho = u[rho_i];
	const auto rhoinv = 1.0 / rho;
	double ek = 0.0;
	for (int dim = 0; dim < NDIM; dim++) {
		ek += std::pow(u[sx_i + dim], 2) * rhoinv * 0.5;
	}
	v = u[sx_i + dim] * rhoinv;
	p = (FGAMMA - 1.0) * std::max(u[egas_i] - ek, 0.0);
}

template<class VECTOR>
void flux(const VECTOR &UL, const VECTOR &UR, VECTOR &F, int dim, double &a) {

	double pr, vr, pl, vl;

	to_prim(UR, pr, vr, dim);
	to_prim(UL, pl, vl, dim);
	if (a < 0.0) {
		double ar, al;
		ar = std::abs(vr) + std::sqrt(FGAMMA * pr / UR[rho_i]);
		al = std::abs(vl) + std::sqrt(FGAMMA * pl / UL[rho_i]);
		a = std::max(al, ar);
	}
	for (int f = 0; f < NF; f++) {
		F[f] = 0.5 * ((vr - a) * UR[f] + (vl + a) * UL[f]);
	}
	F[sx_i + dim] += 0.5 * (pr + pl);
	F[egas_i] += 0.5 * (pr * vr + pl * vl);
}

inline static double minmod(double a, double b) {
	return (std::copysign(0.5, a) + std::copysign(0.5, b)) * std::min(std::abs(a), std::abs(b));
}

inline static double bound_width() {
	int H_BW = 1;
	int next_H_BW = 1;
	for (int dim = 1; dim < NDIM; dim++) {
		next_H_BW *= H_NX;
		H_BW += next_H_BW;
	}
	return H_BW;
}

constexpr int ORDER = 3;

double hydro_flux(std::vector<std::vector<double>> &U, std::vector<std::vector<std::vector<double>>> &F) {

	static thread_local std::vector<std::vector<std::array<double, NDIR>>> L(NF,
			std::vector<std::array<double, NDIR>>(H_N3, { SNAN, SNAN, SNAN}));
static thread_local std::vector<std::vector<std::array<double, NDIR / 2>>> D1(NF,
std::vector<std::array<double, NDIR / 2>>(H_N3, {SNAN, SNAN, SNAN}));
static thread_local std::vector<std::vector<std::array<double, NDIR / 2>>> D2(NF,
std::vector<std::array<double, NDIR / 2>>(H_N3, {SNAN, SNAN, SNAN}));
static thread_local std::vector<std::vector<std::array<double, NDIR>>> Q(NF,
std::vector<std::array<double, NDIR>>(H_N3));
static thread_local std::vector<std::vector<std::vector<std::array<double, NFACEDIR>>>> fluxes(NDIM,
std::vector < std::vector<std::array<double, NFACEDIR>>
> (NF, std::vector<std::array<double, NFACEDIR>>(H_N3)));
static thread_local std::array<double, NF> UR, UL, this_flux;

constexpr auto faces = lower_face_members[NDIM - 1];
constexpr auto weights = quad_weights[NDIM - 1];

constexpr auto dir = directions[NDIM - 1];

int H_BW = bound_width();

for (int f = 0; f < NF; f++) {
	for (int i = H_BW; i < H_N3 - H_BW; i++) {
		for (int d = 0; d < NDIR; d++) {
			Q[f][i][d] = U[f][i];
		}
	}
	if (ORDER > 1) {
		for (int i = H_BW; i < H_N3 - H_BW; i++) {
			for (int d = 0; d < NDIR / 2; d++) {
				const auto di = dir[d];
				D1[f][i][d] = minmod(U[f][i + di] - U[f][i], U[f][i] - U[f][i - di]);
			}
		}
		for (int i = H_BW; i < H_N3 - H_BW; i++) {
			for (int d = 0; d < NDIR / 2; d++) {
				Q[f][i][d] += 0.5 * D1[f][i][d];
				Q[f][i][flip(d)] -= 0.5 * D1[f][i][d];
			}
		}
	}
	if (ORDER > 2) {
		for (int i = 2 * H_BW; i < H_N3 - 2 * H_BW; i++) {
			for (int d = 0; d < NDIR / 2; d++) {
				const auto di = dir[d];
				const auto &d1 = D1[f][i][d];
				auto &d2 = D2[f][i][d];
				d2 = minmod(D1[f][i + di][d] - D1[f][i][d], D1[f][i][d] - D1[f][i - di][d]);
				d2 = std::copysign(std::min(std::abs(d2), std::abs(2.0 * d1)), d2);
			}

		}
		for (int i = H_BW; i < H_N3 - H_BW; i++) {
			double d2avg = 0.0;
			double c0 = 1.0;
			if (NDIM > 1) {
				for (int d = 0; d < NDIR / 2; d++) {
					d2avg += D2[f][i][d];
				}
				d2avg /= (NDIR / 2);
				c0 = double(NDIR - 1) / double(NDIR - 3) / 12.0;
			}
			for (int d = 0; d < NDIR / 2; d++) {
				Q[f][i][d] += c0 * (D2[f][i][d] - d2avg);
				Q[f][i][flip(d)] += c0 * (D2[f][i][d] - d2avg);
			}
		}
	}
}

double amax = 0.0;

for (int dim = 0; dim < NDIM; dim++) {
	for (int i = 2 * H_BW; i < H_N3 - 2 * H_BW; i++) {
		double a = -1.0;
		for (int fi = 0; fi < NFACEDIR; fi++) {
			const auto d = faces[dim][fi];
			const auto di = dir[d];
			for (int f = 0; f < NF; f++) {
				UR[f] = Q[f][i][d];
				UL[f] = Q[f][i + di][flip(d)];
			}
			flux(UL, UR, this_flux, dim, a);
			for (int f = 0; f < NF; f++) {
				fluxes[dim][f][i][fi] = this_flux[f];
			}
		}
		amax = std::max(a, amax);
	}
	for (int f = 0; f < NF; f++) {
		for (int i = H_BW; i < H_N3 - H_BW; i++) {
			F[dim][f][i] = 0.0;
			for (int fi = 0; fi < NFACEDIR; fi++) {
				F[dim][f][i] += weights[fi] * fluxes[dim][f][i][fi];
			}
		}
	}
}
return amax;
}

void boundaries(std::vector<std::vector<double>> &U) {
	for (int f = 0; f < NF; f++) {
		if (NDIM == 1) {
			for (int i = 0; i < H_BW + 20; i++) {
				U[f][i] = U[f][H_BW];
				U[f][H_NX - 1 - i] = U[f][H_NX - H_BW - 1];
			}
		} else if (NDIM == 2) {

			const auto index = [](int i, int j) {
				return i + H_NX * j;
			};

			for (int i = 0; i < H_BW; i++) {
				for (int j = 0; j < H_NX; j++) {
					U[f][index(i, j)] = U[f][index(H_BW, j)];
					U[f][index(j, i)] = U[f][index(j, H_BW)];
					U[f][index(H_NX - 1 - i, j)] = U[f][index(H_NX - 1 - H_BW, j)];
					U[f][index(j, H_NX - 1 - i)] = U[f][index(j, H_NX - 1 - H_BW)];
				}
			}
		} else {
			for (int i = 0; i < H_BW; i++) {
				for (int j = 0; j < H_NX; j++) {
					for (int k = 0; k < H_NX; k++) {
						const int ox = i + j * H_NX + k * H_NX * H_NX;
						const int oy = j + i * H_NX + k * H_NX * H_NX;
						const int oz = k + j * H_NX + j * H_NX * H_NX;
						U[f][ox + i] = U[f][ox + H_BW];
						U[f][oy + i] = U[f][oy + H_BW];
						U[f][oz + i] = U[f][oz + H_BW];
						U[f][ox + H_NX - 1 - i] = U[f][ox + H_NX - H_BW - 1];
						U[f][oy + H_NX - 1 - i] = U[f][oy + H_NX - H_BW - 1];
						U[f][oz + H_NX - 1 - i] = U[f][oz + H_NX - H_BW - 1];
					}
				}
			}
		}
	}
}

void advance(const std::vector<std::vector<double>> &U0, std::vector<std::vector<double>> &U,
		const std::vector<std::vector<std::vector<double>>> &F, double dx, double dt, double beta) {
	int stride = 1;
	int H_BW = bound_width();
	for (int dim = 0; dim < NDIM; dim++) {
		for (int f = 0; f < NF; f++) {
			for (int i = 2 * H_BW; i < H_N3 - 2 * H_BW; i++) {
				double u0 = U0[f][i];
				const auto fr = F[dim][f][i + stride];
				const auto fl = F[dim][f][i];
				const auto dudt = -(fr - fl) / dx;
				double u1 = U[f][i] + dudt * dt;
				U[f][i] = u0 * (1.0 - beta) + u1 * beta;
			}
		}
		stride *= H_NX;
	}
}

void output(const std::vector<std::vector<double>> &U, const std::vector<std::array<double, NDIM>> &X, int num) {
	std::string filename = "Y." + std::to_string(num);
	if (NDIM == 1) {
		filename += ".txt";
		FILE *fp = fopen(filename.c_str(), "wt");
		for (int i = 0; i < H_NX; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				fprintf(fp, "%13.6e ", X[i][dim]);
			}
			for (int f = 0; f < NF; f++) {
				fprintf(fp, "%13.6e ", U[f][i]);
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
	} else {
		filename += ".silo";
		auto db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Uni-tiger", DB_PDB);
		const char *coord_names[] = { "x", "y", "z" };
		double coords[NDIM][H_NX + 1];
		for (int i = 0; i < H_NX + 1; i++) {
			const auto x = double(i - H_BW) / H_NX;
			for (int dim = 0; dim < NDIM; dim++) {
				coords[dim][i] = x;
			}
		}
		void *coords_[] = { coords, coords + 1, coords + 2 };
		int dims1[] = { H_NX + 1, H_NX + 1, H_NX + 1 };
		int dims2[] = { H_NX, H_NX, H_NX };
		DBPutQuadmesh(db, "quadmesh", coord_names, coords_, dims1, NDIM, DB_DOUBLE, DB_COLLINEAR, NULL);
		for (int f = 0; f < NF; f++) {
			DBPutQuadvar1(db, field_names[f], "quadmesh", U[f].data(), dims2, NDIM, NULL, 0, DB_DOUBLE, DB_ZONECENT,
			NULL);
		}
		DBClose(db);
	}

}

#include <fenv.h>

int main() {

	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);

	std::vector<std::array<double, NDIM>> X(H_N3);
	std::vector<std::vector<std::vector<double>>> F(NDIM,
			std::vector<std::vector<double>>(NF, std::vector<double>(H_N3, SNAN)));
	std::vector<std::vector<double>> U(NF, std::vector<double>(H_N3, SNAN));
	std::vector<std::vector<double>> U0(NF, std::vector<double>(H_N3, SNAN));

	const double dx = 1.0 / H_NX;

	for (int i = 0; i < H_N3; i++) {
		int k = i;
		int j = 0;
		while (k) {
			X[i][j] = (((k % H_NX) - H_BW) + 0.5) * dx;
			k /= H_NX;
			j++;
		}
	}

	for (int i = 0; i < H_N3; i++) {
		for (int f = 0; f < NF; f++) {
			U[f][i] = 0.0;
		}
		double xsum = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			xsum += X[i][dim];
		}
		if (xsum < 0.5) {
			U[rho_i][i] = 1.0;
			U[egas_i][i] = 2.5;
		} else {
			U[rho_i][i] = 0.125;
			U[egas_i][i] = 0.25;
		}
	}

	double t = 0.0;
	int iter = 0;
	output(U, X, iter++);
	while (t < tmax) {
		U0 = U;
		auto a = hydro_flux(U, F);
		double dt = 0.4 * dx / a / NDIM;
		advance(U0, U, F, dx, dt, 1.0);
		boundaries(U);
		if (ORDER >= 2) {
			boundaries(U);
			hydro_flux(U, F);
			advance(U0, U, F, dx, dt, 0.5);
		}
		t += dt;
		boundaries(U);
		output(U, X, iter++);
		printf("%e %e\n", t, dt);
	}
	output(U, X, iter++);

	return 0;
}

