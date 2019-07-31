/*
 * hydro.hpp
 *
 *  Created on: Jul 31, 2019
 *      Author: dmarce1
 */

#ifndef OCTOTIGER_UNITIGER_HYDRO_HPP_
#define OCTOTIGER_UNITIGER_HYDRO_HPP_
#include <vector>

template<int NDIM, int INX, int ORDER>
struct hydro_computer {
	double hydro_flux(std::vector<std::vector<double>> U, std::vector<std::vector<std::vector<double>>> &F, std::vector<std::array<double, NDIM>> &X,
			double omega);

	void update_tau(std::vector<std::vector<double>> &U);

	template<class VECTOR>
	void to_prim(VECTOR u, double &p, double &v, int dim);
	template<class VECTOR>
	void flux(const VECTOR &UL, const VECTOR &UR, VECTOR &F, int dim, double &a, std::array<double, NDIM> &vg);
	inline static double minmod(double a, double b);
	inline static double minmod_theta(double a, double b, double c);
	inline static double bound_width();

	void boundaries(std::vector<std::vector<double>> &U);
	void advance(const std::vector<std::vector<double>> &U0, std::vector<std::vector<double>> &U, const std::vector<std::vector<std::vector<double>>> &F,
			double dx, double dt, double beta, double omega);
	void output(const std::vector<std::vector<double>> &U, const std::vector<std::array<double, NDIM>> &X, int num);

	static constexpr int rho_i = 0;
	static constexpr int egas_i = 1;
	static constexpr int tau_i = 2;
	static constexpr int pot_i = 3;
	static constexpr int sx_i = 4;
	static constexpr int sy_i = 5;
	static constexpr int sz_i = 6;
	static constexpr int zx_i = 4 + NDIM;
	static constexpr int zy_i = 5 + NDIM;
	static constexpr int zz_i = 6 + NDIM;
	static constexpr int spc_i = 4 + NDIM + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2));

	int nf;

private:

	int ns;
	static constexpr int H_BW = 3;
	static constexpr int H_NX = (2 * H_BW + INX);
	static constexpr int H_DNX = 1;
	static constexpr int H_DN[3] = { 1, H_NX, H_NX * H_NX };
	static constexpr int H_DNY = H_NX;
	static constexpr int H_DNZ = (H_NX * H_NX);
	static constexpr int H_N3 = std::pow(H_NX, NDIM);
	static constexpr int H_DN0 = 0;
	static constexpr int NDIR = std::pow(3, NDIM);
	static constexpr int nfACEDIR = std::pow(3, NDIM - 1);
	static constexpr int lower_face_members[3][3][9] = { { { 0 } }, { { 3, 6, 0 }, { 1, 0, 2 } }, { { 12, 0, 3, 6, 9, 15, 18 },
			{ 10, 0, 1, 2, 9, 11, 18, 19, 20 }, { 4, 0, 1, 2, 3, 5, 6, 7, 8 } } };

	static constexpr double quad_weights[3][9] = { { 1.0 }, { 2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0 }, { 16. / 36., 1. / 36., 4. / 36., 1. / 36., 4. / 36., 4. / 36.,
			1. / 36., 4. / 36., 1. / 36. } };

	static constexpr int directions[3][27] = { {
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
	std::vector<std::vector<std::array<double, NDIR / 2>>> D1;
	std::vector<std::vector<std::array<double, NDIR / 2>>> D2;
	std::vector<std::vector<std::array<double, NDIR>>> Q;
	std::vector<std::vector<std::vector<std::array<double, nfACEDIR>>>> fluxes;

public:

	hydro_computer(int nspecies) {
		nf = 4 + NDIM + nspecies + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2));
		ns = nspecies;

		D1 = decltype(D1)(nf, std::vector<std::array<double, NDIR / 2>>(H_N3));
		D2 = decltype(D2)(nf, std::vector<std::array<double, NDIR / 2>>(H_N3));
		Q = decltype(Q)(nf, std::vector<std::array<double, NDIR>>(H_N3));
		fluxes = decltype(fluxes)(NDIM,
				std::vector<std::vector<std::array<double, nfACEDIR>> >(nf, std::vector<std::array<double, nfACEDIR>>(H_N3)));

	}

};

#ifndef NOHPX
#include <hpx/include/async.hpp>
#endif
//#include <hpx/include/future.hpp>

#define flip( d ) (NDIR - 1 - (d))

template<int NDIM, int INX, int ORDER>
double hydro_computer<NDIM, INX, ORDER>::hydro_flux(std::vector<std::vector<double>> U, std::vector<std::vector<std::vector<double>>> &F,
		std::vector<std::array<double, NDIM>> &X, double omega) {

	constexpr auto faces = lower_face_members[NDIM - 1];
	constexpr auto weights = quad_weights[NDIM - 1];

	constexpr auto dir = directions[NDIM - 1];

	int bw = bound_width();

//	std::vector<hpx::future<void>> frecon(nf);
//	std::vector<hpx::future<double>> futs2;

//	for (int f = 0; f < nf; f++) {
//		frecon[f] = hpx::make_ready_future<void>();
//	}
//
//	frecon[pot_i] = frecon[pot_i].then([&U](hpx::future<void> &&fut) {
//		fut.get();
	for (int i = 0; i < H_N3; i++) {
		U[pot_i][i] /= U[rho_i][i];
	}
//	});

	for (int f = 0; f < nf; f++) {
//		if (f == rho_i) {
//			continue;
//		}
//		frecon[f] = frecon[f].then([f, bw, &U, &Q, &D1, &D2](hpx::future<void> &&fut) {
//			fut.get();
		for (int i = bw; i < H_N3 - bw; i++) {
			for (int d = 0; d < NDIR; d++) {
				const auto &Uf = U[f];
				auto &Qfi = Q[f][i];
				Qfi[d] = Uf[i];
			}
		}
		if (ORDER > 1) {
			for (int i = bw; i < H_N3 - bw; i++) {
				for (int d = 0; d < NDIR / 2; d++) {
					const auto di = dir[d];
					D1[f][i][d] = minmod_theta(U[f][i + di] - U[f][i], U[f][i] - U[f][i - di], 1.0);
				}
			}
			for (int i = bw; i < H_N3 - bw; i++) {
				for (int d = 0; d < NDIR / 2; d++) {
					Q[f][i][d] += 0.5 * D1[f][i][d];
					Q[f][i][flip(d)] -= 0.5 * D1[f][i][d];
				}
			}
		}
		if (ORDER > 2) {
			for (int i = 2 * bw; i < H_N3 - 2 * bw; i++) {
				for (int d = 0; d < NDIR / 2; d++) {
					const auto di = dir[d];
					const auto &d1 = D1[f][i][d];
					auto &d2 = D2[f][i][d];
					d2 = minmod_theta(D1[f][i + di][d] - D1[f][i][d], D1[f][i][d] - D1[f][i - di][d], 2.0);
					d2 = std::copysign(std::min(std::abs(d2), std::abs(2.0 * d1)), d2);
				}

			}
			for (int i = bw; i < H_N3 - bw; i++) {
				double d2avg = 0.0;
				double c0 = 1.0 / 12.0;
				for (int d = 0; d < NDIR / 2; d++) {
					Q[f][i][d] += c0 * (D2[f][i][d]);
					Q[f][i][flip(d)] += c0 * (D2[f][i][d]);
				}
			}
		}
//		});

	}

//	frecon[rho_i] = frecon[rho_i].then([&](hpx::future<void> &&fut) {
//		fut.get();
//		for (int s = 0; s < ns; s++) {
//			frecon[spc_i + s].wait();
//		}
//		for (int i = 1; i < H_N3 - 1; i++) {
//			for (int d = 0; d < NDIR / 2; d++) {
//				Q[rho_i][i][d] = 0.0;
//				for (int s = 0; s < ns; s++) {
//					Q[rho_i][i][d] += Q[spc_i + s][i][d];
//				}
//			}
//		}
////	});
//
//	frecon[rho_i] = frecon[rho_i].then([&](hpx::future<void> &&fut) {
//		fut.get();
//		frecon[rho_i].wait();
	for (int i = bw; i < H_N3 - bw; i++) {
		for (int d = 0; d < NDIR / 2; d++) {
			auto &q1 = Q[pot_i][i][d];
			auto &q2 = Q[pot_i][i + dir[d]][flip(d)];
			auto avg = 0.5 * (q1 + q2);
			q1 = q2 = avg;
			q1 *= Q[rho_i][i][d];
			q2 *= Q[rho_i][i + dir[d]][flip(d)];
		}
	}
//	});
//
//	for (auto &fut : frecon) {
//		fut.get();
//	}

	double amax = 0.0;
	for (int dim = 0; dim < NDIM; dim++) {
//		futs2.push_back(hpx::async(hpx::launch::async, [&](int dim) {
		std::vector<double> UR(nf), UL(nf), this_flux(nf);
//			double amax = 0.0;
		for (int i = 2 * bw; i < H_N3 - 2 * bw; i++) {
			double a = -1.0;
			for (int fi = 0; fi < nfACEDIR; fi++) {
				const auto d = faces[dim][fi];
				const auto di = dir[d];
				for (int f = 0; f < nf; f++) {
					UR[f] = Q[f][i][d];
					UL[f] = Q[f][i + di][flip(d)];
				}
				std::array<double, NDIM> vg;
				if constexpr (NDIM > 1) {
					vg[0] = -0.5 * omega * (X[i][1] + X[i - H_DN[dim]][1]);
					vg[1] = +0.5 * omega * (X[i][0] + X[i - H_DN[dim]][0]);
					if constexpr (NDIM == 3) {
						vg[2] = 0.0;
					}
				}
				flux(UL, UR, this_flux, dim, a, vg);
				for (int f = 0; f < nf; f++) {
					fluxes[dim][f][i][fi] = this_flux[f];
				}
			}
			amax = std::max(a, amax);
		}
		for (int f = 0; f < nf; f++) {
			for (int i = bw; i < H_N3 - bw; i++) {
				F[dim][f][i] = 0.0;
				for (int fi = 0; fi < nfACEDIR; fi++) {
					F[dim][f][i] += weights[fi] * fluxes[dim][f][i][fi];
				}
			}
		}
//			return amax;
//		},dim));
	}
//	for (auto &fut : futs2) {
//		amax = std::max(amax, fut.get());
//	}

//	for (int f = 0; f < nf; f++) {
//		double sum = 0.0;
//		for (int i = 3 * bw; i < H_N3 - 2 * bw; i++) {
//			for (int d = 0; d < NDIR / 2; d++) {
//				sum += std::abs(Q[f][i][d] - Q[f][i + dir[d]][flip(d)]);
//			}
//		}
//		printf("%e ", sum);
//	}
//	printf("\n");
	//	abort();
	return amax;
}

template<int NDIM, int INX, int ORDER>
template<class VECTOR>
void hydro_computer<NDIM, INX, ORDER>::to_prim(VECTOR u, double &p, double &v, int dim) {
	const auto rho = u[rho_i];
	const auto rhoinv = 1.0 / rho;
	double ek = 0.0;
	for (int dim = 0; dim < NDIM; dim++) {
		ek += std::pow(u[sx_i + dim], 2) * rhoinv * 0.5;
	}
	auto ein = u[egas_i] - ek;
	if (ein < 0.001 * u[egas_i]) {
		ein = std::pow(u[tau_i], FGAMMA);
	}
	v = u[sx_i + dim] * rhoinv;
	p = (FGAMMA - 1.0) * ein;
}

template<int NDIM, int INX, int ORDER>
template<class VECTOR>
void hydro_computer<NDIM, INX, ORDER>::flux(const VECTOR &UL, const VECTOR &UR, VECTOR &F, int dim, double &a, std::array<double, NDIM> &vg) {

	double pr, vr, pl, vl, vr0, vl0;

	to_prim(UR, pr, vr0, dim);
	to_prim(UL, pl, vl0, dim);
	vr = vr0 - vg[dim];
	vl = vl0 - vg[dim];
	if (a < 0.0) {
		double ar, al;
		ar = std::abs(vr) + std::sqrt(FGAMMA * pr / UR[rho_i]);
		al = std::abs(vl) + std::sqrt(FGAMMA * pl / UL[rho_i]);
		a = std::max(al, ar);
	}
	for (int f = 0; f < nf; f++) {
		F[f] = 0.5 * ((vr - a) * UR[f] + (vl + a) * UL[f]);
	}
	F[sx_i + dim] += 0.5 * (pr + pl);
	F[egas_i] += 0.5 * (pr * vr0 + pl * vl0);
}

template<int NDIM, int INX, int ORDER>
inline double hydro_computer<NDIM, INX, ORDER>::minmod(double a, double b) {
	return (std::copysign(0.5, a) + std::copysign(0.5, b)) * std::min(std::abs(a), std::abs(b));
}

template<int NDIM, int INX, int ORDER>
inline double hydro_computer<NDIM, INX, ORDER>::minmod_theta(double a, double b, double c) {
	return minmod(c * minmod(a, b), 0.5 * (a + b));
}

template<int NDIM, int INX, int ORDER>
inline double hydro_computer<NDIM, INX, ORDER>::bound_width() {
	int bw = 1;
	int next_bw = 1;
	for (int dim = 1; dim < NDIM; dim++) {
		next_bw *= H_NX;
		bw += next_bw;
	}
	return bw;
}

template<int NDIM, int INX, int ORDER>
void hydro_computer<NDIM, INX, ORDER>::update_tau(std::vector<std::vector<double>> &U) {
	constexpr auto dir = directions[NDIM - 1];
	int bw = bound_width();
	for (int i = bw; i < H_N3 - bw; i++) {
		double ek = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			ek += U[sx_i + dim][i] * U[sx_i + dim][i];
		}
		ek *= 0.5 / U[rho_i][i];
		auto egas_max = U[egas_i][i];
		for (int d = 0; d < NDIR; d++) {
			egas_max = std::max(egas_max, U[egas_i][i + dir[d]]);
		}
		double ein = U[egas_i][i] - ek;
		if (ein > 0.1 * egas_max) {
			U[tau_i][i] = std::pow(ein, 1.0 / FGAMMA);
		}
	}
}

template<int NDIM, int INX, int ORDER>
void hydro_computer<NDIM, INX, ORDER>::boundaries(std::vector<std::vector<double>> &U) {
	for (int f = 0; f < nf; f++) {
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
			const auto index = [](int i, int j, int k) {
				return i + H_NX * j + k * H_NX * H_NX;
			};

			for (int i = 0; i < H_BW; i++) {
				for (int j = 0; j < H_NX; j++) {
					for (int k = 0; k < H_NX; k++) {
						U[f][index(i, j, k)] = U[f][index(H_BW, j, k)];
						U[f][index(j, i, k)] = U[f][index(j, H_BW, k)];
						U[f][index(j, k, i)] = U[f][index(j, k, H_BW)];
						U[f][index(H_NX - 1 - i, j, k)] = U[f][index(H_NX - 1 - H_BW, j, k)];
						U[f][index(j, H_NX - 1 - i, k)] = U[f][index(j, H_NX - 1 - H_BW, k)];
						U[f][index(j, H_NX - 1 - k, i)] = U[f][index(j, k, H_NX - 1 - H_BW)];
					}
				}
			}
		}
	}
}

template<int NDIM, int INX, int ORDER>
void hydro_computer<NDIM, INX, ORDER>::advance(const std::vector<std::vector<double>> &U0, std::vector<std::vector<double>> &U,
		const std::vector<std::vector<std::vector<double>>> &F, double dx, double dt, double beta, double omega) {
	int stride = 1;
	int bw = bound_width();
	std::vector<std::vector<double>> dudt(nf, std::vector<double>(H_N3, 0.0));
	for (int dim = 0; dim < NDIM; dim++) {
		for (int f = 0; f < nf; f++) {
			for (int i = 2 * bw; i < H_N3 - 2 * bw; i++) {
				const auto fr = F[dim][f][i + stride];
				const auto fl = F[dim][f][i];
				dudt[f][i] -= (fr - fl) / dx;
			}
		}
		stride *= H_NX;
	}
	for (int i = 0; i < H_N3; i++) {
		dudt[sx_i][i] += U[sy_i][i] * omega;
		dudt[sy_i][i] -= U[sx_i][i] * omega;
	}
	for (int f = 0; f < nf; f++) {
		for (int i = 2 * bw; i < H_N3 - 2 * bw; i++) {
			double u0 = U0[f][i];
			double u1 = U[f][i] + dudt[f][i] * dt;
			U[f][i] = u0 * (1.0 - beta) + u1 * beta;
		}
	}

}

template<int NDIM, int INX, int ORDER>
void hydro_computer<NDIM, INX, ORDER>::output(const std::vector<std::vector<double>> &U, const std::vector<std::array<double, NDIM>> &X, int num) {
	std::string filename = "Y." + std::to_string(num);
	if (NDIM == 1) {
		filename += ".txt";
		FILE *fp = fopen(filename.c_str(), "wt");
		for (int i = 0; i < H_NX; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				fprintf(fp, "%13.6e ", X[i][dim]);
			}
			for (int f = 0; f < nf; f++) {
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
		const auto &field_names = NDIM == 2 ? field_names2 : field_names3;
		DBPutQuadmesh(db, "quadmesh", coord_names, coords_, dims1, NDIM, DB_DOUBLE, DB_COLLINEAR, NULL);
		for (int f = 0; f < nf; f++) {
			DBPutQuadvar1(db, field_names[f], "quadmesh", U[f].data(), dims2, NDIM, NULL, 0, DB_DOUBLE, DB_ZONECENT,
			NULL);
		}
		DBClose(db);
	}

}

#endif /* OCTOTIGER_UNITIGER_HYDRO_HPP_ */
