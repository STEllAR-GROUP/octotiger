/*
 * hydro.hpp
 *
 *  Created on: Jul 31, 2019
 *      Author: dmarce1
 */

#ifndef OCTOTIGER_UNITIGER_HYDRO_HPP_
#define OCTOTIGER_UNITIGER_HYDRO_HPP_
#include <vector>

void output_cell2d(FILE *fp, const std::array<double, 9> &C, int ioff, int joff);

void filter_cell1d(std::array<double, 3> &C, double C0);
void filter_cell2d(std::array<double, 9> &C, double C0);
void filter_cell3d(std::array<double, 27> &C, double C0);

template<int NDIM>
struct filter_cell_helper;

template<>
struct filter_cell_helper<1> {
	static constexpr auto func = filter_cell1d;
};

template<>
struct filter_cell_helper<2> {
	static constexpr auto func = filter_cell2d;
};

template<>
struct filter_cell_helper<3> {
	static constexpr auto func = filter_cell1d;
};

template<int NDIM, class VECTOR>
void filter_cell(VECTOR &C, double c0) {
	return (*filter_cell_helper<NDIM>::func)(C, c0);
}

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
	static constexpr int NANGMOM = NDIM == 1 ? 0 : std::pow(3, NDIM - 2);
	static constexpr int kdeltas[3][3][3][3] = { { { { } } }, { { { 0, 1 }, { -1, 0 } } }, { { { 0, 0, 0 }, { 0, 0, 1 }, { 0, -1, 0 } }, { { 0, 0, -1 }, { 0, 0,
			0 }, { 1, 0, 0 } }, { { 0, 1, 0 }, { -1, 0, 0 }, { 0, 0, 0 } } } };
	static constexpr int nfACEDIR = std::pow(3, NDIM - 1);
	static constexpr int lower_face_members[3][3][9] = { { { 0 } }, { { 3, 6, 0 }, { 1, 0, 2 } }, { { 12, 0, 3, 6, 9, 15, 18 },
			{ 10, 0, 1, 2, 9, 11, 18, 19, 20 }, { 4, 0, 1, 2, 3, 5, 6, 7, 8 } } };

	static constexpr double quad_weights[3][9] = { { 1.0 }, { 2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0 }, { 16. / 36., 1. / 36., 4. / 36., 1. / 36., 4. / 36., 4. / 36.,
			1. / 36., 4. / 36., 1. / 36. } };

	static constexpr int face_locs[3][27][3] = {
	/**/{ { -1 }, { 0 }, { 1 } },
	/**/{ { -1, -1 }, { -1, 0 }, { -1, 1 },
	/**/{ +0, -1 }, { +0, 0 }, { +0, 1 },
	/**/{ +1, -1 }, { +1, 0 }, { +1, 1 } },
	/**/{ { -1, -1, -1 }, { -1, -1, +0 }, { -1, -1, +1 },
	/**/{ -1, +0, -1 }, { -1, +0, +0 }, { -1, +0, +1 },
	/**/{ -1, +1, -1 }, { -1, +1, +0 }, { -1, +1, +1 },
	/**/{ +0, -1, -1 }, { +0, -1, +0 }, { +0, -1, +1 },
	/**/{ +0, +0, -1 }, { +0, +0, +0 }, { +0, +0, +1 },
	/**/{ +0, +1, -1 }, { +0, +1, +0 }, { +0, +1, +1 },
	/**/{ +1, -1, -1 }, { +1, -1, +0 }, { +1, -1, +1 },
	/**/{ +1, +0, -1 }, { +1, +0, +0 }, { +1, +0, +1 },
	/**/{ +1, +1, -1 }, { +1, +1, +0 }, { +1, +1, +1 } } };

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
	std::vector<std::vector<std::array<double, NDIR>>> Q;
	std::vector<std::vector<std::array<double, NDIR>>> L;
	std::vector<std::vector<std::vector<std::array<double, nfACEDIR>>>> fluxes;

public:

	hydro_computer(int nspecies) {
		nf = 4 + NDIM + nspecies + (NDIM == 1 ? 0 : std::pow(3, NDIM - 2));
		ns = nspecies;

		D1 = decltype(D1)(nf, std::vector<std::array<double, NDIR / 2>>(H_N3));
		Q = decltype(Q)(nf, std::vector<std::array<double, NDIR>>(H_N3));
		L = decltype(Q)(nf, std::vector<std::array<double, NDIR>>(H_N3));
		fluxes = decltype(fluxes)(NDIM, std::vector<std::vector<std::array<double, nfACEDIR>> >(nf, std::vector<std::array<double, nfACEDIR>>(H_N3)));

	}

};

#ifndef NOHPX
#include <hpx/include/async.hpp>
#include <hpx/include/future.hpp>
#include <hpx/lcos/when_all.hpp>

using namespace hpx;
#endif

#define flip( d ) (NDIR - 1 - (d))

inline void limit_slope(double &ql, double q0, double &qr) {
	const double tmp1 = qr - ql;
	const double tmp2 = qr + ql;

	if (bool(qr < q0) != bool(q0 < ql)) {
		qr = ql = q0;
		return;
	}
	const double tmp3 = tmp1 * tmp1 / 6.0;
	const double tmp4 = tmp1 * (q0 - 0.5 * tmp2);
	if (tmp4 > tmp3) {
		ql = 3.0 * q0 - 2.0 * qr;
	} else if (-tmp3 > tmp4) {
		qr = 3.0 * q0 - 2.0 * ql;
	}
}

template<int NDIM, int INX, int ORDER>
double hydro_computer<NDIM, INX, ORDER>::hydro_flux(std::vector<std::vector<double>> U, std::vector<std::vector<std::vector<double>>> &F,
		std::vector<std::array<double, NDIM>> &X, double omega) {

	const auto find_indices = [=](int lb, int ub) {
		std::vector<int> I;
		for (int i = 0; i < H_N3; i++) {
			int k = i;
			bool interior = true;
			for (int dim = 0; dim < NDIM; dim++) {
				int this_i = k % H_NX;
				if (this_i < lb || this_i >= ub) {
					interior = false;
				} else {
					k /= H_NX;
				}
			}
			if (interior) {
				I.push_back(i);
			}
		}
		return I;
	};

	static constexpr auto faces = lower_face_members[NDIM - 1];
	static constexpr auto weights = quad_weights[NDIM - 1];
	static constexpr auto face_loc = face_locs[NDIM - 1];
	static constexpr auto kdelta = kdeltas[NDIM - 1];
	static constexpr auto dx = 1.0 / INX;

	static constexpr auto dir = directions[NDIM - 1];

	int bw = bound_width();

	static const auto measure_ang_mom = [](std::array<std::array<double, NDIR>,NDIM> C) {
		std::array<double, NANGMOM> L;
		for (int n = 0; n < NANGMOM; n++) {
			for (int m = 0; m < NDIM; m++) {
				for (int l = 0; l < NDIM; l++) {
					for (int d = 0; d < NDIR; d++) {
						L[n] += weights[d] * kdelta[n][m][l] * face_loc[d][m] * C[l][d];
					}
				}
			}
		}
		return L;
	};

	std::vector<future<void>> frecon;
	frecon.reserve(nf);
	for (int f = 0; f < nf; f++) {
		frecon.push_back(hpx::async(hpx::launch::async, [&](int f) {
			if constexpr (ORDER == 1) {
				for (int i = bw; i < H_N3 - bw; i++) {
					for (int d = 0; d < NDIR; d++) {
						Q[f][i][d] = U[f][i];
					}
				}
			} else if constexpr (ORDER == 2) {
				for (int i = bw; i < H_N3 - bw; i++) {
					for (int d = 0; d < NDIR / 2; d++) {
						const auto di = dir[d];
						const auto slp = minmod(U[f][i + di] - U[f][i], U[f][i] - U[f][i - di]);
						Q[f][i][d] = U[f][i] + 0.5 * slp;
						Q[f][i][flip(d)] = U[f][i] - 0.5 * slp;
					}
				}
			} else if constexpr (ORDER == 3) {
				static const auto indices1 = find_indices(1, H_NX - 1);
				for (const auto &i : indices1) {
					for (int d = 0; d < NDIR / 2; d++) {
						const auto di = dir[d];
						D1[f][i][d] = minmod_theta(U[f][i + di] - U[f][i], U[f][i] - U[f][i - di], 2.0);
						const auto slp = minmod(U[f][i + di] - U[f][i], U[f][i] - U[f][i - di]);
						L[f][i][d] = U[f][i] + 0.5 * slp;
						L[f][i][flip(d)] = U[f][i] - 0.5 * slp;
					}
				}
				static const auto indices2 = find_indices(2, H_NX - 1);
				for (const auto &i : indices2) {
					for (int d = 0; d < NDIR / 2; d++) {
						const auto di = dir[d];
						Q[f][i][d] = 0.5 * (U[f][i] + U[f][i + di]);
						Q[f][i][d] += (1.0 / 6.0) * (D1[f][i][d] - D1[f][i + di][d]);
						Q[f][i + di][flip(d)] = Q[f][i][d];

					}
				}
				static const auto indices3 = find_indices(2, H_NX - 2);
				for (const auto &i : indices3) {
					for (int d = 0; d < NDIR / 2; d++) {
						const auto di = dir[d];
						limit_slope(Q[f][i][d], U[f][i], Q[f][i][flip(d)]);
					}
					filter_cell<NDIM>(Q[f][i], U[f][i]);
				}
			}
		}, f));

	}

	hpx::when_all(frecon.begin(), frecon.end()).get();

	std::vector<future<void>> fflux;
	fflux.reserve(NDIM);

	std::array<double, 3> amax = { 0.0, 0.0, 0.0 };
	for (int dim = 0; dim < NDIM; dim++) {
		fflux.push_back(hpx::async(hpx::launch::async, [&](int dim) {
			std::vector<double> UR(nf), UL(nf), this_flux(nf);
			static const auto indices = find_indices(3, H_NX - 2);
			for (const auto &i : indices) {
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
				amax[dim] = std::max(a, amax[dim]);
			}
			for (int f = 0; f < nf; f++) {
				for (int i = bw; i < H_N3 - bw; i++) {
					F[dim][f][i] = 0.0;
					for (int fi = 0; fi < nfACEDIR; fi++) {
						F[dim][f][i] += weights[fi] * fluxes[dim][f][i][fi];
					}
				}
			}
			for (int n = 0; n < NANGMOM; n++) {
				for (int m = 0; m < NDIM; m++) {
					for (int l = 0; l < NDIM; l++) {
						for (int fi = 0; fi < nfACEDIR; fi++) {
							const auto d = faces[dim][fi];
							for (int i = bw; i < H_N3 - bw; i++) {
								F[dim][zx_i + n][i] += kdelta[n][m][l] * face_loc[d][m] * 0.5 * dx * fluxes[dim][sx_i + l][i][fi];
							}
						}
					}
				}
			}
		}, dim));
	}
	hpx::when_all(fflux.begin(), fflux.end()).get();
	for (int d = 1; d < NDIM; d++) {
		amax[0] = std::max(amax[0], amax[d]);
	}
	return amax[0];
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
	auto ein = std::max(u[egas_i] - ek, 0.0);
	if (ein < 0.001 * u[egas_i]) {
		ein = std::pow(std::max(u[tau_i], 0.0), FGAMMA);
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
					int j0 = j;
					j0 = std::max(j0, H_BW);
					j0 = std::min(j0, H_NX - 1 - H_BW);
					int i0 = i;
					i0 = std::max(i0, H_BW);
					i0 = std::min(i0, H_NX - 1 - H_BW);
					U[f][index(i, j)] = U[f][index(H_BW, j0)];
					U[f][index(j, i)] = U[f][index(j0, H_BW)];
					U[f][index(H_NX - 1 - i, j)] = U[f][index(H_NX - 1 - H_BW, j0)];
					U[f][index(j, H_NX - 1 - i)] = U[f][index(j0, H_NX - 1 - H_BW)];
				}
			}
		} else {
			const auto index = [](int i, int j, int k) {
				return i + H_NX * j + k * H_NX * H_NX;
			};

			for (int i = 0; i < H_BW; i++) {
				for (int j = 0; j < H_NX; j++) {
					for (int k = 0; k < H_NX; k++) {
						int j0;
						j0 = std::max(j0, H_BW);
						j0 = std::min(j0, H_NX - 1 - H_BW);
						int i0 = i;
						i0 = std::max(i0, H_BW);
						i0 = std::min(i0, H_NX - 1 - H_BW);
						int k0 = i;
						k0 = std::max(k0, H_BW);
						k0 = std::min(k0, H_NX - 1 - H_BW);
						U[f][index(i, j, k)] = U[f][index(H_BW, j0, k0)];
						U[f][index(j, i, k)] = U[f][index(j0, H_BW, k0)];
						U[f][index(j, k, i)] = U[f][index(j0, k0, H_BW)];
						U[f][index(H_NX - 1 - i, j, k)] = U[f][index(H_NX - 1 - H_BW, j0, k0)];
						U[f][index(j, H_NX - 1 - i, k)] = U[f][index(j0, H_NX - 1 - H_BW, k0)];
						U[f][index(j, H_NX - 1 - k, i)] = U[f][index(j0, k0, H_NX - 1 - H_BW)];
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

	constexpr auto kdelta = kdeltas[NDIM - 1];
	for (int i = 2 * bw; i < H_N3 - 2 * bw; i++) {
		dudt[sx_i][i] += U[sy_i][i] * omega;
		dudt[sy_i][i] -= U[sx_i][i] * omega;
		for (int n = 0; n < NANGMOM; n++) {
			for (int m = 0; m < NDIM; m++) {
				for (int l = 0; l < NDIM; l++) {
					dudt[zx_i + n][i] += kdelta[n][m][l] * F[m][sx_i + l][i];
				}
			}
		}
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
