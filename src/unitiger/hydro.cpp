//============================================================================
// Name        : hydro.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "../../octotiger/unitiger/unitiger.hpp"
#include <hpx/include/async.hpp>
#include <hpx/include/future.hpp>
#include <functional>

constexpr int lower_face_members[3][3][9] = { { { 0 } }, { { 3, 0, 6 }, { 1, 0, 2 } }, { { 12, 0, 3, 6, 9, 15, 18 }, { 10, 0, 1, 2, 9, 11, 18, 19, 20 }, { 4, 0,
		1, 2, 3, 5, 6, 7, 8 } } };

constexpr double quad_weights[3][9] = { { 1.0 }, { 2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0 }, { 16. / 36., 1. / 36., 4. / 36., 1. / 36., 4. / 36., 4. / 36., 1. / 36.,
		4. / 36., 1. / 36. } };

#define flip( d ) (NDIR - 1 - (d))

template<class VECTOR>
void to_prim(VECTOR u, double &p, double &v, int dim) {
	const auto rho = u[rho_i];
	const auto rhoinv = 1.0 / rho;
	double ek = 0.0;
	for (int dim = 0; dim < NDIM; dim++) {
		ek += std::pow(u[sx_i + dim], 2) * rhoinv * 0.5;
	}
	auto ein = u[egas_i] - ek;
	if (ein * 0.01 < u[egas_i]) {
		ein = std::pow(u[tau_i], FGAMMA);
	}
	v = u[sx_i + dim] * rhoinv;
	p = (FGAMMA - 1.0) * ein;
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

inline static double minmod_theta(double a, double b, double c) {
	return minmod(c * minmod(a, b), 0.5 * (a + b));
}

inline static double limit_this(double a, double b) {
	return std::copysign(std::min(std::abs(a), std::abs(b)), a);
}

double limiter(double ql, double qr, double ll, double lr, double ul, double u0, double ur) {
	const auto M = std::max(qr, ql);
	const auto m = std::min(qr, ql);
	const auto M_p = std::max(lr, qr);
	const auto M_m = std::max(ll, ql);
	const auto m_p = std::min(lr, qr);
	const auto m_m = std::min(ll, ql);
	double theta = 1.0;
	double tmp;
	if (ul < u0 && u0 < ur) {
		tmp = M - lr;
		if (tmp != 0.0) {
			theta = std::min(theta, (M_p - lr) / tmp);
		}
		tmp = m - ll;
		if (tmp != 0.0) {
			theta = std::min(theta, (m_m - ll) / tmp);
		}
	} else if (ul > u0 && u0 > ur) {
		tmp = m - lr;
		if (tmp != 0.0) {
			theta = std::min(theta, (m_p - lr) / tmp);
		}
		tmp = M - ll;
		if (tmp != 0.0) {
			theta = std::min(theta, (M_m - ll) / tmp);
		}
	}
	assert(theta >= 0.0 && theta <= 1.0);
	return theta;
}

double hydro_flux(std::vector<std::vector<double>> &U, std::vector<std::vector<std::vector<double>>> &F) {

	std::vector<std::vector<std::array<double, NDIR / 2>>> D1(NF, std::vector<std::array<double, NDIR / 2>>(H_N3));
	std::vector<std::vector<std::array<double, NDIR / 2>>> D2(NF, std::vector<std::array<double, NDIR / 2>>(H_N3));
	std::vector<std::vector<std::array<double, NDIR>>> Q(NF, std::vector<std::array<double, NDIR>>(H_N3));
	std::vector < std::vector
			< std::vector<std::array<double, NFACEDIR>>
					>> fluxes(NDIM, std::vector < std::vector<std::array<double, NFACEDIR>> > (NF, std::vector<std::array<double, NFACEDIR>>(H_N3)));

	constexpr auto faces = lower_face_members[NDIM - 1];
	constexpr auto weights = quad_weights[NDIM - 1];

	constexpr auto dir = directions[NDIM - 1];

	int bw = bound_width();

	std::vector<hpx::future<void>> frecon;
	std::vector<hpx::future<double>> futs2;

	for (int f = 0; f < NF; f++) {

		frecon.push_back(hpx::async(hpx::launch::async, [&](int f) {

			for (int i = bw; i < H_N3 - bw; i++) {
				for (int d = 0; d < NDIR; d++) {
					Q[f][i][d] = U[f][i];
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
		}, f));

	}

	for (auto &fut : frecon) {
		fut.get();
	}

	for (int dim = 0; dim < NDIM; dim++) {
		futs2.push_back(hpx::async(hpx::launch::async, [&](int dim) {
			std::array<double, NF> UR, UL, this_flux;
			double amax = 0.0;
			for (int i = 2 * bw; i < H_N3 - 2 * bw; i++) {
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
				for (int i = bw; i < H_N3 - bw; i++) {
					F[dim][f][i] = 0.0;
					for (int fi = 0; fi < NFACEDIR; fi++) {
						F[dim][f][i] += weights[fi] * fluxes[dim][f][i][fi];
					}
				}
			}
			return amax;
		},dim));
	}
	double amax = 0.0;
	for (auto &fut : futs2) {
		amax = std::max(amax, fut.get());
	}
	return amax;
}
