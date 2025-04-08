//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

//#define TVD_TEST

#include "octotiger/unitiger/physics.hpp"
#include "octotiger/unitiger/physics_impl.hpp"

#include <octotiger/cuda_util/cuda_helper.hpp>
#include <octotiger/cuda_util/cuda_scheduler.hpp>
#include <octotiger/common_kernel/struct_of_array_data.hpp>
#include <octotiger/profiler.hpp>

template<class T>
static inline bool PPM_test(const T &ql, const T &q0, const T &qr) {
	const T tmp1 = qr - ql;
	const T tmp2 = qr + ql;
	const T tmp3 = tmp1 * tmp1 / 6.0;
	const T tmp4 = tmp1 * (q0 - 0.5 * tmp2);
	const auto eps = std::max(std::abs(tmp3), std::abs(tmp4)) * 1.0e-10;
	bool rc;
	if (bool(qr < q0) != bool(q0 < ql)) {
		rc = false;
	} else {
		if (tmp4 > tmp3 + eps) {
			rc = false;
		} else if (-tmp3 > tmp4 + eps) {
			rc = false;
		} else {
			rc = true;
		}
	}
	if (!rc) {
		if (qr != 0.0 || ql != 0.0) {
			const auto test = std::log(std::abs(qr - ql) / (0.5 * (std::abs(qr) + std::abs(ql))));
			if (test < 1.0e-10) {
				rc = true;
			}
		}
	}
	if (!rc) {
		printf("%e %e %e %e %e\n", ql, q0, qr, tmp4, tmp3);
	}

	return rc;
}


template<int NDIM, int INX>
void reconstruct_minmod(std::vector<std::vector<safe_real>> &q, const std::vector<safe_real> &u) {
	PROFILE();
	static const cell_geometry<NDIM, INX> geo;
	static constexpr auto dir = geo.direction();
	for (int d = 0; d < geo.NDIR; d++) {
		const auto di = dir[d];
		for (int j = 0; j < geo.H_NX_XM2; j++) {
			for (int k = 0; k < geo.H_NX_YM2; k++) {
				for (int l = 0; l < geo.H_NX_ZM2; l++) {
					const int i = geo.to_index(j + 1, k + 1, l + 1);
					q[d][i] = u[i] + 0.5 * minmod(u[i + di] - u[i], u[i] - u[i - di]);
				}
			}
		}
	}
}

template<int NDIM, int INX, class PHYSICS>
void hydro_computer<NDIM, INX, PHYSICS>::reconstruct_ppm(std::vector<std::vector<safe_real>> &q,
		const std::vector<safe_real> &u, bool smooth, bool disc_detect, const std::vector<std::vector<double>> &disc) {
	PROFILE();

	static const cell_geometry<NDIM, INX> geo;
	static constexpr auto dir = geo.direction();
	static thread_local auto D1 = std::vector < safe_real > (geo.H_N3, 0.0);
	for (int d = 0; d < geo.NDIR / 2; d++) {
		const auto di = dir[d];
		for (int j = 0; j < geo.H_NX_XM2; j++) {
			for (int k = 0; k < geo.H_NX_YM2; k++) {
#pragma ivdep
				for (int l = 0; l < geo.H_NX_ZM2; l++) {
					const int i = geo.to_index(j + 1, k + 1, l + 1);
					D1[i] = minmod_theta(u[i + di] - u[i], u[i] - u[i - di], 2.0);
				}
			}
		}
		for (int j = 0; j < geo.H_NX_XM2; j++) {
			for (int k = 0; k < geo.H_NX_YM2; k++) {
#pragma ivdep
				for (int l = 0; l < geo.H_NX_ZM2; l++) {
					const int i = geo.to_index(j + 1, k + 1, l + 1);
					q[d][i] = 0.5 * (u[i] + u[i + di]);
					q[d][i] += (1.0 / 6.0) * (D1[i] - D1[i + di]);
					q[geo.flip(d)][i + di] = q[d][i];
				}
			}
		}
	}
	if (!smooth) {
		for (int d = 0; d < geo.NDIR / 2; d++) {
			for (int j = 0; j < geo.H_NX_XM4; j++) {
				for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
					for (int l = 0; l < geo.H_NX_ZM4; l++) {
						const int i = geo.to_index(j + 2, k + 2, l + 2);
						auto &qp = q[geo.flip(d)][i];
						auto &qm = q[d][i];
						make_monotone(qm, u[i], qp);
					}
				}
			}
		}
	}

}

inline safe_real maxmod(safe_real a, safe_real b) {
	return (std::copysign(0.5, a) + std::copysign(0.5, b)) * std::max(std::abs(a), std::abs(b));
}

inline safe_real vanleer(safe_real a, safe_real b) {
	const auto abs_a = std::abs(a);
	const auto abs_b = std::abs(b);
	const auto den = abs_a + abs_b;
	if (den > 0.0) {
		return (a * abs_b + b * abs_a) / den;
	} else {
		return 0.0;
	}
}

inline safe_real ospre(safe_real a, safe_real b) {
	const auto a2 = a * a;
	const auto b2 = b * b;
	if (a * b <= 0.0) {
		return 0.0;
	} else {
		return 1.5 * (a2 * b + b2 * a) / (a2 + b2 + a * b);
	}
}

template<int NDIM, int INX, class PHYS>
const hydro::recon_type<NDIM>& hydro_computer<NDIM, INX, PHYS>::reconstruct(const hydro::state_type &U_,
		const hydro::x_type &X, safe_real omega) {
	PROFILE();
	static thread_local std::vector<std::vector<safe_real>> AM(geo::NANGMOM, std::vector < safe_real > (geo::H_N3));
	static thread_local std::vector<std::vector<std::vector<safe_real>> > Q(nf_,
			std::vector < std::vector < safe_real >> (geo::NDIR, std::vector < safe_real > (geo::H_N3)));

	static constexpr auto xloc = geo::xloc();
	static constexpr auto levi_civita = geo::levi_civita();
	static constexpr auto vw = geo::volume_weight();
	static constexpr auto dir = geo::direction();

	const auto dx = X[0][geo::H_DNX] - X[0][0];
	const auto &U = PHYS::template pre_recon<INX>(U_, X, omega, angmom_index_ != -1);
	const auto &cdiscs = PHYS::template find_contact_discs<INX>(U_);
	if (angmom_index_ == -1 || NDIM == 1) {
		for (int f = 0; f < nf_; f++) {
//			if (f < lx_i || f > lx_i + geo::NANGMOM || NDIM == 1) {
			reconstruct_ppm(Q[f], U[f], false, disc_detect_[f], cdiscs);
//			reconstruct_ppm(Q[f], U[f], smooth_field_[f], disc_detect_[f], cdiscs);
//			} else {
//				reconstruct_minmod<NDIM, INX>(Q[f], U[f]);
//			}
//			reconstruct_minmod<NDIM, INX>(Q[f], U[f]);
		}

	}


#ifdef TVD_TEST
	{
		PROFILE();
		/**** ENSURE TVD TEST***/
		for (int f = 0; f < nf_; f++) {
			if (!smooth_field_[f]) {
				for (int d = 0; d < geo::NDIR / 2; d++) {
					for (int j = 0; j < geo::H_NX_XM4; j++) {
						for (int k = 0; k < geo::H_NX_YM4; k++) {
#pragma ivdep
							for (int l = 0; l < geo::H_NX_ZM4; l++) {
								const int i = geo::to_index(j + 2, k + 2, l + 2);
								const auto up = U[f][i + dir[d]];
								const auto u0 = U[f][i];
								const auto um = U[f][i - dir[d]];
								const auto qp = Q[f][d][i];
								const auto qm = Q[f][geo::flip(d)][i];
								auto norm = std::max(std::abs(u0), std::max(std::abs(up), std::abs(um)));
								norm *= norm;
								if ((qp - qm) * (up - um) < -1.0e-12 * norm) {
									printf("TVD fail 1 %e\n", (qp - qm) * (up - um) / norm);
									abort();
								}
//								if (!PPM_test(qp, u0, qm)) {
//									printf("TVD fail 4\n");
//									abort();
//								}
							}
						}
					}
				}
				for (int d = 0; d < geo::NDIR; d++) {
					if (d != geo::NDIR / 2) {
						for (int j = 0; j < geo::H_NX_XM6; j++) {
							for (int k = 0; k < geo::H_NX_YM6; k++) {
#pragma ivdep
								for (int l = 0; l < geo::H_NX_ZM6; l++) {
									const int i = geo::to_index(j + 3, k + 3, l + 3);
									const auto ur = U[f][i + dir[d]];
									const auto ul = U[f][i];
									const auto ql = Q[f][geo::flip(d)][i + dir[d]];
									const auto qr = Q[f][d][i];
									auto norm = std::max(std::abs(ur), std::abs(ul));
									norm *= norm;
									if ((qr - ul) * (ur - qr) < -1.0e-12 * norm) {
										printf("TVD fail 3 %e\n", (qr - ul) * (ur - qr) / norm);
										abort();
									}
									if ((ql - ul) * (ur - ql) < -1.0e-12 * norm) {
										printf("TVD fail 5 %e\n", (ql - ul) * (ur - ql) / norm);
										abort();
									}
								}
							}
						}
					}
				}
			}
		}
	}

#endif
	PHYS::template post_recon<INX>(Q, X, omega, angmom_index_ != -1);

	return Q;
}

