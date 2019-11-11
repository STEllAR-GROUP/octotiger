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
	const auto eps = std::max(std::abs(tmp3), std::abs(tmp4)) * 1.0e-12;
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
			if (std::log(std::abs(qr - ql) / (0.5 * (std::abs(qr) + std::abs(ql)))) < 1.0e-12) {
				rc = true;
			}
		}
	}
	if (!rc) {
		printf("%e %e %e %e %e\n", ql, q0, qr, tmp4, tmp3);
	}

	return rc;
}

//#ifdef OCTOTIGER_WITH_CUDA
template<int NDIM, int INX, class PHYS>
const hydro::recon_type<NDIM>& hydro_computer<NDIM, INX, PHYS>::reconstruct_cuda(hydro::state_type &U_, const hydro::x_type &X, safe_real omega) {

//	static thread_local octotiger::fmm::struct_of_array_data<std::array<safe_real, geo::NDIR>, safe_real, geo::NDIR, geo::H_N3, 19>
//		D1_SoA;
	/*static thread_local*/auto D1 = std::vector<std::array<safe_real, geo::NDIR / 2>>(geo::H_N3);
	/*static thread_local*/auto Q = std::vector < std::vector<std::array<safe_real, geo::NDIR>>
			> (nf_, std::vector<std::array<safe_real, geo::NDIR>>(geo::H_N3));

	/*static thread_local*/octotiger::fmm::struct_of_array_data<std::array<safe_real, geo::NDIR>, safe_real, geo::NDIR, geo::H_N3, 19> D1_SoA;
	/*static thread_local*/std::vector<octotiger::fmm::struct_of_array_data<std::array<safe_real, geo::NDIR>, safe_real, geo::NDIR, geo::H_N3, 19>> Q_SoA(nf_);

	/* 	std::cout << " U_ " << U_.size();
	 for (int i = 0; i < U_.size(); i++) {
	 std::cout << U_[i].size() << " ";
	 }
	 std::cout << std::endl;
	 std::cout << " X " << X.size();
	 for (int i = 0; i < X.size(); i++) {
	 std::cout << X[i].size() << " ";
	 }
	 std::cout << std::endl;
	 std::cout << "Constants: NDIR:" << geo::NDIR << " H_N3:" << geo::H_N3 << std::endl; */

	octotiger::fmm::struct_of_array_data<std::array<safe_real, geo::NDIR>, safe_real, geo::NDIR, geo::H_N3, 19> U_SoA;
	U_SoA.concatenate_vectors(U_);

	return Q;
}
//#endif

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

template<int NDIM, int INX>
void reconstruct_ppm(std::vector<std::vector<safe_real>> &q, const std::vector<safe_real> &u, bool smooth) {
	PROFILE();

	static const cell_geometry<NDIM, INX> geo;
	static constexpr auto dir = geo.direction();
	static thread_local auto D1 = std::vector<safe_real>(geo.H_N3, 0.0);
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

inline safe_real superbee(safe_real a, safe_real b) {
	return maxmod(minmod(a, 2 * b), minmod(2 * a, b));
}

template<int NDIM, int INX, class PHYS>
const hydro::recon_type<NDIM>& hydro_computer<NDIM, INX, PHYS>::reconstruct(const hydro::state_type &U_, const hydro::x_type &X, safe_real omega) {
	PROFILE();
	static thread_local auto AM = std::vector < safe_real > (geo::H_N3);
	static thread_local auto Q = std::vector < std::vector<std::vector<safe_real>>
			> (nf_, std::vector < std::vector < safe_real >> (geo::NDIR, std::vector < safe_real > (geo::H_N3)));
	static thread_local auto Theta = std::vector < safe_real > (geo::H_N3, 0.0);

	static constexpr auto xloc = geo::xloc();
	static constexpr auto kdelta = geo::kronecker_delta();
	static constexpr auto vw = geo::volume_weight();
	static constexpr auto dir = geo::direction();

	const auto dx = X[0][geo::H_DNX] - X[0][0];
	const auto &U = PHYS::template pre_recon<INX>(U_, X, omega, angmom_count_ > 0);
	if (angmom_count_ == 0 || NDIM == 1) {
		for (int f = 0; f < nf_; f++) {
			if (f < lx_i || f > lx_i + geo::NANGMOM) {
				reconstruct_ppm<NDIM, INX>(Q[f], U[f], smooth_field_[f]);
			} else {
				reconstruct_minmod<NDIM, INX>(Q[f], U[f]);
			}
		}

	} else {
		for (int f = 0; f < angmom_index_; f++) {
			reconstruct_ppm<NDIM, INX>(Q[f], U[f], smooth_field_[f]);
		}

		int sx_i = angmom_index_;
		int zx_i = sx_i + NDIM;

		for (int angmom_pair = 0; angmom_pair < angmom_count_; angmom_pair++) {
			for (int f = sx_i; f < sx_i + NDIM; f++) {
				reconstruct_ppm<NDIM, INX>(Q[f], U[f], true);
			}
			for (int f = zx_i; f < zx_i + geo::NANGMOM; f++) {
				reconstruct_minmod<NDIM, INX>(Q[f], U[f]);
			}

			for (int n = 0; n < geo::NANGMOM; n++) {
				for (int j = 0; j < geo::H_NX_XM4; j++) {
					for (int k = 0; k < geo::H_NX_YM4; k++) {
#pragma ivdep
						for (int l = 0; l < geo::H_NX_ZM4; l++) {
							const int i = geo::to_index(j + 2, k + 2, l + 2);
							AM[i] = U[zx_i + n][i] * U[0][i];
						}
					}
				}
				for (int m = 0; m < NDIM; m++) {
					for (int q = 0; q < NDIM; q++) {
						const auto kd = kdelta[n][m][q];
						if (kd != 0) {
							for (int d = 0; d < geo::NDIR; d++) {
								if (d != geo::NDIR / 2) {
									for (int j = 0; j < geo::H_NX_XM4; j++) {
										for (int k = 0; k < geo::H_NX_YM4; k++) {
#pragma ivdep
											for (int l = 0; l < geo::H_NX_ZM4; l++) {
												const int i = geo::to_index(j + 2, k + 2, l + 2);
												AM[i] -= vw[d] * kd * 0.5 * xloc[d][m] * Q[sx_i + q][d][i] * Q[0][d][i] * dx;
											}
										}
									}
								}
							}
						}
					}
				}
				for (int m = 0; m < NDIM; m++) {
					for (int q = 0; q < NDIM; q++) {
						const auto f = sx_i + q;
						const auto kd = kdelta[n][m][q];
						if (kd != 0) {
							for (int d = 0; d < geo::NDIR / 2; d++) {
								const auto di = dir[d];
								for (int j = 0; j < geo::H_NX_XM4; j++) {
									for (int k = 0; k < geo::H_NX_YM4; k++) {
#pragma ivdep
										for (int l = 0; l < geo::H_NX_ZM4; l++) {
											const int i = geo::to_index(j + 2, k + 2, l + 2);
											const auto &rho_r = Q[0][d][i];
											const auto &rho_l = Q[0][geo::flip(d)][i];
											auto &qr = Q[f][d][i];
											auto &ql = Q[f][geo::flip(d)][i];
											const auto &ur = U[f][i + di];
											const auto &u0 = U[f][i];
											const auto &ul = U[f][i - di];
											auto b = 12.0 * AM[i] * kd * xloc[d][m] / (dx * (rho_l + rho_r)) + (qr - ql);
											auto c = 6.0 * (0.5 * (qr + ql) - u0);
											const auto blim = superbee(ur - u0, u0 - ul);
											b = minmod(blim, b);
											const auto clim = std::min(0.5 * std::abs(b), 3.0 * std::abs(blim - b));
											c = std::copysign(std::min(std::abs(c), clim), c);
											qr = u0 + 0.5 * b + c / 6.0;
											ql = u0 - 0.5 * b + c / 6.0;
										}
									}
								}
							}
						}
					}
				}
			}
			sx_i += geo::NANGMOM + NDIM;
			zx_i += geo::NANGMOM + NDIM;
		}
		for (int f = angmom_index_ + angmom_count_ * (geo::NANGMOM + NDIM); f < nf_; f++) {
			reconstruct_ppm<NDIM, INX>(Q[f], U[f], smooth_field_[f]);
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
								const auto norm = 0.25 * (std::abs(qp) + std::abs(qm)) * (std::abs(up) + std::abs(um));
								if ((qp - qm) * (up - um) < -1.0e-10 * norm) {
									printf("TVD fail 1 %e\n", (qp - qm) * (up - um) / norm);
								}
								if (!PPM_test(qp, u0, qm)) {
									printf("TVD fail 4\n");
								}
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
									const auto norm = 0.25 * (std::abs(ql) + std::abs(qr)) * (std::abs(ur) + std::abs(ul));
									if ((qr - ul) * (ur - qr) < -1.0e-10 * norm) {
										printf("TVD fail 3 %e\n", (qr - ul) * (ur - qr) / norm);
									}
									if ((ql - ul) * (ur - ql) < -1.0e-10 * norm) {
										printf("TVD fail 5 %e\n", (ql - ul) * (ur - ql) / norm);
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
	PHYS::template post_recon<INX>(U, Q, X, omega, angmom_count_ > 0);

	return Q;
}

