//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#define TVD_TEST

#include "octotiger/unitiger/physics.hpp"
#include "octotiger/unitiger/physics_impl.hpp"

#include <octotiger/cuda_util/cuda_helper.hpp>
#include <octotiger/cuda_util/cuda_scheduler.hpp>
#include <octotiger/common_kernel/struct_of_array_data.hpp>
#include <octotiger/profiler.hpp>

template<class T>
static inline void limit_slope(T &ql, T q0, T &qr) {
	const T tmp1 = qr - ql;
	const T tmp2 = qr + ql;

	if (bool(qr < q0) != bool(q0 < ql)) {
		qr = ql = q0;
		return;
	}
	const T tmp3 = tmp1 * tmp1 / 6.0;
	const T tmp4 = tmp1 * (q0 - 0.5 * tmp2);
	constexpr auto eps = 1.0e-12;
	if (tmp4 > tmp3) {
		ql = (3.0 * q0 - 2.0 * qr);
	} else if (-tmp3 > tmp4) {
		qr = (3.0 * q0 - 2.0 * ql);
	}
}

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

void reconstruct_constant(std::vector<std::vector<safe_real>> &q, const std::vector<safe_real> &u) {
	static const cell_geometry<NDIM, INX> geo;
	static constexpr auto dir = geo.direction();
	for (int d = 0; d < geo.NDIR; d++) {
		const auto di = dir[d];
		for (int j = 0; j < geo.H_NX_XM6; j++) {
			for (int k = 0; k < geo.H_NX_YM6; k++) {
				for (int l = 0; l < geo.H_NX_ZM6; l++) {
					const int i = geo.to_index(j + 3, k + 3, l + 3);
					q[d][i] = u[i];
				}
			}
		}
	}
}

void reconstruct_ppm(std::vector<std::vector<safe_real>> &q, const std::vector<safe_real> &u, bool smooth, bool slim) {
	PROFILE()
	;

	static const cell_geometry<NDIM, INX> geo;
	static constexpr auto dir = geo.direction();
	static thread_local auto D1 = std::vector<safe_real>(geo.H_N3, 0.0);
	const int xb1 = slim ? geo.H_NX_XM4 : geo.H_NX_XM2;
	const int yb1 = slim ? geo.H_NX_YM4 : geo.H_NX_YM2;
	const int zb1 = slim ? geo.H_NX_ZM4 : geo.H_NX_ZM2;
	const int xb2 = slim ? geo.H_NX_XM6 : geo.H_NX_XM4;
	const int yb2 = slim ? geo.H_NX_YM6 : geo.H_NX_YM4;
	const int zb2 = slim ? geo.H_NX_ZM6 : geo.H_NX_ZM4;
	const int o1 = slim ? 2 : 1;
	const int o2 = slim ? 3 : 2;
	for (int d = 0; d < geo.NDIR / 2; d++) {
		const auto di = dir[d];
		for (int j = 0; j < xb1; j++) {
			for (int k = 0; k < yb1; k++) {
#pragma ivdep
				for (int l = 0; l < zb1; l++) {
					const int i = geo.to_index(j + o1, k + o1, l + o1);
					D1[i] = minmod_theta(u[i + di] - u[i], u[i] - u[i - di], 2.0);
				}
			}
		}
		for (int j = 0; j < xb1; j++) {
			for (int k = 0; k < yb1; k++) {
#pragma ivdep
				for (int l = 0; l < zb1; l++) {
					const int i = geo.to_index(j + o1, k + o1, l + o1);
					q[d][i] = 0.5 * (u[i] + u[i + di]);
					q[d][i] += (1.0 / 6.0) * (D1[i] - D1[i + di]);
					q[geo.flip(d)][i + di] = q[d][i];
				}
			}
		}
	}
	if (!smooth) {
		for (int d = 0; d < geo.NDIR / 2; d++) {
			for (int j = 0; j < xb2; j++) {
				for (int k = 0; k < yb2; k++) {
#pragma ivdep
					for (int l = 0; l < zb2; l++) {
						const int i = geo.to_index(j + o2, k + o2, l + o2);
						auto &qp = q[geo.flip(d)][i];
						auto &qm = q[d][i];
						limit_slope(qm, u[i], qp);
					}
				}
			}
		}
	}

}
;

template<int NDIM, int INX, class PHYS>
const hydro::recon_type<NDIM>& hydro_computer<NDIM, INX, PHYS>::reconstruct(const hydro::state_type &U_, const hydro::x_type &X, safe_real omega) {
	PROFILE()
	;
	static thread_local auto AM = std::vector < std::vector < safe_real >> (geo::NANGMOM, std::vector < safe_real > (geo::H_N3));
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
			reconstruct_ppm(Q[f], U[f], smooth_field_[f], slim_field_[f]);
		}

	} else {
		for (int f = 0; f < angmom_index_; f++) {
			reconstruct_ppm(Q[f], U[f], smooth_field_[f], slim_field_[f]);
		}

		int sx_i = angmom_index_;
		int zx_i = sx_i + NDIM;

		for (int angmom_pair = 0; angmom_pair < angmom_count_; angmom_pair++) {
			for (int f = sx_i; f < sx_i + NDIM; f++) {
				reconstruct_ppm(Q[f], U[f], false, false);
			}
			for (int f = zx_i; f < zx_i + geo::NANGMOM; f++) {
				reconstruct_constant(Q[f], U[f]);
			}

			for (int n = 0; n < geo::NANGMOM; n++) {
				for (int j = 0; j < geo::H_NX_XM4; j++) {
					for (int k = 0; k < geo::H_NX_YM4; k++) {
#pragma ivdep
						for (int l = 0; l < geo::H_NX_ZM4; l++) {
							const int i = geo::to_index(j + 2, k + 2, l + 2);
							AM[n][i] = U[zx_i + n][i] * U[0][i];
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
												AM[n][i] -= vw[d] * kd * 0.5 * xloc[d][m] * Q[sx_i + q][d][i] * Q[0][d][i] * dx;
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
						const auto kd = kdelta[n][m][q];
						if (kd != 0) {
							for (int d = 0; d < geo::NDIR; d++) {
								if (d != geo::NDIR / 2) {
									for (int j = 0; j < geo::H_NX_XM4; j++) {
										for (int k = 0; k < geo::H_NX_YM4; k++) {
#pragma ivdep
											for (int l = 0; l < geo::H_NX_ZM4; l++) {
												const int i = geo::to_index(j + 2, k + 2, l + 2);
												const auto tmp = 6.0 * AM[n][i] / dx;
												Q[sx_i + q][d][i] += kd * 0.5 * xloc[d][m] * tmp / Q[0][d][i];
											}
										}
									}
								}
							}
						}
					}
				}
			}
			for (int dim = 0; dim < NDIM; dim++) {
				const auto f = sx_i + dim;
				for (int d = 0; d < geo::NDIR; d++) {
					if (d == geo::NDIR / 2) {
						continue;
					}
					for (int j = 0; j < geo::H_NX_XM4; j++) {
						for (int k = 0; k < geo::H_NX_YM4; k++) {
#pragma ivdep
							for (int l = 0; l < geo::H_NX_ZM4; l++) {
								const int i = geo::to_index(j + 2, k + 2, l + 2);
								auto s = Q[sx_i + dim][d][i];
								const auto &ur = U[sx_i + dim][i + dir[d]];
								const auto &ul = U[sx_i + dim][i];
								const auto M = std::max(ul, ur);
								const auto m = std::min(ul, ur);
								s = std::min(s, M);
								Q[sx_i + dim][d][i] = std::max(s, m);
							}
						}
					}
				}
				for (int d = 0; d < geo::NDIR / 2; d++) {
					const auto dp = d;
					const auto dm = geo::flip(d);
					for (int j = 0; j < geo::H_NX_XM4; j++) {
						for (int k = 0; k < geo::H_NX_YM4; k++) {
#pragma ivdep
							for (int l = 0; l < geo::H_NX_ZM4; l++) {
								const int i = geo::to_index(j + 2, k + 2, l + 2);
								const auto ur = U[f][i + dir[d]];
								const auto ul = U[f][i];
								auto &qr = Q[f][geo::flip(d)][i + dir[d]];
								auto &ql = Q[f][d][i];
								if ((qr - ql) * (ur - ul) < 0) {
									qr = ql = 0.5 * (qr + ql);
								}
							}
						}
					}
					for (int j = 0; j < geo::H_NX_XM6; j++) {
						for (int k = 0; k < geo::H_NX_YM6; k++) {
#pragma ivdep
							for (int l = 0; l < geo::H_NX_ZM6; l++) {
								const int i = geo::to_index(j + 3, k + 3, l + 3);
								const auto up = U[f][i + dir[d]];
								const auto u0 = U[f][i];
								const auto um = U[f][i - dir[d]];
								auto &qp = Q[f][dp][i];
								auto &qm = Q[f][dm][i];
								if ((qp - qm) * (up - um) < 0) {
									qp = qm = u0;
								} else {
									limit_slope(qp, u0, qm);
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
			reconstruct_ppm(Q[f], U[f], smooth_field_[f], slim_field_[f]);
		}

	}

#ifdef TVD_TEST

	/**** ENSURE TVD TEST***/
	for (int f = 0; f < nf_; f++) {
		if (!smooth_field_[f]) {
			for (int d = 0; d < geo::NDIR / 2; d++) {
				for (int j = 0; j < geo::H_NX_XM6; j++) {
					for (int k = 0; k < geo::H_NX_YM6; k++) {
#pragma ivdep
						for (int l = 0; l < geo::H_NX_ZM6; l++) {
							const int i = geo::to_index(j + 3, k + 3, l + 3);
							const auto up = U[f][i + dir[d]];
							const auto u0 = U[f][i];
							const auto um = U[f][i - dir[d]];
							const auto qp = Q[f][d][i];
							const auto qm = Q[f][geo::flip(d)][i];
							const auto norm = 0.25 * (std::abs(qp) + std::abs(qm)) * (std::abs(up) + std::abs(um));
							if ((qp - qm) * (up - um) < 0) {
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
					for (int j = 0; j < geo::H_NX_XM8; j++) {
						for (int k = 0; k < geo::H_NX_YM8; k++) {
#pragma ivdep
							for (int l = 0; l < geo::H_NX_ZM8; l++) {
								const int i = geo::to_index(j + 4, k + 4, l + 4);
								const auto ur = U[f][i + dir[d]];
								const auto ul = U[f][i];
								const auto ql = Q[f][geo::flip(d)][i + dir[d]];
								const auto qr = Q[f][d][i];
								const auto norm = 0.25 * (std::abs(ql) + std::abs(qr)) * (std::abs(ur) + std::abs(ul));
								if ((ql - qr) * (ur - ul) < 0) {
									printf("TVD fail 2 %e\n", (ql - qr) * (ur - ul) / norm);
								}
								if ((qr - ul) * (ur - qr) < 0.0 || (ql - ul) * (ur - ql) < 0.0) {
									printf("TVD fail3\n");
								}
							}
						}
					}
				}
			}
		}
	}
#endif
	PHYS::template post_recon<INX>(Q, X, omega, angmom_count_ > 0);


	return Q;
}

