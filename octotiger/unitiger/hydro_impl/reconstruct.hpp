//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#include "octotiger/unitiger/physics.hpp"
#include "octotiger/unitiger/physics_impl.hpp"

#include <octotiger/cuda_util/cuda_helper.hpp>
#include <octotiger/cuda_util/cuda_scheduler.hpp>
#include <octotiger/common_kernel/struct_of_array_data.hpp>
#include <octotiger/profiler.hpp>

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
	PROF_BEGIN
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
				for (int l = 0; l < zb1; l++) {
					const int i = geo.to_index(j + o1, k + o1, l + o1);
					D1[i] = minmod_theta(u[i + di] - u[i], u[i] - u[i - di], 2.0);
				}
			}
		}
		for (int j = 0; j < xb1; j++) {
			for (int k = 0; k < yb1; k++) {
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
	PROF_END;
}
;

template<int NDIM, int INX, class PHYS>
const hydro::recon_type<NDIM>& hydro_computer<NDIM, INX, PHYS>::reconstruct(const hydro::state_type &U_, const hydro::x_type &X, safe_real omega) {
	PROF_BEGIN
	;
	static thread_local auto Q = std::vector < std::vector<std::array<safe_real, geo::NDIR>> > (nf_, std::vector<std::array<safe_real, geo::NDIR>>(geo::H_N3));
	static thread_local auto QS = std::vector < std::vector<std::array<safe_real, geo::NDIR>>
			> (NDIM, std::vector<std::array<safe_real, geo::NDIR>>(geo::H_N3));
	static thread_local auto Q_SoA = std::vector < std::vector<std::vector<safe_real>>
			> (nf_, std::vector < std::vector < safe_real >> (geo::NDIR, std::vector < safe_real > (geo::H_N3)));
	static thread_local auto Theta = std::vector < safe_real > (geo::H_N3, 0.0);

	static const auto SoA2AoS = [](int f1, int f2) {
		for (int f = f1; f < f2; f++) {
			for (int i = 0; i < geo::H_N3; i++) {
				for (int d = 0; d < geo::NDIR; d++) {
					Q[f][i][d] = Q_SoA[f][d][i];
				}
			}
		}
	};

	static const auto AoS2SoA = [](int f1, int f2) {
		for (int f = f1; f < f2; f++) {
			for (int i = 0; i < geo::H_N3; i++) {
				for (int d = 0; d < geo::NDIR; d++) {
					Q_SoA[f][d][i] = Q[f][i][d];
				}
			}
		}
	};

	static constexpr auto xloc = geo::xloc();
	static constexpr auto kdelta = geo::kronecker_delta();
	static constexpr auto vw = geo::volume_weight();
	static constexpr auto dir = geo::direction();

	const auto dx = X[0][geo::H_DNX] - X[0][0];
	const auto &U = PHYS::template pre_recon<INX>(U_, X, omega, angmom_count_ > 0);

	const auto measure_angmom = [dx](const std::array<std::array<safe_real, geo::NDIR>, NDIM> &C) {
		std::array < safe_real, geo::NANGMOM > L;
		for (int n = 0; n < geo::NANGMOM; n++) {
			L[n] = 0.0;
			for (int m = 0; m < NDIM; m++) {
				for (int l = 0; l < NDIM; l++) {
					for (int d = 0; d < geo::NDIR; d++) {
						if (d != geo::NDIR / 2) {
							L[n] += vw[d] * kdelta[n][m][l] * 0.5 * xloc[d][m] * C[l][d] * dx;
						}
					}
				}
			}
		}
		return L;
	};

	const auto add_angmom = [dx](std::array<std::array<safe_real, geo::NDIR>, NDIM> &C, std::array<safe_real, geo::NANGMOM> &Z) {
		for (int d = 0; d < geo::NDIR; d++) {
			if (d != geo::NDIR / 2) {
				for (int n = 0; n < geo::NANGMOM; n++) {
					for (int m = 0; m < NDIM; m++) {
						for (int l = 0; l < NDIM; l++) {
							const auto tmp = 6.0 * Z[n] / dx;
							C[l][d] += kdelta[n][m][l] * 0.5 * xloc[d][m] * tmp;
						}
					}
				}
			}
		}
	};

	if (angmom_count_ == 0 || NDIM == 1) {
		for (int f = 0; f < nf_; f++) {
			reconstruct_ppm(Q_SoA[f], U[f], smooth_field_[f], slim_field_[f]);
		}

	} else {
		for (int f = 0; f < angmom_index_; f++) {
			reconstruct_ppm(Q_SoA[f], U[f], smooth_field_[f], slim_field_[f]);
		}

		int sx_i = angmom_index_;
		int zx_i = sx_i + NDIM;

		SoA2AoS(rho_i, rho_i + 1);

		for (int angmom_pair = 0; angmom_pair < angmom_count_; angmom_pair++) {
			for (int f = sx_i; f < sx_i + NDIM; f++) {
				reconstruct_ppm(Q_SoA[f], U[f], false, false);
			}
			for (int f = zx_i; f < zx_i + geo::NANGMOM; f++) {
				reconstruct_constant(Q_SoA[f], U[f]);
			}

			SoA2AoS(sx_i, sx_i + NDIM);

			for (int j = 0; j < geo::H_NX_XM4; j++) {
				for (int k = 0; k < geo::H_NX_YM4; k++) {
					for (int l = 0; l < geo::H_NX_ZM4; l++) {
						const int i = geo::to_index(j + 2, k + 2, l + 2);

						std::array < safe_real, geo::NANGMOM > Z;
						std::array<std::array<safe_real, geo::NDIR>, NDIM> S;
						for (int n = 0; n < geo::NANGMOM; n++) {
							Z[n] = U[zx_i + n][i];
						}
						for (int dim = 0; dim < NDIM; dim++) {
							for (int d = 0; d < geo::NDIR; d++) {
								S[dim][d] = Q[sx_i + dim][i][d] * Q[0][i][d];
							}
						}

						auto am1 = measure_angmom(S);
						decltype(Z) am2;
						for (int dim = 0; dim < geo::NANGMOM; dim++) {
							am2[dim] = Z[dim] * U[0][i] - am1[dim];
						}
						add_angmom(S, am2);

						for (int dim = 0; dim < NDIM; dim++) {
							for (int d = 0; d < geo::NDIR; d++) {
								if (d != geo::NDIR / 2) {
									auto s = S[dim][d] / Q[0][i][d];
									const auto &up = U[sx_i + dim][i + dir[d]];
									const auto &u0 = U[sx_i + dim][i];
									const auto M = std::max(u0, up);
									const auto m = std::min(u0, up);
									s = std::min(s, M);
									S[dim][d] = std::max(s, m);
								}
							}
						}

						for (int dim = 0; dim < NDIM; dim++) {
							for (int d = 0; d < geo::NDIR; d++) {
								QS[dim][i][d] = S[dim][d];
							}
							for (int d = 0; d < geo::NDIR / 2; d++) {
								const auto dp = d;
								const auto dm = geo::flip(d);
								limit_slope(QS[dim][i][dm], U[sx_i + dim][i], QS[dim][i][dp]);
							}
						}
					}
				}
			}
			for (int d = 0; d < geo::NDIR / 2; d++) {
				for (int f = sx_i; f < sx_i + NDIM; f++) {
					const auto dim = f - sx_i;
					const auto dp = d;
					const auto dm = geo::flip(d);
					for (int j = 0; j < geo::H_NX_XM6; j++) {
						for (int k = 0; k < geo::H_NX_YM6; k++) {
							for (int l = 0; l < geo::H_NX_ZM6; l++) {
								const int i = geo::to_index(j + 3, k + 3, l + 3);
								const auto ur = U[f][i + dir[d]];
								const auto u0 = U[f][i];
								const auto ul = U[f][i - dir[d]];
								const auto qr = (Q[f][i][dp] + Q[f][i + dir[d]][dm]) / 2.0;
								const auto ql = (Q[f][i][dm] + Q[f][i - dir[d]][dp]) / 2.0;
								const auto Mr = std::max(qr, QS[dim][i + dir[d]][dm]);
								const auto mr = std::min(qr, QS[dim][i + dir[d]][dm]);
								const auto Ml = std::max(ql, QS[dim][i - dir[d]][dp]);
								const auto ml = std::min(ql, QS[dim][i - dir[d]][dp]);
								const auto M = std::max(QS[dim][i][dp], QS[dim][i][dm]);
								const auto m = std::min(QS[dim][i][dp], QS[dim][i][dm]);
								double theta = 1.0;
								if (ur > u0 && u0 > ul) {
									if (M - u0 != 0.0) {
										theta = std::min(theta, (Mr - u0) / (M - u0));
									}
									if (m - u0 != 0.0) {
										theta = std::min(theta, (ml - u0) / (m - u0));
									}
								} else if (ur < u0 && u0 < ul) {
									if (M - u0 != 0.0) {
										theta = std::min(theta, (Ml - u0) / (M - u0));
									}
									if (m - u0 != 0.0) {
										theta = std::min(theta, (mr - u0) / (m - u0));
									}
								}
								theta = std::min(theta, 1.0);
								if (theta < 0.0)
									printf("%e\n", theta);
								Theta[i] = theta;
							}
						}
					}
					for (int j = 0; j < geo::H_NX_XM6; j++) {
						for (int k = 0; k < geo::H_NX_YM6; k++) {
							for (int l = 0; l < geo::H_NX_ZM6; l++) {
								const int i = geo::to_index(j + 3, k + 3, l + 3);
								const auto dp = d;
								const auto dm = geo::flip(d);
								for (int d = 0; d < geo::NDIR / 2; d++) {
									const auto &theta = Theta[i];
									Q[f][i][dp] = theta * QS[dim][i][dp] + (1.0 - theta) * Q[f][i][dp];
									Q[f][i][dm] = theta * QS[dim][i][dm] + (1.0 - theta) * Q[f][i][dm];
								}
							}
						}
					}
				}
			}

			AoS2SoA(sx_i, sx_i + NDIM);

			sx_i += geo::NANGMOM + NDIM;
			zx_i += geo::NANGMOM + NDIM;
		}
		for (int f = angmom_index_ + angmom_count_ * (geo::NANGMOM + NDIM); f < nf_; f++) {
			reconstruct_ppm(Q_SoA[f], U[f], smooth_field_[f], slim_field_[f]);
		}

	}

	PHYS::template post_recon<INX>(Q_SoA, X, omega, angmom_count_ > 0);

	PROF_END;
	return Q_SoA;
}

