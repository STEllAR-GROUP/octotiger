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

//#ifdef OCTOTIGER_WITH_CUDA
template<int NDIM, int INX>
const hydro::recon_type<NDIM>& hydro_computer<NDIM, INX>::reconstruct_cuda(hydro::state_type &U_, const hydro::x_type &X, safe_real omega) {

//	static thread_local octotiger::fmm::struct_of_array_data<std::array<safe_real, geo::NDIR>, safe_real, geo::NDIR, geo::H_N3, 19>
//		D1_SoA;
	/*static thread_local*/ auto D1 = std::vector<std::array<safe_real, geo::NDIR / 2>>(geo::H_N3);
	/*static thread_local*/ auto Q = std::vector < std::vector<std::array<safe_real, geo::NDIR>> > (nf_, std::vector<std::array<safe_real, geo::NDIR>>(geo::H_N3));

	/*static thread_local*/ octotiger::fmm::struct_of_array_data<std::array<safe_real, geo::NDIR>, safe_real, geo::NDIR, geo::H_N3, 19> D1_SoA;
	/*static thread_local*/ std::vector<octotiger::fmm::struct_of_array_data<std::array<safe_real, geo::NDIR>, safe_real, geo::NDIR, geo::H_N3, 19>> Q_SoA(nf_);

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
const hydro::recon_type<NDIM>& hydro_computer<NDIM, INX>::reconstruct(hydro::state_type &U_, const hydro::x_type &X, safe_real omega) {

	static thread_local auto Q = std::vector < std::vector<std::array<safe_real, geo::NDIR>> > (nf_, std::vector<std::array<safe_real, geo::NDIR>>(geo::H_N3));

	static thread_local auto Q_SoA = std::vector < std::vector<std::vector<safe_real>>
			> (nf_, std::vector < std::vector < safe_real >> (geo::NDIR, std::vector < safe_real > (geo::H_N3)));
	static thread_local auto D1_SoA = std::vector < std::vector < safe_real >> (geo::NDIR, std::vector < safe_real > (geo::H_N3));

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
	auto U = physics < NDIM > ::template pre_recon<INX>(U_, X, omega, angmom_count_ > 0);

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

	const auto reconstruct_ppm = [this](std::vector<std::vector<safe_real>> &q, const std::vector<safe_real> &u, bool smooth) {

		for (int d = 0; d < geo::NDIR; d++) {
			for (int j = 0; j < geo::H_NX_XM2; j++) {
				for (int k = 0; k < geo::H_NX_YM2; k++) {
					for (int l = 0; l < geo::H_NX_ZM2; l++) {
						const int i = geo::to_index(j + 1, k + 1, l + 1);
						const auto di = dir[d];
						D1_SoA[d][i] = minmod_theta(u[i + di] - u[i], u[i] - u[i - di], 2.0);
					}
				}
			}
		}

		for (int d = 0; d < geo::NDIR; d++) {
			const auto di = dir[d];
			for (int j = 0; j < geo::H_NX_XM4; j++) {
				for (int k = 0; k < geo::H_NX_YM4; k++) {
					for (int l = 0; l < geo::H_NX_ZM4; l++) {
						const int i = geo::to_index(j + 2, k + 2, l + 2);
						q[d][i] = 0.5 * (u[i] + u[i + di]);
						q[d][i] += (1.0 / 6.0) * (D1_SoA[d][i] - D1_SoA[d][i + di]);
						q[geo::flip(d)][i + di] = q[d][i];
					}
				}
			}
		}
#define DISABLE_VERTEX_AVG
#ifndef DISABLE_VERTEX_AVG
		for (int j = 0; j < geo::H_NX_XM2; j++) {
			for (int k = 0; k < geo::H_NX_YM2; k++) {
				for (int l = 0; l < geo::H_NX_ZM2; l++) {
					const int i = geo::to_index(j + 1, k + 1, l + 1);
					for (int gi = 0; gi < geo::group_count(); gi++) {
						safe_real sum = 0.0;
						for (int n = 0; n < geo::group_size(gi); n++) {
							const auto pair = geo::group_pair(gi, n);
							sum += q[pair.second][i + pair.first];
						}
						sum /= safe_real(geo::group_size(gi));
						for (int n = 0; n < geo::group_size(gi); n++) {
							const auto pair = geo::group_pair(gi, n);
							q[pair.second][i + pair.first] = sum;
						}
					}
				}
			}
		}
		if (!smooth) {
			for (int d = 0; d < geo::NDIR; d++) {
				if (d != geo::NDIR / 2) {
					const auto di = dir[d];
					for (int j = 0; j < geo::H_NX_XM2; j++) {
						for (int k = 0; k < geo::H_NX_YM2; k++) {
							for (int l = 0; l < geo::H_NX_ZM2; l++) {
								const int i = geo::to_index(j + 1, k + 1, l + 1);
								const auto M = std::max(u[i], u[i + di]);
								const auto m = std::min(u[i], u[i + di]);
								q[d][i] = std::max(q[d][i], m);
								q[d][i] = std::min(q[d][i], M);
							}
						}
					}
				}
			}
		}
#endif

		if (!smooth) {
			for (int d = 0; d < geo::NDIR / 2; d++) {
				for (int j = 0; j < geo::H_NX_XM4; j++) {
					for (int k = 0; k < geo::H_NX_YM4; k++) {
						for (int l = 0; l < geo::H_NX_ZM4; l++) {
							const int i = geo::to_index(j + 2, k + 2, l + 2);
							auto &qp = q[geo::flip(d)][i];
							auto &qm = q[d][i];
							limit_slope(qm, u[i], qp);
						}
					}
				}
			}
		}
	};

	if (angmom_count_ == 0 || NDIM == 1) {
		for (int f = 0; f < nf_; f++) {
			reconstruct_ppm(Q_SoA[f], U[f], smooth_field_[f]);
		}

	} else {
		for (int f = 0; f < angmom_index_; f++) {
			reconstruct_ppm(Q_SoA[f], U[f], smooth_field_[f]);
		}

		int sx_i = angmom_index_;
		int zx_i = sx_i + NDIM;

		for (int angmom_pair = 0; angmom_pair < angmom_count_; angmom_pair++) {
			for (int f = sx_i; f < sx_i + NDIM; f++) {
				reconstruct_ppm(Q_SoA[f], U[f], true);
			}

			SoA2AoS(rho_i, rho_i + 1);
			SoA2AoS(sx_i, sx_i + NDIM);

			for (int j = 0; j < geo::H_NX_XM4; j++) {
				for (int k = 0; k < geo::H_NX_YM4; k++) {
					for (int l = 0; l < geo::H_NX_ZM4; l++) {
						const int i = geo::to_index(j + 2, k + 2, l + 2);

						std::array < safe_real, geo::NANGMOM > Z;
						std::array<std::array<safe_real, geo::NDIR>, NDIM> S;
						for (int dim = 0; dim < geo::NANGMOM; dim++) {
							Z[dim] = U[zx_i + dim][i];
						}
						for (int dim = 0; dim < NDIM; dim++) {
							for (int d = 0; d < geo::NDIR; d++) {
								S[dim][d] = Q[sx_i + dim][i][d];
							}
						}

						physics < NDIM > ::template pre_angmom<INX>(U, Q, Z, S, i, dx);
						auto am1 = measure_angmom(S);
						decltype(Z) am2;
						for (int dim = 0; dim < geo::NANGMOM; dim++) {
							am2[dim] = Z[dim] - am1[dim];
						}
						add_angmom(S, am2);
						physics < NDIM > ::template post_angmom<INX>(U, Q, Z, S, i, dx);

						for (int dim = 0; dim < NDIM; dim++) {
							for (int d = 0; d < geo::NDIR; d++) {
								if (d != geo::NDIR / 2) {
									auto &s = S[dim][d];
									const auto &q = U[sx_i + dim][i + dir[d]];
									const auto &u0 = U[sx_i + dim][i];
									const auto M = std::max(u0, q);
									const auto m = std::min(u0, q);
									s = std::min(s, M);
									s = std::max(s, m);
								}
							}
						}
						for (int f = sx_i; f < sx_i + NDIM; f++) {
							const auto dim = f - sx_i;
							for (int d = 0; d < geo::NDIR / 2; d++) {
								limit_slope(S[dim][d], U[f][i], S[dim][geo::flip(d)]);
							}
						}

						for (int dim = 0; dim < NDIM; dim++) {
							for (int d = 0; d < geo::NDIR; d++) {
								Q[sx_i + dim][i][d] = S[dim][d];
							}
						}
					}
				}
			}

			AoS2SoA(sx_i, zx_i + geo::NANGMOM);

			for (int f = zx_i; f < zx_i + geo::NANGMOM; f++) {
				reconstruct_ppm(Q_SoA[f], U[f], false);
			}
			sx_i += geo::NANGMOM + NDIM;
			zx_i += geo::NANGMOM + NDIM;
		}
		for (int f = angmom_index_ + angmom_count_ * (geo::NANGMOM + NDIM); f < nf_; f++) {
			reconstruct_ppm(Q_SoA[f], U[f], smooth_field_[f]);
		}

	}

	Q_SoA = physics < NDIM > ::template post_recon<INX>(Q_SoA, X, omega, angmom_count_ > 0);

	SoA2AoS(0, nf_);
	return Q;
}

