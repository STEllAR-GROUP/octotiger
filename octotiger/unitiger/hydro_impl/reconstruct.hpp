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

template<int NDIM, int INX, class PHYSICS>
void hydro_computer<NDIM, INX, PHYSICS>::reconstruct_ppm(std::vector<std::vector<safe_real>> &q, const std::vector<safe_real> &u, bool smooth, bool disc_detect,
		const std::vector<std::vector<double>> &disc) {
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
	if (experiment == 1) {
		for (int j = 0; j < geo.H_NX_XM2; j++) {
			for (int k = 0; k < geo.H_NX_YM2; k++) {
#pragma ivdep
				for (int l = 0; l < geo.H_NX_ZM2; l++) {
					const int i = geo.to_index(j + 1, k + 1, l + 1);
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
		for (int d = 0; d < geo.NDIR; d++) {
			const auto di = dir[d];
			for (int j = 0; j < geo.H_NX_XM2; j++) {
				for (int k = 0; k < geo.H_NX_YM2; k++) {
#pragma ivdep
					for (int l = 0; l < geo.H_NX_ZM2; l++) {
						const int i = geo.to_index(j + 1, k + 1, l + 1);
						const auto mx = std::max(u[i + di], u[i]);
						const auto mn = std::min(u[i + di], u[i]);
						q[d][i] = std::min(mx, q[d][i]);
						q[d][i] = std::max(mn, q[d][i]);
					}
				}
			}
		}
	}
	if (disc_detect) {
		constexpr auto eps = 0.01;
		constexpr auto eps2 = 0.001;
		constexpr auto eta1 = 20.0;
		constexpr auto eta2 = 0.05;
		for (int d = 0; d < geo.NDIR / 2; d++) {
			const auto di = dir[d];
			for (int j = 0; j < geo.H_NX_XM4; j++) {
				for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
					for (int l = 0; l < geo.H_NX_ZM4; l++) {
						const int i = geo.to_index(j + 2, k + 2, l + 2);
						const auto &up = u[i + di];
						const auto &u0 = u[i];
						const auto &um = u[i - di];
						const auto dif = up - um;
						if (std::abs(dif) > disc[d][i] * std::min(std::abs(up), std::abs(um))) {
							if (std::min(std::abs(up), std::abs(um)) / std::max(std::abs(up), std::abs(um)) > eps2) {
								const auto d2p = (1.0 / 6.0) * (u[i + 2 * di] + u0 - 2.0 * u[i + di]);
								const auto d2m = (1.0 / 6.0) * (u0 + u[i - 2 * di] - 2.0 * u[i - di]);
								if (d2p * d2m < 0.0) {
									double eta = 0.0;
									if (std::abs(dif) > eps * std::min(std::abs(up), std::abs(um))) {
										eta = -(d2p - d2m) / dif;
									}
									eta = std::max(0.0, std::min(eta1 * (eta - eta2), 1.0));
									if (eta > 0.0) {
										auto ul = um + 0.5 * minmod_theta(u[i] - um, um - u[i - 2 * di], 2.0);
										auto ur = up - 0.5 * minmod_theta(u[i + 2 * di] - up, up - u[i], 2.0);
										auto &qp = q[d][i];
										auto &qm = q[geo.flip(d)][i];
										qp += eta * (ur - qp);
										qm += eta * (ul - qm);
									}
								}
							}
						}
					}
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
const hydro::recon_type<NDIM>& hydro_computer<NDIM, INX, PHYS>::reconstruct(const hydro::state_type &U_, const hydro::x_type &X, safe_real omega) {
	PROFILE();
	static thread_local std::vector<std::vector<safe_real>> AM(geo::NANGMOM, std::vector < safe_real > (geo::H_N3));
	thread_local std::vector<std::vector<std::vector<safe_real>> > Q(nf_,
			std::vector < std::vector < safe_real >> (geo::NDIR, std::vector < safe_real > (geo::H_N3)));
  for (auto &q_field : Q) {
    for (auto &q_dir : q_field) {
      for (auto &q_cell : q_dir) {
        q_cell = 0.0;
      }
    }
  }

	static constexpr auto xloc = geo::xloc();
	static constexpr auto levi_civita = geo::levi_civita();
	static constexpr auto vw = geo::volume_weight();
	static constexpr auto dir = geo::direction();

	const auto dx = X[0][geo::H_DNX] - X[0][0];
	const auto &U = PHYS::template pre_recon<INX>(U_, X, omega, angmom_index_ != -1);
	const auto &cdiscs = PHYS::template find_contact_discs<INX>(U_);
	if (angmom_index_ == -1 || NDIM == 1) {
		for (int f = 0; f < nf_; f++) {
			if (f < lx_i || f > lx_i + geo::NANGMOM || NDIM == 1) {
				reconstruct_ppm(Q[f], U[f], smooth_field_[f], disc_detect_[f], cdiscs);
			} else {
				reconstruct_minmod<NDIM, INX>(Q[f], U[f]);
			}
		}

	} else {
		for (int f = 0; f < angmom_index_; f++) {
			reconstruct_ppm(Q[f], U[f], smooth_field_[f], disc_detect_[f], cdiscs);
		}

		int sx_i = angmom_index_;
		int zx_i = sx_i + NDIM;

		for (int f = sx_i; f < sx_i + NDIM; f++) {
			reconstruct_ppm(Q[f], U[f], true, false, cdiscs);
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
						AM[n][i] = U[zx_i + n][i] * U[0][i];
					}
				}
			}
			for (int m = 0; m < NDIM; m++) {
				for (int q = 0; q < NDIM; q++) {
					const auto lc = levi_civita[n][m][q];
					if (lc != 0) {
						for (int d = 0; d < geo::NDIR; d++) {
							if (d != geo::NDIR / 2) {
								for (int j = 0; j < geo::H_NX_XM4; j++) {
									for (int k = 0; k < geo::H_NX_YM4; k++) {
#pragma ivdep
										for (int l = 0; l < geo::H_NX_ZM4; l++) {
											const int i = geo::to_index(j + 2, k + 2, l + 2);
											AM[n][i] -= vw[d] * lc * 0.5 * xloc[d][m] * Q[sx_i + q][d][i] * Q[0][d][i] * dx;
										}
									}
								}
							}
						}
					}
				}
			}
		}
		for (int q = 0; q < NDIM; q++) {
			const auto f = sx_i + q;
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
							const auto b0 = qr - ql;
							auto b = b0;
							for (int n = 0; n < geo::NANGMOM; n++) {
								for (int m = 0; m < NDIM; m++) {
									const auto lc = levi_civita[n][m][q];
									b += 12.0 * AM[n][i] * lc * xloc[d][m] / (dx * (rho_l + rho_r));
								}
							}
							double blim;
							if ((ur - u0) * (u0 - ul) <= 0.0) {
								blim = 0.0;
							} else {
								blim = b0;
							}
							b = minmod(blim, b);
							qr += 0.5 * (b - b0);
							ql -= 0.5 * (b - b0);
							if (ur > u0 && u0 > ul) {
								if (qr > ur) {
									ql -= (qr - ur);
									qr = ur;
								} else if (ql < ul) {
									qr -= (ql - ul);
									ql = ul;
								}
							} else if (ur < u0 && u0 < ul) {
								if (qr < ur) {
									ql -= (qr - ur);
									qr = ur;
								} else if (ql > ul) {
									qr -= (ql - ul);
									ql = ul;
								}
							}
							make_monotone(qr, u0, ql);
						}
					}
				}
			}
		}
		for (int f = angmom_index_ + geo::NANGMOM + NDIM; f < nf_; f++) {
			reconstruct_ppm(Q[f], U[f], smooth_field_[f], disc_detect_[f], cdiscs);
		}

	}

#ifdef TVD_TEST
	{
		PROFILE();
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

