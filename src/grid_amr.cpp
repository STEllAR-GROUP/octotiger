//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/grid.hpp"
#include "octotiger/test_problems/amr/amr.hpp"
#include "octotiger/util.hpp"

std::vector<real> grid::get_subset(const std::array<integer, NDIM> &lb, const std::array<integer, NDIM> &ub) {
	std::vector<real> data;
	for (int f = 0; f < opts().n_fields; f++) {
		for (int i = lb[0]; i < ub[0]; i++) {
			for (int j = lb[1]; j < ub[1]; j++) {
				for (int k = lb[2]; k < ub[2]; k++) {
					data.push_back(U[f][hindex(i, j, k)]);
				}
			}
		}
	}
	return std::move(data);
}

void grid::set_hydro_amr_boundary(const std::vector<real> &data, const geo::direction &dir) {

	std::array<integer, NDIM> lb, ub;
	int l = 0;
	get_boundary_size(lb, ub, dir, OUTER, INX / 2, H_BW);
	for (int i = lb[0]; i < ub[0]; i++) {
		for (int j = lb[1]; j < ub[1]; j++) {
			for (int k = lb[2]; k < ub[2]; k++) {
				is_coarse[hSindex(i, j, k)]++;
				assert(i < H_BW || i >= HS_NX - H_BW || j < H_BW || j >= HS_NX - H_BW || k < H_BW || k >= HS_NX - H_BW);
			}
		}
	}

	for (int dim = 0; dim < NDIM; dim++) {
		lb[dim] = std::max(lb[dim] - 1, integer(0));
		ub[dim] = std::min(ub[dim] + 1, integer(HS_NX));
	}

	for (int f = 0; f < opts().n_fields; f++) {
		for (int i = lb[0]; i < ub[0]; i++) {
			for (int j = lb[1]; j < ub[1]; j++) {
				for (int k = lb[2]; k < ub[2]; k++) {
					Ushad[f][hSindex(i, j, k)] = data[l++];
				}
			}
		}
	}
	assert(l == data.size());
}

void grid::complete_hydro_amr_boundary() {

	if (opts().amrbnd_order == 0) {
		for (int f = 0; f < opts().n_fields; f++) {
			for (int ir = 0; ir < H_NX; ir++) {
				for (int jr = 0; jr < H_NX; jr++) {
					for (int kr = 0; kr < H_NX; kr++) {
						const int i0 = (ir + H_BW) / 2;
						const int j0 = (jr + H_BW) / 2;
						const int k0 = (kr + H_BW) / 2;
						const int iii0 = hSindex(i0, j0, k0);
						const int iiir = hindex(ir, jr, kr);
						if (is_coarse[iii0]) {
							assert(ir < H_BW || ir >= H_NX - H_BW || jr < H_BW || jr >= H_NX - H_BW || kr < H_BW || kr >= H_NX - H_BW);
							U[f][iiir] = Ushad[f][iii0];
						}
					}
				}
			}
		}
	} else {
		using oct_array = std::array<std::array<std::array<double, 2>, 2>, 2>;
		static thread_local std::vector<std::vector<oct_array>> Uf(opts().n_fields, std::vector<oct_array>(HS_N3));
		for (int f = 0; f < opts().n_fields; f++) {
			for (int i0 = 1; i0 < HS_NX - 1; i0++) {
				for (int j0 = 1; j0 < HS_NX - 1; j0++) {
					for (int k0 = 1; k0 < HS_NX - 1; k0++) {
						const int iii0 = hSindex(i0, j0, k0);
						if (is_coarse[iii0]) {
							for (int ir = 0; ir < 2; ir++) {
								for (int jr = 0; jr < 2; jr++) {
									for (int kr = 0; kr < 2; kr++) {
										const auto is = ir % 2 ? +1 : -1;
										const auto js = jr % 2 ? +1 : -1;
										const auto ks = kr % 2 ? +1 : -1;
										const auto &u0 = Ushad[f][iii0];
										const auto &uc = Ushad[f];
										constexpr safe_real theta = 0.999 * 64.0 / 37.0;
										const auto s_x = minmod_theta(uc[iii0 + is * HS_DNX] - u0, u0 - uc[iii0 - is * HS_DNX], theta);
										const auto s_y = minmod_theta(uc[iii0 + js * HS_DNY] - u0, u0 - uc[iii0 - js * HS_DNY], theta);
										const auto s_z = minmod_theta(uc[iii0 + ks * HS_DNZ] - u0, u0 - uc[iii0 - ks * HS_DNZ], theta);
										const auto s_xy = minmod_theta(uc[iii0 + is * HS_DNX + js * HS_DNY] - u0, u0 - uc[iii0 - is * HS_DNX - js * HS_DNY], theta);
										const auto s_xz = minmod_theta(uc[iii0 + is * HS_DNX + ks * HS_DNZ] - u0, u0 - uc[iii0 - is * HS_DNX - ks * HS_DNZ], theta);
										const auto s_yz = minmod_theta(uc[iii0 + js * HS_DNY + ks * HS_DNZ] - u0, u0 - uc[iii0 - js * HS_DNY - ks * HS_DNZ], theta);
										const auto s_xyz = minmod_theta(uc[iii0 + is * HS_DNX + js * HS_DNY + ks * HS_DNZ] - u0,
												u0 - uc[iii0 - is * HS_DNX - js * HS_DNY - ks * HS_DNZ], theta);
										auto &uf = Uf[f][iii0][ir][jr][kr];
										uf = u0;
										uf += (9.0 / 64.0) * (s_x + s_y + s_z);
										uf += (3.0 / 64.0) * (s_xy + s_yz + s_xz);
										uf += (1.0 / 64.0) * s_xyz;
									}
								}
							}
						}
					}
				}
			}
		};

//		std::array<double, NDIM> xmin;
//		for (int dim = 0; dim < NDIM; dim++) {
//			xmin[dim] = X[dim][hindex(H_BW, H_BW, H_BW)] - 0.5 * dx;
//		}
//		for (int i0 = 1; i0 < HS_NX - 1; i0++) {
//			for (int j0 = 1; j0 < HS_NX - 1; j0++) {
//				for (int k0 = 1; k0 < HS_NX - 1; k0++) {
//					const int iii0 = hSindex(i0, j0, k0);
//					if (is_coarse[iii0]) {
//						double &lx0 = Ushad[lx_i][iii0];
//						double &ly0 = Ushad[ly_i][iii0];
//						double &lz0 = Ushad[lz_i][iii0];
//						double lx = 0.0;
//						double ly = 0.0;
//						double lz = 0.0;
//						for (int ir = 0; ir < 2; ir++) {
//							for (int jr = 0; jr < 2; jr++) {
//								for (int kr = 0; kr < 2; kr++) {
//									const auto &sx = Uf[sx_i][iii0][ir][jr][kr];
//									const auto &sy = Uf[sy_i][iii0][ir][jr][kr];
//									const auto &sz = Uf[sz_i][iii0][ir][jr][kr];
//									const auto x = (2 * i0 - H_BW + ir + 0.5) * dx + xmin[XDIM];
//									const auto y = (2 * j0 - H_BW + jr + 0.5) * dx + xmin[YDIM];
//									const auto z = (2 * k0 - H_BW + kr + 0.5) * dx + xmin[ZDIM];
//									lx += (y * sz - z * sy) * 0.125;
//									ly -= (x * sz - z * sx) * 0.125;
//									lz += (x * sy - y * sx) * 0.125;
//								}
//							}
//						}
//						if (lz0 != 0.0) {
////							printf("%e %e %e ", lz, lz0, lz - lz0);
//						}
//						const auto dlx = lx - lx0;
//						const auto dly = ly - ly0;
//						const auto dlz = lz - lz0;
//						for (int ir = 0; ir < 2; ir++) {
//							for (int jr = 0; jr < 2; jr++) {
//								for (int kr = 0; kr < 2; kr++) {
//									const auto is = ir % 2 ? +1 : -1;
//									const auto js = jr % 2 ? +1 : -1;
//									const auto ks = kr % 2 ? +1 : -1;
//									auto &sx = Uf[sx_i][iii0][ir][jr][kr];
//									auto &sy = Uf[sy_i][iii0][ir][jr][kr];
//									auto &sz = Uf[sz_i][iii0][ir][jr][kr];
//									const auto x = is * dx;
//									const auto y = js * dx;
//									const auto z = ks * dx;
//									sx += (y * dlz - z * dly) / (dx * dx) * 3.0 / 4.0;
//									sy -= (x * dlz - z * dlx) / (dx * dx) * 3.0 / 4.0;
//									sz += (x * dly - y * dlx) / (dx * dx) * 3.0 / 4.0;
//
//
//								}
//							}
//						}
//						lx = 0.0;
//						ly = 0.0;
//						lz = 0.0;
//						for (int ir = 0; ir < 2; ir++) {
//							for (int jr = 0; jr < 2; jr++) {
//								for (int kr = 0; kr < 2; kr++) {
//									const auto &sx = Uf[sx_i][iii0][ir][jr][kr];
//									const auto &sy = Uf[sy_i][iii0][ir][jr][kr];
//									const auto &sz = Uf[sz_i][iii0][ir][jr][kr];
//									const auto x = (2 * i0 - H_BW + ir + 0.5) * dx + xmin[XDIM];
//									const auto y = (2 * j0 - H_BW + jr + 0.5) * dx + xmin[YDIM];
//									const auto z = (2 * k0 - H_BW + kr + 0.5) * dx + xmin[ZDIM];
//									lx += (y * sz - z * sy) * 0.125;
//									ly -= (x * sz - z * sx) * 0.125;
//									lz += (x * sy - y * sx) * 0.125;
//								}
//							}
//						}
//						if (lz0 != 0.0) {
//							printf("%e\n",lz - lz0);
//						}
//					}
//				}
//			}
//		}

		for (int f = 0; f < opts().n_fields; f++) {
			for (int i = 0; i < H_NX; i++) {
				for (int j = 0; j < H_NX; j++) {
					for (int k = 0; k < H_NX; k++) {
						const int i0 = (i + H_BW) / 2;
						const int j0 = (j + H_BW) / 2;
						const int k0 = (k + H_BW) / 2;
						const int iii0 = hSindex(i0, j0, k0);
						const int iiir = hindex(i, j, k);
						if (is_coarse[iii0]) {
							U[f][iiir] = Uf[f][iii0][1 - (i % 2)][1 - (j % 2)][1 - (k % 2)];
						}
					}
				}
			}
		}
	}

}

std::pair<real, real> grid::amr_error() const {

	const auto is_physical = [this](int i, int j, int k) {
		const integer iii = hindex(i, j, k);
		const auto x = X[XDIM][iii];
		const auto y = X[YDIM][iii];
		const auto z = X[ZDIM][iii];
		if (std::abs(x) > opts().xscale) {
			return true;
		}
		if (std::abs(y) > opts().xscale) {
			return true;
		}
		if (std::abs(z) > opts().xscale) {
			return true;
		}
		return false;
	};

	real sum = 0.0, V = 0.0;
	const real dV = dx * dx * dx;
	for (int i = 0; i < H_NX; i++) {
		for (int j = 0; j < H_NX; j++) {
			for (int k = 0; k < H_NX; k++) {
				const int iii = hindex(i, j, k);
				const auto x = X[XDIM][iii];
				const auto y = X[YDIM][iii];
				const auto z = X[ZDIM][iii];
				if (!is_physical(i, j, k)) {
					const int i0 = (i + H_BW) / 2;
					const int j0 = (j + H_BW) / 2;
					const int k0 = (k + H_BW) / 2;
					const int iii0 = hSindex(i0, j0, k0);
					if (is_coarse[iii0]) {
						const double v0 = amr_test_analytic(x, y, z);
						const double v1 = U[rho_i][iii];
						sum += std::pow(v0 - v1, 2) * dV;
						FILE *fp = fopen("error.txt", "at");
						fprintf(fp, "%e %e %e\n", double(y), double(v0), double(v1));
						fclose(fp);
						V += dV;
					}
				}
			}
		}
	}
	return std::make_pair(sum, V);
}

void grid::clear_amr() {
	std::fill(is_coarse.begin(), is_coarse.end(), 0);
}
