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
		static thread_local std::vector<std::vector<double>> slpx(opts().n_fields, std::vector<double>(HS_N3));
		static thread_local std::vector<std::vector<double>> slpy(opts().n_fields, std::vector<double>(HS_N3));
		static thread_local std::vector<std::vector<double>> slpz(opts().n_fields, std::vector<double>(HS_N3));

		const auto slopes1 = [this](int f) {
			for (int i0 = 1; i0 < HS_NX - 1; i0++) {
				for (int j0 = 1; j0 < HS_NX - 1; j0++) {
					for (int k0 = 1; k0 < HS_NX - 1; k0++) {
						const int iii0 = hSindex(i0, j0, k0);
						if (is_coarse[iii0]) {
							const auto &u0 = Ushad[f][iii0];
							const auto &uc = Ushad[f];
							constexpr auto theta = 2.0;
							const auto s_x = minmod_theta(uc[iii0 + HS_DNX] - u0, u0 - uc[iii0 - HS_DNX], theta);
							const auto s_y = minmod_theta(uc[iii0 + HS_DNY] - u0, u0 - uc[iii0 - HS_DNY], theta);
							const auto s_z = minmod_theta(uc[iii0 + HS_DNZ] - u0, u0 - uc[iii0 - HS_DNZ], theta);
							slpx[f][iii0] = s_x;
							slpy[f][iii0] = s_y;
							slpz[f][iii0] = s_z;
						}
					}
				}
			}
		};

		const auto slopes2 = [this](int f) {
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
										constexpr auto theta = 1.0;
										const auto s_x = is * slpx[f][iii0];
										const auto s_y = js * slpy[f][iii0];
										const auto s_z = ks * slpz[f][iii0];
										const auto s_xy = minmod_theta(uc[iii0 + is * HS_DNX + js * HS_DNY] - u0, u0 - uc[iii0 - is * HS_DNX - js * HS_DNY],
												theta);
										const auto s_xz = minmod_theta(uc[iii0 + is * HS_DNX + ks * HS_DNZ] - u0, u0 - uc[iii0 - is * HS_DNX - ks * HS_DNZ],
												theta);
										const auto s_yz = minmod_theta(uc[iii0 + js * HS_DNY + ks * HS_DNZ] - u0, u0 - uc[iii0 - js * HS_DNY - ks * HS_DNZ],
												theta);
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

		for (int f = 0; f < sx_i; f++) {
			slopes1(f);
			slopes2(f);
		}

		for (int f = sx_i; f <= sz_i; f++) {
			slopes1(f);
		}

		for (int i0 = 1; i0 < HS_NX - 1; i0++) {
			for (int j0 = 1; j0 < HS_NX - 1; j0++) {
				for (int k0 = 1; k0 < HS_NX - 1; k0++) {
					const int iii0 = hSindex(i0, j0, k0);
					if (is_coarse[iii0]) {
						const auto sym_xy = 0.5 * (slpx[sy_i][iii0] + slpy[sx_i][iii0]);
						const auto sym_xz = 0.5 * (slpx[sz_i][iii0] + slpz[sx_i][iii0]);
						const auto sym_yz = 0.5 * (slpy[sz_i][iii0] + slpz[sy_i][iii0]);
						const auto ant_xy = +3.0 * Ushad[zz_i][iii0] / (dx);
						const auto ant_xz = -3.0 * Ushad[zy_i][iii0] / (dx);
						const auto ant_yz = +3.0 * Ushad[zx_i][iii0] / (dx);
						slpx[sy_i][iii0] = minmod(sym_xy + ant_xy, slpx[sy_i][iii0]);
						slpy[sx_i][iii0] = minmod(sym_xy - ant_xy, slpy[sx_i][iii0]);
						slpx[sz_i][iii0] = minmod(sym_xz + ant_xz, slpx[sz_i][iii0]);
						slpz[sx_i][iii0] = minmod(sym_xz - ant_xz, slpz[sx_i][iii0]);
						slpy[sz_i][iii0] = minmod(sym_yz + ant_yz, slpy[sz_i][iii0]);
						slpz[sy_i][iii0] = minmod(sym_yz - ant_yz, slpz[sy_i][iii0]);
					}
				}
			}
		}

		for (int f = sx_i; f <= sz_i; f++) {
			slopes2(f);
		}

		for (int i0 = 1; i0 < HS_NX - 1; i0++) {
			for (int j0 = 1; j0 < HS_NX - 1; j0++) {
				for (int k0 = 1; k0 < HS_NX - 1; k0++) {
					const int iii0 = hSindex(i0, j0, k0);
					if (is_coarse[iii0]) {
						for (int i = 0; i < 2; i++) {
							for (int j = 0; j < 2; j++) {
								for (int k = 0; k < 2; k++) {
									const auto xsgn = 2 * (i % 2) - 1;
									const auto ysgn = 2 * (j % 2) - 1;
									const auto zsgn = 2 * (k % 2) - 1;
									Ushad[zx_i][iii0] += 0.5 * zsgn * Uf[sy_i][iii0][i][j][k] * dx / 8.0;
									Ushad[zx_i][iii0] -= 0.5 * ysgn * Uf[sz_i][iii0][i][j][k] * dx / 8.0;
									Ushad[zy_i][iii0] -= 0.5 * zsgn * Uf[sx_i][iii0][i][j][k] * dx / 8.0;
									Ushad[zy_i][iii0] += 0.5 * xsgn * Uf[sz_i][iii0][i][j][k] * dx / 8.0;
									Ushad[zz_i][iii0] += 0.5 * ysgn * Uf[sx_i][iii0][i][j][k] * dx / 8.0;
									Ushad[zz_i][iii0] -= 0.5 * xsgn * Uf[sy_i][iii0][i][j][k] * dx / 8.0;
								}
							}
						}
					}
				}
			}
		}

		for (int f = sz_i + 1; f < opts().n_fields; f++) {
			slopes1(f);
			slopes2(f);
		}

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
						fprintf(fp, "%e %e %e\n", y, v0, v1);
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
