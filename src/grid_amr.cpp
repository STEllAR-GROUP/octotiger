//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/grid.hpp"
#include "octotiger/test_problems/amr/amr.hpp"
#include "octotiger/unitiger/util.hpp"

double ppm_slope(double am2, double am1, double a00, double ap1, double ap2) {
	const double sm1 = minmod_theta(am1 - am2, a00 - am1, 2.0);
	const double s00 = minmod_theta(a00 - am1, ap1 - a00, 2.0);
	const double sp1 = minmod_theta(ap1 - a00, ap2 - ap1, 2.0);
	double ap = 0.5 * (a00 + ap1) - (1.0 / 6.0) * (sp1 - s00);
	double am = 0.5 * (a00 + am1) - (1.0 / 6.0) * (s00 - sm1);
	limit_slope(am, a00, ap);
	return ap - am;
}

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
					has_coarse[hSindex(i, j, k)]++;
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

		std::array<double, NDIM> xmin;
		for (int dim = 0; dim < NDIM; dim++) {
			xmin[dim] = X[dim][hindex(H_BW, H_BW, H_BW)] - 0.5 * dx;
		}
		const auto dx0 = 2.0 * dx;

		const auto limiter = [](double a, double b) {
			return minmod_theta(a, b, 1.5);
			if (a * b <= 0.0) {
				return 0.0;
			} else {
				constexpr double theta = 64. / 37.;
				constexpr double gamma = 2.0 * (theta - 1.0);
				return theta * (a * a * b + b * b * a) / (a * a + gamma * a * b + b * b);
			}

		};

		for (int i0 = 0; i0 < HS_NX; i0++) {
			for (int j0 = 0; j0 < HS_NX; j0++) {
				for (int k0 = 0; k0 < HS_NX; k0++) {
					const int iii0 = hSindex(i0, j0, k0);
					if (has_coarse[iii0]) {
						for (int f = 0; f < opts().n_fields; f++) {
							if (f != rho_i && f != tau_i && f != egas_i) {
								Ushad[f][iii0] /= Ushad[rho_i][iii0];
							}
						}
					}
//
					const auto x = ((i0 - H_BW) + 0.5) * dx0 + xmin[XDIM];
					const auto y = ((j0 - H_BW) + 0.5) * dx0 + xmin[YDIM];
					const auto z = ((k0 - H_BW) + 0.5) * dx0 + xmin[ZDIM];
					Ushad[lx_i][iii0] -= x * Ushad[sy_i][iii0] - y * Ushad[sx_i][iii0];
					Ushad[ly_i][iii0] += x * Ushad[sz_i][iii0] - z * Ushad[sx_i][iii0];
					Ushad[lz_i][iii0] -= y * Ushad[sz_i][iii0] - z * Ushad[sy_i][iii0];
				}
			}
		}

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
										const auto s_x = limiter(uc[iii0 + is * HS_DNX] - u0, u0 - uc[iii0 - is * HS_DNX]);
										const auto s_y = limiter(uc[iii0 + js * HS_DNY] - u0, u0 - uc[iii0 - js * HS_DNY]);
										const auto s_z = limiter(uc[iii0 + ks * HS_DNZ] - u0, u0 - uc[iii0 - ks * HS_DNZ]);
										const auto s_xy = limiter(uc[iii0 + is * HS_DNX + js * HS_DNY] - u0, u0 - uc[iii0 - is * HS_DNX - js * HS_DNY]);
										const auto s_xz = limiter(uc[iii0 + is * HS_DNX + ks * HS_DNZ] - u0, u0 - uc[iii0 - is * HS_DNX - ks * HS_DNZ]);
										const auto s_yz = limiter(uc[iii0 + js * HS_DNY + ks * HS_DNZ] - u0, u0 - uc[iii0 - js * HS_DNY - ks * HS_DNZ]);
										const auto s_xyz = limiter(uc[iii0 + is * HS_DNX + js * HS_DNY + ks * HS_DNZ] - u0,
												u0 - uc[iii0 - is * HS_DNX - js * HS_DNY - ks * HS_DNZ]);
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

		for (int i = 0; i < H_NX; i++) {
			for (int j = 0; j < H_NX; j++) {
				for (int k = 0; k < H_NX; k++) {
					const int i0 = (i + H_BW) / 2;
					const int j0 = (j + H_BW) / 2;
					const int k0 = (k + H_BW) / 2;
					const int iii0 = hSindex(i0, j0, k0);
					const int iiir = hindex(i, j, k);
					if (is_coarse[iii0]) {
						const auto x = X[XDIM][iiir];
						const auto y = X[YDIM][iiir];
						const auto z = X[ZDIM][iiir];
						U[lx_i][iiir] += x * U[sy_i][iiir] - y * U[sx_i][iiir];
						U[ly_i][iiir] -= x * U[sz_i][iiir] - z * U[sx_i][iiir];
						U[lz_i][iiir] += y * U[sz_i][iiir] - z * U[sy_i][iiir];
						for (int f = 0; f < opts().n_fields; f++) {
							if (f != rho_i && f != tau_i && f != egas_i) {
								U[f][iiir] *= U[rho_i][iiir];
							}
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
	std::fill(has_coarse.begin(), has_coarse.end(), 0);
}
