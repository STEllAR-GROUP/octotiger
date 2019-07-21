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
						assert(
								ir < H_BW || ir >= H_NX - H_BW || jr < H_BW || jr >= H_NX - H_BW || kr < H_BW
										|| kr >= H_NX - H_BW);
						U[f][iiir] = Ushad[f][iii0];
					}
				}
			}
		}
	}
	if (opts().amrbnd_order > 0) {
		static thread_local std::array<std::vector<double>, NDIM> slp;
		for (int d = 0; d < NDIM; d++) {
			slp[d].resize(opts().n_fields);
			slp[d].resize(HS_N3, std::numeric_limits<real>::signaling_NaN());
		}
		for (int f = 0; f < opts().n_fields; f++) {
			for (int i0 = 2; i0 < HS_NX - 2; i0++) {
				for (int j0 = 2; j0 < HS_NX - 2; j0++) {
					for (int k0 = 2; k0 < HS_NX - 2; k0++) {
						const int ir = 2 * i0 - H_BW;
						const int jr = 2 * j0 - H_BW;
						const int kr = 2 * k0 - H_BW;
						const int iii0 = hSindex(i0, j0, k0);
						const int iiir = hindex(ir, jr, kr);
						if (is_coarse[iii0]) {
							//		printf( "%i\n", int(is_coarse[iii0]));

							const auto u0 = Ushad[f][iii0];
							for (int d = 0; d < NDIM; d++) {
								const int da = d == XDIM ? YDIM : XDIM;
								const int db = d == ZDIM ? YDIM : ZDIM;
								const int iiip = iii0 + HS_DN[d];
								const int iiim = iii0 - HS_DN[d];
								double slpp, slpm;
								if (is_coarse[iiip]) {
									slpp = Ushad[f][iiip] - u0;
								} else {
									const int iiir0 = iiir + 2 * H_DN[d];
									assert(iiir0 >= 0);
									assert(iiir0 < H_N3);
									slpp = 0.0;
									slpp += U[f][iiir + 2 * H_DN[d] + 0 * H_DN[da] + 0 * H_DN[db]] - u0;
									slpp += U[f][iiir + 2 * H_DN[d] + 0 * H_DN[da] + 1 * H_DN[db]] - u0;
									slpp += U[f][iiir + 2 * H_DN[d] + 1 * H_DN[da] + 0 * H_DN[db]] - u0;
									slpp += U[f][iiir + 2 * H_DN[d] + 0 * H_DN[da] + 1 * H_DN[db]] - u0;
									slpp /= 3.0;
								}
								if (is_coarse[iiim]) {
									slpm = Ushad[f][iiim] - u0;
								} else {
									const int iiir0 = iiir - H_DN[d];
									assert(iiir0 >= 0);
									assert(iiir0 < H_N3);
									slpm = 0.0;
									slpm += U[f][iiir - H_DN[d] + 0 * H_DN[da] + 0 * H_DN[db]] - u0;
									slpm += U[f][iiir - H_DN[d] + 0 * H_DN[da] + 1 * H_DN[db]] - u0;
									slpm += U[f][iiir - H_DN[d] + 1 * H_DN[da] + 0 * H_DN[db]] - u0;
									slpm += U[f][iiir - H_DN[d] + 0 * H_DN[da] + 1 * H_DN[db]] - u0;
									slpm /= 3.0;
								}
								slp[d][iii0] = minmod(slpp, -slpm);
							}
						}
					}
				}
			}
			for (int ir = 1; ir < H_NX - 1; ir++) {
				for (int jr = 1; jr < H_NX - 1; jr++) {
					for (int kr = 1; kr < H_NX - 1; kr++) {
						const int isgn = ir % 2 ? 1 : -1;
						const int jsgn = jr % 2 ? 1 : -1;
						const int ksgn = kr % 2 ? 1 : -1;
						const int i0 = (ir + H_BW) / 2;
						const int j0 = (jr + H_BW) / 2;
						const int k0 = (kr + H_BW) / 2;
						const int iii0 = hSindex(i0, j0, k0);
						const int iiir = hindex(ir, jr, kr);
						if (is_coarse[iii0]) {
							auto &value = U[f][iiir];
							value -= 0.25 * isgn * slp[XDIM][iii0];
							value -= 0.25 * jsgn * slp[YDIM][iii0];
							value -= 0.25 * ksgn * slp[ZDIM][iii0];
				//			if( f >= zx_i && f <= zz_i ) {
				//				value /= 4.0;
				//			}
							if (opts().angmom) {
								if (f == sx_i) {
									Ushad[zy_i][iii0] -= 0.25 * ksgn * value * dx / 8.0;
									Ushad[zz_i][iii0] += 0.25 * jsgn * value * dx / 8.0;
								} else if (f == sy_i) {
									Ushad[zx_i][iii0] += 0.25 * ksgn * value * dx / 8.0;
									Ushad[zz_i][iii0] -= 0.25 * isgn * value * dx / 8.0;
								} else if (f == sz_i) {
									Ushad[zx_i][iii0] -= 0.25 * jsgn * value * dx / 8.0;
									Ushad[zy_i][iii0] += 0.25 * isgn * value * dx / 8.0;
								}
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
