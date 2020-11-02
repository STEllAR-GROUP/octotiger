#include <vector>
#include <iostream>

#include "octotiger/grid.hpp"

#include <aligned_buffer_util.hpp>
#include <buffer_manager.hpp>

void complete_hydro_amr_boundary_cpu(const double dx, const bool energy_only, const std::vector<std::vector<real>> &Ushad, const std::vector<std::atomic<int>> &is_coarse, const std::array<double, NDIM> &xmin, std::vector<std::vector<real>> &U) {
  //std::cout << "Calling hydro cpu version!" << std::endl;

	//using oct_array = std::array<std::array<std::array<double, 2>, 2>, 2>;
	//static thread_local std::vector<std::vector<oct_array>> Uf(opts().n_fields, std::vector<oct_array>(HS_N3));

  std::vector<double, recycler::aggressive_recycle_aligned<real, 32>> unified_uf(opts().n_fields * HS_N3 * 8);
  constexpr int field_offset = HS_N3 * 8;

	const auto limiter = [](double a, double b) {
		return minmod_theta(a, b, 64./37.);
	};

  // Phase 1: From UShad to Uf
	for (int f = 0; f < opts().n_fields; f++) {
		if (!energy_only || f == egas_i) {

			for (int i0 = 1; i0 < HS_NX - 1; i0++) {
				for (int j0 = 1; j0 < HS_NX - 1; j0++) {
					for (int k0 = 1; k0 < HS_NX - 1; k0++) {
						const int iii0 = hSindex(i0, j0, k0);
						if (is_coarse[iii0]) {
							for (int ir = 0; ir < 2; ir++) {
								for (int jr = 0; jr < 2; jr++) {
									for (int kr = 0; kr < 2; kr++) {
                    const int oct_index = ir * 4 + jr * 2 + kr;

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
										//auto &uf = Uf[f][iii0][ir][jr][kr];
										auto &uf = unified_uf[f * field_offset + 8 * iii0 + oct_index];
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
		}
	}

  // Phase 2: Process Uf
	if (!energy_only) {
		for (int i0 = 1; i0 < HS_NX - 1; i0++) {
			for (int j0 = 1; j0 < HS_NX - 1; j0++) {
				for (int k0 = 1; k0 < HS_NX - 1; k0++) {
					const int iii0 = hSindex(i0, j0, k0);
					if (is_coarse[iii0]) {
						for (int ir = 0; ir < 2; ir++) {
							for (int jr = 0; jr < 2; jr++) {
								for (int kr = 0; kr < 2; kr++) {
                  const int oct_index = ir * 4 + jr * 2 + kr;
									const auto i1 = 2 * i0 - H_BW + ir;
									const auto j1 = 2 * j0 - H_BW + jr;
									const auto k1 = 2 * k0 - H_BW + kr;
									const auto x = (i1) * dx + xmin[XDIM];
									const auto y = (j1) * dx + xmin[YDIM];
									const auto z = (k1) * dx + xmin[ZDIM];
									unified_uf[lx_i * field_offset + 8 * iii0 + oct_index] -= 
                    y * unified_uf[sz_i * field_offset + 8 * iii0 + oct_index] -
                    z * unified_uf[sy_i * field_offset + 8 * iii0 + oct_index];
									unified_uf[ly_i * field_offset + 8 * iii0 + oct_index] +=
                    x * unified_uf[sz_i * field_offset + 8 * iii0 + oct_index] -
                    z * unified_uf[sx_i * field_offset + 8 * iii0 + oct_index];
									unified_uf[lz_i * field_offset + 8 * iii0 + oct_index] -=
                    x * unified_uf[sy_i * field_offset + 8 * iii0 + oct_index] -
                    y * unified_uf[sx_i * field_offset + 8 * iii0 + oct_index];
								}
							}
						}
						double zx = 0, zy = 0, zz = 0, rho = 0;
						for (int ir = 0; ir < 2; ir++) {
							for (int jr = 0; jr < 2; jr++) {
								for (int kr = 0; kr < 2; kr++) {
                  const int oct_index = ir * 4 + jr * 2 + kr;
									zx += unified_uf[lx_i * field_offset + 8 * iii0 + oct_index] / 8.0;
									zy += unified_uf[ly_i * field_offset + 8 * iii0 + oct_index] / 8.0;
									zz += unified_uf[lz_i * field_offset + 8 * iii0 + oct_index] / 8.0;
									//			rho += Uf[rho_i][iii0][ir][jr][kr] / 8.0;
								}
							}
						}
						for (int ir = 0; ir < 2; ir++) {
							for (int jr = 0; jr < 2; jr++) {
								for (int kr = 0; kr < 2; kr++) {
									//					const auto factor = Uf[rho_i][iii0][ir][jr][kr] / rho;
									const auto factor = 1.0;
                  const int oct_index = ir * 4 + jr * 2 + kr;
									unified_uf[lx_i * field_offset + 8 * iii0 + oct_index] = zx * factor;
									unified_uf[ly_i * field_offset + 8 * iii0 + oct_index] = zy * factor;
									unified_uf[lz_i * field_offset + 8 * iii0 + oct_index] = zz * factor;
								}
							}
						}
						for (int ir = 0; ir < 2; ir++) {
							for (int jr = 0; jr < 2; jr++) {
								for (int kr = 0; kr < 2; kr++) {
                  const int oct_index = ir * 4 + jr * 2 + kr;
									const auto i1 = 2 * i0 - H_BW + ir;
									const auto j1 = 2 * j0 - H_BW + jr;
									const auto k1 = 2 * k0 - H_BW + kr;
									const auto x = (i1) * dx + xmin[XDIM];
									const auto y = (j1) * dx + xmin[YDIM];
									const auto z = (k1) * dx + xmin[ZDIM];
									unified_uf[lx_i * field_offset + 8 * iii0 + oct_index] +=
                    y * unified_uf[sz_i * field_offset + 8 * iii0 + oct_index] -
                    z * unified_uf[sy_i * field_offset + 8 * iii0 + oct_index];
									unified_uf[ly_i * field_offset + 8 * iii0 + oct_index] -=
                    x * unified_uf[sz_i * field_offset + 8 * iii0 + oct_index] -
                    z * unified_uf[sx_i * field_offset + 8 * iii0 + oct_index];
									unified_uf[lz_i * field_offset + 8 * iii0 + oct_index] +=
                    x * unified_uf[sy_i * field_offset + 8 * iii0 + oct_index] -
                    y * unified_uf[sx_i * field_offset + 8 * iii0 + oct_index];
								}
							}
						}
					}
				}
			}
		}
	}

  // Phase 3: From Uf to U
	for (int f = 0; f < opts().n_fields; f++) {
		if (!energy_only || f == egas_i) {
			for (int i = 0; i < H_NX; i++) {
				for (int j = 0; j < H_NX; j++) {
					for (int k = 0; k < H_NX; k++) {
						const int i0 = (i + H_BW) / 2;
						const int j0 = (j + H_BW) / 2;
						const int k0 = (k + H_BW) / 2;
						const int iii0 = hSindex(i0, j0, k0);
						const int iiir = hindex(i, j, k);
						if (is_coarse[iii0]) {
							int ir, jr, kr;
							if HOST_CONSTEXPR (H_BW % 2 == 0) {
								ir = i % 2;
								jr = j % 2;
								kr = k % 2;
							} else {
								ir = 1 - (i % 2);
								jr = 1 - (j % 2);
								kr = 1 - (k % 2);
							}
              const int oct_index = ir * 4 + jr * 2 + kr;
							U[f][iiir] = unified_uf[f * field_offset + 8 * iii0 + oct_index];
						}
					}
				}
			}
		}
	}
}
