#include "octotiger/grid.hpp"

#include <buffer_manager.hpp>
#include <aligned_buffer_util.hpp>
#ifdef OCTOTIGER_HAVE_CUDA
#include <cuda_buffer_util.hpp>
#include <cuda_runtime.h>
#include <stream_manager.hpp>
#include "octotiger/cuda_util/cuda_helper.hpp"
#endif

void complete_hydro_amr_boundary_cpu(const double dx, const bool energy_only, const std::vector<std::vector<real>> &ushad, const std::vector<std::atomic<int>> &is_coarse, const std::array<double, NDIM> &xmin, std::vector<std::vector<real>> &u);
void launch_complete_hydro_amr_boundary_cuda(stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor, double dx, bool energy_only, const std::vector<std::vector<real>> &ushad, const std::vector<std::atomic<int>> &is_coarse, const std::array<double, NDIM> &xmin, std::vector<std::vector<real>> &u);

CUDA_GLOBAL_METHOD inline double minmod_cuda(double a, double b) {
	return (copysign(0.5, a) + copysign(0.5, b)) * std::min(std::abs(a), abs(b));
}

CUDA_GLOBAL_METHOD inline double minmod_cuda_theta(double a, double b, double c) {
	return minmod_cuda(c * minmod_cuda(a, b), 0.5 * (a + b));
}

CUDA_GLOBAL_METHOD inline double limiter(const double a, const double b) {
    return minmod_cuda_theta(a, b, 64. / 37.);
}

CUDA_GLOBAL_METHOD inline void complete_hydro_amr_boundary_inner_loop(const double dx, const bool energy_only,
    const double* __restrict__ unified_ushad, const int* __restrict__ coarse,
    const double* __restrict__ xmin, double* __restrict__ unified_uf,
    const int i0, const int j0, const int k0, const int nfields) {
    const int field_offset = HS_N3 * 8;
    const int iii0 = i0 * HS_DNX + j0 * HS_DNY + k0 * HS_DNZ;
    if (coarse[iii0]) {
        for (int ir = 0; ir < 2; ir++) {
            for (int jr = 0; jr < 2; jr++) {
                for (int kr = 0; kr < 2; kr++) {
                    for (int f = 0; f < nfields; f++) {
                        if (!energy_only || f == egas_i) {
                            const int oct_index = ir * 4 + jr * 2 + kr;

                            const auto is = ir % 2 ? +1 : -1;
                            const auto js = jr % 2 ? +1 : -1;
                            const auto ks = kr % 2 ? +1 : -1;
                            // const auto& u0 = Ushad[f][iii0];
                            // const auto& uc = Ushad[f];
                            const auto& u0 = unified_ushad[f * HS_N3 + iii0];
                            const double* uc = unified_ushad + f * HS_N3;
                            const auto s_x =
                                limiter(uc[iii0 + is * HS_DNX] - u0, u0 - uc[iii0 - is * HS_DNX]);
                            const auto s_y =
                                limiter(uc[iii0 + js * HS_DNY] - u0, u0 - uc[iii0 - js * HS_DNY]);
                            const auto s_z =
                                limiter(uc[iii0 + ks * HS_DNZ] - u0, u0 - uc[iii0 - ks * HS_DNZ]);
                            const auto s_xy = limiter(uc[iii0 + is * HS_DNX + js * HS_DNY] - u0,
                                u0 - uc[iii0 - is * HS_DNX - js * HS_DNY]);
                            const auto s_xz = limiter(uc[iii0 + is * HS_DNX + ks * HS_DNZ] - u0,
                                u0 - uc[iii0 - is * HS_DNX - ks * HS_DNZ]);
                            const auto s_yz = limiter(uc[iii0 + js * HS_DNY + ks * HS_DNZ] - u0,
                                u0 - uc[iii0 - js * HS_DNY - ks * HS_DNZ]);
                            const auto s_xyz =
                                limiter(uc[iii0 + is * HS_DNX + js * HS_DNY + ks * HS_DNZ] - u0,
                                    u0 - uc[iii0 - is * HS_DNX - js * HS_DNY - ks * HS_DNZ]);
                            // auto &uf = Uf[f][iii0][ir][jr][kr];
                            auto& uf = unified_uf[f * field_offset + 8 * iii0 + oct_index];
                            uf = u0;
                            uf += (9.0 / 64.0) * (s_x + s_y + s_z);
                            uf += (3.0 / 64.0) * (s_xy + s_yz + s_xz);
                            uf += (1.0 / 64.0) * s_xyz;
                        }
                    }
                    if (!energy_only) {
                        const int oct_index = ir * 4 + jr * 2 + kr;
                        const auto i1 = 2 * i0 - H_BW + ir;
                        const auto j1 = 2 * j0 - H_BW + jr;
                        const auto k1 = 2 * k0 - H_BW + kr;
                        const auto x = (i1) *dx + xmin[XDIM];
                        const auto y = (j1) *dx + xmin[YDIM];
                        const auto z = (k1) *dx + xmin[ZDIM];
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
        }
        if (!energy_only) {
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
                        //					const auto factor =
                        // Uf[rho_i][iii0][ir][jr][kr]
                        /// rho;
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
                        const auto x = (i1) *dx + xmin[XDIM];
                        const auto y = (j1) *dx + xmin[YDIM];
                        const auto z = (k1) *dx + xmin[ZDIM];
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
