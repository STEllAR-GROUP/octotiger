#include "octotiger/grid.hpp"

#include <aligned_buffer_util.hpp>
#include <buffer_manager.hpp>
#ifdef OCTOTIGER_HAVE_CUDA
#include <cuda_buffer_util.hpp>
#include <cuda_runtime.h>
#include <stream_manager.hpp>
#include "octotiger/cuda_util/cuda_helper.hpp"
#endif
#include "octotiger/util/vec_scalar_host_wrapper.hpp"

void complete_hydro_amr_boundary_cpu(const double dx, const bool energy_only,
    const std::vector<std::vector<real>>& ushad, const std::vector<std::atomic<int>>& is_coarse,
    const std::array<double, NDIM>& xmin, std::vector<std::vector<real>>& u);
void complete_hydro_amr_boundary_vc(const double dx, const bool energy_only,
    const std::vector<std::vector<real>>& Ushad, const std::vector<std::atomic<int>>& is_coarse,
    const std::array<double, NDIM>& xmin, std::vector<std::vector<double>>& U);
void launch_complete_hydro_amr_boundary_cuda(
    stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor, double dx,
    bool energy_only, const std::vector<std::vector<real>>& ushad,
    const std::vector<std::atomic<int>>& is_coarse, const std::array<double, NDIM>& xmin,
    std::vector<std::vector<real>>& u);

template <typename T, typename mask_t, typename index_t>
CUDA_GLOBAL_METHOD inline void complete_hydro_amr_boundary_inner_loop(const double dx, const bool energy_only,
    const double* __restrict__ unified_ushad, const int* __restrict__ coarse,
    const double* __restrict__ xmin, const int i0, const int j0,
    const int k0, const int nfields, const mask_t mask, const index_t k , const int iii0 , T * __restrict__ uf_local) {
    const int field_offset = HS_N3 * 8;
    for (int ir = 0; ir < 2; ir++) {
        for (int jr = 0; jr < 2; jr++) {
            #pragma unroll
            for (int kr = 0; kr < 2; kr++) {
                for (int f = 0; f < nfields; f++) {
                    if (!energy_only || f == egas_i) {

                        const int oct_index = ir * 4 + jr * 2 + kr;
                        const auto is = ir % 2 ? +1 : -1;
                        const auto js = jr % 2 ? +1 : -1;
                        const auto ks = kr % 2 ? +1 : -1;

                        const double* uc = unified_ushad + f * HS_N3;
                        const T u0 = load_value<T>(unified_ushad, f * HS_N3 + iii0);

                        const T uc_x = load_value<T>(uc, iii0 + is * HS_DNX);
                        const T uc_y = load_value<T>(uc, iii0 + js * HS_DNY);
                        const T uc_z = load_value<T>(uc, iii0 + ks * HS_DNZ);
                        const T uc_x_neg = load_value<T>(uc, iii0 - is * HS_DNX);
                        const T uc_y_neg = load_value<T>(uc, iii0 - js * HS_DNY);
                        const T uc_z_neg = load_value<T>(uc, iii0 - ks * HS_DNZ);

                        const auto s_x = limiter_wrapper(uc_x - u0, u0 - uc_x_neg);
                        const auto s_y = limiter_wrapper(uc_y - u0, u0 - uc_y_neg);
                        const auto s_z = limiter_wrapper(uc_z - u0, u0 - uc_z_neg);

                        const T uc_xy = load_value<T>(uc, iii0 + is * HS_DNX + js * HS_DNY);
                        const T uc_xz = load_value<T>(uc, iii0 + is * HS_DNX + ks * HS_DNZ);
                        const T uc_yz = load_value<T>(uc, iii0 + js * HS_DNY + ks * HS_DNZ);
                        const T uc_xy_neg = load_value<T>(uc, iii0 - is * HS_DNX - js * HS_DNY);
                        const T uc_xz_neg = load_value<T>(uc, iii0 - is * HS_DNX - ks * HS_DNZ);
                        const T uc_yz_neg = load_value<T>(uc, iii0 - js * HS_DNY - ks * HS_DNZ);

                        const auto s_xy = limiter_wrapper(uc_xy - u0, u0 - uc_xy_neg);
                        const auto s_xz = limiter_wrapper(uc_xz - u0, u0 - uc_xz_neg);
                        const auto s_yz = limiter_wrapper(uc_yz - u0, u0 - uc_yz_neg);

                        const T uc_xyz =
                            load_value<T>(uc, iii0 + is * HS_DNX + js * HS_DNY + ks * HS_DNZ);
                        const T uc_xyz_neg =
                            load_value<T>(uc, iii0 - is * HS_DNX - js * HS_DNY - ks * HS_DNZ);
                        const auto s_xyz = limiter_wrapper(uc_xyz - u0, u0 - uc_xyz_neg);
                        T uf = u0;
                        uf += (9.0 / 64.0) * (s_x + s_y + s_z);
                        uf += (3.0 / 64.0) * (s_xy + s_yz + s_xz);
                        uf += (1.0 / 64.0) * s_xyz;
                        
                        uf_local[f * 8 + oct_index] = uf;
                        select_wrapper<T>(uf_local[f * 8 + oct_index], mask, uf, T(0));
                    }
                }
                if (!energy_only) {
                    const int oct_index = ir * 4 + jr * 2 + kr;
                    const auto i1 = 2 * i0 - H_BW + ir;
                    const auto j1 = 2 * j0 - H_BW + jr;
                    const auto k1 = 2 * (k0 + k) - H_BW + kr;
                    const auto x = (i1) *dx + xmin[XDIM];
                    const auto y = (j1) *dx + xmin[YDIM];
                    const auto z = (k1) *dx + xmin[ZDIM];

                    uf_local[lx_i * 8 + oct_index] -= y * uf_local[sz_i * 8 + oct_index] - z * uf_local[sy_i * 8 + oct_index];
                    uf_local[ly_i * 8 + oct_index] += x * uf_local[sz_i * 8 + oct_index] - z * uf_local[sx_i * 8 + oct_index];
                    uf_local[lz_i * 8 + oct_index] -= x * uf_local[sy_i * 8 + oct_index] - y * uf_local[sx_i * 8 + oct_index];
                }
            }
        }
    }
    if (!energy_only) {
        T zx = 0, zy = 0, zz = 0, rho = 0;
        for (int ir = 0; ir < 2; ir++) {
            for (int jr = 0; jr < 2; jr++) {
                #pragma unroll
                for (int kr = 0; kr < 2; kr++) {
                    const int oct_index = ir * 4 + jr * 2 + kr;
                    zx += uf_local[lx_i * 8 + oct_index] / 8.0;
                    zy += uf_local[ly_i * 8 + oct_index] / 8.0;
                    zz += uf_local[lz_i * 8 + oct_index] / 8.0;
                }
            }
        }
        for (int ir = 0; ir < 2; ir++) {
            for (int jr = 0; jr < 2; jr++) {
                #pragma unroll
                for (int kr = 0; kr < 2; kr++) {
                    const auto factor = 1.0;
                    const int oct_index = ir * 4 + jr * 2 + kr;
                    zx *= factor;
                    zy *= factor;
                    zz *= factor;
                    uf_local[lx_i * 8 + oct_index] = zx;
                    uf_local[ly_i * 8 + oct_index] = zy;
                    uf_local[lz_i * 8 + oct_index] = zz;
                }
            }
        }
        for (int ir = 0; ir < 2; ir++) {
            for (int jr = 0; jr < 2; jr++) {
                #pragma unroll
                for (int kr = 0; kr < 2; kr++) {
                    const int oct_index = ir * 4 + jr * 2 + kr;
                    const auto i1 = 2 * i0 - H_BW + ir;
                    const auto j1 = 2 * j0 - H_BW + jr;
                    const auto k1 = 2 * (k0 + k) - H_BW + kr;
                    const auto x = (i1) *dx + xmin[XDIM];
                    const auto y = (j1) *dx + xmin[YDIM];
                    const auto z = (k1) *dx + xmin[ZDIM];
                    uf_local[lx_i * 8 + oct_index] += y * uf_local[sz_i * 8 + oct_index] - z * uf_local[sy_i * 8 + oct_index];
                    uf_local[ly_i * 8 + oct_index] -= x * uf_local[sz_i * 8 + oct_index] - z * uf_local[sx_i * 8 + oct_index];
                    uf_local[lz_i * 8 + oct_index] += x * uf_local[sy_i * 8 + oct_index] - y * uf_local[sx_i * 8 + oct_index];

                }
            }
        }
    }
}
