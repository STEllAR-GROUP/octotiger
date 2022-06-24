//  Copyright (c) 2020-2022 Gregor Daiß
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//#include <__clang_cuda_builtin_vars.h>
#ifdef OCTOTIGER_HAVE_CUDA
#include "octotiger/util/vec_scalar_device_wrapper.hpp"
#include "octotiger/unitiger/hydro_impl/hydro_boundary_exchange.hpp"



__global__ void
__launch_bounds__((HS_NX - 2) * (HS_NX - 2), 1)
complete_hydro_amr_cuda_kernel(const double *dx_global, const int *energy_only_global,
    double* unified_ushad_global, int* coarse_global,
    double* xmin_global, double* unified_uf_global,
    const int nfields) {
    constexpr int max_nf = OCTOTIGER_MAX_NUMBER_FIELDS;
    const int field_offset = HS_N3 * 8;
    double uf_local[max_nf * 8];
    const int iii0 = (blockIdx.z + 1) * HS_DNX + (threadIdx.y + 1) * HS_DNY + (threadIdx.z + 1) * HS_DNZ;

    // Map arrays to slices
    const int slice_id = blockIdx.x;
    const int * coarse = coarse_global + slice_id * HS_N3;
    const double * unified_ushad = unified_ushad_global + slice_id * nfields * HS_N3;
    double * unified_uf = unified_uf_global + slice_id * nfields * HS_N3 * 8;
    const double * xmin = xmin_global + slice_id * NDIM;
    const bool energy_only = energy_only_global[slice_id];
    const double dx  = dx_global[slice_id];

    if (coarse[iii0]) {
        complete_hydro_amr_boundary_inner_loop<double>(dx, energy_only, unified_ushad, coarse, xmin, 
             blockIdx.z + 1, threadIdx.y + 1, threadIdx.z + 1, nfields, true, 0, iii0, uf_local);
        // TODO Remove uf once we can keep U on the device until the reconstruct kernel is called
        // Until then this kernel makes things slower since we have to copy either U or Uf back to the host just
        // to do some operations and copy it back to the device for reconstruct
        for (int f = 0; f < nfields; f++) {
            for (int ir = 0; ir < 2; ir++) {
                for (int jr = 0; jr < 2; jr++) {
                    #pragma unroll
                    for (int kr = 0; kr < 2; kr++) {
                       const int oct_index = ir * 4 + jr * 2 + kr;
                       unified_uf[f * field_offset + iii0 + oct_index * HS_N3] = uf_local[f * 8 + oct_index];
                    }
                }
            }
        }
    }
}
__global__ void
__launch_bounds__((H_NX) * (H_NX), 1)
complete_hydro_amr_cuda_kernel_phase2(
    const int* coarse_global,
    const double* unified_uf_global,
    double* unified_u_global,
    const int nfields, const bool energy_only) {
    const int field_offset = HS_N3 * 8;

    int i = blockIdx.z;
    int j = threadIdx.y;
    int k = threadIdx.z;
    int f = blockIdx.y;
    if (!energy_only || f == egas_i) {
      const int slice_id = blockIdx.x;

      // Map arrays to slices
      const int * coarse = coarse_global + slice_id * HS_N3;
      const double * unified_uf = unified_uf_global + slice_id * nfields * HS_N3 * 8;
      double * unified_u = unified_u_global + slice_id * nfields * H_N3;


      const int i0 = (i + H_BW) / 2;
      const int j0 = (j + H_BW) / 2;
      const int k0 = (k + H_BW) / 2;
      const int iii0 = hSindex(i0, j0, k0);
      const int iiir = hindex(i, j, k);

      if (coarse[iii0]) {
          int ir, jr, kr;
          if (H_BW % 2 == 0) {
              ir = i % 2;
              jr = j % 2;
              kr = k % 2;
          } else {
              ir = 1 - (i % 2);
              jr = 1 - (j % 2);
              kr = 1 - (k % 2);
          }
          const int oct_index = ir * 4 + jr * 2 + kr;
          unified_u[f * H_N3 + iiir] = unified_uf[f * field_offset + iii0 + oct_index * HS_N3];
      }
    }
}

void launch_complete_hydro_amr_boundary_cuda_post(
    aggregated_executor_t& executor,
    dim3 const grid_spec, dim3 const threads_per_block, void *args[]) {
    executor.post(
    cudaLaunchKernel<decltype(complete_hydro_amr_cuda_kernel)>,
    complete_hydro_amr_cuda_kernel, grid_spec, threads_per_block, args, 0);
}
void launch_complete_hydro_amr_boundary_cuda_phase2_post(
    aggregated_executor_t& executor,
    dim3 const grid_spec, dim3 const threads_per_block, void *args[]) {
    executor.post(
    cudaLaunchKernel<decltype(complete_hydro_amr_cuda_kernel_phase2)>,
    complete_hydro_amr_cuda_kernel_phase2, grid_spec, threads_per_block, args, 0);
}

#endif
