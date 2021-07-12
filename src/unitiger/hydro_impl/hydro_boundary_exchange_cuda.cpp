#ifdef OCTOTIGER_HAVE_CUDA
#include "octotiger/util/vec_scalar_device_wrapper.hpp"
#include "octotiger/unitiger/hydro_impl/hydro_boundary_exchange.hpp"



__global__ void
__launch_bounds__((HS_NX - 2) * (HS_NX - 2), 1)
complete_hydro_amr_cuda_kernel(const double dx, const bool energy_only,
    double* __restrict__ unified_ushad, int* __restrict__ coarse,
    double* __restrict__ xmin, double* __restrict__ unified_uf,
    const int nfields) {
    constexpr int max_nf = 15;
    const int field_offset = HS_N3 * 8;
    double uf_local[max_nf * 8];
    const int iii0 = (blockIdx.z + 1) * HS_DNX + (threadIdx.y + 1) * HS_DNY + (threadIdx.z + 1) * HS_DNZ;
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

void launch_complete_hydro_amr_boundary_cuda_post(
    stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
    dim3 const grid_spec, dim3 const threads_per_block, void *args[]) {
    executor.post(
    cudaLaunchKernel<decltype(complete_hydro_amr_cuda_kernel)>,
    complete_hydro_amr_cuda_kernel, grid_spec, threads_per_block, args, 0);
}

#endif
