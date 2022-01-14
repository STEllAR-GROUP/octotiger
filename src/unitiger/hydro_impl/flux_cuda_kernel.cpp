#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)

#if defined(OCTOTIGER_HAVE_HIP)
#define cudaLaunchKernel hipLaunchKernel
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaError_t hipError_t
#endif

#include <buffer_manager.hpp>
#include <hpx/modules/async_cuda.hpp>
#if defined(OCTOTIGER_HAVE_CUDA)
#include <cuda_buffer_util.hpp>
#include <cuda_runtime.h>
#elif defined(OCTOTIGER_HAVE_HIP)
#include <hip/hip_runtime.h>
#include <hip_buffer_util.hpp>
#endif
#include <stream_manager.hpp>
#include "octotiger/cuda_util/cuda_helper.hpp"
#include "octotiger/options.hpp"

#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/flux_kernel_templates.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp"    // required for xloc definition

__global__ void __launch_bounds__(128, 2)
    flux_cuda_kernel(const double* __restrict__ q_combined, const double* __restrict__ x_combined,
        double* __restrict__ f_combined, double* amax, int* amax_indices, int* amax_d,
        const bool* __restrict__ masks, const double omega, const double dx, const double A_,
        const double B_, const int nf, const double fgamma, const double de_switch_1) {
    __shared__ double sm_amax[128];
    __shared__ int sm_d[128];
    __shared__ int sm_i[128];

    // Set during cmake step with -DOCTOTIGER_WITH_MAX_NUMBER_FIELDS
    double local_f[OCTOTIGER_MAX_NUMBER_FIELDS];
    // assumes maximal number (given by cmake) of species in a simulation.
    // Not the most elegant solution and rather old-fashion but one that works.
    // May be changed to a more flexible sophisticated object.
    for (int f = 0; f < nf; f++) {
        local_f[f] = 0.0;
    }
    double local_x[3] = {0.0, 0.0, 0.0};
    double local_vg[3] = {0.0, 0.0, 0.0};

    double current_amax = 0.0;
    int current_d = 0;

    // 3 dim 1000 i workitems
    const int number_dims = gridDim.z;
    const int blocks_per_dim = gridDim.y;
    const int dim = blockIdx.z;
    const int tid = threadIdx.x * 64 + threadIdx.y * 8 + threadIdx.z;
    const int index = blockIdx.y * 128 + tid;    // + 104;
    for (int f = 0; f < nf; f++) {
        f_combined[dim * nf * q_inx3 + f * q_inx3 + index] = 0.0;
    }
    if (index > q_inx * q_inx + q_inx && index < q_inx3) {
        double mask = masks[index + dim * dim_offset];
        if (mask != 0.0) {
            for (int fi = 0; fi < 9; fi++) {            // 9
                double this_ap = 0.0, this_am = 0.0;    // tmps
                const int d = faces[dim][fi];
                const int flipped_dim = flip_dim(d, dim);
                for (int dim = 0; dim < 3; dim++) {
                    local_x[dim] = x_combined[dim * q_inx3 + index] + (0.5 * xloc[d][dim] * dx);
                }
                local_vg[0] = -omega * (x_combined[q_inx3 + index] + 0.5 * xloc[d][1] * dx);
                local_vg[1] = +omega * (x_combined[index] + 0.5 * xloc[d][0] * dx);
                local_vg[2] = 0.0;
                cell_inner_flux_loop<double>(omega, nf, A_, B_, q_combined, local_f, local_x,
                    local_vg, this_ap, this_am, dim, d, dx, fgamma, de_switch_1,
                    dim_offset * d + index, dim_offset * flipped_dim - compressedH_DN[dim] + index,
                    face_offset);
                this_ap *= mask;
                this_am *= mask;
                const double amax_tmp = max_wrapper(this_ap, (-this_am));
                if (amax_tmp > current_amax) {
                    current_amax = amax_tmp;
                    current_d = d;
                }
                for (int f = 1; f < nf; f++) {
                    f_combined[dim * nf * q_inx3 + f * q_inx3 + index] +=
                        quad_weights[fi] * local_f[f];
                }
            }
        }
        for (int f = 10; f < nf; f++) {
            f_combined[dim * nf * q_inx3 + index] +=
                f_combined[dim * nf * q_inx3 + f * q_inx3 + index];
        }
    }
    // Find maximum:
    sm_amax[tid] = current_amax;
    sm_d[tid] = current_d;
    sm_i[tid] = index;
    __syncthreads();
    // Max reduction with multiple warps
    /*for (int tid_border = 64; tid_border >= 32; tid_border /= 2) {
        if (tid < tid_border) {
            if (sm_amax[tid + tid_border] > sm_amax[tid]) {
                sm_amax[tid] = sm_amax[tid + tid_border];
                sm_d[tid] = sm_d[tid + tid_border];
                sm_i[tid] = sm_i[tid + tid_border];
            }
        }
        __syncthreads();
    }
    // Max reduction within one warps
    for (int tid_border = 16; tid_border >= 1; tid_border /= 2) {
        if (tid < tid_border) {
            if (sm_amax[tid + tid_border] > sm_amax[tid]) {
                sm_amax[tid] = sm_amax[tid + tid_border];
                sm_d[tid] = sm_d[tid + tid_border];
                sm_i[tid] = sm_i[tid + tid_border];
            }
        }
    }*/
    // Find maximum:
    sm_amax[tid] = current_amax;
    sm_d[tid] = current_d;
    sm_i[tid] = index;
    __syncthreads();
    // Max reduction with multiple warps
    for (int tid_border = 64; tid_border >= 32; tid_border /= 2) {
        if (tid < tid_border) {
            if (sm_amax[tid + tid_border] > sm_amax[tid]) {
                sm_amax[tid] = sm_amax[tid + tid_border];
                sm_d[tid] = sm_d[tid + tid_border];
                sm_i[tid] = sm_i[tid + tid_border];
            } else if (sm_amax[tid + tid_border] == sm_amax[tid]) {
                if (sm_i[tid + tid_border] < sm_i[tid]) {
                    sm_amax[tid] = sm_amax[tid + tid_border];
                    sm_d[tid] = sm_d[tid + tid_border];
                    sm_i[tid] = sm_i[tid + tid_border];
                }
            }
        }
        __syncthreads();
    }
    // Max reduction within one warps
    for (int tid_border = 16; tid_border >= 1; tid_border /= 2) {
        if (tid < tid_border) {
            if (sm_amax[tid + tid_border] > sm_amax[tid]) {
                sm_amax[tid] = sm_amax[tid + tid_border];
                sm_d[tid] = sm_d[tid + tid_border];
                sm_i[tid] = sm_i[tid + tid_border];
            } else if (sm_amax[tid + tid_border] == sm_amax[tid]) {
                if (sm_i[tid + tid_border] < sm_i[tid]) {
                    sm_amax[tid] = sm_amax[tid + tid_border];
                    sm_d[tid] = sm_d[tid + tid_border];
                    sm_i[tid] = sm_i[tid + tid_border];
                }
            }
        }
    }

    if (tid == 0) {
        const int block_id = blockIdx.y + dim * blocks_per_dim;
        amax[block_id] = sm_amax[0];
        amax_indices[block_id] = sm_i[0];
        amax_d[block_id] = sm_d[0];

        // Save face to the end of the amax buffer
        const int flipped_dim = flip_dim(sm_d[0], dim);
        for (int f = 0; f < nf; f++) {
            amax[blocks_per_dim * number_dims + block_id * 2 * nf + f] =
                q_combined[sm_i[0] + f * face_offset + dim_offset * sm_d[0]];
            amax[blocks_per_dim * number_dims + block_id * 2 * nf + nf + f] = q_combined[sm_i[0] -
                compressedH_DN[dim] + f * face_offset + dim_offset * flipped_dim];
        }
    }
    return;
}

#if defined(OCTOTIGER_HAVE_HIP)
void flux_hip_kernel_ggl_wrapper(dim3 const grid_spec, dim3 const threads_per_block,
    double* device_q, double* device_x, double* device_f, double* device_amax,
    int* device_amax_indices, int* device_amax_d, const bool* masks, const double omega,
    const double dx, const double A_, const double B_, const size_t nf_, const double fgamma,
    const double de_switch_1, cudaStream_t &stream) {
    hipLaunchKernelGGL(flux_cuda_kernel, grid_spec, threads_per_block, 0, stream, device_q,
       device_x, device_f, device_amax, device_amax_indices, device_amax_d, masks, omega, dx, A_, B_, nf_,
       fgamma, de_switch_1);
}

void launch_flux_hip_kernel_post(
    stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
    dim3 const grid_spec, dim3 const threads_per_block, double* device_q, double* device_x,
    double* device_f, double* device_amax, int* device_amax_indices, int* device_amax_d,
    const bool* masks, const double omega, const double dx, const double A_, const double B_,
    const size_t nf_, const double fgamma, const double de_switch_1) {
    executor.post(flux_hip_kernel_ggl_wrapper, grid_spec, threads_per_block, device_q, device_x,
        device_f, device_amax, device_amax_indices, device_amax_d, masks, omega, dx, A_, B_, nf_, fgamma, de_switch_1);
}
#else
void launch_flux_cuda_kernel_post(
    stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
    dim3 const grid_spec, dim3 const threads_per_block, void* args[]) {
    executor.post(cudaLaunchKernel<decltype(flux_cuda_kernel)>, flux_cuda_kernel, grid_spec,
        threads_per_block, args, 0);
}
#endif

#endif
