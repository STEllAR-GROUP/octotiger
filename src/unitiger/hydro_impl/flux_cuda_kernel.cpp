#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)

#if defined(OCTOTIGER_HAVE_HIP)
#define cudaLaunchKernel hipLaunchKernel
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaError_t hipError_t
#endif

#include <hpx/modules/async_cuda.hpp>
#include "octotiger/cuda_util/cuda_helper.hpp"
#include "octotiger/options.hpp"

#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/flux_kernel_templates.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp"    // required for xloc definition

#include <buffer_manager.hpp>
#if defined(OCTOTIGER_HAVE_CUDA)
#include <cuda_buffer_util.hpp>
#include <cuda_runtime.h>
#elif defined(OCTOTIGER_HAVE_HIP)
#include <hip/hip_runtime.h>
#include <hip_buffer_util.hpp>
#endif
#include <stream_manager.hpp>

#include "octotiger/common_kernel/kokkos_simd.hpp"
using simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::scalar>;
using simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::scalar>;

__global__ void __launch_bounds__(128, 2) flux_cuda_kernel(const double* __restrict__ q_combined,
    const double* __restrict__ x_combined, double* __restrict__ f_combined, double* amax,
    int* amax_indices, int* amax_d, const bool* __restrict__ masks, const double omega,
    const double *dx, const double A_, const double B_, const int nf, const double fgamma,
    const double de_switch_1, const int number_blocks) {
    __shared__ double sm_amax[128];
    __shared__ int sm_d[128];
    __shared__ int sm_i[128];

    const int slice_id = blockIdx.x;
    const int q_slice_offset = (nf * 27 * q_inx3 + 128) * slice_id;
    const int f_slice_offset = (NDIM * nf * q_inx3 + 128) * slice_id;
    const int x_slice_offset = (NDIM * q_inx3 + 128) * slice_id;
    const int amax_slice_offset = NDIM * (1 + 2 * nf) * number_blocks * slice_id;
    const int max_indices_slice_offset = NDIM * number_blocks * slice_id;
    // 3 dim 1000 i workitems
    const int number_dims = gridDim.z;
    const int blocks_per_dim = gridDim.y;
    const int dim = blockIdx.z;
    const int tid = threadIdx.x * 64 + threadIdx.y * 8 + threadIdx.z;
    const int index = blockIdx.y * 128 + tid;    // + 104;

    // Set during cmake step with -DOCTOTIGER_WITH_MAX_NUMBER_FIELDS
    std::array<simd_t, OCTOTIGER_MAX_NUMBER_FIELDS> local_f;
    std::array<simd_t, OCTOTIGER_MAX_NUMBER_FIELDS> local_q;
    std::array<simd_t, OCTOTIGER_MAX_NUMBER_FIELDS> local_q_flipped;
    // assumes maximal number (given by cmake) of species in a simulation.
    // Not the most elegant solution and rather old-fashion but one that works.
    // May be changed to a more flexible sophisticated object.
    for (int f = 0; f < nf; f++) {
        local_f[f] = 0.0;
    }
    std::array<simd_t, NDIM> local_x; 
    std::array<simd_t, NDIM> local_vg; 
    for (int dim = 0; dim < NDIM; dim++) {
        local_x[dim] = simd_t(0.0);
        local_vg[dim] = simd_t(0.0);
    }

    double current_amax = 0.0;
    int current_d = 0;
    int current_i = index;

    for (int f = 0; f < nf; f++) {
        for (int i = 0; i < simd_t::size(); i++) {
            f_combined[dim * nf * q_inx3 + f * q_inx3 + index + i + f_slice_offset] = 0.0;
        }
    }
    if (index + simd_t::size() > q_inx * q_inx + q_inx && index < q_inx3) {
        // Workaround to set the mask - usually I'd like to set it
        // component-wise but kokkos-simd currently does not support this!
        // hence the mask_helpers
        const simd_t mask_helper1(1.0);
        std::array<double, simd_t::size()> mask_helper2_array;
        // TODO make masks double and load directly
        for (int i = 0; i < simd_t::size(); i++) {
            mask_helper2_array[i] = masks[index + dim * dim_offset + i];
        }
        const simd_t mask_helper2(
            mask_helper2_array.data(), SIMD_NAMESPACE::element_aligned_tag{});
        const simd_mask_t mask = mask_helper1 == mask_helper2;
        if (SIMD_NAMESPACE::any_of(mask)) {
            for (int fi = 0; fi < 9; fi++) {            // 9
                simd_t this_ap = 0.0, this_am = 0.0;    // tmps
                const int d = faces[dim][fi];
                const int flipped_dim = flip_dim(d, dim);
                for (int dim = 0; dim < 3; dim++) {
                    local_x[dim] = simd_t(x_combined + dim * q_inx3 + index + x_slice_offset,
                                       SIMD_NAMESPACE::element_aligned_tag{}) +
                        simd_t(0.5 * xloc[d][dim] * dx[slice_id]);
                }
                local_vg[0] = -omega *
                    (simd_t(x_combined + q_inx3 + index + x_slice_offset,
                         SIMD_NAMESPACE::element_aligned_tag{}) +
                        simd_t(0.5 * xloc[d][1] * dx[slice_id]));
                local_vg[1] = +omega *
                    (simd_t(x_combined + index + x_slice_offset,
                         SIMD_NAMESPACE::element_aligned_tag{}) +
                        simd_t(0.5 * xloc[d][0] * dx[slice_id]));
                local_vg[2] = 0.0;
                const double* q_with_offset = q_combined + q_slice_offset;
                for (int f = 0; f < nf; f++) {
                    local_q[f].copy_from(q_with_offset + f * face_offset +
                            dim_offset * d + index,
                        SIMD_NAMESPACE::element_aligned_tag{});
                    local_q_flipped[f].copy_from(q_with_offset +
                            f * face_offset + dim_offset * flipped_dim -
                            compressedH_DN[dim] + index,
                        SIMD_NAMESPACE::element_aligned_tag{});
                    // Results get masked, no need to mask the input:
                    /* local_q[f] = SIMD_NAMESPACE::choose(mask, local_q[f], simd_t(1.0)); */
                    /* local_q_flipped[f] = SIMD_NAMESPACE::choose(mask, local_q_flipped[f], */
                    /*     simd_t(1.0)); */
                }
                cell_inner_flux_loop_simd<simd_t>(omega, nf, A_, B_, local_q, local_q_flipped,
                    local_f, local_x, local_vg, this_ap, this_am, dim, d, dx[slice_id],
                    fgamma, de_switch_1,
                    face_offset);
                this_ap = SIMD_NAMESPACE::choose(mask, this_ap, simd_t(0.0));
                this_am = SIMD_NAMESPACE::choose(mask, this_am, simd_t(0.0));
                const simd_t amax_tmp = SIMD_NAMESPACE::max(this_ap, (-this_am));
                // Reduce
                // TODO Reduce outside of inner loop?
                std::array<double, simd_t::size()> max_helper;
                amax_tmp.copy_to(max_helper.data(), SIMD_NAMESPACE::element_aligned_tag{});
                for (int i = 0; i < simd_t::size(); i++) {
                  if (max_helper[i] > current_amax) {
                      current_amax = max_helper[i];
                      current_d = d;
                      current_i = index + i;
                  }
                }
                for (int f = 1; f < nf; f++) {
                    simd_t current_val(
                        f_combined + dim * nf * q_inx3 + f * q_inx3 + index + f_slice_offset,
                        SIMD_NAMESPACE::element_aligned_tag{});
                    current_val = current_val +
                      SIMD_NAMESPACE::choose(mask, quad_weights[fi] * local_f[f],
                          simd_t(0.0));
                    current_val.copy_to(f_combined + dim
                      * nf * q_inx3 + f * q_inx3 + index + f_slice_offset,
                      SIMD_NAMESPACE::element_aligned_tag{});
                    /* f_combined[dim * nf * q_inx3 + f * q_inx3 + index + f_slice_offset] += */
                    /*     quad_weights[fi] * local_f[f]; */
                }
            }
        }
        simd_t current_val(
          f_combined + dim * nf * q_inx3 + index + f_slice_offset,
          SIMD_NAMESPACE::element_aligned_tag{});
        for (int f = 10; f < nf; f++) {
            simd_t current_field_val(
                f_combined + dim * nf * q_inx3 + f * q_inx3 + index + f_slice_offset,
                SIMD_NAMESPACE::element_aligned_tag{});
            current_val = current_val +
              SIMD_NAMESPACE::choose(mask, current_field_val,
                  simd_t(0.0));
        }
        current_val.copy_to(
          f_combined + dim * nf * q_inx3 + index + f_slice_offset,
          SIMD_NAMESPACE::element_aligned_tag{});
    }
    // Find maximum:
    sm_amax[tid] = current_amax;
    sm_d[tid] = current_d;
    sm_i[tid] = current_i;
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
        amax[block_id + amax_slice_offset] = sm_amax[0];
        amax_indices[block_id + max_indices_slice_offset] = sm_i[0];
        amax_d[block_id + max_indices_slice_offset] = sm_d[0];

        // Save face to the end of the amax buffer
        const int flipped_dim = flip_dim(sm_d[0], dim);
        for (int f = 0; f < nf; f++) {
            amax[blocks_per_dim * number_dims + block_id * 2 * nf + f + amax_slice_offset] =
                q_combined[sm_i[0] + f * face_offset + dim_offset * sm_d[0] + q_slice_offset];
            amax[blocks_per_dim * number_dims + block_id * 2 * nf + nf + f + amax_slice_offset] =
                q_combined[sm_i[0] - compressedH_DN[dim] + f * face_offset +
                    dim_offset * flipped_dim + q_slice_offset];
        }
    }
    return;
}

#if defined(OCTOTIGER_HAVE_HIP)
void flux_hip_kernel_ggl_wrapper(dim3 const grid_spec, dim3 const threads_per_block,
    double* device_q, double* device_x, double* device_f, double* device_amax,
    int* device_amax_indices, int* device_amax_d, const bool* masks, const double omega,
    const double *dx, const double A_, const double B_, const size_t nf_, const double fgamma,
    const double de_switch_1, const int number_blocks, cudaStream_t &stream) {
    hipLaunchKernelGGL(flux_cuda_kernel, grid_spec, threads_per_block, 0,
        stream, device_q, device_x, device_f, device_amax, device_amax_indices,
        device_amax_d, masks, omega, dx, A_, B_, nf_, fgamma, de_switch_1,
        number_blocks);
}

void launch_flux_hip_kernel_post(
    aggregated_executor_t& executor,
    dim3 const grid_spec, dim3 const threads_per_block, double* device_q, double* device_x,
    double* device_f, double* device_amax, int* device_amax_indices, int* device_amax_d,
    const bool* masks, const double omega, const double *dx, const double A_, const double B_,
    const size_t nf_, const double fgamma, const double de_switch_1, const int number_blocks) {
    executor.post(flux_hip_kernel_ggl_wrapper, grid_spec, threads_per_block,
        device_q, device_x, device_f, device_amax, device_amax_indices,
        device_amax_d, masks, omega, dx, A_, B_, nf_, fgamma, de_switch_1,
        number_blocks);
}
#else
void launch_flux_cuda_kernel_post(
    aggregated_executor_t& executor,
    dim3 const grid_spec, dim3 const threads_per_block, void* args[]) {
    executor.post(cudaLaunchKernel<decltype(flux_cuda_kernel)>, flux_cuda_kernel, grid_spec,
        threads_per_block, args, 0);
}
#endif

#endif
