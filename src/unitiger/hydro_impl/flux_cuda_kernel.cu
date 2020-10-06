#ifdef OCTOTIGER_HAVE_CUDA

#include <buffer_manager.hpp>
#include <cuda_buffer_util.hpp>
#include "octotiger/options.hpp"
#include "octotiger/cuda_util/cuda_helper.hpp"
#include <cuda_runtime.h>
#include <stream_manager.hpp>

#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp"

#include <hpx/synchronization/once.hpp>

__device__ inline int flip_dim(const int d, const int flip_dim) {
		int dims[3];
		int k = d;
		for (int dim = 0; dim < 3; dim++) {
			dims[dim] = k % 3;
			k /= 3;
		}
		k = 0;
		dims[flip_dim] = 2 - dims[flip_dim];
		for (int dim = 0; dim < 3; dim++) {
			k *= 3;
			k += dims[2 - dim];
		}
		return k;
}

__device__ const int faces[3][9] = { { 12, 0, 3, 6, 9, 15, 18, 21, 24 }, { 10, 0, 1, 2, 9, 11,
			18, 19, 20 }, { 4, 0, 1, 2, 3, 5, 6, 7, 8 } };

__device__ const int xloc[27][3] = {
	/**/{ -1, -1, -1 }, { +0, -1, -1 }, { +1, -1, -1 },
	/**/{ -1, +0, -1 }, { +0, +0, -1 }, { 1, +0, -1 },
	/**/{ -1, +1, -1 }, { +0, +1, -1 }, { +1, +1, -1 },
	/**/{ -1, -1, +0 }, { +0, -1, +0 }, { +1, -1, +0 },
	/**/{ -1, +0, +0 }, { +0, +0, +0 }, { +1, +0, +0 },
	/**/{ -1, +1, +0 }, { +0, +1, +0 }, { +1, +1, +0 },
	/**/{ -1, -1, +1 }, { +0, -1, +1 }, { +1, -1, +1 },
	/**/{ -1, +0, +1 }, { +0, +0, +1 }, { +1, +0, +1 },
	/**/{ -1, +1, +1 }, { +0, +1, +1 }, { +1, +1, +1 } };

__device__ const double quad_weights[9] = { 16. / 36., 1. / 36., 4. / 36., 1. / 36., 4. / 36., 4.
			/ 36., 1. / 36., 4. / 36., 1. / 36. };

hpx::lcos::local::once_flag flag1;

__host__ void init_gpu_masks(bool *masks) {
  auto masks_boost = create_masks();
  cudaMemcpy(masks, masks_boost.data(), NDIM * 1000 * sizeof(bool), cudaMemcpyHostToDevice);
}

__host__ const bool* get_gpu_masks(void) {
    static bool *masks = recycler::recycle_allocator_cuda_device<bool>{}.allocate(NDIM * 1000);
    hpx::lcos::local::call_once(flag1, init_gpu_masks, masks);
    return masks;
}

__device__ const int offset = 0;
__device__ const int compressedH_DN[3] = {100, 10, 1};
__device__ const int face_offset = 27 * 1000;
__device__ const int dim_offset = 1000;

__global__ void
__launch_bounds__(128, 2)
 flux_cuda_kernel(const double * __restrict__ q_combined, const double * __restrict__ x_combined, double * __restrict__ f_combined,
    double * amax, int * amax_indices, int * amax_d, const bool * __restrict__ masks, const double omega, const double dx, const double A_, const double B_, const double fgamma, const double de_switch_1) {
  __shared__ double sm_amax[128];
  __shared__ int sm_d[128];
  __shared__ int sm_i[128];

  // 3 dim 1000 i workitems
  //const int index = threadIdx.x * 100 + threadIdx.y * 10 + threadIdx.z + 100;

  const int dim = blockIdx.z;
  const int index = (blockIdx.y * 2 + threadIdx.x) * 64 + threadIdx.y * 8 + threadIdx.z + 104;
  const int tid = threadIdx.x * 64 + threadIdx.y * 8 + threadIdx.z;   
  const double mask = masks[index + dim * dim_offset];
  //if(tid == 0)
  // printf("starting...");
  const int nf = 15;

  double local_f[15] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double local_x[3] = {0.0, 0.0, 0.0};
  double local_vg[3] = {0.0, 0.0, 0.0};
  for (int f = 0; f < nf; f++) {
      f_combined[dim * 15 * 1000 + f * 1000 + index] = 0.0;
  }

  double current_amax = 0.0;
  int current_d = 0;
  if(mask != 0.0) {
    for (int fi = 0; fi < 9; fi++) {    // 9
      double this_ap = 0.0, this_am = 0.0;    // tmps
      const int d = faces[dim][fi];
      const int flipped_dim = flip_dim(d, dim);
      for (int dim = 0; dim < 3; dim++) {
          local_x[dim] = x_combined[dim * 1000 + index] + (0.5 * xloc[d][dim] * dx);
      }
      local_vg[0] = -omega * (x_combined[1000 + index] + 0.5 * xloc[d][1] * dx);
      local_vg[1] = +omega * (x_combined[index] + 0.5 * xloc[d][0] * dx);
      local_vg[2] = 0.0;
      inner_flux_loop2<double>(omega, nf, A_, B_, q_combined, local_f, local_x, local_vg,
        this_ap, this_am, dim, d, dx, fgamma, de_switch_1, dim_offset * d + index, dim_offset * flipped_dim - compressedH_DN[dim] + index, face_offset);
      this_ap *= mask;
      this_am *= mask;
      const double amax_tmp = max_wrapper(this_ap, (-this_am));
      if (amax_tmp > current_amax) {
        current_amax = amax_tmp;
        current_d = d;
      }
      for (int f = 1; f < nf; f++) {
        f_combined[dim * 15 * 1000 + f * 1000 + index] += quad_weights[fi] * local_f[f];
        if (f > 9) {
          f_combined[dim * 15 * 1000 + index] += quad_weights[fi] * local_f[f];
        }
      }
   }
 }

 // Find maximum:
 sm_amax[tid] = current_amax;
 sm_d[tid] = current_d;
 sm_i[tid] = index;
 __syncthreads();
 for (int tid_border = 64; tid_border >= 32; tid_border /= 2) {
   if(tid < tid_border) {
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
   if(tid < tid_border) {
     if (sm_amax[tid + tid_border] > sm_amax[tid]) {
       sm_amax[tid] = sm_amax[tid + tid_border];
       sm_d[tid] = sm_d[tid + tid_border];
       sm_i[tid] = sm_i[tid + tid_border];
     }
   }
 }

 if (tid == 0) {
   amax[dim * 7 + blockIdx.y] = sm_amax[0];
   amax_indices[dim * 7 + blockIdx.y] = sm_i[0];
   amax_d[dim * 7 + blockIdx.y] = sm_d[0];
 }
 return;
}

timestep_t launch_flux_cuda(stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
    std::vector<double, recycler::recycle_allocator_cuda_host<double>> &combined_q, double* device_q,
    std::vector<double, recycler::recycle_allocator_cuda_host<double>> &combined_f,
    std::vector<double, recycler::recycle_allocator_cuda_host<double>> &combined_x, double* device_x,
    safe_real omega, const size_t nf_, double dx, size_t device_id) {
    timestep_t ts;
    const cell_geometry<3, 8> geo;


    /*size_t device_id =
      stream_pool::get_next_device_id<hpx::cuda::experimental::cuda_executor,
      pool_strategy>();*/
    //size_t device_id = 0;

    recycler::cuda_device_buffer<double> device_f(NDIM * 15 * 1000 + 32, device_id);
    const bool *masks = get_gpu_masks();

    recycler::cuda_device_buffer<double> device_amax(7 * NDIM);
    recycler::cuda_device_buffer<int> device_amax_indices(7 * NDIM);
    recycler::cuda_device_buffer<int> device_amax_d(7 * NDIM);
    double A_ = physics<NDIM>::A_;
    double B_ = physics<NDIM>::B_;
    double fgamma = physics<NDIM>::fgamma_;
    double de_switch_1 = physics<NDIM>::de_switch_1;

    dim3 const grid_spec(1, 7, 3);
    dim3 const threads_per_block(2, 8, 8);
    void* args[] = {&(device_q),
      &(device_x), &(device_f.device_side_buffer), &(device_amax.device_side_buffer),
      &(device_amax_indices.device_side_buffer), &(device_amax_d.device_side_buffer), &masks, &omega, &dx, &A_, &B_, &fgamma, &de_switch_1};
    executor.post(
    cudaLaunchKernel<decltype(flux_cuda_kernel)>,
    flux_cuda_kernel, grid_spec, threads_per_block, args, 0);

    // Move data to host
    std::vector<double, recycler::recycle_allocator_cuda_host<double>> amax(7 * NDIM);
    std::vector<int, recycler::recycle_allocator_cuda_host<int>> amax_indices(7 * NDIM);
    std::vector<int, recycler::recycle_allocator_cuda_host<int>> amax_d(7 * NDIM);
    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
               cudaMemcpyAsync, amax.data(),
               device_amax.device_side_buffer, 7 * NDIM * sizeof(double),
               cudaMemcpyDeviceToHost);
    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
               cudaMemcpyAsync, amax_indices.data(),
               device_amax_indices.device_side_buffer, 7 * NDIM * sizeof(int),
               cudaMemcpyDeviceToHost);
    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
               cudaMemcpyAsync, amax_d.data(),
               device_amax_d.device_side_buffer, 7 * NDIM * sizeof(int),
               cudaMemcpyDeviceToHost);
    auto fut = hpx::async(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
               cudaMemcpyAsync, combined_f.data(), device_f.device_side_buffer,
               (NDIM * 15 * 1000 + 32) * sizeof(double), cudaMemcpyDeviceToHost);
    fut.get();

    // Find Maximum
    size_t current_id = 0;
    for (size_t i = 1; i < NDIM * 7; i++) {
      if (amax[i] > amax[current_id]) { 
        current_id = i;
      }
    }
    const size_t current_dim = current_id / 7;
    const size_t current_max_index = amax_indices[current_id];
    const size_t current_d = amax_d[current_id];
    //std::cin.get();
    std::vector<double> URs(nf_), ULs(nf_);
    ts.a = amax[current_dim];
    ts.x = combined_x[current_max_index];
    ts.y = combined_x[current_max_index + 1000];
    ts.z = combined_x[current_max_index + 2000];
    const auto flipped_dim = geo.flip_dim(current_d, current_dim);
    constexpr int compressedH_DN[3] = {100, 10, 1};
    for (int f = 0; f < nf_; f++) {
        URs[f] = combined_q[current_max_index + f * face_offset + dim_offset * current_d];
        ULs[f] = combined_q[current_max_index - compressedH_DN[current_dim] + f * face_offset +
            dim_offset * flipped_dim];
    }
    ts.ul = std::move(ULs);
    ts.ur = std::move(URs);
    ts.dim = current_dim;
    return ts;
}


#endif
