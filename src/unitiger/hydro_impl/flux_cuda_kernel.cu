#ifdef OCTOTIGER_HAVE_CUDA

#include <buffer_manager.hpp>
#include <cuda_buffer_util.hpp>
#include "octotiger/options.hpp"
#include "octotiger/cuda_util/cuda_helper.hpp"
#include <cuda_runtime.h>
#include <stream_manager.hpp>

#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp"

#include <mutex>

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

std::once_flag flag1;

__host__ void init_gpu_masks(bool *masks) {
  auto masks_boost = create_masks();
  cudaMemcpy(masks, masks_boost.data(), NDIM * 1000 * sizeof(bool), cudaMemcpyHostToDevice);
}

__host__ const bool* get_gpu_masks(void) {
    static recycler::cuda_device_buffer<bool> masks(NDIM * 1000, 0);
    std::call_once(flag1, init_gpu_masks, masks.device_side_buffer);
    return masks.device_side_buffer;
}

__device__ const int offset = 0;
__device__ const int compressedH_DN[3] = {100, 10, 1};
__device__ const int face_offset = 27 * 1000;
__device__ const int dim_offset = 1000;

__global__ void
__launch_bounds__(480, 1)
 flux_cuda_kernel(const double * __restrict__ q_combined, const double * __restrict__ x_combined, double * __restrict__ f_combined,
    double * __restrict__ amax, int * __restrict__ amax_indices, const bool * __restrict__ masks, const double omega, const double dx, const double A_, const double B_, const double fgamma, const double de_switch_1) {

  // 3 dim 1000 i workitems
  const size_t dim = blockIdx.y;
  const size_t index = blockIdx.z * 128 + threadIdx.z;
  const size_t nf = 15;
  if (index > 110 && index < 1000 && masks[dim * 1000 + index]) {
    double local_f[15] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double local_x[3] = {0.0, 0.0, 0.0};
    double local_vg[3] = {0.0, 0.0, 0.0};
    double local_amax = 0.0;
    int local_amax_index = 0;

    for (int fi = 0; fi < 9; fi++) {    // 9
      double this_ap = 0.0, this_am = 0.0;    // tmps
      const size_t d = faces[dim][fi];
      const size_t flipped_dim = flip_dim(d, dim);
      for (int dim = 0; dim < 3; dim++) {
          local_x[dim] = x_combined[dim * 1000 + index] + (0.5 * xloc[d][dim] * dx);
      }
      local_vg[0] = -omega * (x_combined[1000 + index] + 0.5 * xloc[d][1] * dx);
      local_vg[1] = +omega * (x_combined[index] + 0.5 * xloc[d][0] * dx);
      local_vg[2] = 0.0;
      double this_amax = inner_flux_loop2<double>(omega, nf, A_, B_, q_combined, local_f, local_x, local_vg,
        this_ap, this_am, dim, d, dx, fgamma, de_switch_1, dim_offset * d + index, dim_offset * flipped_dim - compressedH_DN[dim] + index, face_offset);
      for (int f = 0; f < nf; f++) {
        f_combined[dim * 15 * 1000 + f * 1000 + index] = local_f[f];
      }
    }

    // TODO Barrier

    // TODO Move max to the beginning of amax and amax_indidces
 }
 return;
}

timestep_t launch_flux_cuda(const hydro::recon_type<NDIM>& Q, hydro::flux_type& F, hydro::x_type& X,
    safe_real omega, const size_t nf_) {
    timestep_t ts;

    // Check availability
    bool avail = stream_pool::interface_available<hpx::cuda::experimental::cuda_executor,
                 pool_strategy>(opts().cuda_buffer_capacity);
  
    // Call CPU kernel as no stream is free
    if (!avail) {
       return flux_cpu_kernel(Q, F, X, omega, nf_);
    } else {

    size_t device_id =
      stream_pool::get_next_device_id<hpx::cuda::experimental::cuda_executor,
      pool_strategy>();

    stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy> executor;

    std::vector<double, recycler::recycle_allocator_cuda_host<double>> combined_q(
        15 * 27 * 10 * 10 * 10 + 32);
    auto it = combined_q.begin();
    for (auto face = 0; face < 15; face++) {
        for (auto d = 0; d < 27; d++) {
            auto start_offset = 2 * 14 * 14 + 2 * 14 + 2;
            for (auto ix = 2; ix < 2 + INX + 2; ix++) {
                for (auto iy = 2; iy < 2 + INX + 2; iy++) {
                    it = std::copy(Q[face][d].begin() + start_offset,
                        Q[face][d].begin() + start_offset + 10, it);
                    start_offset += 14;
                }
                start_offset += (2 + 2) * 14;
            }
        }
    }
    recycler::cuda_device_buffer<double> device_q(15 * 27 * 10 * 10 * 10 + 32, device_id);
    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
    cudaMemcpyAsync, device_q.device_side_buffer,
    combined_q.data(), (15 * 27 * 10 * 10 * 10 + 32) * sizeof(double), cudaMemcpyHostToDevice);

    std::vector<double, recycler::recycle_allocator_cuda_host<double>> combined_x(NDIM * 1000 + 32);
    auto it_x = combined_x.begin();
    for (size_t dim = 0; dim < NDIM; dim++) {
      auto start_offset = 2 * 14 * 14 + 2 * 14 + 2;
      for (auto ix = 2; ix < 2 + INX + 2; ix++) {
          for (auto iy = 2; iy < 2 + INX + 2; iy++) {
              it_x = std::copy(X[dim].begin() + start_offset,
                  X[dim].begin() + start_offset + 10, it_x);
              start_offset += 14;
          }
          start_offset += (2 + 2) * 14;
      }
    }
    // TODO make INX independ
    double dx = X[0][196] - X[0][0];
	  std::cerr << "Start moving x" << device_id << std::endl;
    recycler::cuda_device_buffer<double> device_x(NDIM * 1000 + 32, device_id);
    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
    cudaMemcpyAsync, device_x.device_side_buffer,
    combined_x.data(), (NDIM * 1000 + 32) * sizeof(double), cudaMemcpyHostToDevice);

	  std::cerr << "Create util stuff" << device_id << std::endl;
    std::vector<double, recycler::recycle_allocator_cuda_host<double>> combined_f(NDIM * 15 * 1000 + 32);
    recycler::cuda_device_buffer<double> device_f(NDIM * 15 * 1000 + 32, device_id);
    recycler::cuda_device_buffer<double> device_amax(NDIM * 1000);
    recycler::cuda_device_buffer<int> device_amax_indices(NDIM * 1000);
    const bool *masks = get_gpu_masks();
    double A_ = physics<NDIM>::A_;
    double B_ = physics<NDIM>::B_;
    double fgamma = physics<NDIM>::fgamma_;
    double de_switch_1 = physics<NDIM>::de_switch_1;

    // TODO Launch kernel
    dim3 const grid_spec(1, 3, 14);
    dim3 const threads_per_block(1, 1, 64);
    std::cerr << "started launching stuff" << std::endl;
    void* args[] = {&(device_q.device_side_buffer),
      &(device_x.device_side_buffer), &(device_f.device_side_buffer), &(device_amax.device_side_buffer),
      &(device_amax_indices.device_side_buffer), &masks, &omega, &dx, &A_, &B_, &fgamma, &de_switch_1};
    executor.post(
    cudaLaunchKernel<decltype(flux_cuda_kernel)>,
    flux_cuda_kernel, grid_spec, threads_per_block, args, 0);

    auto fut = hpx::async(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
               cudaMemcpyAsync, combined_f.data(), device_f.device_side_buffer,
               (NDIM * 15 * 1000 + 32) * sizeof(double), cudaMemcpyDeviceToHost);
    fut.get();
    // TODO Move amax from GPU
    }

    return ts;
}


#endif
