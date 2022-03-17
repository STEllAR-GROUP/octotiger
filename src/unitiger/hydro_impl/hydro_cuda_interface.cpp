#pragma once
#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
#include <buffer_manager.hpp>
#if defined(OCTOTIGER_HAVE_CUDA)
#include <cuda_buffer_util.hpp>
#include <cuda_runtime.h>
#elif defined(OCTOTIGER_HAVE_HIP)
#include <hip/hip_runtime.h>
#include <hip_buffer_util.hpp>
#endif
#include <stream_manager.hpp>
#undef NDEBUG
#include <aggregation_manager.hpp>

#include <hpx/apply.hpp>
#include <hpx/synchronization/once.hpp>

#include "octotiger/cuda_util/cuda_helper.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/options.hpp"

#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/flux_kernel_templates.hpp"    // required for fill_masks
#include "octotiger/unitiger/hydro_impl/hydro_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp"    // required for constants

static const char hydro_cuda_kernel_identifier[] = "hydro_kernel_aggregator_cuda";
using hydro_cuda_agg_executor_pool = aggregation_pool<hydro_cuda_kernel_identifier, hpx::cuda::experimental::cuda_executor,
                                       pool_strategy>;

hpx::lcos::local::once_flag flag1;
hpx::lcos::local::once_flag init_pool_flag;

#if defined(OCTOTIGER_HAVE_CUDA)
template <typename T>
using device_buffer_t = recycler::cuda_device_buffer<T>;
template <typename T, typename Alloc>
using aggregated_device_buffer_t = recycler::cuda_aggregated_device_buffer<T, Alloc>;
template <typename T>
using host_buffer_t = std::vector<T, recycler::recycle_allocator_cuda_host<T>>;
using executor_t = hpx::cuda::experimental::cuda_executor;
template <typename T, typename Alloc>
using aggregated_host_buffer_t = std::vector<T, Alloc>;
#elif defined(OCTOTIGER_HAVE_HIP)
template <typename T>
using device_buffer_t = recycler::hip_device_buffer<T>;
template <typename T>
using host_buffer_t = std::vector<T, recycler::recycle_allocator_hip_host<T>>;
using executor_t = hpx::cuda::experimental::cuda_executor;

#define cudaLaunchKernel hipLaunchKernel
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyAsync hipMemcpyAsync

#endif

void init_aggregation_pool(void) {
    constexpr size_t number_aggregation_executors = 1;
    constexpr size_t max_slices = 1;
    constexpr Aggregated_Executor_Modes executor_mode = Aggregated_Executor_Modes::EAGER;
    hydro_cuda_agg_executor_pool::init(number_aggregation_executors, max_slices, executor_mode);
}

__host__ void init_gpu_masks(bool* masks) {
    boost::container::vector<bool> masks_boost(NDIM * q_inx * q_inx * q_inx);
    fill_masks(masks_boost);
    cudaMemcpy(masks, masks_boost.data(), NDIM * q_inx3 * sizeof(bool), cudaMemcpyHostToDevice);
}

__host__ const bool* get_gpu_masks(void) {
#if defined(OCTOTIGER_HAVE_CUDA)
    static bool* masks = recycler::recycle_allocator_cuda_device<bool>{}.allocate(NDIM * q_inx3);
#elif defined(OCTOTIGER_HAVE_HIP)
    static bool* masks = recycler::recycle_allocator_hip_device<bool>{}.allocate(NDIM * q_inx3);
#endif
    hpx::lcos::local::call_once(flag1, init_gpu_masks, masks);
    return masks;
}


timestep_t launch_flux_cuda(
    stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
    double* device_q, host_buffer_t<double>& combined_f, host_buffer_t<double>& combined_x,
    double* device_x, safe_real omega, const size_t nf_, double dx, size_t device_id) {
    timestep_t ts;
    const cell_geometry<3, 8> geo;
    constexpr int number_blocks = (q_inx3 / 128 + 1);

      // TODO Add slice alloc
    device_buffer_t<double> device_f(NDIM * nf_ * q_inx3 + 128, device_id);
    const bool* masks = get_gpu_masks();
    device_buffer_t<double> device_amax(number_blocks * NDIM * (1 + 2 * nf_));
    device_buffer_t<int> device_amax_indices(number_blocks * NDIM);
    device_buffer_t<int> device_amax_d(number_blocks * NDIM);
    double A_ = physics<NDIM>::A_;
    double B_ = physics<NDIM>::B_;
    double fgamma = physics<NDIM>::fgamma_;
    double de_switch_1 = physics<NDIM>::de_switch_1;
    int nf_local = physics<NDIM>::nf_;

    assert(NDIM == 3);
    dim3 const grid_spec(1, number_blocks, 3);
    dim3 const threads_per_block(2, 8, 8);
#if defined(OCTOTIGER_HAVE_CUDA)
    void* args[] = {&(device_q), &(device_x), &(device_f.device_side_buffer),
        &(device_amax.device_side_buffer), &(device_amax_indices.device_side_buffer),
        &(device_amax_d.device_side_buffer), &masks, &omega, &dx, &A_, &B_, &nf_local, &fgamma,
        &de_switch_1};
      // TODO Wrap in slice executor
    launch_flux_cuda_kernel_post(executor, grid_spec, threads_per_block, args);
#elif defined(OCTOTIGER_HAVE_HIP)
      // TODO Wrap in slice executor
    launch_flux_hip_kernel_post(executor, grid_spec, threads_per_block, device_q, device_x,
        device_f.device_side_buffer, device_amax.device_side_buffer,
        device_amax_indices.device_side_buffer, device_amax_d.device_side_buffer, masks, omega, dx,
        A_, B_, nf_local, fgamma, de_switch_1);
#endif

    // Move data to host
      // TODO Add slice alloc
    host_buffer_t<double> amax(number_blocks * NDIM * (1 + 2 * nf_));
    host_buffer_t<int> amax_indices(number_blocks * NDIM);
    host_buffer_t<int> amax_d(number_blocks * NDIM);

      // TODO Wrap in slice executor
    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor), cudaMemcpyAsync,
        amax.data(), device_amax.device_side_buffer,
        (number_blocks * NDIM * (1 + 2 * nf_)) * sizeof(double), cudaMemcpyDeviceToHost);
    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor), cudaMemcpyAsync,
        amax_indices.data(), device_amax_indices.device_side_buffer,
        number_blocks * NDIM * sizeof(int), cudaMemcpyDeviceToHost);
    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor), cudaMemcpyAsync,
        amax_d.data(), device_amax_d.device_side_buffer, number_blocks * NDIM * sizeof(int),
        cudaMemcpyDeviceToHost);
    auto fut = hpx::async(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
        cudaMemcpyAsync, combined_f.data(), device_f.device_side_buffer,
        (NDIM * nf_ * q_inx3 + 128) * sizeof(double), cudaMemcpyDeviceToHost);
    fut.get();

    // Find Maximum
    size_t current_dim = 0;
    for (size_t dim_i = 1; dim_i < number_blocks * NDIM; dim_i++) {
      if (amax[dim_i] > amax[current_dim]) { 
        current_dim = dim_i;
      } else if (amax[dim_i] == amax[current_dim]) {
        if (amax_indices[dim_i] < amax_indices[current_dim]) {
          current_dim = dim_i;
        }
      }
    }
    std::vector<double> URs(nf_), ULs(nf_);
    const size_t current_max_index = amax_indices[current_dim];
    // const size_t current_d = amax_d[current_dim];
    ts.a = amax[current_dim];
    ts.x = combined_x[current_max_index];
    ts.y = combined_x[current_max_index + q_inx3];
    ts.z = combined_x[current_max_index + 2 * q_inx3];
    
    const size_t current_i = current_dim;
    current_dim = current_dim / number_blocks;
    for (int f = 0; f < nf_; f++) {
        URs[f] = amax[NDIM * number_blocks + current_i * 2 * nf_ + f];
        ULs[f] = amax[NDIM * number_blocks + current_i * 2 * nf_ + nf_ + f];
    }
    ts.ul = std::move(URs);
    ts.ur = std::move(ULs);
    ts.dim = current_dim;
    return ts;
}

// Input U, X, omega, executor, device_id
// Output F
timestep_t launch_hydro_cuda_kernels(const hydro_computer<NDIM, INX, physics<NDIM>>& hydro,
    const std::vector<std::vector<safe_real>>& U, const std::vector<std::vector<safe_real>>& X,
    const double omega, const size_t device_id,
    stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
    std::vector<hydro_state_t<std::vector<safe_real>>>& F) {

    // Init local kernel pool if not done already
    hpx::lcos::local::call_once(init_pool_flag, init_aggregation_pool);
    // Get executor (-slice if aggregated with max_slices > 1) future
    auto executor_slice_fut = hydro_cuda_agg_executor_pool::request_executor_slice();
    // Add continuation to executor future to execute the hydro kernels as soon as it's ready
    auto ret_fut = executor_slice_fut.value().then([&](auto && fut) {
      // Unwrap executor from ready future
      auto exec_slice = fut.get();
      // How many executor slices are working together and what's our ID?
      const size_t slice_id = exec_slice.id;
      const size_t number_slices = exec_slice.number_slices;

      // Get allocators of all the executors working together
      auto alloc_host_double =
          exec_slice
              .template make_allocator<double, recycler::detail::cuda_pinned_allocator<double>>();
      auto alloc_device_double =
          exec_slice
              .template make_allocator<double, recycler::detail::cuda_device_allocator<double>>();
      auto alloc_host_int =
          exec_slice
              .template make_allocator<int, recycler::detail::cuda_pinned_allocator<int>>();
      auto alloc_device_int =
          exec_slice
              .template make_allocator<int, recycler::detail::cuda_device_allocator<int>>();

      static const cell_geometry<NDIM, INX> geo;

      // Device buffers
      aggregated_device_buffer_t<double, decltype(alloc_device_double)>
        device_q(
          hydro.get_nf() * 27 * q_inx * q_inx * q_inx + 128, device_id, alloc_device_double);
      aggregated_device_buffer_t<double, decltype(alloc_device_double)> device_x(
          NDIM * q_inx3 + 128, device_id, alloc_device_double);
      aggregated_device_buffer_t<double, decltype(alloc_device_double)> device_large_x(
          NDIM * H_N3 + 128, device_id, alloc_device_double);
      aggregated_device_buffer_t<double, decltype(alloc_device_double)> device_f(
          NDIM * hydro.get_nf() * q_inx3 + 128, device_id, alloc_device_double);
      aggregated_device_buffer_t<double, decltype(alloc_device_double)> device_u(
          hydro.get_nf() * H_N3 + 128, device_id, alloc_device_double);
      aggregated_device_buffer_t<double, decltype(alloc_device_double)>
        device_unified_discs(
          geo.NDIR / 2 * H_N3 + 128, device_id, alloc_device_double);
      aggregated_device_buffer_t<double, decltype(alloc_device_double)> device_P(
          H_N3 + 128, device_id, alloc_device_double);
      aggregated_device_buffer_t<int, decltype(alloc_device_int)> device_disc_detect(
          hydro.get_nf(), device_id, alloc_device_int);
      aggregated_device_buffer_t<int, decltype(alloc_device_int)> device_smooth_field(
          hydro.get_nf(), device_id, alloc_device_int);
      aggregated_device_buffer_t<double, decltype(alloc_device_double)> device_AM(
          NDIM * q_inx * q_inx * q_inx + 128, device_id, alloc_device_double);

      // Host buffers
      aggregated_host_buffer_t<double, decltype(alloc_host_double)> combined_x(
          NDIM * q_inx3 + 128, double{}, alloc_host_double);
      aggregated_host_buffer_t<double, decltype(alloc_host_double)> combined_large_x(
          NDIM * H_N3 + 128, double{}, alloc_host_double);
      aggregated_host_buffer_t<double, decltype(alloc_host_double)> combined_u(
          hydro.get_nf() * H_N3 + 128, double{}, alloc_host_double);
      aggregated_host_buffer_t<double, decltype(alloc_host_double)> f(
          NDIM * hydro.get_nf() * q_inx3 + 128, double{}, alloc_host_double);
      aggregated_host_buffer_t<int, decltype(alloc_host_int)> disc_detect(
          hydro.get_nf(), int{}, alloc_host_int);
      aggregated_host_buffer_t<int, decltype(alloc_host_int)> smooth_field(
          hydro.get_nf(), int{}, alloc_host_int);

      // TODO Adapt indexing for aggregated buffer
      // Convert input
      convert_x_structure(X, combined_x.data());
      for (int f = 0; f < hydro.get_nf(); f++) {
          std::copy(U[f].begin(), U[f].end(), combined_u.data() + f * H_N3);
      }

      const auto& disc_detect_bool = hydro.get_disc_detect();
      const auto& smooth_bool = hydro.get_smooth_field();
      // TODO Update indexing for aggregated buffers
      for (auto f = 0; f < hydro.get_nf(); f++) {
          disc_detect[f] = disc_detect_bool[f];
          smooth_field[f] = smooth_bool[f];
      }

      // TODO Wrap function calls in exec_slice
      // Move input to device
      hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor), cudaMemcpyAsync,
          device_u.device_side_buffer, combined_u.data(),
          (hydro.get_nf() * H_N3 + 128) * sizeof(double), cudaMemcpyHostToDevice);
      hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor), cudaMemcpyAsync,
          device_x.device_side_buffer, combined_x.data(), (NDIM * q_inx3 + 128) * sizeof(double),
          cudaMemcpyHostToDevice);
      hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor), cudaMemcpyAsync,
          device_disc_detect.device_side_buffer, disc_detect.data(), (hydro.get_nf()) * sizeof(int),
          cudaMemcpyHostToDevice);
      hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor), cudaMemcpyAsync,
          device_smooth_field.device_side_buffer, smooth_field.data(), (hydro.get_nf()) * sizeof(int),
          cudaMemcpyHostToDevice);


      // TODO Wrap function calls in exec_slice
      // get discs
      launch_find_contact_discs_cuda(executor, device_u.device_side_buffer,
          device_P.device_side_buffer, device_unified_discs.device_side_buffer, physics<NDIM>::A_,
          physics<NDIM>::B_, physics<NDIM>::fgamma_, physics<NDIM>::de_switch_1);

      // TODO Update indexing for aggregated buffers
      for (int n = 0; n < NDIM; n++) {
          std::copy(X[n].begin(), X[n].end(), combined_large_x.data() + n * H_N3);
      }
      // TODO Wrap function calls in exec_slice
      hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor), cudaMemcpyAsync,
          device_large_x.device_side_buffer, combined_large_x.data(),
          (NDIM * H_N3 + 128) * sizeof(double), cudaMemcpyHostToDevice);

      // TODO Wrap function calls in exec_slice
      launch_hydro_pre_recon_cuda(executor, device_large_x.device_side_buffer, omega,
          hydro.get_angmom_index() != -1, device_u.device_side_buffer, hydro.get_nf(),
          opts().n_species);

      // TODO Wrap function calls in exec_slice
      launch_reconstruct_cuda(executor, omega, hydro.get_nf(), hydro.get_angmom_index(),
          device_smooth_field.device_side_buffer, device_disc_detect.device_side_buffer,
          device_q.device_side_buffer, device_x.device_side_buffer, device_u.device_side_buffer,
          device_AM.device_side_buffer, X[0][geo.H_DNX] - X[0][0],
          device_unified_discs.device_side_buffer, opts().n_species);

      // Call Flux kernel
      timestep_t ts;
      size_t nf_ = hydro.get_nf();
      double dx = X[0][geo.H_DNX] - X[0][0];
      constexpr int number_blocks = (q_inx3 / 128 + 1);

      const bool* masks = get_gpu_masks();
      aggregated_device_buffer_t<double, decltype(alloc_device_double)> device_amax(
          number_blocks * NDIM * (1 + 2 * nf_), device_id, alloc_device_double);
      aggregated_device_buffer_t<int, decltype(alloc_device_int)>
        device_amax_indices(
          number_blocks * NDIM, device_id, alloc_device_int);
      aggregated_device_buffer_t<int, decltype(alloc_device_int)> device_amax_d(
          number_blocks * NDIM, device_id, alloc_device_int);

      double A_ = physics<NDIM>::A_;
      double B_ = physics<NDIM>::B_;
      double fgamma = physics<NDIM>::fgamma_;
      double de_switch_1 = physics<NDIM>::de_switch_1;
      int nf_local = physics<NDIM>::nf_;

      assert(NDIM == 3);
      dim3 const grid_spec(1, number_blocks, 3);
      dim3 const threads_per_block(2, 8, 8);
      double omega_local = omega;
#if defined(OCTOTIGER_HAVE_CUDA)
      void* args[] = {&(device_q.device_side_buffer), &(device_x.device_side_buffer),
          &(device_f.device_side_buffer), &(device_amax.device_side_buffer),
          &(device_amax_indices.device_side_buffer), &(device_amax_d.device_side_buffer), &masks,
          &omega_local, &dx, &A_, &B_, &nf_local, &fgamma, &de_switch_1};
      // TODO Wrap in slice executor
      launch_flux_cuda_kernel_post(executor, grid_spec, threads_per_block,
          args);
#elif defined(OCTOTIGER_HAVE_HIP)
        // TODO Wrap in slice executor
      launch_flux_hip_kernel_post(executor, grid_spec, threads_per_block,
          device_q.device_side_buffer, device_x.device_side_buffer, device_f.device_side_buffer,
          device_amax.device_side_buffer, device_amax_indices.device_side_buffer,
          device_amax_d.device_side_buffer, masks, omega_local, dx, A_, B_,
          nf_local, fgamma,
          de_switch_1);
#endif

      // Move data to host
      aggregated_host_buffer_t<double, decltype(alloc_host_double)> amax(
          number_blocks * NDIM * (1 + 2 * nf_local), double{}, alloc_host_double);
      aggregated_host_buffer_t<int, decltype(alloc_host_int)> amax_indices(number_blocks * NDIM, int{}, alloc_host_int);
      aggregated_host_buffer_t<int, decltype(alloc_host_int)> amax_d(number_blocks * NDIM, int{}, alloc_host_int);

        // TODO Wrap in slice executor
      hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor), cudaMemcpyAsync,
          amax.data(), device_amax.device_side_buffer,
          (number_blocks * NDIM * (1 + 2 * nf_local)) * sizeof(double), cudaMemcpyDeviceToHost);
      hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor), cudaMemcpyAsync,
          amax_indices.data(), device_amax_indices.device_side_buffer,
          number_blocks * NDIM * sizeof(int), cudaMemcpyDeviceToHost);
      hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor), cudaMemcpyAsync,
          amax_d.data(), device_amax_d.device_side_buffer, number_blocks * NDIM * sizeof(int),
          cudaMemcpyDeviceToHost);
      auto flux_kernel_fut = hpx::async(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
          cudaMemcpyAsync, f.data(), device_f.device_side_buffer,
          (NDIM * nf_local * q_inx3 + 128) * sizeof(double), cudaMemcpyDeviceToHost);
      flux_kernel_fut.get();

      // Find Maximum
      size_t current_dim = 0;
      for (size_t dim_i = 1; dim_i < number_blocks * NDIM; dim_i++) {
        if (amax[dim_i] > amax[current_dim]) { 
          current_dim = dim_i;
        } else if (amax[dim_i] == amax[current_dim]) {
          if (amax_indices[dim_i] < amax_indices[current_dim]) {
            current_dim = dim_i;
          }
        }
      }
      std::vector<double> URs(nf_local), ULs(nf_local);
      const size_t current_max_index = amax_indices[current_dim];
      // const size_t current_d = amax_d[current_dim];
      ts.a = amax[current_dim];
      ts.x = combined_x[current_max_index];
      ts.y = combined_x[current_max_index + q_inx3];
      ts.z = combined_x[current_max_index + 2 * q_inx3];
      
      const size_t current_i = current_dim;
      current_dim = current_dim / number_blocks;
      for (int f = 0; f < nf_local; f++) {
          URs[f] = amax[NDIM * number_blocks + current_i * 2 * nf_local + f];
          ULs[f] = amax[NDIM * number_blocks + current_i * 2 * nf_local + nf_local + f];
      }
      ts.ul = std::move(URs);
      ts.ur = std::move(ULs);
      ts.dim = current_dim;
      auto max_lambda = ts;

      /* auto max_lambda = launch_flux_cuda(executor, device_q.device_side_buffer, f, combined_x, */
      /*     device_x.device_side_buffer, omega, hydro.get_nf(), X[0][geo.H_DNX] - X[0][0], device_id); */

      // TODO Update indexing to slice indexing
      // Convert output
      for (int dim = 0; dim < NDIM; dim++) {
          for (integer field = 0; field != opts().n_fields; ++field) {
              const auto dim_offset = dim * opts().n_fields * q_inx3 + field * q_inx3;
              for (integer i = 0; i <= INX; ++i) {
                  for (integer j = 0; j <= INX; ++j) {
                      for (integer k = 0; k <= INX; ++k) {
                          const auto i0 = findex(i, j, k);
                          const auto input_index =
                              (i + 1) * q_inx * q_inx + (j + 1) * q_inx + (k + 1);
                          F[dim][field][i0] = f[dim_offset + input_index];
                          // std::cout << F[dim][field][i0] << " ";
                      }
                  }
              }
          }
      }
      return max_lambda;
    });
    return ret_fut.get();
}
#endif
