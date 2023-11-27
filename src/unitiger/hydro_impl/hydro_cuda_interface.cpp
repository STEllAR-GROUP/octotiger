#include <hpx/apply.hpp>
#include <hpx/synchronization/once.hpp>

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


#include "octotiger/cuda_util/cuda_helper.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/options.hpp"

#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/flux_kernel_templates.hpp"    // required for fill_masks
#include "octotiger/unitiger/hydro_impl/hydro_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp"    // required for constants

#include "octotiger/unitiger/hydro_impl/hydro_performance_counters.hpp"

static const char hydro_cuda_kernel_identifier[] = "hydro_kernel_aggregator_cuda";
using hydro_cuda_agg_executor_pool = aggregation_pool<hydro_cuda_kernel_identifier,
    hpx::cuda::experimental::cuda_executor, pool_strategy>;

hpx::once_flag flag1;
hpx::once_flag init_hydro_pool_flag;

#if defined(OCTOTIGER_HAVE_CUDA)
template <typename T>
using host_pinned_allocator = recycler::detail::cuda_pinned_allocator<T>;
template <typename T>
using device_allocator = recycler::detail::cuda_device_allocator<T>;

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

template <typename T>
using host_pinned_allocator = recycler::detail::hip_pinned_allocator<T>;
template <typename T>
using device_allocator = recycler::detail::hip_device_allocator<T>;
template <typename T, typename Alloc>
using aggregated_device_buffer_t = recycler::hip_aggregated_device_buffer<T, Alloc>;
template <typename T, typename Alloc>
using aggregated_host_buffer_t = std::vector<T, Alloc>;

#define cudaLaunchKernel hipLaunchKernel
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyAsync hipMemcpyAsync

#endif

void init_hydro_aggregation_pool(void) {
    const size_t max_slices = opts().max_kernels_fused;
    constexpr size_t number_aggregation_executors = 16;
    constexpr Aggregated_Executor_Modes executor_mode = Aggregated_Executor_Modes::EAGER;
    hydro_cuda_agg_executor_pool::init(number_aggregation_executors, max_slices,
        executor_mode, opts().number_gpus);
}

__host__ void init_gpu_masks(std::array<bool*, recycler::max_number_gpus>& masks) {
    boost::container::vector<bool> masks_boost(NDIM * q_inx * q_inx * q_inx);
    fill_masks(masks_boost);
    for (size_t gpu_id = 0; gpu_id < opts().number_gpus; gpu_id++) {
      const size_t location_id = 0;
#if defined(OCTOTIGER_HAVE_CUDA)
      masks[gpu_id] = recycler::detail::buffer_recycler::get<bool,
          typename recycler::recycle_allocator_cuda_device<bool>::underlying_allocator_type>(
          NDIM * q_inx3, false, location_id, gpu_id);
#elif defined(OCTOTIGER_HAVE_HIP)
      masks[gpu_id] = recycler::detail::buffer_recycler::get<bool,
          typename recycler::recycle_allocator_hip_device<bool>::underlying_allocator_type>(
          NDIM * q_inx3, false, location_id, gpu_id);
#endif
      cudaMemcpy(masks[gpu_id], masks_boost.data(), NDIM * q_inx3 * sizeof(bool), cudaMemcpyHostToDevice);
    }
}

__host__ bool* get_gpu_masks(const size_t gpu_id = 0) {
    static std::array<bool*, recycler::max_number_gpus> masks;
    hpx::call_once(flag1, init_gpu_masks, masks);
    return masks[gpu_id];
}

// Input U, X, omega, executor, device_id
// Output F
// TODO remove obsolete executor
timestep_t launch_hydro_cuda_kernels(const hydro_computer<NDIM, INX, physics<NDIM>>& hydro,
    const f_data_t& U_flat, const std::vector<std::vector<safe_real>>& X,
    const double omega, const size_t device_id,
#if defined(OCTOTIGER_HAVE_CUDA) || (defined(OCTOTIGER_HAVE_KOKKOS) && (defined(KOKKOS_ENABLE_CUDA)))
    std::vector<real, recycler::detail::cuda_pinned_allocator<real>>& F_flat) {
#else
    std::vector<real>& F_flat) {
#endif

    // Init local kernel pool if not done already
    hpx::call_once(init_hydro_pool_flag, init_hydro_aggregation_pool);
    // Get executor (-slice if aggregated with max_slices > 1) future
    auto executor_slice_fut = hydro_cuda_agg_executor_pool::request_executor_slice();
    // Add continuation to executor future to execute the hydro kernels as soon as it's ready
    auto ret_fut = executor_slice_fut.value().then(hpx::annotated_function([&](auto && fut) {
      // Unwrap executor from ready future
      aggregated_executor_t exec_slice = fut.get();
      // How many executor slices are working together and what's our ID?
      const size_t slice_id = exec_slice.id;
      const size_t number_slices = exec_slice.number_slices;

      // Get allocators of all the executors working together
      auto alloc_host_double =
          exec_slice
              .template make_allocator<double, host_pinned_allocator<double>>();
      auto alloc_host_int =
          exec_slice
              .template make_allocator<int, host_pinned_allocator<int>>();
      auto alloc_device_int =
          exec_slice
              .template make_allocator<int, device_allocator<int>>();
      auto alloc_device_double =
          exec_slice
              .template make_allocator<double, device_allocator<double>>();

      static const cell_geometry<NDIM, INX> geo;

      const size_t max_slices = opts().max_kernels_fused;

      // Move input to device
      aggregated_host_buffer_t<double, decltype(alloc_host_double)> combined_u(
          (hydro.get_nf() * H_N3 + 128) * max_slices, double{}, alloc_host_double);
      aggregated_device_buffer_t<double, decltype(alloc_device_double)> device_u(
          (hydro.get_nf() * H_N3 + 128) * max_slices, alloc_device_double);


      // Device buffers
      aggregated_device_buffer_t<double, decltype(alloc_device_double)> device_q(
          (hydro.get_nf() * 27 * q_inx * q_inx * q_inx + 128) * max_slices, 
          alloc_device_double);
      aggregated_device_buffer_t<double, decltype(alloc_device_double)> device_x(
          (NDIM * q_inx3 + 128) * max_slices, alloc_device_double);
      aggregated_device_buffer_t<double, decltype(alloc_device_double)> device_large_x(
          (NDIM * H_N3 + 128) * max_slices, alloc_device_double);
      aggregated_device_buffer_t<double, decltype(alloc_device_double)> device_f(
          (NDIM * hydro.get_nf() * q_inx3 + 128) * max_slices, alloc_device_double);
      aggregated_device_buffer_t<double, decltype(alloc_device_double)>
        device_unified_discs(
          (geo.NDIR / 2 * H_N3 + 128) * max_slices, alloc_device_double);
      aggregated_device_buffer_t<double, decltype(alloc_device_double)> device_P(
          (H_N3 + 128) * max_slices, alloc_device_double);
      aggregated_device_buffer_t<int, decltype(alloc_device_int)> device_disc_detect(
          (hydro.get_nf()) * max_slices, alloc_device_int);
      aggregated_device_buffer_t<int, decltype(alloc_device_int)> device_smooth_field(
          (hydro.get_nf()) * max_slices, alloc_device_int);
      aggregated_device_buffer_t<double, decltype(alloc_device_double)> device_AM(
          (NDIM * q_inx * q_inx * q_inx + 128) * max_slices, alloc_device_double);


      // Host buffers
      aggregated_host_buffer_t<double, decltype(alloc_host_double)> combined_x(
          (NDIM * q_inx3 + 128) * max_slices, double{}, alloc_host_double);
      aggregated_host_buffer_t<double, decltype(alloc_host_double)> combined_large_x(
          (NDIM * H_N3 + 128) * max_slices, double{}, alloc_host_double);
      aggregated_host_buffer_t<double, decltype(alloc_host_double)> f(
          (NDIM * hydro.get_nf() * q_inx3 + 128) * max_slices, double{}, alloc_host_double);
      aggregated_host_buffer_t<int, decltype(alloc_host_int)> disc_detect(
          (hydro.get_nf()) * max_slices, int{}, alloc_host_int);
      aggregated_host_buffer_t<int, decltype(alloc_host_int)> smooth_field(
          (hydro.get_nf()) * max_slices, int{}, alloc_host_int);

      // Slice offsets
      const int u_slice_offset = hydro.get_nf() * H_N3 + 128;
      constexpr int x_slice_offset = NDIM * q_inx3 + 128;
      const int disc_detect_slice_offset = hydro.get_nf();
      const int smooth_slice_offset = hydro.get_nf();
      constexpr int large_x_slice_offset = (H_N3 * NDIM + 128); 
      //const int q_slice_offset = (nf_ * 27 * H_N3 + 128) 
      const int f_slice_offset = (NDIM* hydro.get_nf() *  q_inx3 + 128);
      constexpr int disc_offset = geo.NDIR / 2 * H_N3 + 128;

      hpx::annotated_function(
          [&]() {
              // Convert input
              convert_x_structure(X, combined_x.data() + x_slice_offset * slice_id);
              /* for (int f = 0; f < hydro.get_nf(); f++) { */
                  /* std::copy(U[f].begin(), U[f].end(), */
                  /*     combined_u.data() + f * H_N3 + u_slice_offset * slice_id); */
              /* } */
                  std::copy(U_flat.begin(), U_flat.end(),
                      combined_u.data());
          },
          "cuda_hydro_solver::convert_input")();

      const auto& disc_detect_bool = hydro.get_disc_detect();
      const auto& smooth_bool = hydro.get_smooth_field();
      for (auto f = 0; f < hydro.get_nf(); f++) {
          disc_detect[f + disc_detect_slice_offset * slice_id] = disc_detect_bool[f];
          smooth_field[f + smooth_slice_offset * slice_id] = smooth_bool[f];
      }

      hpx::apply(exec_slice, cudaMemcpyAsync, device_u.device_side_buffer, combined_u.data(),
          (hydro.get_nf() * H_N3 + 128) * sizeof(double) * number_slices, cudaMemcpyHostToDevice);
      hpx::apply(exec_slice, cudaMemcpyAsync, device_x.device_side_buffer, combined_x.data(),
          (NDIM * q_inx3 + 128) * sizeof(double) * number_slices, cudaMemcpyHostToDevice);
      hpx::apply(exec_slice, cudaMemcpyAsync, device_disc_detect.device_side_buffer,
          disc_detect.data(), (hydro.get_nf()) * sizeof(int) * number_slices,
          cudaMemcpyHostToDevice);
      hpx::apply(exec_slice, cudaMemcpyAsync, device_smooth_field.device_side_buffer,
          smooth_field.data(), (hydro.get_nf()) * sizeof(int) * number_slices,
          cudaMemcpyHostToDevice);

      // get discs
      launch_find_contact_discs_cuda(exec_slice, device_u.device_side_buffer,
          device_P.device_side_buffer, device_unified_discs.device_side_buffer, physics<NDIM>::A_,
          physics<NDIM>::B_, physics<NDIM>::fgamma_, physics<NDIM>::de_switch_1, hydro.get_nf());

      for (int n = 0; n < NDIM; n++) {
          std::copy(X[n].begin(), X[n].end(),
              combined_large_x.data() + large_x_slice_offset * slice_id + n *
              H_N3);
      }
      hpx::apply(exec_slice, cudaMemcpyAsync, device_large_x.device_side_buffer,
          combined_large_x.data(), (NDIM * H_N3 + 128) * sizeof(double) * number_slices,
          cudaMemcpyHostToDevice);

      launch_hydro_pre_recon_cuda(exec_slice, device_large_x.device_side_buffer, omega,
          hydro.get_angmom_index() != -1, device_u.device_side_buffer, hydro.get_nf(),
          opts().n_species);

      const double dx = X[0][geo.H_DNX] - X[0][0];
      std::vector<double, decltype(alloc_host_double)> dx_host(
          max_slices * 1, double{}, alloc_host_double);
      aggregated_device_buffer_t<double, decltype(alloc_device_double)>
        dx_device(
          max_slices, alloc_device_double);
      dx_host[slice_id] = dx;
      hpx::apply(exec_slice, cudaMemcpyAsync, dx_device.device_side_buffer, dx_host.data(),
          number_slices * sizeof(double), cudaMemcpyHostToDevice);

      launch_reconstruct_cuda(exec_slice, omega, hydro.get_nf(), hydro.get_angmom_index(),
          device_smooth_field.device_side_buffer, device_disc_detect.device_side_buffer,
          device_q.device_side_buffer, device_x.device_side_buffer, device_u.device_side_buffer,
          device_AM.device_side_buffer, dx_device.device_side_buffer,
          device_unified_discs.device_side_buffer, opts().n_species);

      // Call Flux kernel
      timestep_t ts;
      size_t nf_ = hydro.get_nf();

      int number_blocks = (q_inx3 / 128 + 1);

      const bool* masks = get_gpu_masks(exec_slice.parent.gpu_id);
      aggregated_device_buffer_t<double, decltype(alloc_device_double)> device_amax(
          max_slices * number_blocks * NDIM * (1 + 2 * nf_), alloc_device_double);
      aggregated_device_buffer_t<int, decltype(alloc_device_int)>
        device_amax_indices(
          max_slices * number_blocks * NDIM, alloc_device_int);
      aggregated_device_buffer_t<int, decltype(alloc_device_int)> device_amax_d(
          max_slices * number_blocks * NDIM, alloc_device_int);

      double A_ = physics<NDIM>::A_;
      double B_ = physics<NDIM>::B_;
      double fgamma = physics<NDIM>::fgamma_;
      double de_switch_1 = physics<NDIM>::de_switch_1;
      int nf_local = physics<NDIM>::nf_;

      assert(NDIM == 3);
      dim3 const grid_spec(number_slices, number_blocks, 3);
      dim3 const threads_per_block(2, 8, 8);
      double omega_local = omega;
#if defined(OCTOTIGER_HAVE_CUDA)
      void* args[] = {&(device_q.device_side_buffer), &(device_x.device_side_buffer),
          &(device_f.device_side_buffer), &(device_amax.device_side_buffer),
          &(device_amax_indices.device_side_buffer), &(device_amax_d.device_side_buffer), &masks,
          &omega_local, &(dx_device.device_side_buffer), &A_, &B_, &nf_local, &fgamma, &de_switch_1, &number_blocks};
      launch_flux_cuda_kernel_post(exec_slice, grid_spec, threads_per_block,
          args);
#elif defined(OCTOTIGER_HAVE_HIP)
      launch_flux_hip_kernel_post(exec_slice, grid_spec, threads_per_block,
          device_q.device_side_buffer, device_x.device_side_buffer, device_f.device_side_buffer,
          device_amax.device_side_buffer, device_amax_indices.device_side_buffer,
          device_amax_d.device_side_buffer, masks, omega_local, dx_device.device_side_buffer, A_, B_,
          nf_local, fgamma,
          de_switch_1, number_blocks);
#endif

      // Move data to host
      aggregated_host_buffer_t<double, decltype(alloc_host_double)> amax(
          max_slices * number_blocks * NDIM * (1 + 2 * nf_local), double{}, alloc_host_double);
      aggregated_host_buffer_t<int, decltype(alloc_host_int)> amax_indices(
          max_slices * number_blocks * NDIM, int{}, alloc_host_int);
      aggregated_host_buffer_t<int, decltype(alloc_host_int)> amax_d(
          max_slices * number_blocks * NDIM, int{}, alloc_host_int);

      hpx::apply(exec_slice, cudaMemcpyAsync, amax.data(), device_amax.device_side_buffer,
          (number_slices * number_blocks * NDIM * (1 + 2 * nf_local)) *
          sizeof(double),
          cudaMemcpyDeviceToHost);
      hpx::apply(exec_slice, cudaMemcpyAsync, amax_indices.data(),
          device_amax_indices.device_side_buffer,
          number_slices * number_blocks * NDIM * sizeof(int), cudaMemcpyDeviceToHost);
      hpx::apply(exec_slice, cudaMemcpyAsync, amax_d.data(), device_amax_d.device_side_buffer,
          number_slices * number_blocks * NDIM * sizeof(int), cudaMemcpyDeviceToHost);
      auto flux_kernel_fut =
          hpx::async(exec_slice, cudaMemcpyAsync, f.data(), device_f.device_side_buffer,
              number_slices * (NDIM * nf_local * q_inx3 + 128) *
              sizeof(double),
              cudaMemcpyDeviceToHost);

      octotiger::hydro::hydro_cuda_gpu_subgrids_processed++;
      if(slice_id == 0)
        octotiger::hydro::hydro_cuda_gpu_aggregated_subgrids_launches++;

      flux_kernel_fut.get();

      // Find Maximum
      const int amax_slice_offset = NDIM * (1 + 2 * nf_local) * number_blocks * slice_id;
      const int max_indices_slice_offset = NDIM * number_blocks * slice_id;
      size_t current_dim = 0;
      for (size_t dim_i = 1; dim_i < number_blocks * NDIM; dim_i++) {
        if (amax[dim_i + amax_slice_offset] > amax[current_dim + amax_slice_offset]) { 
          current_dim = dim_i;
        } else if (amax[dim_i + amax_slice_offset] == amax[current_dim + amax_slice_offset]) {
          if (amax_indices[dim_i + max_indices_slice_offset] <
              amax_indices[current_dim + max_indices_slice_offset]) {
              current_dim = dim_i;
          }
        }
      }
      std::vector<double> URs(nf_local), ULs(nf_local);
      const size_t current_max_index = amax_indices[current_dim + max_indices_slice_offset];
      // const size_t current_d = amax_d[current_dim];
      ts.a = amax[current_dim + amax_slice_offset];
      ts.x = combined_x[current_max_index + x_slice_offset * slice_id];
      ts.y = combined_x[current_max_index + q_inx3 + x_slice_offset * slice_id];
      ts.z = combined_x[current_max_index + 2 * q_inx3 + x_slice_offset * slice_id];
      
      const size_t current_i = current_dim;
      current_dim = current_dim / number_blocks;
      for (int f = 0; f < nf_local; f++) {
          URs[f] = amax[NDIM * number_blocks + current_i * 2 * nf_local + f + amax_slice_offset];
          ULs[f] = amax[NDIM * number_blocks + current_i * 2 * nf_local +
            nf_local + f +
              amax_slice_offset];
      }
      ts.ul = std::move(URs);
      ts.ur = std::move(ULs);
      ts.dim = current_dim;
      auto max_lambda = ts;

      /* auto max_lambda = launch_flux_cuda(executor, device_q.device_side_buffer, f, combined_x, */
      /*     device_x.device_side_buffer, omega, hydro.get_nf(), X[0][geo.H_DNX] - */
      /*     X[0][0], device_id); */

      // Convert output
      hpx::annotated_function([&]() {
        for (int dim = 0; dim < NDIM; dim++) {
            for (integer field = 0; field != opts().n_fields; ++field) {
                const auto dim_offset = dim * opts().n_fields * q_inx3 + field * q_inx3;
                for (integer i = 0; i <= INX; ++i) {
                    for (integer j = 0; j <= INX; ++j) {
                        for (integer k = 0; k <= INX; ++k) {
                            const auto i0 = findex(i, j, k);
                            const auto input_index =
                                (i + 1) * q_inx * q_inx + (j + 1) * q_inx + (k + 1);
                            /* F[dim][field][i0] = */
                            /*     f[dim_offset + input_index + f_slice_offset * slice_id]; */
                            F_flat[dim * nf_local * F_N3 + field * F_N3 + i0] =
                                f[dim_offset + input_index + f_slice_offset * slice_id];
                            // std::cout << F[dim][field][i0] << " ";
                        }
                    }
                }
            }
        }
      }, "cuda_hydro_solver::convert_output")();
      return max_lambda;

    }, "cuda_hydro_solver"));
    return ret_fut.get();
}
#endif
