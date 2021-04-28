#pragma once
#ifdef OCTOTIGER_HAVE_CUDA
#include <buffer_manager.hpp>
#include <cuda_buffer_util.hpp>
#include <cuda_runtime.h>
#include <stream_manager.hpp>

#include <hpx/apply.hpp>
#include <hpx/synchronization/once.hpp>

#include "octotiger/cuda_util/cuda_helper.hpp"
#include "octotiger/options.hpp"
#include "octotiger/grid.hpp"

#include "octotiger/unitiger/hydro_impl/hydro_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp" 
#include "octotiger/unitiger/hydro_impl/flux_kernel_templates.hpp" // required for fill_masks
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp" 

hpx::lcos::local::once_flag flag1;

__host__ void init_gpu_masks(bool *masks) {
  boost::container::vector<bool> masks_boost(NDIM * 10 * 10 * 10);
  fill_masks(masks_boost);
  cudaMemcpy(masks, masks_boost.data(), NDIM * 1000 * sizeof(bool), cudaMemcpyHostToDevice);
}

__host__ const bool* get_gpu_masks(void) {
    // TODO Create class to handle these read-only, created-once GPU buffers for masks. This is a reoccuring problem
    static bool *masks = recycler::recycle_allocator_cuda_device<bool>{}.allocate(NDIM * 1000);
    hpx::lcos::local::call_once(flag1, init_gpu_masks, masks);
    return masks;
}

timestep_t launch_flux_cuda(stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
    double* device_q,
    std::vector<double, recycler::recycle_allocator_cuda_host<double>> &combined_f,
    std::vector<double, recycler::recycle_allocator_cuda_host<double>> &combined_x, double* device_x,
    safe_real omega, const size_t nf_, double dx, size_t device_id) {
    timestep_t ts;

    const cell_geometry<3, 8> geo;

    recycler::cuda_device_buffer<double> device_f(NDIM * nf_ * 1000 + 32, device_id);
    const bool *masks = get_gpu_masks();

    recycler::cuda_device_buffer<double> device_amax(7 * NDIM * (1 + 2 * nf_));
    recycler::cuda_device_buffer<int> device_amax_indices(7 * NDIM);
    recycler::cuda_device_buffer<int> device_amax_d(7 * NDIM);
    double A_ = physics<NDIM>::A_;
    double B_ = physics<NDIM>::B_;
    double fgamma = physics<NDIM>::fgamma_;
    double de_switch_1 = physics<NDIM>::de_switch_1;
    int nf_local = physics<NDIM>::nf_;

    dim3 const grid_spec(1, 7, 3);
    dim3 const threads_per_block(2, 8, 8);
    void* args[] = {&(device_q),
      &(device_x), &(device_f.device_side_buffer), &(device_amax.device_side_buffer),
      &(device_amax_indices.device_side_buffer), &(device_amax_d.device_side_buffer),
      &masks, &omega, &dx, &A_, &B_, &nf_local, &fgamma, &de_switch_1};
    launch_flux_cuda_kernel_post(executor, grid_spec, threads_per_block, args);

    // Move data to host
    std::vector<double, recycler::recycle_allocator_cuda_host<double>> amax(7 * NDIM * (1 + 2 * nf_));
    std::vector<int, recycler::recycle_allocator_cuda_host<int>> amax_indices(7 * NDIM);
    std::vector<int, recycler::recycle_allocator_cuda_host<int>> amax_d(7 * NDIM);

    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
               cudaMemcpyAsync, amax.data(),
               device_amax.device_side_buffer, (7 * NDIM * (1 + 2 * nf_)) * sizeof(double),
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
               (NDIM * nf_ * 1000 + 32) * sizeof(double), cudaMemcpyDeviceToHost);
    fut.get();

    // Find Maximum
    size_t current_dim = 0;
    for (size_t dim_i = 1; dim_i < 7 * NDIM; dim_i++) {
      if (amax[dim_i] > amax[current_dim]) { 
        current_dim = dim_i;
      }
    }
    std::vector<double> URs(nf_), ULs(nf_);
    const size_t current_max_index = amax_indices[current_dim];
    const size_t current_d = amax_d[current_dim];
    ts.a = amax[current_dim];
    ts.x = combined_x[current_max_index];
    ts.y = combined_x[current_max_index + 1000];
    ts.z = combined_x[current_max_index + 2000];
    const size_t current_i = current_dim;
    current_dim = current_dim / 7;
    const auto flipped_dim = geo.flip_dim(current_d, current_dim);
    constexpr int compressedH_DN[3] = {100, 10, 1};
    for (int f = 0; f < nf_; f++) {
        URs[f] = amax[21 + current_i * 2 * nf_ + f];
        ULs[f] = amax[21 + current_i * 2 * nf_ + nf_ + f];
    }
    ts.ul = std::move(ULs);
    ts.ur = std::move(URs);
    ts.dim = current_dim;
    return ts;
}

// Input U, X, omega, executor, device_id
// Output F
timestep_t launch_hydro_cuda_kernels(const hydro_computer<NDIM, INX, physics<NDIM>>& hydro,
    const std::vector<std::vector<safe_real>>& U, const std::vector<std::vector<safe_real>>& X,
    const double omega, const size_t device_id,
    stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
    std::vector<hydro_state_t<std::vector<safe_real>>> &F) {
    static const cell_geometry<NDIM, INX> geo;

    // Device buffers
    recycler::cuda_device_buffer<double> device_q(
        hydro.get_nf() * 27 * 10 * 10 * 10 + 32, device_id);
    recycler::cuda_device_buffer<double> device_x(NDIM * 1000 + 32, device_id);
    recycler::cuda_device_buffer<double> device_large_x(NDIM * H_N3 + 32, device_id);
    recycler::cuda_device_buffer<double> device_f(NDIM * hydro.get_nf() * 1000 + 32, device_id);
    recycler::cuda_device_buffer<double> device_u(hydro.get_nf() * H_N3 + 32);
    recycler::cuda_device_buffer<double> device_amax(NDIM);
    recycler::cuda_device_buffer<int> device_amax_indices(NDIM);
    recycler::cuda_device_buffer<int> device_amax_d(NDIM);
    recycler::cuda_device_buffer<double> device_unified_discs(geo.NDIR / 2 * H_N3 + 32);
    recycler::cuda_device_buffer<double> device_P(H_N3 + 32);
    recycler::cuda_device_buffer<int> device_disc_detect(hydro.get_nf());
    recycler::cuda_device_buffer<int> device_smooth_field(hydro.get_nf());
    recycler::cuda_device_buffer<double> device_AM(NDIM * 10 * 10 * 10 + 32);

    // Host buffers
    std::vector<double, recycler::recycle_allocator_cuda_host<double>> combined_x(NDIM * 1000 + 32);
    std::vector<double, recycler::recycle_allocator_cuda_host<double>> combined_large_x(
        NDIM * H_N3 + 32);
    std::vector<double, recycler::recycle_allocator_cuda_host<double>> combined_u(
        hydro.get_nf() * H_N3 + 32);
    std::vector<int, recycler::recycle_allocator_cuda_host<int>> disc_detect(hydro.get_nf());
    std::vector<int, recycler::recycle_allocator_cuda_host<int>> smooth_field(hydro.get_nf());
    std::vector<double, recycler::recycle_allocator_cuda_host<double>> f(
        NDIM * hydro.get_nf() * 1000 + 32);

    // Convert input
    convert_x_structure(X, combined_x.data());
    for (int f = 0; f < hydro.get_nf(); f++) {
        std::copy(U[f].begin(), U[f].end(), combined_u.data() + f * H_N3);
    }

    const auto& disc_detect_bool = hydro.get_disc_detect();
    const auto& smooth_bool = hydro.get_smooth_field();
    for (auto f = 0; f < hydro.get_nf(); f++) {
        disc_detect[f] = disc_detect_bool[f];
        smooth_field[f] = smooth_bool[f];
    }

    // Move input to device
    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor), cudaMemcpyAsync,
        device_u.device_side_buffer, combined_u.data(),
        (hydro.get_nf() * H_N3 + 32) * sizeof(double), cudaMemcpyHostToDevice);
    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor), cudaMemcpyAsync,
        device_x.device_side_buffer, combined_x.data(), (NDIM * 1000 + 32) * sizeof(double),
        cudaMemcpyHostToDevice);
    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor), cudaMemcpyAsync,
        device_disc_detect.device_side_buffer, disc_detect.data(), (hydro.get_nf()) * sizeof(int),
        cudaMemcpyHostToDevice);
    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor), cudaMemcpyAsync,
        device_smooth_field.device_side_buffer, smooth_field.data(), (hydro.get_nf()) * sizeof(int),
        cudaMemcpyHostToDevice);

    // get discs
    launch_find_contact_discs_cuda(executor, device_u.device_side_buffer,
        device_P.device_side_buffer, device_unified_discs.device_side_buffer, physics<NDIM>::A_,
        physics<NDIM>::B_, physics<NDIM>::fgamma_, physics<NDIM>::de_switch_1);

    for (int n = 0; n < NDIM; n++) {
        std::copy(X[n].begin(), X[n].end(), combined_large_x.data() + n * H_N3);
    }
    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor), cudaMemcpyAsync,
        device_large_x.device_side_buffer, combined_large_x.data(),
        (NDIM * H_N3 + 32) * sizeof(double), cudaMemcpyHostToDevice);

    launch_hydro_pre_recon_cuda(executor, device_large_x.device_side_buffer, omega,
        hydro.get_angmom_index() != -1, device_u.device_side_buffer, hydro.get_nf(),
        opts().n_species);

    launch_reconstruct_cuda(executor, omega, hydro.get_nf(), hydro.get_angmom_index(),
        device_smooth_field.device_side_buffer, device_disc_detect.device_side_buffer,
        device_q.device_side_buffer, device_x.device_side_buffer, device_u.device_side_buffer,
        device_AM.device_side_buffer, X[0][geo.H_DNX] - X[0][0],
        device_unified_discs.device_side_buffer, opts().n_species);

    // Call Flux kernel
    auto max_lambda = launch_flux_cuda(executor, device_q.device_side_buffer, f, combined_x,
        device_x.device_side_buffer, omega, hydro.get_nf(), X[0][geo.H_DNX] - X[0][0], device_id);
    //     auto max_lambda = flux_cpu_kernel(device_q.device_side_buffer, f,
    //     device_x.device_side_buffer, omega, hydro.get_nf());
    // printf("\n\nAFTER flux\n\n");

    // Convert output
    for (int dim = 0; dim < NDIM; dim++) {
        for (integer field = 0; field != opts().n_fields; ++field) {
            const auto dim_offset = dim * opts().n_fields * 1000 + field * 1000;
#pragma GCC ivdep
            for (integer i = 0; i <= INX; ++i) {
                for (integer j = 0; j <= INX; ++j) {
                    for (integer k = 0; k <= INX; ++k) {
                        const auto i0 = findex(i, j, k);
                        const auto input_index = (i + 1) * 10 * 10 + (j + 1) * 10 + (k + 1);
                        F[dim][field][i0] = f[dim_offset + input_index];
                    }
                }
            }
        }
    }
    return max_lambda;
}
#endif