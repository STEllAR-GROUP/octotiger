#pragma once

#ifdef OCTOTIGER_HAVE_CUDA
#include "octotiger/grid.hpp"
#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp" 
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp"

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
