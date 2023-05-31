//  Copyright (c) 2020-2022 Gregor Daiß
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//#pragma GCC push_options
//#pragma GCC optimize("unroll-loops")

#include <iostream>
#include <vector>

#include "octotiger/hydro_defs.hpp"
#include "octotiger/unitiger/hydro_impl/hydro_boundary_exchange.hpp"
#include "octotiger/util/vec_scalar_host_wrapper.hpp"

#include <hpx/synchronization/once.hpp>


#ifdef OCTOTIGER_HAVE_CUDA
static const char amr_cuda_kernel_identifier[] = "amr_kernel_aggregator_cuda";
using amr_cuda_agg_executor_pool = aggregation_pool<amr_cuda_kernel_identifier, hpx::cuda::experimental::cuda_executor,
                                       pool_strategy>;
hpx::once_flag init_pool_flag;

void init_aggregation_pool(void) {
    const size_t max_slices = opts().max_executor_slices;
    constexpr size_t number_aggregation_executors = 16;
    constexpr Aggregated_Executor_Modes executor_mode = Aggregated_Executor_Modes::EAGER;
    amr_cuda_agg_executor_pool::init(number_aggregation_executors, max_slices, executor_mode);
}

__host__ void launch_complete_hydro_amr_boundary_cuda(double dx, bool
    energy_only,
    const std::vector<std::vector<real>>& Ushad, const std::vector<int>& is_coarse,
    const std::array<double, NDIM>& xmin, std::vector<std::vector<real>>& U) {
    bool early_exit = true;
    for (int i = 0; i < HS_N3; i++) {
        if (early_exit && is_coarse[i]) {
          early_exit = false;
          break;
        }
    }
    if (early_exit)
      return;
    // Init local kernel pool if not done already
    hpx::call_once(init_pool_flag, init_aggregation_pool);

    auto executor_slice_fut =
      amr_cuda_agg_executor_pool::request_executor_slice();

    auto ret_fut = executor_slice_fut.value().then([&](auto&& fut) {
        // Unwrap executor from ready future
        aggregated_executor_t exec_slice = fut.get();
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
            exec_slice.template make_allocator<int, recycler::detail::cuda_pinned_allocator<int>>();
        auto alloc_device_int =
            exec_slice.template make_allocator<int, recycler::detail::cuda_device_allocator<int>>();
        int nfields = opts().n_fields;

        const size_t max_slices = opts().max_executor_slices;
        // Create host buffers
        std::vector<double, decltype(alloc_host_double)> unified_uf(
            max_slices * opts().n_fields * HS_N3 * 8, double{},
            alloc_host_double);
        std::vector<double, decltype(alloc_host_double)> unified_ushad(
            max_slices * opts().n_fields * HS_N3, double{}, alloc_host_double);
        std::vector<int, decltype(alloc_host_int)> coarse(
            max_slices * HS_N3, int{}, alloc_host_int);
        std::vector<double, decltype(alloc_host_double)> x_min(
            max_slices * NDIM, double{}, alloc_host_double);

        std::vector<int, decltype(alloc_host_int)> energy_only_host(
            max_slices * 1, int{}, alloc_host_int);
        std::vector<double, decltype(alloc_host_double)> dx_host(
            max_slices * 1, double{}, alloc_host_double);    

        // Create device buffers
        recycler::cuda_aggregated_device_buffer<double, decltype(alloc_device_double)> device_uf(
            max_slices * opts().n_fields * HS_N3 * 8, 0, alloc_device_double);
        recycler::cuda_aggregated_device_buffer<double, decltype(alloc_device_double)> device_ushad(
            max_slices * opts().n_fields * HS_N3, 0, alloc_device_double);
        recycler::cuda_aggregated_device_buffer<int, decltype(alloc_device_int)> device_coarse(
            max_slices * HS_N3, 0, alloc_device_int);
        recycler::cuda_aggregated_device_buffer<double, decltype(alloc_device_double)> device_xmin(
            max_slices * NDIM, 0, alloc_device_double);

        recycler::cuda_aggregated_device_buffer<int, decltype(alloc_device_int)> energy_only_device(
            max_slices * 1, 0, alloc_device_int);
        recycler::cuda_aggregated_device_buffer<double, decltype(alloc_device_double)> dx_device(
            max_slices * 1, 0, alloc_device_double);

        for (int d = 0; d < NDIM; d++) {
            x_min[d + slice_id * NDIM] = xmin[d];
        }
        exec_slice.post(cudaMemcpyAsync, device_xmin.device_side_buffer, x_min.data(),
            number_slices * (NDIM) * sizeof(double), cudaMemcpyHostToDevice);
        dx_host[slice_id] = dx;
        exec_slice.post(cudaMemcpyAsync, dx_device.device_side_buffer, dx_host.data(),
            number_slices * sizeof(double), cudaMemcpyHostToDevice);

        energy_only_host[slice_id] = energy_only;
        exec_slice.post(cudaMemcpyAsync, energy_only_device.device_side_buffer,
            energy_only_host.data(), number_slices * sizeof(int),
            cudaMemcpyHostToDevice);

        // Fill host buffers
        for (int f = 0; f < opts().n_fields; f++) {
            if (!energy_only || f == egas_i) {
                std::copy(Ushad[f].begin(), Ushad[f].begin() + HS_N3,
                    unified_ushad.begin() + f * HS_N3 + slice_id * nfields * HS_N3);
            }
        }
        exec_slice.post(cudaMemcpyAsync, device_ushad.device_side_buffer, unified_ushad.data(),
            number_slices * (opts().n_fields * HS_N3) * sizeof(double),
            cudaMemcpyHostToDevice);

        for (int i = 0; i < HS_N3; i++) {
            coarse[i + slice_id * HS_N3] = is_coarse[i];
        }
        exec_slice.post(cudaMemcpyAsync, device_coarse.device_side_buffer, coarse.data(),
            number_slices * (HS_N3) * sizeof(int), cudaMemcpyHostToDevice);

        dim3 const grid_spec(number_slices, 1, HS_NX - 2);
        dim3 const threads_per_block(1, HS_NX - 2, HS_NX - 2);
        void* args[] = {&(dx_device.device_side_buffer), &(energy_only_device.device_side_buffer),
            &(device_ushad.device_side_buffer), &(device_coarse.device_side_buffer),
            &(device_xmin.device_side_buffer), &(device_uf.device_side_buffer),
            &nfields};


        launch_complete_hydro_amr_boundary_cuda_post(
            exec_slice, grid_spec, threads_per_block, args);

        /* auto ret_fut = */
        /*     exec_slice.async(cudaMemcpyAsync, unified_uf.data(), device_uf.device_side_buffer, */
        /*         number_slices * (opts().n_fields * HS_N3 * 8) * sizeof(double), */
        /*         cudaMemcpyDeviceToHost); */
        /* constexpr int field_offset = HS_N3 * 8; */
        /* for (int f = 0; f < opts().n_fields; f++) { */
        /*     if (!energy_only || f == egas_i) { */
        /*         /1* std::copy(U[f].begin(), U[f].end(), unified_uf.begin() + f * */
        /*          * H_N3); *1/ */

        /*         for (int i = 0; i < H_NX; i++) { */
        /*             for (int j = 0; j < H_NX; j++) { */
        /*                 for (int k = 0; k < H_NX; k++) { */
        /*                     const int i0 = (i + H_BW) / 2; */
        /*                     const int j0 = (j + H_BW) / 2; */
        /*                     const int k0 = (k + H_BW) / 2; */
        /*                     const int iii0 = hSindex(i0, j0, k0); */
        /*                     const int iiir = hindex(i, j, k); */
        /*                     if (coarse[iii0 + slice_id * HS_N3]) { */
        /*                         int ir, jr, kr; */
        /*                         if HOST_CONSTEXPR (H_BW % 2 == 0) { */
        /*                             ir = i % 2; */
        /*                             jr = j % 2; */
        /*                             kr = k % 2; */
        /*                         } else { */
        /*                             ir = 1 - (i % 2); */
        /*                             jr = 1 - (j % 2); */
        /*                             kr = 1 - (k % 2); */
        /*                         } */
        /*                         const int oct_index = ir * 4 + jr * 2 + kr; */
        /*                         // unified_u[f * H_N3 + iiir] = unified_uf[f * field_offset + 8 * */
        /*                         // iii0 + oct_index]; */
        /*                         U[f][iiir] = unified_uf[f * field_offset + iii0 + */
        /*                             oct_index * HS_N3 + slice_id * nfields * HS_N3 * 8]; */
        /*                     } */
        /*                 } */
        /*             } */
        /*         } */

        /*         // std::copy(unified_u.begin() + f * H_N3, unified_u.begin() + f * H_N3 + H_N3, */
        /*         // U[f].begin()); */
        /*     } */ 
        /* } */ 

        recycler::cuda_aggregated_device_buffer<double, decltype(alloc_device_double)> device_u(
            max_slices * nfields * H_N3, 0, alloc_device_double);
        std::vector<double, decltype(alloc_host_double)> unified_u(
            max_slices * nfields * H_N3, double{},
            alloc_host_double);

        dim3 const grid_spec_phase2(number_slices, nfields, H_NX);
        dim3 const threads_per_block_phase2(1, H_NX, H_NX);
        bool local_energy_only = energy_only;
        void* args_phase2[] = {
            &(device_coarse.device_side_buffer),
            &(device_uf.device_side_buffer),
            &(device_u.device_side_buffer),
            &nfields, &local_energy_only};
        launch_complete_hydro_amr_boundary_cuda_phase2_post(
            exec_slice, grid_spec_phase2, threads_per_block_phase2, args_phase2);

        auto ret_fut =
            exec_slice.async(cudaMemcpyAsync, unified_u.data(), device_u.device_side_buffer,
                number_slices * (nfields * H_N3) * sizeof(double),
                cudaMemcpyDeviceToHost);
        ret_fut.get();

        constexpr int field_offset = HS_N3 * 8;
        for (int f = 0; f < opts().n_fields; f++) {
            if (!energy_only || f == egas_i) {
                /* std::copy(U[f].begin(), U[f].end(), unified_uf.begin() + f *
                 * H_N3); */

                for (int i = 0; i < H_NX; i++) {
                    for (int j = 0; j < H_NX; j++) {
                        for (int k = 0; k < H_NX; k++) {
                            const int i0 = (i + H_BW) / 2;
                            const int j0 = (j + H_BW) / 2;
                            const int k0 = (k + H_BW) / 2;
                            const int iii0 = hSindex(i0, j0, k0);
                            const int iiir = hindex(i, j, k);
                            if (coarse[iii0 + slice_id * HS_N3]) {
                                U[f][iiir] = unified_u[f * H_N3 + iiir + slice_id * nfields * H_N3];
                            }
                        }
                    }
                }
                // std::copy(unified_u.begin() + f * H_N3, unified_u.begin() + f * H_N3 + H_N3,
                // U[f].begin());
            } 
        } 

    }); 
    ret_fut.get();
}

#endif

void complete_hydro_amr_boundary_cpu(const double dx, const bool energy_only,
    const std::vector<std::vector<real>>& Ushad, const std::vector<int>& is_coarse,
    const std::array<double, NDIM>& xmin, std::vector<std::vector<double>>& U) {
    // std::cout << "Calling hydro cpu version!" << std::endl;

    // std::vector<double, recycler::aggressive_recycle_aligned<double, 32>> unified_u(
    //     opts().n_fields * H_N3);
    std::vector<double, recycler::aggressive_recycle_aligned<double, 32>> unified_ushad(
        opts().n_fields * HS_N3);
    // Create non-atomic copy
    std::vector<int, recycler::aggressive_recycle_aligned<int, 32>> coarse(HS_N3);

    for (int f = 0; f < opts().n_fields; f++) {
        if (!energy_only || f == egas_i) {
            std::copy(
                Ushad[f].begin(), Ushad[f].begin() + HS_N3, unified_ushad.begin() + f * HS_N3);
        }
    }

    for (int i = 0; i < HS_N3; i++) {
        coarse[i] = is_coarse[i];
    }
    // constexpr int field_offset = HS_N3 * 8;

    // Phase 1: From UShad to Uf
    constexpr int uf_max = OCTOTIGER_MAX_NUMBER_FIELDS;
    double uf_local[uf_max * 8];
    for (int i0 = 1; i0 < HS_NX - 1; i0++) {
        for (int j0 = 1; j0 < HS_NX - 1; j0++) {
            for (int k0 = 1; k0 < HS_NX - 1; k0++) {
                const int iii0 = i0 * HS_DNX + j0 * HS_DNY + k0 * HS_DNZ;
                if (coarse[iii0]) {
                    complete_hydro_amr_boundary_inner_loop<double>(dx, energy_only,
                        unified_ushad.data(), coarse.data(), xmin.data(), i0, j0,
                        k0, opts().n_fields, true, 0, iii0, uf_local);
                    int i = 2 * i0 - H_BW;
                    int j = 2 * j0 - H_BW;
                    int k = 2 * k0 - H_BW;
                    int ir = 0;
                    if (i < 0)
                        ir = 1;
                    for (; ir < 2 && i + ir < H_NX; ir++) {
                        int jr = 0;
                        if (j < 0)
                            jr = 1;
                        for (; jr < 2 && j + jr < H_NX; jr++) {
                            int kr = 0;
                            if (k < 0)
                                kr = 1;
                            for (; kr < 2 && k + kr < H_NX; kr++) {
                                const int iiir = hindex(i + ir, j + jr, k + kr);
                                const int oct_index = ir * 4 + jr * 2 + kr;
                                for (int f = 0; f < opts().n_fields; f++) {
                                    if (!energy_only || f == egas_i) {
                                        U[f][iiir] =
                                            uf_local[f * 8 + oct_index];
                                    }
                                }
                            }
                        }
                    }
                } 
            }
        }
    }
}
//#pragma GCC pop_options
