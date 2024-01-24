//  Copyright (c) 2021-2022 Gregor Daiß
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/* #undef NDEBUG */
#ifdef OCTOTIGER_HAVE_KOKKOS
#include <hpx/kokkos/executors.hpp>
#include <hpx/kokkos.hpp>
#endif
#include "octotiger/unitiger/hydro_impl/hydro_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/hydro_performance_counters.hpp"
#ifdef OCTOTIGER_HAVE_KOKKOS
#include "octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp"
#endif
#if defined(OCTOTIGER_HAVE_KOKKOS)
hpx::once_flag init_hydro_kokkos_pool_flag;
#if defined(KOKKOS_ENABLE_CUDA)
using device_executor = hpx::kokkos::cuda_executor;
using device_pool_strategy = round_robin_pool<device_executor>;
using executor_interface_t = stream_interface<device_executor, device_pool_strategy>;
#elif defined(KOKKOS_ENABLE_HIP)
using device_executor = hpx::kokkos::hip_executor;
using device_pool_strategy = round_robin_pool<device_executor>;
using executor_interface_t = stream_interface<device_executor, device_pool_strategy>;
#elif defined(KOKKOS_ENABLE_SYCL)
using device_executor = hpx::kokkos::sycl_executor;
using device_pool_strategy = round_robin_pool<device_executor>;
using executor_interface_t = stream_interface<device_executor, device_pool_strategy>;
#endif
#ifdef OCTOTIGER_HYDRO_HOST_HPX_EXECUTOR
using host_executor = hpx::kokkos::hpx_executor;
#else
using host_executor = hpx::kokkos::serial_executor;
#endif
void init_hydro_kokkos_aggregation_pool(void) {
    const size_t max_slices = opts().max_kernels_fused;
    constexpr size_t number_aggregation_executors = 128;
    Aggregated_Executor_Modes executor_mode = Aggregated_Executor_Modes::EAGER;
    if (max_slices == 1) {
      executor_mode = Aggregated_Executor_Modes::STRICT;
    }
    if (opts().executors_per_gpu > 0) {
#if defined(KOKKOS_ENABLE_CUDA)
    hydro_kokkos_agg_executor_pool<hpx::kokkos::cuda_executor>::init(number_aggregation_executors, max_slices, executor_mode, opts().number_gpus);
#elif defined(KOKKOS_ENABLE_HIP)
    hydro_kokkos_agg_executor_pool<hpx::kokkos::hip_executor>::init(number_aggregation_executors, max_slices, executor_mode, opts().number_gpus);
#elif defined(KOKKOS_ENABLE_SYCL)
    hydro_kokkos_agg_executor_pool<hpx::kokkos::sycl_executor>::init(number_aggregation_executors, max_slices, executor_mode, opts().number_gpus);
#endif
    }
    hydro_kokkos_agg_executor_pool<host_executor>::init(number_aggregation_executors, 1, Aggregated_Executor_Modes::STRICT, 1);
}
#endif


#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
#include <hpx/async_cuda/cuda_executor.hpp>
using device_executor_cuda = hpx::cuda::experimental::cuda_executor;
using device_pool_strategy_cuda = round_robin_pool<device_executor_cuda>;
using executor_interface_cuda_t = stream_interface<device_executor_cuda, device_pool_strategy_cuda>;
#endif

timestep_t launch_hydro_kernels(hydro_computer<NDIM, INX, physics<NDIM>>& hydro,
    const std::vector<std::vector<safe_real>>& U, std::vector<std::vector<safe_real>>& X,
    const double omega, std::vector<hydro_state_t<std::vector<safe_real>>>& F,
    const interaction_host_kernel_type host_type, const interaction_device_kernel_type device_type,
    const size_t max_gpu_executor_queue_length) {
    static const cell_geometry<NDIM, INX> geo;

    // interaction_host_kernel_type host_type = opts().hydro_host_kernel_type;
    // interaction_device_kernel_type device_type = opts().hydro_device_kernel_type;

    // Timestep default value
    auto max_lambda = timestep_t{};

    // Try accelerator implementation
    if (device_type != interaction_device_kernel_type::OFF) {
        if (device_type == interaction_device_kernel_type::KOKKOS_CUDA ||
            device_type == interaction_device_kernel_type::KOKKOS_HIP ||
            device_type == interaction_device_kernel_type::KOKKOS_SYCL) {
#if defined(OCTOTIGER_HAVE_KOKKOS) && (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)|| defined(KOKKOS_ENABLE_SYCL))
            // Init local kernel pool if not done already
            hpx::call_once(init_hydro_kokkos_pool_flag, init_hydro_kokkos_aggregation_pool);
            bool avail = true; 
            // Host execution is possible: Check if there is a launch slot for device - if not 
            // we will execute the kernel on the CPU instead
            if (host_type != interaction_host_kernel_type::DEVICE_ONLY) {
                size_t device_id =
                    stream_pool::get_next_device_id<device_executor, device_pool_strategy>(opts().number_gpus);
                avail = stream_pool::interface_available<device_executor, device_pool_strategy>(
                    max_gpu_executor_queue_length, device_id);
            }
            if (avail) {
                // executor_interface_t executor;
                max_lambda = launch_hydro_kokkos_kernels<device_executor>(
                    hydro, U, X, omega, opts().n_species, F);
                return max_lambda;
            }
        }
#else
            std::cerr << "Trying to call hydro Kokkos kernel with no or the wrong kokkos device backend active! "
                         "Aborting..."
                      << std::endl;
            abort();
        }
#endif
        if (device_type == interaction_device_kernel_type::CUDA) {
#ifdef OCTOTIGER_HAVE_CUDA
            bool avail = true;
            // Host execution is possible: Check if there is a launch slot for device - if not 
            // we will execute the kernel on the CPU instead
            if (host_type != interaction_host_kernel_type::DEVICE_ONLY) {
                size_t device_id =
                    stream_pool::get_next_device_id<device_executor_cuda, device_pool_strategy_cuda>(opts().number_gpus);
                avail = stream_pool::interface_available<hpx::cuda::experimental::cuda_executor,
                    device_pool_strategy_cuda>(max_gpu_executor_queue_length, device_id);
            }
            if (avail) {
                size_t device_id = 0;
                max_lambda = launch_hydro_cuda_kernels(hydro, U, X, omega, device_id, F);
                return max_lambda;
            }
        }
#else
            std::cerr << "Trying to call Hydro CUDA device kernels in a non-CUDA build! "
                      << "Aborting..." << std::endl;
            abort();
        }
#endif
        if (device_type == interaction_device_kernel_type::HIP) {
#ifdef OCTOTIGER_HAVE_HIP
            bool avail = true;
            if (host_type != interaction_host_kernel_type::DEVICE_ONLY) {
                size_t device_id =
                    stream_pool::get_next_device_id<device_executor_cuda, device_pool_strategy_cuda>(opts().number_gpus);
              avail = stream_pool::interface_available<hpx::cuda::experimental::cuda_executor,
                  device_pool_strategy_cuda>(max_gpu_executor_queue_length, device_id);
            }
            if (avail) {
                size_t device_id = 0;
                max_lambda = launch_hydro_cuda_kernels(hydro, U, X, omega, device_id, F);
                return max_lambda;
            }
        }
#else
            std::cerr << "Trying to call Hydro HIP device kernels in a non-HIP build! "
                      << "Aborting..." << std::endl;
            abort();
        }
#endif
    }

    // Nothing is available or device execution is disabled - fallback to host execution
    if (host_type == interaction_host_kernel_type::KOKKOS) {
#ifdef OCTOTIGER_HAVE_KOKKOS
        hpx::call_once(init_hydro_kokkos_pool_flag, init_hydro_kokkos_aggregation_pool);
        max_lambda = launch_hydro_kokkos_kernels<host_executor>(
            hydro, U, X, omega, opts().n_species, F);
        return max_lambda;
#else
        std::cerr << "Trying to call Hydro Kokkos kernels in a non-kokkos build! Aborting..."
                  << std::endl;
        abort();
#endif
    } else if (host_type == interaction_host_kernel_type::LEGACY) {
        // Legacy implementation
        static thread_local auto f = std::vector<std::vector<std::vector<safe_real>>>(NDIM,
            std::vector<std::vector<safe_real>>(opts().n_fields, std::vector<safe_real>(H_N3)));
#ifdef HPX_HAVE_APEX
        auto reconstruct_timer = apex::start("kernel hydro_reconstruct legacy");
#endif
        const auto q = hydro.reconstruct(U, X, omega);
#ifdef HPX_HAVE_APEX
        apex::stop(reconstruct_timer);
        auto flux_timer = apex::start("kernel hydro_flux legacy");
#endif
        max_lambda = hydro.flux(U, q, f, X, omega);
        octotiger::hydro::hydro_legacy_subgrids_processed++;
#ifdef HPX_HAVE_APEX
        apex::stop(flux_timer);
#endif

    bool output = true;
    char type = 'n';
    static int dump_id = 0;
    /* do { */
    /*     std::cout << "   Print input/output" << std::endl; */
    /*     std::cin >> type; */
    /* } while (!std::cin.fail() && type != 'y' && type != 'n'); */
    if (type == 'y') {
        dump_id++;
        std::string filename_u = std::string("legacy_input_egas_u_") + std::to_string(dump_id);
        std::ofstream uout(filename_u);
        for (int f = 0; f < opts().n_fields; f++) {
            if (f == egas_i) {
                        for (auto u_i = 0; u_i < 14 * 14 * 14; u_i++) {
                            if (u_i % 14 == 0)
                                uout << std::endl;
                            if (u_i % 14 * 14 == 0)
                                uout << std::endl;
                            if (U[f][u_i] > 1e-16) 
                            uout << U[f][u_i] << " ";
                        }
                        uout << std::endl;
            }
        }
        // Verify symmetry in q
        std::string filename_q = std::string("legacy_output_egas_q_") + std::to_string(dump_id);
        std::ofstream qout(filename_q);
        for (int f = 0; f < opts().n_fields; f++) {
            if (f == egas_i) {
                for (int d = 0; d < geo.NDIR; d++) {
                    /* if (d < geo.NDIR / 2) { */
                        qout << "DIR " << d << ":" << std::endl;
                        for (auto x = 2; x < 12; x++) {
                        for (auto y = 2; y < 12; y++) {
                        for (auto z = 2; z < 12; z++) {
                            auto i = x * 14 * 14 + y * 14 + z;
                            auto q_i = (x - 2) * 10 * 10 + (y - 2) * 10 + z - 2;
                            /* const int start_index_flipped = f * q_face_offset + flipped_di *
                             * q_dir_offset; */
                            if (q_i % 10 == 0)
                                qout << std::endl;
                            if (q_i % 100 == 0)
                                qout << std::endl;
                            if (q[f][d][i] > 1e-16) 
                            qout << q[f][d][i] << " ";
                        }
                        }
                        }
                        qout << std::endl;
                    /* } */
                }
            }
        }
    }
        // Use legacy conversion
        for (int dim = 0; dim < NDIM; dim++) {
            for (integer field = 0; field < opts().n_fields; ++field) {
                for (integer i = 0; i <= INX; ++i) {
                    for (integer j = 0; j <= INX; ++j) {
                        for (integer k = 0; k <= INX; ++k) {
                            const auto i0 = findex(i, j, k);
                            F[dim][field][i0] = f[dim][field][hindex(i + H_BW, j + H_BW, k + H_BW)];
                            real rho_tot = 0.0;
                            for (integer field = spc_i; field != spc_i + opts().n_species;
                                 ++field) {
                                rho_tot += F[dim][field][i0];
                            }
                            F[dim][rho_i][i0] = rho_tot;
                        }
                    }
                }
            }
        }
    if (type == 'y') {
        std::string filename_f = std::string("legacy_output_f_") + std::to_string(dump_id);
        std::ofstream fout(filename_f);
        for (int dim = 0; dim < NDIM; dim++) {
            fout << std::endl << "DIM " << dim << std::endl;
            fout << std::endl;
            for (integer field = 0; field < opts().n_fields; ++field) {
                const auto dim_offset = dim * opts().n_fields * q_inx3 + field * q_inx3;
                fout << std::endl << "Field " << field << std::endl;
                fout << std::endl;
                for (integer i = 0; i <= INX; ++i) {
                    fout << std::endl;
                    for (integer j = 0; j <= INX; ++j) {
                        fout << std::endl;
                        for (integer k = 0; k <= INX; ++k) {
                            if (f[dim][field][hindex(i + H_BW, j + H_BW, k + H_BW)] > 1e-16) 
                            fout << f[dim][field][hindex(i + H_BW, j + H_BW, k + H_BW)] << " ";
                        }
                    }
                }
            }
        }
    }
        return max_lambda;
    } else {
        std::cerr << "No valid hydro kernel type given! " << std::endl;
        std::cerr << "Aborting..." << std::endl;
        abort();
    }
    std::cerr << "Invalid state: Could not call any hydro kernel configuration!" << std::endl;
    std::cerr << "Aborting..." << std::endl;
    abort();
    return max_lambda;
}

void convert_x_structure(const hydro::x_type& X, double* const combined_x) {
    constexpr int length_orig = INX + 6;
    constexpr int length_desired = INX + 2;
    auto it_x = combined_x;
    for (size_t dim = 0; dim < NDIM; dim++) {
        auto start_offset = 2 * length_orig * length_orig + 2 * length_orig + 2;
        for (auto ix = 2; ix < 2 + INX + 2; ix++) {
            for (auto iy = 2; iy < 2 + INX + 2; iy++) {
                std::copy(X[dim].begin() + start_offset,
                    X[dim].begin() + start_offset + length_desired, it_x);
                it_x += length_desired;
                start_offset += length_orig;
            }
            start_offset += (2 + 2) * length_orig;
        }
    }
}
