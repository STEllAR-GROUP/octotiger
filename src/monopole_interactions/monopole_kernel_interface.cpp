//  Copyright (c) 2020-2021 Gregor Daiß
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//

#ifdef OCTOTIGER_HAVE_KOKKOS
#include <hpx/kokkos.hpp>
#endif

#include "octotiger/monopole_interactions/monopole_kernel_interface.hpp"
#include "octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp"
#include "octotiger/options.hpp"

#include "octotiger/common_kernel/interactions_iterators.hpp"
#include "octotiger/monopole_interactions/legacy/monopole_interaction_interface.hpp"
#include "octotiger/monopole_interactions/util/calculate_stencil.hpp"
#include "octotiger/common_kernel/gravity_performance_counters.hpp"
#include "octotiger/options.hpp"

#include <algorithm>
#include <array>
#include <vector>

#include <aligned_buffer_util.hpp>
#include <buffer_manager.hpp>
#include <stream_manager.hpp>

#include "octotiger/options.hpp"

#if defined(OCTOTIGER_HAVE_KOKKOS) && defined(KOKKOS_ENABLE_SYCL)
#if defined(OCTOTIGER_HAVE_INTEL_GPU_WORKAROUND)
#include "octotiger/sycl_initialization_guard.hpp"
static const char module_identifier_monopoles[] = "gravity_solver_monopoles";
/// Dummy variable to ensure the touch_sycl_device_by_running_a_dummy_kernel is being run
const int init_sycl_device_monopoles =
    octotiger::sycl_util::touch_sycl_device_by_running_a_dummy_kernel<module_identifier_monopoles>();
#else
#pragma message "SYCL builds without OCTOTIGER_WITH_INTEL_GPU_WORKAROUND=ON may break on Intel GPUs"
#endif
#endif

#if defined(OCTOTIGER_HAVE_KOKKOS)
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
#ifdef OCTOTIGER_MONOPOLE_HOST_HPX_EXECUTOR
using host_executor = hpx::kokkos::hpx_executor;
#else
using host_executor = hpx::kokkos::serial_executor;
#endif
#endif


#include "octotiger/monopole_interactions/kernel/kokkos_kernel.hpp"
#if defined(OCTOTIGER_HAVE_KOKKOS)
hpx::once_flag init_monopole_kokkos_pool_flag;
void init_monopole_kokkos_aggregation_pool(void) {
    const size_t max_slices = opts().max_kernels_fused;
    constexpr size_t number_aggregation_executors = 128;
    Aggregated_Executor_Modes executor_mode = Aggregated_Executor_Modes::EAGER;
    if (max_slices == 1) {
      executor_mode = Aggregated_Executor_Modes::STRICT;
    }
    if (opts().executors_per_gpu > 0) {
#if defined(KOKKOS_ENABLE_CUDA)
        monopole_kokkos_agg_executor_pool<hpx::kokkos::cuda_executor>::init(
            number_aggregation_executors, max_slices, executor_mode, opts().number_gpus);
#elif defined(KOKKOS_ENABLE_HIP)
        monopole_kokkos_agg_executor_pool<hpx::kokkos::hip_executor>::init(
            number_aggregation_executors, max_slices, executor_mode, opts().number_gpus);
#elif defined(KOKKOS_ENABLE_SYCL)
        monopole_kokkos_agg_executor_pool<hpx::kokkos::sycl_executor>::init(
            number_aggregation_executors, max_slices, executor_mode, opts().number_gpus);
#endif
    }
    monopole_kokkos_agg_executor_pool<host_executor>::init(
        number_aggregation_executors, 1, Aggregated_Executor_Modes::STRICT, 1);
}
#endif

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        void monopole_kernel_interface(std::vector<real>& monopoles,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::shared_ptr<grid>& grid_ptr, const bool contains_multipole_neighbor) {
            interaction_host_kernel_type host_type = opts().monopole_host_kernel_type;
            interaction_device_kernel_type device_type = opts().monopole_device_kernel_type;

            // Try accelerator implementation
            if (device_type != interaction_device_kernel_type::OFF) {
                if (device_type == interaction_device_kernel_type::KOKKOS_CUDA ||
                    device_type == interaction_device_kernel_type::KOKKOS_HIP ||
                    device_type == interaction_device_kernel_type::KOKKOS_SYCL) {
#if defined(OCTOTIGER_HAVE_KOKKOS) && (defined(KOKKOS_ENABLE_CUDA) || \
    defined(KOKKOS_ENABLE_HIP)|| defined(KOKKOS_ENABLE_SYCL))
                    bool avail = true;
                    size_t device_id =
                        stream_pool::get_next_device_id<device_executor, device_pool_strategy>(opts().number_gpus);
                    if (host_type != interaction_host_kernel_type::DEVICE_ONLY) {
                        // Check where we want to run this:
                        avail =
                            stream_pool::interface_available<device_executor, device_pool_strategy>(
                                opts().max_gpu_executor_queue_length, device_id);
                    }
                    // TODO p2m kokkos bug - probably not enough threads for a wavefront
                    // TODO how to identify the intel sycl compile here?
//#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
                    if (contains_multipole_neighbor) // TODO Add device_only warning
                        avail = false;
//#elif (defined(KOKKOS_ENABLE_CUDA) && defined(__clang__) )
                    /* if (contains_multipole_neighbor && opts().detected_intel_compiler) // TODO Add device_only warning */
                    /*     avail = false; */
//#endif
                    if (avail) {
                        if (contains_multipole_neighbor) {
                            executor_interface_t executor{device_id};
                            monopole_kernel<device_executor>(executor, monopoles, com_ptr,
                                neighbors, type, dx, opts().theta, is_direction_empty, grid_ptr,
                                contains_multipole_neighbor, device_id);
                        } else {
                            hpx::call_once(init_monopole_kokkos_pool_flag, init_monopole_kokkos_aggregation_pool);
                            monopole_kernel_agg<device_executor>(monopoles, com_ptr, neighbors,
                                type, dx, opts().theta, is_direction_empty, grid_ptr,
                                contains_multipole_neighbor, device_id);
                        }
                        p2p_kokkos_gpu_subgrids_launched++;

                        return;
                    }
                }
#else
                    std::cerr << "Trying to call P2P Kokkos kernel with no or the wrong kokkos device "
                                 "backend active! Aborting..."
                              << std::endl;
                    abort();
                }
#endif
                if (device_type == interaction_device_kernel_type::CUDA) {
#ifdef OCTOTIGER_HAVE_CUDA
                    cuda_monopole_interaction_interface monopole_interactor{};
                    monopole_interactor.compute_interactions(monopoles, com_ptr, neighbors, type,
                        dx, is_direction_empty, grid_ptr, contains_multipole_neighbor);
                    p2p_cuda_gpu_subgrids_launched++;
                    return;
                }
#else
                    std::cerr << "Trying to call P2P CUDA kernel in a non-CUDA build! "
                              << "Aborting..." << std::endl;
                    abort();
                }
#endif
                if (device_type == interaction_device_kernel_type::HIP) {
#ifdef OCTOTIGER_HAVE_HIP
                    cuda_monopole_interaction_interface monopole_interactor{};
                    monopole_interactor.compute_interactions(monopoles, com_ptr, neighbors, type,
                        dx, is_direction_empty, grid_ptr, contains_multipole_neighbor);
                    p2p_cuda_gpu_subgrids_launched++;
                    return;
                }
#else
                    std::cerr << "Trying to call P2P HIP kernel in a non-HIP build! "
                              << "Aborting..." << std::endl;
                    abort();
                }
#endif
            }    // Nothing is available or device execution is disabled - fallback to host
                 // execution

            if (host_type == interaction_host_kernel_type::KOKKOS) {
#ifdef OCTOTIGER_HAVE_KOKKOS
                host_executor executor{hpx::kokkos::execution_space_mode::independent};
                monopole_kernel<host_executor>(executor, monopoles, com_ptr, neighbors, type, dx,
                    opts().theta, is_direction_empty, grid_ptr, contains_multipole_neighbor, 0);
                p2p_kokkos_cpu_subgrids_launched++;
                return;
#else
                std::cerr << "Trying to call P2P Kokkos kernel in a non-kokkos build! Aborting..."
                          << std::endl;
                abort();
#endif
            } else {
                monopole_interaction_interface monopole_interactor{};
                monopole_interactor.compute_interactions(monopoles, com_ptr, neighbors, type, dx,
                    is_direction_empty, grid_ptr, contains_multipole_neighbor);
                return;
            }

            return;
        }

    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
