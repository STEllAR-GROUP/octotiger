//  Copyright (c) 2020-2021 Gregor Daiß
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//

#ifdef OCTOTIGER_HAVE_KOKKOS
#include <hpx/kokkos.hpp>
#endif

#include "octotiger/multipole_interactions/multipole_kernel_interface.hpp"
#include "octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp"
#include "octotiger/options.hpp"

#include "octotiger/common_kernel/interactions_iterators.hpp"
#include "octotiger/multipole_interactions/legacy/multipole_interaction_interface.hpp"
#include "octotiger/multipole_interactions/util/calculate_stencil.hpp"
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
static const char module_identifier_multipoles[] = "gravity_solver_multipoles";
/// Dummy variable to ensure the touch_sycl_device_by_running_a_dummy_kernel is being run
const int init_sycl_device_multipoles =
    octotiger::sycl_util::touch_sycl_device_by_running_a_dummy_kernel<
        module_identifier_multipoles>();
#else
#warning "SYCL builds without OCTOTIGER_WITH_INTEL_GPU_WORKAROUND=ON may break on Intel GPUs"
#endif
#endif

#ifdef OCTOTIGER_HAVE_KOKKOS
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

#ifdef OCTOTIGER_MULTIPOLE_HOST_HPX_EXECUTOR
using host_executor = hpx::kokkos::hpx_executor;
#else
using host_executor = hpx::kokkos::serial_executor;
#endif
#endif

#include "octotiger/multipole_interactions/kernel/kokkos_kernel.hpp"

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        void multipole_kernel_interface(std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::array<real, NDIM> xbase, std::shared_ptr<grid> grid, const bool use_root_stencil) {
            interaction_host_kernel_type host_type = opts().multipole_host_kernel_type;
            interaction_device_kernel_type device_type = opts().multipole_device_kernel_type;

            // Try accelerator implementation
            if (device_type != interaction_device_kernel_type::OFF) {
                if (device_type == interaction_device_kernel_type::KOKKOS_CUDA ||
                    device_type == interaction_device_kernel_type::KOKKOS_HIP ||
                    device_type == interaction_device_kernel_type::KOKKOS_SYCL) {
#if defined(OCTOTIGER_HAVE_KOKKOS) && (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL))
                    bool avail = true;
                    size_t device_id =
                        stream_pool::get_next_device_id<device_executor, device_pool_strategy>(opts().number_gpus);
                    if (host_type != interaction_host_kernel_type::DEVICE_ONLY) {
                        // Check where we want to run this:
                        avail =
                            stream_pool::interface_available<device_executor, device_pool_strategy>(
                                opts().max_gpu_executor_queue_length, device_id);
                    }
                    if (avail) {

                        executor_interface_t executor{device_id};
                        multipole_kernel<device_executor>(executor, monopoles, M_ptr, com_ptr,
                            neighbors, type, dx, opts().theta, is_direction_empty, xbase, grid,
                            use_root_stencil, device_id);

                        octotiger::fmm::multipole_kokkos_gpu_subgrids_launched++;
                        return;
                    }
                }
#else
                    std::cerr << "Trying to call multipole Kokkos kernel with no or the wrong kokkos "
                                 "device backend active! Aborting..."
                              << std::endl;
                    abort();
                }
#endif
                if (device_type == interaction_device_kernel_type::CUDA) {
#ifdef OCTOTIGER_HAVE_CUDA
                    cuda_multipole_interaction_interface multipole_interactor{};
                    multipole_interactor.set_grid_ptr(grid);
                    multipole_interactor.compute_multipole_interactions(monopoles, M_ptr, com_ptr,
                        neighbors, type, dx, is_direction_empty, xbase, use_root_stencil);
                    octotiger::fmm::multipole_cuda_gpu_subgrids_launched++;
                    return;
                }
#else
                    std::cerr << "Trying to call multipole CUDA kernel in a non-CUDA build! "
                              << "Aborting..." << std::endl;
                    abort();
                }
#endif
                if (device_type == interaction_device_kernel_type::HIP) {
#ifdef OCTOTIGER_HAVE_HIP
                    cuda_multipole_interaction_interface multipole_interactor{};
                    multipole_interactor.set_grid_ptr(grid);
                    multipole_interactor.compute_multipole_interactions(monopoles, M_ptr, com_ptr,
                        neighbors, type, dx, is_direction_empty, xbase, use_root_stencil);
                    octotiger::fmm::multipole_cuda_gpu_subgrids_launched++;
                    return;
                }
#else
                    std::cerr << "Trying to call multipole HIP kernel in a non-HIP build! "
                              << "Aborting..." << std::endl;
                    abort();
                }
#endif
            }    // Nothing is available or device execution is disabled - fallback to host
                 // execution

            if (host_type == interaction_host_kernel_type::KOKKOS) {
#ifdef OCTOTIGER_HAVE_KOKKOS
                host_executor executor{hpx::kokkos::execution_space_mode::independent};
                multipole_kernel<host_executor>(executor, monopoles, M_ptr, com_ptr, neighbors,
                    type, dx, opts().theta, is_direction_empty, xbase, grid, use_root_stencil, 0);
                octotiger::fmm::multipole_kokkos_cpu_subgrids_launched++;
                return;
#else
                std::cerr
                    << "Trying to call Multipole Kokkos kernel in a non-kokkos build! Aborting..."
                    << std::endl;
                abort();
#endif
            } else {
                multipole_interaction_interface multipole_interactor{};
                multipole_interactor.set_grid_ptr(grid);
                multipole_interactor.compute_multipole_interactions(monopoles, M_ptr, com_ptr,
                    neighbors, type, dx, is_direction_empty, xbase, use_root_stencil);
                return;
            }

            return;
        }

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
