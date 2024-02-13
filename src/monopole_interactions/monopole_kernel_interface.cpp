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
#include <CL/sycl.hpp>
// We encounter segfaults on Intel GPUs when running the normal kernels for the first time after
// the program starts. This seems to be some initialization issue as we can simply fix it by
// (non-concurrently) run simple dummy kernel first right after starting octotiger
// (presumably initializes something within the intel gpu runtime).
// Curiousely we have to do this not once per program, but once per lib (octolib and hydrolib).
//
// Somewhat of an ugly workaround but it does the trick and allows us to target Intel GPUs as
// Octo-Tiger runs as expected after applying this workaround.

// TODO(daissgr) Check again in the future to see if the runtime has matured and we don't need this anymore. 
// (last check was 02/2024)

/// Utility function working around segfault on Intel GPU. Initializes something within the runtime by runnning
///a dummy kernel
int touch_sycl_device_by_running_a_dummy_kernel(void) {
    try {
        cl::sycl::queue q(cl::sycl::default_selector_v, cl::sycl::property::queue::in_order{});
        cl::sycl::event my_kernel_event = q.submit(
            [&](cl::sycl::handler& h) {
                h.parallel_for(512, [=](auto i) {});
            },
            cl::sycl::detail::code_location{});
        my_kernel_event.wait();
    } catch (sycl::exception const& e) {
        std::cerr << "(NON-FATAL) ERROR: Caught sycl::exception during SYCL dummy kernel!\n";
        std::cerr << " {what}: " << e.what() << "\n ";
        std::cerr << "Continuing for now as error only occured in the dummy kernel...\n";
        return 2;

    }
    return 1;
}
/// Dummy variable to ensure the touch_sycl_device_by_running_a_dummy_kernel is being run
const int init_sycl_device = touch_sycl_device_by_running_a_dummy_kernel();
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
#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
                    if (contains_multipole_neighbor) // TODO Add device_only warning
                        avail = false;
#elif (defined(KOKKOS_ENABLE_CUDA) && defined(__clang__) )
                    /* if (contains_multipole_neighbor && opts().detected_intel_compiler) // TODO Add device_only warning */
                    /*     avail = false; */
#endif
                    if (avail) {
                        executor_interface_t executor{device_id};
                        monopole_kernel<device_executor>(executor, monopoles, com_ptr, neighbors,
                            type, dx, opts().theta, is_direction_empty, grid_ptr,
                            contains_multipole_neighbor, device_id);
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
