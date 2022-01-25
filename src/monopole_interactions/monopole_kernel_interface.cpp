
#include "octotiger/monopole_interactions/monopole_kernel_interface.hpp"
#include "octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp"
#include "octotiger/options.hpp"

#include "octotiger/common_kernel/interactions_iterators.hpp"
#include "octotiger/monopole_interactions/legacy/monopole_interaction_interface.hpp"
#include "octotiger/monopole_interactions/util/calculate_stencil.hpp"
#include "octotiger/options.hpp"

#include <algorithm>
#include <array>
#include <vector>

#include <aligned_buffer_util.hpp>
#include <buffer_manager.hpp>
#include <stream_manager.hpp>

#ifdef OCTOTIGER_HAVE_KOKKOS
#include <hpx/kokkos.hpp>
#endif

#include "octotiger/options.hpp"

#if defined(OCTOTIGER_HAVE_KOKKOS)
#if defined(KOKKOS_ENABLE_CUDA)
using device_executor = hpx::kokkos::cuda_executor;
using device_pool_strategy = round_robin_pool<device_executor>;
using executor_interface_t = stream_interface<device_executor, device_pool_strategy>;
#elif defined(KOKKOS_ENABLE_HIP)
using device_executor = hpx::kokkos::hip_executor;
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
                    device_type == interaction_device_kernel_type::KOKKOS_HIP) {
#if defined(OCTOTIGER_HAVE_KOKKOS) && (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP))
                    bool avail = true;
                    if (host_type != interaction_host_kernel_type::DEVICE_ONLY) {
                        // Check where we want to run this:
                        avail =
                            stream_pool::interface_available<device_executor, device_pool_strategy>(
                                opts().cuda_buffer_capacity);
                    }
                    // TODO p2m kokkos bug - probably not enough threads for a wavefront
#if defined(KOKKOS_ENABLE_HIP)
                    if (contains_multipole_neighbor) // TODO Add device_only error
                        avail = false;
#endif
                    if (avail) {
                        executor_interface_t executor;
                        monopole_kernel<device_executor>(executor, monopoles, com_ptr, neighbors,
                            type, dx, opts().theta, is_direction_empty, grid_ptr,
                            contains_multipole_neighbor);
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
                host_executor executor(hpx::kokkos::execution_space_mode::independent);
                monopole_kernel<host_executor>(executor, monopoles, com_ptr, neighbors, type, dx,
                    opts().theta, is_direction_empty, grid_ptr, contains_multipole_neighbor);
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
