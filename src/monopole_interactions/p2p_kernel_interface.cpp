
#include "octotiger/monopole_interactions/p2p_kernel_interface.hpp"
#include "octotiger/monopole_interactions/legacy/cuda_p2p_interaction_interface.hpp"
#include "octotiger/options.hpp"

#include "octotiger/common_kernel/interactions_iterators.hpp"
#include "octotiger/monopole_interactions/legacy/p2p_interaction_interface.hpp"
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

enum accelerator_kernel_type
{
    OFF,
    DEVICE_CUDA,
    DEVICE_KOKKOS
};
enum host_kernel_type
{
    LEGACY,
    HOST_VC,
    HOST_KOKKOS
};

#ifdef OCTOTIGER_HAVE_KOKKOS
using device_executor = hpx::kokkos::cuda_executor;
//using host_executor = hpx::kokkos::serial_executor;
using host_executor = hpx::kokkos::hpx_executor;
using device_pool_strategy = round_robin_pool<device_executor>;
using executor_interface_t = stream_interface<device_executor, device_pool_strategy>;
#endif

#include "octotiger/monopole_interactions/kernel/kokkos_kernel.hpp"

void p2p_kernel_interface(std::vector<real>& monopoles,
    std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
    std::array<bool, geo::direction::count()>& is_direction_empty, std::shared_ptr<grid> grid_ptr) {
    //accelerator_kernel_type device_type = DEVICE_CUDA;
    //host_kernel_type host_type = HOST_VC;
    accelerator_kernel_type device_type = DEVICE_KOKKOS;
    host_kernel_type host_type = HOST_KOKKOS;

    // Try accelerator implementation
    if (device_type != OFF) {
        if (device_type == DEVICE_KOKKOS) {
#ifdef OCTOTIGER_HAVE_KOKKOS
            bool avail = stream_pool::interface_available<device_executor, device_pool_strategy>(
                opts().cuda_buffer_capacity);
            if (avail) {
                executor_interface_t executor;
                p2p_kernel<device_executor>(executor, monopoles, neighbors, type, dx, opts().theta,
                    is_direction_empty, grid_ptr);
                return;
            }
#else
            std::cerr << "Trying to call P2P Kokkos kernel in a non-kokkos build! Aborting..." << std::endl;
            abort();
#endif
        }
        if (device_type == DEVICE_CUDA) {
            octotiger::fmm::monopole_interactions::cuda_p2p_interaction_interface p2p_interactor{};
            p2p_interactor.set_grid_ptr(grid_ptr);
            p2p_interactor.compute_p2p_interactions(
                monopoles, neighbors, type, dx, is_direction_empty);
            return;
        }
    }    // Nothing is available or device execution is disabled - fallback to host execution

    if (host_type == HOST_KOKKOS) {
#ifdef OCTOTIGER_HAVE_KOKKOS
        host_executor executor{};
        p2p_kernel<host_executor>(
            executor, monopoles, neighbors, type, dx, opts().theta, is_direction_empty, grid_ptr);
        return;
#else
        std::cerr << "Trying to call P2P Kokkos kernel in a non-kokkos build! Aborting..." << std::endl;
        abort();
#endif
    } else {
        octotiger::fmm::monopole_interactions::p2p_interaction_interface p2p_interactor{};
        p2p_interactor.set_grid_ptr(grid_ptr);
        p2p_interactor.compute_p2p_interactions(monopoles, neighbors, type, dx, is_direction_empty);
        return;
    }

    return;
}
