
#include "octotiger/multipole_interactions/multipole_kernel_interface.hpp"
#include "octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp"
#include "octotiger/options.hpp"

#include "octotiger/common_kernel/interactions_iterators.hpp"
#include "octotiger/multipole_interactions/legacy/multipole_interaction_interface.hpp"
#include "octotiger/multipole_interactions/util/calculate_stencil.hpp"
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

#include "octotiger/multipole_interactions/kernel/kokkos_kernel.hpp"

void multipole_kernel_interface(std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
    std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
    std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
    std::array<bool, geo::direction::count()>& is_direction_empty, std::array<real, NDIM> xbase,
    std::shared_ptr<grid> grid, const bool use_root_stencil) {
    accelerator_kernel_type device_type = DEVICE_CUDA;
    host_kernel_type host_type = HOST_VC;
    //accelerator_kernel_type device_type = DEVICE_KOKKOS;
    //host_kernel_type host_type = HOST_KOKKOS;

    // Try accelerator implementation
    if (device_type != OFF) {
        if (device_type == DEVICE_KOKKOS) {
#ifdef OCTOTIGER_HAVE_KOKKOS
            bool avail = stream_pool::interface_available<device_executor, device_pool_strategy>(
                opts().cuda_buffer_capacity);
            if (avail) {
                executor_interface_t executor;
                multipole_kernel<device_executor>(executor, monopoles, M_ptr, com_ptr,
                neighbors, type, dx, opts().theta, is_direction_empty, xbase, grid, use_root_stencil);
                return;
            }
#else
            std::cerr << "Trying to call Multipole Kokkos kernel in a non-kokkos build! Aborting..." << std::endl;
            abort();
#endif
        }
        if (device_type == DEVICE_CUDA) {

            octotiger::fmm::multipole_interactions::cuda_multipole_interaction_interface
                multipole_interactor{};
            multipole_interactor.set_grid_ptr(grid);
            multipole_interactor.compute_multipole_interactions(monopoles, M_ptr, com_ptr,
                neighbors, type, dx, is_direction_empty, xbase, use_root_stencil);
            return;
        }
    }    // Nothing is available or device execution is disabled - fallback to host execution

    if (host_type == HOST_KOKKOS) {
#ifdef OCTOTIGER_HAVE_KOKKOS
        host_executor executor{};
        multipole_kernel<host_executor>(executor, monopoles, M_ptr, com_ptr,
        neighbors, type, dx, opts().theta, is_direction_empty, xbase, grid, use_root_stencil);
        return;
#else
        std::cerr << "Trying to call Multipole Kokkos kernel in a non-kokkos build! Aborting..." << std::endl;
        abort();
#endif
    } else {
        octotiger::fmm::multipole_interactions::multipole_interaction_interface
            multipole_interactor{};
        multipole_interactor.set_grid_ptr(grid);
        multipole_interactor.compute_multipole_interactions(monopoles, M_ptr, com_ptr,
                neighbors, type, dx, is_direction_empty, xbase, use_root_stencil);
        return;
    }

    return;
}
