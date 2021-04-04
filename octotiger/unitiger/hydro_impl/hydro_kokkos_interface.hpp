#pragma once

#ifdef OCTOTIGER_HAVE_KOKKOS
#include "octotiger/grid.hpp"
#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp"

#include "octotiger/common_kernel/kokkos_util.hpp"

// Input U, X, omega, executor, device_id
// Output F
template<typename executor_t>
timestep_t launch_hydro_kokkos_kernels(const hydro_computer<NDIM, INX, physics<NDIM>>& hydro,
    const std::vector<std::vector<safe_real>>& U, const std::vector<std::vector<safe_real>>& X,
    const double omega, const size_t device_id,
    executor_t& executor,
    std::vector<hydro_state_t<std::vector<safe_real>>> &F) {
    static const cell_geometry<NDIM, INX> geo;

    host_buffer<double> host_monopoles(10);

    // Device buffers
    return timestep_t{};
}
#endif
