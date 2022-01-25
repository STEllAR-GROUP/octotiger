#pragma once

#include "octotiger/interaction_types.hpp"
#include "octotiger/grid.hpp"

// Input U, X, omega, executor, device_id
// Output F
timestep_t launch_hydro_kernels(hydro_computer<NDIM, INX, physics<NDIM>>& hydro,
    const std::vector<std::vector<safe_real>>& U, std::vector<std::vector<safe_real>>& X,
    const double omega, std::vector<hydro_state_t<std::vector<safe_real>>>& F,
    const interaction_host_kernel_type host_type, const interaction_device_kernel_type device_type,
    const size_t cuda_buffer_capacity);

#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
timestep_t launch_hydro_cuda_kernels(const hydro_computer<NDIM, INX, physics<NDIM>>& hydro,
    const std::vector<std::vector<safe_real>>& U, const std::vector<std::vector<safe_real>>& X,
    const double omega, const size_t device_id,
    stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
    std::vector<hydro_state_t<std::vector<safe_real>>> &F);
#endif

// Data conversion functions
void convert_x_structure(const hydro::x_type& X, double * const combined_x);
