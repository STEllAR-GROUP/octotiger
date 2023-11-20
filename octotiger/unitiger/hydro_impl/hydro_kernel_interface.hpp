//  Copyright (c) 2021-2022 Gregor Daiß
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#pragma once

#include "octotiger/interaction_types.hpp"
#include "octotiger/grid.hpp"
#if defined(OCTOTIGER_HAVE_CUDA) || (defined(OCTOTIGER_HAVE_KOKKOS) && (defined(KOKKOS_ENABLE_CUDA)))
#include <cuda_buffer_util.hpp>
#endif

// Input U, X, omega, executor, device_id
// Output F
timestep_t launch_hydro_kernels(hydro_computer<NDIM, INX, physics<NDIM>>& hydro,
    const std::vector<std::vector<safe_real>>& U, std::vector<std::vector<safe_real>>& X,
    const double omega,
    std::vector<hydro_state_t<std::vector<safe_real>>>& F,
#if defined(OCTOTIGER_HAVE_CUDA) || (defined(OCTOTIGER_HAVE_KOKKOS) && (defined(KOKKOS_ENABLE_CUDA)))
    std::vector<real, recycler::detail::cuda_pinned_allocator<real>>& F_flat,
#else
    std::vector<real>& F_flat,
#endif
    const interaction_host_kernel_type host_type, const interaction_device_kernel_type device_type,
    const size_t max_gpu_executor_queue_length);

#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
timestep_t launch_hydro_cuda_kernels(const hydro_computer<NDIM, INX, physics<NDIM>>& hydro,
    const std::vector<std::vector<safe_real>>& U, const std::vector<std::vector<safe_real>>& X,
    const double omega, const size_t device_id,
#if defined(OCTOTIGER_HAVE_CUDA) || (defined(OCTOTIGER_HAVE_KOKKOS) && (defined(KOKKOS_ENABLE_CUDA)))
    std::vector<real, recycler::detail::cuda_pinned_allocator<real>>& F_flat);
#else
    std::vector<real>& F_flat);
#endif
#endif

// Data conversion functions
void convert_x_structure(const hydro::x_type& X, double * const combined_x);
