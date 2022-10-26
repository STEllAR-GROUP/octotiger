//  Copyright (c) 2021-2022 Gregor Daiß
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#pragma once

#include "octotiger/interaction_types.hpp"
#include "octotiger/grid.hpp"

// Input U, X, omega, executor, device_id
// Output F
timestep_t launch_hydro_kernels(hydro_computer<NDIM, INX, physics<NDIM>>& hydro,
    const oct::vector<oct::vector<safe_real>>& U, oct::vector<oct::vector<safe_real>>& X,
    const double omega, oct::vector<hydro_state_t<oct::vector<safe_real>>>& F,
    const interaction_host_kernel_type host_type, const interaction_device_kernel_type device_type,
    const size_t cuda_buffer_capacity);

#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
timestep_t launch_hydro_cuda_kernels(const hydro_computer<NDIM, INX, physics<NDIM>>& hydro,
    const oct::vector<oct::vector<safe_real>>& U, const oct::vector<oct::vector<safe_real>>& X,
    const double omega, const size_t device_id,
    oct::vector<hydro_state_t<oct::vector<safe_real>>> &F);
#endif

// Data conversion functions
void convert_x_structure(const hydro::x_type& X, double * const combined_x);
