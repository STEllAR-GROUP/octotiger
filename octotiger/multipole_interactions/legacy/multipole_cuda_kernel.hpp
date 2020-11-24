//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#ifdef OCTOTIGER_HAVE_CUDA
#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/common_kernel/multiindex.hpp"

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {
        extern __constant__ bool device_stencil_indicator_const[FULL_STENCIL_SIZE];
        extern __constant__ bool device_constant_stencil_masks[FULL_STENCIL_SIZE];
        __host__ void copy_stencil_to_m2m_constant_memory(const float *stencil, const size_t stencil_size);
        __host__ void copy_indicator_to_m2m_constant_memory(const float *indicator, const size_t indicator_size);
        __global__ void cuda_multipole_interactions_kernel_rho(
            const double (&local_monopoles)[NUMBER_LOCAL_MONOPOLE_VALUES],
            const double (&center_of_masses)[NUMBER_MASS_VALUES],
            const double (&multipoles)[NUMBER_LOCAL_EXPANSION_VALUES],
            double (&potential_expansions)[NUMBER_POT_EXPANSIONS],
            double (&angular_corrections)[NUMBER_ANG_CORRECTIONS],
            const double theta, const bool computing_second_half);
        __global__ void cuda_multipole_interactions_kernel_root_rho(
            const double (&center_of_masses)[NUMBER_MASS_VALUES],
            const double (&multipoles)[NUMBER_LOCAL_EXPANSION_VALUES],
            double (&potential_expansions)[NUMBER_POT_EXPANSIONS],
            double (&angular_corrections)[NUMBER_ANG_CORRECTIONS]);
        __global__ void cuda_multipole_interactions_kernel_non_rho(
            const double (&local_monopoles)[NUMBER_LOCAL_MONOPOLE_VALUES],
            const double (&center_of_masses)[NUMBER_MASS_VALUES],
            const double (&multipoles)[NUMBER_LOCAL_EXPANSION_VALUES],
            double (&potential_expansions)[NUMBER_POT_EXPANSIONS],
            const double theta, const bool computing_second_half);
        __global__ void cuda_multipole_interactions_kernel_root_non_rho(
            const double (&center_of_masses)[NUMBER_MASS_VALUES],
            const double (&multipoles)[NUMBER_LOCAL_EXPANSION_VALUES],
            double (&potential_expansions)[NUMBER_POT_EXPANSIONS]);
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
