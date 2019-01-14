#pragma once
#ifdef OCTOTIGER_WITH_CUDA
#include "../common_kernel/interaction_constants.hpp"
#include "../common_kernel/multiindex.hpp"

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {
        __global__ void cuda_multipole_interactions_kernel_rho(
            const double (&local_monopoles)[NUMBER_LOCAL_MONOPOLE_VALUES],
            const double (&center_of_masses)[NUMBER_MASS_VALUES],
            const double (&multipoles)[NUMBER_LOCAL_EXPANSION_VALUES],
            double (&potential_expansions)[NUMBER_POT_EXPANSIONS],
            double (&angular_corrections)[NUMBER_ANG_CORRECTIONS],
            const octotiger::fmm::multiindex<> (&stencil)[STENCIL_SIZE],
            const double (&stencil_phases)[STENCIL_SIZE], const double theta);
        __global__ void cuda_multipole_interactions_kernel_non_rho(
            const double (&local_monopoles)[NUMBER_LOCAL_MONOPOLE_VALUES],
            const double (&center_of_masses)[NUMBER_MASS_VALUES],
            const double (&multipoles)[NUMBER_LOCAL_EXPANSION_VALUES],
            double (&potential_expansions)[NUMBER_POT_EXPANSIONS],
            const octotiger::fmm::multiindex<> (&stencil)[STENCIL_SIZE],
            const double (&stencil_phases)[STENCIL_SIZE], const double theta);
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
