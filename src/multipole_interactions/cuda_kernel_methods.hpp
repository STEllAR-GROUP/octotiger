#pragma once
#ifdef OCTOTIGER_CUDA_ENABLED
#include "../common_kernel/interaction_constants.hpp"
#include "../common_kernel/multiindex.hpp"

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {
        __global__ void cuda_multipole_interactions_kernel(
            double (&local_monopoles)[NUMBER_LOCAL_MONOPOLE_VALUES],
            double (&center_of_masses)[NUMBER_MASS_VALUES],
            double (&multipoles)[NUMBER_LOCAL_EXPANSION_VALUES],
            double (&potential_expansions)[NUMBER_POT_EXPANSIONS],
            double (&angular_corrections)[NUMBER_ANG_CORRECTIONS],
            octotiger::fmm::multiindex<> (&stencil)[STENCIL_SIZE],
            bool (&stencil_phases)[STENCIL_SIZE], double (&factor_half)[20],
            double (&factor_sixth)[20], double theta);
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
