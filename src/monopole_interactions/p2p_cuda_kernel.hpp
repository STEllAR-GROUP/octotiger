#pragma once
#ifdef OCTOTIGER_CUDA_ENABLED
#include "../common_kernel/interaction_constants.hpp"
#include "../common_kernel/multiindex.hpp"

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {
        __global__ void cuda_p2p_interactions_kernel(
            const double (&local_monopoles)[NUMBER_LOCAL_MONOPOLE_VALUES],
            double (&potential_expansions)[NUMBER_POT_EXPANSIONS],
            const octotiger::fmm::multiindex<> (&stencil)[STENCIL_SIZE],
            const double (&four)[4 * STENCIL_SIZE],
            const double theta, const double dx);
    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
