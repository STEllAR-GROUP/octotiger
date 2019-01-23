#pragma once
#ifdef OCTOTIGER_HAVE_CUDA
#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/common_kernel/multiindex.hpp"

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {
        extern __constant__ float device_stencil_masks[FULL_STENCIL_SIZE];
        extern __constant__ double device_four_constants[4 * FULL_STENCIL_SIZE];
        __host__ void copy_stencil_to_p2p_constant_memory(const float *stencil_masks, const size_t full_stencil_size);
        __host__  void copy_constants_to_p2p_constant_memory(const double *constants, const size_t constants_size);
        __global__ void cuda_p2p_interactions_kernel(
            const double (&local_monopoles)[NUMBER_LOCAL_MONOPOLE_VALUES],
            double (&potential_expansions)[NUMBER_POT_EXPANSIONS_SMALL],
            const double theta, const double dx);
        __global__ void cuda_add_pot_blocks(
            double (&potential_expansions)[3 * NUMBER_POT_EXPANSIONS_SMALL]);
    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
