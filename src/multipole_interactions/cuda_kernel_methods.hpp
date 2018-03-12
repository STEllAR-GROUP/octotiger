#pragma once
#ifdef OCTOTIGER_CUDA_ENABLED
#include "compute_kernel_templates.hpp"
#include "../common_kernel/multiindex.hpp"

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {
        __global__ void cuda_multipole_interactions_kernel(double* center_of_masses,
            double* multipoles, double* potential_expansions, double* angular_corrections,
            octotiger::fmm::multiindex<> stencil, bool* stencil_phases,
            double* factor_half, double* factor_sixth);
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
