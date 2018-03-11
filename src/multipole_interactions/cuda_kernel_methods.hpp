#pragma once
#ifdef OCTOTIGER_CUDA_ENABLED
#include "compute_kernel_templates.hpp"
#include "../common_kernel/multiindex.hpp"

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {
        __global__ void cuda_multipole_interactions_kernel(void);
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
