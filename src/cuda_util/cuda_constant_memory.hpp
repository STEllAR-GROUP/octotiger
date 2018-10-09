#ifdef OCTOTIGER_CUDA_ENABLED
#ifndef DEFINED_CONSTANT_CUDA_ARRAYS
#define DEFINED_CONSTANT_CUDA_ARRAYS
#include "../common_kernel/interaction_constants.hpp"
#include "../common_kernel/multiindex.hpp"
namespace octotiger {
namespace fmm {
    void copy_indicator_to_constant_memory(const double *indicator, const size_t indicator_size);
    void copy_stencil_to_constant_memory(const multiindex<> *stencil, const size_t stencil_size);
    void copy_constants_to_constant_memory(const double *constants, const size_t constants_size);
}    // namespace fmm
}    // namespace octotiger
#endif
#endif
