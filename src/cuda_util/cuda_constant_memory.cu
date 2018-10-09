#ifdef OCTOTIGER_CUDA_ENABLED
#include "cuda_constant_memory.hpp"
namespace octotiger {
namespace fmm {
    // These global, __constant__ arrays will be created in the constant memory of each gpu
    // that is used by octotiger.
    __constant__ octotiger::fmm::multiindex<> device_stencil_const[STENCIL_SIZE];
    __constant__ double device_stencil_indicator_const[STENCIL_SIZE];
    void copy_indicator_to_constant_memory(const double *indicator, const size_t indicator_size) {
        cudaMemcpyToSymbol(device_stencil_indicator_const, indicator, indicator_size);
    }
    void copy_stencil_to_constant_memory(const multiindex<> *stencil, const size_t stencil_size) {
        cudaMemcpyToSymbol(device_stencil_const, stencil, stencil_size);
    }
}    // namespace fmm
}    // namespace octotiger
#endif
