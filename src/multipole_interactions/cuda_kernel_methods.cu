#ifdef OCTOTIGER_CUDA_ENABLED
#include "cuda_kernel_methods.hpp"
namespace octotiger {
namespace fmm {
    namespace multipole_interactions {
        __global__ void cuda_multipole_interactions_kernel(void) {
            octotiger::fmm::multiindex<> cell_index_unpadded(threadIdx.x, threadIdx.y, threadIdx.z);
            for (auto i = 0; i < 100; ++i) {
              printf("%i %i %i", threadIdx.x, threadIdx.y, threadIdx.z);
            }

        }
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
