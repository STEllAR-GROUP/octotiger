#ifdef OCTOTIGER_CUDA_ENABLED
#include "cuda_multipole_interaction_interface.hpp"
namespace octotiger {
namespace fmm {
    namespace multipole_interactions {
        thread_local util::cuda_helper cuda_multipole_interaction_interface::gpu_interface;

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
