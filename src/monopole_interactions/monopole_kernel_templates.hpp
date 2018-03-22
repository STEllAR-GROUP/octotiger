#pragma once
#include "../common_kernel/interaction_constants.hpp"
#include "../cuda_util/cuda_global_def.hpp"
namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        template <typename T>
        CUDA_CALLABLE_METHOD inline void compute_monopole_interaction(const T& monopole,
            T (&tmpstore)[4], const T (&four)[4], const T (&d_components)[2]) noexcept {
            tmpstore[0] = tmpstore[0] + four[0] * monopole * d_components[0];
            tmpstore[1] = tmpstore[1] + four[1] * monopole * d_components[1];
            tmpstore[2] = tmpstore[2] + four[2] * monopole * d_components[1];
            tmpstore[3] = tmpstore[3] + four[3] * monopole * d_components[1];
        }
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
