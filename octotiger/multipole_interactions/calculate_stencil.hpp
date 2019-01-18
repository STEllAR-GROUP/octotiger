#pragma once

#include "octotiger/common_kernel/multiindex.hpp"

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        two_phase_stencil calculate_stencil(void);
        std::pair<std::vector<bool>, std::vector<std::array<real, 4>>>
        calculate_stencil_masks(std::vector<multiindex<>> superimposed_stencil);

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
