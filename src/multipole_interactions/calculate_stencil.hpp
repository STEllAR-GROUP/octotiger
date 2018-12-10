#pragma once

#include "../common_kernel/multiindex.hpp"

#include <vector>

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        two_phase_stencil calculate_stencil(void);
        std::pair<std::vector<bool>, std::vector<bool>> calculate_stencil_masks(two_phase_stencil superimposed_stencil);

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
