#pragma once

#include "../common_kernel/multiindex.hpp"

#include <vector>

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        std::pair<std::vector<multiindex<>>, std::vector<std::array<real, 4>>> calculate_stencil();
        std::pair<std::vector<bool>, std::vector<std::array<real, 4>>>
        calculate_stencil_masks(std::vector<multiindex<>> superimposed_stencil);

    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
