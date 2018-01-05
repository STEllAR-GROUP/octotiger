#pragma once

#include "../common_kernel/multiindex.hpp"

#include <vector>

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        std::pair<std::vector<multiindex<>>, std::vector<std::array<real, 4>>> calculate_stencil();

    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
