#pragma once

#include "../common_kernel/multiindex.hpp"

#include <vector>

namespace octotiger {
namespace fmm {
    namespace p2p_kernel {

        std::pair<std::vector<multiindex<>>, std::vector<std::array<real, 4>>> calculate_stencil();

    }    // namespace p2p_kernel
}    // namespace fmm
}    // namespace octotiger
