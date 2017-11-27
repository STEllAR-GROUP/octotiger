#pragma once

#include "../common_kernel/multiindex.hpp"

#include <vector>

namespace octotiger {
namespace fmm {
    namespace m2p_kernel {

        std::vector<multiindex<>> calculate_stencil();

    }    // namespace p2p_kernel
}    // namespace fmm
}    // namespace octotiger
