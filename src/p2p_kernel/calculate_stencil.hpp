#pragma once

#include "multiindex.hpp"

#include <vector>

namespace octotiger {
namespace fmm {
    namespace p2p_kernel {

        std::vector<multiindex<>> calculate_stencil();

    }    // namespace p2p_kernel
}    // namespace fmm
}    // namespace octotiger
