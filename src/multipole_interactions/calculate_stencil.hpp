#pragma once

#include "../common_kernel/multiindex.hpp"

#include <vector>

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        std::vector<multiindex<>> calculate_stencil(bool multipole_interactions);

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
