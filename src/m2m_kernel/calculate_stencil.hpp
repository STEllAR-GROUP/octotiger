#pragma once

#include "../common_kernel/multiindex.hpp"

#include <vector>

namespace octotiger {
namespace fmm {

    std::vector<multiindex<>> calculate_stencil(bool multipole_interactions);

}    // namespace fmm
}    // namespace octotiger
