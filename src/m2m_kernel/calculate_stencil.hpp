#pragma once

#include "multiindex.hpp"

#include <vector>

namespace octotiger {
namespace fmm {

    std::vector<multiindex<>> calculate_stencil();

}    // namespace fmm
}    // namespace octotiger
