#pragma once

#include "octotiger/common_kernel/multiindex.hpp"
#include "octotiger/config/export_definitions.hpp"

#include "octotiger/real.hpp"

#include <array>
#include <utility>
#include <vector>

namespace octotiger { namespace fmm { namespace monopole_interactions {
    OCTOTIGER_EXPORT
    std::pair<std::vector<multiindex<>>, std::vector<std::array<real, 4>>>
    calculate_stencil();

    OCTOTIGER_EXPORT
    std::pair<std::vector<bool>, std::vector<std::array<real, 4>>>
    calculate_stencil_masks(std::vector<multiindex<>> superimposed_stencil);

}}}
