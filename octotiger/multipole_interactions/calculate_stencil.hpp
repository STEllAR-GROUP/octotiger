#pragma once

#include "octotiger/config/export_definitions.hpp"
#include "octotiger/common_kernel/multiindex.hpp"

namespace octotiger { namespace fmm { namespace multipole_interactions {

    OCTOTIGER_EXPORT two_phase_stencil calculate_stencil(void);
    OCTOTIGER_EXPORT std::pair<std::vector<bool>, std::vector<bool>>
    calculate_stencil_masks(two_phase_stencil superimposed_stencil);

}}}
