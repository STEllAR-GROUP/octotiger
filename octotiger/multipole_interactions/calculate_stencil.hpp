//  Copyright (c) 2019 Dominic C Marcello
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "octotiger/common_kernel/multiindex.hpp"
#include "octotiger/config/export_definitions.hpp"

namespace octotiger { namespace fmm { namespace multipole_interactions {

    OCTOTIGER_EXPORT two_phase_stencil calculate_stencil();
    OCTOTIGER_EXPORT std::pair<std::vector<bool>, std::vector<bool>>
    calculate_stencil_masks(two_phase_stencil superimposed_stencil);

}}}
