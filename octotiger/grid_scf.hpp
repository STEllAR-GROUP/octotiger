//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "octotiger/config/export_definitions.hpp"

OCTOTIGER_EXPORT std::vector<particle> scf_binary_particles(real, real, real, real);

namespace scf_options {
    OCTOTIGER_EXPORT void read_option_file();
}
