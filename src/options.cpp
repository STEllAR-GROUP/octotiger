//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/defs.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/real.hpp"
#include "octotiger/common_kernel/interaction_constants.hpp"

#include <boost/program_options.hpp>
#include <hpx/modules/runtime_distributed.hpp>

#include <cmath>
#include <iosfwd>
#include <sstream>
#include <string>
#include <vector>


options& opts() {
	static options opts_;
	return opts_;
}
std::vector<hpx::id_type> options::all_localities = { };
