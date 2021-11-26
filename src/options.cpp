//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/defs.hpp"
#include "octotiger/options.hpp"

#include <boost/program_options.hpp>

#include <vector>


options& opts() {
	static options opts_;
	return opts_;
}
std::vector<hpx::id_type> options::all_localities = { };
