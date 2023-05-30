//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef MESA_STAR_HPP_
#define MESA_STAR_HPP_


#include "octotiger/config/export_definitions.hpp"
#include "octotiger/real.hpp"
#include "octotiger/defs.hpp"
#include <string>
#include <functional>
#include <vector>

std::function<double(double)> build_rho_of_h_from_mesa(
		const std::string& filename);

OCTOTIGER_EXPORT std::vector<real> mesa_star(real, real, real, real);

OCTOTIGER_EXPORT bool mesa_refine_test(integer level, integer maxl, real, real, real,
    std::vector<real> const& U,
    std::array<std::vector<real>, NDIM> const& dudx);

//static mesa_profiles(
//                const std::string& filename);//, std::vector<double>& P, std::vector<double>& rho, std::vector<double>& r, std::vector<double>& omega);

#endif /* MESA_STAR_HPP_ */
