//  Copyright (c) 2020-2021 Gregor Daiß
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//
#include "octotiger/geometry.hpp"
#include "octotiger/grid.hpp"

#include <array>
#include <memory>
#include <vector>

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        void multipole_kernel_interface(oct::vector<real>& monopoles, oct::vector<multipole>& M_ptr,
            oct::vector<std::shared_ptr<oct::vector<space_vector>>>& com_ptr,
            oct::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
            oct::array<bool, geo::direction::count()>& is_direction_empty,
            std::array<real, NDIM> xbase, std::shared_ptr<grid> grid, const bool use_root_stencil);

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
