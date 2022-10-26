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
    namespace monopole_interactions {

        void monopole_kernel_interface(oct::vector<real>& monopoles,
            oct::vector<std::shared_ptr<oct::vector<space_vector>>>& com_ptr,
            oct::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
            oct::array<bool, geo::direction::count()>& is_direction_empty,
            std::shared_ptr<grid>& grid, const bool contains_multipole_neighbor);

    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
