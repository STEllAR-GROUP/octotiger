//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/monopole_interactions/legacy/p2m_interaction_interface.hpp"
#include "octotiger/common_kernel/interactions_iterators.hpp"
#include "octotiger/monopole_interactions/util/calculate_stencil.hpp"
#include "octotiger/options.hpp"
#include "octotiger/real.hpp"
#include "octotiger/monopole_interactions/legacy/p2m_kernel.hpp"

#include <algorithm>
#include <array>
#include <vector>

// Big picture questions:
// - use any kind of tiling?

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {
        std::vector<multiindex<>>& p2m_interaction_interface::stencil() {
            static thread_local std::vector<multiindex<>> stencil_ = calculate_stencil().first;
            return stencil_;
        }

        p2m_interaction_interface::p2m_interaction_interface()
          : neighbor_empty_multipoles(27) {
            this->p2m_type = opts().p2m_kernel_type;
        }

        void p2m_interaction_interface::compute_p2m_interactions(std::vector<real>& monopoles,
            std::vector<multipole>& M_ptr,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type,
            std::array<bool, geo::direction::count()>& is_direction_empty) {
            cpu_expansion_buffer_t local_expansions_staging_area;
            cpu_space_vector_buffer_t center_of_masses_staging_area;

            update_input(M_ptr, com_ptr, neighbors, type, 
                local_expansions_staging_area, center_of_masses_staging_area);
            compute_interactions(type, is_direction_empty, neighbors, 
                local_expansions_staging_area, center_of_masses_staging_area);
        }

        void p2m_interaction_interface::compute_interactions(gsolve_type type,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::vector<neighbor_gravity_type>& all_neighbor_interaction_data,
            const cpu_expansion_buffer_t& local_expansions_staging_area,
            const cpu_space_vector_buffer_t& center_of_masses_staging_area) {
            if (p2m_type == interaction_kernel_type::SOA_CPU) {
                if (multipole_neighbors_exist) {
                    p2m_kernel kernel(neighbor_empty_multipoles);
                    cpu_expansion_result_buffer_t potential_expansions_SoA;
                    cpu_angular_result_t angular_corrections_SoA;
                    kernel.apply_stencil(local_expansions_staging_area,
                        center_of_masses_staging_area, potential_expansions_SoA,
                        angular_corrections_SoA, stencil(), type, x_skip, y_skip, z_skip);
                    potential_expansions_SoA.add_to_non_SoA(grid_ptr->get_L());
                    if (type == RHO) {
                        angular_corrections_SoA.to_non_SoA(grid_ptr->get_L_c());
                    }
                }
            } else {
                // waits for boundary data and then computes boundary interactions
                for (auto const& dir : geo::direction::full_set()) {
                    if (!is_direction_empty[dir]) {
                        neighbor_gravity_type& neighbor_data = all_neighbor_interaction_data[dir];
                        if (!neighbor_data.is_monopole) {
                            grid_ptr->compute_boundary_interactions(type, neighbor_data.direction,
                                neighbor_data.is_monopole, neighbor_data.data);
                        }
                    }
                }
            }
        }
    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
