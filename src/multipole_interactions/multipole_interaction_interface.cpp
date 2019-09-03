//  Copyright (c) 2019
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/multipole_interactions/multipole_interaction_interface.hpp"
#include "octotiger/multipole_interactions/calculate_stencil.hpp"
#include "octotiger/multipole_interactions/multipole_cpu_kernel.hpp"

#include "octotiger/common_kernel/interactions_iterators.hpp"

#include "octotiger/options.hpp"

#include <algorithm>
#include <array>
#include <vector>

// Big picture questions:
// - use any kind of tiling?

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {
        size_t& multipole_interaction_interface::cpu_launch_counter()
        {
            static thread_local size_t cpu_launch_counter_ = 0;
            return cpu_launch_counter_;
        }
        size_t& multipole_interaction_interface::cuda_launch_counter()
        {
            static thread_local size_t cuda_launch_counter_ = 0;
            return cuda_launch_counter_;
        }
        size_t& multipole_interaction_interface::cpu_launch_counter_non_rho()
        {
            static thread_local size_t cpu_launch_counter_non_rho_ = 0;
            return cpu_launch_counter_non_rho_;
        }
        size_t& multipole_interaction_interface::cuda_launch_counter_non_rho()
        {
            static thread_local size_t cuda_launch_counter_non_rho_ = 0;
            return cuda_launch_counter_non_rho_;
        }

        two_phase_stencil& multipole_interaction_interface::stencil()
        {
            static thread_local two_phase_stencil stencil_ = calculate_stencil();
            return stencil_;
        }
        thread_local std::vector<real>
            multipole_interaction_interface::local_monopoles_staging_area(EXPANSION_COUNT_PADDED);
        thread_local struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>
        multipole_interaction_interface::local_expansions_staging_area;
        thread_local struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>
        multipole_interaction_interface::center_of_masses_staging_area;
        std::vector<bool>& multipole_interaction_interface::stencil_masks()
        {
            static thread_local std::vector<bool> stencil_masks_ =
                calculate_stencil_masks(
                    multipole_interaction_interface::stencil())
                    .first;
            return stencil_masks_;
        }
        std::vector<bool>& multipole_interaction_interface::inner_stencil_masks()
        {
            static thread_local std::vector<bool> inner_stencil_masks_ =
                calculate_stencil_masks(
                    multipole_interaction_interface::stencil())
                    .second;
            return inner_stencil_masks_;
        }


        multipole_interaction_interface::multipole_interaction_interface() {
            local_monopoles_staging_area = std::vector<real>(ENTRIES);
            this->m2m_type = opts().m2m_kernel_type;
        }

        void multipole_interaction_interface::compute_multipole_interactions(
            std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::array<real, NDIM> xbase) {
            if (type == RHO)
                cpu_launch_counter()++;
            else
                cpu_launch_counter_non_rho()++;
            update_input(monopoles, M_ptr, com_ptr, neighbors, type, dx, xbase,
                local_monopoles_staging_area, local_expansions_staging_area,
                center_of_masses_staging_area);
            compute_interactions(is_direction_empty, neighbors, local_monopoles_staging_area,
                local_expansions_staging_area, center_of_masses_staging_area);
        }

        void multipole_interaction_interface::compute_interactions(
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::vector<neighbor_gravity_type>& all_neighbor_interaction_data,
            const std::vector<real>& local_monopoles,
            const struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>&
                local_expansions_SoA,
            const struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                center_of_masses_SoA) {
            if (m2m_type == interaction_kernel_type::SOA_CPU) {
                struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>
                    potential_expansions_SoA;
                struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>
                    angular_corrections_SoA;

                multipole_cpu_kernel kernel;
                kernel.apply_stencil_non_blocked(local_expansions_SoA,
                    center_of_masses_SoA, potential_expansions_SoA,
                    angular_corrections_SoA, local_monopoles, stencil_masks(),
                    inner_stencil_masks(), type);

                if (type == RHO) {
                    angular_corrections_SoA.to_non_SoA(grid_ptr->get_L_c());
                }
                potential_expansions_SoA.add_to_non_SoA(grid_ptr->get_L());
            } else {
                // old-style interaction calculation
                // computes inner interactions
                grid_ptr->compute_interactions(type);
                // computes boundary interactions
                for (auto const& dir : geo::direction::full_set()) {
                    if (!is_direction_empty[dir]) {
                        neighbor_gravity_type& neighbor_data = all_neighbor_interaction_data[dir];
                        grid_ptr->compute_boundary_interactions(type, neighbor_data.direction,
                            neighbor_data.is_monopole, neighbor_data.data);
                    }
                }
            }
        }
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
