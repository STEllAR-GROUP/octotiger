//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/monopole_interactions/legacy/p2p_interaction_interface.hpp"
#include "octotiger/monopole_interactions/legacy/p2m_interaction_interface.hpp"
#include "octotiger/common_kernel/interactions_iterators.hpp"
#include "octotiger/monopole_interactions/util/calculate_stencil.hpp"
#include "octotiger/monopole_interactions/legacy/p2p_cpu_kernel.hpp"    //VC ?
#include "octotiger/options.hpp"

#include <algorithm>
#include <array>
#include <vector>

#include <aligned_buffer_util.hpp>
#include <buffer_manager.hpp>

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {
        size_t& p2p_interaction_interface::cpu_launch_counter() {
            static thread_local size_t cpu_launch_counter_ = 0;
            return cpu_launch_counter_;
        }
        size_t& p2p_interaction_interface::cuda_launch_counter() {
            static thread_local size_t cuda_launch_counter_ = 0;
            return cuda_launch_counter_;
        }

        std::vector<multiindex<>>& p2p_interaction_interface::stencil() {
            static thread_local std::vector<multiindex<>> stencil_ = calculate_stencil().first;
            return stencil_;
        }

        std::vector<bool>& p2p_interaction_interface::stencil_masks() {
            static thread_local std::vector<bool> stencil_masks_ =
                calculate_stencil_masks(p2p_interaction_interface::stencil()).first;
            return stencil_masks_;
        }
        std::vector<std::array<real, 4>>& p2p_interaction_interface::four() {
            static thread_local std::vector<std::array<real, 4>> four_ = calculate_stencil().second;
            return four_;
        }
        std::vector<std::array<real, 4>>& p2p_interaction_interface::stencil_four_constants() {
            static thread_local std::vector<std::array<real, 4>> stencil_four_constants_ =
                calculate_stencil_masks(p2p_interaction_interface::stencil()).second;
            return stencil_four_constants_;
        }

        p2p_interaction_interface::p2p_interaction_interface()
          : neighbor_empty_monopoles(27) {
            this->p2p_type = opts().p2p_kernel_type;
        }

        void p2p_interaction_interface::compute_p2p_interactions(const std::vector<real>& monopoles,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type,
            real dx,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::shared_ptr<grid>& grid_ptr,
            const bool contains_multipole_neighbor) {

            cpu_launch_counter()++;
            cpu_monopole_buffer_t local_monopoles_staging_area(ENTRIES);

            update_input(monopoles, neighbors, type, local_monopoles_staging_area, neighbor_empty_monopoles, grid_ptr);
            compute_interactions(
                type, is_direction_empty, neighbors, dx, local_monopoles_staging_area, grid_ptr);
            // Do we need to run the p2m kernel as well?
            if (contains_multipole_neighbor) {
                // runs (and converts neighbor data) for each p2m kernel
                compute_p2m_interactions_neighbors_only(monopoles, com_ptr, neighbors, type, is_direction_empty, grid_ptr);
            }
        }

        void p2p_interaction_interface::compute_interactions(gsolve_type type,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::vector<neighbor_gravity_type>& all_neighbor_interaction_data, real dx,
            const cpu_monopole_buffer_t& local_monopoles_staging_area, std::shared_ptr<grid>& grid_ptr) {
            if (p2p_type == interaction_kernel_type::SOA_CPU) {
                p2p_cpu_kernel kernel_monopoles;
                cpu_expansion_result_buffer_t potential_expansions_SoA;
                kernel_monopoles.apply_stencil(local_monopoles_staging_area,
                    potential_expansions_SoA, stencil_masks(), stencil_four_constants(), dx);
                potential_expansions_SoA.to_non_SoA(grid_ptr->get_L());
            } else {
                grid_ptr->compute_interactions(type);
                // waits for boundary data and then computes boundary interactions
                for (auto const& dir : geo::direction::full_set()) {
                    if (!is_direction_empty[dir]) {
                        neighbor_gravity_type& neighbor_data = all_neighbor_interaction_data[dir];
                        if (neighbor_data.is_monopole) {
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
