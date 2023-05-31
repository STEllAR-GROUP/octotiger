//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/monopole_interactions/legacy/monopole_interaction_interface.hpp"
#include "octotiger/common_kernel/interactions_iterators.hpp"
#include "octotiger/monopole_interactions/legacy/p2m_interaction_interface.hpp"
#include "octotiger/monopole_interactions/legacy/p2p_cpu_kernel.hpp"    //VC ?
#include "octotiger/monopole_interactions/util/calculate_stencil.hpp"
#include "octotiger/options.hpp"

#include <algorithm>
#include <array>
#include <vector>

#include <aligned_buffer_util.hpp>
#include <buffer_manager.hpp>

#ifdef HPX_HAVE_APEX
#include <apex_api.hpp>
#endif
namespace octotiger {
namespace fmm {
    namespace monopole_interactions {
        size_t& monopole_interaction_interface::cpu_launch_counter() {
            static thread_local size_t cpu_launch_counter_ = 0;
            return cpu_launch_counter_;
        }
        size_t& monopole_interaction_interface::cuda_launch_counter() {
            static thread_local size_t cuda_launch_counter_ = 0;
            return cuda_launch_counter_;
        }

        std::vector<multiindex<>>& monopole_interaction_interface::stencil() {
            static thread_local std::vector<multiindex<>>
                stencil_;    // = calculate_stencil().first;
            static thread_local bool initialized = false;
            if (!initialized) {
                stencil_ = calculate_stencil().first;
                initialized = true;
            }
            return stencil_;
        }

        std::vector<bool>& monopole_interaction_interface::stencil_masks() {
            static thread_local std::vector<bool> stencil_masks_;
            static thread_local bool initialized = false;
            if (!initialized) {
                stencil_masks_ =
                    calculate_stencil_masks(monopole_interaction_interface::stencil()).first;
                initialized = true;
            }
            return stencil_masks_;
        }
        std::vector<std::array<real, 4>>& monopole_interaction_interface::four() {
            static thread_local std::vector<std::array<real, 4>> four_;
            static thread_local bool initialized = false;
            if (!initialized) {
                four_ = calculate_stencil().second;
                initialized = true;
            }
            return four_;
        }
        std::vector<std::array<real, 4>>& monopole_interaction_interface::stencil_four_constants() {
            static thread_local std::vector<std::array<real, 4>> stencil_four_constants_;
            static thread_local bool initialized = false;
            if (!initialized) {
                stencil_four_constants_ =
                    calculate_stencil_masks(monopole_interaction_interface::stencil()).second;
                initialized = true;
            }
            return stencil_four_constants_;
        }

        monopole_interaction_interface::monopole_interaction_interface() {
            this->p2p_type = opts().monopole_host_kernel_type;
        }

        void monopole_interaction_interface::compute_interactions(
            const std::vector<real>& monopoles,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::shared_ptr<grid>& grid_ptr, const bool contains_multipole_neighbor) {
            cpu_launch_counter()++;
            cpu_monopole_buffer_t local_monopoles_staging_area(ENTRIES);

            update_input(monopoles, neighbors, type, local_monopoles_staging_area, grid_ptr);
            compute_interactions(
                type, is_direction_empty, neighbors, dx, local_monopoles_staging_area, grid_ptr);
            // Do we need to run the p2m kernel as well?
            if (contains_multipole_neighbor) {
                // runs (and converts neighbor data) for each p2m kernel
                compute_p2m_interactions_neighbors_only(
                    monopoles, com_ptr, neighbors, type, is_direction_empty, grid_ptr);
            }
        }

        void monopole_interaction_interface::compute_interactions(gsolve_type type,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::vector<neighbor_gravity_type>& all_neighbor_interaction_data, real dx,
            const cpu_monopole_buffer_t& local_monopoles_staging_area,
            std::shared_ptr<grid>& grid_ptr) {
            if (p2p_type == interaction_host_kernel_type::VC) {
#ifdef OCTOTIGER_HAVE_VC    // kernel is only compiled with Vc
                p2p_cpu_kernel kernel_monopoles;
                cpu_expansion_result_buffer_t potential_expansions_SoA;
#ifdef HPX_HAVE_APEX
                auto p2p_timer = apex::start("kernel p2p vc");
#endif
                kernel_monopoles.apply_stencil(local_monopoles_staging_area,
                    potential_expansions_SoA, stencil_masks(), stencil_four_constants(), dx);
#ifdef HPX_HAVE_APEX
                apex::stop(p2p_timer);
#endif
                potential_expansions_SoA.to_non_SoA(grid_ptr->get_L());
#else    // should not happen - option gets already checked at application startup
                std::cerr << "Tried to call Vc kernel in non-Vc build!" << std::endl;
                abort();
#endif
            } else if (p2p_type == interaction_host_kernel_type::LEGACY) {
#ifdef HPX_HAVE_APEX
                auto p2p_timer = apex::start("kernel p2p legacy");
#endif
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
#ifdef HPX_HAVE_APEX
                apex::stop(p2p_timer);
#endif
            }
        }
    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
