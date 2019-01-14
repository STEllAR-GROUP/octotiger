#include "octotiger/monopole_interactions/p2m_interaction_interface.hpp"
#include "octotiger/monopole_interactions/calculate_stencil.hpp"

#include "octotiger/common_kernel/interactions_iterators.hpp"

#include "octotiger/options.hpp"
#include "octotiger/real.hpp"

#include <array>
#include <algorithm>
#include <vector>

// Big picture questions:
// - use any kind of tiling?

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {
        const thread_local std::vector<multiindex<>> p2m_interaction_interface::stencil =
            calculate_stencil().first;
        thread_local std::vector<real> p2m_interaction_interface::local_monopoles_staging_area(
            ENTRIES);
        thread_local struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>
            p2m_interaction_interface::local_expansions_staging_area;
        thread_local struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>
            p2m_interaction_interface::center_of_masses_staging_area;

        p2m_interaction_interface::p2m_interaction_interface(void)
          : neighbor_empty_multipoles(27)
          , kernel(neighbor_empty_multipoles) {
            this->p2m_type = opts().p2m_kernel_type;
        }

        void p2m_interaction_interface::compute_p2m_interactions(std::vector<real>& monopoles,
            std::vector<multipole>& M_ptr,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type,
            std::array<bool, geo::direction::count()>& is_direction_empty) {
            update_input(monopoles, M_ptr, com_ptr, neighbors, type, local_monopoles_staging_area,
                local_expansions_staging_area, center_of_masses_staging_area);
            compute_interactions(type, is_direction_empty, neighbors);
        }

        void p2m_interaction_interface::compute_interactions(gsolve_type type,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::vector<neighbor_gravity_type>& all_neighbor_interaction_data) {
            if (p2m_type == interaction_kernel_type::SOA_CPU) {
                if (multipole_neighbors_exist) {
                    struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>
                        potential_expansions_SoA;
                    struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>
                        angular_corrections_SoA;
                    kernel.apply_stencil(local_expansions_staging_area,
                        center_of_masses_staging_area, potential_expansions_SoA,
                        angular_corrections_SoA, stencil, type, x_skip, y_skip, z_skip);
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
