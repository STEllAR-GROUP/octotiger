#include "multipole_interaction_interface.hpp"

#include "../common_kernel/interactions_iterators.hpp"
#include "calculate_stencil.hpp"
#include "m2m_kernel.hpp"
#include "m2p_kernel.hpp"

#include <algorithm>

// Big picture questions:
// - use any kind of tiling?

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        std::vector<multiindex<>> multipole_interaction_interface::stencil_multipole_interactions;
        std::vector<multiindex<>> multipole_interaction_interface::stencil_mixed_interactions;
        multipole_interaction_interface::multipole_interaction_interface(void)
          : neighbor_empty_multipole(27)
          , neighbor_empty_monopole(27) {
            local_expansions = std::vector<expansion>(EXPANSION_COUNT_PADDED);
            center_of_masses = std::vector<space_vector>(EXPANSION_COUNT_PADDED);

            local_monopoles = std::vector<real>(EXPANSION_COUNT_PADDED);
            interact = std::vector<bool>(EXPANSION_COUNT_PADDED);

            potential_expansions = std::vector<expansion>(EXPANSION_COUNT_NOT_PADDED);
            angular_corrections = std::vector<space_vector>(EXPANSION_COUNT_NOT_PADDED);
        }

        void multipole_interaction_interface::update_input(std::vector<real>& monopoles,
            std::vector<multipole>& M_ptr,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type t, real dx,
            std::array<real, NDIM> xbase) {
            type = t;
            dX = dx;
            xBase = xbase;
            std::vector<space_vector> const& com0 = *(com_ptr[0]);

            iterate_inner_cells_padded(
                [this, M_ptr, com0](const multiindex<>& i, const size_t flat_index,
                    const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                    local_expansions.at(flat_index) = M_ptr.at(flat_index_unpadded);
                    center_of_masses.at(flat_index) = com0.at(flat_index_unpadded);
                    interact.at(flat_index) = false;
                });

            for (size_t i = 0; i < neighbor_empty_multipole.size(); i++) {
                neighbor_empty_multipole[i] = false;
                neighbor_empty_monopole[i] = false;
            }

            for (const geo::direction& dir : geo::direction::full_set()) {
                // don't use neighbor.direction, is always zero for empty cells!
                neighbor_gravity_type& neighbor = neighbors[dir];

                // this dir is setup as a multipole
                if (!neighbor.is_monopole) {
                    neighbor_empty_monopole[dir.flat_index_with_center()] = true;
                    if (!neighbor.data.M) {
                        // TODO: ask Dominic why !is_monopole and stuff still empty
                        iterate_inner_cells_padding(
                            dir, [this](const multiindex<>& i, const size_t flat_index,
                                     const multiindex<>&, const size_t) {
                                // initializes whole expansion, relatively expansion
                                local_expansions.at(flat_index) = 0.0;
                                // initializes x,y,z vector
                                center_of_masses.at(flat_index) = 0.0;

                                local_monopoles.at(flat_index) = 0.0;
                                interact.at(flat_index) = false;
                            });
                        neighbor_empty_multipole[dir.flat_index_with_center()] = true;
                    } else {
                        std::vector<multipole>& neighbor_M_ptr = *(neighbor.data.M);
                        std::vector<space_vector>& neighbor_com0 = *(neighbor.data.x);
                        iterate_inner_cells_padding(
                            dir, [this, neighbor_M_ptr, neighbor_com0](const multiindex<>& i,
                                     const size_t flat_index, const multiindex<>& i_unpadded,
                                     const size_t flat_index_unpadded) {
                                local_expansions.at(flat_index) =
                                    neighbor_M_ptr.at(flat_index_unpadded);
                                center_of_masses.at(flat_index) =
                                    neighbor_com0.at(flat_index_unpadded);

                                local_monopoles.at(flat_index) = 0.0;
                                interact.at(flat_index) = false;
                            });
                    }
                } else {
                    neighbor_empty_multipole[dir.flat_index_with_center()] = true;
                    // neighbor has no data - input structure just recevices zeros as padding
                    if (!neighbor.data.m) {
                        iterate_inner_cells_padding(
                            dir, [this](const multiindex<>& i, const size_t flat_index,
                                     const multiindex<>&, const size_t) {
                                // initializes whole expansion, relatively expansion
                                local_monopoles.at(flat_index) = 0.0;
                                local_expansions.at(flat_index) = 0.0;
                                // initializes x,y,z vector
                                center_of_masses.at(flat_index) = 0.0;
                                interact.at(flat_index) = false;
                            });
                        neighbor_empty_monopole[dir.flat_index_with_center()] = true;
                    } else {
                        // Get multipole data into our input structure
                        std::vector<real>& neighbor_mons = *(neighbor.data.m);
                        std::vector<space_vector>& neighbor_com0 = *(neighbor.data.x);
                        iterate_inner_cells_padding(
                            dir,
                            [this, neighbor_mons](const multiindex<>& i, const size_t flat_index,
                                const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                                local_expansions.at(flat_index) = 0.0;
                                center_of_masses.at(flat_index) = 0.0;

                                local_monopoles.at(flat_index) =
                                    neighbor_mons.at(flat_index_unpadded);
                                interact.at(flat_index) = true;
                            });
                        monopole_neighbors_exist = true;
                    }
                }
            }

            neighbor_empty_multipole[13] = false;
            neighbor_empty_monopole[13] = false;

            // TODO/BUG: expansion don't initialize to zero by default
            iterate_inner_cells_not_padded(
                [this](const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                    expansion& e = potential_expansions.at(flat_index_unpadded);
                    e = 0.0;
                });
            // TODO/BUG: expansion don't initialize to zero by default
            iterate_inner_cells_not_padded(
                [this](const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                    space_vector& s = angular_corrections.at(flat_index_unpadded);
                    s = 0.0;
                });
            // std::fill(std::begin(potential_expansions), std::end(potential_expansions), ZERO);

            // std::cout << "local_expansions:" << std::endl;
            // this->print_local_expansions();
            // std::cout << "center_of_masses:" << std::endl;
            // this->print_center_of_masses();
        }

        void multipole_interaction_interface::compute_interactions(interaction_kernel_type m2m_type,
            interaction_kernel_type m2p_type,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::vector<neighbor_gravity_type>& all_neighbor_interaction_data) {
            if (m2m_type == interaction_kernel_type::SOA_CPU &&
                m2p_type == interaction_kernel_type::SOA_CPU) {
                struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>
                    local_expansions_SoA(local_expansions);
                struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>
                    center_of_masses_SoA(center_of_masses);
                struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>
                    potential_expansions_SoA(potential_expansions);
                struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>
                    angular_corrections_SoA(angular_corrections);

                m2m_kernel kernel(neighbor_empty_multipole, type);
                m2p_kernel mixed_interactions_kernel(neighbor_empty_monopole, type, dX, xBase);

                mixed_interactions_kernel.apply_stencil(local_monopoles, local_expansions_SoA,
                    center_of_masses_SoA, potential_expansions_SoA, angular_corrections_SoA,
                    stencil_mixed_interactions, interact);
                kernel.apply_stencil(local_expansions_SoA, center_of_masses_SoA,
                    potential_expansions_SoA, angular_corrections_SoA,
                    stencil_multipole_interactions);

                potential_expansions_SoA.to_non_SoA(potential_expansions);
                angular_corrections_SoA.to_non_SoA(angular_corrections);

                std::vector<expansion>& L = grid_ptr->get_L();
                std::vector<space_vector>& L_c = grid_ptr->get_L_c();
                for (size_t i = 0; i < L.size(); i++) {
                    L[i] = potential_expansions[i];
                }
                for (size_t i = 0; i < L_c.size(); i++) {
                    L_c[i] = angular_corrections[i];
                }

            } else if (m2m_type == interaction_kernel_type::SOA_CPU) {
                struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>
                    local_expansions_SoA(local_expansions);
                struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>
                    center_of_masses_SoA(center_of_masses);
                struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>
                    potential_expansions_SoA(potential_expansions);
                struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>
                    angular_corrections_SoA(angular_corrections);
                m2m_kernel kernel(neighbor_empty_multipole, type);
                kernel.apply_stencil(local_expansions_SoA, center_of_masses_SoA,
                    potential_expansions_SoA, angular_corrections_SoA,
                    stencil_multipole_interactions);

                potential_expansions_SoA.to_non_SoA(potential_expansions);
                angular_corrections_SoA.to_non_SoA(angular_corrections);

                std::vector<expansion>& L = grid_ptr->get_L();
                std::vector<space_vector>& L_c = grid_ptr->get_L_c();
                std::fill(std::begin(L), std::end(L), ZERO);
                std::fill(std::begin(L_c), std::end(L_c), ZERO);
                for (auto const& dir : geo::direction::full_set()) {
                    if (!is_direction_empty[dir]) {
                        neighbor_gravity_type& neighbor_data = all_neighbor_interaction_data[dir];
                        if (neighbor_data.is_monopole) {
                            grid_ptr->compute_boundary_interactions(type, neighbor_data.direction,
                                neighbor_data.is_monopole, neighbor_data.data);
                        }
                    }
                }
                for (size_t i = 0; i < L.size(); i++) {
                    L[i] += potential_expansions[i];
                }
                for (size_t i = 0; i < L_c.size(); i++) {
                    L_c[i] += angular_corrections[i];
                }

            } else if (m2p_type == interaction_kernel_type::SOA_CPU) {
                struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>
                    local_expansions_SoA(local_expansions);
                struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>
                    center_of_masses_SoA(center_of_masses);
                struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>
                    potential_expansions_SoA(potential_expansions);
                struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>
                    angular_corrections_SoA(angular_corrections);
                m2p_kernel mixed_interactions_kernel(neighbor_empty_monopole, type, dX, xBase);
                mixed_interactions_kernel.apply_stencil(local_monopoles, local_expansions_SoA,
                    center_of_masses_SoA, potential_expansions_SoA, angular_corrections_SoA,
                    stencil_mixed_interactions, interact);
                potential_expansions_SoA.to_non_SoA(potential_expansions);
                angular_corrections_SoA.to_non_SoA(angular_corrections);

                std::vector<expansion>& L = grid_ptr->get_L();
                std::vector<space_vector>& L_c = grid_ptr->get_L_c();
                std::fill(std::begin(L), std::end(L), ZERO);
                std::fill(std::begin(L_c), std::end(L_c), ZERO);

                grid_ptr->compute_interactions(type);
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
                for (size_t i = 0; i < L.size(); i++) {
                    L[i] += potential_expansions[i];
                }
                for (size_t i = 0; i < L_c.size(); i++) {
                    L_c[i] += angular_corrections[i];
                }
            } else {
                // old-style interaction calculation
                // computes inner interactions
                grid_ptr->compute_interactions(type);
                // waits for boundary data and then computes boundary interactions
                for (auto const& dir : geo::direction::full_set()) {
                    if (!is_direction_empty[dir]) {
                        neighbor_gravity_type& neighbor_data = all_neighbor_interaction_data[dir];
                        grid_ptr->compute_boundary_interactions(type, neighbor_data.direction,
                            neighbor_data.is_monopole, neighbor_data.data);
                    }
                }
            }
        }

        std::vector<expansion>& multipole_interaction_interface::get_local_expansions() {
            return local_expansions;
        }

        std::vector<space_vector>& multipole_interaction_interface::get_center_of_masses() {
            return center_of_masses;
        }

        std::vector<expansion>& multipole_interaction_interface::get_potential_expansions() {
            return potential_expansions;
        }

        std::vector<space_vector>& multipole_interaction_interface::get_angular_corrections() {
            return angular_corrections;
        }

        void multipole_interaction_interface::print_potential_expansions() {
            print_layered_not_padded(true, [this](const multiindex<>& i, const size_t flat_index) {
                std::cout << " (" << i << ") =[0] " << this->potential_expansions[flat_index][0];
            });
        }

        void multipole_interaction_interface::print_angular_corrections() {
            print_layered_not_padded(true, [this](const multiindex<>& i, const size_t flat_index) {
                std::cout << " (" << i << ") =[0] " << this->angular_corrections[flat_index];
            });
        }

        void multipole_interaction_interface::print_local_expansions() {
            print_layered_padded(true, [this](const multiindex<>& i, const size_t flat_index) {
                std::cout << " " << this->local_expansions[flat_index];
            });
        }

        void multipole_interaction_interface::print_center_of_masses() {
            print_layered_padded(true, [this](const multiindex<>& i, const size_t flat_index) {
                std::cout << this->center_of_masses[flat_index];
            });
        }

        void multipole_interaction_interface::add_to_potential_expansions(
            std::vector<expansion>& L) {
            iterate_inner_cells_not_padded([this, &L](multiindex<>& i, size_t flat_index) {
                potential_expansions[flat_index] += L[flat_index];
            });
        }

        void multipole_interaction_interface::add_to_center_of_masses(
            std::vector<space_vector>& L_c) {
            iterate_inner_cells_not_padded([this, &L_c](multiindex<>& i, size_t flat_index) {
                center_of_masses[flat_index] += L_c[flat_index];
            });
        }

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
