#include "m2p_interactions.hpp"

#include "../common_kernel/interactions_iterators.hpp"
#include "calculate_stencil.hpp"
#include "m2p_kernel.hpp"

#include <algorithm>

// Big picture questions:
// - use any kind of tiling?

namespace octotiger {
namespace fmm {
    namespace m2p_kernel {
        size_t total_neighbors = 0;
        size_t missing_neighbors = 0;

        std::vector<multiindex<>> m2p_interactions::stencil;

        m2p_interactions::m2p_interactions(std::vector<multipole>& multipoles,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type)
          : neighbor_empty(27)
          , type(type) {
            // Create our input structure for the compute kernel
            local_expansions = std::vector<expansion>(EXPANSION_COUNT_PADDED);
            center_of_masses = std::vector<space_vector>(EXPANSION_COUNT_PADDED);
            std::vector<space_vector> const& com0 = *(com_ptr[0]);


        iterate_inner_cells_padded(
            [this, multipoles, com0](const multiindex<>& i, const size_t flat_index,
                const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
              local_expansions.at(flat_index) = 0.0;
                center_of_masses.at(flat_index) = com0.at(flat_index_unpadded);
            });
            total_neighbors += 27;

            size_t current_missing = 0;
            size_t current_monopole = 0;

            for (size_t i = 0; i < neighbor_empty.size(); i++) {
                neighbor_empty[i] = false;
            }

            multipole_neighbors_exist = false;

            // Now look at neighboring data
            for (const geo::direction& dir : geo::direction::full_set()) {
                // don't use neighbor.direction, is always zero for empty cells!
                neighbor_gravity_type& neighbor = neighbors[dir];

                // this dir is setup as a multipole - and we only consider multipoles here
                if (!neighbor.is_monopole) {
                    // neighbor has no data - input structure just recevices zeros as padding
                    if (!neighbor.data.M) {
                        iterate_inner_cells_padding(
                            dir, [this](const multiindex<>& i, const size_t flat_index,
                                     const multiindex<>&, const size_t) {
                                // initializes whole expansion, relatively expansion
                                local_expansions.at(flat_index) = 0.0;
                                // initializes x,y,z vector
                                center_of_masses.at(flat_index) = 0.0;
                            });
                        missing_neighbors += 1;
                        current_missing += 1;
                        neighbor_empty[dir.flat_index_with_center()] = true;
                    } else {
                        // Get multipole data into our input structure
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
                            });
                        multipole_neighbors_exist = true;
                    }
                } else {
                    // in case of monopole, boundary becomes padding in that direction
                    // TODO: setting everything to zero might not be correct to create zero
                    // potentials
                    iterate_inner_cells_padding(
                        dir, [this](const multiindex<>& i, const size_t flat_index,
                                 const multiindex<>&, const size_t) {
                            // initializes whole expansion, relatively expansion
                            local_expansions.at(flat_index) = 0.0;
                            // initializes x,y,z vector
                            center_of_masses.at(flat_index) = 0.0;
                        });
                    missing_neighbors += 1;
                    current_monopole += 1;
                    neighbor_empty[dir.flat_index_with_center()] = true;
                }
            }

            neighbor_empty[13] = false;

            // Allocate our output structure and initialise it
            potential_expansions = std::vector<expansion>(EXPANSION_COUNT_NOT_PADDED);
            iterate_inner_cells_not_padded(
                [this](const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                    expansion& e = potential_expansions.at(flat_index_unpadded);
                    e = 0.0;
                });
            angular_corrections = std::vector<space_vector>(EXPANSION_COUNT_NOT_PADDED);
            iterate_inner_cells_not_padded(
                [this](const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                    space_vector& s = angular_corrections.at(flat_index_unpadded);
                    s = 0.0;
                });
        }

        void m2p_interactions::compute_interactions() {
            if (!multipole_neighbors_exist)
              return;
            // Convert input structure to new datastructure (SoA)
            struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING> local_expansions_SoA(
                local_expansions);
            struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING> center_of_masses_SoA(
                center_of_masses);
            struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>
                potential_expansions_SoA(potential_expansions);
            struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>
                angular_corrections_SoA(angular_corrections);

            // std::cout << "Expansions" << std::endl;
            // for (auto i = 0; i < 100; ++i) {
            //   for (auto x = 0; x < 20; ++x) {
            //     std::cout <<  local_expansions[i][x] << " ";
            //   }
            //   std::cout << std::endl;
            // }
            // std::cout << "Centers" << std::endl;
            // for (auto i = 0; i < 100; ++i) {
            //   for (auto x = 0; x < 3; ++x) {
            //     std::cout <<  center_of_masses[i][x] << " ";
            //   }
            //   std::cout << std::endl;
            // }
            // int count = 0;
            // for (auto i = 0; i < local_expansions.size(); ++i) {
            //   for (auto x = 0; x < 20; ++x) {
            //     if (local_expansions[i][x] != 0)
            //       count++;
            //   }
            // }
            // std::cout << "There are " << count << " non-zero entries in the local expansions"
            //           << std::endl;
            // count = 0;
            // for (auto i = 0; i < center_of_masses.size(); ++i) {
            //   for (auto x = 0; x < 3; ++x) {
            //     if (center_of_masses[i][x] != 0)
            //       count++;
            //   }
            // }
            // std::cout << "There are " << count << " non-zero entries in the center of masses"
            //           << std::endl;


            // std::cin.get();

            m2p_kernel kernel(neighbor_empty, type);

            // for(auto i = 0; i < local_expansions.size(); i++)
            //   std::cout << local_expansions[i] << " ";
            auto start = std::chrono::high_resolution_clock::now();

            kernel.apply_stencil(local_expansions_SoA, center_of_masses_SoA,
                potential_expansions_SoA, angular_corrections_SoA, stencil);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> duration = end - start;
            // copy back SoA data into non-SoA result
            potential_expansions_SoA.to_non_SoA(potential_expansions);
            angular_corrections_SoA.to_non_SoA(angular_corrections);
            // std::cout << "Potential Expansaions" << std::endl;
            // for (auto i = 0; i < 100; ++i) {
            //   for (auto x = 0; x < 20; ++x) {
            //     std::cout <<  potential_expansions[i][x] << " ";
            //   }
            //   std::cout << std::endl;
            // }
            // std::cout << "Centers" << std::endl;
            // for (auto i = 0; i < 100; ++i) {
            //   for (auto x = 0; x < 3; ++x) {
            //     std::cout <<  angular_corrections[i][x] << " ";
            //   }
            //   std::cout << std::endl;
            // }
            // count = 0;
            // for (auto i = 0; i < potential_expansions.size(); ++i) {
            //   for (auto x = 0; x < 20; ++x) {
            //     if (potential_expansions[i][x] != 0)
            //       count++;
            //   }
            // }
            // std::cout << "There are " << count << " non-zero entries in the potential expansions"
            //           << std::endl;
            // count = 0;
            // for (auto i = 0; i < angular_corrections.size(); ++i) {
            //   for (auto x = 0; x < 3; ++x) {
            //     if (angular_corrections[i][x] != 0)
            //       count++;
            //   }
            // }
            // std::cout << "There are " << count << " non-zero entries in the angular corrections"
            //           << std::endl;
            // std::cin.get();
        }

        std::vector<expansion>& m2p_interactions::get_local_expansions() {
            return local_expansions;
        }

        std::vector<space_vector>& m2p_interactions::get_center_of_masses() {
            return center_of_masses;
        }

        std::vector<expansion>& m2p_interactions::get_potential_expansions() {
            return potential_expansions;
        }

        std::vector<space_vector>& m2p_interactions::get_angular_corrections() {
            return angular_corrections;
        }

        void m2p_interactions::print_potential_expansions() {
            print_layered_not_padded(true, [this](const multiindex<>& i, const size_t flat_index) {
                std::cout << " (" << i << ") =[0] " << this->potential_expansions[flat_index][0];
            });
        }

        void m2p_interactions::print_angular_corrections() {
            print_layered_not_padded(true, [this](const multiindex<>& i, const size_t flat_index) {
                std::cout << " (" << i << ") =[0] " << this->angular_corrections[flat_index];
            });
        }

        void m2p_interactions::print_local_expansions() {
            print_layered_padded(true, [this](const multiindex<>& i, const size_t flat_index) {
                std::cout << " " << this->local_expansions[flat_index];
            });
        }

        void m2p_interactions::print_center_of_masses() {
            print_layered_padded(true, [this](const multiindex<>& i, const size_t flat_index) {
                std::cout << this->center_of_masses[flat_index];
            });
        }

        void m2p_interactions::add_to_potential_expansions(std::vector<expansion>& L) {
            if (!multipole_neighbors_exist)
              return;
            iterate_inner_cells_not_padded([this, &L](multiindex<>& i, size_t flat_index) {
                potential_expansions[flat_index] += L[flat_index];
            });
        }

        void m2p_interactions::add_to_center_of_masses(std::vector<space_vector>& L_c) {
            if (!multipole_neighbors_exist)
              return;
            iterate_inner_cells_not_padded([this, &L_c](multiindex<>& i, size_t flat_index) {
                center_of_masses[flat_index] += L_c[flat_index];
            });
        }

    }    // namespace m2p_kernel
}    // namespace fmm
}    // namespace octotiger
