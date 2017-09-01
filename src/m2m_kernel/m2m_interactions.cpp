#include "m2m_interactions.hpp"

#include "calculate_stencil.hpp"
#include "interactions_iterators.hpp"
#include "m2m_kernel.hpp"

#include <algorithm>

// Big picture questions:
// - use any kind of tiling?

size_t total_neighbors = 0;
size_t missing_neighbors = 0;

namespace octotiger {
namespace fmm {

    std::vector<multiindex<>> m2m_interactions::stencil;

    m2m_interactions::m2m_interactions(std::vector<multipole>& M_ptr,
        std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
        // grid& g,
        std::vector<neighbor_gravity_type>& neighbors, gsolve_type type)
      : neighbor_empty(27)
      , type(type) {
        local_expansions = std::vector<expansion>(EXPANSION_COUNT_PADDED);
        center_of_masses = std::vector<space_vector>(EXPANSION_COUNT_PADDED);

        std::vector<space_vector> const& com0 = *(com_ptr[0]);

        iterate_inner_cells_padded(
            [this, M_ptr, com0](const multiindex<>& i, const size_t flat_index,
                const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                local_expansions.at(flat_index) = M_ptr.at(flat_index_unpadded);
                center_of_masses.at(flat_index) = com0.at(flat_index_unpadded);
            });

        total_neighbors += 27;

        size_t current_missing = 0;
        size_t current_monopole = 0;

        for (size_t i = 0; i < neighbor_empty.size(); i++) {
            neighbor_empty[i] = false;
        }

        for (const geo::direction& dir : geo::direction::full_set()) {
            // don't use neighbor.direction, is always zero for empty cells!
            neighbor_gravity_type& neighbor = neighbors[dir];

            // this dir is setup as a multipole
            if (!neighbor.is_monopole) {
                if (!neighbor.data.M) {
                    // TODO: ask Dominic why !is_monopole and stuff still empty
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
                    std::vector<multipole>& neighbor_M_ptr = *(neighbor.data.M);
                    std::vector<space_vector>& neighbor_com0 = *(neighbor.data.x);
                    iterate_inner_cells_padding(
                        dir, [this, neighbor_M_ptr, neighbor_com0](const multiindex<>& i,
                                 const size_t flat_index, const multiindex<>& i_unpadded,
                                 const size_t flat_index_unpadded) {
                            local_expansions.at(flat_index) =
                                neighbor_M_ptr.at(flat_index_unpadded);
                            center_of_masses.at(flat_index) = neighbor_com0.at(flat_index_unpadded);
                        });
                }
            } else {
                // in case of monopole, boundary becomes padding in that direction
                // TODO: setting everything to zero might not be correct to create zero potentials
                iterate_inner_cells_padding(
                    dir, [this](const multiindex<>& i, const size_t flat_index, const multiindex<>&,
                             const size_t) {
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

        // std::cout << current_monopole << " monopoles, " << current_monopole << " missing
        // neighbors"
        //           << std::endl;

        // std::cout << std::boolalpha;
        // for (size_t i = 0; i < 27; i++) {
        //     if (i > 0) {
        //         std::cout << ", ";
        //     }
        //     std::cout << i << " -> " << neighbor_empty[i];
        // }
        // std::cout << std::endl;

        // allocate output variables without padding
        potential_expansions = std::vector<expansion>(EXPANSION_COUNT_NOT_PADDED);
        // TODO/BUG: expansion don't initialize to zero by default
        iterate_inner_cells_not_padded(
            [this](const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                expansion& e = potential_expansions.at(flat_index_unpadded);
                e = 0.0;
            });
        angular_corrections = std::vector<space_vector>(EXPANSION_COUNT_NOT_PADDED);
        // TODO/BUG: expansion don't initialize to zero by default
        iterate_inner_cells_not_padded(
            [this](const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                space_vector& s = angular_corrections.at(flat_index_unpadded);
                s = 0.0;
            });
    }

    void m2m_interactions::compute_interactions() {
        struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING> local_expansions_SoA(
            local_expansions);
        struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING> center_of_masses_SoA(
            center_of_masses);
        struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING> potential_expansions_SoA(
            potential_expansions);
        struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING> angular_corrections_SoA(
            angular_corrections);

        m2m_kernel kernel(
            // local_expansions_SoA,
            //               center_of_masses_SoA, potential_expansions_SoA,
            // angular_corrections_SoA,
            neighbor_empty, type);

        auto start = std::chrono::high_resolution_clock::now();

        kernel.apply_stencil(local_expansions_SoA, center_of_masses_SoA, potential_expansions_SoA,
            angular_corrections_SoA, stencil);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "new interaction kernel (apply only, ms): " << duration.count() << std::endl;

        // TODO: remove this after finalizing conversion
        // copy back SoA data into non-SoA result
        potential_expansions_SoA.to_non_SoA(potential_expansions);
        angular_corrections_SoA.to_non_SoA(angular_corrections);
    }

    std::vector<expansion>& m2m_interactions::get_local_expansions() {
        return local_expansions;
    }

    std::vector<space_vector>& m2m_interactions::get_center_of_masses() {
        return center_of_masses;
    }

    std::vector<expansion>& m2m_interactions::get_potential_expansions() {
        return potential_expansions;
    }

    std::vector<space_vector>& m2m_interactions::get_angular_corrections() {
        return angular_corrections;
    }

    void m2m_interactions::print_potential_expansions() {
        print_layered_not_padded(true, [this](const multiindex<>& i, const size_t flat_index) {
            std::cout << " (" << i << ") =[0] " << this->potential_expansions[flat_index][0];
        });
    }

    void m2m_interactions::print_angular_corrections() {
        print_layered_not_padded(true, [this](const multiindex<>& i, const size_t flat_index) {
            std::cout << " (" << i << ") =[0] " << this->angular_corrections[flat_index];
        });
    }

    void m2m_interactions::print_local_expansions() {
        print_layered_inner_padded(
            true, [this](const multiindex<>& i, const size_t flat_index,
                      const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                std::cout << " " << this->local_expansions[flat_index];
            });
    }

    void m2m_interactions::print_center_of_masses() {
        print_layered_padded(true, [this](const multiindex<>& i, const size_t flat_index) {
            std::cout << this->center_of_masses[flat_index];
        });
    }

    void m2m_interactions::add_to_potential_expansions(std::vector<expansion>& L) {
        iterate_inner_cells_not_padded([this, &L](multiindex<>& i, size_t flat_index) {
            potential_expansions[flat_index] += L[flat_index];
        });
    }

    void m2m_interactions::add_to_center_of_masses(std::vector<space_vector>& L_c) {
        iterate_inner_cells_not_padded([this, &L_c](multiindex<>& i, size_t flat_index) {
            center_of_masses[flat_index] += L_c[flat_index];
        });
    }

}    // namespace fmm
}    // namespace octotiger
