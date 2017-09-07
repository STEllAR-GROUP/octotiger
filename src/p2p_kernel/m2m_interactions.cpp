#include "m2m_interactions.hpp"

#include "calculate_stencil.hpp"
#include "interactions_iterators.hpp"
#include "m2m_kernel.hpp"

#include <algorithm>

// Big picture questions:
// - use any kind of tiling?


namespace octotiger {
namespace fmm {
namespace p2p_kernel {
size_t total_neighbors = 0;
size_t missing_neighbors = 0;

    std::vector<multiindex<>> m2m_interactions::stencil;

    m2m_interactions::m2m_interactions(std::vector<real>& mons,
        std::vector<neighbor_gravity_type>& neighbors,
        gsolve_type type, real dx)
        : neighbor_empty(27),
       type(type), dx(dx) {
      local_expansions = std::vector<real>(EXPANSION_COUNT_PADDED);

        iterate_inner_cells_padded(
            [this, mons](const multiindex<>& i, const size_t flat_index,
                const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                local_expansions.at(flat_index) = mons.at(flat_index_unpadded);
            });

        total_neighbors += 27;

        size_t current_missing = 0;

        for (size_t i = 0; i < neighbor_empty.size(); i++) {
            neighbor_empty[i] = false;
        }

        for (const geo::direction& dir : geo::direction::full_set()) {
            // don't use neighbor.direction, is always zero for empty cells!
            neighbor_gravity_type& neighbor = neighbors[dir];

            // this dir is setup as a multipole
            if (neighbor.is_monopole) {
                if (!neighbor.data.m) {
                    // TODO: ask Dominic why !is_monopole and stuff still empty
                    iterate_inner_cells_padding(
                        dir, [this](const multiindex<>& i, const size_t flat_index,
                                 const multiindex<>&, const size_t) {
                            // initializes whole expansion, relatively expansion
                            local_expansions.at(flat_index) = 0.0;
                        });
                    missing_neighbors += 1;
                    current_missing += 1;
                    neighbor_empty[dir.flat_index_with_center()] = true;
                } else {
                std::vector<real>& neighbor_mons = *(neighbor.data.m);
                iterate_inner_cells_padding(
                    dir, [this, neighbor_mons](const multiindex<>& i, const size_t flat_index, const multiindex<>&,
                             const size_t flat_index_unpadded) {
                        // initializes whole expansion, relatively expansion
                      local_expansions.at(flat_index) = neighbor_mons.at(flat_index_unpadded);
                    });
                }
            } else {
                    iterate_inner_cells_padding(
                        dir, [this](const multiindex<>& i,
                                 const size_t flat_index, const multiindex<>& i_unpadded,
                                 const size_t flat_index_unpadded) {
                            local_expansions.at(flat_index) = 0.0;
                        });
                missing_neighbors += 1;
                neighbor_empty[dir.flat_index_with_center()] = true;
            }
        }

        neighbor_empty[13] = false;

        // allocate output variables without padding
        potential_expansions = std::vector<expansion>(EXPANSION_COUNT_NOT_PADDED);
        // TODO/BUG: expansion don't initialize to zero by default
        iterate_inner_cells_not_padded(
            [this](const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                expansion& e = potential_expansions.at(flat_index_unpadded);
                e = 0.0;
            });
    }

    void m2m_interactions::compute_interactions() {
        struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING> potential_expansions_SoA(
            potential_expansions);

        m2m_kernel kernel(
            neighbor_empty, type, dx);

        // for(auto i = 0; i < local_expansions.size(); i++)
        //   std::cout << local_expansions[i] << " ";
        auto start = std::chrono::high_resolution_clock::now();

        kernel.apply_stencil(local_expansions, potential_expansions_SoA, stencil);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;

        // TODO: remove this after finalizing conversion
        // copy back SoA data into non-SoA result
        potential_expansions_SoA.to_non_SoA(potential_expansions);
    }

    std::vector<real>& m2m_interactions::get_local_expansions() {
        return local_expansions;
    }

    std::vector<expansion>& m2m_interactions::get_potential_expansions() {
        return potential_expansions;
    }

    void m2m_interactions::print_potential_expansions() {
        print_layered_not_padded(true, [this](const multiindex<>& i, const size_t flat_index) {
            std::cout << " (" << i << ") =[0] " << this->potential_expansions[flat_index][0];
        });
    }

    void m2m_interactions::add_to_potential_expansions(std::vector<expansion>& L) {
        iterate_inner_cells_not_padded([this, &L](multiindex<>& i, size_t flat_index) {
            potential_expansions[flat_index] += L[flat_index];
        });
    }

}    // namespace p2p_kernel
}    // namespace fmm
}    // namespace octotiger
