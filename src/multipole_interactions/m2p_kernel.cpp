#include "m2p_kernel.hpp"

#include "../common_kernel/struct_of_array_data.hpp"
#include "defs.hpp"
#include "interaction_types.hpp"
#include "options.hpp"

#include <array>
#include <functional>

// std::vector<interaction_type> ilist_debugging;

extern options opts;

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        m2p_kernel::m2p_kernel(std::vector<bool>& neighbor_empty)
          : neighbor_empty(neighbor_empty)
          , theta_rec_squared(sqr(1.0 / opts.theta))
        // , theta_rec_squared_scalar(sqr(1.0 / opts.theta))
        {
            for (size_t i = 0; i < m2m_int_vector::size(); i++) {
                offset_vector[i] = i;
            }
        }

        void m2p_kernel::apply_stencil(std::vector<real>& mons,
            struct_of_array_data<expansion, real, 20, ENTRIES,
                                           SOA_PADDING>& __restrict__ local_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>& center_of_masses_SoA,
            struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>&
                potential_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>&
                angular_corrections_SoA,
            const std::vector<multiindex<>>& stencil, gsolve_type type, real dX,
                                       std::array<real, NDIM>& xbase, bool (&z_skip)[3][3][3],
                bool (&y_skip)[3][3], bool (&x_skip)[3]) {
            // for(auto i = 0; i < local_expansions.size(); i++)
            //   std::cout << local_expansions[i] << " ";
            // for (multiindex<>& stencil_element : stencil) {
            for (size_t outer_stencil_index = 0; outer_stencil_index < stencil.size();
                 outer_stencil_index += 1) {
                const multiindex<>& stencil_element = stencil[outer_stencil_index];
                // std::cout << "stencil_element: " << stencil_element << std::endl;
                // TODO: remove after proper vectorization
                // multiindex<> se(stencil_element.x, stencil_element.y, stencil_element.z);
                // std::cout << "se: " << se << std::endl;
                // iterate_inner_cells_padded_stencil(se, *this);
                for (size_t i0 = 0; i0 < INNER_CELLS_PER_DIRECTION; i0++) {
                    const size_t x_interaction = i0 + stencil_element.x + INNER_CELLS_PADDING_DEPTH;
                    const size_t x_block = x_interaction / INNER_CELLS_PER_DIRECTION;
                    if (x_skip[x_block])
                        continue;
                    for (size_t i1 = 0; i1 < INNER_CELLS_PER_DIRECTION; i1++) {
                        const size_t y_interaction =
                            i1 + stencil_element.y + INNER_CELLS_PADDING_DEPTH;
                        const size_t y_block = y_interaction / INNER_CELLS_PER_DIRECTION;
                        if (y_skip[x_block][y_block])
                            continue;
                        // for (size_t i2 = 0; i2 < INNER_CELLS_PER_DIRECTION; i2++) {
                        for (size_t i2 = 0; i2 < INNER_CELLS_PER_DIRECTION;
                             i2 += m2m_vector::size()) {
                            const size_t z_interaction =
                                i2 + stencil_element.z + INNER_CELLS_PADDING_DEPTH;
                            const size_t z_block = z_interaction / INNER_CELLS_PER_DIRECTION;
                            const size_t z_interaction2 = i2 + stencil_element.z +
                                m2m_vector::size() - 1 + INNER_CELLS_PADDING_DEPTH;
                            const size_t z_block2 = z_interaction2 / INNER_CELLS_PER_DIRECTION;
                            if (z_skip[x_block][y_block][z_block] &&
                                z_skip[x_block][y_block][z_block2])
                                continue;
                            const multiindex<> cell_index(i0 + INNER_CELLS_PADDING_DEPTH,
                                i1 + INNER_CELLS_PADDING_DEPTH, i2 + INNER_CELLS_PADDING_DEPTH);
                            // BUG: indexing has to be done with uint32_t because of Vc
                            // limitation
                            const int64_t cell_flat_index =
                                to_flat_index_padded(cell_index);    // iii0...
                            const multiindex<> cell_index_unpadded(i0, i1, i2);
                            const int64_t cell_flat_index_unpadded =
                                to_inner_flat_index_not_padded(cell_index_unpadded);

                            // indices on coarser level (for outer stencil boundary)
                            // implicitly broadcasts to vector
                            multiindex<m2m_int_vector> cell_index_coarse(cell_index);
                            for (size_t j = 0; j < m2m_int_vector::size(); j++) {
                                cell_index_coarse.z[j] += j;
                            }
                            // note that this is the same for groups of 2x2x2 elements
                            // -> maps to the same for some SIMD lanes
                            cell_index_coarse.transform_coarse();

                            // calculate position of the monopole

                            if (type == RHO) {
                                this->blocked_interaction_rho(local_expansions_SoA, mons,
                                    center_of_masses_SoA, potential_expansions_SoA,
                                    angular_corrections_SoA, cell_index, cell_flat_index,
                                    cell_index_coarse, cell_index_unpadded,
                                    cell_flat_index_unpadded, stencil, outer_stencil_index, dX,
                                    xbase);
                            } else {
                                this->blocked_interaction_non_rho(mons, center_of_masses_SoA,
                                    potential_expansions_SoA, angular_corrections_SoA, cell_index,
                                    cell_flat_index, cell_index_coarse, cell_index_unpadded,
                                    cell_flat_index_unpadded, stencil, outer_stencil_index, dX,
                                    xbase);
                            }
                        }
                    }
                }
            }
        }

        void m2p_kernel::vectors_check_empty() {
            vector_is_empty = std::vector<bool>(PADDED_STRIDE * PADDED_STRIDE * PADDED_STRIDE);
            for (size_t i0 = 0; i0 < PADDED_STRIDE; i0 += 1) {
                for (size_t i1 = 0; i1 < PADDED_STRIDE; i1 += 1) {
                    for (size_t i2 = 0; i2 < PADDED_STRIDE; i2 += 1) {
                        const multiindex<> cell_index(i0, i1, i2);
                        const int64_t cell_flat_index = to_flat_index_padded(cell_index);

                        const multiindex<> in_boundary_start(
                            (cell_index.x / INNER_CELLS_PER_DIRECTION) - 1,
                            (cell_index.y / INNER_CELLS_PER_DIRECTION) - 1,
                            (cell_index.z / INNER_CELLS_PER_DIRECTION) - 1);

                        const multiindex<> in_boundary_end(in_boundary_start.x, in_boundary_start.y,
                            ((cell_index.z + m2m_int_vector::size()) / INNER_CELLS_PER_DIRECTION) -
                                1);

                        geo::direction dir_start;
                        dir_start.set(
                            in_boundary_start.x, in_boundary_start.y, in_boundary_start.z);
                        geo::direction dir_end;
                        dir_end.set(in_boundary_end.x, in_boundary_end.y, in_boundary_end.z);

                        if (neighbor_empty[dir_start.flat_index_with_center()] &&
                            neighbor_empty[dir_end.flat_index_with_center()]) {
                            vector_is_empty[cell_flat_index] = true;
                        } else {
                            vector_is_empty[cell_flat_index] = false;
                        }
                        // if (i0 >= 8 && i0 < 16 && i1 >= 8 && i1 < 16 && i2 >= 8 && i2 < 16)
                        //     vector_is_empty[cell_flat_index] = true;
                    }
                }
            }
        }
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
