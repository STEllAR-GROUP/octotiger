#include "m2m_kernel.hpp"

#include "../common_kernel/struct_of_array_data.hpp"
#include "defs.hpp"
#include "interaction_types.hpp"
#include "../common_kernel/interaction_constants.hpp"
#include "options.hpp"

#include <array>
#include <functional>

extern options opts;

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        m2m_kernel::m2m_kernel(void)
          : theta_rec_squared(sqr(1.0 / opts.theta))
        {
            for (size_t i = 0; i < m2m_int_vector::size(); i++) {
                offset_vector[i] = i;
            }
        }

        void m2m_kernel::apply_stencil(
            const struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>& local_expansions_SoA,
            const struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>& center_of_masses_SoA,
            struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>&
                potential_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>&
                angular_corrections_SoA,
            const std::vector<real>& mons, const two_phase_stencil& stencil, gsolve_type type) {
            // vectors_check_empty();
            // for (multiindex<>& stencil_element : stencil) {
            for (size_t outer_stencil_index = 0;
                 outer_stencil_index < stencil.stencil_elements.size();
                 outer_stencil_index += STENCIL_BLOCKING) {
                // std::cout << "stencil_element: " << stencil_element << std::endl;
                // TODO: remove after proper vectorization
                // multiindex<> se(stencil_element.x, stencil_element.y, stencil_element.z);
                // std::cout << "se: " << se << std::endl;
                // iterate_inner_cells_padded_stencil(se, *this);
                for (size_t i0 = 0; i0 < INNER_CELLS_PER_DIRECTION; i0++) {
                    for (size_t i1 = 0; i1 < INNER_CELLS_PER_DIRECTION; i1++) {
                        // for (size_t i2 = 0; i2 < INNER_CELLS_PER_DIRECTION; i2++) {
                        for (size_t i2 = 0; i2 < INNER_CELLS_PER_DIRECTION;
                             i2 += m2m_vector::size()) {
                            const multiindex<> cell_index(i0 + INNER_CELLS_PADDING_DEPTH,
                                i1 + INNER_CELLS_PADDING_DEPTH, i2 + INNER_CELLS_PADDING_DEPTH);
                            // BUG: indexing has to be done with uint32_t because of Vc limitation
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

                            if (type == RHO) {
                                this->blocked_interaction_rho(local_expansions_SoA,
                                    center_of_masses_SoA, potential_expansions_SoA,
                                    angular_corrections_SoA, mons,
                                    cell_index, cell_flat_index, cell_index_coarse,
                                    cell_index_unpadded, cell_flat_index_unpadded, stencil,
                                    outer_stencil_index);
                            } else {
                                this->blocked_interaction_non_rho(local_expansions_SoA,
                                    center_of_masses_SoA, potential_expansions_SoA,
                                    angular_corrections_SoA, mons, cell_index, cell_flat_index,
                                    cell_index_coarse, cell_index_unpadded,
                                    cell_flat_index_unpadded, stencil, outer_stencil_index);
                            }
                        }
                    }
                }
            }
        }
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
