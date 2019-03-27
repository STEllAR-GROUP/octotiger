#include "octotiger/monopole_interactions/p2p_cpu_kernel.hpp"
#include "octotiger/monopole_interactions/monopole_kernel_templates.hpp"

#include "octotiger/common_kernel/helper.hpp"
#include "octotiger/common_kernel/struct_of_array_data.hpp"

#include "octotiger/defs.hpp"
#include "octotiger/interaction_types.hpp"
#include "octotiger/options.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        p2p_cpu_kernel::p2p_cpu_kernel(std::vector<bool>& neighbor_empty)
          : neighbor_empty(neighbor_empty)
          , theta_rec_squared(sqr(1.0 / opts().theta))
        {
            for (size_t i = 0; i < m2m_int_vector::size(); i++) {
                offset_vector[i] = i;
            }
        }

        void p2p_cpu_kernel::apply_stencil(std::vector<real>& local_expansions,
            struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>&
                potential_expansions_SoA,
            const std::vector<bool>& stencil_masks, const std::vector<std::array<real, 4>>& four, real dx) {
                for (size_t i0 = 0; i0 < INNER_CELLS_PER_DIRECTION; i0++) {
                    for (size_t i1 = 0; i1 < INNER_CELLS_PER_DIRECTION; i1+=2) {
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

                            this->cell_interactions(local_expansions, potential_expansions_SoA,
                                cell_index, cell_flat_index, cell_index_coarse, cell_index_unpadded,
                                cell_flat_index_unpadded, stencil_masks, four, 0, dx);
                        }
                    }
                }
            }

        void p2p_cpu_kernel::cell_interactions(
            std::vector<real>& mons,
            struct_of_array_data<expansion, real, 20, INNER_CELLS,
                SOA_PADDING>& __restrict__ potential_expansions_SoA,    // L
            const multiindex<>& __restrict__ cell_index,
            const size_t cell_flat_index,    /// iii0
            const multiindex<m2m_int_vector>& __restrict__ cell_index_coarse,
            const multiindex<>& __restrict__ cell_index_unpadded,
            const size_t cell_flat_index_unpadded,
            const std::vector<bool>& __restrict__ stencil,
            const std::vector<std::array<real, 4>>& __restrict__ four_constants,
            const size_t outer_stencil_index, real dx) {

            const m2m_vector d_components[2] = {1.0 / dx, -1.0 / sqr(dx)};
            m2m_vector tmpstore1[4];
            tmpstore1[0] = potential_expansions_SoA.value<0>(cell_flat_index_unpadded);
            tmpstore1[1] = potential_expansions_SoA.value<1>(cell_flat_index_unpadded);
            tmpstore1[2] = potential_expansions_SoA.value<2>(cell_flat_index_unpadded);
            tmpstore1[3] = potential_expansions_SoA.value<3>(cell_flat_index_unpadded);
            m2m_vector tmpstore2[4];
            tmpstore1[0] = potential_expansions_SoA.value<0>(cell_flat_index_unpadded);
            tmpstore1[1] = potential_expansions_SoA.value<1>(cell_flat_index_unpadded);
            tmpstore1[2] = potential_expansions_SoA.value<2>(cell_flat_index_unpadded);
            tmpstore1[3] = potential_expansions_SoA.value<3>(cell_flat_index_unpadded);

            bool data_changed = true;
            size_t skipped = 0;
            size_t calculated = 0;

            for (int stencil_x = STENCIL_MIN; stencil_x <= STENCIL_MAX; stencil_x++) {
                int x = stencil_x - STENCIL_MIN;
                for (int stencil_y = STENCIL_MIN; stencil_y <= STENCIL_MAX; stencil_y++) {
                    int y = stencil_y - STENCIL_MIN;
                    for (int stencil_z = STENCIL_MIN; stencil_z <= STENCIL_MAX; stencil_z++) {
                        const size_t index = x * STENCIL_INX * STENCIL_INX + y * STENCIL_INX + (stencil_z - STENCIL_MIN);
                        if (!stencil[index]) {
                            skipped++;
                            continue;
                        }
                        calculated++;

                        const multiindex<> stencil_element(stencil_x, stencil_y, stencil_z);
                        const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                                                                     cell_index.y + stencil_element.y,
                                                                     cell_index.z + stencil_element.z);
                        const multiindex<> interaction_partner_index2(cell_index.x + stencil_element.x,
                                                                     cell_index.y +
                        stencil_element.y + 1,
                                                                     cell_index.z + stencil_element.z);

                        const size_t interaction_partner_flat_index =
                            to_flat_index_padded(interaction_partner_index);    // iii1n
                        // if (vector_is_empty[interaction_partner_flat_index]) {
                        //     continue;
                        // }

                        // implicitly broadcasts to vector
                        multiindex<m2m_int_vector> interaction_partner_index_coarse(
                            interaction_partner_index);
                        multiindex<m2m_int_vector> interaction_partner_index_coarse2(
                            interaction_partner_index2);
                        interaction_partner_index_coarse.z += offset_vector;
                        interaction_partner_index_coarse2.z += offset_vector;
                        interaction_partner_index_coarse.transform_coarse();
                        interaction_partner_index_coarse2.transform_coarse();

                        m2m_int_vector theta_c_rec_squared_int = detail::distance_squared_reciprocal(
                            cell_index_coarse, interaction_partner_index_coarse);
                        m2m_int_vector theta_c_rec_squared_int2 = detail::distance_squared_reciprocal(
                            cell_index_coarse, interaction_partner_index_coarse2);

                        const m2m_vector theta_c_rec_squared =
                            Vc::simd_cast<m2m_vector>(theta_c_rec_squared_int);
                        const m2m_vector theta_c_rec_squared2 =
                            Vc::simd_cast<m2m_vector>(theta_c_rec_squared_int2);

                        const m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;
                        const m2m_vector::mask_type mask2 = theta_rec_squared > theta_c_rec_squared2;
                        if (Vc::none_of(mask) && Vc::none_of(mask2)) {
                            continue;
                        }
                        data_changed = true;
                        m2m_vector monopole;
                        Vc::where(mask, monopole) = m2m_vector(
                            mons.data() + interaction_partner_flat_index);
                        m2m_vector monopole2;
                        Vc::where(mask2, monopole2) = m2m_vector(
                            mons.data() + interaction_partner_flat_index + INX + 10);

                        const m2m_vector four[4] = {
                            four_constants[index][0],
                            four_constants[index][1],
                            four_constants[index][2],
                            four_constants[index][3]};

                        compute_monopole_interaction<m2m_vector>(monopole, tmpstore1, four, d_components);
                        compute_monopole_interaction<m2m_vector>(monopole2, tmpstore2, four, d_components);
                    }
                }
            }

            if (data_changed) {
                tmpstore1[0].store(potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded));
                tmpstore1[1].store(potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded));
                tmpstore1[2].store(potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded));
                tmpstore1[3].store(potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded));
                tmpstore2[0].store(
                    potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded + INX));
                tmpstore2[1].store(
                    potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded + INX));
                tmpstore2[2].store(
                    potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded + INX));
                tmpstore2[3].store(
                    potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded + INX));
            }
        }
    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
