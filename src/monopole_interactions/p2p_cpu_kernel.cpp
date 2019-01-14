#include "octotiger/monopole_interactions/p2p_cpu_kernel.hpp"
#include "octotiger/monopole_interactions/monopole_kernel_templates.hpp"

#include "octotiger/common_kernel/helper.hpp"
#include "octotiger/common_kernel/struct_of_array_data.hpp"

#include "octotiger/defs.hpp"
#include "octotiger/interaction_types.hpp"
#include "octotiger/options.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <vector>

// std::vector<interaction_type> ilist_debugging;

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        p2p_cpu_kernel::p2p_cpu_kernel(std::vector<bool>& neighbor_empty)
          : neighbor_empty(neighbor_empty)
          , theta_rec_squared(sqr(1.0 / opts().theta))
        // , theta_rec_squared_scalar(sqr(1.0 / opts().theta))
        {
            for (size_t i = 0; i < m2m_int_vector::size(); i++) {
                offset_vector[i] = i;
            }
            // calculate_coarse_indices();
            vectors_check_empty();
        }

        void p2p_cpu_kernel::apply_stencil(std::vector<real>& local_expansions,
            struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>&
                potential_expansions_SoA,
            const std::vector<multiindex<>>& stencil, const std::vector<std::array<real, 4>>& four, real dx) {
            for (size_t outer_stencil_index = 0; outer_stencil_index < stencil.size();
                 outer_stencil_index += P2P_STENCIL_BLOCKING) {
                for (size_t i0 = 0; i0 < INNER_CELLS_PER_DIRECTION; i0++) {
                    for (size_t i1 = 0; i1 < INNER_CELLS_PER_DIRECTION; i1 += 2) {
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

                            this->blocked_interaction(local_expansions, potential_expansions_SoA,
                                cell_index, cell_flat_index, cell_index_coarse, cell_index_unpadded,
                                cell_flat_index_unpadded, stencil, four, outer_stencil_index, dx);
                        }
                    }
                }
            }
        }

        void p2p_cpu_kernel::blocked_interaction(
            std::vector<real>& mons,
            struct_of_array_data<expansion, real, 20, INNER_CELLS,
                SOA_PADDING>& __restrict__ potential_expansions_SoA,    // L
            const multiindex<>& __restrict__ cell_index,
            const size_t cell_flat_index,    /// iii0
            const multiindex<m2m_int_vector>& __restrict__ cell_index_coarse,
            const multiindex<>& __restrict__ cell_index_unpadded,
            const size_t cell_flat_index_unpadded,
            const std::vector<multiindex<>>& __restrict__ stencil,
            const std::vector<std::array<real, 4>>& __restrict__ four_constants,
            const size_t outer_stencil_index, real dx) {
            multiindex<m2m_int_vector> cell_index_coarse2(cell_index_coarse);
            for (size_t j = 0; j < m2m_int_vector::size(); j++)
                cell_index_coarse2.y[j] += 1;
            const m2m_vector d_components[2] = {1.0 / dx, -1.0 / sqr(dx)};
            m2m_vector tmpstore1[4];
            tmpstore1[0] = potential_expansions_SoA.value<0>(cell_flat_index_unpadded);
            tmpstore1[1] = potential_expansions_SoA.value<1>(cell_flat_index_unpadded);
            tmpstore1[2] = potential_expansions_SoA.value<2>(cell_flat_index_unpadded);
            tmpstore1[3] = potential_expansions_SoA.value<3>(cell_flat_index_unpadded);
            m2m_vector tmpstore2[4];
            tmpstore2[0] = potential_expansions_SoA.value<0>(cell_flat_index_unpadded + 8);
            tmpstore2[1] = potential_expansions_SoA.value<1>(cell_flat_index_unpadded + 8);
            tmpstore2[2] = potential_expansions_SoA.value<2>(cell_flat_index_unpadded + 8);
            tmpstore2[3] = potential_expansions_SoA.value<3>(cell_flat_index_unpadded + 8);

            bool data_changed = false;
            for (size_t inner_stencil_index = 0; inner_stencil_index < P2P_STENCIL_BLOCKING &&
                 outer_stencil_index + inner_stencil_index < stencil.size();
                 inner_stencil_index +=
                 1) {    // blocking is done by stepping in die outer_stencil index
                const multiindex<>& stencil_element =
                    stencil[outer_stencil_index + inner_stencil_index];
                const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                    cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);
                const multiindex<> interaction_partner_index2(cell_index.x + stencil_element.x,
                    cell_index.y + stencil_element.y + 1, cell_index.z + stencil_element.z);

                const size_t interaction_partner_flat_index =
                    to_flat_index_padded(interaction_partner_index);    // iii1n
                if (vector_is_empty[interaction_partner_flat_index]) {
                    continue;
                }

                // implicitly broadcasts to vector
                multiindex<m2m_int_vector> interaction_partner_index_coarse(
                    interaction_partner_index);
                multiindex<m2m_int_vector> interaction_partner_index_coarse2(
                    interaction_partner_index2);
                interaction_partner_index_coarse.z += offset_vector;
                interaction_partner_index_coarse2.z += offset_vector;
                // note that this is the same for groups of 2x2x2 elements
                // -> maps to the same for some SIMD lanes
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
                    mons.data() + interaction_partner_flat_index, Vc::Aligned);
                m2m_vector monopole2;
                Vc::where(mask2, monopole2) = m2m_vector(
                    mons.data() + interaction_partner_flat_index + 24, Vc::Aligned);

                const m2m_vector four[4] = {
                    four_constants[outer_stencil_index + inner_stencil_index][0],
                    four_constants[outer_stencil_index + inner_stencil_index][1],
                    four_constants[outer_stencil_index + inner_stencil_index][2],
                    four_constants[outer_stencil_index + inner_stencil_index][3]};

                compute_monopole_interaction<m2m_vector>(monopole, tmpstore1, four, d_components);
                compute_monopole_interaction<m2m_vector>(monopole2, tmpstore2, four, d_components);
            }
            if (data_changed) {
                tmpstore1[0].store(potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore1[1].store(potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore1[2].store(potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore1[3].store(potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore2[0].store(
                    potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded + 8),
                    Vc::Aligned);
                tmpstore2[1].store(
                    potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded + 8),
                    Vc::Aligned);
                tmpstore2[2].store(
                    potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded + 8),
                    Vc::Aligned);
                tmpstore2[3].store(
                    potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded + 8),
                    Vc::Aligned);
            }
        }

        void p2p_cpu_kernel::vectors_check_empty() {
            vector_is_empty = std::vector<bool>(PADDED_STRIDE * PADDED_STRIDE * PADDED_STRIDE);
            for (size_t i0 = 0; i0 < PADDED_STRIDE; i0 += 1) {
                for (size_t i1 = 0; i1 < PADDED_STRIDE; i1 += 1) {
                    for (size_t i2 = 0; i2 < PADDED_STRIDE; i2 += 1) {
                        const multiindex<> cell_index(i0, i1, i2);
                        const int64_t cell_flat_index = to_flat_index_padded(cell_index);
                        // std::cout << "cell_flat_index: " << cell_flat_index << std::endl;

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
                            // std::cout << "prepare true cell_index:" << cell_index << std::endl;
                            // std::cout << "cell_flat_index: " << cell_flat_index << std::endl;
                            // std::cout << "dir_start.flat_index_with_center(): "
                            //           << dir_start.flat_index_with_center() << std::endl;
                            // std::cout << "dir_end.flat_index_with_center(): "
                            //           << dir_end.flat_index_with_center() << std::endl;
                            // std::cout << "in_boundary_end: " << in_boundary_end << std::endl;

                        } else {
                            vector_is_empty[cell_flat_index] = false;
                            // std::cout << "prepare false cell_index:" << cell_index << std::endl;
                            // std::cout << "cell_flat_index: " << cell_flat_index << std::endl;
                            // std::cout << "dir_start.flat_index_with_center(): "
                            //           << dir_start.flat_index_with_center() << std::endl;
                            // std::cout << "dir_end.flat_index_with_center(): "
                            //           << dir_end.flat_index_with_center() << std::endl;
                        }
                    }
                }
            }
        }
    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
