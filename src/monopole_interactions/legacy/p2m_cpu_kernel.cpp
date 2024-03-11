//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef OCTOTIGER_HAVE_VC
#include "octotiger/monopole_interactions/legacy/p2m_cpu_kernel.hpp"

#include "octotiger/common_kernel/helper.hpp"
#include "octotiger/common_kernel/kernel_taylor_set_basis.hpp"
#include "octotiger/common_kernel/struct_of_array_data.hpp"

#include "octotiger/defs.hpp"
#include "octotiger/interaction_types.hpp"
#include "octotiger/options.hpp"
#include "octotiger/real.hpp"

#include <array>
#include <cstddef>

#include "octotiger/monopole_interactions/kernel/monopole_kernel_templates.hpp"

// std::vector<interaction_type> ilist_debugging;

extern taylor<4, real> factor;
extern taylor<4, m2m_vector> factor_half_v;
extern taylor<4, m2m_vector> factor_sixth_v;

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        p2m_kernel::p2m_kernel()
          : theta_rec_squared(sqr(1.0 / opts().theta)) {
            for (size_t i = 0; i < m2m_int_vector::size(); i++) {
                offset_vector[i] = i;
            }
        }

        void p2m_kernel::apply_stencil(const cpu_expansion_buffer_t& local_expansions_SoA,
            const cpu_space_vector_buffer_t& center_of_masses_SoA,
            cpu_expansion_result_buffer_t& potential_expansions_SoA,
            cpu_angular_result_t& angular_corrections_SoA, const std::vector<multiindex<>>& stencil,
            gsolve_type type, bool (&z_skip)[3][3][3], bool (&y_skip)[3][3], bool (&x_skip)[3]) {
            // for (multiindex<>& stencil_element : stencil) {
            for (size_t outer_stencil_index = 0; outer_stencil_index < stencil.size();
                 outer_stencil_index += 1) {
                const multiindex<>& stencil_element = stencil[outer_stencil_index];
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

                            const multiindex<> interaction_partner_index(
                                cell_index.x + stencil_element.x, cell_index.y + stencil_element.y,
                                cell_index.z + stencil_element.z);

                            const size_t interaction_partner_flat_index =
                                to_flat_index_padded(interaction_partner_index);    // iii1n

                            // implicitly broadcasts to vector
                            multiindex<m2m_int_vector> interaction_partner_index_coarse(
                                interaction_partner_index);
                            interaction_partner_index_coarse.z += offset_vector;
                            // note that this is the same for groups of 2x2x2 elements
                            // -> maps to the same for some SIMD lanes
                            interaction_partner_index_coarse.transform_coarse();

                            // calculate position of the monopole

                            if (type == RHO) {
                                this->blocked_interaction_rho(local_expansions_SoA,
                                    center_of_masses_SoA, potential_expansions_SoA,
                                    angular_corrections_SoA, cell_index, cell_flat_index,
                                    cell_index_coarse, cell_index_unpadded,
                                    cell_flat_index_unpadded, interaction_partner_index,
                                    interaction_partner_flat_index,
                                    interaction_partner_index_coarse);
                            } else {
                                this->blocked_interaction_non_rho(local_expansions_SoA,
                                    center_of_masses_SoA, potential_expansions_SoA,
                                    angular_corrections_SoA, cell_index, cell_flat_index,
                                    cell_index_coarse, cell_index_unpadded,
                                    cell_flat_index_unpadded, interaction_partner_index,
                                    interaction_partner_flat_index,
                                    interaction_partner_index_coarse);
                            }
                        }
                    }
                }
            }
        }

        void p2m_kernel::blocked_interaction_rho(const cpu_expansion_buffer_t& local_expansions_SoA,
            const cpu_space_vector_buffer_t& center_of_masses_SoA,
            cpu_expansion_result_buffer_t& potential_expansions_SoA,
            cpu_angular_result_t& angular_corrections_SoA,
            const multiindex<>& __restrict__ cell_index, const size_t cell_flat_index,
            const multiindex<m2m_int_vector>& __restrict__ cell_index_coarse,
            const multiindex<>& __restrict__ cell_index_unpadded,
            const size_t cell_flat_index_unpadded, const multiindex<>& interaction_partner_index,
            const size_t interaction_partner_flat_index,
            multiindex<m2m_int_vector>& interaction_partner_index_coarse) {
            m2m_vector X[NDIM];
            X[0] = center_of_masses_SoA.value<0, m2m_vector>(cell_flat_index);
            X[1] = center_of_masses_SoA.value<1, m2m_vector>(cell_flat_index);
            X[2] = center_of_masses_SoA.value<2, m2m_vector>(cell_flat_index);
            m2m_vector tmpstore[4];
            m2m_vector tmp_corrections[3];

            m2m_int_vector theta_c_rec_squared_int = detail::distance_squared_reciprocal(
                cell_index_coarse, interaction_partner_index_coarse);

            m2m_vector theta_c_rec_squared = Vc::simd_cast<m2m_vector>(theta_c_rec_squared_int);

            m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;

            if (Vc::none_of(mask)) {
                return;
            }

            m2m_vector Y[NDIM];
            Y[0] = center_of_masses_SoA.value<0, m2m_vector>(interaction_partner_flat_index);
            Y[1] = center_of_masses_SoA.value<1, m2m_vector>(interaction_partner_flat_index);
            Y[2] = center_of_masses_SoA.value<2, m2m_vector>(interaction_partner_flat_index);

            m2m_vector m_partner[20];
            m2m_vector cur_pot[4];

            Vc::where(mask, m_partner[0]) =
                local_expansions_SoA.value<0, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[4]) =
                local_expansions_SoA.value<4, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[5]) =
                local_expansions_SoA.value<5, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[6]) =
                local_expansions_SoA.value<6, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[7]) =
                local_expansions_SoA.value<7, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[8]) =
                local_expansions_SoA.value<8, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[9]) =
                local_expansions_SoA.value<9, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[10]) =
                local_expansions_SoA.value<10, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[11]) =
                local_expansions_SoA.value<11, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[12]) =
                local_expansions_SoA.value<12, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[13]) =
                local_expansions_SoA.value<13, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[14]) =
                local_expansions_SoA.value<14, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[15]) =
                local_expansions_SoA.value<15, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[16]) =
                local_expansions_SoA.value<16, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[17]) =
                local_expansions_SoA.value<17, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[18]) =
                local_expansions_SoA.value<18, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[19]) =
                local_expansions_SoA.value<19, m2m_vector>(interaction_partner_flat_index);

            compute_kernel_p2m_rho(X, Y, m_partner, tmpstore, tmp_corrections,
                [](const m2m_vector& one, const m2m_vector& two) -> m2m_vector {
                    return Vc::max(one, two);
                });

            tmpstore[0] = tmpstore[0] +
                potential_expansions_SoA.value<0, m2m_vector>(cell_flat_index_unpadded);
            tmpstore[1] = tmpstore[1] +
                potential_expansions_SoA.value<1, m2m_vector>(cell_flat_index_unpadded);
            tmpstore[2] = tmpstore[2] +
                potential_expansions_SoA.value<2, m2m_vector>(cell_flat_index_unpadded);
            tmpstore[3] = tmpstore[3] +
                potential_expansions_SoA.value<3, m2m_vector>(cell_flat_index_unpadded);
            tmpstore[0].store(potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded));
            tmpstore[1].store(potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded));
            tmpstore[2].store(potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded));
            tmpstore[3].store(potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded));

            tmp_corrections[0] = tmp_corrections[0] +
                angular_corrections_SoA.value<0, m2m_vector>(cell_flat_index_unpadded);
            tmp_corrections[1] = tmp_corrections[1] +
                angular_corrections_SoA.value<1, m2m_vector>(cell_flat_index_unpadded);
            tmp_corrections[2] = tmp_corrections[2] +
                angular_corrections_SoA.value<2, m2m_vector>(cell_flat_index_unpadded);
            tmp_corrections[0].store(angular_corrections_SoA.pointer<0>(cell_flat_index_unpadded));
            tmp_corrections[1].store(angular_corrections_SoA.pointer<1>(cell_flat_index_unpadded));
            tmp_corrections[2].store(angular_corrections_SoA.pointer<2>(cell_flat_index_unpadded));
        }

        void p2m_kernel::blocked_interaction_non_rho(
            const cpu_expansion_buffer_t& local_expansions_SoA,
            const cpu_space_vector_buffer_t& center_of_masses_SoA,
            cpu_expansion_result_buffer_t& potential_expansions_SoA,
            cpu_angular_result_t& angular_corrections_SoA, const multiindex<>& cell_index,
            const size_t cell_flat_index, const multiindex<m2m_int_vector>& cell_index_coarse,
            const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
            const multiindex<>& interaction_partner_index,
            const size_t interaction_partner_flat_index,
            multiindex<m2m_int_vector>& interaction_partner_index_coarse) {
            m2m_vector X[NDIM];
            X[0] = center_of_masses_SoA.value<0, m2m_vector>(cell_flat_index);
            X[1] = center_of_masses_SoA.value<1, m2m_vector>(cell_flat_index);
            X[2] = center_of_masses_SoA.value<2, m2m_vector>(cell_flat_index);
            m2m_vector tmpstore[4];

            m2m_int_vector theta_c_rec_squared_int = detail::distance_squared_reciprocal(
                cell_index_coarse, interaction_partner_index_coarse);

            m2m_vector theta_c_rec_squared = Vc::simd_cast<m2m_vector>(theta_c_rec_squared_int);

            m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;

            if (Vc::none_of(mask)) {
                return;
            }

            m2m_vector Y[NDIM];
            Y[0] = center_of_masses_SoA.value<0, m2m_vector>(interaction_partner_flat_index);
            Y[1] = center_of_masses_SoA.value<1, m2m_vector>(interaction_partner_flat_index);
            Y[2] = center_of_masses_SoA.value<2, m2m_vector>(interaction_partner_flat_index);

            m2m_vector m_partner[20];

            // Array to store the temporary result - was called A in the old style
            Vc::where(mask, m_partner[0]) =
                local_expansions_SoA.value<0, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[1]) =
                local_expansions_SoA.value<1, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[2]) =
                local_expansions_SoA.value<2, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[3]) =
                local_expansions_SoA.value<3, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[4]) =
                local_expansions_SoA.value<4, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[5]) =
                local_expansions_SoA.value<5, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[6]) =
                local_expansions_SoA.value<6, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[7]) =
                local_expansions_SoA.value<7, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[8]) =
                local_expansions_SoA.value<8, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[9]) =
                local_expansions_SoA.value<9, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[10]) =
                local_expansions_SoA.value<10, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[11]) =
                local_expansions_SoA.value<11, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[12]) =
                local_expansions_SoA.value<12, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[13]) =
                local_expansions_SoA.value<13, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[14]) =
                local_expansions_SoA.value<14, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[15]) =
                local_expansions_SoA.value<15, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[16]) =
                local_expansions_SoA.value<16, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[17]) =
                local_expansions_SoA.value<17, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[18]) =
                local_expansions_SoA.value<18, m2m_vector>(interaction_partner_flat_index);
            Vc::where(mask, m_partner[19]) =
                local_expansions_SoA.value<19, m2m_vector>(interaction_partner_flat_index);

            compute_kernel_p2m_non_rho(X, Y, m_partner, tmpstore,
                [](const m2m_vector& one, const m2m_vector& two) -> m2m_vector {
                    return Vc::max(one, two);
                });

            tmpstore[0] = tmpstore[0] +
                potential_expansions_SoA.value<0, m2m_vector>(cell_flat_index_unpadded);
            tmpstore[1] = tmpstore[1] +
                potential_expansions_SoA.value<1, m2m_vector>(cell_flat_index_unpadded);
            tmpstore[2] = tmpstore[2] +
                potential_expansions_SoA.value<2, m2m_vector>(cell_flat_index_unpadded);
            tmpstore[3] = tmpstore[3] +
                potential_expansions_SoA.value<3, m2m_vector>(cell_flat_index_unpadded);
            tmpstore[0].store(potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded));
            tmpstore[1].store(potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded));
            tmpstore[2].store(potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded));
            tmpstore[3].store(potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded));
        }

        template <size_t buffer_size>
        void neighbor_interaction_rho(const multiindex<>& neighbor_size,
            const multiindex<>& start_index, const multiindex<>& end_index,
            const struct_of_array_data<expansion, real, 20, buffer_size, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&
                local_expansions_SoA,
            const struct_of_array_data<space_vector, real, 3, buffer_size, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&
                center_of_masses_SoA,
            const struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&
                center_of_masses_inner_cells_SoA,
            cpu_expansion_result_buffer_t& potential_expansions_SoA,
            cpu_angular_result_t& angular_corrections_SoA, const multiindex<>& cell_index,
            const size_t cell_flat_index, const multiindex<m2m_int_vector>& cell_index_coarse,
            const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
            const std::vector<bool>& stencil_masks, const geo::direction& dir, const double theta) {
            // Load position and init tmp stores & constants
            const m2m_vector theta_rec_squared(sqr(1.0 / theta));
            m2m_vector X[NDIM];
            X[0] = center_of_masses_inner_cells_SoA.template value<0, m2m_vector>(
                cell_flat_index_unpadded);
            X[1] = center_of_masses_inner_cells_SoA.template value<1, m2m_vector>(
                cell_flat_index_unpadded);
            X[2] = center_of_masses_inner_cells_SoA.template value<2, m2m_vector>(
                cell_flat_index_unpadded);
            m2m_vector tmpstore[4];
            m2m_vector tmp_corrections[3];

            for (size_t x = start_index.x; x < end_index.x; x++) {
                for (size_t y = start_index.y; y < end_index.y; y++) {
                    for (size_t z = start_index.z; z < end_index.z; z++) {
                        // Global index (regarding inner cells + all neighbors)
                        // Used to figure out which stencil mask to use
                        const multiindex<> interaction_partner_index(
                            INNER_CELLS_PADDING_DEPTH + dir[0] * INNER_CELLS_PADDING_DEPTH + x,
                            INNER_CELLS_PADDING_DEPTH + dir[1] * INNER_CELLS_PADDING_DEPTH + y,
                            INNER_CELLS_PADDING_DEPTH + dir[2] * INNER_CELLS_PADDING_DEPTH + z);
                        const size_t global_flat_index =
                            to_flat_index_padded(interaction_partner_index);

                        // Stencil element for first element of the SIMD lane
                        multiindex<> stencil_element(
                            interaction_partner_index.x - cell_index.x - STENCIL_MIN,
                            interaction_partner_index.y - cell_index.y - STENCIL_MIN,
                            interaction_partner_index.z - cell_index.z - STENCIL_MIN);
                        const int stencil_flat_index =
                            stencil_element.x * STENCIL_INX * STENCIL_INX +
                            stencil_element.y * STENCIL_INX + stencil_element.z;
                        // Generate stencil mask
                        m2m_vector::mask_type stencil_mask;
                        for (int i = 0; i < m2m_vector::size(); i++) {
                             if (stencil_flat_index - i >= 0 && stencil_flat_index - i < FULL_STENCIL_SIZE)
                                stencil_mask[i] = stencil_masks[stencil_flat_index - i];
                             else
                                stencil_mask[i] = false;
                             if (cell_index_unpadded.z + i >= INX)
                                stencil_mask[i] = false;
                        }
                        // Skip with stencil masks are all 0
                        if (Vc::none_of(stencil_mask)) {
                            continue;
                        }

                        // Set mask element by element (obtain simd::length stencil elements)
                        // Note: All cell_index elements of the SIMD lanes try to interact with the
                        // SAME neighbor cell unlike in the other interaction kernels
                        multiindex<m2m_int_vector> interaction_partner_index_coarse(
                            interaction_partner_index);
                        interaction_partner_index_coarse.transform_coarse();
                        m2m_int_vector theta_c_rec_squared_int =
                            detail::distance_squared_reciprocal(
                                cell_index_coarse, interaction_partner_index_coarse);
                        m2m_vector theta_c_rec_squared =
                            Vc::simd_cast<m2m_vector>(theta_c_rec_squared_int);
                        m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;

                        // Skip with stencil masks are all 0
                        if (Vc::none_of(mask)) {
                            continue;
                        }
                        // combine masks
                        mask = mask && stencil_mask;

                        // Local index
                        // Used to figure out which data element to use
                        const multiindex<> interaction_partner_data_index(
                            x - start_index.x, y - start_index.y, z - start_index.z);
                        const size_t interaction_partner_flat_index =
                            interaction_partner_data_index.x * (neighbor_size.y * neighbor_size.z) +
                            interaction_partner_data_index.y * neighbor_size.z +
                            interaction_partner_data_index.z;

                        // Load required data from interaction partner and broadcast into simd
                        // arrays
                        m2m_vector Y[NDIM];
                        Y[0] = center_of_masses_SoA.template at<0>(interaction_partner_flat_index);
                        Y[1] = center_of_masses_SoA.template at<1>(interaction_partner_flat_index);
                        Y[2] = center_of_masses_SoA.template at<2>(interaction_partner_flat_index);
                        m2m_vector m_partner[20];
                        Vc::where(mask, m_partner[0]) =
                            local_expansions_SoA.template at<0>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[1]) =
                            local_expansions_SoA.template at<1>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[2]) =
                            local_expansions_SoA.template at<2>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[3]) =
                            local_expansions_SoA.template at<3>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[4]) =
                            local_expansions_SoA.template at<4>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[5]) =
                            local_expansions_SoA.template at<5>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[6]) =
                            local_expansions_SoA.template at<6>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[7]) =
                            local_expansions_SoA.template at<7>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[8]) =
                            local_expansions_SoA.template at<8>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[9]) =
                            local_expansions_SoA.template at<9>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[10]) =
                            local_expansions_SoA.template at<10>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[11]) =
                            local_expansions_SoA.template at<11>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[12]) =
                            local_expansions_SoA.template at<12>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[13]) =
                            local_expansions_SoA.template at<13>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[14]) =
                            local_expansions_SoA.template at<14>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[15]) =
                            local_expansions_SoA.template at<15>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[16]) =
                            local_expansions_SoA.template at<16>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[17]) =
                            local_expansions_SoA.template at<17>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[18]) =
                            local_expansions_SoA.template at<18>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[19]) =
                            local_expansions_SoA.template at<19>(interaction_partner_flat_index);

                        // run templated interaction method instanced with Vc type
                        compute_kernel_p2m_rho(X, Y, m_partner, tmpstore, tmp_corrections,
                            [](const m2m_vector& one, const m2m_vector& two) -> m2m_vector {
                                return Vc::max(one, two);
                            });
                    }
                }
            }
            // Move data back into the potential expansions buffer
            tmpstore[0] = tmpstore[0] +
                potential_expansions_SoA.template value<0, m2m_vector>(cell_flat_index_unpadded);
            tmpstore[1] = tmpstore[1] +
                potential_expansions_SoA.template value<1, m2m_vector>(cell_flat_index_unpadded);
            tmpstore[2] = tmpstore[2] +
                potential_expansions_SoA.template value<2, m2m_vector>(cell_flat_index_unpadded);
            tmpstore[3] = tmpstore[3] +
                potential_expansions_SoA.template value<3, m2m_vector>(cell_flat_index_unpadded);
            tmpstore[0].store(
                potential_expansions_SoA.template pointer<0>(cell_flat_index_unpadded));
            tmpstore[1].store(
                potential_expansions_SoA.template pointer<1>(cell_flat_index_unpadded));
            tmpstore[2].store(
                potential_expansions_SoA.template pointer<2>(cell_flat_index_unpadded));
            tmpstore[3].store(
                potential_expansions_SoA.template pointer<3>(cell_flat_index_unpadded));

            // Move data back into the angular corrections buffer
            tmp_corrections[0] = tmp_corrections[0] +
                angular_corrections_SoA.template value<0, m2m_vector>(cell_flat_index_unpadded);
            tmp_corrections[1] = tmp_corrections[1] +
                angular_corrections_SoA.template value<1, m2m_vector>(cell_flat_index_unpadded);
            tmp_corrections[2] = tmp_corrections[2] +
                angular_corrections_SoA.template value<2, m2m_vector>(cell_flat_index_unpadded);
            tmp_corrections[0].store(
                angular_corrections_SoA.template pointer<0>(cell_flat_index_unpadded));
            tmp_corrections[1].store(
                angular_corrections_SoA.template pointer<1>(cell_flat_index_unpadded));
            tmp_corrections[2].store(
                angular_corrections_SoA.template pointer<2>(cell_flat_index_unpadded));
        }

        template <size_t buffer_size>
        void neighbor_interaction_non_rho(const multiindex<>& neighbor_size,
            const multiindex<>& start_index, const multiindex<>& end_index,
            const struct_of_array_data<expansion, real, 20, buffer_size, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&
                local_expansions_SoA,
            const struct_of_array_data<space_vector, real, 3, buffer_size, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&
                center_of_masses_SoA,
            const struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&
                center_of_masses_inner_cells_SoA,
            cpu_expansion_result_buffer_t& potential_expansions_SoA, const multiindex<>& cell_index,
            const size_t cell_flat_index, const multiindex<m2m_int_vector>& cell_index_coarse,
            const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
            const std::vector<bool>& stencil_masks, const geo::direction& dir, const double theta) {
            const m2m_vector theta_rec_squared(sqr(1.0 / theta));
            m2m_vector X[NDIM];
            X[0] = center_of_masses_inner_cells_SoA.template value<0, m2m_vector>(
                cell_flat_index_unpadded);
            X[1] = center_of_masses_inner_cells_SoA.template value<1, m2m_vector>(
                cell_flat_index_unpadded);
            X[2] = center_of_masses_inner_cells_SoA.template value<2, m2m_vector>(
                cell_flat_index_unpadded);
            m2m_vector tmpstore[4];

            for (size_t x = start_index.x; x < end_index.x; x++) {
                for (size_t y = start_index.y; y < end_index.y; y++) {
                    for (size_t z = start_index.z; z < end_index.z; z++) {
                        // Global index (regarding inner cells + all neighbors)
                        // Used to figure out which stencil mask to use
                        const multiindex<> interaction_partner_index(
                            INNER_CELLS_PADDING_DEPTH + dir[0] * INNER_CELLS_PADDING_DEPTH + x,
                            INNER_CELLS_PADDING_DEPTH + dir[1] * INNER_CELLS_PADDING_DEPTH + y,
                            INNER_CELLS_PADDING_DEPTH + dir[2] * INNER_CELLS_PADDING_DEPTH + z);
                        const size_t global_flat_index =
                            to_flat_index_padded(interaction_partner_index);

                        // Stencil element for first element of the SIMD lane
                        multiindex<> stencil_element(
                            interaction_partner_index.x - cell_index.x - STENCIL_MIN,
                            interaction_partner_index.y - cell_index.y - STENCIL_MIN,
                            interaction_partner_index.z - cell_index.z - STENCIL_MIN);
                        const int stencil_flat_index =
                            stencil_element.x * STENCIL_INX * STENCIL_INX +
                            stencil_element.y * STENCIL_INX + stencil_element.z;
                        // Generate stencil mask
                        m2m_vector::mask_type stencil_mask;
                        for (int i = 0; i < m2m_vector::size(); i++) {
                             if (stencil_flat_index - i >= 0 && stencil_flat_index - i < FULL_STENCIL_SIZE)
                                stencil_mask[i] = stencil_masks[stencil_flat_index - i];
                             else
                                stencil_mask[i] = false;
                             if (cell_index_unpadded.z + i >= INX)
                                stencil_mask[i] = false;
                        }
                        // Skip with stencil masks are all 0
                        if (Vc::none_of(stencil_mask)) {
                            continue;
                        }

                        // Set mask element by element (obtain simd::length stencil elements)
                        // Note: All cell_index elements of the SIMD lanes try to interact with the
                        // SAME neighbor cell unlike in the other interaction kernels
                        multiindex<m2m_int_vector> interaction_partner_index_coarse(
                            interaction_partner_index);
                        interaction_partner_index_coarse.transform_coarse();
                        m2m_int_vector theta_c_rec_squared_int =
                            detail::distance_squared_reciprocal(
                                cell_index_coarse, interaction_partner_index_coarse);
                        m2m_vector theta_c_rec_squared =
                            Vc::simd_cast<m2m_vector>(theta_c_rec_squared_int);
                        m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;

                        // Skip with stencil masks are all 0
                        if (Vc::none_of(mask)) {
                            continue;
                        }
                        // combine masks
                        mask = mask && stencil_mask;

                        // Local index
                        // Used to figure out which data element to use
                        const multiindex<> interaction_partner_data_index(
                            x - start_index.x, y - start_index.y, z - start_index.z);
                        const size_t interaction_partner_flat_index =
                            interaction_partner_data_index.x * (neighbor_size.y * neighbor_size.z) +
                            interaction_partner_data_index.y * neighbor_size.z +
                            interaction_partner_data_index.z;

                        // Load required data from interaction partner and broadcast into simd
                        // arrays
                        m2m_vector Y[NDIM];
                        Y[0] = center_of_masses_SoA.template at<0>(interaction_partner_flat_index);
                        Y[1] = center_of_masses_SoA.template at<1>(interaction_partner_flat_index);
                        Y[2] = center_of_masses_SoA.template at<2>(interaction_partner_flat_index);
                        m2m_vector m_partner[20];
                        Vc::where(mask, m_partner[0]) =
                            local_expansions_SoA.template at<0>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[1]) =
                            local_expansions_SoA.template at<1>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[2]) =
                            local_expansions_SoA.template at<2>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[3]) =
                            local_expansions_SoA.template at<3>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[4]) =
                            local_expansions_SoA.template at<4>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[5]) =
                            local_expansions_SoA.template at<5>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[6]) =
                            local_expansions_SoA.template at<6>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[7]) =
                            local_expansions_SoA.template at<7>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[8]) =
                            local_expansions_SoA.template at<8>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[9]) =
                            local_expansions_SoA.template at<9>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[10]) =
                            local_expansions_SoA.template at<10>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[11]) =
                            local_expansions_SoA.template at<11>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[12]) =
                            local_expansions_SoA.template at<12>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[13]) =
                            local_expansions_SoA.template at<13>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[14]) =
                            local_expansions_SoA.template at<14>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[15]) =
                            local_expansions_SoA.template at<15>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[16]) =
                            local_expansions_SoA.template at<16>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[17]) =
                            local_expansions_SoA.template at<17>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[18]) =
                            local_expansions_SoA.template at<18>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[19]) =
                            local_expansions_SoA.template at<19>(interaction_partner_flat_index);

                        // run templated interaction method instanced with Vc type
                        compute_kernel_p2m_non_rho(X, Y, m_partner, tmpstore,
                            [](const m2m_vector& one, const m2m_vector& two) -> m2m_vector {
                                return Vc::max(one, two);
                            });
                    }
                }
            }
            // Move data back into the potential expansions buffer
            tmpstore[0] = tmpstore[0] +
                potential_expansions_SoA.template value<0, m2m_vector>(cell_flat_index_unpadded);
            tmpstore[1] = tmpstore[1] +
                potential_expansions_SoA.template value<1, m2m_vector>(cell_flat_index_unpadded);
            tmpstore[2] = tmpstore[2] +
                potential_expansions_SoA.template value<2, m2m_vector>(cell_flat_index_unpadded);
            tmpstore[3] = tmpstore[3] +
                potential_expansions_SoA.template value<3, m2m_vector>(cell_flat_index_unpadded);
            tmpstore[0].store(
                potential_expansions_SoA.template pointer<0>(cell_flat_index_unpadded));
            tmpstore[1].store(
                potential_expansions_SoA.template pointer<1>(cell_flat_index_unpadded));
            tmpstore[2].store(
                potential_expansions_SoA.template pointer<2>(cell_flat_index_unpadded));
            tmpstore[3].store(
                potential_expansions_SoA.template pointer<3>(cell_flat_index_unpadded));
        }

        template <size_t buffer_size>
        void p2m_kernel::apply_stencil_neighbor(const multiindex<>& neighbor_size,
            const multiindex<>& neighbor_start_index, const multiindex<>& neighbor_end_index,
            const struct_of_array_data<expansion, real, 20, buffer_size, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&
                local_expansions_SoA,
            const struct_of_array_data<space_vector, real, 3, buffer_size, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&
                center_of_masses_SoA,
            const struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&
                center_of_masses_inner_cells_SoA,
            cpu_expansion_result_buffer_t& potential_expansions_SoA,
            cpu_angular_result_t& angular_corrections_SoA, const std::vector<bool>& stencil_masks,
            gsolve_type type, const geo::direction& dir) {
            const double theta = opts().theta;
            // Depending on the location of the current neighbor we don't need to look 
            // at the complete subgrid
            int startx = 0;
            if (dir[0] == 1)
              startx = INX - (STENCIL_MAX + 1);
            int endx = INX;
            if (dir[0] == -1)
              endx = (STENCIL_MAX + 1);
            int starty = 0;
            if (dir[1] == 1)
              starty = INX - (STENCIL_MAX + 1);
            int endy = INX;
            if (dir[1] == -1)
              endy = (STENCIL_MAX + 1);
            int startz = 0;
            if (dir[2] == 1)
              startz = INX - (STENCIL_MAX + 1);
            int endz = INX;
            if (dir[2] == -1)
              endz = (STENCIL_MAX + 1);
            // Iterate over required cells and call kernel
            for (size_t i0 = startx; i0 < endx; i0++) {
                for (size_t i1 = starty; i1 < endy; i1++) {
                    for (size_t i2 = startz; i2 < endz; i2 += m2m_vector::size()) {
                        const multiindex<> cell_index(i0 + INNER_CELLS_PADDING_DEPTH,
                            i1 + INNER_CELLS_PADDING_DEPTH, i2 + INNER_CELLS_PADDING_DEPTH);
                        const int64_t cell_flat_index =
                            to_flat_index_padded(cell_index);    // iii0...
                        const multiindex<> cell_index_unpadded(i0, i1, i2);
                        const int64_t cell_flat_index_unpadded =
                            to_inner_flat_index_not_padded(cell_index_unpadded);
                        multiindex<m2m_int_vector> cell_index_coarse(cell_index);
                        for (size_t j = 0; j < m2m_int_vector::size(); j++) {
                            cell_index_coarse.z[j] += j;
                        }
                        // note that this is the same for groups of 2x2x2 elements
                        // -> maps to the same for some SIMD lanes
                        cell_index_coarse.transform_coarse();

                        if (type == RHO) {
                            neighbor_interaction_rho<buffer_size>(neighbor_size,
                                neighbor_start_index, neighbor_end_index, local_expansions_SoA,
                                center_of_masses_SoA, center_of_masses_inner_cells_SoA,
                                potential_expansions_SoA, angular_corrections_SoA, cell_index,
                                cell_flat_index, cell_index_coarse, cell_index_unpadded,
                                cell_flat_index_unpadded, stencil_masks, dir, theta);
                        } else {
                            neighbor_interaction_non_rho<buffer_size>(neighbor_size,
                                neighbor_start_index, neighbor_end_index, local_expansions_SoA,
                                center_of_masses_SoA, center_of_masses_inner_cells_SoA,
                                potential_expansions_SoA, cell_index, cell_flat_index,
                                cell_index_coarse, cell_index_unpadded, cell_flat_index_unpadded,
                                stencil_masks, dir, theta);
                        }
                    }
                }
            }
        }
        // Required template instances (as template declaration is in the header)
        template void p2m_kernel::apply_stencil_neighbor<((INX == STENCIL_MAX)? INX+1 : INX) * ((INX == STENCIL_MAX)? INX+1 : INX) * STENCIL_MAX>(
            const multiindex<>&, const multiindex<>&, const multiindex<>&,
            const struct_of_array_data<expansion, real, 20, ((INX == STENCIL_MAX)? INX+1 : INX) * ((INX == STENCIL_MAX)? INX+1 : INX) * STENCIL_MAX, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&,
            const struct_of_array_data<space_vector, real, 3, ((INX == STENCIL_MAX)? INX+1 : INX) * ((INX == STENCIL_MAX)? INX+1 : INX) * STENCIL_MAX, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&,
            const struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&,
            cpu_expansion_result_buffer_t&, cpu_angular_result_t&, const std::vector<bool>&,
            gsolve_type, const geo::direction&);
        template void p2m_kernel::apply_stencil_neighbor<((INX == STENCIL_MAX)? INX+1 : INX) * STENCIL_MAX * STENCIL_MAX>(
            const multiindex<>&, const multiindex<>&, const multiindex<>&,
            const struct_of_array_data<expansion, real, 20, ((INX == STENCIL_MAX)? INX+1 : INX) * STENCIL_MAX * STENCIL_MAX, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&,
            const struct_of_array_data<space_vector, real, 3, ((INX == STENCIL_MAX)? INX+1 : INX) * STENCIL_MAX * STENCIL_MAX, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&,
            const struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&,
            cpu_expansion_result_buffer_t&, cpu_angular_result_t&, const std::vector<bool>&,
            gsolve_type, const geo::direction&);
        template void p2m_kernel::apply_stencil_neighbor<STENCIL_MAX * STENCIL_MAX * STENCIL_MAX>(
            const multiindex<>&, const multiindex<>&, const multiindex<>&,
            const struct_of_array_data<expansion, real, 20, STENCIL_MAX * STENCIL_MAX * STENCIL_MAX, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&,
            const struct_of_array_data<space_vector, real, 3, STENCIL_MAX * STENCIL_MAX * STENCIL_MAX, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&,
            const struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&,
            cpu_expansion_result_buffer_t&, cpu_angular_result_t&, const std::vector<bool>&,
            gsolve_type, const geo::direction&);
    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
