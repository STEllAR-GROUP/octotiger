#include "multipole_cpu_kernel.hpp"

#include "../common_kernel/helper.hpp"
#include "../common_kernel/interaction_constants.hpp"
#include "../common_kernel/struct_of_array_data.hpp"
#include "compute_kernel_templates.hpp"
#include "defs.hpp"
#include "interaction_types.hpp"
#include "options.hpp"

#include <array>
#include <functional>

extern options opts;

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        multipole_cpu_kernel::multipole_cpu_kernel(void)
          : theta_rec_squared(sqr(1.0 / opts.theta)) {
            for (size_t i = 0; i < m2m_int_vector::size(); i++) {
                offset_vector[i] = i;
            }
        }

        void multipole_cpu_kernel::apply_stencil(const struct_of_array_data<expansion, real, 20, ENTRIES,
                                           SOA_PADDING>& local_expansions_SoA,
            const struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                center_of_masses_SoA,
            struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>&
                potential_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>&
                angular_corrections_SoA,
            const std::vector<real>& mons, const two_phase_stencil& stencil, gsolve_type type) {
            for (size_t outer_stencil_index = 0;
                 outer_stencil_index < stencil.stencil_elements.size();
                 outer_stencil_index += STENCIL_BLOCKING) {
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
                                    angular_corrections_SoA, mons, cell_index, cell_flat_index,
                                    cell_index_coarse, cell_index_unpadded,
                                    cell_flat_index_unpadded, stencil, outer_stencil_index);
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
        void multipole_cpu_kernel::apply_stencil_non_blocked(const struct_of_array_data<expansion, real, 20, ENTRIES,
                                   SOA_PADDING>& local_expansions_SoA,
                const struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                    center_of_masses_SoA,
                struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>&
                    potential_expansions_SoA,
                struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>&
                    angular_corrections_SoA,
                const std::vector<real>& mons, const std::vector<bool> &stencil, const std::vector<bool>&
                inner_stencil, gsolve_type type) {
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
                                this->non_blocked_interaction_rho(local_expansions_SoA,
                                    center_of_masses_SoA, potential_expansions_SoA,
                                    angular_corrections_SoA, mons, cell_index, cell_flat_index,
                                    cell_index_coarse, cell_index_unpadded,
                                    cell_flat_index_unpadded, stencil, inner_stencil, 0);
                            } else {
                                this->non_blocked_interaction_non_rho(local_expansions_SoA,
                                    center_of_masses_SoA, potential_expansions_SoA,
                                    angular_corrections_SoA, mons, cell_index, cell_flat_index,
                                    cell_index_coarse, cell_index_unpadded,
                                    cell_flat_index_unpadded, stencil, inner_stencil, 0);
                            }
                        }
                    }
                }
        }


        void multipole_cpu_kernel::blocked_interaction_rho(
            const struct_of_array_data<expansion, real, 20, ENTRIES,
                SOA_PADDING>& __restrict__ local_expansions_SoA,
            const struct_of_array_data<space_vector, real, 3, ENTRIES,
                SOA_PADDING>& __restrict__ center_of_masses_SoA,
            struct_of_array_data<expansion, real, 20, INNER_CELLS,
                SOA_PADDING>& __restrict__ potential_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, INNER_CELLS,
                SOA_PADDING>& __restrict__ angular_corrections_SoA,
            const std::vector<real>& mons, const multiindex<>& __restrict__ cell_index,
            const size_t cell_flat_index,
            const multiindex<m2m_int_vector>& __restrict__ cell_index_coarse,
            const multiindex<>& __restrict__ cell_index_unpadded,
            const size_t cell_flat_index_unpadded, const two_phase_stencil& __restrict__ stencil,
            const size_t outer_stencil_index) {
            m2m_vector X[3];
            X[0] = center_of_masses_SoA.value<0>(cell_flat_index);
            X[1] = center_of_masses_SoA.value<1>(cell_flat_index);
            X[2] = center_of_masses_SoA.value<2>(cell_flat_index);
            m2m_vector tmpstore[20];
            m2m_vector tmp_corrections[3];

            m2m_vector m_cell[20];
            m_cell[0] = local_expansions_SoA.value<0>(cell_flat_index);
            m_cell[1] = local_expansions_SoA.value<1>(cell_flat_index);
            m_cell[2] = local_expansions_SoA.value<2>(cell_flat_index);
            m_cell[3] = local_expansions_SoA.value<3>(cell_flat_index);
            m_cell[4] = local_expansions_SoA.value<4>(cell_flat_index);
            m_cell[5] = local_expansions_SoA.value<5>(cell_flat_index);
            m_cell[6] = local_expansions_SoA.value<6>(cell_flat_index);
            m_cell[7] = local_expansions_SoA.value<7>(cell_flat_index);
            m_cell[8] = local_expansions_SoA.value<8>(cell_flat_index);
            m_cell[9] = local_expansions_SoA.value<9>(cell_flat_index);
            m_cell[10] = local_expansions_SoA.value<10>(cell_flat_index);
            m_cell[11] = local_expansions_SoA.value<11>(cell_flat_index);
            m_cell[12] = local_expansions_SoA.value<12>(cell_flat_index);
            m_cell[13] = local_expansions_SoA.value<13>(cell_flat_index);
            m_cell[14] = local_expansions_SoA.value<14>(cell_flat_index);
            m_cell[15] = local_expansions_SoA.value<15>(cell_flat_index);
            m_cell[16] = local_expansions_SoA.value<16>(cell_flat_index);
            m_cell[17] = local_expansions_SoA.value<17>(cell_flat_index);
            m_cell[18] = local_expansions_SoA.value<18>(cell_flat_index);
            m_cell[19] = local_expansions_SoA.value<19>(cell_flat_index);

            m2m_vector Y[3];

            bool changed_data = false;
            for (size_t inner_stencil_index = 0; inner_stencil_index < STENCIL_BLOCKING &&
                 outer_stencil_index + inner_stencil_index < stencil.stencil_elements.size();
                 inner_stencil_index += 1) {
                const bool phase_one =
                    stencil.stencil_phase_indicator[outer_stencil_index + inner_stencil_index];
                const multiindex<>& stencil_element =
                    stencil.stencil_elements[outer_stencil_index + inner_stencil_index];
                const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                    cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);

                const size_t interaction_partner_flat_index =
                    to_flat_index_padded(interaction_partner_index);    // iii1n

                // implicitly broadcasts to vector
                multiindex<m2m_int_vector> interaction_partner_index_coarse(
                    interaction_partner_index);
                interaction_partner_index_coarse.z += offset_vector;
                // note that this is the same for groups of 2x2x2 elements
                // -> maps to the same for some SIMD lanes
                interaction_partner_index_coarse.transform_coarse();

                m2m_int_vector theta_c_rec_squared_int = detail::distance_squared_reciprocal(
                    cell_index_coarse, interaction_partner_index_coarse);

                m2m_vector theta_c_rec_squared =
                    // Vc::static_datapar_cast<double>(theta_c_rec_squared_int);
                    Vc::simd_cast<m2m_vector>(theta_c_rec_squared_int);

                m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;

                if (Vc::none_of(mask)) {
                    continue;
                }
                changed_data = true;

                m2m_vector m_partner[20];
                Y[0] = center_of_masses_SoA.value<0>(interaction_partner_flat_index);
                Y[1] = center_of_masses_SoA.value<1>(interaction_partner_flat_index);
                Y[2] = center_of_masses_SoA.value<2>(interaction_partner_flat_index);

                m2m_vector::mask_type mask_phase_one(phase_one);

                Vc::where(mask, m_partner[0]) = m2m_vector(
                    mons.data() + interaction_partner_flat_index, Vc::Aligned);
                mask = mask & mask_phase_one;    // do not load multipoles outside the inner stencil
                Vc::where(mask, m_partner[0]) =
                    m_partner[0] + local_expansions_SoA.value<0>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[1]) =
                    local_expansions_SoA.value<1>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[2]) =
                    local_expansions_SoA.value<2>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[3]) =
                    local_expansions_SoA.value<3>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[4]) =
                    local_expansions_SoA.value<4>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[5]) =
                    local_expansions_SoA.value<5>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[6]) =
                    local_expansions_SoA.value<6>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[7]) =
                    local_expansions_SoA.value<7>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[8]) =
                    local_expansions_SoA.value<8>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[9]) =
                    local_expansions_SoA.value<9>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[10]) =
                    local_expansions_SoA.value<10>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[11]) =
                    local_expansions_SoA.value<11>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[12]) =
                    local_expansions_SoA.value<12>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[13]) =
                    local_expansions_SoA.value<13>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[14]) =
                    local_expansions_SoA.value<14>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[15]) =
                    local_expansions_SoA.value<15>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[16]) =
                    local_expansions_SoA.value<16>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[17]) =
                    local_expansions_SoA.value<17>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[18]) =
                    local_expansions_SoA.value<18>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[19]) =
                    local_expansions_SoA.value<19>(interaction_partner_flat_index);

                compute_kernel_rho(X, Y, m_partner, tmpstore, tmp_corrections, m_cell,
                    [](const m2m_vector& one, const m2m_vector& two) -> m2m_vector {
                        return Vc::max(one, two);
                    });
            }
            if (changed_data) {
                tmpstore[0] =
                    tmpstore[0] + potential_expansions_SoA.value<0>(cell_flat_index_unpadded);
                tmpstore[1] =
                    tmpstore[1] + potential_expansions_SoA.value<1>(cell_flat_index_unpadded);
                tmpstore[2] =
                    tmpstore[2] + potential_expansions_SoA.value<2>(cell_flat_index_unpadded);
                tmpstore[3] =
                    tmpstore[3] + potential_expansions_SoA.value<3>(cell_flat_index_unpadded);
                tmpstore[4] =
                    tmpstore[4] + potential_expansions_SoA.value<4>(cell_flat_index_unpadded);
                tmpstore[5] =
                    tmpstore[5] + potential_expansions_SoA.value<5>(cell_flat_index_unpadded);
                tmpstore[6] =
                    tmpstore[6] + potential_expansions_SoA.value<6>(cell_flat_index_unpadded);
                tmpstore[7] =
                    tmpstore[7] + potential_expansions_SoA.value<7>(cell_flat_index_unpadded);
                tmpstore[8] =
                    tmpstore[8] + potential_expansions_SoA.value<8>(cell_flat_index_unpadded);
                tmpstore[9] =
                    tmpstore[9] + potential_expansions_SoA.value<9>(cell_flat_index_unpadded);
                tmpstore[10] =
                    tmpstore[10] + potential_expansions_SoA.value<10>(cell_flat_index_unpadded);
                tmpstore[11] =
                    tmpstore[11] + potential_expansions_SoA.value<11>(cell_flat_index_unpadded);
                tmpstore[12] =
                    tmpstore[12] + potential_expansions_SoA.value<12>(cell_flat_index_unpadded);
                tmpstore[13] =
                    tmpstore[13] + potential_expansions_SoA.value<13>(cell_flat_index_unpadded);
                tmpstore[14] =
                    tmpstore[14] + potential_expansions_SoA.value<14>(cell_flat_index_unpadded);
                tmpstore[15] =
                    tmpstore[15] + potential_expansions_SoA.value<15>(cell_flat_index_unpadded);
                tmpstore[16] =
                    tmpstore[16] + potential_expansions_SoA.value<16>(cell_flat_index_unpadded);
                tmpstore[17] =
                    tmpstore[17] + potential_expansions_SoA.value<17>(cell_flat_index_unpadded);
                tmpstore[18] =
                    tmpstore[18] + potential_expansions_SoA.value<18>(cell_flat_index_unpadded);
                tmpstore[19] =
                    tmpstore[19] + potential_expansions_SoA.value<19>(cell_flat_index_unpadded);
                tmpstore[0].store(potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[1].store(potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[2].store(potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[3].store(potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[4].store(potential_expansions_SoA.pointer<4>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[5].store(potential_expansions_SoA.pointer<5>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[6].store(potential_expansions_SoA.pointer<6>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[7].store(potential_expansions_SoA.pointer<7>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[8].store(potential_expansions_SoA.pointer<8>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[9].store(potential_expansions_SoA.pointer<9>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[10].store(
                    potential_expansions_SoA.pointer<10>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[11].store(
                    potential_expansions_SoA.pointer<11>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[12].store(
                    potential_expansions_SoA.pointer<12>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[13].store(
                    potential_expansions_SoA.pointer<13>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[14].store(
                    potential_expansions_SoA.pointer<14>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[15].store(
                    potential_expansions_SoA.pointer<15>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[16].store(
                    potential_expansions_SoA.pointer<16>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[17].store(
                    potential_expansions_SoA.pointer<17>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[18].store(
                    potential_expansions_SoA.pointer<18>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[19].store(
                    potential_expansions_SoA.pointer<19>(cell_flat_index_unpadded),
                    Vc::Aligned);

                tmp_corrections[0] =
                    tmp_corrections[0] + angular_corrections_SoA.value<0>(cell_flat_index_unpadded);
                tmp_corrections[1] =
                    tmp_corrections[1] + angular_corrections_SoA.value<1>(cell_flat_index_unpadded);
                tmp_corrections[2] =
                    tmp_corrections[2] + angular_corrections_SoA.value<2>(cell_flat_index_unpadded);
                tmp_corrections[0].store(
                    angular_corrections_SoA.pointer<0>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmp_corrections[1].store(
                    angular_corrections_SoA.pointer<1>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmp_corrections[2].store(
                    angular_corrections_SoA.pointer<2>(cell_flat_index_unpadded),
                    Vc::Aligned);
            }
        }

        void multipole_cpu_kernel::blocked_interaction_non_rho(
            const struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>&
                local_expansions_SoA,
            const struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                center_of_masses_SoA,
            struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>&
                potential_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>&
                angular_corrections_SoA,
            const std::vector<real>& mons, const multiindex<>& cell_index,
            const size_t cell_flat_index, const multiindex<m2m_int_vector>& cell_index_coarse,
            const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
            const two_phase_stencil& stencil, const size_t outer_stencil_index) {
            m2m_vector X[3];
            X[0] = center_of_masses_SoA.value<0>(cell_flat_index);
            X[1] = center_of_masses_SoA.value<1>(cell_flat_index);
            X[2] = center_of_masses_SoA.value<2>(cell_flat_index);
            m2m_vector tmpstore[20];

            m2m_vector Y[3];

            bool changed_data = false;
            for (size_t inner_stencil_index = 0; inner_stencil_index < STENCIL_BLOCKING &&
                 outer_stencil_index + inner_stencil_index < stencil.stencil_elements.size();
                 inner_stencil_index += 1) {
                const bool phase_one =
                    stencil.stencil_phase_indicator[outer_stencil_index + inner_stencil_index];
                const multiindex<>& stencil_element =
                    stencil.stencil_elements[outer_stencil_index + inner_stencil_index];
                const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                    cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);

                const size_t interaction_partner_flat_index =
                    to_flat_index_padded(interaction_partner_index);    // iii1n

                // implicitly broadcasts to vector
                multiindex<m2m_int_vector> interaction_partner_index_coarse(
                    interaction_partner_index);
                interaction_partner_index_coarse.z += offset_vector;
                // note that this is the same for groups of 2x2x2 elements
                // -> maps to the same for some SIMD lanes
                interaction_partner_index_coarse.transform_coarse();

                m2m_int_vector theta_c_rec_squared_int = detail::distance_squared_reciprocal(
                    cell_index_coarse, interaction_partner_index_coarse);

                m2m_vector theta_c_rec_squared =
                    // Vc::static_datapar_cast<double>(theta_c_rec_squared_int);
                    Vc::simd_cast<m2m_vector>(theta_c_rec_squared_int);

                m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;

                if (Vc::none_of(mask)) {
                    continue;
                }
                changed_data = true;

                m2m_vector m_partner[20];
                Y[0] = center_of_masses_SoA.value<0>(interaction_partner_flat_index);
                Y[1] = center_of_masses_SoA.value<1>(interaction_partner_flat_index);
                Y[2] = center_of_masses_SoA.value<2>(interaction_partner_flat_index);

                m2m_vector::mask_type mask_phase_one(phase_one);

                Vc::where(mask, m_partner[0]) = m2m_vector(
                    mons.data() + interaction_partner_flat_index, Vc::Aligned);
                mask = mask & mask_phase_one;    // do not load multipoles outside the inner stencil
                Vc::where(mask, m_partner[0]) =
                    m_partner[0] + local_expansions_SoA.value<0>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[1]) =
                    local_expansions_SoA.value<1>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[2]) =
                    local_expansions_SoA.value<2>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[3]) =
                    local_expansions_SoA.value<3>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[4]) =
                    local_expansions_SoA.value<4>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[5]) =
                    local_expansions_SoA.value<5>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[6]) =
                    local_expansions_SoA.value<6>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[7]) =
                    local_expansions_SoA.value<7>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[8]) =
                    local_expansions_SoA.value<8>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[9]) =
                    local_expansions_SoA.value<9>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[10]) =
                    local_expansions_SoA.value<10>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[11]) =
                    local_expansions_SoA.value<11>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[12]) =
                    local_expansions_SoA.value<12>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[13]) =
                    local_expansions_SoA.value<13>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[14]) =
                    local_expansions_SoA.value<14>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[15]) =
                    local_expansions_SoA.value<15>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[16]) =
                    local_expansions_SoA.value<16>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[17]) =
                    local_expansions_SoA.value<17>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[18]) =
                    local_expansions_SoA.value<18>(interaction_partner_flat_index);
                Vc::where(mask, m_partner[19]) =
                    local_expansions_SoA.value<19>(interaction_partner_flat_index);

                compute_kernel_non_rho(X, Y, m_partner, tmpstore,
                    [](const m2m_vector& one, const m2m_vector& two) -> m2m_vector {
                        return Vc::max(one, two);
                    });
            }
            if (changed_data) {
                tmpstore[0] =
                    tmpstore[0] + potential_expansions_SoA.value<0>(cell_flat_index_unpadded);
                tmpstore[1] =
                    tmpstore[1] + potential_expansions_SoA.value<1>(cell_flat_index_unpadded);
                tmpstore[2] =
                    tmpstore[2] + potential_expansions_SoA.value<2>(cell_flat_index_unpadded);
                tmpstore[3] =
                    tmpstore[3] + potential_expansions_SoA.value<3>(cell_flat_index_unpadded);
                tmpstore[4] =
                    tmpstore[4] + potential_expansions_SoA.value<4>(cell_flat_index_unpadded);
                tmpstore[5] =
                    tmpstore[5] + potential_expansions_SoA.value<5>(cell_flat_index_unpadded);
                tmpstore[6] =
                    tmpstore[6] + potential_expansions_SoA.value<6>(cell_flat_index_unpadded);
                tmpstore[7] =
                    tmpstore[7] + potential_expansions_SoA.value<7>(cell_flat_index_unpadded);
                tmpstore[8] =
                    tmpstore[8] + potential_expansions_SoA.value<8>(cell_flat_index_unpadded);
                tmpstore[9] =
                    tmpstore[9] + potential_expansions_SoA.value<9>(cell_flat_index_unpadded);
                tmpstore[10] =
                    tmpstore[10] + potential_expansions_SoA.value<10>(cell_flat_index_unpadded);
                tmpstore[11] =
                    tmpstore[11] + potential_expansions_SoA.value<11>(cell_flat_index_unpadded);
                tmpstore[12] =
                    tmpstore[12] + potential_expansions_SoA.value<12>(cell_flat_index_unpadded);
                tmpstore[13] =
                    tmpstore[13] + potential_expansions_SoA.value<13>(cell_flat_index_unpadded);
                tmpstore[14] =
                    tmpstore[14] + potential_expansions_SoA.value<14>(cell_flat_index_unpadded);
                tmpstore[15] =
                    tmpstore[15] + potential_expansions_SoA.value<15>(cell_flat_index_unpadded);
                tmpstore[16] =
                    tmpstore[16] + potential_expansions_SoA.value<16>(cell_flat_index_unpadded);
                tmpstore[17] =
                    tmpstore[17] + potential_expansions_SoA.value<17>(cell_flat_index_unpadded);
                tmpstore[18] =
                    tmpstore[18] + potential_expansions_SoA.value<18>(cell_flat_index_unpadded);
                tmpstore[19] =
                    tmpstore[19] + potential_expansions_SoA.value<19>(cell_flat_index_unpadded);
                tmpstore[0].store(potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[1].store(potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[2].store(potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[3].store(potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[4].store(potential_expansions_SoA.pointer<4>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[5].store(potential_expansions_SoA.pointer<5>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[6].store(potential_expansions_SoA.pointer<6>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[7].store(potential_expansions_SoA.pointer<7>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[8].store(potential_expansions_SoA.pointer<8>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[9].store(potential_expansions_SoA.pointer<9>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[10].store(
                    potential_expansions_SoA.pointer<10>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[11].store(
                    potential_expansions_SoA.pointer<11>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[12].store(
                    potential_expansions_SoA.pointer<12>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[13].store(
                    potential_expansions_SoA.pointer<13>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[14].store(
                    potential_expansions_SoA.pointer<14>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[15].store(
                    potential_expansions_SoA.pointer<15>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[16].store(
                    potential_expansions_SoA.pointer<16>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[17].store(
                    potential_expansions_SoA.pointer<17>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[18].store(
                    potential_expansions_SoA.pointer<18>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[19].store(
                    potential_expansions_SoA.pointer<19>(cell_flat_index_unpadded),
                    Vc::Aligned);
            }
        }

        void multipole_cpu_kernel::non_blocked_interaction_rho(
            const struct_of_array_data<expansion, real, 20, ENTRIES,
            SOA_PADDING>& local_expansions_SoA,
            const struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
            center_of_masses_SoA,
            struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>&
            potential_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>&
            angular_corrections_SoA,
            const std::vector<real>& mons, const multiindex<>& cell_index,
            const size_t cell_flat_index, const multiindex<m2m_int_vector>& cell_index_coarse,
            const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
            const std::vector<bool>& stencil, const
            std::vector<bool>& inner_mask, const size_t outer_stencil_index) {
            m2m_vector X[3];
            X[0] = center_of_masses_SoA.value<0>(cell_flat_index);
            X[1] = center_of_masses_SoA.value<1>(cell_flat_index);
            X[2] = center_of_masses_SoA.value<2>(cell_flat_index);
            m2m_vector tmpstore[20];
            m2m_vector tmp_corrections[3];

            m2m_vector m_cell[20];
            m_cell[0] = local_expansions_SoA.value<0>(cell_flat_index);
            m_cell[1] = local_expansions_SoA.value<1>(cell_flat_index);
            m_cell[2] = local_expansions_SoA.value<2>(cell_flat_index);
            m_cell[3] = local_expansions_SoA.value<3>(cell_flat_index);
            m_cell[4] = local_expansions_SoA.value<4>(cell_flat_index);
            m_cell[5] = local_expansions_SoA.value<5>(cell_flat_index);
            m_cell[6] = local_expansions_SoA.value<6>(cell_flat_index);
            m_cell[7] = local_expansions_SoA.value<7>(cell_flat_index);
            m_cell[8] = local_expansions_SoA.value<8>(cell_flat_index);
            m_cell[9] = local_expansions_SoA.value<9>(cell_flat_index);
            m_cell[10] = local_expansions_SoA.value<10>(cell_flat_index);
            m_cell[11] = local_expansions_SoA.value<11>(cell_flat_index);
            m_cell[12] = local_expansions_SoA.value<12>(cell_flat_index);
            m_cell[13] = local_expansions_SoA.value<13>(cell_flat_index);
            m_cell[14] = local_expansions_SoA.value<14>(cell_flat_index);
            m_cell[15] = local_expansions_SoA.value<15>(cell_flat_index);
            m_cell[16] = local_expansions_SoA.value<16>(cell_flat_index);
            m_cell[17] = local_expansions_SoA.value<17>(cell_flat_index);
            m_cell[18] = local_expansions_SoA.value<18>(cell_flat_index);
            m_cell[19] = local_expansions_SoA.value<19>(cell_flat_index);

            m2m_vector Y[3];

            bool changed_data = false;
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
                        const bool phase_one =
                            inner_mask[index];
                        const multiindex<> stencil_element(stencil_x, stencil_y, stencil_z);
                        const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                                                                     cell_index.y + stencil_element.y,
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

                        m2m_int_vector theta_c_rec_squared_int = detail::distance_squared_reciprocal(
                            cell_index_coarse, interaction_partner_index_coarse);

                        m2m_vector theta_c_rec_squared =
                            // Vc::static_datapar_cast<double>(theta_c_rec_squared_int);
                            Vc::simd_cast<m2m_vector>(theta_c_rec_squared_int);

                        m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;

                        if (Vc::none_of(mask)) {
                            continue;
                        }
                        changed_data = true;

                        m2m_vector m_partner[20];
                        Y[0] = center_of_masses_SoA.value<0>(interaction_partner_flat_index);
                        Y[1] = center_of_masses_SoA.value<1>(interaction_partner_flat_index);
                        Y[2] = center_of_masses_SoA.value<2>(interaction_partner_flat_index);

                        m2m_vector::mask_type mask_phase_one(phase_one);

                        Vc::where(mask, m_partner[0]) = m2m_vector(
                            mons.data() + interaction_partner_flat_index, Vc::Aligned);
                        mask = mask & mask_phase_one;    // do not load multipoles outside the inner stencil
                        Vc::where(mask, m_partner[0]) =
                            m_partner[0] + local_expansions_SoA.value<0>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[1]) =
                            local_expansions_SoA.value<1>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[2]) =
                            local_expansions_SoA.value<2>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[3]) =
                            local_expansions_SoA.value<3>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[4]) =
                            local_expansions_SoA.value<4>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[5]) =
                            local_expansions_SoA.value<5>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[6]) =
                            local_expansions_SoA.value<6>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[7]) =
                            local_expansions_SoA.value<7>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[8]) =
                            local_expansions_SoA.value<8>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[9]) =
                            local_expansions_SoA.value<9>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[10]) =
                            local_expansions_SoA.value<10>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[11]) =
                            local_expansions_SoA.value<11>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[12]) =
                            local_expansions_SoA.value<12>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[13]) =
                            local_expansions_SoA.value<13>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[14]) =
                            local_expansions_SoA.value<14>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[15]) =
                            local_expansions_SoA.value<15>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[16]) =
                            local_expansions_SoA.value<16>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[17]) =
                            local_expansions_SoA.value<17>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[18]) =
                            local_expansions_SoA.value<18>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[19]) =
                            local_expansions_SoA.value<19>(interaction_partner_flat_index);

                        compute_kernel_rho(X, Y, m_partner, tmpstore, tmp_corrections, m_cell,
                                           [](const m2m_vector& one, const m2m_vector& two) -> m2m_vector {
                                               return Vc::max(one, two);
                                           });
                    }
                }
            }
            if (changed_data) {
                tmpstore[0] =
                    tmpstore[0] + potential_expansions_SoA.value<0>(cell_flat_index_unpadded);
                tmpstore[1] =
                    tmpstore[1] + potential_expansions_SoA.value<1>(cell_flat_index_unpadded);
                tmpstore[2] =
                    tmpstore[2] + potential_expansions_SoA.value<2>(cell_flat_index_unpadded);
                tmpstore[3] =
                    tmpstore[3] + potential_expansions_SoA.value<3>(cell_flat_index_unpadded);
                tmpstore[4] =
                    tmpstore[4] + potential_expansions_SoA.value<4>(cell_flat_index_unpadded);
                tmpstore[5] =
                    tmpstore[5] + potential_expansions_SoA.value<5>(cell_flat_index_unpadded);
                tmpstore[6] =
                    tmpstore[6] + potential_expansions_SoA.value<6>(cell_flat_index_unpadded);
                tmpstore[7] =
                    tmpstore[7] + potential_expansions_SoA.value<7>(cell_flat_index_unpadded);
                tmpstore[8] =
                    tmpstore[8] + potential_expansions_SoA.value<8>(cell_flat_index_unpadded);
                tmpstore[9] =
                    tmpstore[9] + potential_expansions_SoA.value<9>(cell_flat_index_unpadded);
                tmpstore[10] =
                    tmpstore[10] + potential_expansions_SoA.value<10>(cell_flat_index_unpadded);
                tmpstore[11] =
                    tmpstore[11] + potential_expansions_SoA.value<11>(cell_flat_index_unpadded);
                tmpstore[12] =
                    tmpstore[12] + potential_expansions_SoA.value<12>(cell_flat_index_unpadded);
                tmpstore[13] =
                    tmpstore[13] + potential_expansions_SoA.value<13>(cell_flat_index_unpadded);
                tmpstore[14] =
                    tmpstore[14] + potential_expansions_SoA.value<14>(cell_flat_index_unpadded);
                tmpstore[15] =
                    tmpstore[15] + potential_expansions_SoA.value<15>(cell_flat_index_unpadded);
                tmpstore[16] =
                    tmpstore[16] + potential_expansions_SoA.value<16>(cell_flat_index_unpadded);
                tmpstore[17] =
                    tmpstore[17] + potential_expansions_SoA.value<17>(cell_flat_index_unpadded);
                tmpstore[18] =
                    tmpstore[18] + potential_expansions_SoA.value<18>(cell_flat_index_unpadded);
                tmpstore[19] =
                    tmpstore[19] + potential_expansions_SoA.value<19>(cell_flat_index_unpadded);
                tmpstore[0].store(potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[1].store(potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[2].store(potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[3].store(potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[4].store(potential_expansions_SoA.pointer<4>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[5].store(potential_expansions_SoA.pointer<5>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[6].store(potential_expansions_SoA.pointer<6>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[7].store(potential_expansions_SoA.pointer<7>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[8].store(potential_expansions_SoA.pointer<8>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[9].store(potential_expansions_SoA.pointer<9>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[10].store(
                    potential_expansions_SoA.pointer<10>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[11].store(
                    potential_expansions_SoA.pointer<11>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[12].store(
                    potential_expansions_SoA.pointer<12>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[13].store(
                    potential_expansions_SoA.pointer<13>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[14].store(
                    potential_expansions_SoA.pointer<14>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[15].store(
                    potential_expansions_SoA.pointer<15>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[16].store(
                    potential_expansions_SoA.pointer<16>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[17].store(
                    potential_expansions_SoA.pointer<17>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[18].store(
                    potential_expansions_SoA.pointer<18>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[19].store(
                    potential_expansions_SoA.pointer<19>(cell_flat_index_unpadded),
                    Vc::Aligned);

                tmp_corrections[0] =
                    tmp_corrections[0] + angular_corrections_SoA.value<0>(cell_flat_index_unpadded);
                tmp_corrections[1] =
                    tmp_corrections[1] + angular_corrections_SoA.value<1>(cell_flat_index_unpadded);
                tmp_corrections[2] =
                    tmp_corrections[2] + angular_corrections_SoA.value<2>(cell_flat_index_unpadded);
                tmp_corrections[0].store(
                    angular_corrections_SoA.pointer<0>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmp_corrections[1].store(
                    angular_corrections_SoA.pointer<1>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmp_corrections[2].store(
                    angular_corrections_SoA.pointer<2>(cell_flat_index_unpadded),
                    Vc::Aligned);
            }
        }

        void multipole_cpu_kernel::non_blocked_interaction_non_rho(
            const struct_of_array_data<expansion, real, 20,
            ENTRIES, SOA_PADDING>& local_expansions_SoA,
            const struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
            center_of_masses_SoA,
            struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>&
            potential_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>&
            angular_corrections_SoA,
            const std::vector<real>& mons, const multiindex<>& cell_index,
            const size_t cell_flat_index, const multiindex<m2m_int_vector>& cell_index_coarse,
            const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
            const std::vector<bool>& stencil, const
            std::vector<bool>& inner_mask, const size_t outer_stencil_index) {
            m2m_vector X[3];
            X[0] = center_of_masses_SoA.value<0>(cell_flat_index);
            X[1] = center_of_masses_SoA.value<1>(cell_flat_index);
            X[2] = center_of_masses_SoA.value<2>(cell_flat_index);
            m2m_vector tmpstore[20];

            m2m_vector Y[3];

            bool changed_data = false;
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

                        const bool phase_one = inner_mask[index];

                        const size_t interaction_partner_flat_index =
                            to_flat_index_padded(interaction_partner_index);    // iii1n

                        // implicitly broadcasts to vector
                        multiindex<m2m_int_vector> interaction_partner_index_coarse(
                            interaction_partner_index);
                        interaction_partner_index_coarse.z += offset_vector;
                        // note that this is the same for groups of 2x2x2 elements
                        // -> maps to the same for some SIMD lanes
                        interaction_partner_index_coarse.transform_coarse();

                        m2m_int_vector theta_c_rec_squared_int = detail::distance_squared_reciprocal(
                            cell_index_coarse, interaction_partner_index_coarse);

                        m2m_vector theta_c_rec_squared =
                            // Vc::static_datapar_cast<double>(theta_c_rec_squared_int);
                            Vc::simd_cast<m2m_vector>(theta_c_rec_squared_int);

                        m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;

                        if (Vc::none_of(mask)) {
                            continue;
                        }
                        changed_data = true;

                        m2m_vector m_partner[20];
                        Y[0] = center_of_masses_SoA.value<0>(interaction_partner_flat_index);
                        Y[1] = center_of_masses_SoA.value<1>(interaction_partner_flat_index);
                        Y[2] = center_of_masses_SoA.value<2>(interaction_partner_flat_index);

                        m2m_vector::mask_type mask_phase_one(phase_one);

                        Vc::where(mask, m_partner[0]) = m2m_vector(
                            mons.data() + interaction_partner_flat_index, Vc::Aligned);
                        mask = mask & mask_phase_one;    // do not load multipoles outside the inner stencil
                        Vc::where(mask, m_partner[0]) =
                            m_partner[0] + local_expansions_SoA.value<0>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[1]) =
                            local_expansions_SoA.value<1>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[2]) =
                            local_expansions_SoA.value<2>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[3]) =
                            local_expansions_SoA.value<3>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[4]) =
                            local_expansions_SoA.value<4>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[5]) =
                            local_expansions_SoA.value<5>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[6]) =
                            local_expansions_SoA.value<6>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[7]) =
                            local_expansions_SoA.value<7>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[8]) =
                            local_expansions_SoA.value<8>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[9]) =
                            local_expansions_SoA.value<9>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[10]) =
                            local_expansions_SoA.value<10>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[11]) =
                            local_expansions_SoA.value<11>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[12]) =
                            local_expansions_SoA.value<12>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[13]) =
                            local_expansions_SoA.value<13>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[14]) =
                            local_expansions_SoA.value<14>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[15]) =
                            local_expansions_SoA.value<15>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[16]) =
                            local_expansions_SoA.value<16>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[17]) =
                            local_expansions_SoA.value<17>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[18]) =
                            local_expansions_SoA.value<18>(interaction_partner_flat_index);
                        Vc::where(mask, m_partner[19]) =
                            local_expansions_SoA.value<19>(interaction_partner_flat_index);

                        compute_kernel_non_rho(X, Y, m_partner, tmpstore,
                                               [](const m2m_vector& one, const m2m_vector& two) -> m2m_vector {
                                                   return Vc::max(one, two);
                                               });
                    }
                }
            }
            if (changed_data) {
                tmpstore[0] =
                    tmpstore[0] + potential_expansions_SoA.value<0>(cell_flat_index_unpadded);
                tmpstore[1] =
                    tmpstore[1] + potential_expansions_SoA.value<1>(cell_flat_index_unpadded);
                tmpstore[2] =
                    tmpstore[2] + potential_expansions_SoA.value<2>(cell_flat_index_unpadded);
                tmpstore[3] =
                    tmpstore[3] + potential_expansions_SoA.value<3>(cell_flat_index_unpadded);
                tmpstore[4] =
                    tmpstore[4] + potential_expansions_SoA.value<4>(cell_flat_index_unpadded);
                tmpstore[5] =
                    tmpstore[5] + potential_expansions_SoA.value<5>(cell_flat_index_unpadded);
                tmpstore[6] =
                    tmpstore[6] + potential_expansions_SoA.value<6>(cell_flat_index_unpadded);
                tmpstore[7] =
                    tmpstore[7] + potential_expansions_SoA.value<7>(cell_flat_index_unpadded);
                tmpstore[8] =
                    tmpstore[8] + potential_expansions_SoA.value<8>(cell_flat_index_unpadded);
                tmpstore[9] =
                    tmpstore[9] + potential_expansions_SoA.value<9>(cell_flat_index_unpadded);
                tmpstore[10] =
                    tmpstore[10] + potential_expansions_SoA.value<10>(cell_flat_index_unpadded);
                tmpstore[11] =
                    tmpstore[11] + potential_expansions_SoA.value<11>(cell_flat_index_unpadded);
                tmpstore[12] =
                    tmpstore[12] + potential_expansions_SoA.value<12>(cell_flat_index_unpadded);
                tmpstore[13] =
                    tmpstore[13] + potential_expansions_SoA.value<13>(cell_flat_index_unpadded);
                tmpstore[14] =
                    tmpstore[14] + potential_expansions_SoA.value<14>(cell_flat_index_unpadded);
                tmpstore[15] =
                    tmpstore[15] + potential_expansions_SoA.value<15>(cell_flat_index_unpadded);
                tmpstore[16] =
                    tmpstore[16] + potential_expansions_SoA.value<16>(cell_flat_index_unpadded);
                tmpstore[17] =
                    tmpstore[17] + potential_expansions_SoA.value<17>(cell_flat_index_unpadded);
                tmpstore[18] =
                    tmpstore[18] + potential_expansions_SoA.value<18>(cell_flat_index_unpadded);
                tmpstore[19] =
                    tmpstore[19] + potential_expansions_SoA.value<19>(cell_flat_index_unpadded);
                tmpstore[0].store(potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[1].store(potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[2].store(potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[3].store(potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[4].store(potential_expansions_SoA.pointer<4>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[5].store(potential_expansions_SoA.pointer<5>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[6].store(potential_expansions_SoA.pointer<6>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[7].store(potential_expansions_SoA.pointer<7>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[8].store(potential_expansions_SoA.pointer<8>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[9].store(potential_expansions_SoA.pointer<9>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[10].store(
                    potential_expansions_SoA.pointer<10>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[11].store(
                    potential_expansions_SoA.pointer<11>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[12].store(
                    potential_expansions_SoA.pointer<12>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[13].store(
                    potential_expansions_SoA.pointer<13>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[14].store(
                    potential_expansions_SoA.pointer<14>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[15].store(
                    potential_expansions_SoA.pointer<15>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[16].store(
                    potential_expansions_SoA.pointer<16>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[17].store(
                    potential_expansions_SoA.pointer<17>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[18].store(
                    potential_expansions_SoA.pointer<18>(cell_flat_index_unpadded),
                    Vc::Aligned);
                tmpstore[19].store(
                    potential_expansions_SoA.pointer<19>(cell_flat_index_unpadded),
                    Vc::Aligned);
            }
        }
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
