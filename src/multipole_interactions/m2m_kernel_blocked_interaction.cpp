#include "m2m_kernel.hpp"

#include "../common_kernel/helper.hpp"
#include "../common_kernel/kernel_taylor_set_basis.hpp"
#include "../common_kernel/struct_of_array_data.hpp"
#include "grid_flattened_indices.hpp"
#include "compute_kernel_templates.hpp"

extern taylor<4, real> factor;
extern taylor<4, m2m_vector> factor_half_v;
extern taylor<4, m2m_vector> factor_sixth_v;

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {
        // TODO:
        // - check codegen and fix in Vc
        // - check for amount of temporaries
        // - try to replace expensive operations like sqrt
        // - remove all sqr()
        // - increase INX

        void m2m_kernel::blocked_interaction_rho(
            struct_of_array_data<expansion, real, 20, ENTRIES,
                SOA_PADDING>& __restrict__ local_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, ENTRIES,
                SOA_PADDING>& __restrict__ center_of_masses_SoA,
            struct_of_array_data<expansion, real, 20, INNER_CELLS,
                SOA_PADDING>& __restrict__ potential_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, INNER_CELLS,
                SOA_PADDING>& __restrict__ angular_corrections_SoA,
            std::vector<real>& mons, const multiindex<>& __restrict__ cell_index,
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
            std::array<m2m_vector, 15> D_upper;
            std::array<m2m_vector, 20> D_lower;
            std::array<m2m_vector, 10> cur_pot;

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

            m2m_vector factor_half[20];
            m2m_vector factor_sixth[20];
            for (auto i = 0; i < 20; ++i) {
                factor_half[i] = factor_half_v[i];
                factor_sixth[i] = factor_sixth_v[i];
            }

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
                    Vc::static_datapar_cast_double_to_int(theta_c_rec_squared_int);

                m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;

                if (Vc::none_of(mask)) {
                    continue;
                }
                changed_data = true;

                m2m_vector Y[3];
                Y[0] = center_of_masses_SoA.value<0>(interaction_partner_flat_index);
                Y[1] = center_of_masses_SoA.value<1>(interaction_partner_flat_index);
                Y[2] = center_of_masses_SoA.value<2>(interaction_partner_flat_index);

                m2m_vector m_partner[20];
                m2m_vector::mask_type mask_phase_one(phase_one);

                Vc::where(mask, m_partner[0]) = m2m_vector(
                    mons.data() + interaction_partner_flat_index, Vc::flags::element_aligned);
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

                compute_kernel_rho(X, Y, m_partner, tmpstore, tmp_corrections, m_cell, factor_half,
                    factor_sixth, [](const m2m_vector& one, const m2m_vector& two) -> m2m_vector {
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
                tmpstore[0].memstore(potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[1].memstore(potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[2].memstore(potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[3].memstore(potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[4].memstore(potential_expansions_SoA.pointer<4>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[5].memstore(potential_expansions_SoA.pointer<5>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[6].memstore(potential_expansions_SoA.pointer<6>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[7].memstore(potential_expansions_SoA.pointer<7>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[8].memstore(potential_expansions_SoA.pointer<8>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[9].memstore(potential_expansions_SoA.pointer<9>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[10].memstore(
                    potential_expansions_SoA.pointer<10>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[11].memstore(
                    potential_expansions_SoA.pointer<11>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[12].memstore(
                    potential_expansions_SoA.pointer<12>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[13].memstore(
                    potential_expansions_SoA.pointer<13>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[14].memstore(
                    potential_expansions_SoA.pointer<14>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[15].memstore(
                    potential_expansions_SoA.pointer<15>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[16].memstore(
                    potential_expansions_SoA.pointer<16>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[17].memstore(
                    potential_expansions_SoA.pointer<17>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[18].memstore(
                    potential_expansions_SoA.pointer<18>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[19].memstore(
                    potential_expansions_SoA.pointer<19>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp_corrections[0] =
                    tmp_corrections[0] + angular_corrections_SoA.value<0>(cell_flat_index_unpadded);
                tmp_corrections[1] =
                    tmp_corrections[1] + angular_corrections_SoA.value<1>(cell_flat_index_unpadded);
                tmp_corrections[2] =
                    tmp_corrections[2] + angular_corrections_SoA.value<2>(cell_flat_index_unpadded);
                tmp_corrections[0].memstore(
                    angular_corrections_SoA.pointer<0>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmp_corrections[1].memstore(
                    angular_corrections_SoA.pointer<1>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmp_corrections[2].memstore(
                    angular_corrections_SoA.pointer<2>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
            }
        }

        void m2m_kernel::blocked_interaction_non_rho(
            struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>& local_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>& center_of_masses_SoA,
            struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>&
                potential_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>&
                angular_corrections_SoA,
            std::vector<real>& mons, const multiindex<>& cell_index, const size_t cell_flat_index,
            const multiindex<m2m_int_vector>& cell_index_coarse,
            const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
            const two_phase_stencil& stencil, const size_t outer_stencil_index) {
            // TODO: should change name to something better (not taylor, but space_vector)
            // struct_of_array_taylor<space_vector, real, 3> X =
            //     center_of_masses_SoA.get_view(cell_flat_index);

            std::array<m2m_vector, 20> tmpstore;
            // tmpstore[0] = potential_expansions_SoA.value<0>(cell_flat_index_unpadded);
            // tmpstore[1] = potential_expansions_SoA.value<1>(cell_flat_index_unpadded);
            // tmpstore[2] = potential_expansions_SoA.value<2>(cell_flat_index_unpadded);
            // tmpstore[3] = potential_expansions_SoA.value<3>(cell_flat_index_unpadded);
            // tmpstore[4] = potential_expansions_SoA.value<4>(cell_flat_index_unpadded);
            // tmpstore[5] = potential_expansions_SoA.value<5>(cell_flat_index_unpadded);
            // tmpstore[6] = potential_expansions_SoA.value<6>(cell_flat_index_unpadded);
            // tmpstore[7] = potential_expansions_SoA.value<7>(cell_flat_index_unpadded);
            // tmpstore[8] = potential_expansions_SoA.value<8>(cell_flat_index_unpadded);
            // tmpstore[9] = potential_expansions_SoA.value<9>(cell_flat_index_unpadded);
            // tmpstore[10] = potential_expansions_SoA.value<10>(cell_flat_index_unpadded);
            // tmpstore[11] = potential_expansions_SoA.value<11>(cell_flat_index_unpadded);
            // tmpstore[12] = potential_expansions_SoA.value<12>(cell_flat_index_unpadded);
            // tmpstore[13] = potential_expansions_SoA.value<13>(cell_flat_index_unpadded);
            // tmpstore[14] = potential_expansions_SoA.value<14>(cell_flat_index_unpadded);
            // tmpstore[15] = potential_expansions_SoA.value<15>(cell_flat_index_unpadded);
            // tmpstore[16] = potential_expansions_SoA.value<16>(cell_flat_index_unpadded);
            // tmpstore[17] = potential_expansions_SoA.value<17>(cell_flat_index_unpadded);
            // tmpstore[18] = potential_expansions_SoA.value<18>(cell_flat_index_unpadded);
            // tmpstore[19] = potential_expansions_SoA.value<19>(cell_flat_index_unpadded);
            // struct_of_array_iterator<space_vector, real, 3> X(center_of_masses_SoA,
            // cell_flat_index);
            std::array<m2m_vector, 20> D_lower;
            std::array<m2m_vector, 10>
                cur_pot;    // current potential = cur_pot = A in the old style
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

                const int64_t interaction_partner_flat_index =
                    to_flat_index_padded(interaction_partner_index);

                // check whether all vector elements are in empty border
                // if (vector_is_empty[interaction_partner_flat_index]) {
                //     continue;
                // }

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
                    Vc::static_datapar_cast_double_to_int(theta_c_rec_squared_int);

                m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;

                if (Vc::none_of(mask)) {
                    continue;
                }
                changed_data = true;

                // struct_of_array_iterator<space_vector, real, 3> Y(
                //     center_of_masses_SoA, interaction_partner_flat_index);

                // distance between cells in all dimensions
                // TODO: replace by m2m_vector for vectorization or get rid of temporary
                std::array<m2m_vector, NDIM> dX;
                dX[0] = center_of_masses_SoA.value<0>(cell_flat_index) -
                    center_of_masses_SoA.value<0>(interaction_partner_flat_index);
                dX[1] = center_of_masses_SoA.value<1>(cell_flat_index) -
                    center_of_masses_SoA.value<1>(interaction_partner_flat_index);
                dX[2] = center_of_masses_SoA.value<2>(cell_flat_index) -
                    center_of_masses_SoA.value<2>(interaction_partner_flat_index);

                // expansion_v m_partner;
                std::array<m2m_vector, 20> m_partner;
                m2m_vector::mask_type mask_phase_one(phase_one);

                Vc::where(mask, m_partner[0]) = m2m_vector(
                    mons.data() + interaction_partner_flat_index, Vc::flags::element_aligned);
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
                // R_i in paper is the dX in the code
                // D is taylor expansion value for a given X expansion of the gravitational
                // potential
                // (multipole expansion)

                // calculates all D-values, calculate all coefficients of 1/r (not the
                // potential),
                // formula (6)-(9) and (19)
                D_split D_calculator(dX);
                D_calculator.calculate_D_lower(D_lower);

                // 10-19 are not cached!

                // the following loops calculate formula (10), potential from B->A
                cur_pot[0] = m_partner[0] * D_lower[0];
                cur_pot[1] = m_partner[0] * D_lower[1];
                cur_pot[2] = m_partner[0] * D_lower[2];
                cur_pot[3] = m_partner[0] * D_lower[3];

                cur_pot[0] -= m_partner[1] * D_lower[1];
                cur_pot[1] -= m_partner[1] * D_lower[4];
                cur_pot[1] -= m_partner[1] * D_lower[5];
                cur_pot[1] -= m_partner[1] * D_lower[6];

                cur_pot[0] -= m_partner[2] * D_lower[2];
                cur_pot[2] -= m_partner[2] * D_lower[5];
                cur_pot[2] -= m_partner[2] * D_lower[7];
                cur_pot[2] -= m_partner[2] * D_lower[8];

                cur_pot[0] -= m_partner[3] * D_lower[3];
                cur_pot[3] -= m_partner[3] * D_lower[6];
                cur_pot[3] -= m_partner[3] * D_lower[8];
                cur_pot[3] -= m_partner[3] * D_lower[9];

                cur_pot[0] += m_partner[4] * (D_lower[4] * factor_half_v[4]);
                cur_pot[1] += m_partner[4] * (D_lower[10] * factor_half_v[4]);
                cur_pot[2] += m_partner[4] * (D_lower[11] * factor_half_v[4]);
                cur_pot[3] += m_partner[4] * (D_lower[12] * factor_half_v[4]);

                cur_pot[0] += m_partner[5] * (D_lower[5] * factor_half_v[5]);
                cur_pot[1] += m_partner[5] * (D_lower[11] * factor_half_v[5]);
                cur_pot[2] += m_partner[5] * (D_lower[13] * factor_half_v[5]);
                cur_pot[3] += m_partner[5] * (D_lower[14] * factor_half_v[5]);

                cur_pot[0] += m_partner[6] * (D_lower[6] * factor_half_v[6]);
                cur_pot[1] += m_partner[6] * (D_lower[12] * factor_half_v[6]);
                cur_pot[2] += m_partner[6] * (D_lower[14] * factor_half_v[6]);
                cur_pot[3] += m_partner[6] * (D_lower[15] * factor_half_v[6]);

                cur_pot[0] += m_partner[7] * (D_lower[7] * factor_half_v[7]);
                cur_pot[1] += m_partner[7] * (D_lower[13] * factor_half_v[7]);
                cur_pot[2] += m_partner[7] * (D_lower[16] * factor_half_v[7]);
                cur_pot[3] += m_partner[7] * (D_lower[17] * factor_half_v[7]);

                cur_pot[0] += m_partner[8] * (D_lower[8] * factor_half_v[8]);
                cur_pot[1] += m_partner[8] * (D_lower[14] * factor_half_v[8]);
                cur_pot[2] += m_partner[8] * (D_lower[17] * factor_half_v[8]);
                cur_pot[3] += m_partner[8] * (D_lower[18] * factor_half_v[8]);

                cur_pot[0] += m_partner[9] * (D_lower[9] * factor_half_v[9]);
                cur_pot[1] += m_partner[9] * (D_lower[15] * factor_half_v[9]);
                cur_pot[2] += m_partner[9] * (D_lower[18] * factor_half_v[9]);
                cur_pot[3] += m_partner[9] * (D_lower[19] * factor_half_v[9]);

                cur_pot[0] -= m_partner[10] * (D_lower[10] * factor_sixth_v[10]);
                cur_pot[0] -= m_partner[11] * (D_lower[11] * factor_sixth_v[11]);
                cur_pot[0] -= m_partner[12] * (D_lower[12] * factor_sixth_v[12]);
                cur_pot[0] -= m_partner[13] * (D_lower[13] * factor_sixth_v[13]);
                cur_pot[0] -= m_partner[14] * (D_lower[14] * factor_sixth_v[14]);
                cur_pot[0] -= m_partner[15] * (D_lower[15] * factor_sixth_v[15]);
                cur_pot[0] -= m_partner[16] * (D_lower[16] * factor_sixth_v[16]);
                cur_pot[0] -= m_partner[17] * (D_lower[17] * factor_sixth_v[17]);
                cur_pot[0] -= m_partner[18] * (D_lower[18] * factor_sixth_v[18]);
                cur_pot[0] -= m_partner[19] * (D_lower[19] * factor_sixth_v[19]);

                cur_pot[4] = m_partner[0] * D_lower[4];
                cur_pot[5] = m_partner[0] * D_lower[5];
                cur_pot[6] = m_partner[0] * D_lower[6];
                cur_pot[7] = m_partner[0] * D_lower[7];
                cur_pot[8] = m_partner[0] * D_lower[8];
                cur_pot[9] = m_partner[0] * D_lower[9];

                cur_pot[4] -= m_partner[1] * D_lower[10];
                cur_pot[5] -= m_partner[1] * D_lower[11];
                cur_pot[6] -= m_partner[1] * D_lower[12];
                cur_pot[7] -= m_partner[1] * D_lower[13];
                cur_pot[8] -= m_partner[1] * D_lower[14];
                cur_pot[9] -= m_partner[1] * D_lower[15];

                cur_pot[4] -= m_partner[2] * D_lower[11];
                cur_pot[5] -= m_partner[2] * D_lower[13];
                cur_pot[6] -= m_partner[2] * D_lower[14];
                cur_pot[7] -= m_partner[2] * D_lower[16];
                cur_pot[8] -= m_partner[2] * D_lower[17];
                cur_pot[9] -= m_partner[2] * D_lower[18];

                cur_pot[4] -= m_partner[3] * D_lower[12];
                cur_pot[5] -= m_partner[3] * D_lower[14];
                cur_pot[6] -= m_partner[3] * D_lower[15];
                cur_pot[7] -= m_partner[3] * D_lower[17];
                cur_pot[8] -= m_partner[3] * D_lower[18];
                cur_pot[9] -= m_partner[3] * D_lower[19];

                tmpstore[0] = tmpstore[0] + cur_pot[0];
                tmpstore[1] = tmpstore[1] + cur_pot[1];
                tmpstore[2] = tmpstore[2] + cur_pot[2];
                tmpstore[3] = tmpstore[3] + cur_pot[3];
                tmpstore[4] = tmpstore[4] + cur_pot[4];
                tmpstore[5] = tmpstore[5] + cur_pot[5];
                tmpstore[6] = tmpstore[6] + cur_pot[6];
                tmpstore[7] = tmpstore[7] + cur_pot[7];
                tmpstore[8] = tmpstore[8] + cur_pot[8];
                tmpstore[9] = tmpstore[9] + cur_pot[9];

                /* Maps to
                for (integer i = taylor_sizes[2]; i < taylor_sizes[3]; ++i) {
                    A0[i] = m0[0] * D[i];
                }*/
                tmpstore[10] = tmpstore[10] + m_partner[0] * D_lower[10];
                tmpstore[11] = tmpstore[11] + m_partner[0] * D_lower[11];
                tmpstore[12] = tmpstore[12] + m_partner[0] * D_lower[12];
                tmpstore[13] = tmpstore[13] + m_partner[0] * D_lower[13];
                tmpstore[14] = tmpstore[14] + m_partner[0] * D_lower[14];
                tmpstore[15] = tmpstore[15] + m_partner[0] * D_lower[15];
                tmpstore[16] = tmpstore[16] + m_partner[0] * D_lower[16];
                tmpstore[17] = tmpstore[17] + m_partner[0] * D_lower[17];
                tmpstore[18] = tmpstore[18] + m_partner[0] * D_lower[18];
                tmpstore[19] = tmpstore[19] + m_partner[0] * D_lower[19];
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
                tmpstore[0].memstore(potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[1].memstore(potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[2].memstore(potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[3].memstore(potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[4].memstore(potential_expansions_SoA.pointer<4>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[5].memstore(potential_expansions_SoA.pointer<5>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[6].memstore(potential_expansions_SoA.pointer<6>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[7].memstore(potential_expansions_SoA.pointer<7>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[8].memstore(potential_expansions_SoA.pointer<8>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[9].memstore(potential_expansions_SoA.pointer<9>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[10].memstore(
                    potential_expansions_SoA.pointer<10>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[11].memstore(
                    potential_expansions_SoA.pointer<11>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[12].memstore(
                    potential_expansions_SoA.pointer<12>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[13].memstore(
                    potential_expansions_SoA.pointer<13>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[14].memstore(
                    potential_expansions_SoA.pointer<14>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[15].memstore(
                    potential_expansions_SoA.pointer<15>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[16].memstore(
                    potential_expansions_SoA.pointer<16>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[17].memstore(
                    potential_expansions_SoA.pointer<17>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[18].memstore(
                    potential_expansions_SoA.pointer<18>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmpstore[19].memstore(
                    potential_expansions_SoA.pointer<19>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
            }
        }
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
