#include "p2m_kernel.hpp"

#include "../common_kernel/helper.hpp"
#include "../common_kernel/kernel_taylor_set_basis.hpp"
#include "../common_kernel/struct_of_array_data.hpp"
#include "grid_flattened_indices.hpp"

extern taylor<4, real> factor;
extern taylor<4, m2m_vector> factor_half_v;
extern taylor<4, m2m_vector> factor_sixth_v;

namespace octotiger {
namespace fmm {
    namespace p2m_kernel {

        // TODO:
        // - check codegen and fix in Vc
        // - check for amount of temporaries
        // - try to replace expensive operations like sqrt
        // - remove all sqr()
        // - increase INX

        void p2m_kernel::blocked_interaction_rho(
            struct_of_array_data<expansion, real, 20, ENTRIES,
                SOA_PADDING>& __restrict__ local_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, ENTRIES,
                SOA_PADDING>& __restrict__ center_of_masses_SoA,
            struct_of_array_data<expansion, real, 20, ENTRIES,
                SOA_PADDING>& __restrict__ potential_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, ENTRIES,
                SOA_PADDING>& __restrict__ angular_corrections_SoA,
            const multiindex<>& __restrict__ cell_index, const size_t cell_flat_index,
            const multiindex<m2m_int_vector>& __restrict__ cell_index_coarse,
            const multiindex<>& __restrict__ cell_index_unpadded,
            const size_t cell_flat_index_unpadded,
            const std::vector<multiindex<>>& __restrict__ stencil, const size_t outer_stencil_index,
            std::vector<bool>& interact) {
            std::array<m2m_vector, NDIM> X;
            X[0] = center_of_masses_SoA.value<0>(cell_flat_index);
            X[1] = center_of_masses_SoA.value<1>(cell_flat_index);
            X[2] = center_of_masses_SoA.value<2>(cell_flat_index);
            std::array<m2m_vector, 4> tmpstore;
            tmpstore[0] = potential_expansions_SoA.value<0>(cell_flat_index_unpadded);
            tmpstore[1] = potential_expansions_SoA.value<1>(cell_flat_index_unpadded);
            tmpstore[2] = potential_expansions_SoA.value<2>(cell_flat_index_unpadded);
            tmpstore[3] = potential_expansions_SoA.value<3>(cell_flat_index_unpadded);
            std::array<m2m_vector, 4> tmp_corrections;
            tmp_corrections[0] = angular_corrections_SoA.value<0>(cell_flat_index_unpadded);
            tmp_corrections[1] = angular_corrections_SoA.value<1>(cell_flat_index_unpadded);
            tmp_corrections[2] = angular_corrections_SoA.value<2>(cell_flat_index_unpadded);
            tmp_corrections[3] = angular_corrections_SoA.value<3>(cell_flat_index_unpadded);
            for (size_t inner_stencil_index = 0; inner_stencil_index < P2M_STENCIL_BLOCKING &&
                 outer_stencil_index + inner_stencil_index < stencil.size();
                 inner_stencil_index += 1) {
                const multiindex<>& stencil_element =
                    stencil[outer_stencil_index + inner_stencil_index];
                const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                    cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);

                const size_t interaction_partner_flat_index =
                    to_flat_index_padded(interaction_partner_index);    // iii1n
                if (!interact[interaction_partner_flat_index] &&
                    !interact[interaction_partner_flat_index + 1] &&
                    !interact[interaction_partner_flat_index + 2] &&
                    !interact[interaction_partner_flat_index + 3])
                    continue;

                // check whether all vector elements are in empty border
                if (vector_is_empty[interaction_partner_flat_index]) {
                    continue;
                }

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

                std::array<m2m_vector, NDIM> Y;
                Y[0] = center_of_masses_SoA.value<0>(interaction_partner_flat_index);
                Y[1] = center_of_masses_SoA.value<1>(interaction_partner_flat_index);
                Y[2] = center_of_masses_SoA.value<2>(interaction_partner_flat_index);
                std::array<m2m_vector, NDIM> dX;
                dX[0] = X[0] - Y[0];
                dX[1] = X[1] - Y[1];
                dX[2] = X[2] - Y[2];

                D_split D_calculator(dX);
                std::array<m2m_vector, 20> D_lower;
                D_calculator.calculate_D_lower(D_lower);

                std::array<m2m_vector, 20> m_partner;

                // Array to store the temporary result - was called A in the old style
                std::array<m2m_vector, 10> cur_pot;
                m_partner[0] = local_expansions_SoA.value<0>(interaction_partner_flat_index);
                cur_pot[0] = m_partner[0] * D_lower[0];
                cur_pot[1] = m_partner[0] * D_lower[1];
                cur_pot[2] = m_partner[0] * D_lower[2];
                cur_pot[3] = m_partner[0] * D_lower[3];

                m_partner[4] = local_expansions_SoA.value<4>(interaction_partner_flat_index);
                m_partner[5] = local_expansions_SoA.value<5>(interaction_partner_flat_index);
                cur_pot[0] += m_partner[4] * (D_lower[4] * factor_half_v[4]);
                cur_pot[1] += m_partner[4] * (D_lower[10] * factor_half_v[4]);
                cur_pot[2] += m_partner[4] * (D_lower[11] * factor_half_v[4]);
                cur_pot[3] += m_partner[4] * (D_lower[12] * factor_half_v[4]);

                cur_pot[0] += m_partner[5] * (D_lower[5] * factor_half_v[5]);
                cur_pot[1] += m_partner[5] * (D_lower[11] * factor_half_v[5]);
                cur_pot[2] += m_partner[5] * (D_lower[13] * factor_half_v[5]);
                cur_pot[3] += m_partner[5] * (D_lower[14] * factor_half_v[5]);

                m_partner[6] = local_expansions_SoA.value<6>(interaction_partner_flat_index);
                m_partner[7] = local_expansions_SoA.value<7>(interaction_partner_flat_index);
                cur_pot[0] += m_partner[6] * (D_lower[6] * factor_half_v[6]);
                cur_pot[1] += m_partner[6] * (D_lower[12] * factor_half_v[6]);
                cur_pot[2] += m_partner[6] * (D_lower[14] * factor_half_v[6]);
                cur_pot[3] += m_partner[6] * (D_lower[15] * factor_half_v[6]);

                cur_pot[0] += m_partner[7] * (D_lower[7] * factor_half_v[7]);
                cur_pot[1] += m_partner[7] * (D_lower[13] * factor_half_v[7]);
                cur_pot[2] += m_partner[7] * (D_lower[16] * factor_half_v[7]);
                cur_pot[3] += m_partner[7] * (D_lower[17] * factor_half_v[7]);

                m_partner[8] = local_expansions_SoA.value<8>(interaction_partner_flat_index);
                m_partner[9] = local_expansions_SoA.value<9>(interaction_partner_flat_index);
                cur_pot[0] += m_partner[8] * (D_lower[8] * factor_half_v[8]);
                cur_pot[1] += m_partner[8] * (D_lower[14] * factor_half_v[8]);
                cur_pot[2] += m_partner[8] * (D_lower[17] * factor_half_v[8]);
                cur_pot[3] += m_partner[8] * (D_lower[18] * factor_half_v[8]);

                cur_pot[0] += m_partner[9] * (D_lower[9] * factor_half_v[9]);
                cur_pot[1] += m_partner[9] * (D_lower[15] * factor_half_v[9]);
                cur_pot[2] += m_partner[9] * (D_lower[18] * factor_half_v[9]);
                cur_pot[3] += m_partner[9] * (D_lower[19] * factor_half_v[9]);

                m_partner[10] = local_expansions_SoA.value<10>(interaction_partner_flat_index);
                m_partner[11] = local_expansions_SoA.value<11>(interaction_partner_flat_index);
                cur_pot[0] -= m_partner[10] * (D_lower[10] * factor_sixth_v[10]);
                cur_pot[0] -= m_partner[11] * (D_lower[11] * factor_sixth_v[11]);
                m_partner[12] = local_expansions_SoA.value<12>(interaction_partner_flat_index);
                m_partner[13] = local_expansions_SoA.value<13>(interaction_partner_flat_index);
                cur_pot[0] -= m_partner[12] * (D_lower[12] * factor_sixth_v[12]);
                cur_pot[0] -= m_partner[13] * (D_lower[13] * factor_sixth_v[13]);
                m_partner[14] = local_expansions_SoA.value<14>(interaction_partner_flat_index);
                m_partner[15] = local_expansions_SoA.value<15>(interaction_partner_flat_index);
                cur_pot[0] -= m_partner[14] * (D_lower[14] * factor_sixth_v[14]);
                cur_pot[0] -= m_partner[15] * (D_lower[15] * factor_sixth_v[15]);
                m_partner[16] = local_expansions_SoA.value<16>(interaction_partner_flat_index);
                m_partner[17] = local_expansions_SoA.value<17>(interaction_partner_flat_index);
                cur_pot[0] -= m_partner[16] * (D_lower[16] * factor_sixth_v[16]);
                cur_pot[0] -= m_partner[17] * (D_lower[17] * factor_sixth_v[17]);
                m_partner[18] = local_expansions_SoA.value<18>(interaction_partner_flat_index);
                m_partner[19] = local_expansions_SoA.value<19>(interaction_partner_flat_index);
                cur_pot[0] -= m_partner[18] * (D_lower[18] * factor_sixth_v[18]);
                cur_pot[0] -= m_partner[19] * (D_lower[19] * factor_sixth_v[19]);

                Vc::where(mask, tmpstore[0]) = tmpstore[0] + cur_pot[0];
                Vc::where(mask, tmpstore[1]) = tmpstore[1] + cur_pot[1];
                Vc::where(mask, tmpstore[2]) = tmpstore[2] + cur_pot[2];
                Vc::where(mask, tmpstore[3]) = tmpstore[3] + cur_pot[3];

                // Was B0 in old style, represents the angular corrections
                m2m_vector current_angular_correction[NDIM];
                std::array<m2m_vector, 15> D_upper;
                D_upper[0] =
                    D_calculator.X[0] * D_calculator.X[0] * D_calculator.d3 + 2.0 * D_calculator.d2;
                m2m_vector d3_X00 = D_calculator.d3 * D_calculator.X_00;
                D_upper[0] += D_calculator.d2;
                D_upper[0] += 5.0 * d3_X00;
                m2m_vector d3_X01 = D_calculator.d3 * D_calculator.X[0] * D_calculator.X[1];
                D_upper[1] = 3.0 * d3_X01;
                m2m_vector X_02 = D_calculator.X[0] * D_calculator.X[2];
                m2m_vector d3_X02 = D_calculator.d3 * X_02;
                D_upper[2] = 3.0 * d3_X02;
                // n0 needs to be a whole m2m vector
                m2m_vector n0_tmp = - m_partner[10];
                // m_cell_iterator++; // 11

                current_angular_correction[0] = n0_tmp * (D_upper[0] * factor_sixth_v[10]);
                current_angular_correction[1] = n0_tmp * (D_upper[1] * factor_sixth_v[10]);
                current_angular_correction[2] = n0_tmp * (D_upper[2] * factor_sixth_v[10]);

                D_upper[3] = D_calculator.d2;
                m2m_vector d3_X11 = D_calculator.d3 * D_calculator.X_11;
                D_upper[3] += d3_X11;
                D_upper[3] += D_calculator.d3 * D_calculator.X_00;
                m2m_vector d3_X12 = D_calculator.d3 * D_calculator.X[1] * D_calculator.X[2];
                D_upper[4] = d3_X12;

                n0_tmp = m_partner[11];

                current_angular_correction[0] -= n0_tmp * (D_upper[1] * factor_sixth_v[11]);
                current_angular_correction[1] -= n0_tmp * (D_upper[3] * factor_sixth_v[11]);
                current_angular_correction[2] -= n0_tmp * (D_upper[4] * factor_sixth_v[11]);

                D_upper[5] = D_calculator.d2;
                m2m_vector d3_X22 = D_calculator.d3 * D_calculator.X_22;
                D_upper[5] += d3_X22;
                D_upper[5] += d3_X00;

                n0_tmp = m_partner[12];

                current_angular_correction[0] -= n0_tmp * (D_upper[2] * factor_sixth_v[12]);
                current_angular_correction[1] -= n0_tmp * (D_upper[4] * factor_sixth_v[12]);
                current_angular_correction[2] -= n0_tmp * (D_upper[5] * factor_sixth_v[12]);

                D_upper[6] = 3.0 * d3_X01;
                D_upper[7] = D_calculator.d3 * X_02;

                n0_tmp = m_partner[13];

                current_angular_correction[0] -= n0_tmp * (D_upper[3] * factor_sixth_v[13]);
                current_angular_correction[1] -= n0_tmp * (D_upper[6] * factor_sixth_v[13]);
                current_angular_correction[2] -= n0_tmp * (D_upper[7] * factor_sixth_v[13]);

                D_upper[8] = D_calculator.d3 * D_calculator.X[0] * D_calculator.X[1];

                n0_tmp = m_partner[14];

                current_angular_correction[0] -= n0_tmp * (D_upper[4] * factor_sixth_v[14]);
                current_angular_correction[1] -= n0_tmp * (D_upper[7] * factor_sixth_v[14]);
                current_angular_correction[2] -= n0_tmp * (D_upper[8] * factor_sixth_v[14]);

                D_upper[9] = 3.0 * d3_X02;

                n0_tmp = m_partner[15];

                current_angular_correction[0] -= n0_tmp * (D_upper[5] * factor_sixth_v[15]);
                current_angular_correction[1] -= n0_tmp * (D_upper[8] * factor_sixth_v[15]);
                current_angular_correction[2] -= n0_tmp * (D_upper[9] * factor_sixth_v[15]);

                D_upper[10] =
                    D_calculator.X[1] * D_calculator.X[1] * D_calculator.d3 + 2.0 * D_calculator.d2;
                D_upper[10] += D_calculator.d2;
                D_upper[10] += 5.0 * d3_X11;

                D_upper[11] = 3.0 * d3_X12;

                n0_tmp = m_partner[16];

                current_angular_correction[0] -= n0_tmp * (D_upper[6] * factor_sixth_v[16]);
                current_angular_correction[1] -= n0_tmp * (D_upper[10] * factor_sixth_v[16]);
                current_angular_correction[2] -= n0_tmp * (D_upper[11] * factor_sixth_v[16]);

                D_upper[12] = D_calculator.d2;
                D_upper[12] += d3_X22;
                D_upper[12] += d3_X11;

                n0_tmp = m_partner[17];

                current_angular_correction[0] -= n0_tmp * (D_upper[7] * factor_sixth_v[17]);
                current_angular_correction[1] -= n0_tmp * (D_upper[11] * factor_sixth_v[17]);
                current_angular_correction[2] -= n0_tmp * (D_upper[12] * factor_sixth_v[17]);

                D_upper[13] = 3.0 * d3_X12;

                n0_tmp = m_partner[18];

                current_angular_correction[0] -= n0_tmp * (D_upper[8] * factor_sixth_v[18]);
                current_angular_correction[1] -= n0_tmp * (D_upper[12] * factor_sixth_v[18]);
                current_angular_correction[2] -= n0_tmp * (D_upper[13] * factor_sixth_v[18]);

                D_upper[14] =
                    D_calculator.X[2] * D_calculator.X[2] * D_calculator.d3 + 2.0 * D_calculator.d2;
                D_upper[14] += D_calculator.d2;
                D_upper[14] += 5.0 * d3_X22;

                n0_tmp = m_partner[19];

                current_angular_correction[0] -= n0_tmp * (D_upper[9] * factor_sixth_v[19]);
                current_angular_correction[1] -= n0_tmp * (D_upper[13] * factor_sixth_v[19]);
                current_angular_correction[2] -= n0_tmp * (D_upper[14] * factor_sixth_v[19]);

                Vc::where(mask, tmp_corrections[0]) =
                    tmp_corrections[0] + current_angular_correction[0];
                Vc::where(mask, tmp_corrections[1]) =
                    tmp_corrections[1] + current_angular_correction[1];
                Vc::where(mask, tmp_corrections[2]) =
                    tmp_corrections[2] + current_angular_correction[2];
                Vc::where(mask, tmp_corrections[3]) =
                    tmp_corrections[3] + current_angular_correction[3];
            }
            tmpstore[0].memstore(potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore[1].memstore(potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore[2].memstore(potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore[3].memstore(potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmp_corrections[0].memstore(
                angular_corrections_SoA.pointer<0>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmp_corrections[1].memstore(
                angular_corrections_SoA.pointer<1>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmp_corrections[2].memstore(
                angular_corrections_SoA.pointer<2>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmp_corrections[3].memstore(
                angular_corrections_SoA.pointer<3>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
        }

        void p2m_kernel::blocked_interaction_non_rho(
            struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>& local_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>& center_of_masses_SoA,
            struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>&
                potential_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                angular_corrections_SoA,
            const multiindex<>& cell_index, const size_t cell_flat_index,
            const multiindex<m2m_int_vector>& cell_index_coarse,
            const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
            const std::vector<multiindex<>>& stencil, const size_t outer_stencil_index,
            std::vector<bool>& interact) {
            // TODO: should change name to something better (not taylor, but space_vector)
            // struct_of_array_taylor<space_vector, real, 3> X =
            //     center_of_masses_SoA.get_view(cell_flat_index);

            std::array<m2m_vector, NDIM> X;
            X[0] = center_of_masses_SoA.value<0>(cell_flat_index);
            X[1] = center_of_masses_SoA.value<1>(cell_flat_index);
            X[2] = center_of_masses_SoA.value<2>(cell_flat_index);
            std::array<m2m_vector, 4> tmpstore;
            tmpstore[0] = potential_expansions_SoA.value<0>(cell_flat_index_unpadded);
            tmpstore[1] = potential_expansions_SoA.value<1>(cell_flat_index_unpadded);
            tmpstore[2] = potential_expansions_SoA.value<2>(cell_flat_index_unpadded);
            tmpstore[3] = potential_expansions_SoA.value<3>(cell_flat_index_unpadded);
            for (size_t inner_stencil_index = 0; inner_stencil_index < P2M_STENCIL_BLOCKING &&
                 outer_stencil_index + inner_stencil_index < stencil.size();
                 inner_stencil_index += 1) {
                const multiindex<>& stencil_element =
                    stencil[outer_stencil_index + inner_stencil_index];
                const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                    cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);

                const size_t interaction_partner_flat_index =
                    to_flat_index_padded(interaction_partner_index);    // iii1n
                if (!interact[interaction_partner_flat_index] &&
                    !interact[interaction_partner_flat_index + 1] &&
                    !interact[interaction_partner_flat_index + 2] &&
                    !interact[interaction_partner_flat_index + 3])
                    continue;

                // check whether all vector elements are in empty border
                if (vector_is_empty[interaction_partner_flat_index]) {
                    continue;
                }

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

                std::array<m2m_vector, NDIM> Y;
                Y[0] = center_of_masses_SoA.value<0>(interaction_partner_flat_index);
                Y[1] = center_of_masses_SoA.value<1>(interaction_partner_flat_index);
                Y[2] = center_of_masses_SoA.value<2>(interaction_partner_flat_index);
                std::array<m2m_vector, NDIM> dX;
                dX[0] = X[0] - Y[0];
                dX[1] = X[1] - Y[1];
                dX[2] = X[2] - Y[2];

                D_split D_calculator(dX);
                std::array<m2m_vector, 20> D_lower;
                D_calculator.calculate_D_lower(D_lower);

                std::array<m2m_vector, 20> m_partner;

                // Array to store the temporary result - was called A in the old style
                std::array<m2m_vector, 10> cur_pot;
                m_partner[0] = local_expansions_SoA.value<0>(interaction_partner_flat_index);
                cur_pot[0] = m_partner[0] * D_lower[0];
                cur_pot[1] = m_partner[0] * D_lower[1];
                cur_pot[2] = m_partner[0] * D_lower[2];
                cur_pot[3] = m_partner[0] * D_lower[3];

                m_partner[4] = local_expansions_SoA.value<4>(interaction_partner_flat_index);
                m_partner[5] = local_expansions_SoA.value<5>(interaction_partner_flat_index);
                cur_pot[0] += m_partner[4] * (D_lower[4] * factor_half_v[4]);
                cur_pot[1] += m_partner[4] * (D_lower[10] * factor_half_v[4]);
                cur_pot[2] += m_partner[4] * (D_lower[11] * factor_half_v[4]);
                cur_pot[3] += m_partner[4] * (D_lower[12] * factor_half_v[4]);

                cur_pot[0] += m_partner[5] * (D_lower[5] * factor_half_v[5]);
                cur_pot[1] += m_partner[5] * (D_lower[11] * factor_half_v[5]);
                cur_pot[2] += m_partner[5] * (D_lower[13] * factor_half_v[5]);
                cur_pot[3] += m_partner[5] * (D_lower[14] * factor_half_v[5]);

                m_partner[6] = local_expansions_SoA.value<6>(interaction_partner_flat_index);
                m_partner[7] = local_expansions_SoA.value<7>(interaction_partner_flat_index);
                cur_pot[0] += m_partner[6] * (D_lower[6] * factor_half_v[6]);
                cur_pot[1] += m_partner[6] * (D_lower[12] * factor_half_v[6]);
                cur_pot[2] += m_partner[6] * (D_lower[14] * factor_half_v[6]);
                cur_pot[3] += m_partner[6] * (D_lower[15] * factor_half_v[6]);

                cur_pot[0] += m_partner[7] * (D_lower[7] * factor_half_v[7]);
                cur_pot[1] += m_partner[7] * (D_lower[13] * factor_half_v[7]);
                cur_pot[2] += m_partner[7] * (D_lower[16] * factor_half_v[7]);
                cur_pot[3] += m_partner[7] * (D_lower[17] * factor_half_v[7]);

                m_partner[8] = local_expansions_SoA.value<8>(interaction_partner_flat_index);
                m_partner[9] = local_expansions_SoA.value<9>(interaction_partner_flat_index);
                cur_pot[0] += m_partner[8] * (D_lower[8] * factor_half_v[8]);
                cur_pot[1] += m_partner[8] * (D_lower[14] * factor_half_v[8]);
                cur_pot[2] += m_partner[8] * (D_lower[17] * factor_half_v[8]);
                cur_pot[3] += m_partner[8] * (D_lower[18] * factor_half_v[8]);

                cur_pot[0] += m_partner[9] * (D_lower[9] * factor_half_v[9]);
                cur_pot[1] += m_partner[9] * (D_lower[15] * factor_half_v[9]);
                cur_pot[2] += m_partner[9] * (D_lower[18] * factor_half_v[9]);
                cur_pot[3] += m_partner[9] * (D_lower[19] * factor_half_v[9]);

                m_partner[10] = local_expansions_SoA.value<10>(interaction_partner_flat_index);
                m_partner[11] = local_expansions_SoA.value<11>(interaction_partner_flat_index);
                cur_pot[0] -= m_partner[10] * (D_lower[10] * factor_sixth_v[10]);
                cur_pot[0] -= m_partner[11] * (D_lower[11] * factor_sixth_v[11]);
                m_partner[12] = local_expansions_SoA.value<12>(interaction_partner_flat_index);
                m_partner[13] = local_expansions_SoA.value<13>(interaction_partner_flat_index);
                cur_pot[0] -= m_partner[12] * (D_lower[12] * factor_sixth_v[12]);
                cur_pot[0] -= m_partner[13] * (D_lower[13] * factor_sixth_v[13]);
                m_partner[14] = local_expansions_SoA.value<14>(interaction_partner_flat_index);
                m_partner[15] = local_expansions_SoA.value<15>(interaction_partner_flat_index);
                cur_pot[0] -= m_partner[14] * (D_lower[14] * factor_sixth_v[14]);
                cur_pot[0] -= m_partner[15] * (D_lower[15] * factor_sixth_v[15]);
                m_partner[16] = local_expansions_SoA.value<16>(interaction_partner_flat_index);
                m_partner[17] = local_expansions_SoA.value<17>(interaction_partner_flat_index);
                cur_pot[0] -= m_partner[16] * (D_lower[16] * factor_sixth_v[16]);
                cur_pot[0] -= m_partner[17] * (D_lower[17] * factor_sixth_v[17]);
                m_partner[18] = local_expansions_SoA.value<18>(interaction_partner_flat_index);
                m_partner[19] = local_expansions_SoA.value<19>(interaction_partner_flat_index);
                cur_pot[0] -= m_partner[18] * (D_lower[18] * factor_sixth_v[18]);
                cur_pot[0] -= m_partner[19] * (D_lower[19] * factor_sixth_v[19]);

                Vc::where(mask, tmpstore[0]) = tmpstore[0] + cur_pot[0];
                Vc::where(mask, tmpstore[1]) = tmpstore[1] + cur_pot[1];
                Vc::where(mask, tmpstore[2]) = tmpstore[2] + cur_pot[2];
                Vc::where(mask, tmpstore[3]) = tmpstore[3] + cur_pot[3];
            }
            tmpstore[0].memstore(potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore[1].memstore(potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore[2].memstore(potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore[3].memstore(potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
        }
    }    // namespace p2m_kernel
}    // namespace fmm
}    // namespace octotiger
