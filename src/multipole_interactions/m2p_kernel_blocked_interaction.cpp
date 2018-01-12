#include "m2p_kernel.hpp"

#include "../common_kernel/helper.hpp"
#include "../common_kernel/kernel_taylor_set_basis.hpp"
#include "../common_kernel/struct_of_array_data.hpp"
#include "grid_flattened_indices.hpp"

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

        void m2p_kernel::blocked_interaction_rho(
            struct_of_array_data<expansion, real, 20, ENTRIES,
                SOA_PADDING>& __restrict__ local_expansions_SoA,
            std::vector<real>& mons, struct_of_array_data<space_vector, real, 3, ENTRIES,
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
            real dX, std::array<real, NDIM>& xbase) {
            // TODO: should change name to something better (not taylor, but space_vector)
            // struct_of_array_taylor<space_vector, real, 3> X =
            //     center_of_masses_SoA.get_view(cell_flat_index);

            std::array<m2m_vector, NDIM> X;
            X[0] = center_of_masses_SoA.value<0>(cell_flat_index);
            X[1] = center_of_masses_SoA.value<1>(cell_flat_index);
            X[2] = center_of_masses_SoA.value<2>(cell_flat_index);
            std::array<m2m_vector, 20> tmpstore;
            tmpstore[0] = potential_expansions_SoA.value<0>(cell_flat_index_unpadded);
            tmpstore[1] = potential_expansions_SoA.value<1>(cell_flat_index_unpadded);
            tmpstore[2] = potential_expansions_SoA.value<2>(cell_flat_index_unpadded);
            tmpstore[3] = potential_expansions_SoA.value<3>(cell_flat_index_unpadded);
            tmpstore[4] = potential_expansions_SoA.value<4>(cell_flat_index_unpadded);
            tmpstore[5] = potential_expansions_SoA.value<5>(cell_flat_index_unpadded);
            tmpstore[6] = potential_expansions_SoA.value<6>(cell_flat_index_unpadded);
            tmpstore[7] = potential_expansions_SoA.value<7>(cell_flat_index_unpadded);
            tmpstore[8] = potential_expansions_SoA.value<8>(cell_flat_index_unpadded);
            tmpstore[9] = potential_expansions_SoA.value<9>(cell_flat_index_unpadded);
            tmpstore[10] = potential_expansions_SoA.value<10>(cell_flat_index_unpadded);
            tmpstore[11] = potential_expansions_SoA.value<11>(cell_flat_index_unpadded);
            tmpstore[12] = potential_expansions_SoA.value<12>(cell_flat_index_unpadded);
            tmpstore[13] = potential_expansions_SoA.value<13>(cell_flat_index_unpadded);
            tmpstore[14] = potential_expansions_SoA.value<14>(cell_flat_index_unpadded);
            tmpstore[15] = potential_expansions_SoA.value<15>(cell_flat_index_unpadded);
            tmpstore[16] = potential_expansions_SoA.value<16>(cell_flat_index_unpadded);
            tmpstore[17] = potential_expansions_SoA.value<17>(cell_flat_index_unpadded);
            tmpstore[18] = potential_expansions_SoA.value<18>(cell_flat_index_unpadded);
            tmpstore[19] = potential_expansions_SoA.value<19>(cell_flat_index_unpadded);

            std::array<m2m_vector, 3> tmp_corrections;
            tmp_corrections[0] = angular_corrections_SoA.value<0>(cell_flat_index_unpadded);
            tmp_corrections[1] = angular_corrections_SoA.value<1>(cell_flat_index_unpadded);
            tmp_corrections[2] = angular_corrections_SoA.value<2>(cell_flat_index_unpadded);
            bool changed_data = false;
            for (size_t inner_stencil_index = 0; inner_stencil_index < STENCIL_BLOCKING &&
                 outer_stencil_index + inner_stencil_index < stencil.size();
                 inner_stencil_index += 1) {
                const multiindex<>& stencil_element =
                    stencil[outer_stencil_index + inner_stencil_index];
                const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                    cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);
                const multiindex<> interaction_partner_index_unpadded(
                    cell_index_unpadded.x + stencil_element.x,
                    cell_index_unpadded.y + stencil_element.y,
                    cell_index_unpadded.z + stencil_element.z);

                const size_t interaction_partner_flat_index =
                    to_flat_index_padded(interaction_partner_index);    // iii1n

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
                changed_data = true;

                std::array<m2m_vector, NDIM> Y;
                for (auto i = 0; i < m2m_vector::size(); ++i) {
                    Y[0][i] = (interaction_partner_index_unpadded.x) * dX + xbase[0];
                    Y[1][i] = (interaction_partner_index_unpadded.y) * dX + xbase[1];
                    Y[2][i] = (interaction_partner_index_unpadded.z + i) * dX + xbase[2];
                }

                std::array<m2m_vector, NDIM> dX;
                dX[0] = X[0] - Y[0];
                dX[1] = X[1] - Y[1];
                dX[2] = X[2] - Y[2];

                m2m_vector monopole;
                Vc::where(mask, monopole) = m2m_vector(
                    mons.data() + interaction_partner_flat_index, Vc::flags::element_aligned);

                D_split D_calculator(dX);
                std::array<m2m_vector, 20> D_lower;
                D_calculator.calculate_D_lower(D_lower);

                tmpstore[0] = tmpstore[0] + monopole * D_lower[0];
                tmpstore[1] = tmpstore[1] + monopole * D_lower[1];
                tmpstore[2] = tmpstore[2] + monopole * D_lower[2];
                tmpstore[3] = tmpstore[3] + monopole * D_lower[3];
                tmpstore[4] = tmpstore[4] + monopole * D_lower[4];
                tmpstore[5] = tmpstore[5] + monopole * D_lower[5];
                tmpstore[6] = tmpstore[6] + monopole * D_lower[6];
                tmpstore[7] = tmpstore[7] + monopole * D_lower[7];
                tmpstore[8] = tmpstore[8] + monopole * D_lower[8];
                tmpstore[9] = tmpstore[9] + monopole * D_lower[9];
                tmpstore[10] = tmpstore[10] + monopole * D_lower[10];
                tmpstore[11] = tmpstore[11] + monopole * D_lower[11];
                tmpstore[12] = tmpstore[12] + monopole * D_lower[12];
                tmpstore[13] = tmpstore[13] + monopole * D_lower[13];
                tmpstore[14] = tmpstore[14] + monopole * D_lower[14];
                tmpstore[15] = tmpstore[15] + monopole * D_lower[15];
                tmpstore[16] = tmpstore[16] + monopole * D_lower[16];
                tmpstore[17] = tmpstore[17] + monopole * D_lower[17];
                tmpstore[18] = tmpstore[18] + monopole * D_lower[18];
                tmpstore[19] = tmpstore[19] + monopole * D_lower[19];

                m2m_vector const n0_constant =
                    monopole / local_expansions_SoA.value<0>(cell_flat_index);
                // std::cout << n0_constant << std::endl;
                // std::cin.get();
                std::array<m2m_vector, 15> D_upper;
                // D_calculator.calculate_D_upper(D_upper);

                m2m_vector current_angular_correction[NDIM];
                current_angular_correction[0] = 0.0;
                current_angular_correction[1] = 0.0;
                current_angular_correction[2] = 0.0;

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
                m2m_vector n0_tmp = -local_expansions_SoA.value<10>(cell_flat_index) * n0_constant;
                // m_cell_iterator++; // 11

                // B0?
                current_angular_correction[0] -= n0_tmp * (D_upper[0] * factor_sixth_v[10]);
                current_angular_correction[1] -= n0_tmp * (D_upper[1] * factor_sixth_v[10]);
                current_angular_correction[2] -= n0_tmp * (D_upper[2] * factor_sixth_v[10]);

                D_upper[3] = D_calculator.d2;
                m2m_vector d3_X11 = D_calculator.d3 * D_calculator.X_11;
                D_upper[3] += d3_X11;
                D_upper[3] += D_calculator.d3 * D_calculator.X_00;
                m2m_vector d3_X12 = D_calculator.d3 * D_calculator.X[1] * D_calculator.X[2];
                D_upper[4] = d3_X12;

                n0_tmp = -local_expansions_SoA.value<11>(cell_flat_index) * n0_constant;

                current_angular_correction[0] -= n0_tmp * (D_upper[1] * factor_sixth_v[11]);
                current_angular_correction[1] -= n0_tmp * (D_upper[3] * factor_sixth_v[11]);
                current_angular_correction[2] -= n0_tmp * (D_upper[4] * factor_sixth_v[11]);

                D_upper[5] = D_calculator.d2;
                m2m_vector d3_X22 = D_calculator.d3 * D_calculator.X_22;
                D_upper[5] += d3_X22;
                D_upper[5] += d3_X00;

                n0_tmp = -local_expansions_SoA.value<12>(cell_flat_index) * n0_constant;

                current_angular_correction[0] -= n0_tmp * (D_upper[2] * factor_sixth_v[12]);
                current_angular_correction[1] -= n0_tmp * (D_upper[4] * factor_sixth_v[12]);
                current_angular_correction[2] -= n0_tmp * (D_upper[5] * factor_sixth_v[12]);

                D_upper[6] = 3.0 * d3_X01;
                D_upper[7] = D_calculator.d3 * X_02;

                n0_tmp = -local_expansions_SoA.value<13>(cell_flat_index) * n0_constant;

                current_angular_correction[0] -= n0_tmp * (D_upper[3] * factor_sixth_v[13]);
                current_angular_correction[1] -= n0_tmp * (D_upper[6] * factor_sixth_v[13]);
                current_angular_correction[2] -= n0_tmp * (D_upper[7] * factor_sixth_v[13]);

                D_upper[8] = D_calculator.d3 * D_calculator.X[0] * D_calculator.X[1];

                n0_tmp = -local_expansions_SoA.value<14>(cell_flat_index) * n0_constant;

                current_angular_correction[0] -= n0_tmp * (D_upper[4] * factor_sixth_v[14]);
                current_angular_correction[1] -= n0_tmp * (D_upper[7] * factor_sixth_v[14]);
                current_angular_correction[2] -= n0_tmp * (D_upper[8] * factor_sixth_v[14]);

                D_upper[9] = 3.0 * d3_X02;

                n0_tmp = -local_expansions_SoA.value<15>(cell_flat_index) * n0_constant;

                current_angular_correction[0] -= n0_tmp * (D_upper[5] * factor_sixth_v[15]);
                current_angular_correction[1] -= n0_tmp * (D_upper[8] * factor_sixth_v[15]);
                current_angular_correction[2] -= n0_tmp * (D_upper[9] * factor_sixth_v[15]);

                D_upper[10] =
                    D_calculator.X[1] * D_calculator.X[1] * D_calculator.d3 + 2.0 * D_calculator.d2;
                D_upper[10] += D_calculator.d2;
                D_upper[10] += 5.0 * d3_X11;

                D_upper[11] = 3.0 * d3_X12;

                n0_tmp = -local_expansions_SoA.value<16>(cell_flat_index) * n0_constant;

                current_angular_correction[0] -= n0_tmp * (D_upper[6] * factor_sixth_v[16]);
                current_angular_correction[1] -= n0_tmp * (D_upper[10] * factor_sixth_v[16]);
                current_angular_correction[2] -= n0_tmp * (D_upper[11] * factor_sixth_v[16]);

                D_upper[12] = D_calculator.d2;
                D_upper[12] += d3_X22;
                D_upper[12] += d3_X11;

                n0_tmp = -local_expansions_SoA.value<17>(cell_flat_index) * n0_constant;

                current_angular_correction[0] -= n0_tmp * (D_upper[7] * factor_sixth_v[17]);
                current_angular_correction[1] -= n0_tmp * (D_upper[11] * factor_sixth_v[17]);
                current_angular_correction[2] -= n0_tmp * (D_upper[12] * factor_sixth_v[17]);

                D_upper[13] = 3.0 * d3_X12;

                n0_tmp = -local_expansions_SoA.value<18>(cell_flat_index) * n0_constant;

                current_angular_correction[0] -= n0_tmp * (D_upper[8] * factor_sixth_v[18]);
                current_angular_correction[1] -= n0_tmp * (D_upper[12] * factor_sixth_v[18]);
                current_angular_correction[2] -= n0_tmp * (D_upper[13] * factor_sixth_v[18]);

                D_upper[14] =
                    D_calculator.X[2] * D_calculator.X[2] * D_calculator.d3 + 2.0 * D_calculator.d2;
                D_upper[14] += D_calculator.d2;
                D_upper[14] += 5.0 * d3_X22;

                n0_tmp = -local_expansions_SoA.value<19>(cell_flat_index) * n0_constant;

                current_angular_correction[0] -= n0_tmp * (D_upper[9] * factor_sixth_v[19]);
                current_angular_correction[1] -= n0_tmp * (D_upper[13] * factor_sixth_v[19]);
                current_angular_correction[2] -= n0_tmp * (D_upper[14] * factor_sixth_v[19]);

                tmp_corrections[0] = tmp_corrections[0] + current_angular_correction[0];
                tmp_corrections[1] = tmp_corrections[1] + current_angular_correction[1];
                tmp_corrections[2] = tmp_corrections[2] + current_angular_correction[2];
                tmp_corrections[3] = tmp_corrections[3] + current_angular_correction[3];
            }
            if (changed_data) {
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
        }

        void m2p_kernel::blocked_interaction_non_rho(std::vector<real>& mons,
            struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>& center_of_masses_SoA,
            struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>&
                potential_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                angular_corrections_SoA,
            const multiindex<>& cell_index, const size_t cell_flat_index,
            const multiindex<m2m_int_vector>& cell_index_coarse,
            const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
            const std::vector<multiindex<>>& stencil, const size_t outer_stencil_index, real dX,
            std::array<real, NDIM>& xbase) {
            // TODO: should change name to something better (not taylor, but space_vector)
            // struct_of_array_taylor<space_vector, real, 3> X =
            //     center_of_masses_SoA.get_view(cell_flat_index);

            std::array<m2m_vector, NDIM> X;
            X[0] = center_of_masses_SoA.value<0>(cell_flat_index);
            X[1] = center_of_masses_SoA.value<1>(cell_flat_index);
            X[2] = center_of_masses_SoA.value<2>(cell_flat_index);
            std::array<m2m_vector, 20> tmpstore;
            tmpstore[0] = potential_expansions_SoA.value<0>(cell_flat_index_unpadded);
            tmpstore[1] = potential_expansions_SoA.value<1>(cell_flat_index_unpadded);
            tmpstore[2] = potential_expansions_SoA.value<2>(cell_flat_index_unpadded);
            tmpstore[3] = potential_expansions_SoA.value<3>(cell_flat_index_unpadded);
            tmpstore[4] = potential_expansions_SoA.value<4>(cell_flat_index_unpadded);
            tmpstore[5] = potential_expansions_SoA.value<5>(cell_flat_index_unpadded);
            tmpstore[6] = potential_expansions_SoA.value<6>(cell_flat_index_unpadded);
            tmpstore[7] = potential_expansions_SoA.value<7>(cell_flat_index_unpadded);
            tmpstore[8] = potential_expansions_SoA.value<8>(cell_flat_index_unpadded);
            tmpstore[9] = potential_expansions_SoA.value<9>(cell_flat_index_unpadded);
            tmpstore[10] = potential_expansions_SoA.value<10>(cell_flat_index_unpadded);
            tmpstore[11] = potential_expansions_SoA.value<11>(cell_flat_index_unpadded);
            tmpstore[12] = potential_expansions_SoA.value<12>(cell_flat_index_unpadded);
            tmpstore[13] = potential_expansions_SoA.value<13>(cell_flat_index_unpadded);
            tmpstore[14] = potential_expansions_SoA.value<14>(cell_flat_index_unpadded);
            tmpstore[15] = potential_expansions_SoA.value<15>(cell_flat_index_unpadded);
            tmpstore[16] = potential_expansions_SoA.value<16>(cell_flat_index_unpadded);
            tmpstore[17] = potential_expansions_SoA.value<17>(cell_flat_index_unpadded);
            tmpstore[18] = potential_expansions_SoA.value<18>(cell_flat_index_unpadded);
            tmpstore[19] = potential_expansions_SoA.value<19>(cell_flat_index_unpadded);
            bool changed_data = false;
            for (size_t inner_stencil_index = 0; inner_stencil_index < STENCIL_BLOCKING &&
                 outer_stencil_index + inner_stencil_index < stencil.size();
                 inner_stencil_index += 1) {
                const multiindex<>& stencil_element =
                    stencil[outer_stencil_index + inner_stencil_index];
                const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                    cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);
                const multiindex<> interaction_partner_index_unpadded(
                    cell_index_unpadded.x + stencil_element.x,
                    cell_index_unpadded.y + stencil_element.y,
                    cell_index_unpadded.z + stencil_element.z);

                const size_t interaction_partner_flat_index =
                    to_flat_index_padded(interaction_partner_index);    // iii1n

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
                changed_data = true;

                std::array<m2m_vector, NDIM> X;
                X[0] = center_of_masses_SoA.value<0>(cell_flat_index);
                X[1] = center_of_masses_SoA.value<1>(cell_flat_index);
                X[2] = center_of_masses_SoA.value<2>(cell_flat_index);

                std::array<m2m_vector, NDIM> Y;
                for (auto i = 0; i < m2m_vector::size(); ++i) {
                    Y[0][i] = (interaction_partner_index_unpadded.x) * dX + xbase[0];
                    Y[1][i] = (interaction_partner_index_unpadded.y) * dX + xbase[1];
                    Y[2][i] = (interaction_partner_index_unpadded.z + i) * dX + xbase[2];
                }

                std::array<m2m_vector, NDIM> dX;
                dX[0] = X[0] - Y[0];
                dX[1] = X[1] - Y[1];
                dX[2] = X[2] - Y[2];

                m2m_vector monopole;
                Vc::where(mask, monopole) = m2m_vector(
                    mons.data() + interaction_partner_flat_index, Vc::flags::element_aligned);

                D_split D_calculator(dX);
                std::array<m2m_vector, 20> D_lower;
                D_calculator.calculate_D_lower(D_lower);

                tmpstore[0] = tmpstore[0] + monopole * D_lower[0];
                tmpstore[1] = tmpstore[1] + monopole * D_lower[1];
                tmpstore[2] = tmpstore[2] + monopole * D_lower[2];
                tmpstore[3] = tmpstore[3] + monopole * D_lower[3];
                tmpstore[4] = tmpstore[4] + monopole * D_lower[4];
                tmpstore[5] = tmpstore[5] + monopole * D_lower[5];
                tmpstore[6] = tmpstore[6] + monopole * D_lower[6];
                tmpstore[7] = tmpstore[7] + monopole * D_lower[7];
                tmpstore[8] = tmpstore[8] + monopole * D_lower[8];
                tmpstore[9] = tmpstore[9] + monopole * D_lower[9];
                tmpstore[10] = tmpstore[10] + monopole * D_lower[10];
                tmpstore[11] = tmpstore[11] + monopole * D_lower[11];
                tmpstore[12] = tmpstore[12] + monopole * D_lower[12];
                tmpstore[13] = tmpstore[13] + monopole * D_lower[13];
                tmpstore[14] = tmpstore[14] + monopole * D_lower[14];
                tmpstore[15] = tmpstore[15] + monopole * D_lower[15];
                tmpstore[16] = tmpstore[16] + monopole * D_lower[16];
                tmpstore[17] = tmpstore[17] + monopole * D_lower[17];
                tmpstore[18] = tmpstore[18] + monopole * D_lower[18];
                tmpstore[19] = tmpstore[19] + monopole * D_lower[19];
            }
            if (changed_data) {
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
