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
    namespace m2p_kernel {

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
            const std::vector<multiindex<>>& __restrict__ stencil,
            const size_t outer_stencil_index) {
            // TODO: should change name to something better (not taylor, but space_vector)
            // struct_of_array_taylor<space_vector, real, 3> X =
            //     center_of_masses_SoA.get_view(cell_flat_index);

            for (size_t inner_stencil_index = 0; inner_stencil_index < STENCIL_BLOCKING &&
                 outer_stencil_index + inner_stencil_index < stencil.size();
                 inner_stencil_index += 1) {
                const multiindex<>& stencil_element =
                    stencil[outer_stencil_index + inner_stencil_index];
                const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                    cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);

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
                std::array<m2m_vector, NDIM> X;
                X[0] = center_of_masses_SoA.value<0>(cell_flat_index);
                X[1] = center_of_masses_SoA.value<1>(cell_flat_index);
                X[2] = center_of_masses_SoA.value<2>(cell_flat_index);

                std::array<m2m_vector, NDIM> Y;
                Y[0] =
                    center_of_masses_SoA.value<0>(interaction_partner_flat_index) * dX + xBase[0];
                Y[1] =
                    center_of_masses_SoA.value<1>(interaction_partner_flat_index) * dX + xBase[1];
                Y[2] =
                    center_of_masses_SoA.value<2>(interaction_partner_flat_index) * dX + xBase[2];

                std::array<m2m_vector, NDIM> dX;
                dX[0] = X[0] - Y[0];
                dX[1] = X[1] - Y[1];
                dX[2] = X[2] - Y[2];

                m2m_vector monopole(
                    mons.data() + interaction_partner_flat_index, Vc::flags::element_aligned);

                D_split D_calculator(dX);
                std::array<m2m_vector, 20> D_lower;
                D_calculator.calculate_D_lower(D_lower);

                std::array<m2m_vector, 20> cur_pot;
                cur_pot[0] = monopole * D_lower[0];
                cur_pot[1] = monopole * (D_lower[1]);
                cur_pot[2] = monopole * (D_lower[2]);
                cur_pot[3] = monopole * (D_lower[3]);
                cur_pot[4] = monopole * (D_lower[4]);
                cur_pot[5] = monopole * (D_lower[5]);
                cur_pot[6] = monopole * (D_lower[6]);
                cur_pot[7] = monopole * (D_lower[7]);
                cur_pot[8] = monopole * (D_lower[8]);
                cur_pot[9] = monopole * (D_lower[9]);
                cur_pot[10] = monopole * (D_lower[10]);
                cur_pot[11] = monopole * (D_lower[11]);
                cur_pot[12] = monopole * (D_lower[12]);
                cur_pot[13] = monopole * (D_lower[13]);
                cur_pot[14] = monopole * (D_lower[14]);
                cur_pot[15] = monopole * (D_lower[15]);
                cur_pot[16] = monopole * (D_lower[16]);
                cur_pot[17] = monopole * (D_lower[17]);
                cur_pot[18] = monopole * (D_lower[18]);
                cur_pot[19] = monopole * (D_lower[19]);

                m2m_vector tmp =
                    potential_expansions_SoA.value<0>(cell_flat_index_unpadded) + cur_pot[0];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = potential_expansions_SoA.value<1>(cell_flat_index_unpadded) + cur_pot[1];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = potential_expansions_SoA.value<2>(cell_flat_index_unpadded) + cur_pot[2];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = potential_expansions_SoA.value<3>(cell_flat_index_unpadded) + cur_pot[3];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = potential_expansions_SoA.value<4>(cell_flat_index_unpadded) + cur_pot[4];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<4>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = potential_expansions_SoA.value<5>(cell_flat_index_unpadded) + cur_pot[5];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<5>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = potential_expansions_SoA.value<6>(cell_flat_index_unpadded) + cur_pot[6];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<6>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = potential_expansions_SoA.value<7>(cell_flat_index_unpadded) + cur_pot[7];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<7>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = potential_expansions_SoA.value<8>(cell_flat_index_unpadded) + cur_pot[8];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<8>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = potential_expansions_SoA.value<9>(cell_flat_index_unpadded) + cur_pot[9];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<9>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = potential_expansions_SoA.value<10>(cell_flat_index_unpadded) + cur_pot[10];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<10>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = potential_expansions_SoA.value<11>(cell_flat_index_unpadded) + cur_pot[11];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<11>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = potential_expansions_SoA.value<12>(cell_flat_index_unpadded) + cur_pot[12];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<12>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = potential_expansions_SoA.value<13>(cell_flat_index_unpadded) + cur_pot[13];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<13>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = potential_expansions_SoA.value<14>(cell_flat_index_unpadded) + cur_pot[14];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<14>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = potential_expansions_SoA.value<15>(cell_flat_index_unpadded) + cur_pot[15];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<15>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = potential_expansions_SoA.value<16>(cell_flat_index_unpadded) + cur_pot[16];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<16>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = potential_expansions_SoA.value<17>(cell_flat_index_unpadded) + cur_pot[17];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<17>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = potential_expansions_SoA.value<18>(cell_flat_index_unpadded) + cur_pot[18];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<18>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = potential_expansions_SoA.value<19>(cell_flat_index_unpadded) + cur_pot[19];
                Vc::where(mask, tmp).memstore(
                    potential_expansions_SoA.pointer<19>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                m2m_vector const n0_constant =
                    monopole / local_expansions_SoA.value<0>(cell_flat_index);
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
                m2m_vector n0_tmp = local_expansions_SoA.value<10>(cell_flat_index) * n0_constant;
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

                n0_tmp = local_expansions_SoA.value<11>(cell_flat_index) * n0_constant;

                current_angular_correction[0] -= n0_tmp * (D_upper[1] * factor_sixth_v[11]);
                current_angular_correction[1] -= n0_tmp * (D_upper[3] * factor_sixth_v[11]);
                current_angular_correction[2] -= n0_tmp * (D_upper[4] * factor_sixth_v[11]);

                D_upper[5] = D_calculator.d2;
                m2m_vector d3_X22 = D_calculator.d3 * D_calculator.X_22;
                D_upper[5] += d3_X22;
                D_upper[5] += d3_X00;

                n0_tmp = local_expansions_SoA.value<12>(cell_flat_index) * n0_constant;

                current_angular_correction[0] -= n0_tmp * (D_upper[2] * factor_sixth_v[12]);
                current_angular_correction[1] -= n0_tmp * (D_upper[4] * factor_sixth_v[12]);
                current_angular_correction[2] -= n0_tmp * (D_upper[5] * factor_sixth_v[12]);

                D_upper[6] = 3.0 * d3_X01;
                D_upper[7] = D_calculator.d3 * X_02;

                n0_tmp = local_expansions_SoA.value<13>(cell_flat_index) * n0_constant;

                current_angular_correction[0] -= n0_tmp * (D_upper[3] * factor_sixth_v[13]);
                current_angular_correction[1] -= n0_tmp * (D_upper[6] * factor_sixth_v[13]);
                current_angular_correction[2] -= n0_tmp * (D_upper[7] * factor_sixth_v[13]);

                D_upper[8] = D_calculator.d3 * D_calculator.X[0] * D_calculator.X[1];

                n0_tmp = local_expansions_SoA.value<14>(cell_flat_index) * n0_constant;

                current_angular_correction[0] -= n0_tmp * (D_upper[4] * factor_sixth_v[14]);
                current_angular_correction[1] -= n0_tmp * (D_upper[7] * factor_sixth_v[14]);
                current_angular_correction[2] -= n0_tmp * (D_upper[8] * factor_sixth_v[14]);

                D_upper[9] = 3.0 * d3_X02;

                n0_tmp = local_expansions_SoA.value<15>(cell_flat_index) * n0_constant;

                current_angular_correction[0] -= n0_tmp * (D_upper[5] * factor_sixth_v[15]);
                current_angular_correction[1] -= n0_tmp * (D_upper[8] * factor_sixth_v[15]);
                current_angular_correction[2] -= n0_tmp * (D_upper[9] * factor_sixth_v[15]);

                D_upper[10] =
                    D_calculator.X[1] * D_calculator.X[1] * D_calculator.d3 + 2.0 * D_calculator.d2;
                D_upper[10] += D_calculator.d2;
                D_upper[10] += 5.0 * d3_X11;

                D_upper[11] = 3.0 * d3_X12;

                n0_tmp = local_expansions_SoA.value<16>(cell_flat_index) * n0_constant;

                current_angular_correction[0] -= n0_tmp * (D_upper[6] * factor_sixth_v[16]);
                current_angular_correction[1] -= n0_tmp * (D_upper[10] * factor_sixth_v[16]);
                current_angular_correction[2] -= n0_tmp * (D_upper[11] * factor_sixth_v[16]);

                D_upper[12] = D_calculator.d2;
                D_upper[12] += d3_X22;
                D_upper[12] += d3_X11;

                n0_tmp = local_expansions_SoA.value<17>(cell_flat_index) * n0_constant;

                current_angular_correction[0] -= n0_tmp * (D_upper[7] * factor_sixth_v[17]);
                current_angular_correction[1] -= n0_tmp * (D_upper[11] * factor_sixth_v[17]);
                current_angular_correction[2] -= n0_tmp * (D_upper[12] * factor_sixth_v[17]);

                D_upper[13] = 3.0 * d3_X12;

                n0_tmp = local_expansions_SoA.value<18>(cell_flat_index) * n0_constant;

                current_angular_correction[0] -= n0_tmp * (D_upper[8] * factor_sixth_v[18]);
                current_angular_correction[1] -= n0_tmp * (D_upper[12] * factor_sixth_v[18]);
                current_angular_correction[2] -= n0_tmp * (D_upper[13] * factor_sixth_v[18]);

                D_upper[14] =
                    D_calculator.X[2] * D_calculator.X[2] * D_calculator.d3 + 2.0 * D_calculator.d2;
                D_upper[14] += D_calculator.d2;
                D_upper[14] += 5.0 * d3_X22;

                n0_tmp = local_expansions_SoA.value<19>(cell_flat_index) * n0_constant;

                current_angular_correction[0] -= n0_tmp * (D_upper[9] * factor_sixth_v[19]);
                current_angular_correction[1] -= n0_tmp * (D_upper[13] * factor_sixth_v[19]);
                current_angular_correction[2] -= n0_tmp * (D_upper[14] * factor_sixth_v[19]);

                // L?
                tmp = angular_corrections_SoA.value<0>(cell_flat_index_unpadded) +
                    current_angular_correction[0];
                Vc::where(mask, tmp).memstore(
                    angular_corrections_SoA.pointer<0>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = angular_corrections_SoA.value<1>(cell_flat_index_unpadded) +
                    current_angular_correction[1];
                Vc::where(mask, tmp).memstore(
                    angular_corrections_SoA.pointer<1>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);

                tmp = angular_corrections_SoA.value<2>(cell_flat_index_unpadded) +
                    current_angular_correction[2];
                Vc::where(mask, tmp).memstore(
                    angular_corrections_SoA.pointer<2>(cell_flat_index_unpadded),
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
            const std::vector<multiindex<>>& stencil, const size_t outer_stencil_index) {
            // TODO: should change name to something better (not taylor, but space_vector)
            // struct_of_array_taylor<space_vector, real, 3> X =
            //     center_of_masses_SoA.get_view(cell_flat_index);

            std::array<m2m_vector, 4> d_components;

            for (size_t inner_stencil_index = 0; inner_stencil_index < STENCIL_BLOCKING &&
                 outer_stencil_index + inner_stencil_index < stencil.size();
                 inner_stencil_index += 1) {
                {
                    const multiindex<>& stencil_element =
                        stencil[outer_stencil_index + inner_stencil_index];
                    const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                        cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);

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
                    std::array<m2m_vector, NDIM> X;
                    X[0] = center_of_masses_SoA.value<0>(cell_flat_index);
                    X[1] = center_of_masses_SoA.value<1>(cell_flat_index);
                    X[2] = center_of_masses_SoA.value<2>(cell_flat_index);

                    std::array<m2m_vector, NDIM> Y;
                    Y[0] = center_of_masses_SoA.value<0>(interaction_partner_flat_index) * dX +
                        xBase[0];
                    Y[1] = center_of_masses_SoA.value<1>(interaction_partner_flat_index) * dX +
                        xBase[1];
                    Y[2] = center_of_masses_SoA.value<2>(interaction_partner_flat_index) * dX +
                        xBase[2];

                    std::array<m2m_vector, NDIM> dX;
                    dX[0] = X[0] - Y[0];
                    dX[1] = X[1] - Y[1];
                    dX[2] = X[2] - Y[2];

                    m2m_vector monopole(
                        mons.data() + interaction_partner_flat_index, Vc::flags::element_aligned);

                    D_split D_calculator(dX);
                    std::array<m2m_vector, 20> D_lower;
                    D_calculator.calculate_D_lower(D_lower);

                    std::array<m2m_vector, 20> cur_pot;
                    cur_pot[0] = monopole * D_lower[0];
                    cur_pot[1] = monopole * (D_lower[1]);
                    cur_pot[2] = monopole * (D_lower[2]);
                    cur_pot[3] = monopole * (D_lower[3]);
                    cur_pot[4] = monopole * (D_lower[4]);
                    cur_pot[5] = monopole * (D_lower[5]);
                    cur_pot[6] = monopole * (D_lower[6]);
                    cur_pot[7] = monopole * (D_lower[7]);
                    cur_pot[8] = monopole * (D_lower[8]);
                    cur_pot[9] = monopole * (D_lower[9]);
                    cur_pot[10] = monopole * (D_lower[10]);
                    cur_pot[11] = monopole * (D_lower[11]);
                    cur_pot[12] = monopole * (D_lower[12]);
                    cur_pot[13] = monopole * (D_lower[13]);
                    cur_pot[14] = monopole * (D_lower[14]);
                    cur_pot[15] = monopole * (D_lower[15]);
                    cur_pot[16] = monopole * (D_lower[16]);
                    cur_pot[17] = monopole * (D_lower[17]);
                    cur_pot[18] = monopole * (D_lower[18]);
                    cur_pot[19] = monopole * (D_lower[19]);

                    m2m_vector tmp =
                        potential_expansions_SoA.value<0>(cell_flat_index_unpadded) + cur_pot[0];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);

                    tmp = potential_expansions_SoA.value<1>(cell_flat_index_unpadded) + cur_pot[1];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);

                    tmp = potential_expansions_SoA.value<2>(cell_flat_index_unpadded) + cur_pot[2];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);

                    tmp = potential_expansions_SoA.value<3>(cell_flat_index_unpadded) + cur_pot[3];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);

                    tmp = potential_expansions_SoA.value<4>(cell_flat_index_unpadded) + cur_pot[4];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<4>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);

                    tmp = potential_expansions_SoA.value<5>(cell_flat_index_unpadded) + cur_pot[5];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<5>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);

                    tmp = potential_expansions_SoA.value<6>(cell_flat_index_unpadded) + cur_pot[6];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<6>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);

                    tmp = potential_expansions_SoA.value<7>(cell_flat_index_unpadded) + cur_pot[7];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<7>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);

                    tmp = potential_expansions_SoA.value<8>(cell_flat_index_unpadded) + cur_pot[8];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<8>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);

                    tmp = potential_expansions_SoA.value<9>(cell_flat_index_unpadded) + cur_pot[9];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<9>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);

                    tmp =
                        potential_expansions_SoA.value<10>(cell_flat_index_unpadded) + cur_pot[10];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<10>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);

                    tmp =
                        potential_expansions_SoA.value<11>(cell_flat_index_unpadded) + cur_pot[11];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<11>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);

                    tmp =
                        potential_expansions_SoA.value<12>(cell_flat_index_unpadded) + cur_pot[12];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<12>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);

                    tmp =
                        potential_expansions_SoA.value<13>(cell_flat_index_unpadded) + cur_pot[13];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<13>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);

                    tmp =
                        potential_expansions_SoA.value<14>(cell_flat_index_unpadded) + cur_pot[14];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<14>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);

                    tmp =
                        potential_expansions_SoA.value<15>(cell_flat_index_unpadded) + cur_pot[15];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<15>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);

                    tmp =
                        potential_expansions_SoA.value<16>(cell_flat_index_unpadded) + cur_pot[16];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<16>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);

                    tmp =
                        potential_expansions_SoA.value<17>(cell_flat_index_unpadded) + cur_pot[17];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<17>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);

                    tmp =
                        potential_expansions_SoA.value<18>(cell_flat_index_unpadded) + cur_pot[18];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<18>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);

                    tmp =
                        potential_expansions_SoA.value<19>(cell_flat_index_unpadded) + cur_pot[19];
                    Vc::where(mask, tmp).memstore(
                        potential_expansions_SoA.pointer<19>(cell_flat_index_unpadded),
                        Vc::flags::element_aligned);
                }
            }
        }
    }    // namespace m2p_kernel
}    // namespace fmm
}    // namespace octotiger
