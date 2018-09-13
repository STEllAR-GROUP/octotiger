#include "p2m_kernel.hpp"

#include "../common_kernel/helper.hpp"
#include "../common_kernel/struct_of_array_data.hpp"
#include "../common_kernel/kernel_taylor_set_basis.hpp"
#include "defs.hpp"
#include "interaction_types.hpp"
#include "options.hpp"

#include <array>
#include <functional>

// std::vector<interaction_type> ilist_debugging;

extern options opts;
extern taylor<4, real> factor;
extern taylor<4, m2m_vector> factor_half_v;
extern taylor<4, m2m_vector> factor_sixth_v;

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        p2m_kernel::p2m_kernel(std::vector<bool>& neighbor_empty)
          : neighbor_empty(neighbor_empty)
          , theta_rec_squared(sqr(1.0 / opts.theta)) {
            for (size_t i = 0; i < m2m_int_vector::size(); i++) {
                offset_vector[i] = i;
            }
        }

        void p2m_kernel::apply_stencil(
            struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>& local_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>& center_of_masses_SoA,
            struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>&
                potential_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>&
                angular_corrections_SoA,
            const std::vector<multiindex<>>& stencil, gsolve_type type, bool (&z_skip)[3][3][3],
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

        void p2m_kernel::vectors_check_empty() {
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

        void p2m_kernel::blocked_interaction_rho(
            struct_of_array_data<expansion, real, 20, ENTRIES,
                SOA_PADDING>& __restrict__ local_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, ENTRIES,
                SOA_PADDING>& __restrict__ center_of_masses_SoA,
            struct_of_array_data<expansion, real, 20, INNER_CELLS,
                SOA_PADDING>& __restrict__ potential_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, INNER_CELLS,
                SOA_PADDING>& __restrict__ angular_corrections_SoA,
            const multiindex<>& __restrict__ cell_index, const size_t cell_flat_index,
            const multiindex<m2m_int_vector>& __restrict__ cell_index_coarse,
            const multiindex<>& __restrict__ cell_index_unpadded,
            const size_t cell_flat_index_unpadded, const multiindex<>& interaction_partner_index,
            const size_t interaction_partner_flat_index,
            multiindex<m2m_int_vector>& interaction_partner_index_coarse) {
            std::array<m2m_vector, NDIM> X;
            X[0] = center_of_masses_SoA.value<0>(cell_flat_index);
            X[1] = center_of_masses_SoA.value<1>(cell_flat_index);
            X[2] = center_of_masses_SoA.value<2>(cell_flat_index);
            std::array<m2m_vector, 4> tmpstore;
            // tmpstore[0] = potential_expansions_SoA.value<0>(cell_flat_index_unpadded);
            // tmpstore[1] = potential_expansions_SoA.value<1>(cell_flat_index_unpadded);
            // tmpstore[2] = potential_expansions_SoA.value<2>(cell_flat_index_unpadded);
            // tmpstore[3] = potential_expansions_SoA.value<3>(cell_flat_index_unpadded);
            std::array<m2m_vector, 3> tmp_corrections;
            // tmp_corrections[0] = angular_corrections_SoA.value<0>(cell_flat_index_unpadded);
            // tmp_corrections[1] = angular_corrections_SoA.value<1>(cell_flat_index_unpadded);
            // tmp_corrections[2] = angular_corrections_SoA.value<2>(cell_flat_index_unpadded);
            // tmp_corrections[3] = angular_corrections_SoA.value<3>(cell_flat_index_unpadded);
            // bool data_changed = false;

            m2m_int_vector theta_c_rec_squared_int = detail::distance_squared_reciprocal(
                cell_index_coarse, interaction_partner_index_coarse);

            m2m_vector theta_c_rec_squared =
                // Vc::static_datapar_cast<double>(theta_c_rec_squared_int);
                Vc::static_datapar_cast_double_to_int(theta_c_rec_squared_int);

            m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;

            if (Vc::none_of(mask)) {
                return;
            }
            // data_changed = true;

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

            std::array<m2m_vector, 17> m_partner;

            // Array to store the temporary result - was called A in the old style
            std::array<m2m_vector, 4> cur_pot;
            m_partner[0] = local_expansions_SoA.value<0>(interaction_partner_flat_index);
            cur_pot[0] = m_partner[0] * D_lower[0];
            cur_pot[1] = m_partner[0] * D_lower[1];
            cur_pot[2] = m_partner[0] * D_lower[2];
            cur_pot[3] = m_partner[0] * D_lower[3];

            m_partner[1] = local_expansions_SoA.value<4>(interaction_partner_flat_index);
            m_partner[2] = local_expansions_SoA.value<5>(interaction_partner_flat_index);
            cur_pot[0] += m_partner[1] * (D_lower[4] * factor_half_v[4]);
            cur_pot[1] += m_partner[1] * (D_lower[10] * factor_half_v[4]);
            cur_pot[2] += m_partner[1] * (D_lower[11] * factor_half_v[4]);
            cur_pot[3] += m_partner[1] * (D_lower[12] * factor_half_v[4]);

            cur_pot[0] += m_partner[2] * (D_lower[5] * factor_half_v[5]);
            cur_pot[1] += m_partner[2] * (D_lower[11] * factor_half_v[5]);
            cur_pot[2] += m_partner[2] * (D_lower[13] * factor_half_v[5]);
            cur_pot[3] += m_partner[2] * (D_lower[14] * factor_half_v[5]);

            m_partner[3] = local_expansions_SoA.value<6>(interaction_partner_flat_index);
            m_partner[4] = local_expansions_SoA.value<7>(interaction_partner_flat_index);
            cur_pot[0] += m_partner[3] * (D_lower[6] * factor_half_v[6]);
            cur_pot[1] += m_partner[3] * (D_lower[12] * factor_half_v[6]);
            cur_pot[2] += m_partner[3] * (D_lower[14] * factor_half_v[6]);
            cur_pot[3] += m_partner[3] * (D_lower[15] * factor_half_v[6]);

            cur_pot[0] += m_partner[4] * (D_lower[7] * factor_half_v[7]);
            cur_pot[1] += m_partner[4] * (D_lower[13] * factor_half_v[7]);
            cur_pot[2] += m_partner[4] * (D_lower[16] * factor_half_v[7]);
            cur_pot[3] += m_partner[4] * (D_lower[17] * factor_half_v[7]);

            m_partner[5] = local_expansions_SoA.value<8>(interaction_partner_flat_index);
            m_partner[6] = local_expansions_SoA.value<9>(interaction_partner_flat_index);
            cur_pot[0] += m_partner[5] * (D_lower[8] * factor_half_v[8]);
            cur_pot[1] += m_partner[5] * (D_lower[14] * factor_half_v[8]);
            cur_pot[2] += m_partner[5] * (D_lower[17] * factor_half_v[8]);
            cur_pot[3] += m_partner[5] * (D_lower[18] * factor_half_v[8]);

            cur_pot[0] += m_partner[6] * (D_lower[9] * factor_half_v[9]);
            cur_pot[1] += m_partner[6] * (D_lower[15] * factor_half_v[9]);
            cur_pot[2] += m_partner[6] * (D_lower[18] * factor_half_v[9]);
            cur_pot[3] += m_partner[6] * (D_lower[19] * factor_half_v[9]);

            m_partner[7] = local_expansions_SoA.value<10>(interaction_partner_flat_index);
            m_partner[8] = local_expansions_SoA.value<11>(interaction_partner_flat_index);
            cur_pot[0] -= m_partner[7] * (D_lower[10] * factor_sixth_v[10]);
            cur_pot[0] -= m_partner[8] * (D_lower[11] * factor_sixth_v[11]);
            m_partner[9] = local_expansions_SoA.value<12>(interaction_partner_flat_index);
            m_partner[10] = local_expansions_SoA.value<13>(interaction_partner_flat_index);
            cur_pot[0] -= m_partner[9] * (D_lower[12] * factor_sixth_v[12]);
            cur_pot[0] -= m_partner[10] * (D_lower[13] * factor_sixth_v[13]);
            m_partner[11] = local_expansions_SoA.value<14>(interaction_partner_flat_index);
            m_partner[12] = local_expansions_SoA.value<15>(interaction_partner_flat_index);
            cur_pot[0] -= m_partner[11] * (D_lower[14] * factor_sixth_v[14]);
            cur_pot[0] -= m_partner[12] * (D_lower[15] * factor_sixth_v[15]);
            m_partner[13] = local_expansions_SoA.value<16>(interaction_partner_flat_index);
            m_partner[14] = local_expansions_SoA.value<17>(interaction_partner_flat_index);
            cur_pot[0] -= m_partner[13] * (D_lower[16] * factor_sixth_v[16]);
            cur_pot[0] -= m_partner[14] * (D_lower[17] * factor_sixth_v[17]);
            m_partner[15] = local_expansions_SoA.value<18>(interaction_partner_flat_index);
            m_partner[16] = local_expansions_SoA.value<19>(interaction_partner_flat_index);
            cur_pot[0] -= m_partner[15] * (D_lower[18] * factor_sixth_v[18]);
            cur_pot[0] -= m_partner[16] * (D_lower[19] * factor_sixth_v[19]);

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

            current_angular_correction[0] = -m_partner[7] * (D_upper[0] * factor_sixth_v[10]);
            current_angular_correction[1] = -m_partner[7] * (D_upper[1] * factor_sixth_v[10]);
            current_angular_correction[2] = -m_partner[7] * (D_upper[2] * factor_sixth_v[10]);

            D_upper[3] = D_calculator.d2;
            m2m_vector d3_X11 = D_calculator.d3 * D_calculator.X_11;
            D_upper[3] += d3_X11;
            D_upper[3] += D_calculator.d3 * D_calculator.X_00;
            m2m_vector d3_X12 = D_calculator.d3 * D_calculator.X[1] * D_calculator.X[2];
            D_upper[4] = d3_X12;

            current_angular_correction[0] -= m_partner[8] * (D_upper[1] * factor_sixth_v[11]);
            current_angular_correction[1] -= m_partner[8] * (D_upper[3] * factor_sixth_v[11]);
            current_angular_correction[2] -= m_partner[8] * (D_upper[4] * factor_sixth_v[11]);

            D_upper[5] = D_calculator.d2;
            m2m_vector d3_X22 = D_calculator.d3 * D_calculator.X_22;
            D_upper[5] += d3_X22;
            D_upper[5] += d3_X00;

            current_angular_correction[0] -= m_partner[9] * (D_upper[2] * factor_sixth_v[12]);
            current_angular_correction[1] -= m_partner[9] * (D_upper[4] * factor_sixth_v[12]);
            current_angular_correction[2] -= m_partner[9] * (D_upper[5] * factor_sixth_v[12]);

            D_upper[6] = 3.0 * d3_X01;
            D_upper[7] = D_calculator.d3 * X_02;

            current_angular_correction[0] -= m_partner[10] * (D_upper[3] * factor_sixth_v[13]);
            current_angular_correction[1] -= m_partner[10] * (D_upper[6] * factor_sixth_v[13]);
            current_angular_correction[2] -= m_partner[10] * (D_upper[7] * factor_sixth_v[13]);

            D_upper[8] = D_calculator.d3 * D_calculator.X[0] * D_calculator.X[1];

            current_angular_correction[0] -= m_partner[11] * (D_upper[4] * factor_sixth_v[14]);
            current_angular_correction[1] -= m_partner[11] * (D_upper[7] * factor_sixth_v[14]);
            current_angular_correction[2] -= m_partner[11] * (D_upper[8] * factor_sixth_v[14]);

            D_upper[9] = 3.0 * d3_X02;

            current_angular_correction[0] -= m_partner[12] * (D_upper[5] * factor_sixth_v[15]);
            current_angular_correction[1] -= m_partner[12] * (D_upper[8] * factor_sixth_v[15]);
            current_angular_correction[2] -= m_partner[12] * (D_upper[9] * factor_sixth_v[15]);

            D_upper[10] =
                D_calculator.X[1] * D_calculator.X[1] * D_calculator.d3 + 2.0 * D_calculator.d2;
            D_upper[10] += D_calculator.d2;
            D_upper[10] += 5.0 * d3_X11;

            D_upper[11] = 3.0 * d3_X12;

            current_angular_correction[0] -= m_partner[13] * (D_upper[6] * factor_sixth_v[16]);
            current_angular_correction[1] -= m_partner[13] * (D_upper[10] * factor_sixth_v[16]);
            current_angular_correction[2] -= m_partner[13] * (D_upper[11] * factor_sixth_v[16]);

            D_upper[12] = D_calculator.d2;
            D_upper[12] += d3_X22;
            D_upper[12] += d3_X11;

            current_angular_correction[0] -= m_partner[14] * (D_upper[7] * factor_sixth_v[17]);
            current_angular_correction[1] -= m_partner[14] * (D_upper[11] * factor_sixth_v[17]);
            current_angular_correction[2] -= m_partner[14] * (D_upper[12] * factor_sixth_v[17]);

            D_upper[13] = 3.0 * d3_X12;

            current_angular_correction[0] -= m_partner[15] * (D_upper[8] * factor_sixth_v[18]);
            current_angular_correction[1] -= m_partner[15] * (D_upper[12] * factor_sixth_v[18]);
            current_angular_correction[2] -= m_partner[15] * (D_upper[13] * factor_sixth_v[18]);

            D_upper[14] =
                D_calculator.X[2] * D_calculator.X[2] * D_calculator.d3 + 2.0 * D_calculator.d2;
            D_upper[14] += D_calculator.d2;
            D_upper[14] += 5.0 * d3_X22;

            current_angular_correction[0] -= m_partner[16] * (D_upper[9] * factor_sixth_v[19]);
            current_angular_correction[1] -= m_partner[16] * (D_upper[13] * factor_sixth_v[19]);
            current_angular_correction[2] -= m_partner[16] * (D_upper[14] * factor_sixth_v[19]);

            Vc::where(mask, tmp_corrections[0]) =
                tmp_corrections[0] + current_angular_correction[0];
            Vc::where(mask, tmp_corrections[1]) =
                tmp_corrections[1] + current_angular_correction[1];
            Vc::where(mask, tmp_corrections[2]) =
                tmp_corrections[2] + current_angular_correction[2];
            // if (data_changed) {
            tmpstore[0] = tmpstore[0] + potential_expansions_SoA.value<0>(cell_flat_index_unpadded);
            tmpstore[1] = tmpstore[1] + potential_expansions_SoA.value<1>(cell_flat_index_unpadded);
            tmpstore[2] = tmpstore[2] + potential_expansions_SoA.value<2>(cell_flat_index_unpadded);
            tmpstore[3] = tmpstore[3] + potential_expansions_SoA.value<3>(cell_flat_index_unpadded);
            tmpstore[0].memstore(potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore[1].memstore(potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore[2].memstore(potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore[3].memstore(potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded),
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
            // }
        }

        void p2m_kernel::blocked_interaction_non_rho(
            struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>& local_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>& center_of_masses_SoA,
            struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>&
                potential_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>&
                angular_corrections_SoA,
            const multiindex<>& cell_index, const size_t cell_flat_index,
            const multiindex<m2m_int_vector>& cell_index_coarse,
            const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
            const multiindex<>& interaction_partner_index,
            const size_t interaction_partner_flat_index,
            multiindex<m2m_int_vector>& interaction_partner_index_coarse) {
            // TODO: should change name to something better (not taylor, but space_vector)
            // struct_of_array_taylor<space_vector, real, 3> X =
            //     center_of_masses_SoA.get_view(cell_flat_index);

            std::array<m2m_vector, NDIM> X;
            X[0] = center_of_masses_SoA.value<0>(cell_flat_index);
            X[1] = center_of_masses_SoA.value<1>(cell_flat_index);
            X[2] = center_of_masses_SoA.value<2>(cell_flat_index);
            std::array<m2m_vector, 4> tmpstore;
            // tmpstore[0] = potential_expansions_SoA.value<0>(cell_flat_index_unpadded);
            // tmpstore[1] = potential_expansions_SoA.value<1>(cell_flat_index_unpadded);
            // tmpstore[2] = potential_expansions_SoA.value<2>(cell_flat_index_unpadded);
            // tmpstore[3] = potential_expansions_SoA.value<3>(cell_flat_index_unpadded);

            m2m_int_vector theta_c_rec_squared_int = detail::distance_squared_reciprocal(
                cell_index_coarse, interaction_partner_index_coarse);

            m2m_vector theta_c_rec_squared =
                // Vc::static_datapar_cast<double>(theta_c_rec_squared_int);
                Vc::static_datapar_cast_double_to_int(theta_c_rec_squared_int);

            m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;

            if (Vc::none_of(mask)) {
                return;
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
            std::array<m2m_vector, 4> cur_pot;
            m_partner[0] = local_expansions_SoA.value<0>(interaction_partner_flat_index);
            cur_pot[0] = m_partner[0] * D_lower[0];
            cur_pot[1] = m_partner[0] * D_lower[1];
            cur_pot[2] = m_partner[0] * D_lower[2];
            cur_pot[3] = m_partner[0] * D_lower[3];

            m_partner[1] = local_expansions_SoA.value<1>(interaction_partner_flat_index);
            m_partner[2] = local_expansions_SoA.value<2>(interaction_partner_flat_index);
            m_partner[3] = local_expansions_SoA.value<3>(interaction_partner_flat_index);

            cur_pot[0] -= m_partner[1] * D_lower[1];
            cur_pot[0] -= m_partner[2] * D_lower[2];
            cur_pot[0] -= m_partner[3] * D_lower[3];

            cur_pot[1] -= m_partner[1] * D_lower[4];
            cur_pot[1] -= m_partner[1] * D_lower[5];
            cur_pot[1] -= m_partner[1] * D_lower[6];

            cur_pot[2] -= m_partner[2] * D_lower[5];
            cur_pot[2] -= m_partner[2] * D_lower[7];
            cur_pot[2] -= m_partner[2] * D_lower[8];

            cur_pot[3] -= m_partner[3] * D_lower[6];
            cur_pot[3] -= m_partner[3] * D_lower[8];
            cur_pot[3] -= m_partner[3] * D_lower[9];

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
            tmpstore[0] = tmpstore[0] + potential_expansions_SoA.value<0>(cell_flat_index_unpadded);
            tmpstore[1] = tmpstore[1] + potential_expansions_SoA.value<1>(cell_flat_index_unpadded);
            tmpstore[2] = tmpstore[2] + potential_expansions_SoA.value<2>(cell_flat_index_unpadded);
            tmpstore[3] = tmpstore[3] + potential_expansions_SoA.value<3>(cell_flat_index_unpadded);
            tmpstore[0].memstore(potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore[1].memstore(potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore[2].memstore(potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore[3].memstore(potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
        }
    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
