#ifdef OCTOTIGER_CUDA_ENABLED
#include "compute_kernel_templates.hpp"
#include "cuda_kernel_methods.hpp"
namespace octotiger {
namespace fmm {
    // This specialization is only required on cuda devices since T::value_type is not supported!
    template <>
    CUDA_CALLABLE_METHOD inline void multiindex<int32_t>::transform_coarse() {
        const int32_t patch_size = static_cast<int32_t>(INX);
        const int32_t subtract = static_cast<int32_t>(INX / 2);
        x = ((x + patch_size) >> 1) - subtract;
        y = ((y + patch_size) >> 1) - subtract;
        z = ((z + patch_size) >> 1) - subtract;
    }

    CUDA_CALLABLE_METHOD inline int32_t distance_squared_reciprocal(
        const multiindex<>& i, const multiindex<>& j) {
        return (sqr(i.x - j.x) + sqr(i.y - j.y) + sqr(i.z - j.z));
    }

    namespace multipole_interactions {

        constexpr size_t component_length = ENTRIES + SOA_PADDING;
        constexpr size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;
        __global__ void cuda_multipole_interactions_kernel(
            double (&local_monopoles)[NUMBER_LOCAL_MONOPOLE_VALUES],
            double (&center_of_masses)[NUMBER_MASS_VALUES],
            double (&multipoles)[NUMBER_LOCAL_EXPANSION_VALUES],
            double (&potential_expansions)[NUMBER_POT_EXPANSIONS],
            double (&angular_corrections)[NUMBER_ANG_CORRECTIONS],
            octotiger::fmm::multiindex<> (&stencil)[STENCIL_SIZE],
            double (&stencil_phases)[STENCIL_SIZE], double (&factor_half)[20],
            double (&factor_sixth)[20], double theta) {
            // printf("yay %f", theta);

            // Set cell indices
            const octotiger::fmm::multiindex<> cell_index(threadIdx.x + INNER_CELLS_PADDING_DEPTH,
                threadIdx.y + INNER_CELLS_PADDING_DEPTH, threadIdx.z + INNER_CELLS_PADDING_DEPTH);
            octotiger::fmm::multiindex<> cell_index_coarse(cell_index);
            cell_index_coarse.transform_coarse();
            const size_t cell_flat_index = octotiger::fmm::to_flat_index_padded(cell_index);
            octotiger::fmm::multiindex<> cell_index_unpadded(threadIdx.x, threadIdx.y, threadIdx.z);
            const size_t cell_flat_index_unpadded =
                octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded);

            double factor_half_f[20];
            double factor_sixth_f[20];
            for (auto i = 0; i < 20; ++i) {
              factor_half_f[i] = factor_half[i];
              factor_sixth_f[i] = factor_sixth[i];
            }

            // Load multipoles for this cell
            double m_cell[20];
            m_cell[0] = multipoles[0 * component_length + cell_flat_index];
            m_cell[1] = multipoles[1 * component_length + cell_flat_index];
            m_cell[2] = multipoles[2 * component_length + cell_flat_index];
            m_cell[3] = multipoles[3 * component_length + cell_flat_index];
            m_cell[4] = multipoles[4 * component_length + cell_flat_index];
            m_cell[5] = multipoles[5 * component_length + cell_flat_index];
            m_cell[6] = multipoles[6 * component_length + cell_flat_index];
            m_cell[7] = multipoles[7 * component_length + cell_flat_index];
            m_cell[8] = multipoles[8 * component_length + cell_flat_index];
            m_cell[9] = multipoles[9 * component_length + cell_flat_index];
            m_cell[10] = multipoles[10 * component_length + cell_flat_index];
            m_cell[11] = multipoles[11 * component_length + cell_flat_index];
            m_cell[12] = multipoles[12 * component_length + cell_flat_index];
            m_cell[13] = multipoles[13 * component_length + cell_flat_index];
            m_cell[14] = multipoles[14 * component_length + cell_flat_index];
            m_cell[15] = multipoles[15 * component_length + cell_flat_index];
            m_cell[16] = multipoles[16 * component_length + cell_flat_index];
            m_cell[17] = multipoles[17 * component_length + cell_flat_index];
            m_cell[18] = multipoles[18 * component_length + cell_flat_index];
            m_cell[19] = multipoles[19 * component_length + cell_flat_index];
            double X[NDIM];
            X[0] = center_of_masses[cell_flat_index];
            X[1] = center_of_masses[1 * component_length + cell_flat_index];
            X[2] = center_of_masses[2 * component_length + cell_flat_index];

            // for (auto i = 0; i < component_length; ++i) {
            //   for (auto com = 0; com < 20; ++com) {
            //     printf ("%.2e ", multipoles[component_length*com+i]);
            //   }

            //     printf ("\n");
            // }


            // Create and set result arrays
            double tmpstore[20];
            double tmp_corrections[3];
            for (size_t i = 0; i < 20; ++i)
                tmpstore[i] = 0.0;
            for (size_t i = 0; i < 3; ++i)
                tmp_corrections[i] = 0.0;
            // Required for mask
            const double theta_rec_squared = sqr(1.0 / theta);
            double m_partner[20];
            double Y[NDIM];

            // calculate interactions between this cell and each stencil element
            for (size_t stencil_index = 0; stencil_index < STENCIL_SIZE; stencil_index++) {
                // Get phase indicator (indicates whether multipole multipole interactions still
                // needs to be done)
                const double mask_phase_one = stencil_phases[stencil_index];

                // Get interaction partner indices
                const multiindex<>& stencil_element = stencil[stencil_index];
                const multiindex<> partner_index(cell_index.x + stencil_element.x,
                    cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);
                const size_t partner_flat_index = to_flat_index_padded(partner_index);
                multiindex<> partner_index_coarse(partner_index);
                partner_index_coarse.transform_coarse();

                // Create mask - TODO is this really necessay in the non-vectorized code..?
                const double theta_c_rec_squared = static_cast<double>(
                    distance_squared_reciprocal(cell_index_coarse, partner_index_coarse));
                const bool mask_b = theta_rec_squared > theta_c_rec_squared;
                double mask = mask_b ? 1.0 : 0.0;

                // Load data of interaction partner
                Y[0] = center_of_masses[partner_flat_index];
                Y[1] = center_of_masses[1 * component_length + partner_flat_index];
                Y[2] = center_of_masses[2 * component_length + partner_flat_index];
                m_partner[0] = local_monopoles[partner_flat_index] * mask;
                mask = mask * mask_phase_one;    // do not load multipoles outside the inner stencil
                m_partner[0] += multipoles[partner_flat_index] * mask;
                m_partner[1] = multipoles[1 * component_length + partner_flat_index] * mask;
                m_partner[2] = multipoles[2 * component_length + partner_flat_index] * mask;
                m_partner[3] = multipoles[3 * component_length + partner_flat_index] * mask;
                // printf("%d ", m_partner[3]);
                m_partner[4] = multipoles[4 * component_length + partner_flat_index] * mask;
                m_partner[5] = multipoles[5 * component_length + partner_flat_index] * mask;
                m_partner[6] = multipoles[6 * component_length + partner_flat_index] * mask;
                m_partner[7] = multipoles[7 * component_length + partner_flat_index] * mask;
                m_partner[8] = multipoles[8 * component_length + partner_flat_index] * mask;
                m_partner[9] = multipoles[9 * component_length + partner_flat_index] * mask;
                m_partner[10] = multipoles[10 * component_length + partner_flat_index] * mask;
                m_partner[11] = multipoles[11 * component_length + partner_flat_index] * mask;
                m_partner[12] = multipoles[12 * component_length + partner_flat_index] * mask;
                m_partner[13] = multipoles[13 * component_length + partner_flat_index] * mask;
                m_partner[14] = multipoles[14 * component_length + partner_flat_index] * mask;
                m_partner[15] = multipoles[15 * component_length + partner_flat_index] * mask;
                m_partner[16] = multipoles[16 * component_length + partner_flat_index] * mask;
                m_partner[17] = multipoles[17 * component_length + partner_flat_index] * mask;
                m_partner[18] = multipoles[18 * component_length + partner_flat_index] * mask;
                m_partner[19] = multipoles[19 * component_length + partner_flat_index] * mask;
                // printf("%f ", multipoles[0 * component_length + partner_flat_index]);
                // printf("%f ", multipoles[1 * component_length + partner_flat_index]);
                // printf("%f ", multipoles[2 * component_length + partner_flat_index]);
                // printf("%f ", multipoles[3 * component_length + partner_flat_index]);
                // printf("%f ", multipoles[4 * component_length + partner_flat_index]);
                // printf("%f ", multipoles[5 * component_length + partner_flat_index]);
                // printf("%f ", multipoles[6 * component_length + partner_flat_index]);
                // printf("%f ", multipoles[7 * component_length + partner_flat_index]);
                // printf("%f ", multipoles[8 * component_length + partner_flat_index]);
                // printf("%f ", multipoles[9 * component_length + partner_flat_index]);
                // printf("%f ", multipoles[10 * component_length + partner_flat_index]);
                // printf("%f ", multipoles[11 * component_length + partner_flat_index]);
                // printf("%f ", multipoles[12 * component_length + partner_flat_index]);
                // printf("%f ", multipoles[13 * component_length + partner_flat_index]);
                // printf("%f ", multipoles[14 * component_length + partner_flat_index]);
                // printf("%f ", multipoles[15 * component_length + partner_flat_index]);
                // printf("%f ", multipoles[16 * component_length + partner_flat_index]);
                // printf("%f ", multipoles[17 * component_length + partner_flat_index]);
                // printf("\n");
                // for (auto i = 0; i < 20; ++i) {
                //   tmpstore[i] += m_partner[i] * factor_sixth[9];
                // }





                // Do the actual calculations
                compute_kernel_rho<double>(
                    X, Y, m_partner, tmpstore, tmp_corrections, m_cell, factor_half_f, factor_sixth_f);
            // double dX[NDIM];
            // dX[0] = X[0] - Y[0];
            // dX[1] = X[1] - Y[1];
            // dX[2] = X[2] - Y[2];
            // double X_00, X_11, X_22;
            // double d0, d1, d2, d3;
            // X_00 = dX[0] * dX[0];
            // X_11 = dX[1] * dX[1];
            // X_22 = dX[2] * dX[2];

            // // const double r2 = X_00 + X_11 + X_22;
            // // const double r2inv = 1.0 / std::max(r2, 1.0e-20);
            // double r2 = X_00 + X_11 + X_22;
            // if (r2 < 1.0e-20) {
            //   printf("ho");
            //   r2 = 1.0e20;
            // }
            // double r2inv = 1.0 / r2;

            // d0 = -sqrt(r2inv);
            // d1 = -d0 * r2inv;
            // d2 = -3.0 * d1 * r2inv;
            // d3 = -5.0 * d2 * r2inv;

            // double D_lower[20];

            // D_lower[0] = d0;

            // D_lower[1] = dX[0] * d1;
            // D_lower[2] = dX[1] * d1;
            // D_lower[3] = dX[2] * d1;

            // const double X_12 = dX[1] * dX[2];
            // const double X_01 = dX[0] * dX[1];
            // const double X_02 = dX[0] * dX[2];

            // D_lower[4] = d2 * X_00;
            // D_lower[4] += d1;
            // D_lower[5] = d2 * X_01;
            // D_lower[6] = d2 * X_02;

            // D_lower[7] = d2 * X_11;
            // D_lower[7] += d1;
            // D_lower[8] = d2 * X_12;

            // D_lower[9] = d2 * X_22;
            // D_lower[9] += d1;

            // D_lower[10] = d3 * X_00 * dX[0];
            // const double d2_X0 = d2 * dX[0];
            // D_lower[10] += 3.0 * d2_X0;
            // D_lower[11] = d3 * X_00 * dX[1];
            // D_lower[11] += d2 * dX[1];
            // D_lower[12] = d3 * X_00 * dX[2];
            // D_lower[12] += d2 * dX[2];

            // D_lower[13] = d3 * dX[0] * X_11;
            // D_lower[13] += d2 * dX[0];
            // D_lower[14] = d3 * dX[0] * X_12;

            // D_lower[15] = d3 * dX[0] * X_22;
            // D_lower[15] += d2_X0;

            // D_lower[16] = d3 * X_11 * dX[1];
            // const double d2_X1 = d2 * dX[1];
            // D_lower[16] += 3.0 * d2_X1;

            // D_lower[17] = d3 * X_11 * dX[2];
            // D_lower[17] += d2 * dX[2];

            // D_lower[18] = d3 * dX[1] * X_22;
            // D_lower[18] += d2 * dX[1];

            // D_lower[19] = d3 * X_22 * dX[2];
            // const double d2_X2 = d2 * dX[2];
            // D_lower[19] += 3.0 * d2_X2;

            // double cur_pot[10];

            // cur_pot[0] = m_partner[0] * D_lower[0];
            // cur_pot[1] = m_partner[0] * D_lower[1];
            // cur_pot[2] = m_partner[0] * D_lower[2];
            // cur_pot[3] = m_partner[0] * D_lower[3];

            // cur_pot[0] += m_partner[4] * (D_lower[4] * factor_half[4]);
            // cur_pot[1] += m_partner[4] * (D_lower[10] * factor_half[4]);
            // cur_pot[2] += m_partner[4] * (D_lower[11] * factor_half[4]);
            // cur_pot[3] += m_partner[4] * (D_lower[12] * factor_half[4]);

            // cur_pot[0] += m_partner[5] * (D_lower[5] * factor_half[5]);
            // cur_pot[1] += m_partner[5] * (D_lower[11] * factor_half[5]);
            // cur_pot[2] += m_partner[5] * (D_lower[13] * factor_half[5]);
            // cur_pot[3] += m_partner[5] * (D_lower[14] * factor_half[5]);

            // cur_pot[0] += m_partner[6] * (D_lower[6] * factor_half[6]);
            // cur_pot[1] += m_partner[6] * (D_lower[12] * factor_half[6]);
            // cur_pot[2] += m_partner[6] * (D_lower[14] * factor_half[6]);
            // cur_pot[3] += m_partner[6] * (D_lower[15] * factor_half[6]);

            // cur_pot[0] += m_partner[7] * (D_lower[7] * factor_half[7]);
            // cur_pot[1] += m_partner[7] * (D_lower[13] * factor_half[7]);
            // cur_pot[2] += m_partner[7] * (D_lower[16] * factor_half[7]);
            // cur_pot[3] += m_partner[7] * (D_lower[17] * factor_half[7]);

            // cur_pot[0] += m_partner[8] * (D_lower[8] * factor_half[8]);
            // cur_pot[1] += m_partner[8] * (D_lower[14] * factor_half[8]);
            // cur_pot[2] += m_partner[8] * (D_lower[17] * factor_half[8]);
            // cur_pot[3] += m_partner[8] * (D_lower[18] * factor_half[8]);

            // cur_pot[0] += m_partner[9] * (D_lower[9] * factor_half[9]);
            // cur_pot[1] += m_partner[9] * (D_lower[15] * factor_half[9]);
            // cur_pot[2] += m_partner[9] * (D_lower[18] * factor_half[9]);
            // cur_pot[3] += m_partner[9] * (D_lower[19] * factor_half[9]);

            // cur_pot[0] -= m_partner[10] * (D_lower[10] * factor_sixth[10]);
            // cur_pot[0] -= m_partner[11] * (D_lower[11] * factor_sixth[11]);
            // cur_pot[0] -= m_partner[12] * (D_lower[12] * factor_sixth[12]);
            // cur_pot[0] -= m_partner[13] * (D_lower[13] * factor_sixth[13]);
            // cur_pot[0] -= m_partner[14] * (D_lower[14] * factor_sixth[14]);
            // cur_pot[0] -= m_partner[15] * (D_lower[15] * factor_sixth[15]);
            // cur_pot[0] -= m_partner[16] * (D_lower[16] * factor_sixth[16]);
            // cur_pot[0] -= m_partner[17] * (D_lower[17] * factor_sixth[17]);
            // cur_pot[0] -= m_partner[18] * (D_lower[18] * factor_sixth[18]);
            // cur_pot[0] -= m_partner[19] * (D_lower[19] * factor_sixth[19]);

            // cur_pot[4] = m_partner[0] * D_lower[4];
            // cur_pot[5] = m_partner[0] * D_lower[5];
            // cur_pot[6] = m_partner[0] * D_lower[6];
            // cur_pot[7] = m_partner[0] * D_lower[7];
            // cur_pot[8] = m_partner[0] * D_lower[8];
            // cur_pot[9] = m_partner[0] * D_lower[9];

            // cur_pot[4] -= m_partner[1] * D_lower[10];
            // cur_pot[5] -= m_partner[1] * D_lower[11];
            // cur_pot[6] -= m_partner[1] * D_lower[12];
            // cur_pot[7] -= m_partner[1] * D_lower[13];
            // cur_pot[8] -= m_partner[1] * D_lower[14];
            // cur_pot[9] -= m_partner[1] * D_lower[15];

            // cur_pot[4] -= m_partner[2] * D_lower[11];
            // cur_pot[5] -= m_partner[2] * D_lower[13];
            // cur_pot[6] -= m_partner[2] * D_lower[14];
            // cur_pot[7] -= m_partner[2] * D_lower[16];
            // cur_pot[8] -= m_partner[2] * D_lower[17];
            // cur_pot[9] -= m_partner[2] * D_lower[18];

            // cur_pot[4] -= m_partner[3] * D_lower[12];
            // cur_pot[5] -= m_partner[3] * D_lower[14];
            // cur_pot[6] -= m_partner[3] * D_lower[15];
            // cur_pot[7] -= m_partner[3] * D_lower[17];
            // cur_pot[8] -= m_partner[3] * D_lower[18];
            // cur_pot[9] -= m_partner[3] * D_lower[19];
            //     tmpstore[0] = tmpstore[0] + cur_pot[0];
            //     tmpstore[1] = tmpstore[1] + cur_pot[1];
            //     tmpstore[2] = tmpstore[2] + cur_pot[2];
            //     tmpstore[3] = tmpstore[3] + cur_pot[3];
            //     tmpstore[4] = tmpstore[4] + cur_pot[4];
            //     tmpstore[5] = tmpstore[5] + cur_pot[5];
            //     tmpstore[6] = tmpstore[6] + cur_pot[6];
            //     tmpstore[7] = tmpstore[7] + cur_pot[7];
            //     tmpstore[8] = tmpstore[8] + cur_pot[8];
            //     tmpstore[9] = tmpstore[9] + cur_pot[9];

            //     /* Maps to
            //     for (integer i = taylor_sizes[2]; i < taylor_sizes[3]; ++i) {
            //         A0[i] = m0[0] * D[i];
            //     }*/
            //     tmpstore[10] = tmpstore[10] + m_partner[0] * D_lower[10];
            //     tmpstore[11] = tmpstore[11] + m_partner[0] * D_lower[11];
            //     tmpstore[12] = tmpstore[12] + m_partner[0] * D_lower[12];
            //     tmpstore[13] = tmpstore[13] + m_partner[0] * D_lower[13];
            //     tmpstore[14] = tmpstore[14] + m_partner[0] * D_lower[14];
            //     tmpstore[15] = tmpstore[15] + m_partner[0] * D_lower[15];
            //     tmpstore[16] = tmpstore[16] + m_partner[0] * D_lower[16];
            //     tmpstore[17] = tmpstore[17] + m_partner[0] * D_lower[17];
            //     tmpstore[18] = tmpstore[18] + m_partner[0] * D_lower[18];
            //     tmpstore[19] = tmpstore[19] + m_partner[0] * D_lower[19];

            // tmpstore[0] = tmpstore[0] + D_lower[0];
            // tmpstore[1] = tmpstore[1] + D_lower[1];
            // tmpstore[2] = tmpstore[2] + D_lower[2];
            // tmpstore[3] = tmpstore[3] + D_lower[3];
            // tmpstore[4] = tmpstore[4] + r2;
            // tmpstore[5] = tmpstore[5] + r2inv;
            // tmpstore[6] = tmpstore[6] + dX[0];
            // tmpstore[7] = tmpstore[7] + dX[1];
            // tmpstore[8] = tmpstore[8] + dX[2];
            // tmpstore[9] = tmpstore[9] + d0;

            /* Maps to
            for (integer i = taylor_sizes[2]; i < taylor_sizes[3]; ++i) {
                A0[i] = m0[0] * D[i];
            }*/
            // tmpstore[10] = tmpstore[10] + d1;
            // tmpstore[11] = tmpstore[11] + d2;
            // tmpstore[12] = tmpstore[12] + d3;
            // tmpstore[13] = 0;
            // tmpstore[14] = 0;
            // tmpstore[15] = 0;
            // tmpstore[16] = 0;
            // tmpstore[17] = 0;
            // tmpstore[18] = 0;
            // tmpstore[19] = 0;

            }
            // Store results in output arrays
            potential_expansions[cell_flat_index_unpadded] = tmpstore[0];
            potential_expansions[1 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[1];
            potential_expansions[2 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[2];
            potential_expansions[3 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[3];
            potential_expansions[4 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[4];
            // printf("%d ", tmpstore[3]);
            potential_expansions[5 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[5];
            potential_expansions[6 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[6];
            potential_expansions[7 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[7];
            potential_expansions[8 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[8];
            potential_expansions[9 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[9];
            potential_expansions[10 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[10];
            potential_expansions[11 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[11];
            potential_expansions[12 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[12];
            potential_expansions[13 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[13];
            potential_expansions[14 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[14];
            potential_expansions[15 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[15];
            potential_expansions[16 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[16];
            potential_expansions[17 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[17];
            potential_expansions[18 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[18];
            potential_expansions[19 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[19];

            angular_corrections[cell_flat_index_unpadded] = tmp_corrections[0];
            angular_corrections[1 * component_length_unpadded + cell_flat_index_unpadded] =
            tmp_corrections[1];
            angular_corrections[2 * component_length_unpadded + cell_flat_index_unpadded] =
            tmp_corrections[2];
        }
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
