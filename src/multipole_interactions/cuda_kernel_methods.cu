#ifdef OCTOTIGER_CUDA_ENABLED
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
        constexpr size_t padded_entries_per_component = ENTRIES + SOA_PADDING;
        __global__ void cuda_multipole_interactions_kernel(double* local_monopoles,
            double* center_of_masses, double* multipoles, double* potential_expansions,
            double* angular_corrections, octotiger::fmm::multiindex<>* stencil,
            bool* stencil_phases, double* factor_half, double* factor_sixth, double theta) {
            printf("yay %f", theta);

            // Set cell indices
            const octotiger::fmm::multiindex<> cell_index(threadIdx.x + INNER_CELLS_PADDING_DEPTH,
                threadIdx.y + INNER_CELLS_PADDING_DEPTH, threadIdx.z + INNER_CELLS_PADDING_DEPTH);
            octotiger::fmm::multiindex<> cell_index_coarse(cell_index);
            cell_index_coarse.transform_coarse();
            const size_t cell_flat_index = octotiger::fmm::to_flat_index_padded(cell_index);
            octotiger::fmm::multiindex<> cell_index_unpadded(threadIdx.x, threadIdx.y, threadIdx.z);
            const size_t cell_flat_index_unpadded =
                octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded);

            // Create and set result arrays
            double tmpstore[20];
            double tmp_corrections[20];
            for (size_t i = 0; i < 20; ++i) {
                tmpstore[i] = 0.0;
                tmp_corrections[i] = 0.0;
            }
            // Required for mask
            const double theta_rec_squared = sqr(1.0 / theta);

            // calculate interactions between this cell and each stencil element
            for (size_t stencil_index = 0; stencil_index < 743; stencil_index++) {
                // Get phase indicator (indicates whether multipole multipole interactions still
                // needs to be done)
                const double mask_phase_one = static_cast<double>(stencil_phases[stencil_index]);

                // Get interaction partner indices
                const multiindex<>& stencil_element = stencil[stencil_index];
                const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                    cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);
                const size_t interaction_partner_flat_index =
                    to_flat_index_padded(interaction_partner_index);
                multiindex<> interaction_partner_index_coarse(interaction_partner_index);
                interaction_partner_index_coarse.transform_coarse();

                // Create mask - TODO is this really necessay in the non-vectorized code..?
                const double theta_c_rec_squared = static_cast<double>(distance_squared_reciprocal(
                    cell_index_coarse, interaction_partner_index_coarse));
                const bool mask_b = theta_rec_squared > theta_c_rec_squared;
                double mask = mask_b ? 1.0 : 0.0;

                double Y[NDIM];
                Y[0] = center_of_masses[interaction_partner_flat_index];
                Y[1] = center_of_masses[1 * padded_entries_per_component +
                    interaction_partner_flat_index];
                Y[2] = center_of_masses[2 * padded_entries_per_component +
                    interaction_partner_flat_index];

                double m_partner[20];
                m_partner[0] = local_monopoles[interaction_partner_flat_index] * mask;
                mask = mask * mask_phase_one;    // do not load multipoles outside the inner stencil
                m_partner[0] += multipoles[interaction_partner_flat_index] * mask;
                m_partner[1] =
                    multipoles[1 * padded_entries_per_component + interaction_partner_flat_index] *
                    mask;
                m_partner[2] =
                    multipoles[2 * padded_entries_per_component + interaction_partner_flat_index] *
                    mask;
                m_partner[3] =
                    multipoles[3 * padded_entries_per_component + interaction_partner_flat_index] *
                    mask;
                m_partner[4] =
                    multipoles[4 * padded_entries_per_component + interaction_partner_flat_index] *
                    mask;
                m_partner[5] =
                    multipoles[5 * padded_entries_per_component + interaction_partner_flat_index] *
                    mask;
                m_partner[6] =
                    multipoles[6 * padded_entries_per_component + interaction_partner_flat_index] *
                    mask;
                m_partner[7] =
                    multipoles[7 * padded_entries_per_component + interaction_partner_flat_index] *
                    mask;
                m_partner[8] =
                    multipoles[8 * padded_entries_per_component + interaction_partner_flat_index] *
                    mask;
                m_partner[9] =
                    multipoles[9 * padded_entries_per_component + interaction_partner_flat_index] *
                    mask;
                m_partner[10] =
                    multipoles[10 * padded_entries_per_component + interaction_partner_flat_index] *
                    mask;
                m_partner[11] =
                    multipoles[11 * padded_entries_per_component + interaction_partner_flat_index] *
                    mask;
                m_partner[12] =
                    multipoles[12 * padded_entries_per_component + interaction_partner_flat_index] *
                    mask;
                m_partner[13] =
                    multipoles[13 * padded_entries_per_component + interaction_partner_flat_index] *
                    mask;
                m_partner[14] =
                    multipoles[14 * padded_entries_per_component + interaction_partner_flat_index] *
                    mask;
                m_partner[15] =
                    multipoles[15 * padded_entries_per_component + interaction_partner_flat_index] *
                    mask;
                m_partner[16] =
                    multipoles[16 * padded_entries_per_component + interaction_partner_flat_index] *
                    mask;
                m_partner[17] =
                    multipoles[17 * padded_entries_per_component + interaction_partner_flat_index] *
                    mask;
                m_partner[18] =
                    multipoles[18 * padded_entries_per_component + interaction_partner_flat_index] *
                    mask;
                m_partner[19] =
                    multipoles[19 * padded_entries_per_component + interaction_partner_flat_index] *
                    mask;
            }
        }
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
