#ifdef OCTOTIGER_WITH_CUDA
#include "monopole_kernel_templates.hpp"
#include "p2p_cuda_kernel.hpp"
namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        __device__ HPX_CONSTEXPR size_t component_length = ENTRIES + SOA_PADDING;
        __device__ HPX_CONSTEXPR size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;

        __global__ void
        __launch_bounds__(512, 3)
        cuda_p2p_interactions_kernel(
            const double (&local_monopoles)[NUMBER_LOCAL_MONOPOLE_VALUES],
            double (&potential_expansions)[3 * NUMBER_POT_EXPANSIONS_SMALL],
            const octotiger::fmm::multiindex<> (&stencil)[STENCIL_SIZE],
            const double (&four_constants)[4 * STENCIL_SIZE], const double theta, const double dx) {
            // Set cell indices
            const octotiger::fmm::multiindex<> cell_index(threadIdx.x + INNER_CELLS_PADDING_DEPTH,
                threadIdx.y + INNER_CELLS_PADDING_DEPTH, threadIdx.z + INNER_CELLS_PADDING_DEPTH);
            octotiger::fmm::multiindex<> cell_index_coarse(cell_index);
            cell_index_coarse.transform_coarse();
            const size_t cell_flat_index = octotiger::fmm::to_flat_index_padded(cell_index);
            octotiger::fmm::multiindex<> cell_index_unpadded(threadIdx.x, threadIdx.y, threadIdx.z);
            const size_t cell_flat_index_unpadded =
                octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded);

            // Required for mask
            const double theta_rec_squared = sqr(1.0 / theta);
            const double d_components[2] = {1.0 / dx, -1.0 / sqr(dx)};
            double tmpstore[4];
            tmpstore[0] = 0.0;
            tmpstore[1] = 0.0;
            tmpstore[2] = 0.0;
            tmpstore[3] = 0.0;

            const size_t block_offset = blockIdx.x * NUMBER_POT_EXPANSIONS_SMALL;
            const size_t block_start = blockIdx.x * 358;
            const size_t block_end = 358 + blockIdx.x * 358;

            // calculate interactions between this cell and each stencil element
            for (size_t stencil_index = block_start; stencil_index < block_end;
                 stencil_index++) {
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
                double monopole = local_monopoles[partner_flat_index] * mask;

                const double four[4] = {four_constants[stencil_index * 4 + 0],
                    four_constants[stencil_index * 4 + 1], four_constants[stencil_index * 4 + 2],
                    four_constants[stencil_index * 4 + 3]};

                // Do the actual calculations
                compute_monopole_interaction<double>(monopole, tmpstore, four, d_components);
            }
            // Store results in output arrays
            potential_expansions[block_offset + cell_flat_index_unpadded] = tmpstore[0];
            potential_expansions[block_offset + 1 * component_length_unpadded +
                cell_flat_index_unpadded] = tmpstore[1];
            potential_expansions[block_offset + 2 * component_length_unpadded +
                cell_flat_index_unpadded] = tmpstore[2];
            potential_expansions[block_offset + 3 * component_length_unpadded +
                cell_flat_index_unpadded] = tmpstore[3];
        }
        __global__ void cuda_add_pot_blocks(
            double (&potential_expansions)[3 * NUMBER_POT_EXPANSIONS_SMALL]) {
            size_t id = threadIdx.x;
            potential_expansions[id] += potential_expansions[NUMBER_POT_EXPANSIONS_SMALL + id];
            potential_expansions[1 * component_length_unpadded + id] +=
                potential_expansions[1 * component_length_unpadded + NUMBER_POT_EXPANSIONS_SMALL +
                    id];
            potential_expansions[2 * component_length_unpadded + id] +=
                potential_expansions[2 * component_length_unpadded + NUMBER_POT_EXPANSIONS_SMALL +
                    id];
            potential_expansions[3 * component_length_unpadded + id] +=
                potential_expansions[3 * component_length_unpadded + NUMBER_POT_EXPANSIONS_SMALL +
                    id];
            potential_expansions[id] += potential_expansions[2 * NUMBER_POT_EXPANSIONS_SMALL + id];
            potential_expansions[1 * component_length_unpadded + id] +=
                potential_expansions[1 * component_length_unpadded + 2 * NUMBER_POT_EXPANSIONS_SMALL +
                    id];
            potential_expansions[2 * component_length_unpadded + id] +=
                potential_expansions[2 * component_length_unpadded + 2 * NUMBER_POT_EXPANSIONS_SMALL +
                    id];
            potential_expansions[3 * component_length_unpadded + id] +=
                potential_expansions[3 * component_length_unpadded + 2 * NUMBER_POT_EXPANSIONS_SMALL +
                    id];
        }
    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
