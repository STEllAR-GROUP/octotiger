#ifdef OCTOTIGER_HAVE_CUDA
#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/monopole_interactions/monopole_kernel_templates.hpp"
#include "octotiger/monopole_interactions/p2p_cuda_kernel.hpp"

namespace octotiger {
    namespace fmm {
        namespace monopole_interactions {
            // __constant__ octotiger::fmm::multiindex<> device_stencil_const[P2P_PADDED_STENCIL_SIZE];
            __device__ __constant__ bool device_stencil_masks[FULL_STENCIL_SIZE];
            // __device__ __constant__ double device_four_constants[FULL_STENCIL_SIZE * 4];

            //__device__ const size_t component_length = ENTRIES + SOA_PADDING;
            __device__ const size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;
            __device__ const size_t cache_line_length = INX + 2 * STENCIL_MAX;
            __device__ const size_t cache_offset = INX + STENCIL_MIN;

            __global__ void
            __launch_bounds__(INX * INX, 1)
                cuda_p2p_interactions_kernel(
                    const double (&local_monopoles)[NUMBER_LOCAL_MONOPOLE_VALUES],
                    double (&potential_expansions)[NUMBER_POT_EXPANSIONS_SMALL],
                    const double theta, const double dx) {

                // Declare shared memory arrays/index
                // Each holds two slices
                // __shared__ double monopole_cache[2 * cache_line_length * cache_line_length];
                // __shared__ multiindex<> coarse_index_cache[2 * cache_line_length * cache_line_length];
                // int local_id = threadIdx.y * INX + threadIdx.z;

                // use in case of debug prints
                //bool first_thread = (blockIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0);
                // Set cell indices
                const octotiger::fmm::multiindex<> cell_index((threadIdx.x + blockIdx.x * 2) + INNER_CELLS_PADDING_DEPTH,
                                                              threadIdx.y + INNER_CELLS_PADDING_DEPTH,
                                                              threadIdx.z + INNER_CELLS_PADDING_DEPTH);
                octotiger::fmm::multiindex<> cell_index_coarse(cell_index);
                cell_index_coarse.transform_coarse();
                const size_t cell_flat_index = octotiger::fmm::to_flat_index_padded(cell_index);
                octotiger::fmm::multiindex<> cell_index_unpadded((threadIdx.x + blockIdx.x * 2), threadIdx.y, threadIdx.z);
                const size_t cell_flat_index_unpadded =
                    octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded);


                const octotiger::fmm::multiindex<> cell_index2((threadIdx.x + blockIdx.x * 2 + 1) + INNER_CELLS_PADDING_DEPTH,
                                                               threadIdx.y + INNER_CELLS_PADDING_DEPTH,
                                                               threadIdx.z + INNER_CELLS_PADDING_DEPTH);
                octotiger::fmm::multiindex<> cell_index_coarse2(cell_index2);
                cell_index_coarse2.transform_coarse();
                const size_t cell_flat_index2 = octotiger::fmm::to_flat_index_padded(cell_index2);
                octotiger::fmm::multiindex<> cell_index_unpadded2((threadIdx.x + blockIdx.x * 2 + 1), threadIdx.y, threadIdx.z);
                const size_t cell_flat_index_unpadded2 =
                    octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded2);

                // Calculate required constants
                const double theta_rec_squared = sqr(1.0 / theta);
                const double d_components[2] = {1.0 / dx, -1.0 / dx};
                double tmpstore[4] = {0.0, 0.0, 0.0, 0.0};
                double tmpstore2[4] = {0.0, 0.0, 0.0, 0.0};
                const size_t index_base = (threadIdx.y + STENCIL_MAX) * cache_line_length + threadIdx.z + STENCIL_MAX;
                const size_t index_base2 = (threadIdx.y + STENCIL_MAX) * cache_line_length + threadIdx.z + STENCIL_MAX + cache_line_length * cache_line_length;

                // int load_offset = 0;
                // int load_id = local_id; // which id should be loaded during the manual caching
                // if (local_id >= cache_line_length) {
                //     load_offset = 1; // will load a different line
                //     load_id = load_id - cache_line_length; // naturally the line starts at a new beginning
                // }

                // Go through all possible stance elements for the two cells this thread is responsible for
                for (int stencil_x = STENCIL_MIN; stencil_x <= STENCIL_MAX; stencil_x++) {
                    int x = stencil_x - STENCIL_MIN;

                
                    // // Phase 1 Manual caching - minizes global memory accesses and aligns them
                    // __syncthreads();
                    // if (local_id < cache_line_length * 2) {
                    //     // we cache over two slides of our cube
                    //     for (int x = 0; x < 2; x++) {
                    //         // we always cache two lines
                    //         for (int i = 0; i < cache_line_length / 2; i++) {
                    //             const multiindex<> partner_index(INNER_CELLS_PADDING_DEPTH + blockIdx.x * 2
                    //                                              + stencil_x + x,
                    //                                              2*i + load_offset + cache_offset,
                    //                                              cache_offset + load_id);
                    //             const size_t partner_flat_index = to_flat_index_padded(partner_index);
                    //             multiindex<> partner_index_coarse(partner_index);
                    //             partner_index_coarse.transform_coarse();
                    //             coarse_index_cache[
                    //                 cache_line_length*(2*i + load_offset + x * cache_line_length) + load_id] =
                    //                 partner_index_coarse;
                    //             monopole_cache[
                    //                 cache_line_length*(2*i + load_offset + x * cache_line_length) + load_id] =
                    //                 local_monopoles[partner_flat_index];
                    //         }
                    //     }
                    // }
                    // // Phase 2: Actual calculations for the cached content
                    // __syncthreads();
                    for (int stencil_y = STENCIL_MIN; stencil_y <= STENCIL_MAX; stencil_y++) {
                        int y = stencil_y - STENCIL_MIN;
                        for (int stencil_z = STENCIL_MIN; stencil_z <= STENCIL_MAX; stencil_z++) {
                            // Each iteration calculates two interactions, one for each of the two cells


                            // Overall index (required for accessing stencil related arrays)
                            const size_t index = x * STENCIL_INX * STENCIL_INX + y * STENCIL_INX + (stencil_z - STENCIL_MIN);
                            if (!device_stencil_masks[index]) {
                                // element not needed according to the stencil -> skip
                                // Note: that this will happen to all threads of the wrap
                                continue;
                            }

                            // partner index
                            const multiindex<> partner_index1(cell_index.x + stencil_x, cell_index.y + stencil_y, cell_index.z + stencil_z);
                            const multiindex<> partner_index2(cell_index2.x + stencil_x, cell_index2.y + stencil_y, cell_index2.z + stencil_z);
                            const size_t partner_flat_index1 = to_flat_index_padded(partner_index1);
                            const size_t partner_flat_index2 = to_flat_index_padded(partner_index2);
                            multiindex<> partner_index_coarse1(partner_index1);
                            multiindex<> partner_index_coarse2(partner_index2);
                            partner_index_coarse1.transform_coarse();
                            partner_index_coarse2.transform_coarse();

                            const double theta_c_rec_squared = static_cast<double>(
                                distance_squared_reciprocal(cell_index_coarse, partner_index_coarse1));
                            const double theta_c_rec_squared2 = static_cast<double>(
                                distance_squared_reciprocal(cell_index_coarse2, partner_index_coarse2));

                            // Where are we in the cache? Get required indices
                            // const size_t cache_index = index_base + stencil_y * cache_line_length + stencil_z;
                            // const size_t cache_index2 = index_base2 + stencil_y * cache_line_length + stencil_z;

                            // Use cached indices to create required masks
                            // const double theta_c_rec_squared = static_cast<double>(
                            //     distance_squared_reciprocal(cell_index_coarse, coarse_index_cache[cache_index]));
                            // const double theta_c_rec_squared2 = static_cast<double>(
                            //     distance_squared_reciprocal(cell_index_coarse2, coarse_index_cache[cache_index2]));
                            const bool mask_b = theta_rec_squared > theta_c_rec_squared;
                            const bool mask_b2 = theta_rec_squared > theta_c_rec_squared2;
                            const double mask = mask_b ? 1.0 : 0.0;
                            const double mask2 = mask_b2 ? 1.0 : 0.0;

                            const double r = std::sqrt(stencil_x * stencil_x + stencil_y * stencil_y + stencil_z * stencil_z);
                            const double r3 = r *r *r;
                            const double four[4] = {-1.0 /r, stencil_x / r3, stencil_y / r3, stencil_z / r3};
                            // Load required constants (same for both interactions)
                            // const double four[4] = {device_four_constants[index * 4 + 0],
                            //                         device_four_constants[index * 4 + 1],
                            //                         device_four_constants[index * 4 + 2],
                            //                         device_four_constants[index * 4 + 3]};

                            // Load masked monopoles from cache
                            // const double monopole = monopole_cache[cache_index] * mask * d_components[0];
                            // const double monopole2 = monopole_cache[cache_index2] * mask2 * d_components[0];

                            const double monopole = local_monopoles[partner_flat_index1] * mask * d_components[0];
                            const double monopole2 = local_monopoles[partner_flat_index2] * mask2 * d_components[0];


                            // Calculate the actual interactions
                            tmpstore[0] = tmpstore[0] + four[0] * monopole;
                            tmpstore2[0] = tmpstore2[0] + four[0] * monopole2;
                            tmpstore[1] = tmpstore[1] + four[1] * monopole * d_components[1];
                            tmpstore2[1] = tmpstore2[1] + four[1] * monopole2 * d_components[1];
                            tmpstore[2] = tmpstore[2] + four[2] * monopole * d_components[1];
                            tmpstore2[2] = tmpstore2[2] + four[2] * monopole2 * d_components[1];
                            tmpstore[3] = tmpstore[3] + four[3] * monopole * d_components[1];
                            tmpstore2[3] = tmpstore2[3] + four[3] * monopole2 * d_components[1];
                        }
                    }
                }

                // Store results in output arrays
                potential_expansions[cell_flat_index_unpadded] = tmpstore[0];
                potential_expansions[1 * component_length_unpadded +
                                     cell_flat_index_unpadded] = tmpstore[1];
                potential_expansions[2 * component_length_unpadded +
                                     cell_flat_index_unpadded] = tmpstore[2];
                potential_expansions[3 * component_length_unpadded +
                                     cell_flat_index_unpadded] = tmpstore[3];

                potential_expansions[cell_flat_index_unpadded2] = tmpstore2[0];
                potential_expansions[1 * component_length_unpadded +
                                     cell_flat_index_unpadded2] = tmpstore2[1];
                potential_expansions[2 * component_length_unpadded +
                                     cell_flat_index_unpadded2] = tmpstore2[2];
                potential_expansions[3 * component_length_unpadded +
                                     cell_flat_index_unpadded2] = tmpstore2[3];
            }
        }    // namespace monopole_interactions
    }    // namespace fmm
}    // namespace octotiger
#endif
