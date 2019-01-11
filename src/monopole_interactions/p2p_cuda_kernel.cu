#ifdef OCTOTIGER_CUDA_ENABLED
#include <sstream>
#include "monopole_kernel_templates.hpp"
#include "p2p_cuda_kernel.hpp"
namespace octotiger {
namespace fmm {
    namespace monopole_interactions {
        // __constant__ octotiger::fmm::multiindex<> device_stencil_const[P2P_PADDED_STENCIL_SIZE];
        __constant__ double device_stencil_masks[FULL_STENCIL_SIZE];
        __constant__ double device_four_constants[FULL_STENCIL_SIZE * 4];

        void copy_stencil_to_p2p_constant_memory(const double *stencil_masks, const size_t full_stencil_size) {
            cudaError_t err = cudaMemcpyToSymbol(device_stencil_masks, stencil_masks, full_stencil_size);
            if (err != cudaSuccess) {
                std::stringstream temp;
                temp << "Copy stencil to constant memory returned error code " << cudaGetErrorString(err);
                throw std::runtime_error(temp.str());
            }
        }
        void copy_constants_to_p2p_constant_memory(const double *constants, const size_t constants_size) {
            cudaError_t err = cudaMemcpyToSymbol(device_four_constants, constants, constants_size);
            if (err != cudaSuccess) {
                std::stringstream temp;
                temp << "Copy four-constants to constant memory returned error code " << cudaGetErrorString(err);
                throw std::runtime_error(temp.str());
            }
        }


        __device__ constexpr size_t component_length = ENTRIES + SOA_PADDING;
        __device__ constexpr size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;

        __global__ void
        __launch_bounds__(64, 8)
        cuda_p2p_interactions_kernel(
            const double (&local_monopoles)[NUMBER_LOCAL_MONOPOLE_VALUES],
            double (&potential_expansions)[NUMBER_POT_EXPANSIONS_SMALL],
            const double theta, const double dx) {
            __shared__ double monopole_cache[8 * 18];
            __shared__ multiindex<> coarse_index_cache[8 * 18];
            // get local id
            int local_id = threadIdx.y * 8 + threadIdx.z;

            // use in case of debug prints
            // bool first_thread = (blockIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0);
            // Set cell indices
            const octotiger::fmm::multiindex<> cell_index((threadIdx.x + blockIdx.x * 1) + INNER_CELLS_PADDING_DEPTH,
                threadIdx.y + INNER_CELLS_PADDING_DEPTH, threadIdx.z + INNER_CELLS_PADDING_DEPTH);
            octotiger::fmm::multiindex<> cell_index_coarse(cell_index);
            cell_index_coarse.transform_coarse();
            const size_t cell_flat_index = octotiger::fmm::to_flat_index_padded(cell_index);
            octotiger::fmm::multiindex<> cell_index_unpadded((threadIdx.x + blockIdx.x * 1), threadIdx.y, threadIdx.z);
            const size_t cell_flat_index_unpadded =
                octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded);
            const int cache_index_base = cell_index_unpadded.y * 18 +
                            cell_index.z - 3;

            // Required for mask
            const double theta_rec_squared = sqr(1.0 / theta);
            const double d_components[2] = {1.0 / dx, -1.0 / dx};
            double tmpstore[4] = {0.0, 0.0, 0.0, 0.0};

            for (int stencil_x = STENCIL_MIN; stencil_x <= STENCIL_MAX; stencil_x++) {
                int x = stencil_x - STENCIL_MIN;
                __syncthreads();
                if (local_id < 18) {
                    for (int i = 0; i < 8; i++) {
                        const multiindex<> partner_index(INNER_CELLS_PADDING_DEPTH + blockIdx.x + stencil_x,
                                                            INNER_CELLS_PADDING_DEPTH + STENCIL_MIN + i,
                                                            3 + local_id);
                        const size_t partner_flat_index = to_flat_index_padded(partner_index);
                        multiindex<> partner_index_coarse(partner_index);
                        partner_index_coarse.transform_coarse();
                        coarse_index_cache[18*i + local_id] = partner_index_coarse;
                        monopole_cache[18*i + local_id] = local_monopoles[partner_flat_index];
                    }
                }
                __syncthreads();
                for (int stencil_y = STENCIL_MIN; stencil_y <= STENCIL_MAX; stencil_y++) {
                    int y = stencil_y - STENCIL_MIN;
                    for (int stencil_z = STENCIL_MIN; stencil_z <= STENCIL_MAX; stencil_z++) {
                        const size_t index = x * STENCIL_INX * STENCIL_INX + y * STENCIL_INX + (stencil_z - STENCIL_MIN);
                        if (!device_stencil_masks[index]) {
                            continue;
                        }

                        int cache_index = cache_index_base + stencil_z;

                        // Create mask
                        const double theta_c_rec_squared = static_cast<double>(
                            distance_squared_reciprocal(cell_index_coarse, coarse_index_cache[cache_index]));
                        const bool mask_b = theta_rec_squared > theta_c_rec_squared;
                        double mask = mask_b ? 1.0 : 0.0;


                        double monopole = monopole_cache[cache_index] * mask * d_components[0];
                        const double four[4] = {device_four_constants[index * 4 + 0],
                                                device_four_constants[index * 4 + 1],
                                                device_four_constants[index * 4 + 2],
                                                device_four_constants[index * 4 + 3]};
                        tmpstore[0] = tmpstore[0] + four[0] * monopole;
                        tmpstore[1] = tmpstore[1] + four[1] * monopole * d_components[1];
                        tmpstore[2] = tmpstore[2] + four[2] * monopole * d_components[1];
                        tmpstore[3] = tmpstore[3] + four[3] * monopole * d_components[1];
                        // compute_monopole_interaction<double>(monopole, tmpstore, four, d_components);
                    }
                    if (stencil_y < STENCIL_MAX && local_id < 18) {
                        // move stencil
                        __syncthreads();
                        for (int i = 0; i < 7; i++) {
                            coarse_index_cache[18*i + local_id] = coarse_index_cache[18*(i + 1) + local_id];
                            monopole_cache[18*i + local_id] = monopole_cache[18*(i + 1) + local_id];
                            __syncthreads();
                        }
                        // Load new row
                        const multiindex<> partner_index(INNER_CELLS_PADDING_DEPTH + blockIdx.x + stencil_x,
                                                         INNER_CELLS_PADDING_DEPTH + (stencil_y + 1) + 7,
                                                            3 + local_id);
                        const size_t partner_flat_index = to_flat_index_padded(partner_index);
                        multiindex<> partner_index_coarse(partner_index);
                        partner_index_coarse.transform_coarse();
                        coarse_index_cache[18*7 + local_id] = partner_index_coarse;
                        monopole_cache[18*7 + local_id] = local_monopoles[partner_flat_index];
                        __syncthreads();
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
