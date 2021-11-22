#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
#include "octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp"
#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/cuda_util/cuda_scheduler.hpp"
#include "octotiger/monopole_interactions/kernel/monopole_kernel_templates.hpp"

#if defined(OCTOTIGER_HAVE_HIP)
#define cudaSetDevice hipSetDevice
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#endif

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {
        // __constant__ octotiger::fmm::multiindex<> device_stencil_const[P2P_PADDED_STENCIL_SIZE];
        __device__ __constant__ bool device_stencil_masks[FULL_STENCIL_SIZE];
        __device__ __constant__ double device_four_constants[FULL_STENCIL_SIZE * 4];

        __host__ void init_stencil(size_t gpu_id, std::unique_ptr<bool[]> stencil_masks,
            std::unique_ptr<double[]> four_constants_tmp) {
            cudaSetDevice(gpu_id);
            cudaMemcpyToSymbol(monopole_interactions::device_stencil_masks, stencil_masks.get(),
                full_stencil_size / sizeof(double) * sizeof(bool));
            cudaMemcpyToSymbol(monopole_interactions::device_four_constants,
                four_constants_tmp.get(), full_stencil_size * 4);
        }

        //__device__ const size_t component_length = ENTRIES + SOA_PADDING;
        __device__ const size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;
        __device__ const size_t cache_line_length = INX + 2 * STENCIL_MAX;
        __device__ const size_t cache_offset = INX + STENCIL_MIN;

#if defined(OCTOTIGER_HAVE_HIP)
        __global__ void cuda_p2p_interactions_kernel(
#else
        __global__ void __launch_bounds__(INX* INX, 2) cuda_p2p_interactions_kernel(
#endif
            const double *local_monopoles,
            double *potential_expansions, const double theta,
            const double dx) {

            const int block_id = blockIdx.y; 

            // use in case of debug prints
            // bool first_thread = (blockIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0);
            // Set cell indices
            const octotiger::fmm::multiindex<> cell_index(
                (threadIdx.x + blockIdx.z * 1) + INNER_CELLS_PADDING_DEPTH,
                threadIdx.y + INNER_CELLS_PADDING_DEPTH, threadIdx.z + INNER_CELLS_PADDING_DEPTH);
            octotiger::fmm::multiindex<> cell_index_coarse(cell_index);
            cell_index_coarse.transform_coarse();
            const size_t cell_flat_index = octotiger::fmm::to_flat_index_padded(cell_index);
            octotiger::fmm::multiindex<> cell_index_unpadded(
                (threadIdx.x + blockIdx.z * 1), threadIdx.y, threadIdx.z);
            const size_t cell_flat_index_unpadded =
                octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded);

            // Calculate required constants
            const double theta_rec_squared = sqr(1.0 / theta);
            const double d_components[2] = {1.0 / dx, -1.0 / dx};
            double tmpstore[4] = {0.0, 0.0, 0.0, 0.0};
            const size_t index_base =
                (threadIdx.y + STENCIL_MAX) * cache_line_length + threadIdx.z + STENCIL_MAX;

            const int x = block_id;
            const int stencil_x = x + STENCIL_MIN;

            for (int stencil_y = STENCIL_MIN; stencil_y <= STENCIL_MAX; stencil_y++) {
                int y = stencil_y - STENCIL_MIN;
                for (int stencil_z = STENCIL_MIN; stencil_z <= STENCIL_MAX; stencil_z++) {
                    // Each iteration calculates two interactions, one for each of the two cells

                    // Overall index (required for accessing stencil related arrays)
                    const size_t index = x * STENCIL_INX * STENCIL_INX + y * STENCIL_INX +
                        (stencil_z - STENCIL_MIN);
                    if (!device_stencil_masks[index]) {
                        // element not needed according to the stencil -> skip
                        // Note: that this will happen to all threads of the wrap
                        continue;
                    }

                    // partner index
                    const multiindex<> partner_index1(cell_index.x + stencil_x,
                        cell_index.y + stencil_y, cell_index.z + stencil_z);
                    const size_t partner_flat_index1 = to_flat_index_padded(partner_index1);
                    multiindex<> partner_index_coarse1(partner_index1);
                    partner_index_coarse1.transform_coarse();

                    const double theta_c_rec_squared = static_cast<double>(
                        distance_squared_reciprocal(cell_index_coarse, partner_index_coarse1));

                    const bool mask_b = theta_rec_squared > theta_c_rec_squared;
                    const double mask = mask_b ? 1.0 : 0.0;

                    // Load required constants (same for both interactions)
                    const double four[4] = {device_four_constants[index * 4 + 0],
                        device_four_constants[index * 4 + 1],
                        device_four_constants[index * 4 + 2],
                        device_four_constants[index * 4 + 3]};


                    const double monopole =
                        local_monopoles[partner_flat_index1] * mask * d_components[0];

                    // Calculate the actual interactions
                    tmpstore[0] = tmpstore[0] + four[0] * monopole;
                    tmpstore[1] = tmpstore[1] + four[1] * monopole * d_components[1];
                    tmpstore[2] = tmpstore[2] + four[2] * monopole * d_components[1];
                    tmpstore[3] = tmpstore[3] + four[3] * monopole * d_components[1];
                }
            }

            // Store results in output arrays
            potential_expansions[block_id * NUMBER_POT_EXPANSIONS_SMALL + cell_flat_index_unpadded] = tmpstore[0];
            potential_expansions[block_id * NUMBER_POT_EXPANSIONS_SMALL +
              1 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[1];
            potential_expansions[block_id * NUMBER_POT_EXPANSIONS_SMALL +
              2 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[2];
            potential_expansions[block_id * NUMBER_POT_EXPANSIONS_SMALL +
              3 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[3];
        }
        __global__ void cuda_sum_p2p_results(
            double *tmp_potential_expansions,
            double *potential_expansions) {
            octotiger::fmm::multiindex<> cell_index_unpadded(
                (threadIdx.x + blockIdx.z), threadIdx.y, threadIdx.z);
            const size_t cell_flat_index_unpadded =
                octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded);
            double tmpstore[4] = {0.0, 0.0, 0.0, 0.0};
            for (int i = 0; i < NUMBER_P2P_BLOCKS; i++) {
                    tmpstore[0] = tmpstore[0] +
                      tmp_potential_expansions[i * NUMBER_POT_EXPANSIONS_SMALL + cell_flat_index_unpadded];
                    tmpstore[1] = tmpstore[1] +
                      tmp_potential_expansions[i * NUMBER_POT_EXPANSIONS_SMALL +
                      1 * component_length_unpadded + cell_flat_index_unpadded];
                    tmpstore[2] = tmpstore[2] +
                      tmp_potential_expansions[i * NUMBER_POT_EXPANSIONS_SMALL +
                      2 * component_length_unpadded + cell_flat_index_unpadded];
                    tmpstore[3] = tmpstore[3] +
                      tmp_potential_expansions[i * NUMBER_POT_EXPANSIONS_SMALL +
                      3 * component_length_unpadded + cell_flat_index_unpadded];
            }
            potential_expansions[cell_flat_index_unpadded] = tmpstore[0];
            potential_expansions[1 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[1];
            potential_expansions[2 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[2];
            potential_expansions[3 * component_length_unpadded + cell_flat_index_unpadded] =
                tmpstore[3];
        }
#if defined(OCTOTIGER_HAVE_HIP)
        void hip_p2p_interactions_kernel_ggl_wrapper(dim3 const grid_spec,
            dim3 const threads_per_block, const double *monopoles, 
            double *tmp_potential_expansions,
            const double theta, const double dx,
            hipStream_t& stream) {
            hipLaunchKernelGGL(cuda_p2p_interactions_kernel, grid_spec, threads_per_block,
                0, stream, monopoles, tmp_potential_expansions, 
                theta, dx);
        }
        void hip_p2p_interactions_kernel_post(
            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
            dim3 const grid_spec, dim3 const threads_per_block, const double *monopoles,
            double *tmp_potential_expansions,
            const double theta, const double dx) {
            executor.post(hip_p2p_interactions_kernel_ggl_wrapper, grid_spec,
                threads_per_block, monopoles, tmp_potential_expansions,
                theta, dx);
        }
        void hip_sum_p2p_results_ggl_wrapper(dim3 const grid_spec,
            dim3 const threads_per_block, 
            double *tmp_potential_expansions,
            double *potential_expansions,
            hipStream_t& stream) {
            hipLaunchKernelGGL(cuda_sum_p2p_results, grid_spec, threads_per_block,
                0, stream, tmp_potential_expansions, 
                potential_expansions);
        }
        void hip_sum_p2p_results_post(
            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
            dim3 const grid_spec, dim3 const threads_per_block, 
            double *tmp_potential_expansions,
            double *potential_expansions) {
            executor.post(hip_sum_p2p_results_ggl_wrapper, grid_spec,
                threads_per_block, tmp_potential_expansions,
                potential_expansions);
        }
#else
        void launch_sum_p2p_results_post(stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
            dim3 const grid_spec, dim3 const threads_per_block, void *args[]) {
            executor.post(
            cudaLaunchKernel<decltype(cuda_sum_p2p_results)>,
            cuda_sum_p2p_results, grid_spec, threads_per_block, args, 0);
        }

        void launch_p2p_cuda_kernel_post(stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
            dim3 const grid_spec, dim3 const threads_per_block, void *args[]) {
            executor.post(
            cudaLaunchKernel<decltype(cuda_p2p_interactions_kernel)>,
            cuda_p2p_interactions_kernel, grid_spec, threads_per_block, args, 0);
        }
#endif

#ifndef OCTOTIGER_HAVE_HIP
        __global__ void __launch_bounds__(INX* INX, 2)
            cuda_p2m_interaction_rho(const double* __restrict__ expansions_neighbors_soa,
                const double* __restrict__ center_of_mass_neighbor_soa,
                const double* __restrict__ center_of_mass_cells_soa,
                double* __restrict__ potential_expansions, double* __restrict__ angular_corrections,
                const multiindex<> neighbor_size, const multiindex<> start_index,
                const multiindex<> end_index, const multiindex<> dir, const double theta,
                multiindex<> cells_start) {
            int index_x = threadIdx.x + blockIdx.x;
            const int component_length_neighbor =
                neighbor_size.x * neighbor_size.y * neighbor_size.z + SOA_PADDING;
            // Set cell indices
            const octotiger::fmm::multiindex<> cell_index(
                index_x + INNER_CELLS_PADDING_DEPTH + cells_start.x,
                threadIdx.y + INNER_CELLS_PADDING_DEPTH + cells_start.y,
                threadIdx.z + INNER_CELLS_PADDING_DEPTH + cells_start.z);
            octotiger::fmm::multiindex<> cell_index_coarse(cell_index);
            cell_index_coarse.transform_coarse();
            const size_t cell_flat_index = octotiger::fmm::to_flat_index_padded(cell_index);
            octotiger::fmm::multiindex<> cell_index_unpadded(
                index_x + cells_start.x, threadIdx.y + cells_start.y, threadIdx.z + cells_start.z);
            const size_t cell_flat_index_unpadded =
                octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded);

            const double theta_rec_squared = sqr(1.0 / theta);

            double X[NDIM];
            X[0] = center_of_mass_cells_soa[cell_flat_index_unpadded];
            X[1] =
                center_of_mass_cells_soa[1 * component_length_unpadded + cell_flat_index_unpadded];
            X[2] =
                center_of_mass_cells_soa[2 * component_length_unpadded + cell_flat_index_unpadded];

            // Create and set result arrays
            double tmpstore[4];
#pragma unroll
            for (size_t i = 0; i < 4; ++i)
                tmpstore[i] = 0.0;
            double tmp_corrections[3];
#pragma unroll
            for (size_t i = 0; i < 3; ++i)
                tmp_corrections[i] = 0.0;
            double m_partner[20];
            double Y[NDIM];
            for (size_t x = start_index.x; x < end_index.x; x++) {
                for (size_t y = start_index.y; y < end_index.y; y++) {
                    for (size_t z = start_index.z; z < end_index.z; z++) {
                        // Global index (regarding inner cells + all neighbors)
                        // Used to figure out which stencil mask to use
                        const multiindex<> interaction_partner_index(
                            INNER_CELLS_PADDING_DEPTH + dir.x * INNER_CELLS_PADDING_DEPTH + x,
                            INNER_CELLS_PADDING_DEPTH + dir.y * INNER_CELLS_PADDING_DEPTH + y,
                            INNER_CELLS_PADDING_DEPTH + dir.z * INNER_CELLS_PADDING_DEPTH + z);

                        // Get stencil mask and skip if necessary
                        multiindex<> stencil_element(
                            interaction_partner_index.x - cell_index.x - STENCIL_MIN,
                            interaction_partner_index.y - cell_index.y - STENCIL_MIN,
                            interaction_partner_index.z - cell_index.z - STENCIL_MIN);
                        const size_t stencil_flat_index =
                            stencil_element.x * STENCIL_INX * STENCIL_INX +
                            stencil_element.y * STENCIL_INX + stencil_element.z;
                        if (!device_stencil_masks[stencil_flat_index])
                            continue;

                        multiindex<> partner_index_coarse(interaction_partner_index);
                        partner_index_coarse.transform_coarse();
                        const double theta_c_rec_squared = static_cast<double>(
                            distance_squared_reciprocal(cell_index_coarse, partner_index_coarse));
                        const bool mask_b = theta_rec_squared > theta_c_rec_squared;
                        double mask = mask_b ? 1.0 : 0.0;
                        if (!mask_b)
                            continue;

                        // Local index
                        // Used to figure out which data element to use
                        const multiindex<> interaction_partner_data_index(
                            x - start_index.x, y - start_index.y, z - start_index.z);
                        const size_t interaction_partner_flat_index =
                            interaction_partner_data_index.x * (neighbor_size.y * neighbor_size.z) +
                            interaction_partner_data_index.y * neighbor_size.z +
                            interaction_partner_data_index.z;

                        // Load data of interaction partner
                        Y[0] = center_of_mass_neighbor_soa[interaction_partner_flat_index];
                        Y[1] = center_of_mass_neighbor_soa[1 * component_length_neighbor +
                            interaction_partner_flat_index];
                        Y[2] = center_of_mass_neighbor_soa[2 * component_length_neighbor +
                            interaction_partner_flat_index];
#pragma unroll
                        for (size_t i = 0; i < 20; ++i)
                            m_partner[i] = expansions_neighbors_soa[i * component_length_neighbor +
                                               interaction_partner_flat_index] *
                                mask;

                        // run templated interaction method instanced with double type
                        compute_kernel_p2m_rho(X, Y, m_partner, tmpstore, tmp_corrections,
                            [](const double& one, const double& two) -> double {
                                return std::max(one, two);
                            });
                    }
                }
            }
// Store results in output arrays
#pragma unroll
            for (size_t i = 0; i < 4; ++i) {
                potential_expansions[i * component_length_unpadded + cell_flat_index_unpadded] +=
                    tmpstore[i];
            }
            angular_corrections[cell_flat_index_unpadded] += tmp_corrections[0];
            angular_corrections[1 * component_length_unpadded + cell_flat_index_unpadded] +=
                tmp_corrections[1];
            angular_corrections[2 * component_length_unpadded + cell_flat_index_unpadded] +=
                tmp_corrections[2];
        }
        void launch_p2m_rho_cuda_kernel_post(stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
            dim3 const grid_spec, dim3 const threads_per_block, void *args[]) {
            executor.post(
            cudaLaunchKernel<decltype(cuda_p2m_interaction_rho)>,
            cuda_p2m_interaction_rho, grid_spec, threads_per_block, args, 0);
        }

        __global__ void __launch_bounds__(INX* INX, 2)
            cuda_p2m_interaction_non_rho(const double* __restrict__ expansions_neighbors_soa,
                const double* __restrict__ center_of_mass_neighbor_soa,
                const double* __restrict__ center_of_mass_cells_soa,
                double* __restrict__ potential_expansions, const multiindex<> neighbor_size,
                const multiindex<> start_index, const multiindex<> end_index,
                const multiindex<> dir, const double theta, multiindex<> cells_start) {
            int index_x = threadIdx.x + blockIdx.x;
            const int component_length_neighbor =
                neighbor_size.x * neighbor_size.y * neighbor_size.z + SOA_PADDING;
            // Set cell indices
            const octotiger::fmm::multiindex<> cell_index(
                index_x + INNER_CELLS_PADDING_DEPTH + cells_start.x,
                threadIdx.y + INNER_CELLS_PADDING_DEPTH + cells_start.y,
                threadIdx.z + INNER_CELLS_PADDING_DEPTH + cells_start.z);
            octotiger::fmm::multiindex<> cell_index_coarse(cell_index);
            cell_index_coarse.transform_coarse();
            const size_t cell_flat_index = octotiger::fmm::to_flat_index_padded(cell_index);
            octotiger::fmm::multiindex<> cell_index_unpadded(
                index_x + cells_start.x, threadIdx.y + cells_start.y, threadIdx.z + cells_start.z);
            const size_t cell_flat_index_unpadded =
                octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded);

            const double theta_rec_squared = sqr(1.0 / theta);

            double X[NDIM];
            X[0] = center_of_mass_cells_soa[cell_flat_index_unpadded];
            X[1] =
                center_of_mass_cells_soa[1 * component_length_unpadded + cell_flat_index_unpadded];
            X[2] =
                center_of_mass_cells_soa[2 * component_length_unpadded + cell_flat_index_unpadded];

            // Create and set result arrays
            double tmpstore[4];
#pragma unroll
            for (size_t i = 0; i < 4; ++i)
                tmpstore[i] = 0.0;
            double m_partner[20];
            double Y[NDIM];
            for (size_t x = start_index.x; x < end_index.x; x++) {
                for (size_t y = start_index.y; y < end_index.y; y++) {
                    for (size_t z = start_index.z; z < end_index.z; z++) {
                        // Global index (regarding inner cells + all neighbors)
                        // Used to figure out which stencil mask to use
                        const multiindex<> interaction_partner_index(
                            INNER_CELLS_PADDING_DEPTH + dir.x * INNER_CELLS_PADDING_DEPTH + x,
                            INNER_CELLS_PADDING_DEPTH + dir.y * INNER_CELLS_PADDING_DEPTH + y,
                            INNER_CELLS_PADDING_DEPTH + dir.z * INNER_CELLS_PADDING_DEPTH + z);

                        // Get stencil mask and skip if necessary
                        multiindex<> stencil_element(
                            interaction_partner_index.x - cell_index.x - STENCIL_MIN,
                            interaction_partner_index.y - cell_index.y - STENCIL_MIN,
                            interaction_partner_index.z - cell_index.z - STENCIL_MIN);
                        const size_t stencil_flat_index =
                            stencil_element.x * STENCIL_INX * STENCIL_INX +
                            stencil_element.y * STENCIL_INX + stencil_element.z;
                        if (!device_stencil_masks[stencil_flat_index])
                            continue;

                        multiindex<> partner_index_coarse(interaction_partner_index);
                        partner_index_coarse.transform_coarse();
                        const double theta_c_rec_squared = static_cast<double>(
                            distance_squared_reciprocal(cell_index_coarse, partner_index_coarse));
                        const bool mask_b = theta_rec_squared > theta_c_rec_squared;
                        double mask = mask_b ? 1.0 : 0.0;
                        if (!mask_b)
                            continue;

                        // Local index
                        // Used to figure out which data element to use
                        const multiindex<> interaction_partner_data_index(
                            x - start_index.x, y - start_index.y, z - start_index.z);
                        const size_t interaction_partner_flat_index =
                            interaction_partner_data_index.x * (neighbor_size.y * neighbor_size.z) +
                            interaction_partner_data_index.y * neighbor_size.z +
                            interaction_partner_data_index.z;

                        // Load data of interaction partner
                        Y[0] = center_of_mass_neighbor_soa[interaction_partner_flat_index];
                        Y[1] = center_of_mass_neighbor_soa[1 * component_length_neighbor +
                            interaction_partner_flat_index];
                        Y[2] = center_of_mass_neighbor_soa[2 * component_length_neighbor +
                            interaction_partner_flat_index];
#pragma unroll
                        for (size_t i = 0; i < 20; ++i)
                            m_partner[i] = expansions_neighbors_soa[i * component_length_neighbor +
                                               interaction_partner_flat_index] *
                                mask;

                        // run templated interaction method instanced with double type
                        compute_kernel_p2m_non_rho(X, Y, m_partner, tmpstore,
                            [](const double& one, const double& two) -> double {
                                return std::max(one, two);
                            });
                    }
                }
            }
// Store results in output arrays
#pragma unroll
            for (size_t i = 0; i < 4; ++i)
                potential_expansions[i * component_length_unpadded + cell_flat_index_unpadded] +=
                    tmpstore[i];
        }
        void launch_p2m_non_rho_cuda_kernel_post(
            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
            dim3 const grid_spec, dim3 const threads_per_block, void *args[]) {
            executor.post(
            cudaLaunchKernel<decltype(cuda_p2m_interaction_non_rho)>,
            cuda_p2m_interaction_non_rho, grid_spec, threads_per_block, args, 0);
        }
#endif
    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
