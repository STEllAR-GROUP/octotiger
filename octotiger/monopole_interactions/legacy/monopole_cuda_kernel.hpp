//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/common_kernel/multiindex.hpp"
#include "octotiger/cuda_util/cuda_helper.hpp"

#include <stream_manager.hpp>

#include <memory>

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {
        // extern __constant__ bool device_stencil_masks[FULL_STENCIL_SIZE];
        // extern __constant__ double device_four_constants[4 * FULL_STENCIL_SIZE];
        __host__ void init_stencil(size_t gpu_id, std::unique_ptr<bool[]> stencil_masks,
            std::unique_ptr<double[]> four_constants_tmp);
        constexpr int NUMBER_P2P_BLOCKS = STENCIL_MAX - STENCIL_MIN + 1;
        /*__global__ void cuda_p2p_interactions_kernel(
            const double (&local_monopoles)[NUMBER_LOCAL_MONOPOLE_VALUES],
            double (&potential_expansions)[NUMBER_POT_EXPANSIONS_SMALL], const double theta,
            const double dx);
        __global__ void cuda_p2m_interaction_rho(
            const double* __restrict__ expansions_neighbors_soa,
            const double* __restrict__ center_of_mass_neighbor_soa,
            const double* __restrict__ center_of_mass_cells_soa,
            double* __restrict__ potential_expansions, double* __restrict__ angular_corrections,
            const multiindex<> neighbor_size, const multiindex<> start_index,
            const multiindex<> dir, const multiindex<> end_index, const double theta,
            multiindex<> cells_start);
        __global__ void cuda_p2m_interaction_non_rho(
            const double* __restrict__ expansions_neighbors_soa,
            const double* __restrict__ center_of_mass_neighbor_soa,
            const double* __restrict__ center_of_mass_cells_soa,
            double* __restrict__ potential_expansions, const multiindex<> neighbor_size,
            const multiindex<> start_index, const multiindex<> end_index, const multiindex<> dir,
            const double theta, multiindex<> cells_start);*/
#ifdef OCTOTIGER_HAVE_CUDA
        void launch_p2p_cuda_kernel_post(
            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
            dim3 const grid_spec, dim3 const threads_per_block, void *args[]);
        void launch_sum_p2p_results_post(
            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
            dim3 const grid_spec, dim3 const threads_per_block, void *args[]);

        void launch_p2m_rho_cuda_kernel_post(
            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
            dim3 const grid_spec, dim3 const threads_per_block, void *args[]);
        void launch_p2m_non_rho_cuda_kernel_post(
            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
            dim3 const grid_spec, dim3 const threads_per_block, void *args[]);
#else
        void hip_p2p_interactions_kernel_post(
            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
            dim3 const grid_spec, dim3 const threads_per_block, const double *monopoles,
            double *tmp_potential_expansions,
            const double theta, const double dx);
        void hip_sum_p2p_results_post(
            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
            dim3 const grid_spec, dim3 const threads_per_block, 
            double *tmp_potential_expansions,
            double *potential_expansions);

#endif

    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
