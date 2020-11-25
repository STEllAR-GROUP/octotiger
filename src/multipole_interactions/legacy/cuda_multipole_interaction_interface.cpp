//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef OCTOTIGER_HAVE_CUDA
#include "octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp"
#include "octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp"
#include "octotiger/multipole_interactions/util/calculate_stencil.hpp"

#include "octotiger/defs.hpp"
#include "octotiger/options.hpp"

#include <array>
#include <vector>

#include <buffer_manager.hpp>
#include <cuda_buffer_util.hpp>
#include <cuda_runtime.h>
#include <stream_manager.hpp>

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        cuda_multipole_interaction_interface::cuda_multipole_interaction_interface()
          : multipole_interaction_interface()
          , theta(opts().theta) {}

        void cuda_multipole_interaction_interface::compute_multipole_interactions(
            std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::array<real, NDIM> xbase, const bool use_root_stencil) {
            bool avail = stream_pool::interface_available<hpx::cuda::experimental::cuda_executor,
                pool_strategy>(opts().cuda_buffer_capacity);
            if (!avail || m2m_type == interaction_kernel_type::OLD || (use_root_stencil && !opts().root_node_on_device)) {
                // Run fallback CPU implementation
                multipole_interaction_interface::compute_multipole_interactions(monopoles, M_ptr,
                    com_ptr, neighbors, type, dx, is_direction_empty, xbase, use_root_stencil);
            } else {    // run on cuda device
                if (type == RHO)
                    cuda_launch_counter()++;
                else
                    cuda_launch_counter_non_rho()++;

                size_t device_id =
                    stream_pool::get_next_device_id<hpx::cuda::experimental::cuda_executor,
                        pool_strategy>();
                stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy> executor;

                cuda_monopole_buffer_t local_monopoles(ENTRIES);
                cuda_expansion_buffer_t local_expansions_SoA;
                cuda_space_vector_buffer_t center_of_masses_SoA;
                cuda_expansion_result_buffer_t potential_expansions_SoA;
                cuda_angular_result_t angular_corrections_SoA;

                recycler::cuda_device_buffer<double> device_local_monopoles(ENTRIES, device_id);
                recycler::cuda_device_buffer<double> device_local_expansions(
                    NUMBER_LOCAL_EXPANSION_VALUES, device_id);
                recycler::cuda_device_buffer<double> device_centers(NUMBER_MASS_VALUES, device_id);
                recycler::cuda_device_buffer<double> device_erg_exp(
                    NUMBER_POT_EXPANSIONS, device_id);
                recycler::cuda_device_buffer<double> device_erg_corrs(NUMBER_ANG_CORRECTIONS);

                // Move data into SoA arrays
                this->dX = dx;
                this->xBase = xbase;
                this->type = type;
                update_input(monopoles, M_ptr, com_ptr, neighbors, type, dx, xbase, local_monopoles,
                    local_expansions_SoA, center_of_masses_SoA, grid_ptr, use_root_stencil);

                if (!use_root_stencil) {
                    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                        cudaMemcpyAsync, device_local_monopoles.device_side_buffer,
                        local_monopoles.data(), local_monopoles_size, cudaMemcpyHostToDevice);
                }
                hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                    cudaMemcpyAsync, device_local_expansions.device_side_buffer,
                    local_expansions_SoA.get_pod(), local_expansions_size, cudaMemcpyHostToDevice);
                hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                    cudaMemcpyAsync, device_centers.device_side_buffer,
                    center_of_masses_SoA.get_pod(), center_of_masses_size, cudaMemcpyHostToDevice);

                if (use_root_stencil) {
                    dim3 const grid_spec(INX, 1, 1);
                    dim3 const threads_per_block(1, INX, INX);
                    if (type == RHO) {
                        void* args[] = {&(device_centers.device_side_buffer),
                            &(device_local_expansions.device_side_buffer),
                            &(device_erg_exp.device_side_buffer),
                            &(device_erg_corrs.device_side_buffer)};
                        executor.post(
                            cudaLaunchKernel<decltype(cuda_multipole_interactions_kernel_root_rho)>,
                            cuda_multipole_interactions_kernel_root_rho, grid_spec,
                            threads_per_block, args, 0);
                        hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                            cudaMemcpyAsync, angular_corrections_SoA.get_pod(),
                            device_erg_corrs.device_side_buffer, angular_corrections_size,
                            cudaMemcpyDeviceToHost);
                    } else {
                        void* args[] = {&(device_centers.device_side_buffer),
                            &(device_local_expansions.device_side_buffer),
                            &(device_erg_exp.device_side_buffer)};
                        executor.post(cudaLaunchKernel<decltype(
                                          cuda_multipole_interactions_kernel_root_non_rho)>,
                            cuda_multipole_interactions_kernel_root_non_rho, grid_spec,
                            threads_per_block, args, 0);
                    }
                } else {
                    // Launch kernel and queue copying of results
                    dim3 const grid_spec(INX, 1, 1);
                    dim3 const threads_per_block(1, INX, INX);
                    if (type == RHO) {
                        bool second_phase = false;
                        void* args[] = {&(device_local_monopoles.device_side_buffer),
                            &(device_centers.device_side_buffer),
                            &(device_local_expansions.device_side_buffer),
                            &(device_erg_exp.device_side_buffer),
                            &(device_erg_corrs.device_side_buffer), &theta, &second_phase};
                        executor.post(
                            cudaLaunchKernel<decltype(cuda_multipole_interactions_kernel_rho)>,
                            cuda_multipole_interactions_kernel_rho, grid_spec, threads_per_block,
                            args, 0);
                        hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                            cudaMemcpyAsync, angular_corrections_SoA.get_pod(),
                            device_erg_corrs.device_side_buffer, angular_corrections_size,
                            cudaMemcpyDeviceToHost);
                    } else {
                        bool second_phase = false;
                        void* args[] = {&(device_local_monopoles.device_side_buffer),
                            &(device_centers.device_side_buffer),
                            &(device_local_expansions.device_side_buffer),
                            &(device_erg_exp.device_side_buffer), &theta, &second_phase};
                        executor.post(
                            cudaLaunchKernel<decltype(cuda_multipole_interactions_kernel_non_rho)>,
                            cuda_multipole_interactions_kernel_non_rho, grid_spec,
                            threads_per_block, args, 0);
                    }
                }
                auto fut = hpx::async(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                    cudaMemcpyAsync, potential_expansions_SoA.get_pod(),
                    device_erg_exp.device_side_buffer, potential_expansions_size,
                    cudaMemcpyDeviceToHost);

                // Wait for stream to finish and allow thread to jump away in the meantime
                fut.get();

                // Copy results back into non-SoA array
                potential_expansions_SoA.add_to_non_SoA(grid_ptr->get_L());
                if (type == RHO)
                    angular_corrections_SoA.to_non_SoA(grid_ptr->get_L_c());
            }
        }

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
