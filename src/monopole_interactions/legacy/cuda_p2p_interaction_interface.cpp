//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef OCTOTIGER_HAVE_CUDA

#include "octotiger/monopole_interactions/legacy/cuda_p2p_interaction_interface.hpp"
#include "octotiger/monopole_interactions/legacy/p2m_interaction_interface.hpp"
#include "octotiger/monopole_interactions/legacy/p2p_cuda_kernel.hpp"
#include "octotiger/monopole_interactions/util/calculate_stencil.hpp"

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
    namespace monopole_interactions {
        cuda_p2p_interaction_interface::cuda_p2p_interaction_interface()
          : p2p_interaction_interface()
          , theta(opts().theta) {}

        void cuda_p2p_interaction_interface::compute_p2p_interactions(std::vector<real>& monopoles,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::shared_ptr<grid>& grid_ptr, const bool contains_multipole_neighbor) {
            // Check where we want to run this:
            bool avail = stream_pool::interface_available<hpx::cuda::experimental::cuda_executor,
                pool_strategy>(opts().cuda_buffer_capacity);
            //avail = true;
            if (!avail || p2p_type == interaction_kernel_type::OLD) {
                // Run CPU implementation
                p2p_interaction_interface::compute_p2p_interactions(monopoles, com_ptr, neighbors,
                    type, dx, is_direction_empty, grid_ptr, contains_multipole_neighbor);
            } else {
                // run on CUDA device
                cuda_launch_counter()++;

                // Pick device and stream
                size_t device_id =
                    stream_pool::get_next_device_id<hpx::cuda::experimental::cuda_executor,
                        pool_strategy>();
                stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy> executor;

                cuda_expansion_result_buffer_t potential_expansions_SoA;
                cuda_monopole_buffer_t local_monopoles(ENTRIES);
                recycler::cuda_device_buffer<double> device_local_monopoles(ENTRIES, device_id);
                recycler::cuda_device_buffer<double> erg(NUMBER_POT_EXPANSIONS_SMALL, device_id);

                // Move data into staging buffers
                update_input(monopoles, neighbors, type, local_monopoles, neighbor_empty_monopoles,
                    grid_ptr);

                hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                    cudaMemcpyAsync, device_local_monopoles.device_side_buffer,
                    local_monopoles.data(), local_monopoles_size, cudaMemcpyHostToDevice);

                // Launch kernel and queue copying of results
                dim3 const grid_spec(1, 1, INX);
                dim3 const threads_per_block(1, INX, INX);
                void* args[] = {&(device_local_monopoles.device_side_buffer),
                    &(erg.device_side_buffer), &theta, &dx};
                // hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                //     cudalaunchkernel<decltype(cuda_p2p_interactions_kernel)>,
                //     cuda_p2p_interactions_kernel, grid_spec, threads_per_block, args, 0);
                executor.post(cudaLaunchKernel<decltype(cuda_p2p_interactions_kernel)>,
                    cuda_p2p_interactions_kernel, grid_spec, threads_per_block, args, 0);

                //if (contains_multipole_neighbor && opts().p2m_kernel_type == SOA_CPU) {
                //if (contains_multipole_neighbor && type == RHO) {
                //    compute_p2m_interactions_neighbors_only(monopoles, com_ptr, neighbors, type, is_direction_empty, grid_ptr);
                if (contains_multipole_neighbor) {
                //} else if (contains_multipole_neighbor && opts().p2m_kernel_type == SOA_CUDA) {
                    // Convert and move innter cells coms to device
                recycler::cuda_device_buffer<double> device_erg_corrs(NUMBER_ANG_CORRECTIONS);
                hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                    cudaMemsetAsync, device_erg_corrs.device_side_buffer, 0,

                    (INNER_CELLS + SOA_PADDING) * 3 * sizeof(double));
                    std::vector<space_vector> const& com0 = *(com_ptr[0]);
                    struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING,
                        std::vector<real, recycler::recycle_allocator_cuda_host<real>>>
                        center_of_masses_inner_cells_staging_area;
                    iterate_inner_cells_padded(
                        [&center_of_masses_inner_cells_staging_area, com0](const multiindex<>& i,
                            const size_t flat_index, const multiindex<>& i_unpadded,
                            const size_t flat_index_unpadded) {
                            center_of_masses_inner_cells_staging_area.set_AoS_value(
                                std::move(com0.at(flat_index_unpadded)), flat_index_unpadded);
                        });
                    recycler::cuda_device_buffer<double> center_of_masses_inner_cells(
                        (INNER_CELLS + SOA_PADDING) * 3, device_id);
                    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                        cudaMemcpyAsync, center_of_masses_inner_cells.device_side_buffer,
                        center_of_masses_inner_cells_staging_area.get_pod(),
                        (INNER_CELLS + SOA_PADDING) * 3 * sizeof(double), cudaMemcpyHostToDevice);

                    // Iterate through neighbors
                    for (const geo::direction& dir : geo::direction::full_set()) {
                        neighbor_gravity_type& neighbor = neighbors[dir];
                        // Multipole neighbor that actually contains any data?
                         if (!neighbor.is_monopole && neighbor.data.M) {
                            int size = 1;
                            for (int i = 0; i < 3; i++) {
                                if (dir[i] == 0)
                                    size *= INX;
                                else
                                    size *= STENCIL_MAX;
                            }
                            assert(size == INX * INX * STENCIL_MAX ||
                                size == INX * STENCIL_MAX * STENCIL_MAX ||
                                size == STENCIL_MAX * STENCIL_MAX * STENCIL_MAX);
                            multiindex<> start_index = get_padding_start_indices(dir);
                            multiindex<> end_index = get_padding_end_indices(dir);
                            multiindex<> neighbor_size = get_padding_real_size(dir);
                            multiindex<> dir_index;
                            dir_index.x = dir[0];
                            dir_index.y = dir[1];
                            dir_index.z = dir[2];
                            multiindex<> cells_start(0, 0, 0);
                            multiindex<> cells_end(INX, INX, INX);
                            if (dir[0] == 1)
                                cells_start.x = INX - (STENCIL_MAX + 1);
                            if (dir[0] == -1)
                                cells_end.x = (STENCIL_MAX + 1);
                            if (dir[1] == 1)
                                cells_start.y = INX - (STENCIL_MAX + 1);
                            if (dir[1] == -1)
                                cells_end.y = (STENCIL_MAX + 1);
                            if (dir[2] == 1)
                                cells_start.z = INX - (STENCIL_MAX + 1);
                            if (dir[2] == -1)
                                cells_end.z = (STENCIL_MAX + 1);

                            // TODO Get array of these?! Otherwise I need to overwrite these which
                            // would be absolutely awful Possiblity 2: sync after kernel... <-- do
                            // this
                            if (size == INX * INX * STENCIL_MAX) {
                                constexpr size_t buffer_size = INX * INX * STENCIL_MAX;
                                struct_of_array_data<expansion, real, 20, buffer_size, SOA_PADDING,
                                    std::vector<real, recycler::recycle_allocator_cuda_host<real>>>
                                    local_expansions_staging_area;
                                struct_of_array_data<space_vector, real, 3, buffer_size,
                                    SOA_PADDING,
                                    std::vector<real, recycler::recycle_allocator_cuda_host<real>>>
                                    center_of_masses_staging_area;
                                update_neighbor_input(dir, com_ptr, neighbors, type,
                                    local_expansions_staging_area, center_of_masses_staging_area,
                                    grid_ptr);
                                recycler::cuda_device_buffer<double> local_expansions(
                                    (buffer_size + SOA_PADDING) * 20, device_id);
                                recycler::cuda_device_buffer<double> center_of_masses(
                                    (buffer_size + SOA_PADDING) * 3, device_id);
                                hpx::apply(
                                    static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                                    cudaMemcpyAsync, local_expansions.device_side_buffer,
                                    local_expansions_staging_area.get_pod(),
                                    (buffer_size + SOA_PADDING) * 20 * sizeof(double),
                                    cudaMemcpyHostToDevice);
                                hpx::apply(
                                    static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                                    cudaMemcpyAsync, center_of_masses.device_side_buffer,
                                    center_of_masses_staging_area.get_pod(),
                                    (buffer_size + SOA_PADDING) * 3 * sizeof(double),
                                    cudaMemcpyHostToDevice);

                                if (type == RHO) {
                                    dim3 const grid_spec(cells_end.x - cells_start.x, 1, 1);
                                    dim3 const threads_per_block(1, cells_end.y - cells_start.y, cells_end.z - cells_start.z);
                                    void* args[] = {&(local_expansions.device_side_buffer),
                                        &(center_of_masses.device_side_buffer),
                                        &(center_of_masses_inner_cells.device_side_buffer),
                                        &(erg.device_side_buffer),
                                        &(device_erg_corrs.device_side_buffer), &neighbor_size,
                                        &start_index, &end_index, &dir_index, &theta, &cells_start};
                                    auto fut = executor.async_execute(
                                        cudaLaunchKernel<decltype(cuda_p2m_interaction_rho)>,
                                        cuda_p2m_interaction_rho, grid_spec, threads_per_block,
                                        args, 0);
                                    fut.get();
                                } else {
                                    dim3 const grid_spec(cells_end.x - cells_start.x, 1, 1);
                                    dim3 const threads_per_block(1, cells_end.y - cells_start.y, cells_end.z - cells_start.z);
                                    void* args[] = {&(local_expansions.device_side_buffer),
                                        &(center_of_masses.device_side_buffer),
                                        &(center_of_masses_inner_cells.device_side_buffer),
                                        &(erg.device_side_buffer), &neighbor_size, &start_index,
                                        &end_index, &dir_index, &theta, &cells_start};
                                    auto fut = executor.async_execute(
                                        cudaLaunchKernel<decltype(cuda_p2m_interaction_non_rho)>,
                                        cuda_p2m_interaction_non_rho, grid_spec, threads_per_block,
                                        args, 0);
                                    fut.get();
                                }
                            } else if (size == INX * STENCIL_MAX * STENCIL_MAX) {
                                constexpr size_t buffer_size = INX * STENCIL_MAX * STENCIL_MAX;
                                struct_of_array_data<expansion, real, 20, buffer_size, SOA_PADDING,
                                    std::vector<real, recycler::recycle_allocator_cuda_host<real>>>
                                    local_expansions_staging_area;
                                struct_of_array_data<space_vector, real, 3, buffer_size,
                                    SOA_PADDING,
                                    std::vector<real, recycler::recycle_allocator_cuda_host<real>>>
                                    center_of_masses_staging_area;
                                update_neighbor_input(dir, com_ptr, neighbors, type,
                                    local_expansions_staging_area, center_of_masses_staging_area,
                                    grid_ptr);
                                recycler::cuda_device_buffer<double> local_expansions(
                                    (buffer_size + SOA_PADDING) * 20, device_id);
                                recycler::cuda_device_buffer<double> center_of_masses(
                                    (buffer_size + SOA_PADDING) * 3, device_id);
                                hpx::apply(
                                    static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                                    cudaMemcpyAsync, local_expansions.device_side_buffer,
                                    local_expansions_staging_area.get_pod(),
                                    (buffer_size + SOA_PADDING) * 20 * sizeof(double),
                                    cudaMemcpyHostToDevice);
                                hpx::apply(
                                    static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                                    cudaMemcpyAsync, center_of_masses.device_side_buffer,
                                    center_of_masses_staging_area.get_pod(),
                                    (buffer_size + SOA_PADDING) * 3 * sizeof(double),
                                    cudaMemcpyHostToDevice);

                                if (type == RHO) {
                                    dim3 const grid_spec(cells_end.x - cells_start.x, 1, 1);
                                    dim3 const threads_per_block(1, cells_end.y - cells_start.y, cells_end.z - cells_start.z);
                                    void* args[] = {&(local_expansions.device_side_buffer),
                                        &(center_of_masses.device_side_buffer),
                                        &(center_of_masses_inner_cells.device_side_buffer),
                                        &(erg.device_side_buffer),
                                        &(device_erg_corrs.device_side_buffer), &neighbor_size,
                                        &start_index, &end_index, &dir_index, &theta, &cells_start};
                                    auto fut = executor.async_execute(
                                        cudaLaunchKernel<decltype(cuda_p2m_interaction_rho)>,
                                        cuda_p2m_interaction_rho, grid_spec, threads_per_block,
                                        args, 0);
                                    fut.get();
                                } else {
                                    dim3 const grid_spec(cells_end.x - cells_start.x, 1, 1);
                                    dim3 const threads_per_block(1, cells_end.y - cells_start.y, cells_end.z - cells_start.z);
                                    void* args[] = {&(local_expansions.device_side_buffer),
                                        &(center_of_masses.device_side_buffer),
                                        &(center_of_masses_inner_cells.device_side_buffer),
                                        &(erg.device_side_buffer), &neighbor_size, &start_index,
                                        &end_index, &dir_index, &theta, &cells_start};
                                    auto fut = executor.async_execute(
                                        cudaLaunchKernel<decltype(cuda_p2m_interaction_non_rho)>,
                                        cuda_p2m_interaction_non_rho, grid_spec, threads_per_block,
                                        args, 0);
                                    fut.get();
                                }
                            } else if (size == STENCIL_MAX * STENCIL_MAX * STENCIL_MAX) {
                                constexpr size_t buffer_size =
                                    STENCIL_MAX * STENCIL_MAX * STENCIL_MAX;
                                struct_of_array_data<expansion, real, 20, buffer_size, SOA_PADDING,
                                    std::vector<real, recycler::recycle_allocator_cuda_host<real>>>
                                    local_expansions_staging_area;
                                struct_of_array_data<space_vector, real, 3, buffer_size,
                                    SOA_PADDING,
                                    std::vector<real, recycler::recycle_allocator_cuda_host<real>>>
                                    center_of_masses_staging_area;
                                update_neighbor_input(dir, com_ptr, neighbors, type,
                                    local_expansions_staging_area, center_of_masses_staging_area,
                                    grid_ptr);

                                recycler::cuda_device_buffer<double> local_expansions(
                                    (buffer_size + SOA_PADDING) * 20, device_id);
                                recycler::cuda_device_buffer<double> center_of_masses(
                                    (buffer_size + SOA_PADDING) * 3, device_id);
                                hpx::apply(
                                    static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                                    cudaMemcpyAsync, local_expansions.device_side_buffer,
                                    local_expansions_staging_area.get_pod(),
                                    (buffer_size + SOA_PADDING) * 20 * sizeof(double),
                                    cudaMemcpyHostToDevice);
                                hpx::apply(
                                    static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                                    cudaMemcpyAsync, center_of_masses.device_side_buffer,
                                    center_of_masses_staging_area.get_pod(),
                                    (buffer_size + SOA_PADDING) * 3 * sizeof(double),
                                    cudaMemcpyHostToDevice);

                                if (type == RHO) {
                                    dim3 const grid_spec(cells_end.x - cells_start.x, 1, 1);
                                    dim3 const threads_per_block(1, cells_end.y - cells_start.y, cells_end.z - cells_start.z);
                                    void* args[] = {&(local_expansions.device_side_buffer),
                                        &(center_of_masses.device_side_buffer),
                                        &(center_of_masses_inner_cells.device_side_buffer),
                                        &(erg.device_side_buffer),
                                        &(device_erg_corrs.device_side_buffer), &neighbor_size,
                                        &start_index, &end_index, &dir_index, &theta, &cells_start};
                                    auto fut = executor.async_execute(
                                        cudaLaunchKernel<decltype(cuda_p2m_interaction_rho)>,
                                        cuda_p2m_interaction_rho, grid_spec, threads_per_block,
                                        args, 0);
                                    fut.get();
                                } else {
                                    dim3 const grid_spec(cells_end.x - cells_start.x, 1, 1);
                                    dim3 const threads_per_block(1, cells_end.y - cells_start.y, cells_end.z - cells_start.z);
                                    void* args[] = {&(local_expansions.device_side_buffer),
                                        &(center_of_masses.device_side_buffer),
                                        &(center_of_masses_inner_cells.device_side_buffer),
                                        &(erg.device_side_buffer), &neighbor_size, &start_index,
                                        &end_index, &dir_index, &theta, &cells_start};
                                    auto fut = executor.async_execute(
                                        cudaLaunchKernel<decltype(cuda_p2m_interaction_non_rho)>,
                                        cuda_p2m_interaction_non_rho, grid_spec, threads_per_block,
                                        args, 0);
                                    fut.get();
                                }
                            }
                        }
                    }
                    if (type == RHO) {
                        cuda_angular_result_t angular_corrections_SoA;
                        auto fut = hpx::async(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                            cudaMemcpyAsync, angular_corrections_SoA.get_pod(),
                            device_erg_corrs.device_side_buffer, angular_corrections_size,
                            cudaMemcpyDeviceToHost);
                        fut.get();
                        angular_corrections_SoA.to_non_SoA(grid_ptr->get_L_c());
                        // std::cout << "Cuda version.." << std::endl;
                        // std::cout << std::endl << std::endl;
                        // std::cin.get();
                        // //fut.get();
                        // angular_corrections_SoA.print(std::cout);
                        // std::cin.get();
                    }
                }
                auto fut = hpx::async(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                    cudaMemcpyAsync, potential_expansions_SoA.get_pod(), erg.device_side_buffer,
                    potential_expansions_small_size, cudaMemcpyDeviceToHost);
                //auto fut = hpx::async(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                //    cudaStreamSynchronize);

                // Wait for stream to finish and allow thread to jump away in the meantime
                fut.get();

                // Copy results back into non-SoA array
                potential_expansions_SoA.add_to_non_SoA(grid_ptr->get_L());
                //if (type == RHO && contains_multipole_neighbor) // && opts().p2m_kernel_type == SOA_CUDA)
                 //   angular_corrections_SoA.to_non_SoA(grid_ptr->get_L_c());
            }
        }

    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
