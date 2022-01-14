//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)

#include "octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp"
#include "octotiger/monopole_interactions/legacy/p2m_interaction_interface.hpp"
#include "octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp"
#include "octotiger/monopole_interactions/util/calculate_stencil.hpp"

#include "octotiger/defs.hpp"
#include "octotiger/options.hpp"

#include <array>
#include <vector>

#include <buffer_manager.hpp>
#if defined(OCTOTIGER_HAVE_CUDA)
#include <cuda_buffer_util.hpp>
#include <cuda_runtime.h>
#elif defined(OCTOTIGER_HAVE_HIP)
#include <hip/hip_runtime.h>
#include <hip_buffer_util.hpp>
#endif
#include <stream_manager.hpp>

#if defined(OCTOTIGER_HAVE_CUDA)
template <typename T>
using device_buffer_t = recycler::cuda_device_buffer<T>;
template <typename T>
using host_buffer_t = std::vector<T, recycler::recycle_allocator_cuda_host<T>>;
using executor_t = hpx::cuda::experimental::cuda_executor;
#elif defined(OCTOTIGER_HAVE_HIP)
template <typename T>
using device_buffer_t = recycler::hip_device_buffer<T>;
template <typename T>
using host_buffer_t = std::vector<T, recycler::recycle_allocator_hip_host<T>>;
using executor_t = hpx::cuda::experimental::cuda_executor;

#define cudaLaunchKernel hipLaunchKernel
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyAsync hipMemcpyAsync

#endif

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {
#ifndef OCTOTIGER_HAVE_HIP
        template <size_t buffer_size>
        inline void run_p2m_kernel(gsolve_type type, double theta, double* device_center_of_masses,
            double* device_local_expansions, double* device_center_of_masses_inner_cells,
            double* device_erg, double* device_erg_corrs,
            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
            multiindex<>& start_index, multiindex<>& end_index, multiindex<>& neighbor_size,
            multiindex<>& dir_index, multiindex<>& cells_start, multiindex<>& cells_end) {
            if (type == RHO) {
                dim3 const grid_spec(cells_end.x - cells_start.x, 1, 1);
                dim3 const threads_per_block(
                    1, cells_end.y - cells_start.y, cells_end.z - cells_start.z);
                void* args[] = {&(device_local_expansions), &(device_center_of_masses),
                    &(device_center_of_masses_inner_cells), &(device_erg), &(device_erg_corrs),
                    &neighbor_size, &start_index, &end_index, &dir_index, &theta, &cells_start};
                // executor.post(cudaLaunchKernel<decltype(cuda_p2m_interaction_rho)>,
                //     cuda_p2m_interaction_rho, grid_spec, threads_per_block, args, 0);
                launch_p2m_rho_cuda_kernel_post(executor, grid_spec, threads_per_block, args);
            } else {
                dim3 const grid_spec(cells_end.x - cells_start.x, 1, 1);
                dim3 const threads_per_block(
                    1, cells_end.y - cells_start.y, cells_end.z - cells_start.z);
                void* args[] = {&(device_local_expansions), &(device_center_of_masses),
                    &(device_center_of_masses_inner_cells), &(device_erg), &neighbor_size,
                    &start_index, &end_index, &dir_index, &theta, &cells_start};
                // executor.post(cudaLaunchKernel<decltype(cuda_p2m_interaction_non_rho)>,
                //     cuda_p2m_interaction_non_rho, grid_spec, threads_per_block, args, 0);
                launch_p2m_non_rho_cuda_kernel_post(executor, grid_spec, threads_per_block, args);
            }
        }
#endif

        cuda_monopole_interaction_interface::cuda_monopole_interaction_interface()
          : monopole_interaction_interface()
          , theta(opts().theta) {}

        void cuda_monopole_interaction_interface::compute_interactions(std::vector<real>& monopoles,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::shared_ptr<grid>& grid_ptr, const bool contains_multipole_neighbor) {
            // Check where we want to run this:
            bool avail = true;
            if (p2p_type != interaction_host_kernel_type::DEVICE_ONLY) {
                // Check where we want to run this:
                avail = stream_pool::interface_available<hpx::cuda::experimental::cuda_executor,
                    pool_strategy>(opts().cuda_buffer_capacity);
            }
#if defined(OCTOTIGER_HAVE_HIP)
            if (contains_multipole_neighbor) // TODO Add DEVICE_ONLY error/warning
              avail = false;
#endif

            if (!avail) {
                // Run CPU implementation
                monopole_interaction_interface::compute_interactions(monopoles, com_ptr, neighbors,
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
                device_buffer_t<double> device_local_monopoles(ENTRIES, device_id);
                device_buffer_t<double> tmp_ergs(NUMBER_P2P_BLOCKS * NUMBER_POT_EXPANSIONS_SMALL, device_id);
                device_buffer_t<double> erg(NUMBER_POT_EXPANSIONS_SMALL, device_id);

                // Move data into staging buffers
                update_input(monopoles, neighbors, type, local_monopoles, 
                    grid_ptr);

                hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                    cudaMemcpyAsync, device_local_monopoles.device_side_buffer,
                    local_monopoles.data(), local_monopoles_size, cudaMemcpyHostToDevice);

                // Launch kernel and queue copying of results
                dim3 const grid_spec(1, NUMBER_P2P_BLOCKS, INX);
                dim3 const threads_per_block(1, INX, INX);
                dim3 const grid_spec_sum(1, 1, INX);
#if defined(OCTOTIGER_HAVE_CUDA)
                void* args[] = {&(device_local_monopoles.device_side_buffer),
                    &(tmp_ergs.device_side_buffer), &theta, &dx};
                // hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                //     cudalaunchkernel<decltype(cuda_p2p_interactions_kernel)>,
                //     cuda_p2p_interactions_kernel, grid_spec, threads_per_block, args, 0);
                // executor.post(cudaLaunchKernel<decltype(cuda_p2p_interactions_kernel)>,
                //  cuda_p2p_interactions_kernel, grid_spec, threads_per_block, args, 0);

                launch_p2p_cuda_kernel_post(executor, grid_spec, threads_per_block, args);
                void* args_sum[] = {&(tmp_ergs.device_side_buffer),
                    &(erg.device_side_buffer), &theta, &dx};
                launch_sum_p2p_results_post(executor, grid_spec_sum, threads_per_block, args_sum);
#elif defined(OCTOTIGER_HAVE_HIP)
                hip_p2p_interactions_kernel_post(executor, grid_spec,
                  threads_per_block, device_local_monopoles.device_side_buffer, tmp_ergs.device_side_buffer, theta, dx);
                hip_sum_p2p_results_post(executor, grid_spec_sum,
                  threads_per_block, tmp_ergs.device_side_buffer, erg.device_side_buffer);
#endif

#ifndef OCTOTIGER_HAVE_HIP
                if (contains_multipole_neighbor) {
                    // Depending on the size of the neighbor there are 3 possible p2m kernels
                    // We need to check how many of which to launch and get appropriate input
                    // buffers. As the struct of arrays datastructure has the size encoded in the type
                    // these input buffers need to be constructured for all three types
                    
                    // TODO Rewrite with less kernel duplication (this requires struct_of_array datatype to
                    // NOT encode its size in the datatype -> larger job than it should be)

                    // Kernel type with INX * INX * STENCIL_MAX elements
                    size_t number_kernel_type1 = 0;
                    constexpr size_t buffer_size_kernel_type1 = INX * INX * STENCIL_MAX;
                    // Kernel type with INX * STENCIL_MAX * STENCIL_MAX elements
                    size_t number_kernel_type2 = 0;
                    constexpr size_t buffer_size_kernel_type2 = INX * STENCIL_MAX * STENCIL_MAX;
                    // Kernel type with STENCIL_MAX * STENCIL_MAX * STENCIL_MAX elements
                    size_t number_kernel_type3 = 0;
                    constexpr size_t buffer_size_kernel_type3 =
                        STENCIL_MAX * STENCIL_MAX * STENCIL_MAX;
                    
                    // Get and reset (as it is recycled) buffer for the angular correction results
                    // Same for all p2m kernel types
                    device_buffer_t<double> device_erg_corrs(NUMBER_ANG_CORRECTIONS);
                    if (type == RHO)
                      hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                          cudaMemsetAsync, device_erg_corrs.device_side_buffer, 0,
                          (INNER_CELLS + SOA_PADDING) * 3 * sizeof(double));
                    // Convert and move inner cells coms to device
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
                    device_buffer_t<double> center_of_masses_inner_cells(
                        (INNER_CELLS + SOA_PADDING) * 3, device_id);
                    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                        cudaMemcpyAsync, center_of_masses_inner_cells.device_side_buffer,
                        center_of_masses_inner_cells_staging_area.get_pod(),
                        (INNER_CELLS + SOA_PADDING) * 3 * sizeof(double), cudaMemcpyHostToDevice);


                    // Check how many p2m kernels of each type we need to launch
                    for (const geo::direction& dir : geo::direction::full_set()) {
                        neighbor_gravity_type& neighbor = neighbors[dir];
                        if (!neighbor.is_monopole && neighbor.data.M) {
                            int size = 1;
                            for (int i = 0; i < 3; i++) {
                                if (dir[i] == 0)
                                    size *= INX;
                                else
                                    size *= STENCIL_MAX;
                            }
                            if (size == buffer_size_kernel_type1) {
                                number_kernel_type1++;
                            } else if (size == buffer_size_kernel_type2) {
                                number_kernel_type2++;
                            } else if (size == buffer_size_kernel_type3) {
                                number_kernel_type3++;
                            }
                        }
                    }
                    // Input kernel buffers for p2m kernels - 2 host-side and 2 device-side buffers
                    // for each of the three kernel types
                    std::vector<struct_of_array_data<expansion, real, 20, buffer_size_kernel_type1,
                        SOA_PADDING,
                        std::vector<real, recycler::recycle_allocator_cuda_host<real>>>>
                        local_expansions_staging_area_type1(number_kernel_type1);
                    std::vector<struct_of_array_data<space_vector, real, 3,
                        buffer_size_kernel_type1, SOA_PADDING,
                        std::vector<real, recycler::recycle_allocator_cuda_host<real>>>>
                        center_of_masses_staging_area_type1(number_kernel_type1);
                    device_buffer_t<double> local_expansions_type1(
                        (buffer_size_kernel_type1 + SOA_PADDING) * 20 * number_kernel_type1 + 32,
                        device_id);
                    device_buffer_t<double> center_of_masses_type1(
                        (buffer_size_kernel_type1 + SOA_PADDING) * 3 * number_kernel_type1 + 32,
                        device_id);
                    // Input buffers for type 2
                    std::vector<struct_of_array_data<expansion, real, 20, buffer_size_kernel_type2,
                        SOA_PADDING,
                        std::vector<real, recycler::recycle_allocator_cuda_host<real>>>>
                        local_expansions_staging_area_type2(number_kernel_type2);
                    std::vector<struct_of_array_data<space_vector, real, 3,
                        buffer_size_kernel_type2, SOA_PADDING,
                        std::vector<real, recycler::recycle_allocator_cuda_host<real>>>>
                        center_of_masses_staging_area_type2(number_kernel_type2);
                    device_buffer_t<double> local_expansions_type2(
                        (buffer_size_kernel_type2 + SOA_PADDING) * 20 * number_kernel_type2 + 32,
                        device_id);
                    device_buffer_t<double> center_of_masses_type2(
                        (buffer_size_kernel_type2 + SOA_PADDING) * 3 * number_kernel_type2 + 32,
                        device_id);
                    // Input buffers for type 3
                    std::vector<struct_of_array_data<expansion, real, 20, buffer_size_kernel_type3,
                        SOA_PADDING,
                        std::vector<real, recycler::recycle_allocator_cuda_host<real>>>>
                        local_expansions_staging_area_type3(number_kernel_type3);
                    std::vector<struct_of_array_data<space_vector, real, 3,
                        buffer_size_kernel_type3, SOA_PADDING,
                        std::vector<real, recycler::recycle_allocator_cuda_host<real>>>>
                        center_of_masses_staging_area_type3(number_kernel_type3);
                    device_buffer_t<double> local_expansions_type3(
                        (buffer_size_kernel_type3 + SOA_PADDING) * 20 * number_kernel_type3 + 32,
                        device_id);
                    device_buffer_t<double> center_of_masses_type3(
                        (buffer_size_kernel_type3 + SOA_PADDING) * 3 * number_kernel_type3 + 32,
                        device_id);

                    // Loop that collects input data for the p2m kernels
                    size_t counter_kernel_type1 = 0;
                    size_t counter_kernel_type2 = 0;
                    size_t counter_kernel_type3 = 0;
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
                            // Convert AoS input into the appropriate SoA datastructure
                            // for the correct p2m kernel size
                            if (size == INX * INX * STENCIL_MAX) {
                                update_neighbor_input(dir, com_ptr, neighbors, type,
                                    local_expansions_staging_area_type1[counter_kernel_type1],
                                    center_of_masses_staging_area_type1[counter_kernel_type1],
                                    grid_ptr, buffer_size_kernel_type1 + SOA_PADDING);
                                counter_kernel_type1++;
                            } else if (size == INX * STENCIL_MAX * STENCIL_MAX) {
                                update_neighbor_input(dir, com_ptr, neighbors, type,
                                    local_expansions_staging_area_type2[counter_kernel_type2],
                                    center_of_masses_staging_area_type2[counter_kernel_type2],
                                    grid_ptr, buffer_size_kernel_type2 + SOA_PADDING);
                                counter_kernel_type2++;
                            } else if (size == STENCIL_MAX * STENCIL_MAX * STENCIL_MAX) {
                                update_neighbor_input(dir, com_ptr, neighbors, type,
                                    local_expansions_staging_area_type3[counter_kernel_type3],
                                    center_of_masses_staging_area_type3[counter_kernel_type3],
                                    grid_ptr, buffer_size_kernel_type3 + SOA_PADDING);
                                counter_kernel_type3++;
                            }
                        }
                    }

                    // Reset type counters for actually running the kernels 
                    counter_kernel_type1 = 0;
                    counter_kernel_type2 = 0;
                    counter_kernel_type3 = 0;
                    // Loop that launches p2m cuda kernels for appropriate neighbors
                    for (const geo::direction& dir : geo::direction::full_set()) {
                        neighbor_gravity_type& neighbor = neighbors[dir];
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
                            // Indices to address the interaction and stencil data
                            multiindex<> start_index = get_padding_start_indices(dir);
                            multiindex<> end_index = get_padding_end_indices(dir);
                            multiindex<> neighbor_size = get_padding_real_size(dir);
                            multiindex<> dir_index;
                            dir_index.x = dir[0];
                            dir_index.y = dir[1];
                            dir_index.z = dir[2];
                            // Save Computation time by only considering cells that actually can change
                            // These are their start and stop indices which are used for the later kernel launch
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

                            // Move data and launch p2m kernel - no synchronization yet
                            // Each branch handles a different p2m kernels type and hence
                            // uses a different input buffer and a different launch size 
                            // Otherwise the launches are identical
                            if (size == INX * INX * STENCIL_MAX) {
                                hpx::apply(
                                    static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                                    cudaMemcpyAsync,
                                    (local_expansions_type1.device_side_buffer) +
                                        counter_kernel_type1 *
                                            (buffer_size_kernel_type1 + SOA_PADDING) * 20,
                                    local_expansions_staging_area_type1[counter_kernel_type1]
                                        .get_pod(),
                                    (buffer_size_kernel_type1 + SOA_PADDING) * 20 * sizeof(double),
                                    cudaMemcpyHostToDevice);
                                hpx::apply(
                                    static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                                    cudaMemcpyAsync,
                                    center_of_masses_type1.device_side_buffer +
                                        counter_kernel_type1 *
                                            (buffer_size_kernel_type1 + SOA_PADDING) * 3,
                                    center_of_masses_staging_area_type1[counter_kernel_type1]
                                        .get_pod(),
                                    (buffer_size_kernel_type1 + SOA_PADDING) * 3 * sizeof(double),
                                    cudaMemcpyHostToDevice);

                                run_p2m_kernel<INX * INX * STENCIL_MAX>(type, theta,
                                    center_of_masses_type1.device_side_buffer +
                                        counter_kernel_type1 *
                                            (buffer_size_kernel_type1 + SOA_PADDING) * 3,
                                    (local_expansions_type1.device_side_buffer) +
                                        counter_kernel_type1 *
                                            (buffer_size_kernel_type1 + SOA_PADDING) * 20,
                                    center_of_masses_inner_cells.device_side_buffer,
                                    erg.device_side_buffer, device_erg_corrs.device_side_buffer,
                                    executor, start_index, end_index, neighbor_size, dir_index,
                                    cells_start, cells_end);

                                counter_kernel_type1++;
                            } else if (size == INX * STENCIL_MAX * STENCIL_MAX) {
                                hpx::apply(
                                    static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                                    cudaMemcpyAsync,
                                    (local_expansions_type2.device_side_buffer) +
                                        counter_kernel_type2 *
                                            (buffer_size_kernel_type2 + SOA_PADDING) * 20,
                                    local_expansions_staging_area_type2[counter_kernel_type2]
                                        .get_pod(),
                                    (buffer_size_kernel_type2 + SOA_PADDING) * 20 * sizeof(double),
                                    cudaMemcpyHostToDevice);
                                hpx::apply(
                                    static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                                    cudaMemcpyAsync,
                                    center_of_masses_type2.device_side_buffer +
                                        counter_kernel_type2 *
                                            (buffer_size_kernel_type2 + SOA_PADDING) * 3,
                                    center_of_masses_staging_area_type2[counter_kernel_type2]
                                        .get_pod(),
                                    (buffer_size_kernel_type2 + SOA_PADDING) * 3 * sizeof(double),
                                    cudaMemcpyHostToDevice);

                                run_p2m_kernel<INX * INX * STENCIL_MAX>(type, theta,
                                    center_of_masses_type2.device_side_buffer +
                                        counter_kernel_type2 *
                                            (buffer_size_kernel_type2 + SOA_PADDING) * 3,
                                    (local_expansions_type2.device_side_buffer) +
                                        counter_kernel_type2 *
                                            (buffer_size_kernel_type2 + SOA_PADDING) * 20,
                                    center_of_masses_inner_cells.device_side_buffer,
                                    erg.device_side_buffer, device_erg_corrs.device_side_buffer,
                                    executor, start_index, end_index, neighbor_size, dir_index,
                                    cells_start, cells_end);
                                counter_kernel_type2++;
                            } else if (size == STENCIL_MAX * STENCIL_MAX * STENCIL_MAX) {
                                hpx::apply(
                                    static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                                    cudaMemcpyAsync,
                                    (local_expansions_type3.device_side_buffer) +
                                        counter_kernel_type3 *
                                            (buffer_size_kernel_type3 + SOA_PADDING) * 20,
                                    local_expansions_staging_area_type3[counter_kernel_type3]
                                        .get_pod(),
                                    (buffer_size_kernel_type3 + SOA_PADDING) * 20 * sizeof(double),
                                    cudaMemcpyHostToDevice);
                                hpx::apply(
                                    static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                                    cudaMemcpyAsync,
                                    center_of_masses_type3.device_side_buffer +
                                        counter_kernel_type3 *
                                            (buffer_size_kernel_type3 + SOA_PADDING) * 3,
                                    center_of_masses_staging_area_type3[counter_kernel_type3]
                                        .get_pod(),
                                    (buffer_size_kernel_type3 + SOA_PADDING) * 3 * sizeof(double),
                                    cudaMemcpyHostToDevice);

                                run_p2m_kernel<INX * INX * STENCIL_MAX>(type, theta,
                                    center_of_masses_type3.device_side_buffer +
                                        counter_kernel_type3 *
                                            (buffer_size_kernel_type3 + SOA_PADDING) * 3,
                                    (local_expansions_type3.device_side_buffer) +
                                        counter_kernel_type3 *
                                            (buffer_size_kernel_type3 + SOA_PADDING) * 20,
                                    center_of_masses_inner_cells.device_side_buffer,
                                    erg.device_side_buffer, device_erg_corrs.device_side_buffer,
                                    executor, start_index, end_index, neighbor_size, dir_index,
                                    cells_start, cells_end);
                                counter_kernel_type3++;
                            }
                        }
                    }

                    // Handle results of both p2p and p2m kernels
                    // as they share their results buffers
                    cuda_angular_result_t angular_corrections_SoA;
                    if (type == RHO) {
                        hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                            cudaMemcpyAsync, angular_corrections_SoA.get_pod(),
                            device_erg_corrs.device_side_buffer, angular_corrections_size,
                            cudaMemcpyDeviceToHost);
                    }
                    auto fut = hpx::async(
                        static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                        cudaMemcpyAsync, potential_expansions_SoA.get_pod(), erg.device_side_buffer,
                        potential_expansions_small_size, cudaMemcpyDeviceToHost);
                    angular_corrections_SoA.to_non_SoA(grid_ptr->get_L_c());
                    fut.get();
                    potential_expansions_SoA.add_to_non_SoA(grid_ptr->get_L());
                    if (type == RHO)
                        angular_corrections_SoA.to_non_SoA(grid_ptr->get_L_c());
                }
#endif

                // Handle results in case we did not need any p2m kernels or used the CPU p2m
                // kernels
                if (!contains_multipole_neighbor) {
                    auto fut = hpx::async(
                        static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                        cudaMemcpyAsync, potential_expansions_SoA.get_pod(), erg.device_side_buffer,
                        potential_expansions_small_size, cudaMemcpyDeviceToHost);
                    // Wait for stream to finish and allow thread to jump away in the meantime
                    fut.get();

                    // Copy results back into non-SoA array
                    potential_expansions_SoA.add_to_non_SoA(grid_ptr->get_L());
                }
            }
        }

    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
