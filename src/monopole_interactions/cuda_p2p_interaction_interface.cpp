//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef OCTOTIGER_HAVE_CUDA
#include "octotiger/common_kernel/kokkos_util.hpp"
#include <Kokkos_Core.hpp>
#include <hpx/kokkos.hpp>

#include "octotiger/monopole_interactions/cuda_p2p_interaction_interface.hpp"
#include "octotiger/monopole_interactions/p2p_cuda_kernel.hpp"
//TODO(daissgr) Remove this header and the source file
#include "octotiger/monopole_interactions/p2p_kokkos_kernel.hpp"
#include "octotiger/monopole_interactions/calculate_stencil.hpp"

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
          , theta(opts().theta) {
            // TODO(daissgr) Replace this
            kokkos_stencil_masks = new bool[14*14*14];
            auto p2p_stencil_pair = monopole_interactions::calculate_stencil();
            auto p2p_stencil_mask_pair =
                monopole_interactions::calculate_stencil_masks(p2p_stencil_pair.first);
            auto p2p_stencil_mask = p2p_stencil_mask_pair.first;
            auto p2p_four_constants = p2p_stencil_mask_pair.second;
            for (auto i = 0; i < FULL_STENCIL_SIZE; ++i) {
                if (p2p_stencil_mask[i]) {
                    kokkos_stencil_masks[i] = true;
                } else {
                    kokkos_stencil_masks[i] = false;
                }
            }
        }

        void cuda_p2p_interaction_interface::compute_p2p_interactions(std::vector<real>& monopoles,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
            std::array<bool, geo::direction::count()>& is_direction_empty) {
            // Check where we want to run this:
            bool avail = stream_pool::interface_available<hpx::cuda::experimental::cuda_executor, pool_strategy>(
                opts().cuda_buffer_capacity);
            bool run_dummy = true;
            if (!avail || p2p_type == interaction_kernel_type::OLD) {
                // Run CPU implementation
                p2p_interaction_interface::compute_p2p_interactions(
                    monopoles, neighbors, type, dx, is_direction_empty);
            } else if (run_dummy) {
                // TODO(daissgr) Use executors for this
                // TODO(daissgr) Move executors into pools and use different streams
                auto host_space =
                    hpx::kokkos::make_execution_space<Kokkos::DefaultHostExecutionSpace>();
                auto stream_space = hpx::kokkos::make_execution_space<Kokkos::Cuda>();

                recycled_pinned_view<double> pinnedMonopoles(NUMBER_LOCAL_MONOPOLE_VALUES);
                recycled_device_view<double> deviceMonopoles(NUMBER_LOCAL_MONOPOLE_VALUES);
                update_input(monopoles, neighbors, type, pinnedMonopoles);

                hpx::kokkos::deep_copy_async(stream_space, deviceMonopoles, pinnedMonopoles);

                recycled_pinned_view<bool> hostmasks(FULL_STENCIL_SIZE);
                recycled_device_view<bool> devicemasks(FULL_STENCIL_SIZE);
                for (auto i = 0; i < FULL_STENCIL_SIZE; i++) {
                    hostmasks[i] = kokkos_stencil_masks[i];
                }
                hpx::kokkos::deep_copy_async(stream_space, devicemasks, hostmasks);

                recycled_pinned_view<double> pinnedResultView(NUMBER_POT_EXPANSIONS_SMALL);
                recycled_device_view<double> deviceResultView(NUMBER_POT_EXPANSIONS_SMALL);

                Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy_1(stream_space, {0, 0, 0}, {8, 8, 8});

                // TODO(daissgr) Rename theta2
                double theta2 = theta;
                constexpr size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;
                Kokkos::parallel_for("kernel p2p", policy_1,
                    [deviceMonopoles, deviceResultView, devicemasks, dx, theta2] CUDA_GLOBAL_METHOD(
                        int idx, int idy, int idz) {
                        const octotiger::fmm::multiindex<> cell_index(
                            idx + INNER_CELLS_PADDING_DEPTH, idy + INNER_CELLS_PADDING_DEPTH,
                            idz + INNER_CELLS_PADDING_DEPTH);
                        octotiger::fmm::multiindex<> cell_index_coarse(cell_index);
                        cell_index_coarse.transform_coarse();
                        const size_t cell_flat_index =
                            octotiger::fmm::to_flat_index_padded(cell_index);
                        octotiger::fmm::multiindex<> cell_index_unpadded(idx, idy, idz);
                        const size_t cell_flat_index_unpadded =
                            octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded);

                        const double theta_rec_squared = (1.0 / theta2) * (1.0 / theta2);
                        const double d_components[2] = {1.0 / dx, -1.0 / dx};
                        double tmpstore[4] = {0.0, 0.0, 0.0, 0.0};

                        // Go through all possible stance elements for the two cells this thread
                        // is responsible for
                        for (int stencil_x = STENCIL_MIN; stencil_x <= STENCIL_MAX; stencil_x++) {
                            int x = stencil_x - STENCIL_MIN;
                            for (int stencil_y = STENCIL_MIN; stencil_y <= STENCIL_MAX;
                                 stencil_y++) {
                                int y = stencil_y - STENCIL_MIN;
                                for (int stencil_z = STENCIL_MIN; stencil_z <= STENCIL_MAX;
                                     stencil_z++) {
                                    // Overall index (required for accessing stencil related
                                    // arrays)
                                    const size_t index = x * STENCIL_INX * STENCIL_INX +
                                        y * STENCIL_INX + (stencil_z - STENCIL_MIN);

                                    if (!devicemasks[index]) {
                                        continue;
                                    }

                                    // partner index
                                    const multiindex<> partner_index1(cell_index.x + stencil_x,
                                        cell_index.y + stencil_y, cell_index.z + stencil_z);
                                    const size_t partner_flat_index1 =
                                        to_flat_index_padded(partner_index1);
                                    multiindex<> partner_index_coarse1(partner_index1);
                                    partner_index_coarse1.transform_coarse();

                                    const double theta_c_rec_squared =
                                        static_cast<double>(distance_squared_reciprocal(
                                            cell_index_coarse, partner_index_coarse1));

                                    const bool mask_b = theta_rec_squared > theta_c_rec_squared;
                                    const double mask = mask_b ? 1.0 : 0.0;

                                    const double r =
                                        std::sqrt(static_cast<double>(stencil_x * stencil_x +
                                            stencil_y * stencil_y + stencil_z * stencil_z));
                                    // const double r = std::sqrt(static_cast<double>(4));

                                    const double r3 = r * r * r;
                                    const double four[4] = {
                                        -1.0 / r, stencil_x / r3, stencil_y / r3, stencil_z / r3};

                                    const double monopole =
                                        deviceMonopoles[partner_flat_index1] * mask * d_components[0];

                                    // Calculate the actual interactions
                                    tmpstore[0] = tmpstore[0] + four[0] * monopole;
                                    tmpstore[1] =
                                        tmpstore[1] + four[1] * monopole * d_components[1];
                                    tmpstore[2] =
                                        tmpstore[2] + four[2] * monopole * d_components[1];
                                    tmpstore[3] =
                                        tmpstore[3] + four[3] * monopole * d_components[1];
                                }
                            }
                        }
                        deviceResultView[cell_flat_index_unpadded] = tmpstore[0];
                        deviceResultView[1 * component_length_unpadded + cell_flat_index_unpadded] =
                            tmpstore[1];
                        deviceResultView[2 * component_length_unpadded + cell_flat_index_unpadded] =
                            tmpstore[2];
                        deviceResultView[3 * component_length_unpadded + cell_flat_index_unpadded] =
                            tmpstore[3];
                    });

                auto fut =
                    hpx::kokkos::deep_copy_async(stream_space, pinnedResultView, deviceResultView);
                fut.get();

                auto org = grid_ptr->get_L();
                auto padded_entries_per_component = NUMBER_POT_EXPANSIONS_SMALL;
                for (size_t component = 0; component < 4; component++) {
                    for (size_t entry = 0; entry < org.size(); entry++) {
                        org[entry][component] += pinnedResultView[component * padded_entries_per_component
                        + entry];
                    }
                }

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
                update_input(monopoles, neighbors, type, local_monopoles);

                hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                    cudaMemcpyAsync, device_local_monopoles.device_side_buffer,
                    local_monopoles.data(), local_monopoles_size, cudaMemcpyHostToDevice);

                // Launch kernel and queue copying of results
                dim3 const grid_spec(INX / 2, 1, 1);
                dim3 const threads_per_block(1, INX, INX);
                void* args[] = {&(device_local_monopoles.device_side_buffer),
                    &(erg.device_side_buffer), &theta, &dx};
                // hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                //     cudalaunchkernel<decltype(cuda_p2p_interactions_kernel)>,
                //     cuda_p2p_interactions_kernel, grid_spec, threads_per_block, args, 0);
                executor.post(
                    cudaLaunchKernel<decltype(cuda_p2p_interactions_kernel)>,
                    cuda_p2p_interactions_kernel, grid_spec, threads_per_block, args, 0);
                auto fut = hpx::async(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                    cudaMemcpyAsync, potential_expansions_SoA.get_pod(), erg.device_side_buffer,
                    potential_expansions_small_size, cudaMemcpyDeviceToHost);

                // Wait for stream to finish and allow thread to jump away in the meantime
                fut.get();

                // Copy results back into non-SoA array
                potential_expansions_SoA.add_to_non_SoA(grid_ptr->get_L());
            }
        }

    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
