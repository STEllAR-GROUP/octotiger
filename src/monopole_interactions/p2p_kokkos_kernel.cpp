#ifdef OCTOTIGER_HAVE_CUDA
#include <iostream>

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>
#include <hpx/kokkos.hpp>

#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/monopole_interactions/monopole_kernel_templates.hpp"
#include "octotiger/monopole_interactions/p2p_kokkos_kernel.hpp"

#include <cuda_buffer_util.hpp>
#include <kokkos_buffer_util.hpp>
#include <stream_manager.hpp>

template <class T>
using kokkos_um_device_array = Kokkos::View<T*, Kokkos::CudaSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_device_view = recycler::recycled_view<kokkos_um_device_array<T>,
    recycler::recycle_allocator_cuda_device<T>, T>;

// NOTE: Must use the same layout to be able to use e.g. cudaMemcpyAsync
template <class T>
using kokkos_um_array = Kokkos::View<T*, typename kokkos_um_device_array<T>::array_layout,
    Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_host_view = recycler::recycled_view<kokkos_um_array<T>, recycler::recycle_std<T>, T>;

// NOTE: Must use the same layout to be able to use e.g. cudaMemcpyAsync
template <class T>
using kokkos_um_pinned_array = Kokkos::View<T*, typename kokkos_um_device_array<T>::array_layout,
    Kokkos::CudaHostPinnedSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_pinned_view =
    recycler::recycled_view<kokkos_um_pinned_array<T>, recycler::recycle_allocator_cuda_host<T>, T>;

template <typename Executor, typename ViewType>
auto get_iteration_policy(const Executor&& executor, const ViewType& view_to_iterate) {
    return get_iteration_policy(executor, view_to_iterate);
}

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {
        const size_t component_length_unpadded = INNER_CELLS + SOA_PADDING;
        void kokkos_p2p_interactions(std::vector<real, recycler::recycle_std<real>>& buffer,
            std::vector<real, recycler::recycle_std<real>>& output, const double dx,
            const double theta, bool *masks) {
            recycled_pinned_view<double> pinnedView(NUMBER_LOCAL_MONOPOLE_VALUES);
            recycled_device_view<double> deviceView(NUMBER_LOCAL_MONOPOLE_VALUES);
            for (auto i = 0; i < NUMBER_LOCAL_MONOPOLE_VALUES; i++) {
                pinnedView[i] = buffer[i];
            }
            recycled_pinned_view<bool> hostmasks(FULL_STENCIL_SIZE);
            recycled_device_view<bool> devicemasks(FULL_STENCIL_SIZE);
            for (auto i = 0; i < FULL_STENCIL_SIZE; i++) {
                hostmasks[i] = masks[i];
            }

            recycled_pinned_view<double> pinnedResultView(NUMBER_POT_EXPANSIONS_SMALL);
            recycled_device_view<double> deviceResultView(NUMBER_POT_EXPANSIONS_SMALL);
            auto host_space =
                hpx::kokkos::make_execution_space<Kokkos::DefaultHostExecutionSpace>();
            // auto policy_host = get_iteration_policy(host_space, pinnedView);

            // auto stream_space = hpx::kokkos::make_execution_space();
            auto stream_space = hpx::kokkos::make_execution_space<Kokkos::Cuda>();
            // auto policy_stream = get_iteration_policy(stream_space, pinnedView);
            Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy_1(stream_space, {0, 0, 0}, {8, 8, 8});
            // Kokkos:::MDRangePolicy<Rank<3>> policy_1({0,0,0},{8,8,8});

            hpx::kokkos::deep_copy_async(stream_space, deviceView, pinnedView);
            hpx::kokkos::deep_copy_async(stream_space, devicemasks, hostmasks);

            Kokkos::parallel_for("kernel p2p", policy_1,
                [deviceView, deviceResultView, devicemasks, dx, theta] CUDA_GLOBAL_METHOD(int idx, int idy, int idz) {
                    const octotiger::fmm::multiindex<> cell_index(idx + INNER_CELLS_PADDING_DEPTH,
                        idy + INNER_CELLS_PADDING_DEPTH, idz + INNER_CELLS_PADDING_DEPTH);
                    octotiger::fmm::multiindex<> cell_index_coarse(cell_index);
                    cell_index_coarse.transform_coarse();
                    const size_t cell_flat_index = octotiger::fmm::to_flat_index_padded(cell_index);
                    octotiger::fmm::multiindex<> cell_index_unpadded(idx, idy, idz);
                    const size_t cell_flat_index_unpadded =
                        octotiger::fmm::to_inner_flat_index_not_padded(cell_index_unpadded);

                    const double theta_rec_squared = (1.0 / theta) * (1.0 / theta);
                    const double d_components[2] = {1.0 / dx, -1.0 / dx};
                    double tmpstore[4] = {0.0, 0.0, 0.0, 0.0};

                    // Go through all possible stance elements for the two cells this thread
                    // is responsible for
                    for (int stencil_x = STENCIL_MIN; stencil_x <= STENCIL_MAX; stencil_x++) {
                        int x = stencil_x - STENCIL_MIN;
                        for (int stencil_y = STENCIL_MIN; stencil_y <= STENCIL_MAX; stencil_y++) {
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
                                    deviceView[partner_flat_index1] * mask * d_components[0];

                                // Calculate the actual interactions
                                tmpstore[0] = tmpstore[0] + four[0] * monopole;
                                tmpstore[1] = tmpstore[1] + four[1] * monopole * d_components[1];
                                tmpstore[2] = tmpstore[2] + four[2] * monopole * d_components[1];
                                tmpstore[3] = tmpstore[3] + four[3] * monopole * d_components[1];
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

            for (auto i = 0; i < NUMBER_POT_EXPANSIONS_SMALL; i++) {
                output[i] = pinnedResultView[i];
            }
        }

    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
