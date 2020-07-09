#ifdef OCTOTIGER_HAVE_CUDA
#include <iostream>

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
        void kokkos_p2p_interactions(std::vector<real, recycler::recycle_std<real>>& buffer,
            std::vector<real, recycler::recycle_std<real>>& output) {
            recycled_pinned_view<double> pinnedView(NUMBER_LOCAL_MONOPOLE_VALUES);
            recycled_device_view<double> deviceView(NUMBER_LOCAL_MONOPOLE_VALUES);

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

            Kokkos::parallel_for(
                "kernel p2p", policy_1, KOKKOS_LAMBDA(int x, int y, int z) {
                    int index = x * 8 * 8 + y * 8 + z;
                    deviceResultView[index] = deviceView[index] + x + y + z;
                });

            auto fut = hpx::kokkos::deep_copy_async(stream_space, pinnedResultView, deviceResultView);
            fut.get();
        }

    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
