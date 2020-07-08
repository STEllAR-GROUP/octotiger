#ifdef OCTOTIGER_HAVE_CUDA
#include <iostream>

#include <Kokkos_Core.hpp>
#include <hpx/kokkos.hpp>

#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/monopole_interactions/monopole_kernel_templates.hpp"
#include "octotiger/monopole_interactions/p2p_kokkos_kernel.hpp"

#include "../include/buffer_manager.hpp"
#include "../include/cuda_buffer_util.hpp"
#include "../include/kokkos_buffer_util.hpp"

constexpr size_t view_size_0 = 10;
constexpr size_t view_size_1 = 50;
using type_in_view = float[view_size_1][view_size_0];
constexpr size_t view_size = view_size_0 * view_size_1;
using kokkos_array = Kokkos::View<type_in_view, Kokkos::HostSpace>;

template <class T>
using kokkos_um_device_array = Kokkos::View<T**, Kokkos::CudaSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_device_view = recycler::recycled_view<kokkos_um_device_array<T>,
    recycler::recycle_allocator_cuda_device<T>, T>;

// NOTE: Must use the same layout to be able to use e.g. cudaMemcpyAsync
template <class T>
using kokkos_um_array = Kokkos::View<T**, typename kokkos_um_device_array<T>::array_layout,
    Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_host_view = recycler::recycled_view<kokkos_um_array<T>, recycler::recycle_std<T>, T>;

// NOTE: Must use the same layout to be able to use e.g. cudaMemcpyAsync
template <class T>
using kokkos_um_pinned_array = Kokkos::View<T**, typename kokkos_um_device_array<T>::array_layout,
    Kokkos::CudaHostPinnedSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_pinned_view =
    recycler::recycled_view<kokkos_um_pinned_array<T>, recycler::recycle_allocator_cuda_host<T>, T>;

template <typename Executor, typename ViewType>
auto get_iteration_policy(const Executor&& executor, const ViewType& view_to_iterate) {
    return get_iteration_policy(executor, view_to_iterate);
}

template <typename Viewtype, typename Policytype>
KOKKOS_INLINE_FUNCTION void
kernel_add_kokkos(const Viewtype &first, const Viewtype &second,
                  Viewtype &output, const Policytype &policy) {
  Kokkos::parallel_for(
      "kernel add", policy, KOKKOS_LAMBDA(int j, int k) {
        // useless loop to make the computation last longer in the profiler
        for (volatile double i = 0.; i < 100.;) {
          ++i;
        }
        output(j, k) = first(j, k) + second(j, k);
      });
}

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {
        void kokkos_sample_kernel() {
            std::cout << "Starting kokkos test!!" << std::endl;
            const int numIterations = 40;
            static double d = 0;
            ++d;
            double t = d;

            recycled_host_view<double> hostView(view_size_0, view_size_1);
            recycled_pinned_view<double> pinnedView(view_size_0, view_size_1);
            recycled_device_view<double> deviceView(view_size_0, view_size_1);

            {
                auto host_space =
                    hpx::kokkos::make_execution_space<Kokkos::DefaultHostExecutionSpace>();
                auto policy_host = get_iteration_policy(host_space, pinnedView);

                auto copy_finished = hpx::kokkos::parallel_for_async(
                    "pinned host init", policy_host, KOKKOS_LAMBDA(int n, int o) {
                        hostView(n, o) = t;
                        pinnedView(n, o) = hostView(n, o);
                    });

                // auto stream_space = hpx::kokkos::make_execution_space();
                auto stream_space = hpx::kokkos::make_execution_space<Kokkos::Cuda>();
                auto policy_stream = get_iteration_policy(stream_space, pinnedView);

                // TODO(pollinta): How to make a nice continuation from HPX future to CUDA
                // stream (i.e. without using wait)?
                copy_finished.wait();

                // All of the following deep copies and kernels are sequenced because they
                // use the same instance. It is enough to wait for the last future. // The
                // views must have compatible layouts to actually use cudaMemcpyAsync.
                hpx::kokkos::deep_copy_async(stream_space, deviceView, pinnedView);

                {
                    // auto totalTimer = scoped_timer("async device");
                    for (int i = 0; i < numIterations; ++i) {
                        kernel_add_kokkos(deviceView, deviceView, deviceView, policy_stream);
                    }
                }

                hpx::kokkos::deep_copy_async(stream_space, pinnedView, deviceView);
                hpx::kokkos::deep_copy_async(stream_space, hostView, pinnedView).wait();

                // test values in hostView
                // printf("%f %f hd ", hostView.data()[0], t);
                assert(std::abs(hostView.data()[0] -
                           t * (static_cast<unsigned long>(1) << numIterations)) < 1e-6);
            }
        }
    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
