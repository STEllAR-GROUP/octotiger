//  Copyright (c) 2020-2022 Gregor Daiß
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#pragma once
#include <aggregation_manager.hpp>
#include <hpx/kokkos/executors.hpp>
#ifdef OCTOTIGER_HAVE_KOKKOS
//#define KOKKOS_OPT_RANGE_AGGRESSIVE_VECTORIZATION
#include <Kokkos_Core.hpp>
#include <hpx/kokkos.hpp>

#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/common_kernel/struct_of_array_data.hpp"


// ============================================================ 
// Executor types / helpers
// ============================================================ 
#include <stream_manager.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
using kokkos_device_executor = hpx::kokkos::cuda_executor;
#elif defined(KOKKOS_ENABLE_HIP)
using kokkos_device_executor = hpx::kokkos::hip_executor;
#endif

template <typename T>
struct always_false
{
    enum
    {
        value = false
    };
};
template <class T>
struct is_kokkos_host_executor
  : std::integral_constant<bool,
        std::is_same<hpx::kokkos::serial_executor, typename std::remove_cv<T>::type>::value ||
            std::is_same<hpx::kokkos::hpx_executor, typename std::remove_cv<T>::type>::value>
{};

template <class T>
struct is_kokkos_device_executor
  : std::integral_constant<bool,
#ifdef KOKKOS_ENABLE_CUDA 
        std::is_same<hpx::kokkos::cuda_executor, typename std::remove_cv<T>::type>::value>
#elif defined(KOKKOS_ENABLE_HIP)
        std::is_same<hpx::kokkos::hip_executor, typename std::remove_cv<T>::type>::value>
#else
        false>
#endif
{};

// ============================================================ 
// Buffer defines / types
// ============================================================ 

#include <kokkos_buffer_util.hpp>
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_buffer_util.hpp>
#endif
#ifdef KOKKOS_ENABLE_HIP
#include <hip_buffer_util.hpp>
#endif

#if defined(KOKKOS_ENABLE_CUDA)
template <class T>
using kokkos_um_device_array = Kokkos::View<T*, Kokkos::CudaSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using kokkos_device_array = Kokkos::View<T*, Kokkos::CudaSpace>;
template <class T>
using recycled_device_view = recycler::recycled_view<kokkos_um_device_array<T>,
    recycler::recycle_allocator_cuda_device<T>, T>;
#elif defined(KOKKOS_ENABLE_HIP)
template <class T>
using kokkos_um_device_array = Kokkos::View<T*, Kokkos::Experimental::HIPSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using kokkos_device_array = Kokkos::View<T*, Kokkos::Experimental::HIPSpace>;
template <class T>
using recycled_device_view = recycler::recycled_view<kokkos_um_device_array<T>,
    recycler::recycle_allocator_hip_device<T>, T>;
#else
// Fallback without cuda
template <class T>
using kokkos_um_device_array = Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using kokkos_device_array = Kokkos::View<T*, Kokkos::HostSpace>;
template <class T>
using recycled_device_view = recycler::recycled_view<kokkos_um_device_array<T>,
    recycler::recycle_std<T>, T>;
#endif


// NOTE: Must use the same layout to be able to use e.g. cudaMemcpyAsync
template <class T>
using kokkos_um_array = Kokkos::View<T*, typename kokkos_um_device_array<T>::array_layout,
    Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using kokkos_host_array =
    Kokkos::View<T*, typename kokkos_device_array<T>::array_layout, Kokkos::HostSpace>;
template <class T>
using recycled_host_view = recycler::recycled_view<kokkos_um_array<T>, recycler::recycle_std<T>, T>;

// NOTE: Must use the same layout to be able to use e.g. cudaMemcpyAsync
#if defined(KOKKOS_ENABLE_CUDA)
template <typename T>
using kokkos_host_allocator = recycler::detail::cuda_pinned_allocator<T>;
template <typename T>
using kokkos_device_allocator = recycler::detail::cuda_device_allocator<T>;
template <class T>
using kokkos_um_pinned_array = Kokkos::View<T*, typename kokkos_um_device_array<T>::array_layout,
    Kokkos::CudaHostPinnedSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_pinned_view =
    recycler::recycled_view<kokkos_um_pinned_array<T>, recycler::recycle_allocator_cuda_host<T>, T>;
#elif defined(KOKKOS_ENABLE_HIP)
template <typename T>
using kokkos_host_allocator = recycler::detail::hip_pinned_allocator<T>;
template <typename T>
using kokkos_device_allocator = recycler::detail::hip_device_allocator<T>;
template <class T>
using kokkos_um_pinned_array = Kokkos::View<T*, typename kokkos_um_device_array<T>::array_layout,
    Kokkos::Experimental::HIPHostPinnedSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_pinned_view =
    recycler::recycled_view<kokkos_um_pinned_array<T>, recycler::recycle_allocator_hip_host<T>, T>;
#else
template <typename T>
using kokkos_host_allocator = std::allocator<T>;
template <typename T>
using kokkos_device_allocator = std::allocator<T>;
template <class T>
using kokkos_um_pinned_array = Kokkos::View<T*, typename kokkos_um_device_array<T>::array_layout,
    Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_pinned_view =
    recycler::recycled_view<kokkos_um_pinned_array<T>, recycler::recycle_std<T>, T>;
#endif

template <typename Executor, typename ViewType>
auto get_iteration_policy(const Executor&& executor, const ViewType& view_to_iterate) {
    return get_iteration_policy(executor, view_to_iterate);
}

template <typename executor_t>
inline void sync_kokkos_host_kernel(executor_t& exec) {
    exec.instance().fence();    // All kokkos executor should support this
}
template <>
inline void sync_kokkos_host_kernel(hpx::kokkos::hpx_executor& exec) {
    exec.instance().fence();    // All kokkos executor should support this
    //auto fut = exec.instance().impl_get_future();
    //fut.get();
}

template <class T, typename executor_t>
using agg_recycled_host_view =
    recycler::aggregated_recycled_view<kokkos_um_pinned_array<T>, Allocator_Slice<T, kokkos_host_allocator<T>, executor_t>, T>;
template <typename T, typename executor_t>
using aggregated_host_buffer = agg_recycled_host_view<T, executor_t>;

template <class T, typename executor_t>
using agg_recycled_device_view =
    recycler::aggregated_recycled_view<kokkos_um_device_array<T>, Allocator_Slice<T, kokkos_device_allocator<T>, executor_t>, T>;
template <typename T, typename executor_t>
using aggregated_device_buffer = agg_recycled_device_view<T, executor_t>;

template <typename T>
using host_buffer = recycled_pinned_view<T>;
template <typename T>
using device_buffer = recycled_device_view<T>;
template <typename T>
using normal_host_buffer = kokkos_host_array<T>;
template <typename T>
using normal_device_buffer = kokkos_device_array<T>;

#endif
