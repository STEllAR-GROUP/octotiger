#pragma once
#ifdef OCTOTIGER_HAVE_KOKKOS
//#define KOKKOS_OPT_RANGE_AGGRESSIVE_VECTORIZATION 
#include <Kokkos_Core.hpp>
#include <hpx/kokkos.hpp>

#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/common_kernel/struct_of_array_data.hpp"

#include <kokkos_buffer_util.hpp>
#include <stream_manager.hpp>

#ifdef OCTOTIGER_HAVE_CUDA
#include <cuda_buffer_util.hpp>

template <class T>
using kokkos_um_device_array = Kokkos::View<T*, Kokkos::CudaSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using kokkos_device_array = Kokkos::View<T*, Kokkos::CudaSpace>;
template <class T>
using recycled_device_view = recycler::recycled_view<kokkos_um_device_array<T>,
    recycler::recycle_allocator_cuda_device<T>, T>;

// NOTE: Must use the same layout to be able to use e.g. cudaMemcpyAsync
template <class T>
using kokkos_um_array = Kokkos::View<T*, typename kokkos_um_device_array<T>::array_layout,
    Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using kokkos_host_array = Kokkos::View<T*, typename kokkos_device_array<T>::array_layout,
    Kokkos::HostSpace>;
template <class T>
using recycled_host_view = recycler::recycled_view<kokkos_um_array<T>, recycler::recycle_std<T>, T>;

// NOTE: Must use the same layout to be able to use e.g. cudaMemcpyAsync
template <class T>
using kokkos_um_pinned_array = Kokkos::View<T*, typename kokkos_um_device_array<T>::array_layout,
    Kokkos::CudaHostPinnedSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_pinned_view =
    recycler::recycled_view<kokkos_um_pinned_array<T>, recycler::recycle_allocator_cuda_host<T>, T>;
#endif

template <typename Executor, typename ViewType>
auto get_iteration_policy(const Executor&& executor, const ViewType& view_to_iterate) {
    return get_iteration_policy(executor, view_to_iterate);
}

template< typename T >
struct always_false { 
    enum { value = false };  
};


template <typename T>
using host_buffer = recycled_pinned_view<T>;
template <typename T>
using device_buffer = recycled_device_view<T>;
template <typename T>
using normal_host_buffer = kokkos_host_array<T>;
template <typename T>
using normal_device_buffer = kokkos_device_array<T>;

#endif