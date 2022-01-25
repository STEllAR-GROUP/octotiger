#pragma once
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
template <class T>
using kokkos_um_pinned_array = Kokkos::View<T*, typename kokkos_um_device_array<T>::array_layout,
    Kokkos::CudaHostPinnedSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_pinned_view =
    recycler::recycled_view<kokkos_um_pinned_array<T>, recycler::recycle_allocator_cuda_host<T>, T>;
#elif defined(KOKKOS_ENABLE_HIP)
template <class T>
using kokkos_um_pinned_array = Kokkos::View<T*, typename kokkos_um_device_array<T>::array_layout,
    Kokkos::Experimental::HIPHostPinnedSpace, Kokkos::MemoryUnmanaged>;
template <class T>
using recycled_pinned_view =
    recycler::recycled_view<kokkos_um_pinned_array<T>, recycler::recycle_allocator_hip_host<T>, T>;
#else
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

template <typename T>
using host_buffer = recycled_pinned_view<T>;
template <typename T>
using device_buffer = recycled_device_view<T>;
template <typename T>
using normal_host_buffer = kokkos_host_array<T>;
template <typename T>
using normal_device_buffer = kokkos_device_array<T>;

// =================================================================================================
// SIMD types
// =================================================================================================

// defines HPX_COMPUTE_HOST_CODE and HPX_COMPUTE_DEVICE_CODE accordingly for the device passes
// useful for picking the correct simd type!
#include <hpx/config/compiler_specific.hpp> 
// SIMD settings
#include <simd.hpp>
using device_simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::scalar>;
using device_simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::scalar>;
#if !defined(HPX_COMPUTE_DEVICE_CODE) && !defined(OCTOTIGER_FORCE_SCALAR_KOKKOS_SIMD)
#if defined(__VSX__)
// NVCC does not play fair with Altivec! See another project with similar issues:
// See https://github.com/dealii/dealii/issues/7328
#ifdef __CUDACC__ // hence: Use scalar when using nvcc
using host_simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::scalar>;
using host_simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::scalar>;
#warning "Using scalar SIMD types"
#else // no nvcc: We can try to use the altivec vectorization
#include <vsx.hpp>
// TODO Actually test with a non-cuda kokkos build and/or clang
// as it should get around the vsx problem
using host_simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::vsx>;
using host_simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::vsx>;
#warning "Using VSX SIMD types"
#endif
#elif defined(__AVX512F__)
#include <avx512.hpp>
using host_simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::avx512>;
using host_simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::avx512>;
#warning "Using AVX512 SIMD types"
#elif defined(__AVX2__) || defined(__AVX__)
#include <avx.hpp>
using host_simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::avx>;
using host_simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::avx>;
#warning "Using AVX SIMD types"
#else
#error "Could not detect any supported SIMD instruction set. Define OCTOTIGER_FORCE_SCALAR_KOKKOS_SIMD to continue anyway (or fix your arch flags if your platform supports AVX)!"
#endif
#else
// drop in for nvcc device pass - is used on host side if FORCE_SCALAR_KOKKOS_SIMD is on
// otherwise only used for compilation
using host_simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::scalar>;
using host_simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::scalar>;
#warning "Using scalar SIMD types"
#endif

#endif
