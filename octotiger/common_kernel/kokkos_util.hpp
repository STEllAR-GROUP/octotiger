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
using kokkos_host_array =
    Kokkos::View<T*, typename kokkos_device_array<T>::array_layout, Kokkos::HostSpace>;
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

template <typename executor_t>
inline void sync_kokkos_host_kernel(executor_t& exec) {
    exec.instance().fence();    // All kokkos executor should support this
}
template <>
inline void sync_kokkos_host_kernel(hpx::kokkos::hpx_executor& exec) {
    auto fut = exec.instance().impl_get_future();
    fut.get();
}

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
        std::is_same<hpx::kokkos::cuda_executor, typename std::remove_cv<T>::type>::value>
{};

template <typename T>
using host_buffer = recycled_pinned_view<T>;
template <typename T>
using device_buffer = recycled_device_view<T>;
template <typename T>
using normal_host_buffer = kokkos_host_array<T>;
template <typename T>
using normal_device_buffer = kokkos_device_array<T>;

// defines HPX_COMPUTE_HOST_CODE and HPX_COMPUTE_DEVICE_CODE accordingly for the device passes
// useful for picking the correct simd type!
#include <hpx/config/compiler_specific.hpp> 
// SIMD settings
#include <simd.hpp>
using device_simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::scalar>;
using device_simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::scalar>;
#if defined(HPX_COMPUTE_HOST_CODE)
#if defined(__VSX__)
// NVCC does not play fair with Altivec! See another project with similar issues:
// See https://github.com/dealii/dealii/issues/7328
#ifdef __CUDACC__ // hence: Use scalar when using nvcc
// TODO Does this work with clang?!
using host_simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::scalar>;
using host_simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::scalar>;
#else // no nvcc: We can try to use the altivec vectorization
#include <vsx.hpp>
// TODO Actually test with a non-cuda kokkos build
using host_simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::vsx>;
using host_simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::vsx>;
#endif
#elif defined(__AVX512F__)
#include <avx512.hpp>
using host_simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::avx512>;
using host_simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::avx512>;
#elif defined(__AVX2__) || defined(__AVX__)
#include <avx.hpp>
using host_simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::avx>;
using host_simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::avx>;
#endif
//using host_simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::scalar>;
//using host_simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::scalar>;
#else
// drop in for nvcc device pass - shouldnt be used on host code but required for compilation
using host_simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::scalar>;
using host_simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::scalar>;
#endif

#endif
