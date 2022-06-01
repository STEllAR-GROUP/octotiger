#pragma once
#include <array>
#include <cmath>
#include "octotiger/cuda_util/cuda_global_def.hpp"
#include "simd_common.hpp"

// =================================================================================================
// SIMD types
// =================================================================================================

// defines HPX_COMPUTE_HOST_CODE and HPX_COMPUTE_DEVICE_CODE accordingly for the device passes
// useful for picking the correct simd type!
#include <hpx/config/compiler_specific.hpp> 
// SIMD settings
#if defined(OCTOTIGER_HAVE_STD_EXPERIMENTAL_SIMD)
#include "octotiger/common_kernel/std_simd.hpp"
#pragma message "Using std-experimental-simd SIMD types"
#else
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

// =================================================================================================
// SIMD functions with fallback  
// =================================================================================================

namespace simd_fallbacks {
namespace detail {
// traits that check for overloads using the current simd_t

// should consider the SIMD_NAMESPACE for overloads due to Argument-dependent name lookup
template <class simd_t, class = void>
struct has_simd_sqrt : std::false_type
{
};
template <class simd_t>
struct has_simd_sqrt<simd_t, std::void_t<decltype(sqrt(std::declval<simd_t>()))>>
  : std::true_type
{
};
template <class simd_t, class = void>
struct has_simd_pow : std::false_type
{
};
template<class simd_t>
struct has_simd_pow<simd_t,
  std::void_t<decltype(pow(std::declval<simd_t>(),std::declval<double>()))>>
  : std::true_type
{
};
template<class simd_t, class=void>
struct has_simd_asinh : std::false_type {};
template <class simd_t>
struct has_simd_asinh<simd_t, std::void_t<decltype(asinh(std::declval<simd_t>()))>>
  : std::true_type
{
};
template<class simd_t, class=void>
struct has_simd_copysign : std::false_type {};
template <class simd_t>
struct has_simd_copysign<simd_t,
    std::void_t<decltype(copysign(std::declval<simd_t>(), std::declval<simd_t>()))>>
  : std::true_type
{
};
} // end namespace detail

template <typename simd_t>
CUDA_GLOBAL_METHOD inline simd_t sqrt_with_serial_fallback(const simd_t input) {
  if constexpr (detail::has_simd_sqrt<simd_t>::value) {
    // should consider the SIMD_NAMESPACE for overloads due to Argument-dependent name lookup
    return sqrt(input);
  } else {
    static_assert(!std::is_same<simd_t, simd_t>::value, "Using sqrt serial fallback!"
        "This will impact If this is intentional please remove this static assert for your build");
    std::array<double, simd_t::size()> sqrt_helper;
    std::array<double, simd_t::size()> sqrt_helper_input;
    input.copy_to(sqrt_helper_input.data(), SIMD_NAMESPACE::element_aligned_tag{});
    for (auto i = 0; i < simd_t::size(); i++) {
      sqrt_helper[i] = std::sqrt(sqrt_helper_input[i]);
    }
    return simd_t{sqrt_helper.data(), SIMD_NAMESPACE::element_aligned_tag{}};
  }
}
template <typename simd_t>
CUDA_GLOBAL_METHOD inline simd_t pow_with_serial_fallback(const simd_t input, const double exponent) {
  if constexpr (detail::has_simd_pow<simd_t>::value) {
    // should consider the SIMD_NAMESPACE for overloads due to argument-dependent name lookup
    return pow(input, exponent);
  } else {
    /* static_assert(!std::is_same<simd_t, simd_t>::value, "Using pow serial fallback! " */
    /*     "If this is intentional please remove this static assert for your build"); */
    std::array<double, simd_t::size()> pow_helper;
    std::array<double, simd_t::size()> pow_helper_input;
    input.copy_to(pow_helper_input.data(), SIMD_NAMESPACE::element_aligned_tag{});
    for (auto i = 0; i < simd_t::size(); i++) {
      pow_helper[i] = std::pow(pow_helper_input[i], exponent);
    }
    return simd_t{pow_helper.data(), SIMD_NAMESPACE::element_aligned_tag{}};
  }
}
template <typename simd_t>
CUDA_GLOBAL_METHOD inline simd_t asinh_with_serial_fallback(const simd_t input) {
  if constexpr (detail::has_simd_asinh<simd_t>::value) {
    // should consider the SIMD_NAMESPACE for overloads due to argument-dependent name lookup
    return asinh(input);
  } else {
    /* static_assert(!std::is_same<simd_t, simd_t>::value, "Using asinh serial fallback!" */
    /*     "If this is intentional please remove this static assert for your build"); */
    std::array<double, simd_t::size()> asinh_helper;
    std::array<double, simd_t::size()> asinh_helper_input;
    input.copy_to(asinh_helper_input.data(), SIMD_NAMESPACE::element_aligned_tag{});
    for (auto i = 0; i < simd_t::size(); i++) {
      asinh_helper[i] = std::asinh(asinh_helper_input[i]);
    }
    return simd_t{asinh_helper.data(), SIMD_NAMESPACE::element_aligned_tag{}};
  }
}
template <typename simd_t>
CUDA_GLOBAL_METHOD inline simd_t copysign_with_serial_fallback(const simd_t input1, const simd_t input2) {
  if constexpr (detail::has_simd_copysign<simd_t>::value) {
    // should consider the SIMD_NAMESPACE for overloads due to argument-dependent name lookup
    return copysign(input1, input2);
  } else {
    /* static_assert(!std::is_same<simd_t, simd_t>::value, "Using asinh serial fallback!" */
    /*     "If this is intentional please remove this static assert for your build"); */
    std::array<double, simd_t::size()> copysign_helper;
    std::array<double, simd_t::size()> copysign_helper_input1;
    std::array<double, simd_t::size()> copysign_helper_input2;
    input1.copy_to(copysign_helper_input1.data(), SIMD_NAMESPACE::element_aligned_tag{});
    input2.copy_to(copysign_helper_input2.data(), SIMD_NAMESPACE::element_aligned_tag{});
    for (auto i = 0; i < simd_t::size(); i++) {
      copysign_helper[i] = std::copysign(copysign_helper_input1[i], copysign_helper_input2[i]);
    }
    return simd_t{copysign_helper.data(), SIMD_NAMESPACE::element_aligned_tag{}};
  }
}
} // end namespace simd_fallbacks
