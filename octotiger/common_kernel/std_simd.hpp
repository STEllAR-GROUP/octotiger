//  Copyright (c) 2022 Srinivas Yadav
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#pragma once

#include <hpx/config/compiler_specific.hpp>


#if !defined(HPX_COMPUTE_DEVICE_CODE) && !defined(OCTOTIGER_KOKKOS_SIMD_SCALAR)
#if defined(OCTOTIGER_KOKKOS_SIMD_AUTOMATIC_DISCOVERY)
#include <experimental/simd>
namespace SIMD_NAMESPACE = std::experimental;
using host_simd_t = std::experimental::native_simd<double>;
using host_simd_mask_t = std::experimental::native_simd_mask<double>;
#elif defined(OCTOTIGER_KOKKOS_SIMD_AVX512)
#ifndef __AVX512F__
#error "AVX512 Kokkos kernels are specified explicitly but build is without AVX512 support."
#endif
#include <experimental/simd>
namespace SIMD_NAMESPACE = std::experimental;
using host_simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::__avx512>;
using host_simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::__avx512>;
static_assert(host_simd_t::size() == 8);
static_assert(host_simd_mask_t::size() == 8);
#elif defined(OCTOTIGER_KOKKOS_SIMD_AVX)
#if !(defined(__AVX2__) || defined(__AVX__))
#error "AVX Kokkos kernels are specified explicitly but build is without AVX support."
#endif
#include <experimental/simd>
namespace SIMD_NAMESPACE = std::experimental;
using host_simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::__avx>;
using host_simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::__avx>;
static_assert(host_simd_t::size() == 4);
static_assert(host_simd_mask_t::size() == 4);
#elif defined(OCTOTIGER_KOKKOS_SIMD_NEON)
#if !(defined(__ARM_NEON))
#error "NEON Kokkos kernels are specified explicitly but build is without NEON support."
#endif
#include <experimental/simd>
namespace SIMD_NAMESPACE = std::experimental;
using host_simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::__neon>;
using host_simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::__neon>;
static_assert(host_simd_t::size() == 2);
static_assert(host_simd_mask_t::size() == 2);
#elif defined(OCTOTIGER_KOKKOS_SIMD_SVE)
// Use custom sve types from Srinivas
#include <sve/sve.hpp> 
#warning "SVE build"
namespace SIMD_NAMESPACE = sve::experimental::parallelism_v2;
using host_simd_t = sve::experimental::native_simd<double>;
using host_simd_mask_t = sve::experimental::native_simd_mask<double>;
static_assert(host_simd_t::size() == 8 || host_simd_t::size() == 4 || host_simd_t::size() == 2);
static_assert(host_simd_mask_t::size() == 8 || host_simd_mask_t::size() == 4 ||
    host_simd_mask_t::size() == 2);
#elif defined(OCTOTIGER_KOKKOS_SIMD_VSX)
#if !(defined(__VSX__)
#error "VSX Kokkos kernels are specified explicitly but build is without VSX support."
#endif
#include <experimental/simd>
namespace SIMD_NAMESPACE = std::experimental;
#error "Not yet implemented! Try using OCTOTIGER_KOKKOS_SIMD_EXTENSION=DISCOVERY to fall back to native type "
#else
#error "Could not detect any supported SIMD instruction set. For STD experimental SIMD, please set OCTOTIGER_KOKKOS_SIMD_EXTENSION to either SCALAR or DISCOVERY !"
#endif
#else
#include <experimental/simd>
namespace SIMD_NAMESPACE = std::experimental;
using host_simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::scalar>;
using host_simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::scalar>;
static_assert(host_simd_t::size() == 1);
static_assert(host_simd_mask_t::size() == 1);
#endif

using device_simd_t = SIMD_NAMESPACE::simd<double, SIMD_NAMESPACE::simd_abi::scalar>;
using device_simd_mask_t = SIMD_NAMESPACE::simd_mask<double, SIMD_NAMESPACE::simd_abi::scalar>;

namespace std{
    namespace experimental {
        template <typename Vector, typename Mask>
        inline Vector choose(const Mask& m, const Vector& tr, const Vector& fl)
        {
            Vector res = fl;
            where(m, res) = tr;
            return res;
        }
    }
}
