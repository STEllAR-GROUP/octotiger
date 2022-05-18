//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "octotiger/geometry.hpp"
#if defined(OCTOTIGER_LEGACY_VC)
#include "octotiger/simd.hpp"
#endif
#include "octotiger/space_vector.hpp"
#include "octotiger/taylor.hpp"

#include <hpx/synchronization/counting_semaphore.hpp>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

using multipole_pass_type = std::pair<std::vector<multipole>, std::vector<space_vector>>;
using expansion_pass_type = std::pair<std::vector<expansion>, std::vector<space_vector>>;
using semaphore = hpx::lcos::local::counting_semaphore;

struct gravity_boundary_type
{
    std::shared_ptr<std::vector<multipole>> M;
    std::shared_ptr<std::vector<real>> m;
    std::shared_ptr<std::vector<space_vector>> x;
    semaphore* local_semaphore;
    gravity_boundary_type()
      : M(nullptr)
      , m(nullptr)
      , x(nullptr) {}
    void allocate() {
        local_semaphore = nullptr;
        if (M == nullptr) {
            M = std::make_shared<std::vector<multipole>>();
            m = std::make_shared<std::vector<real>>();
            x = std::make_shared<std::vector<space_vector>>();
        }
    }
    template <class Archive>
    void serialize(Archive& arc, unsigned) {
        allocate();
        std::uintptr_t tmp = reinterpret_cast<std::uintptr_t>(local_semaphore);
        arc& M;
        arc& m;
        arc& x;
        arc& tmp;
        local_semaphore = reinterpret_cast<decltype(local_semaphore)>(tmp);
    }
};
#if defined(OCTOTIGER_LEGACY_VC)
Vc_DECLARE_ALLOCATOR(gravity_boundary_type)
#endif

struct neighbor_gravity_type {
    gravity_boundary_type data;
    bool is_monopole;
    geo::direction direction;
};
#if defined(OCTOTIGER_LEGACY_VC)
Vc_DECLARE_ALLOCATOR(neighbor_gravity_type)
#endif

// struct neighbor_gravity_type
// {
//     gravity_boundary_type data;
//     bool is_monopole;
//     geo::direction direction;

//     template <class Archive>
//     void serialize(Archive& arc, unsigned) {
//         arc& data;
//         arc& is_monopole;
//         arc& direction;
//     }
// };

class interaction_type
{
 public:
    // linear index in cell of first interaction partner
    std::uint16_t first;
    // linear index in cell of second interaction partner
    // (in case of pointing to neighbor, inner linear index in neighbor cell)
    std::uint16_t second;
    // index vector in cell
    alignas(32) space_vector x;
    // precomputed values: (-1.0/r, (i0-j0)/r^3, (i1-j1)/r^3, (i2-j2)/r^3), r - distance(i - j)
#if defined(OCTOTIGER_LEGACY_VC)
    alignas(32) v4sd four;
#else
    alignas(32) std::array<real, 4> four;
#endif
    // // helper variable for vectorization
    // std::uint32_t inner_loop_stop;

    interaction_type() {
        first = 0;
        second = 0;
        x = {};
        four = {};
    }

    interaction_type(const interaction_type &other) {
        first = other.first;
        second = other.second;
        x = other.x;
        four = other.four;
    }

    // interaction_type& operator=(const interaction_type &other) {
    //     first = other.first;
    //     second = other.second;
    //     x = other.x;
    //     four = other.four;
    // }

    // interaction_type(const interaction_type &&other) {
    //     first = other.first;
    //     second = other.second;
    //     x = other.x;
    //     four = other.four;
    // }
};
#if defined(OCTOTIGER_LEGACY_VC)
Vc_DECLARE_ALLOCATOR(interaction_type)
#endif

struct boundary_interaction_type
{
    // // all interaction partners, if first.size() == 1, else the current index
    // std::vector<std::uint16_t> second;
    std::uint16_t second;
    // all interaction partners, if second.size() == 1, else the current index
    std::vector<std::uint16_t> first;
    // precomputed values, as in interaction_type
#if defined(__AVX2__) && defined(OCTOTIGER_LEGACY_VC)
    std::vector<v4sd> four;
#else
    std::vector<std::array<real, 4>> four;
#endif
    // index vector in cell
    space_vector x;
};
#if defined(OCTOTIGER_LEGACY_VC)
//Vc_DECLARE_ALLOCATOR(boundary_interaction_type)
#endif


#include "options_enum.hpp"
#include <boost/algorithm/string.hpp>

COMMAND_LINE_ENUM(interaction_host_kernel_type,DEVICE_ONLY,LEGACY,VC,KOKKOS);
COMMAND_LINE_ENUM(interaction_device_kernel_type,OFF,CUDA,HIP,KOKKOS_CUDA,KOKKOS_HIP);
COMMAND_LINE_ENUM(amr_boundary_type,AMR_LEGACY,AMR_OPTIMIZED,AMR_CUDA);

