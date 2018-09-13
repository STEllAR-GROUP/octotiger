#pragma once

#include <Vc/Vc>

#include "geometry.hpp"
#include "simd.hpp"
#include "space_vector.hpp"
#include "taylor.hpp"

#include <vector>

// typedef std::pair<std::vector<multipole>, std::vector<space_vector>> multipole_pass_type;
// typedef std::pair<std::vector<expansion>, std::vector<space_vector>> expansion_pass_type;

// struct gravity_boundary_type
// {
//     std::shared_ptr<std::vector<multipole>> M;
//     std::shared_ptr<std::vector<real>> m;
//     std::shared_ptr<std::vector<space_vector>> x;
//     bool is_local;
//     gravity_boundary_type()
//       : M(nullptr)
//       , m(nullptr)
//       , x(nullptr) {}
//     void allocate() {
//         if (M == nullptr) {
//             M = std::make_shared<std::vector<multipole>>();
//             m = std::make_shared<std::vector<real>>();
//             x = std::make_shared<std::vector<space_vector>>();
//         }
//     }
//     template <class Archive>
//     void serialize(Archive& arc, unsigned) {
//         allocate();
//         arc& M;
//         arc& m;
//         arc& x;
//         arc& is_local;
//     }
// };

typedef std::pair<std::vector<multipole>, std::vector<space_vector>> multipole_pass_type;
typedef std::pair<std::vector<expansion>, std::vector<space_vector>> expansion_pass_type;

#include <hpx/lcos/local/counting_semaphore.hpp>
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
Vc_DECLARE_ALLOCATOR(gravity_boundary_type)

struct neighbor_gravity_type {
    gravity_boundary_type data;
    bool is_monopole;
    geo::direction direction;
};
Vc_DECLARE_ALLOCATOR(neighbor_gravity_type)

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
    alignas(32) v4sd four;
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
Vc_DECLARE_ALLOCATOR(interaction_type)

struct boundary_interaction_type
{
    // // all interaction partners, if first.size() == 1, else the current index
    // std::vector<std::uint16_t> second;
    std::uint16_t second;
    // all interaction partners, if second.size() == 1, else the current index
    std::vector<std::uint16_t> first;
    // precomputed values, as in interaction_type
    std::vector<v4sd> four;
    // index vector in cell
    space_vector x;
};
Vc_DECLARE_ALLOCATOR(boundary_interaction_type)

enum interaction_kernel_type : uint32_t {
    SOA_CPU = 0,
    OLD,
    SOA_CUDA
};
