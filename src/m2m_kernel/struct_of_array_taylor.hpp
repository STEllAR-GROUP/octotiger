#pragma once

#include "m2m_simd_types.hpp"
#include "struct_of_array_data.hpp"

namespace octotiger {
namespace fmm {

    // template <typename AoS_type, typename component_type, size_t num_components>
    // class struct_of_array_taylor
    //     : public struct_of_array_view<AoS_type, component_type, num_components>
    // {
    // private:
    // public:
    //     // to be initialized via conversion, for interoperability with struct_of_array_data
    //     struct_of_array_taylor(
    //         const struct_of_array_view<AoS_type, component_type, num_components>& other)
    //       : struct_of_array_view<AoS_type, component_type, num_components>(other) {}

    //     // accesses the coefficient of the first gradient of gravitational potential
    //     inline m2m_vector grad_0() const {
    //         return m2m_vector(this->component_pointer(0), Vc::flags::element_aligned);
    //     }

    //     // accesses the coefficients of the second gradient
    //     inline m2m_vector grad_1(size_t grad1_index) const {
    //         // skip grad 0 component
    //         return m2m_vector(this->component_pointer(grad1_index + 1), Vc::flags::element_aligned);
    //     }
    // };

}    // namespace fmm
}    // namespace octotiger
