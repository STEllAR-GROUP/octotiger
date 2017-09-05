#include "m2m_kernel.hpp"

#include "grid_flattened_indices.hpp"
#include "helper.hpp"
#include "m2m_taylor_set_basis.hpp"
#include "struct_of_array_taylor.hpp"

extern taylor<4, real> factor;
extern taylor<4, m2m_vector> factor_half_v;
extern taylor<4, m2m_vector> factor_sixth_v;

namespace octotiger {
namespace fmm {
namespace p2p_kernel {

    // TODO:
    // - check codegen and fix in Vc
    // - check for amount of temporaries
    // - try to replace expensive operations like sqrt
    // - remove all sqr()
    // - increase INX

    void m2m_kernel::blocked_interaction_rho(
        struct_of_array_data<expansion, real, 20, ENTRIES,
        SOA_PADDING>& __restrict__ local_expansions_SoA, // M_ptr
        struct_of_array_data<space_vector, real, 3, ENTRIES,
        SOA_PADDING>& __restrict__ center_of_masses_SoA, // com0
        struct_of_array_data<expansion, real, 20, ENTRIES,
        SOA_PADDING>& __restrict__ potential_expansions_SoA, // L
        struct_of_array_data<space_vector, real, 3, ENTRIES,
        SOA_PADDING>& __restrict__ angular_corrections_SoA, // L_c
        const multiindex<>& __restrict__ cell_index, const size_t cell_flat_index, ///iii0
        const multiindex<m2m_int_vector>& __restrict__ cell_index_coarse,
        const multiindex<>& __restrict__ cell_index_unpadded, const size_t cell_flat_index_unpadded,
        const std::vector<multiindex<>>& __restrict__ stencil, const size_t outer_stencil_index) {
        // TODO: should change name to something better (not taylor, but space_vector)
        // struct_of_array_taylor<space_vector, real, 3> X =
        //     center_of_masses_SoA.get_view(cell_flat_index);

        std::array<m2m_vector, 4> d_components;
        d_components[0] = 1.0 / dx;
        d_components[1] = 1.0 / sqr(dx);
        d_components[2] = 1.0 / sqr(dx);
        d_components[3] = 1.0 / sqr(dx);

        for (size_t inner_stencil_index = 0; inner_stencil_index < STENCIL_BLOCKING &&
             outer_stencil_index + inner_stencil_index < stencil.size();
             inner_stencil_index += 1) {
            const multiindex<>& stencil_element =
                stencil[outer_stencil_index + inner_stencil_index];
            const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);

            const size_t interaction_partner_flat_index =
                to_flat_index_padded(interaction_partner_index); // iii1n

            // all relevant components (4)
            std::array<m2m_vector, 20> m_partner; //m0 from mpole from neighbors?
            m_partner[0] = local_expansions_SoA.value<0>(interaction_partner_flat_index);
            m_partner[1] = local_expansions_SoA.value<1>(interaction_partner_flat_index);
            m_partner[2] = local_expansions_SoA.value<2>(interaction_partner_flat_index);
            m_partner[3] = local_expansions_SoA.value<3>(interaction_partner_flat_index);

            m_partner[0] *= d_components[0];
            m_partner[1] *= d_components[1];
            m_partner[2] *= d_components[2];
            m_partner[3] *= d_components[3];

            // do these values really map to i0 - i1, j0 - j1, k0 - k1?
            // gotta complain to david if they do not
            const real x = stencil_element.x;
            const real y = stencil_element.y;
            const real z = stencil_element.z;
            const real tmp = sqr(x) + sqr(y) + sqr(z);
            const real r = (tmp == 0) ? 0 : std::sqrt(tmp);
            const real r3 = r * r * r;
            std::array<m2m_vector, 4> four;
            if (r > 0.0) {
              four[0] = -1.0 / r;
              four[1] = x / r3;
              four[2] = y / r3;
              four[3] = z / r3;
            } else {
              for (integer i = 0; i != 4; ++i) {
                four[i] = 0.0;
              }
            }

            m_partner[0] *= four[0];
            m_partner[1] *= four[1];
            m_partner[2] *= four[2];
            m_partner[3] *= four[3];


            m2m_vector tmpstore = potential_expansions_SoA.value<0>(cell_flat_index_unpadded) + m_partner[0];
            tmpstore.memstore(
                potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore = potential_expansions_SoA.value<1>(cell_flat_index_unpadded) + m_partner[1];
            tmpstore.memstore(
                potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore = potential_expansions_SoA.value<2>(cell_flat_index_unpadded) + m_partner[2];
            tmpstore.memstore(
                potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore = potential_expansions_SoA.value<3>(cell_flat_index_unpadded) + m_partner[3];
            tmpstore.memstore(
                potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
        }
    }

    void m2m_kernel::blocked_interaction_non_rho(
        struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>& local_expansions_SoA,
        struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>& center_of_masses_SoA,
        struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>& potential_expansions_SoA,
        struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>& angular_corrections_SoA,
        const multiindex<>& cell_index, const size_t cell_flat_index,
        const multiindex<m2m_int_vector>& cell_index_coarse,
        const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
        const std::vector<multiindex<>>& stencil, const size_t outer_stencil_index) {
      std::cout << "Why are you here?" << std::endl;
    }
}    // namespace p2p_kernel
}    // namespace fmm
}    // namespace octotiger
