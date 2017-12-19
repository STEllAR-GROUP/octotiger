#include "p2p_kernel.hpp"

#include "../common_kernel/helper.hpp"
#include "../common_kernel/kernel_taylor_set_basis.hpp"
#include "../common_kernel/struct_of_array_data.hpp"
#include "grid_flattened_indices.hpp"

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

        void p2p_kernel::blocked_interaction_rho(
            std::vector<real>& mons,
            struct_of_array_data<expansion, real, 20, ENTRIES,
                SOA_PADDING>& __restrict__ potential_expansions_SoA,    // L
            const multiindex<>& __restrict__ cell_index,
            const size_t cell_flat_index,    /// iii0
            const multiindex<m2m_int_vector>& __restrict__ cell_index_coarse,
            const multiindex<>& __restrict__ cell_index_unpadded,
            const size_t cell_flat_index_unpadded,
            const std::vector<multiindex<>>& __restrict__ stencil,
            const std::vector<std::array<real, 4>>& __restrict__ four_constants,
            const size_t outer_stencil_index) {
            // TODO: should change name to something better (not taylor, but space_vector)
            // struct_of_array_taylor<space_vector, real, 3> X =
            //     center_of_masses_SoA.get_view(cell_flat_index);

            std::array<m2m_vector, 4> d_components;
            d_components[0] = 1.0 / dx;
            d_components[1] = -1.0 / sqr(dx);
            d_components[2] = -1.0 / sqr(dx);
            d_components[3] = -1.0 / sqr(dx);
            std::array<m2m_vector, 4> tmpstore;
            tmpstore[0] = potential_expansions_SoA.value<0>(cell_flat_index_unpadded);
            tmpstore[1] = potential_expansions_SoA.value<1>(cell_flat_index_unpadded);
            tmpstore[2] = potential_expansions_SoA.value<2>(cell_flat_index_unpadded);
            tmpstore[3] = potential_expansions_SoA.value<3>(cell_flat_index_unpadded);

            for (size_t inner_stencil_index = 0; inner_stencil_index < P2P_STENCIL_BLOCKING &&
                 outer_stencil_index + inner_stencil_index < stencil.size();
                 inner_stencil_index +=
                 1) {    // blocking is done by stepping in die outer_stencil index
                const multiindex<>& stencil_element =
                    stencil[outer_stencil_index + inner_stencil_index];
                const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                    cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);

                const size_t interaction_partner_flat_index =
                    to_flat_index_padded(interaction_partner_index);    // iii1n

                // implicitly broadcasts to vector
                multiindex<m2m_int_vector> interaction_partner_index_coarse(
                    interaction_partner_index);
                interaction_partner_index_coarse.z += offset_vector;
                // note that this is the same for groups of 2x2x2 elements
                // -> maps to the same for some SIMD lanes
                interaction_partner_index_coarse.transform_coarse();

                m2m_int_vector theta_c_rec_squared_int = detail::distance_squared_reciprocal(
                    cell_index_coarse, interaction_partner_index_coarse);

                m2m_vector theta_c_rec_squared =
                    // Vc::static_datapar_cast<double>(theta_c_rec_squared_int);
                    Vc::static_datapar_cast_double_to_int(theta_c_rec_squared_int);

                m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;

                m2m_vector monopole(
                    mons.data() + interaction_partner_flat_index, Vc::flags::element_aligned);
                // real monopole = mons[interaction_partner_flat_index];

                std::array<m2m_vector, 4> four;
                Vc::where(mask, four[0]) =
                    four_constants[outer_stencil_index + inner_stencil_index][0];
                Vc::where(mask, four[1]) =
                    four_constants[outer_stencil_index + inner_stencil_index][1];
                Vc::where(mask, four[2]) =
                    four_constants[outer_stencil_index + inner_stencil_index][2];
                Vc::where(mask, four[3]) =
                    four_constants[outer_stencil_index + inner_stencil_index][3];

                four[0] *= monopole;
                four[1] *= monopole;
                four[2] *= monopole;
                four[3] *= monopole;
                tmpstore[0] = tmpstore[0] + four[0] * d_components[0];
                tmpstore[1] = tmpstore[1] + four[1] * d_components[1];
                tmpstore[2] = tmpstore[2] + four[2] * d_components[2];
                tmpstore[3] = tmpstore[3] + four[3] * d_components[3];
            }
            tmpstore[0].memstore(potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore[1].memstore(potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore[2].memstore(potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore[3].memstore(potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
        }

        void p2p_kernel::blocked_interaction_non_rho(
            std::vector<real>& mons,
            struct_of_array_data<expansion, real, 20, ENTRIES,
                SOA_PADDING>& __restrict__ potential_expansions_SoA,    // L
            const multiindex<>& __restrict__ cell_index,
            const size_t cell_flat_index,    /// iii0
            const multiindex<m2m_int_vector>& __restrict__ cell_index_coarse,
            const multiindex<>& __restrict__ cell_index_unpadded,
            const size_t cell_flat_index_unpadded,
            const std::vector<multiindex<>>& __restrict__ stencil,
            const std::vector<std::array<real, 4>>& __restrict__ four_constants,
            const size_t outer_stencil_index) {
            // TODO: should change name to something better (not taylor, but space_vector)
            // struct_of_array_taylor<space_vector, real, 3> X =
            //     center_of_masses_SoA.get_view(cell_flat_index);

            std::array<m2m_vector, 4> d_components;
            d_components[0] = 1.0 / dx;
            d_components[1] = -1.0 / sqr(dx);
            d_components[2] = -1.0 / sqr(dx);
            d_components[3] = -1.0 / sqr(dx);
            std::array<m2m_vector, 4> tmpstore;
            tmpstore[0] = potential_expansions_SoA.value<0>(cell_flat_index_unpadded);
            tmpstore[1] = potential_expansions_SoA.value<1>(cell_flat_index_unpadded);
            tmpstore[2] = potential_expansions_SoA.value<2>(cell_flat_index_unpadded);
            tmpstore[3] = potential_expansions_SoA.value<3>(cell_flat_index_unpadded);

            for (size_t inner_stencil_index = 0; inner_stencil_index < P2P_STENCIL_BLOCKING &&
                 outer_stencil_index + inner_stencil_index < stencil.size();
                 inner_stencil_index +=
                 1) {    // blocking is done by stepping in die outer_stencil index
                const multiindex<>& stencil_element =
                    stencil[outer_stencil_index + inner_stencil_index];
                const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                    cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);

                const size_t interaction_partner_flat_index =
                    to_flat_index_padded(interaction_partner_index);    // iii1n

                // implicitly broadcasts to vector
                multiindex<m2m_int_vector> interaction_partner_index_coarse(
                    interaction_partner_index);
                interaction_partner_index_coarse.z += offset_vector;
                // note that this is the same for groups of 2x2x2 elements
                // -> maps to the same for some SIMD lanes
                interaction_partner_index_coarse.transform_coarse();

                m2m_int_vector theta_c_rec_squared_int = detail::distance_squared_reciprocal(
                    cell_index_coarse, interaction_partner_index_coarse);

                m2m_vector theta_c_rec_squared =
                    // Vc::static_datapar_cast<double>(theta_c_rec_squared_int);
                    Vc::static_datapar_cast_double_to_int(theta_c_rec_squared_int);

                m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;

                m2m_vector monopole(
                    mons.data() + interaction_partner_flat_index, Vc::flags::element_aligned);
                // real monopole = mons[interaction_partner_flat_index];

                std::array<m2m_vector, 4> four;
                Vc::where(mask, four[0]) =
                    four_constants[outer_stencil_index + inner_stencil_index][0];
                Vc::where(mask, four[1]) =
                    four_constants[outer_stencil_index + inner_stencil_index][1];
                Vc::where(mask, four[2]) =
                    four_constants[outer_stencil_index + inner_stencil_index][2];
                Vc::where(mask, four[3]) =
                    four_constants[outer_stencil_index + inner_stencil_index][3];

                four[0] *= monopole;
                four[1] *= monopole;
                four[2] *= monopole;
                four[3] *= monopole;
                tmpstore[0] = tmpstore[0] + four[0] * d_components[0];
                tmpstore[1] = tmpstore[1] + four[1] * d_components[1];
                tmpstore[2] = tmpstore[2] + four[2] * d_components[2];
                tmpstore[3] = tmpstore[3] + four[3] * d_components[3];
            }
            tmpstore[0].memstore(potential_expansions_SoA.pointer<0>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore[1].memstore(potential_expansions_SoA.pointer<1>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore[2].memstore(potential_expansions_SoA.pointer<2>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
            tmpstore[3].memstore(potential_expansions_SoA.pointer<3>(cell_flat_index_unpadded),
                Vc::flags::element_aligned);
        }
    }    // namespace p2p_kernel
}    // namespace fmm
}    // namespace octotiger
