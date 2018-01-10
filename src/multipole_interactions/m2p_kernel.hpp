#pragma once

#include "../common_kernel/interaction_constants.hpp"
#include "../common_kernel/interactions_iterators.hpp"
#include "../common_kernel/kernel_simd_types.hpp"
#include "../common_kernel/multiindex.hpp"
#include "../common_kernel/struct_of_array_data.hpp"
#include "taylor.hpp"

#include <vector>

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {
        constexpr uint64_t M2P_STENCIL_BLOCKING = 1;

        class m2p_kernel
        {
        private:
            /// Stores which neighbors are empty so that we can just skip them
            std::vector<bool>& neighbor_empty;

            /// so skip non-existing interaction partners faster, one entry per vector variable
            std::vector<bool> vector_is_empty;

            /// determines how the system is going to be solved: rho or non-rho
            gsolve_type type;

            const m2m_vector theta_rec_squared;
            m2m_int_vector offset_vector;

            real dX;
            std::array<real, NDIM> xBase;

            /// Calculates the monopole multipole boundary interactions with solve type rho
            void blocked_interaction_rho(struct_of_array_data<expansion, real, 20, ENTRIES,
                                             SOA_PADDING>& __restrict__ local_expansions_SoA,
                std::vector<real>& mons, struct_of_array_data<space_vector, real, 3, ENTRIES,
                                             SOA_PADDING>& center_of_masses_SoA,
                struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>&
                    potential_expansions_SoA,
                struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                    angular_corrections_SoA,
                const multiindex<>& cell_index, const size_t cell_flat_index,
                const multiindex<m2m_int_vector>& cell_index_coarse,
                const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
                const std::vector<multiindex<>>& stencil, const size_t outer_stencil_index,
                std::vector<bool>& interact);

            /// Calculates the monopole multipole boundary interactions without the solve type rho
            void blocked_interaction_non_rho(std::vector<real>& mons,
                struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                    center_of_masses_SoA,
                struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>&
                    potential_expansions_SoA,
                struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                    angular_corrections_SoA,
                const multiindex<>& cell_index, const size_t cell_flat_index,
                const multiindex<m2m_int_vector>& cell_index_coarse,
                const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
                const std::vector<multiindex<>>& stencil, const size_t outer_stencil_index,
                std::vector<bool>& interact);

            void vectors_check_empty();

        public:
            m2p_kernel(std::vector<bool>& neighbor_empty, gsolve_type type, real dX,
                std::array<real, NDIM> xbase);

            m2p_kernel(m2p_kernel& other) = delete;
            m2p_kernel(const m2p_kernel& other) = delete;
            m2p_kernel operator=(const m2p_kernel& other) = delete;

            void apply_stencil(std::vector<real>& local_expansions,
                struct_of_array_data<expansion, real, 20, ENTRIES,
                                   SOA_PADDING>& __restrict__ local_expansions_SoA,
                struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                    center_of_masses_SoA,
                struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>&
                    potential_expansions_SoA,
                struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                    angular_corrections_SoA,
                std::vector<multiindex<>>& stencil, std::vector<bool>& interact);
        };

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
