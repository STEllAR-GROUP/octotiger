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
    namespace p2m_kernel {

        class p2m_kernel
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

            /// Calculates the monopole multipole boundary interactions with solve type rho
            void blocked_interaction_rho(struct_of_array_data<expansion, real, 20, ENTRIES,
                                             SOA_PADDING>& local_expansions_SoA,
                struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                    center_of_masses_SoA,
                struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>&
                    potential_expansions_SoA,
                struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                    angular_corrections_SoA,
                const multiindex<>& cell_index, const size_t cell_flat_index,
                const multiindex<m2m_int_vector>& cell_index_coarse,
                const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
                                         const std::vector<multiindex<>>& stencil, const size_t
                                         outer_stencil_index, std::vector<bool>& interact);

            /// Calculates the monopole multipole boundary interactions without the solve type rho
            void blocked_interaction_non_rho(struct_of_array_data<expansion, real, 20, ENTRIES,
                                                 SOA_PADDING>& local_expansions_SoA,
                struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                    center_of_masses_SoA,
                struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>&
                    potential_expansions_SoA,
                struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                    angular_corrections_SoA,
                const multiindex<>& cell_index, const size_t cell_flat_index,
                const multiindex<m2m_int_vector>& cell_index_coarse,
                const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
                                             const std::vector<multiindex<>>& stencil, const
                                             size_t outer_stencil_index, std::vector<bool>& interact);

            void vectors_check_empty();

        public:
            p2m_kernel(std::vector<bool>& neighbor_empty, gsolve_type type);

            p2m_kernel(p2m_kernel& other) = delete;
            p2m_kernel(const p2m_kernel& other) = delete;
            p2m_kernel operator=(const p2m_kernel& other) = delete;

            void apply_stencil(struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>&
                                   local_expansions_SoA,
                struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                    center_of_masses_SoA,
                struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>&
                    potential_expansions_SoA,
                struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                    angular_corrections_SoA,
                std::vector<multiindex<>>& stencil, std::vector<bool>& interact);
        };

    }    // namespace p2m_kernel
}    // namespace fmm
}    // namespace octotiger
