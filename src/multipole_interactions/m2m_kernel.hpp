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

        class m2m_kernel
        {
        private:
            const m2m_vector theta_rec_squared;
            m2m_int_vector offset_vector;

            void blocked_interaction_rho(const struct_of_array_data<expansion, real, 20, ENTRIES,
                                             SOA_PADDING>& local_expansions_SoA,
                const struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                    center_of_masses_SoA,
                struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>&
                    potential_expansions_SoA,
                struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>&
                    angular_corrections_SoA,
                const std::vector<real>& mons, const multiindex<>& cell_index,
                const size_t cell_flat_index, const multiindex<m2m_int_vector>& cell_index_coarse,
                const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
                const two_phase_stencil& stencil, const size_t outer_stencil_index);

            void blocked_interaction_non_rho(const struct_of_array_data<expansion, real, 20,
                                                 ENTRIES, SOA_PADDING>& local_expansions_SoA,
                const struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                    center_of_masses_SoA,
                struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>&
                    potential_expansions_SoA,
                struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>&
                    angular_corrections_SoA,
                const std::vector<real>& mons, const multiindex<>& cell_index,
                const size_t cell_flat_index, const multiindex<m2m_int_vector>& cell_index_coarse,
                const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
                const two_phase_stencil& stencil, const size_t outer_stencil_index);
        public:
            m2m_kernel(void);

            m2m_kernel(m2m_kernel& other) = delete;

            m2m_kernel(const m2m_kernel& other) = delete;

            m2m_kernel operator=(const m2m_kernel& other) = delete;

            void apply_stencil(const struct_of_array_data<expansion, real, 20, ENTRIES,
                                   SOA_PADDING>& local_expansions_SoA,
                const struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                    center_of_masses_SoA,
                struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>&
                    potential_expansions_SoA,
                struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>&
                    angular_corrections_SoA,
                const std::vector<real>& mons, const two_phase_stencil& stencil, gsolve_type type);
        };

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
