//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/common_kernel/interactions_iterators.hpp"
#include "octotiger/common_kernel/kernel_simd_types.hpp"
#include "octotiger/common_kernel/multiindex.hpp"
#include "octotiger/common_kernel/struct_of_array_data.hpp"
#include "octotiger/taylor.hpp"

#include <cstdint>
#include <vector>

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {
        constexpr uint64_t P2M_STENCIL_BLOCKING = 1;

        class p2m_kernel
        {
        private:
            const m2m_vector theta_rec_squared;
            m2m_int_vector offset_vector;

            /// Calculates the monopole multipole boundary interactions with solve type rho
            void blocked_interaction_rho(const cpu_expansion_buffer_t& local_expansions_SoA,
                const cpu_space_vector_buffer_t& center_of_masses_SoA,
                cpu_expansion_result_buffer_t& potential_expansions_SoA,
                cpu_angular_result_t& angular_corrections_SoA, const multiindex<>& cell_index,
                const size_t cell_flat_index, const multiindex<m2m_int_vector>& cell_index_coarse,
                const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
                const multiindex<>& interaction_partner_index,
                const size_t interaction_partner_flat_index,
                multiindex<m2m_int_vector>& interaction_partner_index_coarse);

            /// Calculates the monopole multipole boundary interactions without the solve type rho
            void blocked_interaction_non_rho(const cpu_expansion_buffer_t& local_expansions_SoA,
                const cpu_space_vector_buffer_t& center_of_masses_SoA,
                cpu_expansion_result_buffer_t& potential_expansions_SoA,
                cpu_angular_result_t& angular_corrections_SoA, const multiindex<>& cell_index,
                const size_t cell_flat_index, const multiindex<m2m_int_vector>& cell_index_coarse,
                const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
                const multiindex<>& interaction_partner_index,
                const size_t interaction_partner_flat_index,
                multiindex<m2m_int_vector>& interaction_partner_index_coarse);
            void vectors_check_empty();

        public:
            p2m_kernel();

            p2m_kernel(p2m_kernel& other) = delete;
            p2m_kernel(const p2m_kernel& other) = delete;
            p2m_kernel operator=(const p2m_kernel& other) = delete;

            void apply_stencil(const cpu_expansion_buffer_t& local_expansions_SoA,
                const cpu_space_vector_buffer_t& center_of_masses_SoA,
                cpu_expansion_result_buffer_t& potential_expansions_SoA,
                cpu_angular_result_t& angular_corrections_SoA,
                const std::vector<multiindex<>>& stencil, gsolve_type type, bool (&z_skip)[3][3][3],
                bool (&y_skip)[3][3], bool (&x_skip)[3]);
            template <size_t buffer_size>
            void apply_stencil_neighbor(const multiindex<>& neighbor_size,
                const multiindex<>& neighbor_start_index, const multiindex<>& neighbor_end_index,
                const struct_of_array_data<expansion, real, 20, buffer_size, SOA_PADDING,
                    std::vector<real,
                        recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&
                    local_expansions_SoA,
                const struct_of_array_data<space_vector, real, 3, buffer_size, SOA_PADDING,
                    std::vector<real,
                        recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&
                    center_of_masses_SoA,
                const struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING,
                    std::vector<real,
                        recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&
                    center_of_masses_inner_cells_SoA,
                cpu_expansion_result_buffer_t& potential_expansions_SoA,
                cpu_angular_result_t& angular_corrections_SoA,
                const std::vector<bool>& stencil_masks, gsolve_type type,
                const geo::direction& dir);
        };

    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
