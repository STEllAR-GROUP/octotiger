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

#include "octotiger/real.hpp"
#include "octotiger/taylor.hpp"

#include <cstddef>
#include <vector>

#include <aligned_buffer_util.hpp>
#include <buffer_manager.hpp>

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        /** Controls the order in which the cpu multipole FMM interactions are calculated
         * (blocking). The actual numeric operations are found in compute_kernel_templates.hpp. This
         * class is mostly responsible for loading data and control the order to increase cache
         * efficieny on the cpu
         */
        class multipole_cpu_kernel
        {
        private:
            const m2m_vector theta_rec_squared;
            m2m_int_vector offset_vector;

            /// Executes a small block of RHO interactions (size is controlled by STENCIL_BLOCKING)
            void blocked_interaction_rho(const cpu_expansion_buffer_t& local_expansions_SoA,
                const cpu_space_vector_buffer_t& center_of_masses_SoA,
                cpu_expansion_result_buffer_t& potential_expansions_SoA,
                cpu_angular_result_t& angular_corrections_SoA, const cpu_monopole_buffer_t& mons,
                const multiindex<>& cell_index, const size_t cell_flat_index,
                const multiindex<m2m_int_vector>& cell_index_coarse,
                const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
                const two_phase_stencil& stencil, const size_t outer_stencil_index);

            /// Executes a small block of non-RHO interactions (size is controlled by
            /// STENCIL_BLOCKING)
            void blocked_interaction_non_rho(const cpu_expansion_buffer_t& local_expansions_SoA,
                const cpu_space_vector_buffer_t& center_of_masses_SoA,
                cpu_expansion_result_buffer_t& potential_expansions_SoA,
                cpu_angular_result_t& angular_corrections_SoA, const cpu_monopole_buffer_t& mons,
                const multiindex<>& cell_index, const size_t cell_flat_index,
                const multiindex<m2m_int_vector>& cell_index_coarse,
                const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
                const two_phase_stencil& stencil, const size_t outer_stencil_index);

            void non_blocked_interaction_rho(const cpu_expansion_buffer_t& local_expansions_SoA,
                const cpu_space_vector_buffer_t& center_of_masses_SoA,
                cpu_expansion_result_buffer_t& potential_expansions_SoA,
                cpu_angular_result_t& angular_corrections_SoA, const cpu_monopole_buffer_t& mons,
                const multiindex<>& cell_index, const size_t cell_flat_index,
                const multiindex<m2m_int_vector>& cell_index_coarse,
                const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
                const std::vector<bool>& stencil, const std::vector<bool>& inner_mask,
                const size_t outer_stencil_index);

            void non_blocked_interaction_non_rho(const cpu_expansion_buffer_t& local_expansions_SoA,
                const cpu_space_vector_buffer_t& center_of_masses_SoA,
                cpu_expansion_result_buffer_t& potential_expansions_SoA,
                cpu_angular_result_t& angular_corrections_SoA, const cpu_monopole_buffer_t& mons,
                const multiindex<>& cell_index, const size_t cell_flat_index,
                const multiindex<m2m_int_vector>& cell_index_coarse,
                const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
                const std::vector<bool>& stencil, const std::vector<bool>& inner_mask,
                const size_t outer_stencil_index);
            
            void non_blocked_root_interaction_rho(
                const cpu_expansion_buffer_t& local_expansions_SoA,
                const cpu_space_vector_buffer_t& center_of_masses_SoA,
                cpu_expansion_result_buffer_t& potential_expansions_SoA,
                cpu_angular_result_t& angular_corrections_SoA,
                const multiindex<>& cell_index, const size_t cell_flat_index,
                const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
                const std::vector<bool>& inner_mask);

            void non_blocked_root_interaction_non_rho(
                const cpu_expansion_buffer_t& local_expansions_SoA,
                const cpu_space_vector_buffer_t& center_of_masses_SoA,
                cpu_expansion_result_buffer_t& potential_expansions_SoA,
                cpu_angular_result_t& angular_corrections_SoA,
                const multiindex<>& cell_index, const size_t cell_flat_index,
                const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
                const std::vector<bool>& inner_mask);

        public:
            multipole_cpu_kernel();

            multipole_cpu_kernel(multipole_cpu_kernel& other) = delete;

            multipole_cpu_kernel(const multipole_cpu_kernel& other) = delete;

            multipole_cpu_kernel operator=(const multipole_cpu_kernel& other) = delete;

            /// Calculate all multipole interactions for this kernel (runs the kernel)
            void apply_stencil(const cpu_expansion_buffer_t& local_expansions_SoA,
                const cpu_space_vector_buffer_t& center_of_masses_SoA,
                cpu_expansion_result_buffer_t& potential_expansions_SoA,
                cpu_angular_result_t& angular_corrections_SoA, const cpu_monopole_buffer_t& mons,
                const two_phase_stencil& stencil, gsolve_type type);

            void apply_stencil_non_blocked(const cpu_expansion_buffer_t& local_expansions_SoA,
                const cpu_space_vector_buffer_t& center_of_masses_SoA,
                cpu_expansion_result_buffer_t& potential_expansions_SoA,
                cpu_angular_result_t& angular_corrections_SoA, const cpu_monopole_buffer_t& mons,
                const std::vector<bool>& stencil, const std::vector<bool>& inner_stencil,
                gsolve_type type);

            void apply_stencil_root_non_blocked(
                const cpu_expansion_buffer_t& local_expansions_SoA,
                const cpu_space_vector_buffer_t& center_of_masses_SoA,
                cpu_expansion_result_buffer_t& potential_expansions_SoA,
                cpu_angular_result_t& angular_corrections_SoA,
                const std::vector<bool>& inner_stencil,
                gsolve_type type);
        };

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
