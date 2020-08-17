//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/common_kernel/kernel_simd_types.hpp"
#include "octotiger/common_kernel/multiindex.hpp"
#include "octotiger/common_kernel/struct_of_array_data.hpp"

#include "octotiger/real.hpp"
#include "octotiger/taylor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <aligned_buffer_util.hpp>
#include <buffer_manager.hpp>

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        constexpr uint64_t P2P_STENCIL_BLOCKING = 24;
        class p2p_cpu_kernel
        {
        private:
            // so skip non-existing interaction partners faster, one entry per vector variable
            std::vector<bool> vector_is_empty;

            const m2m_vector theta_rec_squared;
            m2m_int_vector offset_vector;

            void cell_interactions(const cpu_monopole_buffer_t& __restrict__ mons,
                cpu_expansion_result_buffer_t& __restrict__ potential_expansions_SoA,    // L
                const multiindex<>& __restrict__ cell_index,
                const size_t cell_flat_index,    /// iii0
                const multiindex<m2m_int_vector>& __restrict__ cell_index_coarse,
                const multiindex<>& __restrict__ cell_index_unpadded,
                const size_t cell_flat_index_unpadded,
                const std::vector<bool>& __restrict__ stencil,
                const std::vector<std::array<real, 4>>& __restrict__ four_constants,
                const size_t outer_stencil_index, real dx);

        public:
            p2p_cpu_kernel();

            p2p_cpu_kernel(p2p_cpu_kernel& other) = delete;

            p2p_cpu_kernel(const p2p_cpu_kernel& other) = delete;

            p2p_cpu_kernel operator=(const p2p_cpu_kernel& other) = delete;

            void apply_stencil(const cpu_monopole_buffer_t& local_expansions,
                cpu_expansion_result_buffer_t& potential_expansions_SoA,
                const std::vector<bool>& stencil, const std::vector<std::array<real, 4>>& four,
                real dx);
        };

    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
