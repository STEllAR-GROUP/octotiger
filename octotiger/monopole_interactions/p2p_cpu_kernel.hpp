#pragma once

#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/common_kernel/interactions_iterators.hpp"
#include "octotiger/common_kernel/kernel_simd_types.hpp"
#include "octotiger/common_kernel/multiindex.hpp"
#include "octotiger/common_kernel/struct_of_array_data.hpp"

#include "octotiger/real.hpp"
#include "octotiger/taylor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        constexpr uint64_t P2P_STENCIL_BLOCKING = 24;
        class p2p_cpu_kernel
        {
        private:
            std::vector<bool>& neighbor_empty;

            // so skip non-existing interaction partners faster, one entry per vector variable
            std::vector<bool> vector_is_empty;

            const m2m_vector theta_rec_squared;
            m2m_int_vector offset_vector;

            void cell_interactions(
                std::vector<real>& mons,
                struct_of_array_data<expansion, real, 20, INNER_CELLS,
                    SOA_PADDING>& __restrict__ potential_expansions_SoA,    // L
                const multiindex<>& __restrict__ cell_index,
                const size_t cell_flat_index,    /// iii0
                const multiindex<m2m_int_vector>& __restrict__ cell_index_coarse,
                const multiindex<>& __restrict__ cell_index_unpadded,
                const size_t cell_flat_index_unpadded,
                const std::vector<bool>& __restrict__ stencil,
                const std::vector<std::array<real, 4>>& __restrict__ four_constants,
                const size_t outer_stencil_index, real dx);

        public:
            p2p_cpu_kernel(std::vector<bool>& neighbor_empty);

            p2p_cpu_kernel(p2p_cpu_kernel& other) = delete;

            p2p_cpu_kernel(const p2p_cpu_kernel& other) = delete;

            p2p_cpu_kernel operator=(const p2p_cpu_kernel& other) = delete;

            void apply_stencil(std::vector<real>& mons,
                struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>&
                    potential_expansions_SoA,
                const std::vector<bool>& stencil, const
                std::vector<std::array<real, 4>>& four,
                real dx);
        };

    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
