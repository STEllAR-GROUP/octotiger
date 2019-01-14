#pragma once

#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/common_kernel/kernel_simd_types.hpp"
#include "octotiger/common_kernel/multiindex.hpp"

#include "octotiger/geometry.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/interaction_types.hpp"
#include "octotiger/monopole_interactions/p2p_cpu_kernel.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        /// Interface for the new monopole-monopole compute kernel
        class p2p_interaction_interface
        {
        public:
            p2p_interaction_interface(void);
            /** Takes AoS data, converts it, calculates FMM monopole-monopole interactions,
              * stores results in L */
            void compute_p2p_interactions(std::vector<real>& monopoles,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
                std::array<bool, geo::direction::count()>& is_direction_empty);
            /// Sets the grid pointer - usually only required once
            void set_grid_ptr(std::shared_ptr<grid> ptr) {
                grid_ptr = ptr;
            }

        protected:
            template <typename monopole_container>
            void update_input(std::vector<real>& mons,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type,
                monopole_container& local_monopoles);
            void compute_interactions(gsolve_type type,
                std::array<bool, geo::direction::count()>& is_direction_empty,
                std::vector<neighbor_gravity_type>& all_neighbor_interaction_data, real dx);

            std::shared_ptr<grid> grid_ptr;
        private:
            /// The stencil is used to identify the neighbors
            static thread_local const std::vector<multiindex<>> stencil;
            static thread_local const std::vector<std::array<real, 4>> four;
            static thread_local std::vector<real> local_monopoles_staging_area;
            std::vector<bool> neighbor_empty_monopoles;

            interaction_kernel_type p2p_type;
            p2p_cpu_kernel kernel_monopoles;
        };

        template <typename monopole_container>
        void p2p_interaction_interface::update_input(std::vector<real>& mons,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type,
            monopole_container& local_monopoles) {
            iterate_inner_cells_padded(
                [&local_monopoles, mons](const multiindex<>& i, const size_t flat_index,
                    const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                    local_monopoles.at(flat_index) = mons.at(flat_index_unpadded);
                });

            for (size_t i = 0; i < neighbor_empty_monopoles.size(); i++) {
                neighbor_empty_monopoles[i] = false;
            }

            // Now look at neighboring data
            for (const geo::direction& dir : geo::direction::full_set()) {
                // don't use neighbor.direction, is always zero for empty cells!
                neighbor_gravity_type& neighbor = neighbors[dir];
                if (!neighbor.is_monopole) {
                    neighbor_empty_monopoles[dir.flat_index_with_center()] = true;
                    // neighbor has no data - input structure just recevices zeros as padding
                    iterate_inner_cells_padding(
                        dir, [&local_monopoles](const multiindex<>& i, const size_t flat_index,
                                 const multiindex<>&,
                                 const size_t) { local_monopoles.at(flat_index) = 0.0; });
                } else {
                    if (neighbor.is_monopole) {
                        if (!neighbor.data.m) {
                            iterate_inner_cells_padding(dir,
                                [&local_monopoles](const multiindex<>& i, const size_t flat_index,
                                                            const multiindex<>&, const size_t) {
                                    // initializes whole expansion, relatively expansion
                                    local_monopoles.at(flat_index) = 0.0;
                                });
                            neighbor_empty_monopoles[dir.flat_index_with_center()] = true;
                        } else {
                            std::vector<real>& neighbor_mons = *(neighbor.data.m);
                            const bool fullsizes = neighbor_mons.size() == INNER_CELLS;
                            if (fullsizes) {
                                iterate_inner_cells_padding(
                                    dir, [&local_monopoles, neighbor_mons](const multiindex<>& i,
                                             const size_t flat_index, const multiindex<>&,
                                             const size_t flat_index_unpadded) {
                                        // initializes whole expansion, relatively expansion
                                        local_monopoles.at(flat_index) =
                                            neighbor_mons.at(flat_index_unpadded);
                                    });
                            } else {
                                iterate_inner_cells_padding(
                                    dir, [&local_monopoles](const multiindex<>& i,
                                             const size_t flat_index, const multiindex<>&,
                                             const size_t) {
                                        // initializes whole expansion, relatively expansion
                                        local_monopoles.at(flat_index) = 0.0;
                                    });
                                auto list = grid_ptr->get_ilist_n_bnd(dir);
                                size_t counter = 0;
                                for (auto i : list) {
                                    const integer iii = i.second;
                                    const multiindex<> offset =
                                        flat_index_to_multiindex_not_padded(iii);
                                    const multiindex<> m(offset.x + INNER_CELLS_PADDING_DEPTH +
                                            dir[0] * INNER_CELLS_PADDING_DEPTH,
                                        offset.y + INNER_CELLS_PADDING_DEPTH +
                                            dir[1] * INNER_CELLS_PADDING_DEPTH,
                                        offset.z + INNER_CELLS_PADDING_DEPTH +
                                            dir[2] * INNER_CELLS_PADDING_DEPTH);
                                    const size_t flat_index = to_flat_index_padded(m);
                                    local_monopoles.at(flat_index) = neighbor_mons.at(counter);
                                    counter++;
                                }
                            }
                        }
                    }
                }
            }
            neighbor_empty_monopoles[13] = false;
        }
    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
