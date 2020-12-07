//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/common_kernel/interactions_iterators.hpp"
#include "octotiger/common_kernel/multiindex.hpp"
#include "octotiger/config/export_definitions.hpp"
#include "octotiger/geometry.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/interaction_types.hpp"
#include "octotiger/taylor.hpp"

#include <array>
#include <memory>
#include <vector>

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        /// The stencil is used to identify which neighbors to interact with
        static OCTOTIGER_EXPORT std::vector<multiindex<>>& p2m_stencil();
        /// Uses a cube with true/flags instead of the spherical multiindex stencil
        static OCTOTIGER_EXPORT std::vector<bool>& p2m_stencil_masks();
        void compute_p2m_interactions_neighbors_only(const std::vector<real>& monopoles,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::shared_ptr<grid>& grid_ptr);
        /// DEPRECATED! Interface for the monopole-multipole compute kernel
        class p2m_interaction_interface
        {
        public:
            p2m_interaction_interface();
            /** Takes AoS data, converts it, calculates monopole-multipole FMM interactions,
                stores results in L, L_c */
            void compute_p2m_interactions(std::vector<real>& monopoles,
                std::vector<multipole>& M_ptr,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type,
                std::array<bool, geo::direction::count()>& is_direction_empty);
            void compute_p2m_interactions_neighbors_only(const std::vector<real>& monopoles,
                std::vector<multipole>& M_ptr,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type,
                std::array<bool, geo::direction::count()>& is_direction_empty,
                std::shared_ptr<grid>& grid_ptr);
            /// Sets the grid pointer - usually only required once
            void set_grid_ptr(std::shared_ptr<grid> ptr) {
                grid_ptr = ptr;
            }

        public:
            /// The stencil is used to identify which neighbors to interact with
            static OCTOTIGER_EXPORT std::vector<multiindex<>>& stencil();

        protected:
            /// Converts AoS input data into SoA data
            template <typename expansion_soa_container, typename masses_soa_container>
            bool update_input(std::vector<multipole>& multipoles,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type,
                expansion_soa_container& local_expansions_SoA,
                masses_soa_container& center_of_masses_SoA);
            void compute_interactions(gsolve_type type,
                std::array<bool, geo::direction::count()>& is_direction_empty,
                std::vector<neighbor_gravity_type>& all_neighbor_interaction_data,
                const cpu_expansion_buffer_t& local_expansions_staging_area,
                const cpu_space_vector_buffer_t& center_of_masses_staging_area);

        private:
            bool multipole_neighbors_exist;
            std::vector<bool> neighbor_empty_multipoles;

            std::shared_ptr<grid> grid_ptr;
            interaction_kernel_type p2m_type;

            bool z_skip[3];
            bool y_skip[3][3];
            bool x_skip[3][3][3];
        };
        template <typename expansion_soa_container, typename masses_soa_container>
        void update_neighbor_input(const geo::direction& neighbor_dir,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type,
            expansion_soa_container& local_expansions_SoA,
            masses_soa_container& center_of_masses_SoA, std::shared_ptr<grid>& grid_ptr) {
            std::vector<space_vector> const& com0 = *(com_ptr[0]);
            neighbor_gravity_type& neighbor = neighbors[neighbor_dir];

            // neighbor must be refined and contain data
            assert(!neighbor.is_monopole && neighbor.data.M);

            std::vector<multipole>& neighbor_M_ptr = *(neighbor.data.M);
            std::vector<space_vector>& neighbor_com0 = *(neighbor.data.x);
            // Check whether we even have all the required data
            const bool fullsizes =
                neighbor_M_ptr.size() == INNER_CELLS && neighbor_com0.size() == INNER_CELLS;
            if (fullsizes) {
                // Get multipole data into our input structure
                iterate_padding(neighbor_dir,
                    [&local_expansions_SoA, &center_of_masses_SoA, neighbor_M_ptr, neighbor_com0](
                        const multiindex<>& i, const size_t flat_index,
                        const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                        local_expansions_SoA.set_AoS_value(
                            std::move(neighbor_M_ptr.at(flat_index_unpadded)), flat_index);
                        center_of_masses_SoA.set_AoS_value(
                            std::move(neighbor_com0.at(flat_index_unpadded)), flat_index);
                    });
            } else {
                iterate_padding(neighbor_dir,
                    [&local_expansions_SoA, &center_of_masses_SoA](const multiindex<>& i,
                        const size_t flat_index, const multiindex<>&, const size_t) {
                        local_expansions_SoA.set_AoS_value(std::move(expansion()), flat_index);
                        center_of_masses_SoA.set_AoS_value(std::move(space_vector()), flat_index);
                    });
                auto list = grid_ptr->get_ilist_n_bnd(neighbor_dir);
                multiindex<> start_index = get_padding_start_indices(neighbor_dir);
                multiindex<> size = get_padding_real_size(neighbor_dir);
                size_t counter = 0;
                for (auto i : list) {
                    const integer iii = i.second;
                    const multiindex<> offset = flat_index_to_multiindex_not_padded(iii);
                    const multiindex<> m_padding_index(offset.x - start_index.x,
                        offset.y - start_index.y, offset.z - start_index.z);
                    const size_t flat_index = m_padding_index.x * (size.y * size.z) +
                        m_padding_index.y * size.z + m_padding_index.z;
                    local_expansions_SoA.set_AoS_value(
                        std::move(neighbor_M_ptr.at(counter)), flat_index);
                    center_of_masses_SoA.set_AoS_value(
                        std::move(neighbor_com0.at(counter)), flat_index);
                    counter++;
                }
            }
        }

        template <typename expansion_soa_container, typename masses_soa_container>
        bool p2m_interaction_interface::update_input(std::vector<multipole>& multipoles,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type,
            expansion_soa_container& local_expansions_SoA,
            masses_soa_container& center_of_masses_SoA) {
            std::vector<space_vector> const& com0 = *(com_ptr[0]);

            iterate_inner_cells_padded(
                [&center_of_masses_SoA, &local_expansions_SoA, multipoles, com0](
                    const multiindex<>& i, const size_t flat_index, const multiindex<>& i_unpadded,
                    const size_t flat_index_unpadded) {
                    center_of_masses_SoA.set_AoS_value(
                        std::move(com0.at(flat_index_unpadded)), flat_index);
                    local_expansions_SoA.set_AoS_value(std::move(expansion()), flat_index);
                });

            for (size_t i = 0; i < neighbor_empty_multipoles.size(); i++) {
                neighbor_empty_multipoles[i] = false;
            }

            multipole_neighbors_exist = false;

            // Now look at neighboring data
            for (const geo::direction& dir : geo::direction::full_set()) {
                // don't use neighbor.direction, is always zero for empty cells!
                neighbor_gravity_type& neighbor = neighbors[dir];
                auto x = dir.operator[](ZDIM) + 1;
                auto y = dir.operator[](YDIM) + 1;
                auto z = dir.operator[](XDIM) + 1;

                // this dir is setup as a multipole - and we only consider multipoles here
                if (!neighbor.is_monopole) {
                    // neighbor has no data - input structure just recevices zeros as padding
                    if (!neighbor.data.M) {
                        neighbor_empty_multipoles[dir.flat_index_with_center()] = true;
                        x_skip[z][y][x] = true;
                        iterate_inner_cells_padding(dir,
                            [&local_expansions_SoA, &center_of_masses_SoA](const multiindex<>& i,
                                const size_t flat_index, const multiindex<>&, const size_t) {
                                local_expansions_SoA.set_AoS_value(
                                    std::move(expansion()), flat_index);
                                center_of_masses_SoA.set_AoS_value(
                                    std::move(space_vector()), flat_index);
                            });
                    } else {
                        multipole_neighbors_exist = true;
                        x_skip[z][y][x] = false;
                        std::vector<multipole>& neighbor_M_ptr = *(neighbor.data.M);
                        std::vector<space_vector>& neighbor_com0 = *(neighbor.data.x);
                        const bool fullsizes = neighbor_M_ptr.size() == INNER_CELLS &&
                            neighbor_com0.size() == INNER_CELLS;
                        if (fullsizes) {
                            // Get multipole data into our input structure
                            iterate_inner_cells_padding(dir,
                                [&local_expansions_SoA, &center_of_masses_SoA, neighbor_M_ptr,
                                    neighbor_com0](const multiindex<>& i, const size_t flat_index,
                                    const multiindex<>& i_unpadded,
                                    const size_t flat_index_unpadded) {
                                    local_expansions_SoA.set_AoS_value(
                                        std::move(neighbor_M_ptr.at(flat_index_unpadded)),
                                        flat_index);
                                    center_of_masses_SoA.set_AoS_value(
                                        std::move(neighbor_com0.at(flat_index_unpadded)),
                                        flat_index);
                                });
                        } else {
                            iterate_inner_cells_padding(dir,
                                [&local_expansions_SoA, &center_of_masses_SoA](
                                    const multiindex<>& i, const size_t flat_index,
                                    const multiindex<>&, const size_t) {
                                    local_expansions_SoA.set_AoS_value(
                                        std::move(expansion()), flat_index);
                                    center_of_masses_SoA.set_AoS_value(
                                        std::move(space_vector()), flat_index);
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
                                local_expansions_SoA.set_AoS_value(
                                    std::move(neighbor_M_ptr.at(counter)), flat_index);
                                center_of_masses_SoA.set_AoS_value(
                                    std::move(neighbor_com0.at(counter)), flat_index);
                                counter++;
                            }
                        }
                    }
                } else {
                    neighbor_empty_multipoles[dir.flat_index_with_center()] = true;
                    x_skip[z][y][x] = true;
                    iterate_inner_cells_padding(dir,
                        [&local_expansions_SoA, &center_of_masses_SoA](const multiindex<>& i,
                            const size_t flat_index, const multiindex<>&, const size_t) {
                            local_expansions_SoA.set_AoS_value(std::move(expansion()), flat_index);
                            center_of_masses_SoA.set_AoS_value(
                                std::move(space_vector()), flat_index);
                        });
                }
            }
            x_skip[1][1][1] = true;
            for (auto zi = 0; zi < 3; ++zi) {
                z_skip[zi] = true;
                for (auto yi = 0; yi < 3; ++yi) {
                    y_skip[zi][yi] = true;
                    for (auto xi = 0; xi < 3; ++xi) {
                        if (!x_skip[zi][yi][xi]) {
                            y_skip[zi][yi] = false;
                            break;
                        }
                    }
                    if (!y_skip[zi][yi])
                        z_skip[zi] = false;
                }
            }
            neighbor_empty_multipoles[13] = true;
            return true;
        }
    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
