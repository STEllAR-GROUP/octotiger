#pragma once

#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/common_kernel/kernel_simd_types.hpp"
#include "octotiger/common_kernel/multiindex.hpp"
#include "octotiger/config/export_definitions.hpp"
#include "octotiger/geometry.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/interaction_types.hpp"
#include "octotiger/monopole_interactions/p2m_kernel.hpp"
#include "octotiger/taylor.hpp"

#include <array>
#include <memory>
#include <vector>

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        /// Interface for the monopole-multipole compute kernel
        class p2m_interaction_interface
        {
        public:
            p2m_interaction_interface(void);
            /** Takes AoS data, converts it, calculates monopole-multipole FMM interactions,
                stores results in L, L_c */
            void compute_p2m_interactions(std::vector<real>& monopoles,
                std::vector<multipole>& M_ptr,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type,
                std::array<bool, geo::direction::count()>& is_direction_empty);
            /// Sets the grid pointer - usually only required once
            void set_grid_ptr(std::shared_ptr<grid> ptr) {
                grid_ptr = ptr;
            }

        public:
            /// The stencil is used to identify the neighbors
            static OCTOTIGER_EXPORT std::vector<multiindex<>>& stencil();

        protected:
            /// Converts AoS input data into SoA data
            template <typename monopole_container, typename expansion_soa_container,
                typename masses_soa_container>
            void update_input(std::vector<real>& mons, std::vector<multipole>& multipoles,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type,
                monopole_container& local_monopoles, expansion_soa_container& local_expansions_SoA,
                masses_soa_container& center_of_masses_SoA);
            void compute_interactions(gsolve_type type,
                std::array<bool, geo::direction::count()>& is_direction_empty,
                std::vector<neighbor_gravity_type>& all_neighbor_interaction_data);

        private:
            static thread_local std::vector<real> local_monopoles_staging_area;
            /// Expansions for all the multipoles the current monopole is neighboring
            static thread_local struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>
                local_expansions_staging_area;
            /// com_ptr - Center of masses, required for the angular corrections
            static thread_local struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>
                center_of_masses_staging_area;

            bool multipole_neighbors_exist;
            std::vector<bool> neighbor_empty_multipoles;

            std::shared_ptr<grid> grid_ptr;
            interaction_kernel_type p2m_type;
            p2m_kernel kernel;

            bool z_skip[3];
            bool y_skip[3][3];
            bool x_skip[3][3][3];
        };

        template <typename monopole_container, typename expansion_soa_container,
            typename masses_soa_container>
        void p2m_interaction_interface::update_input(std::vector<real>& mons,
            std::vector<multipole>& multipoles,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type,
            monopole_container& local_monopoles, expansion_soa_container& local_expansions_SoA,
            masses_soa_container& center_of_masses_SoA) {
            std::vector<space_vector> const& com0 = *(com_ptr[0]);

            iterate_inner_cells_padded([&center_of_masses_SoA, &local_expansions_SoA,
                                        &local_monopoles, mons, multipoles, com0]
                                       (const multiindex<>& i, const size_t flat_index,
                const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                center_of_masses_SoA.set_AoS_value(
                    std::move(com0.at(flat_index_unpadded)), flat_index);
                local_expansions_SoA.set_AoS_value(
                    std::move(expansion()), flat_index);
                local_monopoles.at(flat_index) = mons.at(flat_index_unpadded);
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
                        iterate_inner_cells_padding(
                            dir, [&local_expansions_SoA, &center_of_masses_SoA, &local_monopoles]
                            (const multiindex<>& i, const size_t flat_index,
                             const multiindex<>&,
                             const size_t) {
                                local_expansions_SoA.set_AoS_value(
                                    std::move(expansion()), flat_index);
                                center_of_masses_SoA.set_AoS_value(
                                    std::move(space_vector()), flat_index);
                                local_monopoles.at(flat_index) = 0.0;
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
                            iterate_inner_cells_padding(
                                dir,
                                [&local_monopoles, &local_expansions_SoA, &center_of_masses_SoA,
                                    neighbor_M_ptr, neighbor_com0](const multiindex<>& i,
                                    const size_t flat_index, const multiindex<>& i_unpadded,
                                    const size_t flat_index_unpadded) {
                                    local_expansions_SoA.set_AoS_value(
                                        std::move(neighbor_M_ptr.at(flat_index_unpadded)),
                                        flat_index);
                                    center_of_masses_SoA.set_AoS_value(
                                        std::move(neighbor_com0.at(flat_index_unpadded)),
                                        flat_index);

                                    local_monopoles.at(flat_index) = 0.0;
                                });
                        } else {
                            iterate_inner_cells_padding(
                                dir, [&local_expansions_SoA, &center_of_masses_SoA, &local_monopoles]
                                (const multiindex<>& i, const size_t flat_index,
                                                        const multiindex<>&,
                                                        const size_t) {
                                    local_expansions_SoA.set_AoS_value(
                                        std::move(expansion()), flat_index);
                                    center_of_masses_SoA.set_AoS_value(
                                        std::move(space_vector()), flat_index);
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
                    iterate_inner_cells_padding(
                        dir, [&local_expansions_SoA, &center_of_masses_SoA, &local_monopoles]
                        (const multiindex<>& i, const size_t flat_index,
                            const multiindex<>&,
                            const size_t) {
                            local_expansions_SoA.set_AoS_value(
                                std::move(expansion()), flat_index);
                            center_of_masses_SoA.set_AoS_value(
                                std::move(space_vector()), flat_index);
                            local_monopoles.at(flat_index) = 0.0;
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
        }
    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
