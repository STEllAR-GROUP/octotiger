//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/common_kernel/interactions_iterators.hpp"
#include "octotiger/common_kernel/multiindex.hpp"
#include "octotiger/common_kernel/struct_of_array_data.hpp"
#include "octotiger/config/export_definitions.hpp"
#include "octotiger/geometry.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/interaction_types.hpp"
#include "octotiger/taylor.hpp"

#include <array>
#include <memory>
#include <vector>

#include <aligned_buffer_util.hpp>
#include <buffer_manager.hpp>

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        /// Interface to the SoA FMM interaction kernels
        class multipole_interaction_interface
        {
        public:
            multipole_interaction_interface();
            /// Takes AoS data, converts it, calculates FMM interactions, stores results in L, L_c
            void compute_multipole_interactions(std::vector<real>& monopoles,
                std::vector<multipole>& M_ptr,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
                std::array<bool, geo::direction::count()>& is_direction_empty,
                std::array<real, NDIM> xbase, const bool use_root_stencil);
            /// Sets the grid pointer - usually only required once
            void set_grid_ptr(std::shared_ptr<grid> ptr) {
                grid_ptr = ptr;
            }

        public:
            static OCTOTIGER_EXPORT size_t& cpu_launch_counter();
            static OCTOTIGER_EXPORT size_t& cuda_launch_counter();
            static OCTOTIGER_EXPORT size_t& cpu_launch_counter_non_rho();
            static OCTOTIGER_EXPORT size_t& cuda_launch_counter_non_rho();

        protected:
            /// Calls FMM kernels with SoA data (assumed to be stored in the static members)
            void compute_interactions(std::array<bool, geo::direction::count()>& is_direction_empty,
                std::vector<neighbor_gravity_type>& all_neighbor_interaction_data,
                const cpu_monopole_buffer_t& local_monopoles,
                const cpu_expansion_buffer_t& local_expansions_SoA,
                const cpu_space_vector_buffer_t& center_of_masses_SoA, const bool use_root_stencil);

        protected:
            gsolve_type type;
            real dX;
            /// Needed for the center of masses calculation for multipole-monopole interactions
            std::array<real, NDIM> xBase;
            std::shared_ptr<grid> grid_ptr;
            /// Option whether SoA Kernels should be called or the old AoS methods
            interaction_kernel_type m2m_type;

        public:
            /// Stencil for stencil based FMM kernels
            static OCTOTIGER_EXPORT two_phase_stencil& stencil();
            static OCTOTIGER_EXPORT std::vector<bool>& stencil_masks();
            static OCTOTIGER_EXPORT std::vector<bool>& inner_stencil_masks();
        };

        template <size_t padded_entries_per_component, size_t num_components, typename container_t,
            typename AoS_temp_type>
        void set_AoS_value(container_t& buffer, AoS_temp_type&& value, size_t flatindex) {
            for (size_t component = 0; component < num_components; component++) {
                buffer[component * padded_entries_per_component + flatindex] = value[component];
            }
        }

        /// Converts AoS input data into SoA data
        template <typename monopole_container, typename expansion_soa_container,
            typename masses_soa_container>
        void update_input(std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type t, real dx,
            std::array<real, NDIM> xbase, monopole_container& local_monopoles,
            expansion_soa_container& local_expansions_SoA,
            masses_soa_container& center_of_masses_SoA, std::shared_ptr<grid>& grid_ptr,
            bool is_root) {
            std::vector<space_vector> const& com0 = *(com_ptr[0]);

            if (is_root) {
                iterate_inner_cells_padded(
                    [M_ptr, com0, &local_expansions_SoA, &center_of_masses_SoA](
                        const multiindex<>& i, const size_t flat_index,
                        const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                        set_AoS_value<ENTRIES + SOA_PADDING, 20>(local_expansions_SoA,
                            std::move(M_ptr.at(flat_index_unpadded)), flat_index);
                        set_AoS_value<ENTRIES + SOA_PADDING, 3>(center_of_masses_SoA,
                            std::move(com0.at(flat_index_unpadded)), flat_index);
                    });
            } else {
                iterate_inner_cells_padded(
                    [M_ptr, com0, &local_expansions_SoA, &center_of_masses_SoA, &local_monopoles](
                        const multiindex<>& i, const size_t flat_index,
                        const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                        set_AoS_value<ENTRIES + SOA_PADDING, 20>(local_expansions_SoA,
                            std::move(M_ptr.at(flat_index_unpadded)), flat_index);
                        set_AoS_value<ENTRIES + SOA_PADDING, 3>(center_of_masses_SoA,
                            std::move(com0.at(flat_index_unpadded)), flat_index);
                        local_monopoles[flat_index] = 0.0;
                    });

                for (const geo::direction& dir : geo::direction::full_set()) {
                    // don't use neighbor.direction, is always zero for empty cells!
                    neighbor_gravity_type& neighbor = neighbors[dir];

                    // this dir is setup as a multipole
                    if (!neighbor.is_monopole) {
                        if (!neighbor.data.M) {
                            iterate_inner_cells_padding(dir,
                                [&local_expansions_SoA, &center_of_masses_SoA, &local_monopoles](
                                    const multiindex<>& i, const size_t flat_index,
                                    const multiindex<>&, const size_t) {
                                    set_AoS_value<ENTRIES + SOA_PADDING, 20>(
                                        local_expansions_SoA, std::move(expansion()), flat_index);
                                    set_AoS_value<ENTRIES + SOA_PADDING, 3>(center_of_masses_SoA,
                                        std::move(space_vector()), flat_index);
                                    local_monopoles[flat_index] = 0.0;
                                });
                        } else {
                            std::vector<multipole>& neighbor_M_ptr = *(neighbor.data.M);
                            std::vector<space_vector>& neighbor_com0 = *(neighbor.data.x);
                            const bool fullsizes = neighbor_M_ptr.size() == INNER_CELLS &&
                                neighbor_com0.size() == INNER_CELLS;
                            if (fullsizes) {
                                iterate_inner_cells_padding(dir,
                                    [&local_expansions_SoA, &center_of_masses_SoA, &local_monopoles,
                                        neighbor_M_ptr, neighbor_com0](const multiindex<>& i,
                                        const size_t flat_index, const multiindex<>& i_unpadded,
                                        const size_t flat_index_unpadded) {
                                        set_AoS_value<ENTRIES + SOA_PADDING, 20>(
                                            local_expansions_SoA,
                                            std::move(neighbor_M_ptr.at(flat_index_unpadded)),
                                            flat_index);
                                        set_AoS_value<ENTRIES + SOA_PADDING, 3>(
                                            center_of_masses_SoA,
                                            std::move(neighbor_com0.at(flat_index_unpadded)),
                                            flat_index);
                                        local_monopoles[flat_index] = 0.0;
                                    });
                            } else {
                                // Reset to default values
                                iterate_inner_cells_padding(dir,
                                    [&local_expansions_SoA, &center_of_masses_SoA,
                                        &local_monopoles](const multiindex<>& i,
                                        const size_t flat_index, const multiindex<>&,
                                        const size_t) {
                                        set_AoS_value<ENTRIES + SOA_PADDING, 20>(
                                            local_expansions_SoA, std::move(expansion()),
                                            flat_index);
                                        set_AoS_value<ENTRIES + SOA_PADDING, 3>(
                                            center_of_masses_SoA, std::move(space_vector()),
                                            flat_index);
                                        local_monopoles[flat_index] = 0.0;
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
                                    set_AoS_value<ENTRIES + SOA_PADDING, 20>(local_expansions_SoA,
                                        std::move(neighbor_M_ptr.at(counter)), flat_index);
                                    set_AoS_value<ENTRIES + SOA_PADDING, 3>(center_of_masses_SoA,
                                        std::move(neighbor_com0.at(counter)), flat_index);
                                    counter++;
                                }
                            }
                        }
                    } else {
                        // neighbor has no data - input structure just recevices zeros as padding
                        if (!neighbor.data.m) {
                            iterate_inner_cells_padding(dir,
                                [&local_expansions_SoA, &center_of_masses_SoA, &local_monopoles](
                                    const multiindex<>& i, const size_t flat_index,
                                    const multiindex<>&, const size_t) {
                                    set_AoS_value<ENTRIES + SOA_PADDING, 20>(
                                        local_expansions_SoA, std::move(expansion()), flat_index);
                                    set_AoS_value<ENTRIES + SOA_PADDING, 3>(center_of_masses_SoA,
                                        std::move(space_vector()), flat_index);
                                    local_monopoles[flat_index] = 0.0;
                                });
                        } else {
                            // Get multipole data into our input structure
                            std::vector<real>& neighbor_mons = *(neighbor.data.m);
                            const bool fullsizes = neighbor_mons.size() == INNER_CELLS;
                            if (fullsizes) {
                                iterate_inner_cells_padding(dir,
                                    [&local_expansions_SoA, &center_of_masses_SoA, &local_monopoles,
                                        neighbor_mons, xbase, dx](const multiindex<>& i,
                                        const size_t flat_index, const multiindex<>& i_unpadded,
                                        const size_t flat_index_unpadded) {
                                        space_vector e;
                                        e[0] =
                                            (i.x) * dx + xbase[0] - INNER_CELLS_PER_DIRECTION * dx;
                                        e[1] =
                                            (i.y) * dx + xbase[1] - INNER_CELLS_PER_DIRECTION * dx;
                                        e[2] =
                                            (i.z) * dx + xbase[2] - INNER_CELLS_PER_DIRECTION * dx;
                                        set_AoS_value<ENTRIES + SOA_PADDING, 3>(
                                            center_of_masses_SoA, std::move(e), flat_index);
                                        set_AoS_value<ENTRIES + SOA_PADDING, 20>(
                                            local_expansions_SoA, std::move(expansion()),
                                            flat_index);
                                        local_monopoles[flat_index] =
                                            neighbor_mons.at(flat_index_unpadded);
                                    });
                            } else {
                                // Reset to default values
                                iterate_inner_cells_padding(dir,
                                    [&local_expansions_SoA, &center_of_masses_SoA, &local_monopoles,
                                        dx, xbase](const multiindex<>& i, const size_t flat_index,
                                        const multiindex<>&, const size_t) {
                                        space_vector e;
                                        e[0] =
                                            (i.x) * dx + xbase[0] - INNER_CELLS_PER_DIRECTION * dx;
                                        e[1] =
                                            (i.y) * dx + xbase[1] - INNER_CELLS_PER_DIRECTION * dx;
                                        e[2] =
                                            (i.z) * dx + xbase[2] - INNER_CELLS_PER_DIRECTION * dx;
                                        set_AoS_value<ENTRIES + SOA_PADDING, 3>(
                                            center_of_masses_SoA, std::move(e), flat_index);
                                        set_AoS_value<ENTRIES + SOA_PADDING, 20>(
                                            local_expansions_SoA, std::move(expansion()),
                                            flat_index);
                                        local_monopoles[flat_index] = 0.0;
                                    });
                                // Load relevant values
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
                                    local_monopoles[flat_index] = neighbor_mons.at(counter);
                                    counter++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
