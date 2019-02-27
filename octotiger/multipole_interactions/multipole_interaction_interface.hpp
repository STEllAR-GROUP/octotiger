#pragma once

#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/common_kernel/interactions_iterators.hpp"
#include "octotiger/common_kernel/kernel_simd_types.hpp"
#include "octotiger/common_kernel/multiindex.hpp"
#include "octotiger/common_kernel/struct_of_array_data.hpp"

#include "octotiger/geometry.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/interaction_types.hpp"
#include "octotiger/taylor.hpp"

#include <array>
#include <memory>
#include <vector>

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        /// Interface to the SoA FMM interaction kernels
        class multipole_interaction_interface
        {
        public:
            multipole_interaction_interface(void);
            /// Takes AoS data, converts it, calculates FMM interactions, stores results in L, L_c
            void compute_multipole_interactions(std::vector<real>& monopoles,
                std::vector<multipole>& M_ptr,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
                std::array<bool, geo::direction::count()>& is_direction_empty,
                std::array<real, NDIM> xbase);
            /// Sets the grid pointer - usually only required once
            void set_grid_ptr(std::shared_ptr<grid> ptr) {
                grid_ptr = ptr;
            }

        protected:
            /// Converts AoS input data into SoA data
            template <typename monopole_container, typename expansion_soa_container,
                typename masses_soa_container>
            void update_input(std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type t, real dx,
                std::array<real, NDIM> xbase, monopole_container& local_monopoles,
                expansion_soa_container& local_expansions_SoA,
                masses_soa_container& center_of_masses_SoA);
            /// Calls FMM kernels with SoA data (assumed to be stored in the static members)
            void compute_interactions(std::array<bool, geo::direction::count()>& is_direction_empty,
                std::vector<neighbor_gravity_type>& all_neighbor_interaction_data,
                const std::vector<real>& local_monopoles,
                const struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>&
                    local_expansions_SoA,
                const struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                    center_of_masses_SoA);

        protected:
            gsolve_type type;
            real dX;
            /// Needed for the center of masses calculation for multipole-monopole interactions
            std::array<real, NDIM> xBase;
            std::shared_ptr<grid> grid_ptr;
            /// Option whether SoA Kernels should be called or the old AoS methods
            interaction_kernel_type m2m_type;

        private:
            /// SoA conversion area - used as input for compute_interactions
            static thread_local std::vector<real> local_monopoles_staging_area;
            /// SoA conversion area - used as input for compute_interactions
            static thread_local struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>
                local_expansions_staging_area;
            /// SoA conversion area - used as input for compute_interactions
            static thread_local struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>
                center_of_masses_staging_area;
            static thread_local bool is_initialized;
        public:
            /// Stencil for stencil based FMM kernels
            static thread_local two_phase_stencil stencil;
            static std::vector<bool>& stencil_masks();
            static std::vector<bool>& inner_stencil_masks();
        };

        template <typename monopole_container, typename expansion_soa_container,
            typename masses_soa_container>
        void multipole_interaction_interface::update_input(std::vector<real>& monopoles,
            std::vector<multipole>& M_ptr,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type t, real dx,
            std::array<real, NDIM> xbase, monopole_container& local_monopoles,
            expansion_soa_container& local_expansions_SoA,
            masses_soa_container& center_of_masses_SoA) {
            type = t;
            dX = dx;
            xBase = xbase;
            std::vector<space_vector> const& com0 = *(com_ptr[0]);

            iterate_inner_cells_padded([M_ptr, com0, &local_expansions_SoA, &center_of_masses_SoA,
                &local_monopoles](const multiindex<>& i, const size_t flat_index,
                const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                local_expansions_SoA.set_AoS_value(
                    std::move(M_ptr.at(flat_index_unpadded)), flat_index);
                center_of_masses_SoA.set_AoS_value(
                    std::move(com0.at(flat_index_unpadded)), flat_index);
                local_monopoles.at(flat_index) = 0.0;
            });

            for (const geo::direction& dir : geo::direction::full_set()) {
                // don't use neighbor.direction, is always zero for empty cells!
                neighbor_gravity_type& neighbor = neighbors[dir];

                // this dir is setup as a multipole
                if (!neighbor.is_monopole) {
                    if (!neighbor.data.M) {
                        iterate_inner_cells_padding(
                            dir, [&local_expansions_SoA, &center_of_masses_SoA, &local_monopoles](
                                     const multiindex<>& i, const size_t flat_index,
                                     const multiindex<>&, const size_t) {
                                local_expansions_SoA.set_AoS_value(
                                    std::move(expansion()), flat_index);
                                center_of_masses_SoA.set_AoS_value(
                                    std::move(space_vector()), flat_index);
                                local_monopoles.at(flat_index) = 0.0;
                            });
                    } else {
                        std::vector<multipole>& neighbor_M_ptr = *(neighbor.data.M);
                        std::vector<space_vector>& neighbor_com0 = *(neighbor.data.x);
                        const bool fullsizes = neighbor_M_ptr.size() == INNER_CELLS &&
                            neighbor_com0.size() == INNER_CELLS;
                        if (fullsizes) {
                            iterate_inner_cells_padding(
                                dir,
                                [&local_expansions_SoA, &center_of_masses_SoA, &local_monopoles,
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
                            // Reset to default values
                            iterate_inner_cells_padding(
                                dir, [&local_expansions_SoA, &center_of_masses_SoA, &local_monopoles](
                                    const multiindex<>& i, const size_t flat_index,
                                    const multiindex<>&, const size_t) {
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
                    // neighbor has no data - input structure just recevices zeros as padding
                    if (!neighbor.data.m) {
                        iterate_inner_cells_padding(
                            dir, [&local_expansions_SoA, &center_of_masses_SoA, &local_monopoles](
                                const multiindex<>& i, const size_t flat_index,
                                const multiindex<>&, const size_t) {
                                local_expansions_SoA.set_AoS_value(
                                    std::move(expansion()), flat_index);
                                center_of_masses_SoA.set_AoS_value(
                                    std::move(space_vector()), flat_index);
                                local_monopoles.at(flat_index) = 0.0;
                            });
                    } else {
                        // Get multipole data into our input structure
                        std::vector<real>& neighbor_mons = *(neighbor.data.m);
                        const bool fullsizes = neighbor_mons.size() == INNER_CELLS;
                        if (fullsizes) {
                            iterate_inner_cells_padding(
                                dir,
                                [&local_expansions_SoA, &center_of_masses_SoA, &local_monopoles,
                                 neighbor_mons, xbase, dx](const multiindex<>& i,
                                                           const size_t flat_index, const multiindex<>& i_unpadded,
                                                           const size_t flat_index_unpadded) {
                                    space_vector e;
                                    e[0] = (i.x) * dx + xbase[0] - INNER_CELLS_PER_DIRECTION * dx;
                                    e[1] = (i.y) * dx + xbase[1] - INNER_CELLS_PER_DIRECTION * dx;
                                    e[2] = (i.z) * dx + xbase[2] - INNER_CELLS_PER_DIRECTION * dx;
                                    center_of_masses_SoA.set_AoS_value(std::move(e), flat_index);
                                    local_expansions_SoA.set_AoS_value(
                                        std::move(expansion()), flat_index);
                                    local_monopoles.at(flat_index) =
                                        neighbor_mons.at(flat_index_unpadded);
                                });
                        } else {
                            // Reset to default values
                            iterate_inner_cells_padding(
                                dir, [&local_expansions_SoA, &center_of_masses_SoA,
                                      &local_monopoles, dx, xbase](
                                    const multiindex<>& i, const size_t flat_index,
                                    const multiindex<>&, const size_t) {
                                    space_vector e;
                                    e[0] = (i.x) * dx + xbase[0] - INNER_CELLS_PER_DIRECTION * dx;
                                    e[1] = (i.y) * dx + xbase[1] - INNER_CELLS_PER_DIRECTION * dx;
                                    e[2] = (i.z) * dx + xbase[2] - INNER_CELLS_PER_DIRECTION * dx;
                                    center_of_masses_SoA.set_AoS_value(std::move(e), flat_index);
                                    local_expansions_SoA.set_AoS_value(
                                        std::move(expansion()), flat_index);
                                    local_monopoles.at(flat_index) = 0.0;

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
                                local_monopoles.at(flat_index) = neighbor_mons.at(counter);
                                counter++;
                            }
                        }
                    }
                }
            }
        }
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
