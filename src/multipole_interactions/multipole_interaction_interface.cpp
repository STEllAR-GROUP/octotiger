#include "multipole_interaction_interface.hpp"

#include "../common_kernel/interactions_iterators.hpp"
#include "calculate_stencil.hpp"
#include "m2m_kernel.hpp"
#include "m2p_kernel.hpp"
#include "options.hpp"

#include <algorithm>

// Big picture questions:
// - use any kind of tiling?

extern options opts;

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        thread_local const two_phase_stencil multipole_interaction_interface::stencil =
            calculate_stencil();
        // thread_local std::vector<real> multipole_interaction_interface::local_monopoles(
        //     EXPANSION_COUNT_PADDED);
        // thread_local struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>
        //     multipole_interaction_interface::local_expansions_SoA;
        // thread_local struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>
        //     multipole_interaction_interface::center_of_masses_SoA;
        multipole_interaction_interface::multipole_interaction_interface(void) {
            local_monopoles_staging_area = std::vector<real>(ENTRIES);
            this->m2m_type = opts.m2m_kernel_type;
        }

        void multipole_interaction_interface::compute_multipole_interactions(
            std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::array<real, NDIM> xbase) {
            update_input(monopoles, M_ptr, com_ptr, neighbors, type, dx, xbase,
                local_monopoles_staging_area, local_expansions_staging_area,
                center_of_masses_staging_area);
            compute_interactions(is_direction_empty, neighbors, local_monopoles_staging_area,
                local_expansions_staging_area, center_of_masses_staging_area);
        }

        void multipole_interaction_interface::update_input(std::vector<real>& monopoles,
            std::vector<multipole>& M_ptr,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type t, real dx,
            std::array<real, NDIM> xbase, std::vector<real>& local_monopoles,
            struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>& local_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                center_of_masses_SoA) {
            type = t;
            dX = dx;
            xBase = xbase;
            std::vector<space_vector> const& com0 = *(com_ptr[0]);

            iterate_inner_cells_padded([M_ptr, com0, &local_expansions_SoA, &center_of_masses_SoA,
                &local_monopoles](const multiindex<>& i, const size_t flat_index,
                const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                // local_expansions.at(flat_index) = M_ptr.at(flat_index_unpadded);
                // center_of_masses.at(flat_index) = com0.at(flat_index_unpadded);
                local_expansions_SoA.set_AoS_value(
                    std::move(M_ptr.at(flat_index_unpadded)), flat_index);
                center_of_masses_SoA.set_AoS_value(
                    std::move(com0.at(flat_index_unpadded)), flat_index);
                local_monopoles.at(flat_index) = 0.0;
            });

            for (const geo::direction& dir : geo::direction::full_set()) {
                // don't use neighbor.direction, is always zero for empty cells!
                neighbor_gravity_type& neighbor = neighbors[dir];
                // Switch x and z dimension since the stencil code and the old octotiger code
                // a different order for some reason
                auto x = dir.operator[](ZDIM) + 1;
                auto y = dir.operator[](YDIM) + 1;
                auto z = dir.operator[](XDIM) + 1;

                // this dir is setup as a multipole
                if (!neighbor.is_monopole) {
                    if (!neighbor.data.M) {
                        // TODO: ask Dominic why !is_monopole and stuff still empty
                        iterate_inner_cells_padding(
                            dir, [&local_expansions_SoA, &center_of_masses_SoA, &local_monopoles](
                                     const multiindex<>& i, const size_t flat_index,
                                     const multiindex<>&, const size_t) {
                                // // initializes whole expansion, relatively expansion
                                // local_expansions.at(flat_index) = 0.0;
                                // // initializes x,y,z vector
                                // center_of_masses.at(flat_index) = 0.0;
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
                                    // local_expansions.at(flat_index) =
                                    //     neighbor_M_ptr.at(flat_index_unpadded);
                                    // center_of_masses.at(flat_index) =
                                    //     neighbor_com0.at(flat_index_unpadded);
                                    local_expansions_SoA.set_AoS_value(
                                        std::move(neighbor_M_ptr.at(flat_index_unpadded)),
                                        flat_index);
                                    center_of_masses_SoA.set_AoS_value(
                                        std::move(neighbor_com0.at(flat_index_unpadded)),
                                        flat_index);
                                    local_monopoles.at(flat_index) = 0.0;

                                });
                        } else {
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
                                // initializes whole expansion, relatively expansion
                                // local_expansions.at(flat_index) = 0.0;
                                // // initializes x,y,z vector
                                // center_of_masses.at(flat_index) = 0.0;
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
                                    // local_expansions.at(flat_index) = 0.0;
                                    // center_of_masses.at(flat_index) = 0.0;

                                    space_vector e;
                                    e[0] = (i.x) * dx + xbase[0] - INNER_CELLS_PER_DIRECTION * dx;
                                    e[1] = (i.y) * dx + xbase[1] - INNER_CELLS_PER_DIRECTION * dx;
                                    e[2] = (i.z) * dx + xbase[2] - INNER_CELLS_PER_DIRECTION * dx;
                                    center_of_masses_SoA.set_AoS_value(std::move(e), flat_index);
                                    // local_monopoles.at(flat_index) =
                                    // neighbor_mons.at(flat_index_unpadded);
                                    local_expansions_SoA.set_AoS_value(
                                        std::move(expansion()), flat_index);
                                    // local_expansions_SoA.set_value(
                                    //     std::move(neighbor_mons.at(flat_index_unpadded)),
                                    //     flat_index);
                                    local_monopoles.at(flat_index) =
                                        neighbor_mons.at(flat_index_unpadded);
                                });
                        } else {
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
                                });
                        }
                    }
                }
            }
        }

        void multipole_interaction_interface::compute_interactions(
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::vector<neighbor_gravity_type>& all_neighbor_interaction_data,
            const std::vector<real>& local_monopoles,
            const struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>&
                local_expansions_SoA,
            const struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                center_of_masses_SoA) {
            if (m2m_type == interaction_kernel_type::SOA_CPU) {
                struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>
                    potential_expansions_SoA;
                struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>
                    angular_corrections_SoA;

                m2m_kernel kernel;
                kernel.apply_stencil(local_expansions_SoA, center_of_masses_SoA,
                    potential_expansions_SoA, angular_corrections_SoA, local_monopoles, stencil,
                    type);

                if (type == RHO) {
                    angular_corrections_SoA.to_non_SoA(grid_ptr->get_L_c());
                }
                potential_expansions_SoA.add_to_non_SoA(grid_ptr->get_L());
            } else {
                // old-style interaction calculation
                // computes inner interactions
                grid_ptr->compute_interactions(type);
                // computes boundary interactions
                for (auto const& dir : geo::direction::full_set()) {
                    if (!is_direction_empty[dir]) {
                        neighbor_gravity_type& neighbor_data = all_neighbor_interaction_data[dir];
                        grid_ptr->compute_boundary_interactions(type, neighbor_data.direction,
                            neighbor_data.is_monopole, neighbor_data.data);
                    }
                }
            }
        }
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
