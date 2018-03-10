#include "multipole_interaction_interface.hpp"

#include "../common_kernel/interactions_iterators.hpp"
#include "calculate_stencil.hpp"
#include "m2m_kernel.hpp"
#include "m2p_kernel.hpp"

#include <algorithm>

// Big picture questions:
// - use any kind of tiling?

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        thread_local const two_phase_stencil multipole_interaction_interface::stencil =
            calculate_stencil();
        thread_local std::vector<real> multipole_interaction_interface::local_monopoles(
            EXPANSION_COUNT_PADDED);
        thread_local struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>
            multipole_interaction_interface::local_expansions_SoA;
        thread_local struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>
            multipole_interaction_interface::center_of_masses_SoA;
        multipole_interaction_interface::multipole_interaction_interface(void)
          : neighbor_empty_multipole(27)
          , neighbor_empty_monopole(27)
          , mixed_interactions_kernel(neighbor_empty_monopole) {}

        void multipole_interaction_interface::update_input(std::vector<real>& monopoles,
            std::vector<multipole>& M_ptr,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type t, real dx,
            std::array<real, NDIM> xbase) {
            type = t;
            dX = dx;
            xBase = xbase;
            std::vector<space_vector> const& com0 = *(com_ptr[0]);

            iterate_inner_cells_padded(
                [this, M_ptr, com0](const multiindex<>& i, const size_t flat_index,
                    const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                    // local_expansions.at(flat_index) = M_ptr.at(flat_index_unpadded);
                    // center_of_masses.at(flat_index) = com0.at(flat_index_unpadded);
                    local_expansions_SoA.set_AoS_value(
                        std::move(M_ptr.at(flat_index_unpadded)), flat_index);
                    center_of_masses_SoA.set_AoS_value(
                        std::move(com0.at(flat_index_unpadded)), flat_index);
                    local_monopoles.at(flat_index) = 0.0;
                });

            for (size_t i = 0; i < neighbor_empty_multipole.size(); i++) {
                neighbor_empty_multipole[i] = false;
                neighbor_empty_monopole[i] = false;
            }

            monopole_neighbors_exist = false;
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
                    neighbor_empty_monopole[dir.flat_index_with_center()] = true;
                    x_skip[z][y][x] = true;
                    if (!neighbor.data.M) {
                        // TODO: ask Dominic why !is_monopole and stuff still empty
                        iterate_inner_cells_padding(dir, [this](const multiindex<>& i,
                                                             const size_t flat_index,
                                                             const multiindex<>&, const size_t) {
                            // // initializes whole expansion, relatively expansion
                            // local_expansions.at(flat_index) = 0.0;
                            // // initializes x,y,z vector
                            // center_of_masses.at(flat_index) = 0.0;
                            local_expansions_SoA.set_AoS_value(std::move(expansion()), flat_index);
                            center_of_masses_SoA.set_AoS_value(
                                std::move(space_vector()), flat_index);
                            local_monopoles.at(flat_index) = 0.0;
                        });
                        neighbor_empty_multipole[dir.flat_index_with_center()] = true;
                    } else {
                        std::vector<multipole>& neighbor_M_ptr = *(neighbor.data.M);
                        std::vector<space_vector>& neighbor_com0 = *(neighbor.data.x);
                        const bool fullsizes = neighbor_M_ptr.size() == INNER_CELLS &&
                            neighbor_com0.size() == INNER_CELLS;
                        if (fullsizes) {
                            iterate_inner_cells_padding(dir, [this, neighbor_M_ptr, neighbor_com0](
                                                                 const multiindex<>& i,
                                                                 const size_t flat_index,
                                                                 const multiindex<>& i_unpadded,
                                                                 const size_t flat_index_unpadded) {
                                // local_expansions.at(flat_index) =
                                //     neighbor_M_ptr.at(flat_index_unpadded);
                                // center_of_masses.at(flat_index) =
                                //     neighbor_com0.at(flat_index_unpadded);
                                local_expansions_SoA.set_AoS_value(
                                    std::move(neighbor_M_ptr.at(flat_index_unpadded)), flat_index);
                                center_of_masses_SoA.set_AoS_value(
                                    std::move(neighbor_com0.at(flat_index_unpadded)), flat_index);
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
                    neighbor_empty_multipole[dir.flat_index_with_center()] = true;
                    // neighbor has no data - input structure just recevices zeros as padding
                    if (!neighbor.data.m) {
                        iterate_inner_cells_padding(dir, [this](const multiindex<>& i,
                                                             const size_t flat_index,
                                                             const multiindex<>&, const size_t) {
                            // initializes whole expansion, relatively expansion
                            // local_expansions.at(flat_index) = 0.0;
                            // // initializes x,y,z vector
                            // center_of_masses.at(flat_index) = 0.0;
                            local_expansions_SoA.set_AoS_value(std::move(expansion()), flat_index);
                            center_of_masses_SoA.set_AoS_value(
                                std::move(space_vector()), flat_index);
                            local_monopoles.at(flat_index) = 0.0;

                        });
                        neighbor_empty_monopole[dir.flat_index_with_center()] = true;
                        x_skip[z][y][x] = true;
                    } else {
                        // Get multipole data into our input structure
                        std::vector<real>& neighbor_mons = *(neighbor.data.m);
                        const bool fullsizes = neighbor_mons.size() == INNER_CELLS;
                        if (fullsizes) {
                            iterate_inner_cells_padding(
                                dir, [this, neighbor_mons, xbase, dx](const multiindex<>& i,
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
                                dir, [this, neighbor_mons, xbase, dx](const multiindex<>& i,
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
                        monopole_neighbors_exist = true;
                        x_skip[z][y][x] = false;
                    }
                }
            }

            neighbor_empty_multipole[13] = false;
            neighbor_empty_monopole[13] = true;

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

            // std::fill(std::begin(potential_expansions), std::end(potential_expansions),
            // ZERO);

            // std::cout << "local_expansions:" << std::endl;
            // this->print_local_expansions();
            // std::cout << "center_of_masses:" << std::endl;
            // this->print_center_of_masses();
        }

        void multipole_interaction_interface::compute_interactions(interaction_kernel_type m2m_type,
            interaction_kernel_type m2p_type,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::vector<neighbor_gravity_type>& all_neighbor_interaction_data) {
            if (m2m_type == interaction_kernel_type::SOA_CPU &&
                m2p_type == interaction_kernel_type::SOA_CPU) {
                struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>
                    potential_expansions_SoA;
                struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>
                    angular_corrections_SoA;

                // if (monopole_neighbors_exist) {
                //     mixed_interactions_kernel.apply_stencil(local_monopoles,
                //     local_expansions_SoA,
                //         center_of_masses_SoA, potential_expansions_SoA,
                //         angular_corrections_SoA,
                //         stencil_mixed_interactions, type, dX, xBase, x_skip, y_skip, z_skip);
                // }
                m2m_kernel kernel(neighbor_empty_multipole);
                kernel.apply_stencil(local_expansions_SoA, center_of_masses_SoA,
                    potential_expansions_SoA, angular_corrections_SoA, local_monopoles, stencil,
                    type);

                if (type == RHO) {
                    angular_corrections_SoA.to_non_SoA(grid_ptr->get_L_c());
                }

                potential_expansions_SoA.add_to_non_SoA(grid_ptr->get_L());

            } else if (m2m_type == interaction_kernel_type::SOA_CPU) {
                struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>
                    potential_expansions_SoA;
                struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>
                    angular_corrections_SoA;
                m2m_kernel kernel(neighbor_empty_multipole);
                kernel.apply_stencil(local_expansions_SoA, center_of_masses_SoA,
                    potential_expansions_SoA, angular_corrections_SoA, local_monopoles, stencil,
                    type);

                std::vector<expansion>& L = grid_ptr->get_L();
                std::vector<space_vector>& L_c = grid_ptr->get_L_c();
                std::fill(std::begin(L), std::end(L), ZERO);
                std::fill(std::begin(L_c), std::end(L_c), ZERO);
                for (auto const& dir : geo::direction::full_set()) {
                    if (!is_direction_empty[dir]) {
                        neighbor_gravity_type& neighbor_data = all_neighbor_interaction_data[dir];
                        if (neighbor_data.is_monopole) {
                            grid_ptr->compute_boundary_interactions(type, neighbor_data.direction,
                                neighbor_data.is_monopole, neighbor_data.data);
                        }
                    }
                }
                potential_expansions_SoA.add_to_non_SoA(grid_ptr->get_L());
                angular_corrections_SoA.add_to_non_SoA(grid_ptr->get_L_c());

            } else if (m2p_type == interaction_kernel_type::SOA_CPU) {
                struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>
                    potential_expansions_SoA;
                struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>
                    angular_corrections_SoA;
                // if (monopole_neighbors_exist) {
                //     mixed_interactions_kernel.apply_stencil(local_monopoles,
                //     local_expansions_SoA,
                //         center_of_masses_SoA, potential_expansions_SoA,
                //         angular_corrections_SoA,
                //         stencil_mixed_interactions, type, dX, xBase, x_skip, y_skip, z_skip);
                // }
                m2m_kernel kernel(neighbor_empty_multipole);
                kernel.apply_stencil(local_expansions_SoA, center_of_masses_SoA,
                    potential_expansions_SoA, angular_corrections_SoA, local_monopoles, stencil,
                    type);

                std::vector<expansion>& L = grid_ptr->get_L();
                std::vector<space_vector>& L_c = grid_ptr->get_L_c();
                std::fill(std::begin(L), std::end(L), ZERO);
                std::fill(std::begin(L_c), std::end(L_c), ZERO);

                grid_ptr->compute_interactions(type);
                // waits for boundary data and then computes boundary interactions
                for (auto const& dir : geo::direction::full_set()) {
                    if (!is_direction_empty[dir]) {
                        neighbor_gravity_type& neighbor_data = all_neighbor_interaction_data[dir];
                        if (!neighbor_data.is_monopole) {
                            grid_ptr->compute_boundary_interactions(type, neighbor_data.direction,
                                neighbor_data.is_monopole, neighbor_data.data);
                        }
                    }
                }
                potential_expansions_SoA.add_to_non_SoA(grid_ptr->get_L());
                angular_corrections_SoA.add_to_non_SoA(grid_ptr->get_L_c());
            } else {
                // old-style interaction calculation
                // computes inner interactions
                grid_ptr->compute_interactions(type);
                // waits for boundary data and then computes boundary interactions
                for (auto const& dir : geo::direction::full_set()) {
                    if (!is_direction_empty[dir]) {
                        neighbor_gravity_type& neighbor_data = all_neighbor_interaction_data[dir];
                        grid_ptr->compute_boundary_interactions(type, neighbor_data.direction,
                            neighbor_data.is_monopole, neighbor_data.data);
                    }
                }
            }
        }
        // template <typename T>
        // inline void interactions_kernel_non_rho(T* cell_com, T* tmpstore, T* m_partner, T* cur_pot,
        //     T* X, T* Y, const T* factor_half_v, const T* factor_sixth_v) {
        //     T dX[NDIM];
        //     dX[0] = X[0] - Y[0];
        //     dX[1] = X[1] - Y[1];
        //     dX[2] = X[2] - Y[2];
        //     double X_00;
        //     double X_11;
        //     double X_22;

        //     double r2;
        //     double r2inv;

        //     double d0;
        //     double d1;
        //     double d2;
        //     double d3;
        //         X_00 = dX[0] * dX[0];
        //         X_11 = dX[1] * dX[1];
        //         X_22 = dX[2] * dX[2];

        //         r2 = X_00 + X_11 + X_22;
        //         r2inv = 1.0 / max(r2, 1.0e-20);

        //         // parts of formula (6)
        //         d0 = -sqrt(r2inv);
        //         // parts of formula (7)
        //         d1 = -d0 * r2inv;
        //         // parts of formula (8)
        //         d2 = -3.0 * d1 * r2inv;
        //         // parts of  formula (9)
        //         d3 = -5.0 * d2 * r2inv;
        //     D_split D_calculator(dX);
        //     // TODO: this translates to an initialization loop, bug!
        //     D_calculator.calculate_D_lower(D_lower);

        //     // 10-19 are not cached!

        //     // the following loops calculate formula (10), potential from B->A

        //     cur_pot[0] = m_partner[0] * D_lower[0];
        //     cur_pot[1] = m_partner[0] * D_lower[1];
        //     cur_pot[2] = m_partner[0] * D_lower[2];
        //     cur_pot[3] = m_partner[0] * D_lower[3];

        //     cur_pot[0] += m_partner[4] * (D_lower[4] * factor_half_v[4]);
        //     cur_pot[1] += m_partner[4] * (D_lower[10] * factor_half_v[4]);
        //     cur_pot[2] += m_partner[4] * (D_lower[11] * factor_half_v[4]);
        //     cur_pot[3] += m_partner[4] * (D_lower[12] * factor_half_v[4]);

        //     cur_pot[0] += m_partner[5] * (D_lower[5] * factor_half_v[5]);
        //     cur_pot[1] += m_partner[5] * (D_lower[11] * factor_half_v[5]);
        //     cur_pot[2] += m_partner[5] * (D_lower[13] * factor_half_v[5]);
        //     cur_pot[3] += m_partner[5] * (D_lower[14] * factor_half_v[5]);

        //     cur_pot[0] += m_partner[6] * (D_lower[6] * factor_half_v[6]);
        //     cur_pot[1] += m_partner[6] * (D_lower[12] * factor_half_v[6]);
        //     cur_pot[2] += m_partner[6] * (D_lower[14] * factor_half_v[6]);
        //     cur_pot[3] += m_partner[6] * (D_lower[15] * factor_half_v[6]);

        //     cur_pot[0] += m_partner[7] * (D_lower[7] * factor_half_v[7]);
        //     cur_pot[1] += m_partner[7] * (D_lower[13] * factor_half_v[7]);
        //     cur_pot[2] += m_partner[7] * (D_lower[16] * factor_half_v[7]);
        //     cur_pot[3] += m_partner[7] * (D_lower[17] * factor_half_v[7]);

        //     cur_pot[0] += m_partner[8] * (D_lower[8] * factor_half_v[8]);
        //     cur_pot[1] += m_partner[8] * (D_lower[14] * factor_half_v[8]);
        //     cur_pot[2] += m_partner[8] * (D_lower[17] * factor_half_v[8]);
        //     cur_pot[3] += m_partner[8] * (D_lower[18] * factor_half_v[8]);

        //     cur_pot[0] += m_partner[9] * (D_lower[9] * factor_half_v[9]);
        //     cur_pot[1] += m_partner[9] * (D_lower[15] * factor_half_v[9]);
        //     cur_pot[2] += m_partner[9] * (D_lower[18] * factor_half_v[9]);
        //     cur_pot[3] += m_partner[9] * (D_lower[19] * factor_half_v[9]);

        //     cur_pot[0] -= m_partner[10] * (D_lower[10] * factor_sixth_v[10]);
        //     cur_pot[0] -= m_partner[11] * (D_lower[11] * factor_sixth_v[11]);
        //     cur_pot[0] -= m_partner[12] * (D_lower[12] * factor_sixth_v[12]);
        //     cur_pot[0] -= m_partner[13] * (D_lower[13] * factor_sixth_v[13]);
        //     cur_pot[0] -= m_partner[14] * (D_lower[14] * factor_sixth_v[14]);
        //     cur_pot[0] -= m_partner[15] * (D_lower[15] * factor_sixth_v[15]);
        //     cur_pot[0] -= m_partner[16] * (D_lower[16] * factor_sixth_v[16]);
        //     cur_pot[0] -= m_partner[17] * (D_lower[17] * factor_sixth_v[17]);
        //     cur_pot[0] -= m_partner[18] * (D_lower[18] * factor_sixth_v[18]);
        //     cur_pot[0] -= m_partner[19] * (D_lower[19] * factor_sixth_v[19]);

        //     /* Maps to:
        //     for (integer i = 0; i != 6; ++i) {
        //         int const* abc_idx_map6 = to_abc_idx_map6[i];

        //         integer const ab_idx = ab_idx_map6[i];
        //         A0[ab_idx] = m0() * D[ab_idx];
        //         for (integer c = 0; c < NDIM; ++c) {
        //             A0[ab_idx] -= m0(c) * D[abc_idx_map6[c]];
        //         }
        //     }

        //     for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
        //         A0[i] = m0[0] * D[i];
        //     }
        //     from the old style */
        //     cur_pot[4] = m_partner[0] * D_lower[4];
        //     cur_pot[5] = m_partner[0] * D_lower[5];
        //     cur_pot[6] = m_partner[0] * D_lower[6];
        //     cur_pot[7] = m_partner[0] * D_lower[7];
        //     cur_pot[8] = m_partner[0] * D_lower[8];
        //     cur_pot[9] = m_partner[0] * D_lower[9];

        //     cur_pot[4] -= m_partner[1] * D_lower[10];
        //     cur_pot[5] -= m_partner[1] * D_lower[11];
        //     cur_pot[6] -= m_partner[1] * D_lower[12];
        //     cur_pot[7] -= m_partner[1] * D_lower[13];
        //     cur_pot[8] -= m_partner[1] * D_lower[14];
        //     cur_pot[9] -= m_partner[1] * D_lower[15];

        //     cur_pot[4] -= m_partner[2] * D_lower[11];
        //     cur_pot[5] -= m_partner[2] * D_lower[13];
        //     cur_pot[6] -= m_partner[2] * D_lower[14];
        //     cur_pot[7] -= m_partner[2] * D_lower[16];
        //     cur_pot[8] -= m_partner[2] * D_lower[17];
        //     cur_pot[9] -= m_partner[2] * D_lower[18];

        //     cur_pot[4] -= m_partner[3] * D_lower[12];
        //     cur_pot[5] -= m_partner[3] * D_lower[14];
        //     cur_pot[6] -= m_partner[3] * D_lower[15];
        //     cur_pot[7] -= m_partner[3] * D_lower[17];
        //     cur_pot[8] -= m_partner[3] * D_lower[18];
        //     cur_pot[9] -= m_partner[3] * D_lower[19];

        //     tmpstore[0] = tmpstore[0] + cur_pot[0];
        //     tmpstore[1] = tmpstore[1] + cur_pot[1];
        //     tmpstore[2] = tmpstore[2] + cur_pot[2];
        //     tmpstore[3] = tmpstore[3] + cur_pot[3];
        //     tmpstore[4] = tmpstore[4] + cur_pot[4];
        //     tmpstore[5] = tmpstore[5] + cur_pot[5];
        //     tmpstore[6] = tmpstore[6] + cur_pot[6];
        //     tmpstore[7] = tmpstore[7] + cur_pot[7];
        //     tmpstore[8] = tmpstore[8] + cur_pot[8];
        //     tmpstore[9] = tmpstore[9] + cur_pot[9];

        //     /* Maps to
        //     for (integer i = taylor_sizes[2]; i < taylor_sizes[3]; ++i) {
        //         A0[i] = m0[0] * D[i];
        //     }*/
        //     tmpstore[10] = tmpstore[10] + m_partner[0] * D_lower[10];
        //     tmpstore[11] = tmpstore[11] + m_partner[0] * D_lower[11];
        //     tmpstore[12] = tmpstore[12] + m_partner[0] * D_lower[12];
        //     tmpstore[13] = tmpstore[13] + m_partner[0] * D_lower[13];
        //     tmpstore[14] = tmpstore[14] + m_partner[0] * D_lower[14];
        //     tmpstore[15] = tmpstore[15] + m_partner[0] * D_lower[15];
        //     tmpstore[16] = tmpstore[16] + m_partner[0] * D_lower[16];
        //     tmpstore[17] = tmpstore[17] + m_partner[0] * D_lower[17];
        //     tmpstore[18] = tmpstore[18] + m_partner[0] * D_lower[18];
        //     tmpstore[19] = tmpstore[19] + m_partner[0] * D_lower[19];
        // }
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
