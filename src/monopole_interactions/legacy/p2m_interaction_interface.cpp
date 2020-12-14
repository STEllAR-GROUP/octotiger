//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/monopole_interactions/legacy/p2m_interaction_interface.hpp"
#include "octotiger/common_kernel/interactions_iterators.hpp"
#include "octotiger/monopole_interactions/legacy/p2m_kernel.hpp"
#include "octotiger/monopole_interactions/util/calculate_stencil.hpp"
#include "octotiger/options.hpp"
#include "octotiger/real.hpp"

#include <algorithm>
#include <array>
#include <vector>

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        template <size_t buffer_size>
        bool check_neighbor_conversion(
            struct_of_array_data<expansion, real, 20, buffer_size, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&
                local_expansions_staging_area,
            struct_of_array_data<space_vector, real, 3, buffer_size, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>&
                center_of_masses_staging_area,
            cpu_expansion_buffer_t& local_expansions_compare,
            cpu_space_vector_buffer_t& center_of_masses_compare, const geo::direction& dir) {
            bool correct = true;

            multiindex<> start_index = get_padding_start_indices(dir);
            multiindex<> end_index = get_padding_end_indices(dir);
            multiindex<> size = get_padding_real_size(dir);
            for (size_t x = start_index.x; x < end_index.x; x++) {
                for (size_t y = start_index.y; y < end_index.y; y++) {
                    for (size_t z = start_index.z; z < end_index.z; z++) {
                        const multiindex<> global_padded_index(
                            INNER_CELLS_PADDING_DEPTH + dir[0] * INNER_CELLS_PADDING_DEPTH + x,
                            INNER_CELLS_PADDING_DEPTH + dir[1] * INNER_CELLS_PADDING_DEPTH + y,
                            INNER_CELLS_PADDING_DEPTH + dir[2] * INNER_CELLS_PADDING_DEPTH + z);
                        const size_t global_flat_index = to_flat_index_padded(global_padded_index);
                        const multiindex<> m_padding_index(
                            x - start_index.x, y - start_index.y, z - start_index.z);
                        const size_t flat_index = m_padding_index.x * (size.y * size.z) +
                            m_padding_index.y * size.z + m_padding_index.z;
                        if (local_expansions_compare.at<0>(global_flat_index) !=
                            local_expansions_staging_area.template at<0>(flat_index)) {
                            std::cerr << " Exp 0 Error at " << global_padded_index << " vs  "
                                      << m_padding_index << " value: "
                                      << local_expansions_compare.at<0>(global_flat_index) << " vs "
                                      << local_expansions_staging_area.template at<0>(flat_index)
                                      << std::endl;
                            correct = false;
                        }
                        if (local_expansions_compare.at<19>(global_flat_index) !=
                            local_expansions_staging_area.template at<19>(flat_index)) {
                            std::cerr
                                << "Exp 19 Error at " << global_padded_index << " vs  "
                                << m_padding_index
                                << " value: " << local_expansions_compare.at<19>(global_flat_index)
                                << " vs "
                                << local_expansions_staging_area.template at<19>(flat_index)
                                << std::endl;
                            correct = false;
                        }
                        if (center_of_masses_compare.at<0>(global_flat_index) !=
                            center_of_masses_staging_area.template at<0>(flat_index)) {
                            std::cerr << "Mass 0 Error at " << global_padded_index << " vs  "
                                      << m_padding_index << " value: "
                                      << center_of_masses_compare.at<0>(global_flat_index) << " vs "
                                      << center_of_masses_staging_area.template at<0>(flat_index)
                                      << std::endl;
                            correct = false;
                        }
                        if (center_of_masses_compare.at<2>(global_flat_index) !=
                            center_of_masses_staging_area.template at<2>(flat_index)) {
                            std::cerr << "Mass 2 Error at " << global_padded_index << " vs  "
                                      << m_padding_index << " value: "
                                      << center_of_masses_compare.at<2>(global_flat_index) << " vs "
                                      << center_of_masses_staging_area.template at<2>(flat_index)
                                      << std::endl;
                            correct = false;
                        }
                    }
                }
            }
            return correct;
        }

        std::vector<multiindex<>>& p2m_interaction_interface::stencil() {
            static thread_local std::vector<multiindex<>> stencil_ = calculate_stencil().first;
            return stencil_;
        }
        std::vector<multiindex<>>& p2m_stencil() {
            static thread_local std::vector<multiindex<>> stencil_ = calculate_stencil().first;
            return stencil_;
        }
        std::vector<bool>& p2m_stencil_masks() {
            static thread_local std::vector<bool> stencil_masks_ =
                calculate_stencil_masks(p2m_stencil()).first;
            return stencil_masks_;
        }

        p2m_interaction_interface::p2m_interaction_interface()
          : neighbor_empty_multipoles(27) {
            this->p2m_type = opts().p2m_kernel_type;
        }

        void p2m_interaction_interface::compute_p2m_interactions(std::vector<real>& monopoles,
            std::vector<multipole>& M_ptr,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type,
            std::array<bool, geo::direction::count()>& is_direction_empty) {
            cpu_expansion_buffer_t local_expansions_staging_area;
            cpu_space_vector_buffer_t center_of_masses_staging_area;

            update_input(M_ptr, com_ptr, neighbors, type, local_expansions_staging_area,
                center_of_masses_staging_area);
            compute_interactions(type, is_direction_empty, neighbors, local_expansions_staging_area,
                center_of_masses_staging_area);
        }
        void compute_p2m_interactions_neighbors_only(const std::vector<real>& monopoles,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::shared_ptr<grid>& grid_ptr) {
            if (opts().p2m_kernel_type == interaction_kernel_type::OLD) {
                // waits for boundary data and then computes boundary interactions
                for (auto const& dir : geo::direction::full_set()) {
                    if (!is_direction_empty[dir]) {
                        neighbor_gravity_type& neighbor_data = neighbors[dir];
                        if (!neighbor_data.is_monopole) {
                            grid_ptr->compute_boundary_interactions(type, neighbor_data.direction,
                                neighbor_data.is_monopole, neighbor_data.data);
                        }
                    }
                }
                return;
            } 
            cpu_expansion_buffer_t local_expansions_compare;
            p2m_kernel kernel;
            cpu_space_vector_buffer_t center_of_masses_compare;

            // TODO remove m_ptr from legacy update input for the sake of testing
            // Required for comparisons in later asserts
            // assert(update_input(M_ptr, com_ptr, neighbors, type, local_expansions_compare,
            //    center_of_masses_compare));

            struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING,
                std::vector<real, recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>
                center_of_masses_inner_cells_staging_area;
            std::vector<space_vector> const& com0 = *(com_ptr[0]);
            cpu_expansion_result_buffer_t potential_expansions_SoA;
            cpu_angular_result_t angular_corrections_SoA;

            iterate_inner_cells_padded(
                [&center_of_masses_inner_cells_staging_area, com0, &angular_corrections_SoA](const multiindex<>& i,
                    const size_t flat_index, const multiindex<>& i_unpadded,
                    const size_t flat_index_unpadded) {
                    center_of_masses_inner_cells_staging_area.set_AoS_value(
                        std::move(com0.at(flat_index_unpadded)), flat_index_unpadded);
                    angular_corrections_SoA.set_AoS_value(space_vector(), flat_index_unpadded);
                });

            for (const geo::direction& dir : geo::direction::full_set()) {
                neighbor_gravity_type& neighbor = neighbors[dir];
                if (!neighbor.is_monopole && neighbor.data.M) {
                    // SoA datastructure only supports constexpr size
                    int size = 1;
                    for (int i = 0; i < 3; i++) {
                        if (dir[i] == 0)
                            size *= INX;
                        else
                            size *= STENCIL_MAX;
                    }
                    assert(size == INX * INX * STENCIL_MAX ||
                        size == INX * STENCIL_MAX * STENCIL_MAX ||
                        size == STENCIL_MAX * STENCIL_MAX * STENCIL_MAX);
                    multiindex<> start_index = get_padding_start_indices(dir);
                    multiindex<> end_index = get_padding_end_indices(dir);
                    multiindex<> neighbor_size = get_padding_real_size(dir);
                    if (size == INX * INX * STENCIL_MAX) {
                        constexpr size_t buffer_size = INX * INX * STENCIL_MAX;
                        struct_of_array_data<expansion, real, 20, buffer_size, SOA_PADDING,
                            std::vector<real,
                                recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>
                            local_expansions_staging_area;
                        struct_of_array_data<space_vector, real, 3, buffer_size, SOA_PADDING,
                            std::vector<real,
                                recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>
                            center_of_masses_staging_area;

                        update_neighbor_input(dir, com_ptr, neighbors, type,
                            local_expansions_staging_area, center_of_masses_staging_area, grid_ptr);
                        /*assert(check_neighbor_conversion(local_expansions_staging_area,
                            center_of_masses_staging_area, local_expansions_compare,
                            center_of_masses_compare, dir));*/

                        kernel.apply_stencil_neighbor<INX * INX * STENCIL_MAX>(neighbor_size,
                            start_index, end_index, local_expansions_staging_area,
                            center_of_masses_staging_area,
                            center_of_masses_inner_cells_staging_area, potential_expansions_SoA,
                            angular_corrections_SoA, p2m_stencil_masks(), type, dir);
                    } else if (size == INX * STENCIL_MAX * STENCIL_MAX) {
                        constexpr size_t buffer_size = INX * STENCIL_MAX * STENCIL_MAX;
                        struct_of_array_data<expansion, real, 20, buffer_size, SOA_PADDING,
                            std::vector<real,
                                recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>
                            local_expansions_staging_area;
                        struct_of_array_data<space_vector, real, 3, buffer_size, SOA_PADDING,
                            std::vector<real,
                                recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>
                            center_of_masses_staging_area;

                        update_neighbor_input(dir, com_ptr, neighbors, type,
                            local_expansions_staging_area, center_of_masses_staging_area, grid_ptr);
                        /*assert(check_neighbor_conversion(local_expansions_staging_area,
                            center_of_masses_staging_area, local_expansions_compare,
                            center_of_masses_compare, dir));*/

                        kernel.apply_stencil_neighbor<INX * STENCIL_MAX * STENCIL_MAX>(
                            neighbor_size, start_index, end_index, local_expansions_staging_area,
                            center_of_masses_staging_area,
                            center_of_masses_inner_cells_staging_area, potential_expansions_SoA,
                            angular_corrections_SoA, p2m_stencil_masks(), type, dir);
                    } else if (size == STENCIL_MAX * STENCIL_MAX * STENCIL_MAX) {
                        constexpr size_t buffer_size = STENCIL_MAX * STENCIL_MAX * STENCIL_MAX;
                        struct_of_array_data<expansion, real, 20, buffer_size, SOA_PADDING,
                            std::vector<real,
                                recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>
                            local_expansions_staging_area;
                        struct_of_array_data<space_vector, real, 3, buffer_size, SOA_PADDING,
                            std::vector<real,
                                recycler::aggressive_recycle_aligned<real, SIMD_LENGTH_BYTES>>>
                            center_of_masses_staging_area;

                        update_neighbor_input(dir, com_ptr, neighbors, type,
                            local_expansions_staging_area, center_of_masses_staging_area, grid_ptr);
                        /*assert(check_neighbor_conversion(local_expansions_staging_area,
                            center_of_masses_staging_area, local_expansions_compare,
                            center_of_masses_compare, dir));*/

                        kernel.apply_stencil_neighbor<STENCIL_MAX * STENCIL_MAX * STENCIL_MAX>(
                            neighbor_size, start_index, end_index, local_expansions_staging_area,
                            center_of_masses_staging_area,
                            center_of_masses_inner_cells_staging_area, potential_expansions_SoA,
                            angular_corrections_SoA, p2m_stencil_masks(), type, dir);
                    }
                }
            }
            potential_expansions_SoA.add_to_non_SoA(grid_ptr->get_L());
            if (type == RHO) {
                angular_corrections_SoA.to_non_SoA(grid_ptr->get_L_c());
            }
        }

        void p2m_interaction_interface::compute_interactions(gsolve_type type,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::vector<neighbor_gravity_type>& all_neighbor_interaction_data,
            const cpu_expansion_buffer_t& local_expansions_staging_area,
            const cpu_space_vector_buffer_t& center_of_masses_staging_area) {
            if (p2m_type == interaction_kernel_type::SOA_CPU) {
                if (multipole_neighbors_exist) {
                    p2m_kernel kernel;
                    cpu_expansion_result_buffer_t potential_expansions_SoA;
                    cpu_angular_result_t angular_corrections_SoA;
                    kernel.apply_stencil(local_expansions_staging_area,
                        center_of_masses_staging_area, potential_expansions_SoA,
                        angular_corrections_SoA, stencil(), type, x_skip, y_skip, z_skip);
                    potential_expansions_SoA.add_to_non_SoA(grid_ptr->get_L());
                    if (type == RHO) {
                        angular_corrections_SoA.to_non_SoA(grid_ptr->get_L_c());
                    }
                }
            } else {
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
            }
        }
    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
