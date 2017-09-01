#include "m2m_kernel.hpp"

#include "defs.hpp"
#include "interaction_types.hpp"
#include "options.hpp"
#include "struct_of_array_taylor.hpp"

#include <array>
#include <functional>

// std::vector<interaction_type> ilist_debugging;

extern options opts;

namespace octotiger {
namespace fmm {

    m2m_kernel::m2m_kernel(
        // struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>& local_expansions_SoA,
        // struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>& center_of_masses_SoA,
        // struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>&
        // potential_expansions_SoA,
        // struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
        // angular_corrections_SoA,
        std::vector<bool>& neighbor_empty,
        gsolve_type type)
      :    // local_expansions(local_expansions),
      // local_expansions_SoA(local_expansions_SoA)
      // , center_of_masses(center_of_masses),
      // center_of_masses_SoA(center_of_masses_SoA)
      // , potential_expansions(potential_expansions)
      // , potential_expansions_SoA(potential_expansions_SoA)
      // , angular_corrections(angular_corrections)
      // , angular_corrections_SoA(angular_corrections_SoA),
      neighbor_empty(neighbor_empty)
      , type(type)
      , theta_rec_squared(sqr(1.0 / opts.theta))
    // , theta_rec_squared_scalar(sqr(1.0 / opts.theta))
    {
        for (size_t i = 0; i < m2m_int_vector::size(); i++) {
            offset_vector[i] = i;
        }
        vectors_check_empty();
        // calculate_coarse_indices();
    }

    void m2m_kernel::apply_stencil(
        struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>& local_expansions_SoA,
        struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>& center_of_masses_SoA,
        struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>& potential_expansions_SoA,
        struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>& angular_corrections_SoA,
        std::vector<multiindex<>>& stencil) {
        // for (multiindex<>& stencil_element : stencil) {
        for (size_t outer_stencil_index = 0; outer_stencil_index < stencil.size();
             outer_stencil_index += STENCIL_BLOCKING) {
            // std::cout << "stencil_element: " << stencil_element << std::endl;
            // TODO: remove after proper vectorization
            // multiindex<> se(stencil_element.x, stencil_element.y, stencil_element.z);
            // std::cout << "se: " << se << std::endl;
            // iterate_inner_cells_padded_stencil(se, *this);
            for (size_t i0 = 0; i0 < INNER_CELLS_PER_DIRECTION; i0++) {
                for (size_t i1 = 0; i1 < INNER_CELLS_PER_DIRECTION; i1++) {
                    // for (size_t i2 = 0; i2 < INNER_CELLS_PER_DIRECTION; i2++) {
                    for (size_t i2 = 0; i2 < INNER_CELLS_PER_DIRECTION; i2 += m2m_vector::size()) {
                        const multiindex<> cell_index(i0 + INNER_CELLS_PADDING_DEPTH,
                            i1 + INNER_CELLS_PADDING_DEPTH, i2 + INNER_CELLS_PADDING_DEPTH);
                        // BUG: indexing has to be done with uint32_t because of Vc limitation
                        const int64_t cell_flat_index = to_flat_index_padded(cell_index);
                        const multiindex<> cell_index_unpadded(i0, i1, i2);
                        const int64_t cell_flat_index_unpadded =
                            to_inner_flat_index_not_padded(cell_index_unpadded);

                        // indices on coarser level (for outer stencil boundary)
                        // implicitly broadcasts to vector
                        multiindex<m2m_int_vector> cell_index_coarse(cell_index);
                        for (size_t j = 0; j < m2m_int_vector::size(); j++) {
                            cell_index_coarse.z[j] += j;
                        }
                        // note that this is the same for groups of 2x2x2 elements
                        // -> maps to the same for some SIMD lanes
                        cell_index_coarse.transform_coarse();

                        if (type == RHO) {
                            this->blocked_interaction_rho(local_expansions_SoA,
                                center_of_masses_SoA, potential_expansions_SoA,
                                angular_corrections_SoA,

                                cell_index, cell_flat_index, cell_index_coarse, cell_index_unpadded,
                                cell_flat_index_unpadded, stencil, outer_stencil_index);
                        } else {
                            this->blocked_interaction_non_rho(local_expansions_SoA,
                                center_of_masses_SoA, potential_expansions_SoA,
                                angular_corrections_SoA, cell_index, cell_flat_index,
                                cell_index_coarse, cell_index_unpadded, cell_flat_index_unpadded,
                                stencil, outer_stencil_index);
                        }
                    }
                }
            }
        }
    }

    // void m2m_kernel::apply_stencil_element(multiindex& stencil_element) {
    //     // execute kernel for individual interactions for every inner cell
    //     // use this object as a functor for the iteration
    //     iterate_inner_cells_padded_stencil(stencil_element, *this);

    //     // std::cout << "potential_expansions.size(): " <<
    //     potential_expansions.size() <<
    //     std::endl;
    //     // std::cout << "potential_expansions[0]: " << potential_expansions[0];
    //     // std::cout << std::endl;
    // }

    void m2m_kernel::vectors_check_empty() {
        vector_is_empty = std::vector<bool>(PADDED_STRIDE * PADDED_STRIDE * PADDED_STRIDE);
        for (size_t i0 = 0; i0 < PADDED_STRIDE; i0 += 1) {
            for (size_t i1 = 0; i1 < PADDED_STRIDE; i1 += 1) {
                for (size_t i2 = 0; i2 < PADDED_STRIDE; i2 += 1) {
                    const multiindex<> cell_index(i0, i1, i2);
                    const int64_t cell_flat_index = to_flat_index_padded(cell_index);
                    // std::cout << "cell_flat_index: " << cell_flat_index << std::endl;

                    const multiindex<> in_boundary_start(
                        (cell_index.x / INNER_CELLS_PER_DIRECTION) - 1,
                        (cell_index.y / INNER_CELLS_PER_DIRECTION) - 1,
                        (cell_index.z / INNER_CELLS_PER_DIRECTION) - 1);

                    const multiindex<> in_boundary_end(in_boundary_start.x, in_boundary_start.y,
                        ((cell_index.z + m2m_int_vector::size()) / INNER_CELLS_PER_DIRECTION) - 1);

                    geo::direction dir_start;
                    dir_start.set(in_boundary_start.x, in_boundary_start.y, in_boundary_start.z);
                    geo::direction dir_end;
                    dir_end.set(in_boundary_end.x, in_boundary_end.y, in_boundary_end.z);

                    if (neighbor_empty[dir_start.flat_index_with_center()] &&
                        neighbor_empty[dir_end.flat_index_with_center()]) {
                        vector_is_empty[cell_flat_index] = true;
                        // std::cout << "prepare true cell_index:" << cell_index << std::endl;
                        // std::cout << "cell_flat_index: " << cell_flat_index << std::endl;
                        // std::cout << "dir_start.flat_index_with_center(): "
                        //           << dir_start.flat_index_with_center() << std::endl;
                        // std::cout << "dir_end.flat_index_with_center(): "
                        //           << dir_end.flat_index_with_center() << std::endl;
                        // std::cout << "in_boundary_end: " << in_boundary_end << std::endl;

                    } else {
                        vector_is_empty[cell_flat_index] = false;
                        // std::cout << "prepare false cell_index:" << cell_index << std::endl;
                        // std::cout << "cell_flat_index: " << cell_flat_index << std::endl;
                        // std::cout << "dir_start.flat_index_with_center(): "
                        //           << dir_start.flat_index_with_center() << std::endl;
                        // std::cout << "dir_end.flat_index_with_center(): "
                        //           << dir_end.flat_index_with_center() << std::endl;
                    }
                }
            }
        }
    }

    // void m2m_kernel::calculate_coarse_indices() {
    //     coarse_x = std::vector<int32_t>(PADDED_STRIDE * PADDED_STRIDE * PADDED_STRIDE);
    //     coarse_y = std::vector<int32_t>(PADDED_STRIDE * PADDED_STRIDE * PADDED_STRIDE);
    //     coarse_z = std::vector<int32_t>(PADDED_STRIDE * PADDED_STRIDE * PADDED_STRIDE);

    //     const int32_t patch_size = static_cast<int32_t>(INX);
    //     const int32_t subtract = static_cast<int32_t>(INX / 2);
    //     for (size_t i0 = 0; i0 < PADDED_STRIDE; i0 += 1) {
    //         for (size_t i1 = 0; i1 < PADDED_STRIDE; i1 += 1) {
    //             for (size_t i2 = 0; i2 < PADDED_STRIDE; i2 += 1) {
    //                 const multiindex<> cell_index(i0, i1, i2);
    //                 const int64_t cell_flat_index = to_flat_index_padded(cell_index);
    //                 coarse_x[cell_flat_index] = ((cell_index.x + patch_size) >> 1) - subtract;
    //                 coarse_y[cell_flat_index] = ((cell_index.y + patch_size) >> 1) - subtract;
    //                 coarse_z[cell_flat_index] = ((cell_index.z + patch_size) >> 1) - subtract;
    //             }
    //         }
    //     }
    // }

}    // namespace fmm
}    // namespace octotiger
