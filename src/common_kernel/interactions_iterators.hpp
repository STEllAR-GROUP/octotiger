#pragma once

#include "geometry.hpp"
#include "interaction_constants.hpp"
// #include "interaction_types.hpp"
#include "multiindex.hpp"
#include "taylor.hpp"

#include <vector>

namespace octotiger {
namespace fmm {
        // meant to iterate the input data structure
        template <typename F>
        void iterate_inner_cells_padded(const F& f) {
            for (size_t i0 = 0; i0 < INNER_CELLS_PER_DIRECTION; i0++) {
                for (size_t i1 = 0; i1 < INNER_CELLS_PER_DIRECTION; i1++) {
                    for (size_t i2 = 0; i2 < INNER_CELLS_PER_DIRECTION; i2++) {
                        const multiindex<> m(i0 + INNER_CELLS_PADDING_DEPTH,
                            i1 + INNER_CELLS_PADDING_DEPTH, i2 + INNER_CELLS_PADDING_DEPTH);
                        const size_t inner_flat_index = to_flat_index_padded(m);
                        const multiindex<> m_unpadded(i0, i1, i2);
                        const size_t inner_flat_index_unpadded =
                            to_inner_flat_index_not_padded(m_unpadded);
                        f(m, inner_flat_index, m_unpadded, inner_flat_index_unpadded);
                    }
                }
            }
        }

        // meant to iterate the input data structure
        template <typename F>
        void iterate_inner_cells_padding(const geo::direction& dir, const F& f) {
            // TODO: implementation not finished
            for (size_t i0 = 0; i0 < INNER_CELLS_PER_DIRECTION; i0++) {
                for (size_t i1 = 0; i1 < INNER_CELLS_PER_DIRECTION; i1++) {
                    for (size_t i2 = 0; i2 < INNER_CELLS_PER_DIRECTION; i2++) {
                        // shift to central cube and apply dir-based offset
                        const multiindex<> m(
                            i0 + INNER_CELLS_PADDING_DEPTH + dir[0] * INNER_CELLS_PADDING_DEPTH,
                            i1 + INNER_CELLS_PADDING_DEPTH + dir[1] * INNER_CELLS_PADDING_DEPTH,
                            i2 + INNER_CELLS_PADDING_DEPTH + dir[2] * INNER_CELLS_PADDING_DEPTH);
                        const size_t inner_flat_index = to_flat_index_padded(m);
                        const multiindex<> m_unpadded(i0, i1, i2);
                        const size_t inner_flat_index_unpadded =
                            to_inner_flat_index_not_padded(m_unpadded);
                        f(m, inner_flat_index, m_unpadded, inner_flat_index_unpadded);
                    }
                }
            }
        }

        // meant to iterate the input data structure
        template <typename F>
        void iterate_cells_padded(const F& f) {
            // TODO: implementation not finished
            for (size_t i0 = 0; i0 < PADDED_STRIDE; i0++) {
                for (size_t i1 = 0; i1 < PADDED_STRIDE; i1++) {
                    for (size_t i2 = 0; i2 < PADDED_STRIDE; i2++) {
                        const multiindex<> m(i0, i1, i2);
                        const size_t flat_index = to_flat_index_padded(m);
                        f(m, flat_index);
                    }
                }
            }
        }

        // meant to iterate the output data structure
        template <typename F>
        void iterate_inner_cells_not_padded(const F& f) {
            for (size_t i0 = 0; i0 < INNER_CELLS_PER_DIRECTION; i0++) {
                for (size_t i1 = 0; i1 < INNER_CELLS_PER_DIRECTION; i1++) {
                    for (size_t i2 = 0; i2 < INNER_CELLS_PER_DIRECTION; i2++) {
                        multiindex<> i(i0, i1, i2);
                        size_t inner_flat_index = to_inner_flat_index_not_padded(i);
                        f(i, inner_flat_index);
                    }
                }
            }
        }

        // iterate non-padded cube
        template <typename component_printer>
        void print_layered_not_padded(bool print_index, const component_printer& printer) {
            iterate_inner_cells_not_padded([&printer, print_index](
                multiindex<>& i, size_t flat_index) {
                if (i.y % INNER_CELLS_PER_DIRECTION == 0 && i.z % INNER_CELLS_PER_DIRECTION == 0) {
                    std::cout << "-------- next layer: " << i.x << "---------" << std::endl;
                }
                // std::cout << this->potential_expansions[flat_index];
                if (i.z % INNER_CELLS_PER_DIRECTION != 0) {
                    std::cout << ", ";
                }
                if (print_index) {
                    std::cout << " (" << i << ") = ";
                }
                printer(i, flat_index);
                if ((i.z + 1) % INNER_CELLS_PER_DIRECTION == 0) {
                    std::cout << std::endl;
                }
            });
        }

        // iterates only the central cube in a padded data structure
        template <typename component_printer>
        void print_layered_inner_padded(bool print_index, const component_printer& printer) {
            iterate_inner_cells_padded([&printer, print_index](const multiindex<>& i,
                const size_t flat_index, const multiindex<>& i_unpadded,
                const size_t flat_index_unpadded) {
                if (i.y % INNER_CELLS_PER_DIRECTION == 0 && i.z % INNER_CELLS_PER_DIRECTION == 0) {
                    std::cout << "-------- next layer: " << i.x << "---------" << std::endl;
                }
                if (i.z % INNER_CELLS_PER_DIRECTION != 0) {
                    std::cout << ", ";
                }
                if (print_index) {
                    std::cout << " (" << i << ") = ";
                }
                printer(i, flat_index, i_unpadded, flat_index_unpadded);
                if ((i.z + 1) % INNER_CELLS_PER_DIRECTION == 0) {
                    std::cout << std::endl;
                }
            });
        }

        // iterate everything including padding
        template <typename component_printer>
        void print_layered_padded(bool print_index, const component_printer& printer) {
            iterate_cells_padded(
                [&printer, print_index](const multiindex<>& i, const size_t flat_index) {
                    if (i.y % PADDED_STRIDE == 0 && i.z % PADDED_STRIDE == 0) {
                        std::cout << "-------- next layer: " << i.x << "---------" << std::endl;
                    }
                    if (i.z % PADDED_STRIDE != 0) {
                        std::cout << ", ";
                    }
                    if (print_index) {
                        std::cout << " (" << i << ") = ";
                    }
                    printer(i, flat_index);
                    if ((i.z + 1) % PADDED_STRIDE == 0) {
                        std::cout << std::endl;
                    }
                });
        }

        // iterate one of the padding directions
        template <typename component_printer>
        void print_layered_padding(
            geo::direction& dir, bool print_index, const component_printer& printer) {
            iterate_inner_cells_padding(
                dir, [&printer, print_index](const multiindex<>& i, const size_t flat_index,
                         const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                    if (i.y % INNER_CELLS_PER_DIRECTION == 0 &&
                        i.z % INNER_CELLS_PER_DIRECTION == 0) {
                        std::cout << "-------- next layer: " << i.x << "---------" << std::endl;
                    }
                    // std::cout << this->potential_expansions[flat_index];
                    if (i.z % INNER_CELLS_PER_DIRECTION != 0) {
                        std::cout << ", ";
                    }
                    if (print_index) {
                        std::cout << " (" << i << ") = ";
                    }
                    printer(i, flat_index, i_unpadded, flat_index_unpadded);
                    if ((i.z + 1) % INNER_CELLS_PER_DIRECTION == 0) {
                        std::cout << std::endl;
                    }
                });
        }

        bool expansion_comparator(const expansion& ref, const expansion& mine);

        bool space_vector_comparator(const space_vector& ref, const space_vector& mine);

        template <typename T, typename compare_functional>
        bool compare_inner_padded_with_non_padded(const std::vector<T>& ref_array,
            const std::vector<T>& mine_array, const compare_functional& c) {
            bool all_ok = true;
            iterate_inner_cells_padded([&c, &all_ok, &ref_array, &mine_array](const multiindex<>& i,
                const size_t flat_index, const multiindex<>& i_unpadded,
                const size_t flat_index_unpadded) {
                const T& ref = ref_array[flat_index_unpadded];
                const T& mine = mine_array[flat_index];
                bool ok = c(ref, mine);
                if (!ok) {
                    std::cout << "error for i:" << i << " i_unpadded: " << i_unpadded << std::endl;
                    all_ok = ok;
                }
            });

            if (!all_ok) {
                std::cout << "error: comparison failed!" << std::endl;
                exit(1);
            } else {
                std::cout << "comparison success!" << std::endl;
            }
            return all_ok;
        }

        template <typename T, typename compare_functional>
        bool compare_non_padded_with_non_padded(const std::vector<T>& ref_array,
            const std::vector<T>& mine_array, const compare_functional& c) {
            bool all_ok = true;
            iterate_inner_cells_not_padded([&c, &all_ok, &ref_array, &mine_array](
                const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                const T& ref = ref_array[flat_index_unpadded];
                const T& mine = mine_array[flat_index_unpadded];
                bool ok = c(ref, mine);
                if (!ok) {
                    std::cout << "error for i_unpadded: " << i_unpadded << std::endl;
                    all_ok = ok;
                }
            });
            if (!all_ok) {
                std::cout << "error: comparison failed!" << std::endl;
                exit(1);
            } else {
                std::cout << "comparison success!" << std::endl;
            }
            return all_ok;
        }

        template <typename T, typename compare_functional>
        bool compare_padded_with_non_padded(
            std::array<std::shared_ptr<std::vector<T>>, geo::direction::count()>& all_neighbors_ref,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            const std::vector<T>& mine_array, const compare_functional& c) {
            bool all_ok = true;
            for (const geo::direction& dir : geo::direction::full_set()) {
                std::cout << "comparing dir: " << dir;
                std::cout << ", is_empty: " << std::boolalpha << is_direction_empty[dir];
                std::cout << ", is_multipole: " << std::boolalpha
                          << (all_neighbors_ref[dir].operator bool()) << std::endl;
                // second condition implies that neighbor is monopole
                if (!is_direction_empty[dir] && all_neighbors_ref[dir]) {
                    std::vector<T>& neighbor_ref = *(all_neighbors_ref[dir]);
                    iterate_inner_cells_padding(
                        dir, [&c, &all_ok, &neighbor_ref, &mine_array, &dir, &all_neighbors_ref](
                                 const multiindex<>& i, const size_t flat_index,
                                 const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                            const T& ref = neighbor_ref[flat_index_unpadded];
                            const T& mine = mine_array[flat_index];
                            if (dir == 0) {
                                std::cout << "ref  !empty-multi d0: " << ref << std::endl;
                                std::cout << "mine !empty-multi d0: " << mine << std::endl;
                            }
                            bool ok = c(ref, mine);
                            if (!ok) {
                                std::cout << "error for i: " << i << " i_unpadded: " << i_unpadded
                                          << std::endl;
                                std::cout << "trying to find value in other neighbor..."
                                          << std::endl;
                                for (const geo::direction& dir_inner : geo::direction::full_set()) {
                                    if (all_neighbors_ref[dir_inner]) {
                                        const T& ref_inner =
                                            (*all_neighbors_ref[dir_inner])[flat_index_unpadded];
                                        bool ok_inner = c(ref_inner, mine);
                                        if (ok_inner) {
                                            std::cout << "found missing entry in dir: " << dir_inner
                                                      << std::endl;
                                        }
                                    }
                                }

                                all_ok = ok;
                            }
                        });
                } else {
                    T ref_dummy;
                    ref_dummy = 0.0;
                    iterate_inner_cells_padding(
                        dir, [&c, &all_ok, &ref_dummy, &mine_array](const multiindex<>& i,
                                 const size_t flat_index, const multiindex<>& i_unpadded,
                                 const size_t flat_index_unpadded) {
                            const T& mine = mine_array[flat_index];
                            bool ok = c(ref_dummy, mine);
                            if (!ok) {
                                std::cout << "error for i:" << i << " i_unpadded: " << i_unpadded
                                          << " (empty dir)" << std::endl;
                                all_ok = ok;
                            }
                        });
                }
            }

            if (!all_ok) {
                std::cout << "error: comparison failed!" << std::endl;
                exit(1);
            } else {
                std::cout << "comparison success!" << std::endl;
            }
            return all_ok;
        }
}    // namespace fmm
}    // namespace octotiger
