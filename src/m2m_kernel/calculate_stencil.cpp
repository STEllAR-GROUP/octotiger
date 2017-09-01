#include "defs.hpp"
#include "geometry.hpp"
#include "options.hpp"

#include "calculate_stencil.hpp"
#include "helper.hpp"

extern options opts;

namespace octotiger {
namespace fmm {

    std::vector<multiindex<>> calculate_stencil() {
        std::array<std::vector<multiindex<>>, 8> stencils;

        // used to check the radiuses of the outer and inner sphere
        const real theta0 = opts.theta;

        // int64_t i0 = 0;
        // int64_t i1 = 0;
        // int64_t i2 = 0;

        for (int64_t i0 = 0; i0 < 2; ++i0) {
            for (int64_t i1 = 0; i1 < 2; ++i1) {
                for (int64_t i2 = 0; i2 < 2; ++i2) {
                    std::vector<multiindex<>> stencil;
                    for (int64_t j0 = i0 - INX; j0 < i0 + INX; ++j0) {
                        for (int64_t j1 = i1 - INX; j1 < i1 + INX; ++j1) {
                            for (int64_t j2 = i2 - INX; j2 < i2 + INX; ++j2) {
                                // don't interact with yourself!
                                if (i0 == j0 && i1 == j1 && i2 == j2) {
                                    continue;
                                }

                                // indices on coarser level (for outer stencil boundary)
                                const int64_t i0_c = (i0 + INX) / 2 - INX / 2;
                                const int64_t i1_c = (i1 + INX) / 2 - INX / 2;
                                const int64_t i2_c = (i2 + INX) / 2 - INX / 2;

                                const int64_t j0_c = (j0 + INX) / 2 - INX / 2;
                                const int64_t j1_c = (j1 + INX) / 2 - INX / 2;
                                const int64_t j2_c = (j2 + INX) / 2 - INX / 2;

                                const real theta_f =
                                    detail::reciprocal_distance(i0, i1, i2, j0, j1, j2);
                                const real theta_c =
                                    detail::reciprocal_distance(i0_c, i1_c, i2_c, j0_c, j1_c, j2_c);

                                // not in inner sphere (theta_c > theta0), but in outer sphere
                                if (theta_c > theta0 && theta_f <= theta0) {
                                    stencil.emplace_back(j0 - i0, j1 - i1, j2 - i2);
                                }
                            }
                        }
                    }
                    stencils[i0 * 4 + i1 * 2 + i2] = stencil;
                }
            }
        }

        std::vector<multiindex<>> superimposed_stencil;
        for (size_t i = 0; i < 8; i++) {
            for (multiindex<>& stencil_element : stencils[i]) {
                bool found = false;
                for (multiindex<>& super_element : superimposed_stencil) {
                    if (stencil_element.compare(super_element)) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    superimposed_stencil.push_back(stencil_element);
                }
            }
        }
        // std::cout << "superimposed_stencil.size(): " << superimposed_stencil.size() << std::endl;

        // for (size_t i = 0; i < 8; i++) {
        //     uint64_t common_elements = 0;
        //     for (auto& element : stencils[i]) {
        //         for (auto& super_element : superimposed_stencil) {
        //             if (element.compare(super_element)) {
        //                 common_elements += 1;
        //                 break;
        //             }
        //         }
        //     }
        //     std::cout << "total_elements: " << stencils[i].size()
        //               << " common_elements: " << common_elements
        //               << " masked_elements: " << (superimposed_stencil.size() - common_elements)
        //               << std::endl;
        // }

        // for (size_t i = 0; i < 8; i++) {
        //     std::cout << "-------------- " << i << " ---------------" << std::endl;
        //     std::cout << "x, y, z" << std::endl;
        //     std::vector<multiindex>& stencil = stencils[i];
        //     for (multiindex& stencil_element : stencil) {
        //         std::cout << stencil_element << std::endl;
        //     }
        //     std::cout << std::endl;
        // }

        // std::vector<multiindex> common_stencil;
        // // use the first stencil to filter the elements that are in all other stencils
        // std::vector<multiindex>& first_stencil = stencils[0];
        // for (multiindex& first_element : first_stencil) {
        //     bool all_found = true;
        //     for (size_t i = 1; i < 8; i++) {
        //         bool found = false;
        //         for (multiindex& second_element : stencils[i]) {
        //             if (first_element.compare(second_element)) {
        //                 found = true;
        //                 break;
        //             }
        //         }
        //         if (!found) {
        //             all_found = false;
        //             break;
        //         }
        //     }
        //     if (all_found) {
        //         common_stencil.push_back(first_element);
        //     }
        // }

        // std::cout << "-------------- common_stencil"
        //           << " size: " << common_stencil.size() << " ---------------" << std::endl;
        // std::cout << "x, y, z" << std::endl;
        // for (multiindex& stencil_element : common_stencil) {
        //     std::cout << stencil_element << std::endl;
        // }
        // std::cout << std::endl;

        // std::array<std::vector<multiindex>, 8> diff_stencils;

        // for (size_t i = 0; i < 8; i++) {
        //     for (multiindex& element : stencils[i]) {
        //         bool found = false;
        //         for (multiindex& common_element : common_stencil) {
        //             if (element.compare(common_element)) {
        //                 found = true;
        //                 break;
        //             }
        //         }
        //         if (!found) {
        //             diff_stencils[i].push_back(element);
        //         }
        //     }
        // }

        // for (size_t i = 0; i < 8; i++) {
        //     std::cout << "-------------- diff_stencil: " << i
        //               << " size: " << diff_stencils[i].size() << " ---------------" << std::endl;
        //     std::cout << "x, y, z" << std::endl;
        //     std::vector<multiindex>& stencil = diff_stencils[i];
        //     for (multiindex& stencil_element : stencil) {
        //         std::cout << stencil_element << std::endl;
        //     }
        //     std::cout << std::endl;
        // }

        return superimposed_stencil;
    }

}    // namespace fmm
}    // namespace octotiger
