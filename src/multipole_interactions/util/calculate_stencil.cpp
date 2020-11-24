//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/multipole_interactions/util/calculate_stencil.hpp"

#include "octotiger/common_kernel/helper.hpp"

#include "octotiger/defs.hpp"
#include "octotiger/geometry.hpp"
#include "octotiger/options.hpp"
#include "octotiger/real.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        two_phase_stencil calculate_stencil() {
            std::array<two_phase_stencil, 8> stencils;

            // used to check the radii of the outer and inner sphere
            const real theta0 = opts().theta;

            int predicted_max = STENCIL_WIDTH;

            for (int64_t i0 = 0; i0 < 2; ++i0) {
                for (int64_t i1 = 0; i1 < 2; ++i1) {
                    for (int64_t i2 = 0; i2 < 2; ++i2) {
                        two_phase_stencil stencil;
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
                                    const real theta_c = detail::reciprocal_distance(
                                        i0_c, i1_c, i2_c, j0_c, j1_c, j2_c);

                                    // not in inner sphere (theta_c > theta0), but in outer
                                    // sphere
                                    if (theta_c > theta0 && theta_f <= theta0) {
                                        stencil.stencil_elements.emplace_back(
                                            j0 - i0, j1 - i1, j2 - i2);
                                        stencil.stencil_phase_indicator.emplace_back(true);
                                    } else if (theta_c > theta0) {
                                        stencil.stencil_elements.emplace_back(
                                            j0 - i0, j1 - i1, j2 - i2);
                                        stencil.stencil_phase_indicator.emplace_back(false);
                                    }
                                }
                            }
                        }
                        stencils[i0 * 4 + i1 * 2 + i2] = stencil;
                    }
                }
            }

            two_phase_stencil superimposed_stencil;
            for (size_t i = 0; i < 8; i++) {
                for (auto element_index = 0; element_index < stencils[i].stencil_elements.size();
                     ++element_index) {
                    multiindex<>& stencil_element = stencils[i].stencil_elements[element_index];
                    bool found = false;
                    for (multiindex<>& super_element : superimposed_stencil.stencil_elements) {
                        if (stencil_element.compare(super_element)) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        superimposed_stencil.stencil_elements.push_back(stencil_element);
                        superimposed_stencil.stencil_phase_indicator.push_back(
                            stencils[i].stencil_phase_indicator[element_index]);
                    }
                }
                // std::cout << "Stencil size: " << stencils[i].stencil_elements.size() << std::endl;
            }
            int i = 0;
            int minx = 0;
            int maxx = 0;
            int miny = 0;
            int maxy = 0;
            int minz = 0;
            int maxz = 0;
            for (auto &element : superimposed_stencil.stencil_elements) {
                if (element.x < minx)
                    minx = element.x;
                if (element.x > maxx)
                    maxx = element.x;
                if (element.y < miny)
                    miny = element.y;
                if (element.y > maxy)
                    maxy = element.y;
                if (element.z < minz)
                    minz = element.z;
                if (element.z > maxz)
                    maxz = element.z;
                i++;
            }
            if (maxz != maxy || maxy != maxx || minz != miny || miny != minx || minz != -maxz) {
                std::stringstream error_string;
                error_string << "ERROR: Stencil should be symetrical but it is not." << std::endl;
                error_string << "This indicates that something went wrong during the creation of the stencil!" << std::endl;
                error_string << "Please use another value for theta and contact the developer about this." << std::endl;
                std::cerr << error_string.str();
                // TODO Why do these exceptions not work?
                //throw std::logic_error(error_string.str());
                // since the exceptions do not stop the execution use this for now to avoid hanging large jobs...
                exit(EXIT_FAILURE);
            }
            if (predicted_max < maxx) {
                std::stringstream error_string;
                error_string << "ERROR: Maximum stencil size seems to be wrong. " << std::endl;
                error_string << "Please recompile with an appropriate minumal value for theta" << std::endl;
                error_string << "Max stencil length is " << predicted_max << ", actual stencil length is " << maxx << std::endl;
                std::cerr << error_string.str();
                // TODO Why do these exceptions not work?
                // throw std::logic_error(error_string.str());
                // since the exceptions do not stop the execution use this for now to avoid hanging large jobs...
                exit(EXIT_FAILURE);

            }
            return superimposed_stencil;
        }
        std::pair<std::vector<bool>, std::vector<bool>>
        calculate_stencil_masks(two_phase_stencil superimposed_stencil) {
            std::vector<bool> stencil_masks(FULL_STENCIL_SIZE, false);
            std::vector<bool> inner_stencil_masks(FULL_STENCIL_SIZE, true);
            auto inner_index = 0;
            for (auto stencil_element : superimposed_stencil.stencil_elements) {
                const int x = stencil_element.x + STENCIL_MAX;
                const int y = stencil_element.y + STENCIL_MAX;
                const int z = stencil_element.z + STENCIL_MAX;
                size_t index = x * STENCIL_INX * STENCIL_INX + y * STENCIL_INX + z;
                stencil_masks[index] = true;
                if (!superimposed_stencil.stencil_phase_indicator[inner_index])
                    inner_stencil_masks[index] = false;
                inner_index++;
            }
            return std::pair<std::vector<bool>, std::vector<bool>>(stencil_masks, inner_stencil_masks);

        }

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
