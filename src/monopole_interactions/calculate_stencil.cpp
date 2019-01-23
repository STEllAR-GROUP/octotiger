#include "octotiger/monopole_interactions/calculate_stencil.hpp"

#include "octotiger/common_kernel/helper.hpp"
#include "octotiger/common_kernel/interaction_constants.hpp"

#include "octotiger/defs.hpp"
#include "octotiger/geometry.hpp"
#include "octotiger/options.hpp"

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        std::pair<std::vector<multiindex<>>, std::vector<std::array<real, 4>>> calculate_stencil() {
            std::array<std::vector<multiindex<>>, 8> stencils;

            // used to check the radii of the outer and inner sphere
            const real theta0 = opts().theta;

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
                                    const real theta_c = detail::reciprocal_distance(
                                        i0_c, i1_c, i2_c, j0_c, j1_c, j2_c);

                                    if (theta_c > theta0) {
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
                // std::cout << "Stencil size: " << stencils[i].size() << std::endl;
            }

            std::vector<std::array<real, 4>> four_constants;
            for (auto stencil_element : superimposed_stencil) {
                const real x = stencil_element.x;
                const real y = stencil_element.y;
                const real z = stencil_element.z;
                const real tmp = sqr(x) + sqr(y) + sqr(z);
                const real r = std::sqrt(tmp);
                const real r3 = r * r * r;
                std::array<real, 4> four;
                four[0] = -1.0 / r;
                four[1] = x / r3;
                four[2] = y / r3;
                four[3] = z / r3;
                four_constants.push_back(four);
            }

            return std::pair<std::vector<multiindex<>>, std::vector<std::array<real, 4>>>(
                superimposed_stencil, four_constants);
        }
        std::pair<std::vector<bool>, std::vector<std::array<real, 4>>>
        calculate_stencil_masks(std::vector<multiindex<>> superimposed_stencil) {

            std::array<real, 4> four_constants_defaults = {0, 0, 0, 0};
            std::vector<bool> stencil_masks(FULL_STENCIL_SIZE, false);
            std::vector<std::array<real, 4>> four_constants_stencil(FULL_STENCIL_SIZE, four_constants_defaults);
            for (auto stencil_element : superimposed_stencil) {
                const int x = stencil_element.x + 5;
                const int y = stencil_element.y + 5;
                const int z = stencil_element.z + 5;
                size_t index = x * 11 * 11 + y * 11 + z;
                stencil_masks[index] = true;
            }
            for (auto stencil_element : superimposed_stencil) {
                const real x = stencil_element.x;
                const real y = stencil_element.y;
                const real z = stencil_element.z;
                const real tmp = sqr(x) + sqr(y) + sqr(z);
                const real r = std::sqrt(tmp);
                const real r3 = r * r * r;
                std::array<real, 4> four;
                four[0] = -1.0 / r;
                four[1] = x / r3;
                four[2] = y / r3;
                four[3] = z / r3;
                size_t index = (x + 5) * 11 * 11 + (y + 5) * 11 + (z + 5);
                four_constants_stencil[index] = four;
            }
            return std::pair<std::vector<bool>, std::vector<std::array<real, 4>>>(stencil_masks, four_constants_stencil);

        }

    } // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
