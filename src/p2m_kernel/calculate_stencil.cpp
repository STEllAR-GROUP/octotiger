#include "defs.hpp"
#include "geometry.hpp"
#include "options.hpp"

#include "calculate_stencil.hpp"
#include "../common_kernel/helper.hpp"

extern options opts;

namespace octotiger {
namespace fmm {
    namespace p2m_kernel {

        std::vector<multiindex<>> calculate_stencil() {
            std::array<std::vector<multiindex<>>, 8> stencils;
            std::cerr << "Calculating new stencil" << std::endl;

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
                std::cout << "Stencil size: " << stencils[i].size() << std::endl;
            }
            return superimposed_stencil;
        }

    }    // namespace p2p_kernel
}    // namespace fmm
}    // namespace octotiger
