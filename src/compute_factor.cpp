#include "octotiger/common_kernel/kernel_simd_types.hpp"
#include "octotiger/compute_factor.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/real.hpp"
#include "octotiger/taylor.hpp"

taylor<4, real> factor;
taylor<4, m2m_vector> factor_half_v;
taylor<4, m2m_vector> factor_sixth_v;
m2m_vector factor_half[20];
m2m_vector factor_sixth[20];

void compute_factor() {
    factor = 0.0;
    factor() += 1.0;
    for (integer a = 0; a < NDIM; ++a) {
        factor(a) += 1.0;
        for (integer b = 0; b < NDIM; ++b) {
            factor(a, b) += 1.0;
            for (integer c = 0; c < NDIM; ++c) {
                factor(a, b, c) += 1.0;
            }
        }
    }

    const m2m_vector half_v(1.0 / 2.0);
    const m2m_vector sixth_v(1.0 / 6.0);
    for (size_t i = 0; i < factor.size(); i++) {
        factor_half_v[i] = half_v * factor[i];
        factor_sixth_v[i] = sixth_v * factor[i];
    }
    for (auto i = 0; i < 20; ++i) {
        factor_half[i] = factor_half_v[i];
        factor_sixth[i] = factor_sixth_v[i];
    }
}
