//  Copyright (c) 2020-2022 Gregor Daiß
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#pragma once
#include <cmath>
#include "octotiger/cuda_util/cuda_global_def.hpp"
#include "octotiger/hydro_defs.hpp"
#include "octotiger/common_kernel/kokkos_simd.hpp"

#if defined(__clang__)
constexpr int faces[3][9] = {{12, 0, 3, 6, 9, 15, 18, 21, 24}, {10, 0, 1, 2, 9, 11, 18, 19, 20},
    {4, 0, 1, 2, 3, 5, 6, 7, 8}};

constexpr double quad_weights[9] = {
    16. / 36., 1. / 36., 4. / 36., 1. / 36., 4. / 36., 4. / 36., 1. / 36., 4. / 36., 1. / 36.};

constexpr int offset = 0;
// TODO Change hard coded offsets
constexpr int dimension_length = INX + 2;
constexpr int compressedH_DN[3] = {dimension_length * dimension_length, dimension_length, 1};
constexpr int dim_offset = dimension_length * dimension_length * dimension_length;
constexpr int face_offset = 27 * dim_offset;
#else
CUDA_CALLABLE_METHOD const int faces[3][9] = {{12, 0, 3, 6, 9, 15, 18, 21, 24},
    {10, 0, 1, 2, 9, 11, 18, 19, 20}, {4, 0, 1, 2, 3, 5, 6, 7, 8}};

CUDA_CALLABLE_METHOD const double quad_weights[9] = {
    16. / 36., 1. / 36., 4. / 36., 1. / 36., 4. / 36., 4. / 36., 1. / 36., 4. / 36., 1. / 36.};

CUDA_CALLABLE_METHOD const int offset = 0;
// TODO Change hard coded offsets
constexpr int dimension_length = INX + 2;
CUDA_CALLABLE_METHOD const int compressedH_DN[3] = {dimension_length * dimension_length, dimension_length, 1};
CUDA_CALLABLE_METHOD const int dim_offset = dimension_length * dimension_length * dimension_length;
CUDA_CALLABLE_METHOD const int face_offset = 27 * dim_offset;
#endif

CUDA_GLOBAL_METHOD inline int flip_dim(const int d, const int flip_dim) {
    int dims[3];
    int k = d;
    for (int dim = 0; dim < 3; dim++) {
        dims[dim] = k % 3;
        k /= 3;
    }
    k = 0;
    dims[flip_dim] = 2 - dims[flip_dim];
    for (int dim = 0; dim < 3; dim++) {
        k *= 3;
        k += dims[2 - dim];
    }
    return k;
}

template <typename simd_t>
CUDA_GLOBAL_METHOD inline simd_t cell_inner_flux_loop_simd(const double omega, const size_t nf_,
    const double A_, const double B_, const std::array<simd_t, OCTOTIGER_MAX_NUMBER_FIELDS>& local_q,
    const std::array<simd_t, OCTOTIGER_MAX_NUMBER_FIELDS>& local_q_flipped,
    std::array<simd_t, OCTOTIGER_MAX_NUMBER_FIELDS> &this_flux, const std::array<simd_t, NDIM>& x,
    const std::array<simd_t, NDIM>& vg, simd_t& ap, simd_t& am, const size_t dim, const size_t d_tmp,
    const double dx, const double fgamma, const double de_switch_1,
    const size_t face_offset_tmp) {
    simd_t amr, apr, aml, apl;
    simd_t this_ap, this_am;    // tmps

    auto rho = local_q[rho_i];
    auto rhoinv = (1.) / rho;
    simd_t pdeg = static_cast<simd_t>(0.0), edeg = static_cast<simd_t>(0.0),
     dpdeg_drho = static_cast<simd_t>(0.0);

    // all workitems choose the same path
    if (A_ != 0.0) {
      // TODO Renable support for different EoS
      // probaly by adding serial fallback for the missing math functins
        const auto Binv = 1.0 / B_;
        const auto x = simd_fallbacks::pow_with_serial_fallback(rho * Binv, 1.0 / 3.0);

        const auto x_sqr = x * x;
        const auto x_sqr_sqrt = simd_fallbacks::sqrt_with_serial_fallback(x_sqr + simd_t(1.0));
        const auto x_pow_5 = x_sqr * x_sqr * x;
        const simd_t hdeg = 8.0 * A_ * Binv * (x_sqr_sqrt - 1.0);

        const simd_t pdeg_tmp1 =
            A_ * (x * (2.0 * x_sqr - 3.0) * x_sqr_sqrt + 3.0 *
                simd_fallbacks::asinh_with_serial_fallback(x));
        const simd_t pdeg_tmp2 = 1.6 * A_ * x_pow_5;

        const auto pdeg_mask = x < simd_t(0.001);
        pdeg = SIMD_NAMESPACE::choose(pdeg_mask, pdeg_tmp2, pdeg_tmp1);

        const simd_t edeg_tmp1 = rho * hdeg - pdeg;
        const simd_t edeg_tmp2 = 2.4 * A_ * x_pow_5;
        const auto edeg_mask = simd_t(0.001) < x;
        edeg = SIMD_NAMESPACE::choose(edeg_mask, edeg_tmp1, edeg_tmp2);

        dpdeg_drho = 8.0 / 3.0 * A_ * Binv * x_sqr / x_sqr_sqrt;
    }
    simd_t ek = 0.0;
    simd_t ein;
    for (int dim = 0; dim < NDIM; dim++) {
        ek += local_q[(sx_i + dim)] *
            local_q[(sx_i + dim)] * rhoinv * 0.5;
    }
    const auto ein1_tmp2 = local_q[egas_i] - ek - edeg;
    const auto ein1_mask =
        (ein1_tmp2 < (de_switch_1 * local_q[egas_i]));

    if (SIMD_NAMESPACE::any_of(ein1_mask)) {
        const auto ein1_tmp1 =
            simd_fallbacks::pow_with_serial_fallback(local_q[tau_i], fgamma);
        ein = SIMD_NAMESPACE::choose(ein1_mask, ein1_tmp1, ein1_tmp2);
    } else {
        ein = ein1_tmp2;
    }
    const auto dp_drho = dpdeg_drho + (fgamma - 1.0) * ein * rhoinv;
    const auto dp_deps = (fgamma - 1.0) * rho;
    const auto v0 = local_q[(sx_i + dim)] * rhoinv;
    const auto p = (fgamma - 1.0) * ein + pdeg;
    const auto c = simd_fallbacks::sqrt_with_serial_fallback(p * rhoinv * rhoinv * dp_deps + dp_drho);
    const auto v = v0 - vg[dim];
    amr = v - c;
    apr = v + c;

    rho = local_q_flipped[rho_i];
    rhoinv = (1.) / rho;
    pdeg = static_cast<simd_t>(0.0);
    edeg = static_cast<simd_t>(0.0);
    dpdeg_drho = static_cast<simd_t>(0.0);

    // all workitems choose the same path
    // from to_prim
    if (A_ != 0.0) {
        const auto Binv = 1.0 / B_;
        const auto x = simd_fallbacks::pow_with_serial_fallback(rho * Binv, 1.0 / 3.0);

        const auto x_sqr = x * x;
        const auto x_sqr_sqrt = simd_fallbacks::sqrt_with_serial_fallback(x_sqr + 1.0);
        const auto x_pow_5 = x_sqr * x_sqr * x;
        const simd_t hdeg = 8.0 * A_ * Binv * (x_sqr_sqrt - 1.0);

        const simd_t pdeg_tmp1 =
            A_ * (x * (2.0 * x_sqr - 3.0) * x_sqr_sqrt + 3.0 * simd_fallbacks::asinh_with_serial_fallback(x));
        const simd_t pdeg_tmp2 = 1.6 * A_ * x_pow_5;
        const auto pdeg_mask = x < simd_t(0.001);
        pdeg = SIMD_NAMESPACE::choose(pdeg_mask, pdeg_tmp2, pdeg_tmp1);

        const simd_t edeg_tmp1 = rho * hdeg - pdeg;
        const simd_t edeg_tmp2 = 2.4 * A_ * x_pow_5;
        const auto edeg_mask = simd_t(0.001) < x;
        edeg = SIMD_NAMESPACE::choose(edeg_mask, edeg_tmp1, edeg_tmp2);

        dpdeg_drho = 8.0 / 3.0 * A_ * Binv * x_sqr / x_sqr_sqrt;
    }
    ek = 0.0;
    for (int dim = 0; dim < NDIM; dim++) {
        ek += local_q_flipped[(sx_i + dim)] *
            local_q_flipped[(sx_i + dim)] * rhoinv * 0.5;
    }
    const auto ein2_tmp2 =
        local_q_flipped[egas_i] - ek - edeg;
    const auto ein2_mask =
        (ein2_tmp2 < (de_switch_1 * local_q_flipped[egas_i]));
    if (SIMD_NAMESPACE::any_of(ein2_mask)) {
        const auto ein2_tmp1 = simd_fallbacks::pow_with_serial_fallback(local_q_flipped[tau_i], fgamma);
        ein = SIMD_NAMESPACE::choose(ein2_mask, ein2_tmp1, ein2_tmp2);
    } else {
        ein = ein2_tmp2;
    }
    const auto dp_drho2 = dpdeg_drho + (fgamma - 1.0) * ein * rhoinv;
    const auto dp_deps2 = (fgamma - 1.0) * rho;
    const auto v02 = local_q_flipped[(sx_i + dim)] * rhoinv;
    const auto p2 = (fgamma - 1.0) * ein + pdeg;
    const auto c2 = simd_fallbacks::sqrt_with_serial_fallback(p2 * rhoinv * rhoinv * dp_deps2 + dp_drho2);
    const auto v2 = v02 - vg[dim];
    aml = v2 - c2;
    apl = v2 + c2;

    this_ap = SIMD_NAMESPACE::max(SIMD_NAMESPACE::max(apr, apl), simd_t(0.0));
    this_am = SIMD_NAMESPACE::min(SIMD_NAMESPACE::min(amr, aml), simd_t(0.0));
    const auto amp_mask = (this_ap - this_am == simd_t(0.0));
    for (int f = 0; f < nf_; f++) {
        simd_t fr = v * local_q[f];
        simd_t fl = v2 * local_q_flipped[f];

        if (f == sx_i + dim) {
            fr += p;
            fl += p2;
        } else if (f == egas_i) {
            fr += v0 * p;
            fl += v02 * p2;
        }
        if (dim == 0) {
            // levi_civita 1 2 0
            if (f == lx_i + 1) {
                fr += x[2] * p;
                fl += x[2] * p2;
            } else if (f == lx_i + 2) {
                // levi_civita 2 1 0
                fr -= x[1] * p;
                fl -= x[1] * p2;
            }
        } else if (dim == 1) {
            // levi_civita 0 2 1
            if (f == lx_i + 0) {
                fr -= x[2] * p;
                fl -= x[2] * p2;
                // 2 0 1
            } else if (f == lx_i + 2) {
                fr += x[0] * p;
                fl += x[0] * p2;
            }
        } else if (dim == 2) {
            if (f == lx_i) {
                // levi_civita 0 1 2
                fr += x[1] * p;
                fl += x[1] * p2;
                // 1 0 2
            } else if (f == lx_i + 1) {
                fr -= x[0] * p;
                fl -= x[0] * p2;
            }
        }

        if (SIMD_NAMESPACE::any_of(amp_mask)) {
            const simd_t flux_tmp1 =
                (this_ap * fl - this_am * fr +
                    this_ap * this_am *
                        (local_q[f] -
                            local_q_flipped[f])) /
                (this_ap - this_am);
            const simd_t flux_tmp2 = (fl + fr) / 2.0;
            this_flux[f] = SIMD_NAMESPACE::choose(amp_mask, flux_tmp2, flux_tmp1);
        } else {
            this_flux[f] = (this_ap * fl - this_am * fr +
                               this_ap * this_am *
                                   (local_q[f] -
                                       local_q_flipped[f])) /
                (this_ap - this_am);
        }
    }

    am = SIMD_NAMESPACE::min(am, this_am);
    ap = SIMD_NAMESPACE::max(ap, this_ap);
    return SIMD_NAMESPACE::max(ap, simd_t(-am));
}

/// Fills masks required for the flux kernel. Container needs to be able to hold NDIM * 1000 elements
template<typename mask_buffer_t>
void fill_masks(mask_buffer_t &masks) {
    //boost::container::vector<bool> masks(NDIM * 10 * 10 * 10);
    constexpr int length = INX + 2;
    constexpr int length_short = INX + 1;
    constexpr size_t dim_offset = length * length * length;
    const cell_geometry<3, 8> geo;
    for (int dim = 0; dim < NDIM; dim++) {
        std::array<int, NDIM> ubs = {length_short, length_short, length_short};
        for (int dimension = 0; dimension < NDIM; dimension++) {
            ubs[dimension] = geo.xloc()[geo.face_pts()[dim][0]][dimension] == -1 ? (length) : (length_short);
        }
        for (size_t ix = 0; ix < length; ix++) {
            for (size_t iy = 0; iy < length; iy++) {
                for (size_t iz = 0; iz < length; iz++) {
                    const size_t index = ix * length * length + iy * length + iz + dim_offset * dim;
                    if (ix > 0 && iy > 0 && iz > 0 && ix < ubs[0] && iy < ubs[1] && iz < ubs[2])
                        masks[index] = true;
                    else
                        masks[index] = false;
                }
            }
        }
    }
    return;
}
