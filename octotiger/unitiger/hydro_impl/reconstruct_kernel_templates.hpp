//  Copyright (c) 2020-2022 Gregor Daiß
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//


#pragma once
#include "octotiger/common_kernel/kokkos_simd.hpp"
#if defined(__clang__)
constexpr int number_dirs = 27;
constexpr int inx_large = INX + 6;
constexpr int inx_normal = INX + 4;
constexpr int q_inx = INX + 2;
constexpr int q_inx2 = q_inx * q_inx;
constexpr int q_inx3 = q_inx * q_inx * q_inx;
constexpr int q_face_offset = number_dirs * q_inx3;
constexpr int u_face_offset = H_N3;
constexpr int x_offset = H_N3;
constexpr int q_dir_offset = q_inx3;
constexpr int am_offset = q_inx3;
constexpr int HR_DNX = H_NX * H_NX;
constexpr int HR_DNY = H_NX;
constexpr int HR_DNZ = 1;
constexpr int HR_DN0 = 0;
//constexpr int NDIR = 27;
constexpr int disc_offset = H_NX * H_NX * H_NX;
constexpr int dir[27] = {
    /**/ -HR_DNX - HR_DNY - HR_DNZ, +HR_DN0 - HR_DNY - HR_DNZ, +HR_DNX - HR_DNY - HR_DNZ, /**/
    /**/ -HR_DNX + HR_DN0 - HR_DNZ, +HR_DN0 + HR_DN0 - HR_DNZ, +HR_DNX + HR_DN0 - HR_DNZ, /**/
    /**/ -HR_DNX + HR_DNY - HR_DNZ, +HR_DN0 + HR_DNY - HR_DNZ, +HR_DNX + HR_DNY - HR_DNZ, /**/
    /**/ -HR_DNX - HR_DNY + HR_DN0, +HR_DN0 - HR_DNY + HR_DN0, +HR_DNX - HR_DNY + HR_DN0, /**/
    /**/ -HR_DNX + HR_DN0 + HR_DN0, +HR_DN0 + HR_DN0 + HR_DN0, +HR_DNX + HR_DN0 + HR_DN0, /**/
    /**/ -HR_DNX + HR_DNY + HR_DN0, +HR_DN0 + HR_DNY + HR_DN0, +HR_DNX + HR_DNY + HR_DN0, /**/
    /**/ -HR_DNX - HR_DNY + HR_DNZ, +HR_DN0 - HR_DNY + HR_DNZ, +HR_DNX - HR_DNY + HR_DNZ, /**/
    /**/ -HR_DNX + HR_DN0 + HR_DNZ, +HR_DN0 + HR_DN0 + HR_DNZ, +HR_DNX + HR_DN0 + HR_DNZ, /**/
    /**/ -HR_DNX + HR_DNY + HR_DNZ, +HR_DN0 + HR_DNY + HR_DNZ, +HR_DNX + HR_DNY + HR_DNZ  /**/

};
constexpr safe_real vw[27] = {
    /**/ 1. / 216., 4. / 216., 1. / 216., 4. / 216., 16. / 216., 4. / 216., 1. / 216., 4. / 216.,
    1. / 216.,
    /****/ 4. / 216., 16. / 216., 4. / 216., 16. / 216., 64. / 216., 16. / 216., 4. / 216.,
    16. / 216., 4. / 216.,
    /****/ 1. / 216., 4. / 216., 1. / 216., 4. / 216., 16. / 216., 4. / 216., 1. / 216., 4. / 216.,
    1. / 216.};

constexpr int xloc[27][3] = {
    /**/ {-1, -1, -1}, {+0, -1, -1}, {+1, -1, -1},
    /**/ {-1, +0, -1}, {+0, +0, -1}, {1, +0, -1},
    /**/ {-1, +1, -1}, {+0, +1, -1}, {+1, +1, -1},
    /**/ {-1, -1, +0}, {+0, -1, +0}, {+1, -1, +0},
    /**/ {-1, +0, +0}, {+0, +0, +0}, {+1, +0, +0},
    /**/ {-1, +1, +0}, {+0, +1, +0}, {+1, +1, +0},
    /**/ {-1, -1, +1}, {+0, -1, +1}, {+1, -1, +1},
    /**/ {-1, +0, +1}, {+0, +0, +1}, {+1, +0, +1},
    /**/ {-1, +1, +1}, {+0, +1, +1}, {+1, +1, +1}};
#else
CUDA_CALLABLE_METHOD const int number_dirs = 27;
constexpr int inx_large = INX + 6;
constexpr int inx_normal = INX + 4;
CUDA_CALLABLE_METHOD const int q_inx = INX + 2;
CUDA_CALLABLE_METHOD const int q_inx2 = q_inx * q_inx;
CUDA_CALLABLE_METHOD const int q_inx3 = q_inx * q_inx * q_inx;
CUDA_CALLABLE_METHOD const int q_face_offset = number_dirs * q_inx3;
CUDA_CALLABLE_METHOD const int u_face_offset = H_N3;
CUDA_CALLABLE_METHOD const int x_offset = H_N3;
CUDA_CALLABLE_METHOD const int q_dir_offset = q_inx3;
CUDA_CALLABLE_METHOD const int am_offset = q_inx3;
CUDA_CALLABLE_METHOD const int HR_DNX = H_NX * H_NX;
CUDA_CALLABLE_METHOD const int HR_DNY = H_NX;
CUDA_CALLABLE_METHOD const int HR_DNZ = 1;
CUDA_CALLABLE_METHOD const int HR_DN0 = 0;
//CUDA_GLOBAL_METHOD const int NDIR = 27;
CUDA_CALLABLE_METHOD const int disc_offset = H_NX * H_NX * H_NX;
CUDA_CALLABLE_METHOD const int dir[27] = {
    /**/ -HR_DNX - HR_DNY - HR_DNZ, +HR_DN0 - HR_DNY - HR_DNZ, +HR_DNX - HR_DNY - HR_DNZ, /**/
    /**/ -HR_DNX + HR_DN0 - HR_DNZ, +HR_DN0 + HR_DN0 - HR_DNZ, +HR_DNX + HR_DN0 - HR_DNZ, /**/
    /**/ -HR_DNX + HR_DNY - HR_DNZ, +HR_DN0 + HR_DNY - HR_DNZ, +HR_DNX + HR_DNY - HR_DNZ, /**/
    /**/ -HR_DNX - HR_DNY + HR_DN0, +HR_DN0 - HR_DNY + HR_DN0, +HR_DNX - HR_DNY + HR_DN0, /**/
    /**/ -HR_DNX + HR_DN0 + HR_DN0, +HR_DN0 + HR_DN0 + HR_DN0, +HR_DNX + HR_DN0 + HR_DN0, /**/
    /**/ -HR_DNX + HR_DNY + HR_DN0, +HR_DN0 + HR_DNY + HR_DN0, +HR_DNX + HR_DNY + HR_DN0, /**/
    /**/ -HR_DNX - HR_DNY + HR_DNZ, +HR_DN0 - HR_DNY + HR_DNZ, +HR_DNX - HR_DNY + HR_DNZ, /**/
    /**/ -HR_DNX + HR_DN0 + HR_DNZ, +HR_DN0 + HR_DN0 + HR_DNZ, +HR_DNX + HR_DN0 + HR_DNZ, /**/
    /**/ -HR_DNX + HR_DNY + HR_DNZ, +HR_DN0 + HR_DNY + HR_DNZ, +HR_DNX + HR_DNY + HR_DNZ  /**/

};
CUDA_CALLABLE_METHOD const safe_real vw[27] = {
    /**/ 1. / 216., 4. / 216., 1. / 216., 4. / 216., 16. / 216., 4. / 216., 1. / 216., 4. / 216.,
    1. / 216.,
    /****/ 4. / 216., 16. / 216., 4. / 216., 16. / 216., 64. / 216., 16. / 216., 4. / 216.,
    16. / 216., 4. / 216.,
    /****/ 1. / 216., 4. / 216., 1. / 216., 4. / 216., 16. / 216., 4. / 216., 1. / 216., 4. / 216.,
    1. / 216.};

CUDA_CALLABLE_METHOD const int xloc[27][3] = {
    /**/ {-1, -1, -1}, {+0, -1, -1}, {+1, -1, -1},
    /**/ {-1, +0, -1}, {+0, +0, -1}, {1, +0, -1},
    /**/ {-1, +1, -1}, {+0, +1, -1}, {+1, +1, -1},
    /**/ {-1, -1, +0}, {+0, -1, +0}, {+1, -1, +0},
    /**/ {-1, +0, +0}, {+0, +0, +0}, {+1, +0, +0},
    /**/ {-1, +1, +0}, {+0, +1, +0}, {+1, +1, +0},
    /**/ {-1, -1, +1}, {+0, -1, +1}, {+1, -1, +1},
    /**/ {-1, +0, +1}, {+0, +0, +1}, {+1, +0, +1},
    /**/ {-1, +1, +1}, {+0, +1, +1}, {+1, +1, +1}};
#endif
// Utility functions

CUDA_GLOBAL_METHOD inline double deg_pres(double x, double A_) {
    double p;
    if (x < 0.001) {
        p = 1.6 * A_ * pow(x, 5);
    } else {
        p = A_ * (x * (2 * x * x - 3) * sqrt(x * x + 1) + 3 * asinh(x));
    }
    return p;
}
CUDA_GLOBAL_METHOD inline int to_q_index(const int j, const int k, const int l) {
    return j * q_inx * q_inx + k * q_inx + l;
}
CUDA_GLOBAL_METHOD inline int flip(const int d) {
    return NDIR - 1 - d;
}
// =================================================================================================
// Find contact discs kernel methods
// =================================================================================================

template <typename container_t, typename const_container_t>
CUDA_GLOBAL_METHOD inline void cell_find_contact_discs_phase1(container_t &P,
    const_container_t &combined_u, const double A_, const double B_, const double fgamma_,
    const double de_switch_1, const int nf, const unsigned int x, const unsigned int y,
    const unsigned int z, const int slice_id) {
    const int i = (x + 1) * inx_large * inx_large + (y + 1) * inx_large + (z + 1);

    const int u_slice_offset = (H_N3 * nf + 128) * slice_id; 
    const int p_slice_offset = (H_N3 + 128) * slice_id; 

    const auto rho = combined_u[u_slice_offset + rho_i * u_face_offset + i];
    const auto rhoinv = 1.0 / rho;
    double pdeg = 0.0, edeg = 0.0;

    if (A_ != 0.0) {
        const auto x = std::pow(rho / B_, 1.0 / 3.0);
        const double hdeg = 8.0 * A_ / B_ * (std::sqrt(x * x + 1.0) - 1.0);
        pdeg = deg_pres(x, A_);
        edeg = rho * hdeg - pdeg;
    }

    safe_real ek = 0.0;
    for (int dim = 0; dim < NDIM; dim++) {
        ek += combined_u[u_slice_offset + (sx_i + dim) * u_face_offset + i] *
            combined_u[u_slice_offset + (sx_i + dim) * u_face_offset + i] * rhoinv * 0.5;
    }

    auto ein = combined_u[u_slice_offset + egas_i * u_face_offset + i] - ek - edeg;
    if (ein < de_switch_1 * combined_u[u_slice_offset + egas_i * u_face_offset + i]) {
        ein = pow(combined_u[u_slice_offset + tau_i * u_face_offset + i], fgamma_);
    }
    P[p_slice_offset + i] = (fgamma_ - 1.0) * ein + pdeg;
}

template <typename container_t, typename const_container_t>
CUDA_GLOBAL_METHOD inline void cell_find_contact_discs_phase2(
    container_t &disc, const_container_t &P, const double fgamma_,
    const int ndir, const unsigned int x, const unsigned int y, const unsigned int z, const int slice_id) {

    const int p_slice_offset = (H_N3 + 128) * slice_id; 
    const int disc_slice_offset = (ndir / 2 * H_N3 + 128) * slice_id; 

    const int disc_offset = inx_large * inx_large * inx_large;

    const double K0 = 0.1;
    const int i = (x + 2) * inx_large * inx_large + (y + 2) * inx_large + (z + 2);
    for (int d = 0; d < ndir / 2; d++) {
        const auto di = dir[d];
        const double P_r = P[p_slice_offset + i + di];
        const double P_l = P[p_slice_offset + i - di];
        const double tmp1 = fgamma_ * K0;
        const double tmp2 = abs(P_r - P_l) / std::min(std::abs(P_r), abs(P_l));
        disc[disc_slice_offset + d * disc_offset + i] = tmp2 / tmp1;
    }
}

// =================================================================================================
// Pre Recon Methods
// =================================================================================================

template <typename container_t, typename const_container_t>
CUDA_GLOBAL_METHOD inline void cell_hydro_pre_recon(const_container_t& X, safe_real omega, bool angmom,
        container_t &u, const int nf, const int n_species_, const unsigned int x, const unsigned int y, const unsigned int z, const int slice_id) {
    const int i = (x) * inx_large * inx_large + (y) * inx_large + (z);
    const int u_slice_offset = (H_N3 * nf + 128) * slice_id; 
    const int large_x_slice_offset = (H_N3 * NDIM + 128) * slice_id; 

    const auto rho = u[u_slice_offset + rho_i * u_face_offset + i];
    const auto rhoinv = 1.0 / rho;
    for (int dim = 0; dim < NDIM; dim++) {
        auto& s = u[u_slice_offset + (sx_i + dim) * u_face_offset + i];
        u[u_slice_offset + egas_i * u_face_offset + i] -= 0.5 * s * s * rhoinv;
        s *= rhoinv;
    }
    for (int si = 0; si < n_species_; si++) {
        u[u_slice_offset + (spc_i + si) * u_face_offset + i] *= rhoinv;
    }
    u[u_slice_offset + pot_i * u_face_offset + i] *= rhoinv;

    u[u_slice_offset + (lx_i + 0) * u_face_offset + i] *= rhoinv;
    u[u_slice_offset + (lx_i + 1) * u_face_offset + i] *= rhoinv;
    u[u_slice_offset + (lx_i + 2) * u_face_offset + i] *= rhoinv;

    // Levi civita n m q -> lc
    // Levi civita 0 1 2 -> 1
    u[u_slice_offset + (lx_i + 0) * u_face_offset + i] -=
        1.0 * X[large_x_slice_offset + 1 * x_offset + i] * u[u_slice_offset + (sx_i + 2) * u_face_offset + i];
    // Levi civita n m q -> lc
    // Levi civita 0 2 1 -> -1
    u[u_slice_offset + (lx_i + 0) * u_face_offset + i] -=
        -1.0 * X[large_x_slice_offset + 2 * x_offset + i] * u[u_slice_offset + (sx_i + 1) * u_face_offset + i];
    // Levi civita n m q -> lc
    // Levi civita 1 0 2 -> -1
    u[u_slice_offset + (lx_i + 1) * u_face_offset + i] -=
        -1.0 * X[large_x_slice_offset + i] * u[u_slice_offset + (sx_i + 2) * u_face_offset + i];
    // Levi civita n m q -> lc
    // Levi civita 1 2 0 -> 1
    u[u_slice_offset + (lx_i + 1) * u_face_offset + i] -=
        1.0 * X[large_x_slice_offset + 2 * x_offset + i] * u[u_slice_offset + (sx_i + 0) * u_face_offset + i];
    // Levi civita n m q -> lc
    // Levi civita 2 0 1 -> 1
    u[u_slice_offset + (lx_i + 2) * u_face_offset + i] -=
        1.0 * X[large_x_slice_offset + i] * u[u_slice_offset + (sx_i + 1) * u_face_offset + i];
    // Levi civita n m q -> lc
    // Levi civita 2 1 0 -> -1
    u[u_slice_offset + (lx_i + 2) * u_face_offset + i] -=
        -1.0 * X[large_x_slice_offset + 1 * x_offset + i] * u[u_slice_offset + (sx_i + 0) * u_face_offset + i];

    u[u_slice_offset + sx_i * u_face_offset + i] += omega * X[large_x_slice_offset + 1 * x_offset + i];
    u[u_slice_offset + sy_i * u_face_offset + i] -= omega * X[large_x_slice_offset + 0 * x_offset + i];
}
// =================================================================================================
// Reconstruct simd methods
// =================================================================================================

template<typename simd_t, typename simd_mask_t>
CUDA_GLOBAL_METHOD inline void make_monotone_simd(simd_t& ql, simd_t q0, simd_t& qr) {
    const simd_t tmp1 = qr - ql;
    const simd_t tmp2 = qr + ql;

    const simd_mask_t mask1 = ((qr < q0) && (q0 < ql)) || (!(qr < q0) && !(q0 < ql));
    const simd_mask_t mask2 = !mask1;
    qr = SIMD_NAMESPACE::choose(mask2, q0, qr);
    ql = SIMD_NAMESPACE::choose(mask2, q0, ql);
    if (SIMD_NAMESPACE::all_of(mask2))
      return;
    const simd_t tmp3 = tmp1 * tmp1 / 6.0;
    const simd_t tmp4 = tmp1 * (q0 - 0.5 * tmp2);
    const simd_mask_t mask3 = (tmp3 < tmp4) && mask1;
    const simd_mask_t mask4 = (tmp4 < -tmp3) && mask1;
    ql = SIMD_NAMESPACE::choose(mask3, (3.0 * q0 - 2.0 * qr), ql);
    qr = SIMD_NAMESPACE::choose(mask4, (3.0 * q0 - 2.0 * ql), qr);
}

template <typename simd_t>
CUDA_GLOBAL_METHOD inline simd_t minmod_simd(simd_t a, simd_t b) {
    return (simd_fallbacks::copysign_with_serial_fallback<simd_t>(0.5, a) +
               simd_fallbacks::copysign_with_serial_fallback<simd_t>(0.5, b)) *
        SIMD_NAMESPACE::min(SIMD_NAMESPACE::abs(a), SIMD_NAMESPACE::abs(b));
}
template <typename simd_t>
CUDA_GLOBAL_METHOD inline simd_t minmod_theta_simd(simd_t a, simd_t b, simd_t c) {
    return minmod_simd(c * minmod_simd(a, b), 0.5 * (a + b));
}
template <typename simd_t, typename simd_mask_t>
CUDA_GLOBAL_METHOD inline void cell_reconstruct_minmod_simd(double* __restrict__ combined_q,
    const double* __restrict__ combined_u_face, int d, int f, int i, int q_i) {
    const auto di = dir[d];
    const int start_index = f * q_face_offset + d * q_dir_offset;
    simd_t result;

    simd_t u_plus_di;
    simd_t u_zero;
    simd_t u_minus_di;
    /* As combined_u and combined_q are differently indexed (i and q_i respectively),
     * we need to take when loading an entire simd lane of u values as those might
     * across 2 bars in the cube and are thus not necessarily consecutive in memory.
     * Thus we first check if the values are consecutive in memory - if yes we load
     * them immediately, if not we load the values manually from the first and
     * second bar in the else branch (element-wise unfortunately) */
    if (q_i%q_inx + simd_t::size() - 1 < q_inx) { 
        // values are all in the first line/bar and can simply be loaded
        u_plus_di.copy_from(combined_u_face + f * u_face_offset + i + di,
            SIMD_NAMESPACE::element_aligned_tag{});
        u_zero.copy_from(combined_u_face + f * u_face_offset + i,
            SIMD_NAMESPACE::element_aligned_tag{});
        u_minus_di.copy_from(combined_u_face + f * u_face_offset + i - di,
            SIMD_NAMESPACE::element_aligned_tag{});
    } else {
        // TODO std::simd should have a specialization for partial loads
        // which would allow us to skip this inefficient implementation of element-wise copies
        std::array<double, simd_t::size()> u_zero_helper;
        std::array<double, simd_t::size()> u_minus_di_helper;
        std::array<double, simd_t::size()> u_plus_di_helper;
        size_t simd_i = 0;
        // load from first bar
        for(size_t i_line = q_i%q_inx; i_line < q_inx; i_line++, simd_i++) {
          u_zero_helper[simd_i] = combined_u_face[f * u_face_offset + i + simd_i];
          u_minus_di_helper[simd_i] = combined_u_face[f * u_face_offset + i - di + simd_i];
          u_plus_di_helper[simd_i] = combined_u_face[f * u_face_offset + i + di + simd_i];
        }
        // calculate indexing offset to check where the second line/bar is starting
        size_t offset = (inx_large - q_inx);
        if constexpr (q_inx2 % simd_t::size() != 0) {
          if ((q_i + simd_i)%q_inx2 == 0) {
            offset += (inx_large - q_inx) * inx_large;
          }
        } 
        // Load relevant values from second line/bar 
        for(; simd_i < simd_t::size(); simd_i++) {
          u_zero_helper[simd_i] = combined_u_face[f * u_face_offset + i + simd_i + offset];
          u_minus_di_helper[simd_i] = combined_u_face[f * u_face_offset + i - di + simd_i + offset];
          u_plus_di_helper[simd_i] = combined_u_face[f * u_face_offset + i + di + simd_i + offset];
        }
        // Copy from tmp helpers into working buffers
        u_plus_di.copy_from(u_plus_di_helper.data(),
            SIMD_NAMESPACE::element_aligned_tag{});
        u_zero.copy_from(u_zero_helper.data(),
            SIMD_NAMESPACE::element_aligned_tag{});
        u_minus_di.copy_from(u_minus_di_helper.data(),
            SIMD_NAMESPACE::element_aligned_tag{});
    }
    result = u_zero +
        0.5 *
            minmod_simd(u_plus_di -
                    u_zero,
                u_zero -
                    u_minus_di);
    result.copy_to(combined_q + q_i + start_index, SIMD_NAMESPACE::element_aligned_tag{});
}

template <typename simd_t, typename simd_mask_t>
CUDA_GLOBAL_METHOD inline void cell_reconstruct_ppm_simd(double *__restrict__ combined_q,
    const double* __restrict__ combined_u_face, bool smooth, bool disc_detect,
    const double* __restrict__ disc, const int d, const int f, int i, int q_i,
    int d_i) {
    const auto di = dir[d];
    const auto flipped_di = flip(d);

    const int start_index = f * q_face_offset + d * q_dir_offset;
    const int start_index_flipped = f * q_face_offset + flipped_di * q_dir_offset;

    simd_t u_plus_2di;
    simd_t u_plus_di;
    simd_t u_zero;
    simd_t u_minus_di;
    simd_t u_minus_2di;
    /* As combined_u and combined_q are differently indexed (i and q_i respectively),
     * we need to take when loading an entire simd lane of u values as those might
     * across 2 bars in the cube and are thus not necessarily consecutive in memory.
     * Thus we first check if the values are consecutive in memory - if yes we load
     * them immediately, if not we load the values manually from the first and
     * second bar in the else branch (element-wise unfortunately) */
    if (q_i%q_inx + simd_t::size() - 1 < q_inx) { 
        // values are all in the first line/bar and can simply be loaded
        u_plus_2di.copy_from(combined_u_face + f * u_face_offset + i + 2 * di,
            SIMD_NAMESPACE::element_aligned_tag{});
        u_plus_di.copy_from(combined_u_face + f * u_face_offset + i + di,
            SIMD_NAMESPACE::element_aligned_tag{});
        u_zero.copy_from(combined_u_face + f * u_face_offset + i,
            SIMD_NAMESPACE::element_aligned_tag{});
        u_minus_di.copy_from(combined_u_face + f * u_face_offset + i - di,
            SIMD_NAMESPACE::element_aligned_tag{});
        u_minus_2di.copy_from(combined_u_face + f * u_face_offset + i - 2 * di,
            SIMD_NAMESPACE::element_aligned_tag{});
    } else {
        // TODO std::simd should have a specialization for partial loads
        // which would allow us to skip this inefficient implementation of element-wise copies
        //
        std::array<double, simd_t::size()> u_zero_helper;
        std::array<double, simd_t::size()> u_minus_di_helper;
        std::array<double, simd_t::size()> u_minus_2di_helper;
        std::array<double, simd_t::size()> u_plus_di_helper;
        std::array<double, simd_t::size()> u_plus_2di_helper;
        size_t simd_i = 0;
        // load from first bar
        for(size_t i_line = q_i%q_inx; i_line < q_inx; i_line++, simd_i++) {
          u_zero_helper[simd_i] = combined_u_face[f * u_face_offset + i + simd_i];
          u_minus_di_helper[simd_i] = combined_u_face[f * u_face_offset + i - di + simd_i];
          u_minus_2di_helper[simd_i] = combined_u_face[f * u_face_offset + i - 2 * di + simd_i];
          u_plus_di_helper[simd_i] = combined_u_face[f * u_face_offset + i + di + simd_i];
          u_plus_2di_helper[simd_i] = combined_u_face[f * u_face_offset + i + 2 * di + simd_i];
        }
        // calculate indexing offset to check where the second line/bar is starting
        size_t offset = (inx_large - q_inx);
        if constexpr (q_inx2 % simd_t::size() != 0) {
          if ((q_i + simd_i)%q_inx2 == 0) {
            offset += (inx_large - q_inx) * inx_large;
          }
        } 
        // Load relevant values from second line/bar 
        for(; simd_i < simd_t::size(); simd_i++) {
          u_zero_helper[simd_i] = combined_u_face[f * u_face_offset + i + simd_i + offset];
          u_minus_di_helper[simd_i] = combined_u_face[f * u_face_offset + i - di + simd_i + offset];
          u_minus_2di_helper[simd_i] = combined_u_face[f * u_face_offset + i - 2 * di + simd_i + offset];
          u_plus_di_helper[simd_i] = combined_u_face[f * u_face_offset + i + di + simd_i + offset];
          u_plus_2di_helper[simd_i] = combined_u_face[f * u_face_offset + i + 2 * di + simd_i + offset];
        }
        // Copy from tmp helpers into working buffers
        u_plus_2di.copy_from(u_plus_2di_helper.data(),
            SIMD_NAMESPACE::element_aligned_tag{});
        u_plus_di.copy_from(u_plus_di_helper.data(),
            SIMD_NAMESPACE::element_aligned_tag{});
        u_zero.copy_from(u_zero_helper.data(),
            SIMD_NAMESPACE::element_aligned_tag{});
        u_minus_di.copy_from(u_minus_di_helper.data(),
            SIMD_NAMESPACE::element_aligned_tag{});
        u_minus_2di.copy_from(u_minus_2di_helper.data(),
            SIMD_NAMESPACE::element_aligned_tag{});
    }

    const simd_t diff_u_plus = u_plus_di - u_zero;
    const simd_t diff_u_2plus = u_plus_2di - u_plus_di;

    const simd_t diff_u_minus = u_zero - u_minus_di;
    const simd_t diff_u_2minus = u_minus_di - u_minus_2di;
    const simd_t d1 = minmod_theta_simd(diff_u_plus, diff_u_minus, simd_t(2.0));
    const simd_t d1_plus = minmod_theta_simd(diff_u_2plus, diff_u_plus, simd_t(2.0));
    const simd_t d1_minus = minmod_theta_simd(diff_u_minus, diff_u_2minus, simd_t(2.0));

    const simd_t results = 0.5 * (u_zero + u_plus_di) + (1.0 / 6.0) * (d1 - d1_plus);
    const simd_t results_flipped = 0.5 * (u_minus_di + u_zero) + (1.0 / 6.0) * (d1_minus - d1);

    simd_t current_q_results(combined_q + start_index + q_i,
        SIMD_NAMESPACE::element_aligned_tag{});
    simd_t current_q_results_flipped(combined_q + start_index_flipped + q_i,
        SIMD_NAMESPACE::element_aligned_tag{});
    const simd_t old_results = current_q_results;
    const simd_t old_results_flipped = current_q_results_flipped;

    current_q_results = results;
    current_q_results_flipped = results_flipped;

    if (disc_detect) {
        constexpr auto eps = 0.01;
        constexpr auto eps2 = 0.001;
        constexpr auto eta1 = 20.0;
        constexpr auto eta2 = 0.05;
        const auto dif = u_plus_di - u_minus_di;
        simd_t disc_val;
        /* Same issue as with the loading from combined_u: we first need to check 
         * if the values are consecutive in memory of if we are dealing with two lines/
         * bars of the cube and need to load them separetly (else branch)*/
        if (q_i%q_inx + simd_t::size() - 1 < q_inx) { 
            // values are consecutive
            disc_val.copy_from(disc + d * disc_offset + d_i, SIMD_NAMESPACE::element_aligned_tag{});
        } else {
            // values need to be loaded from two lines/bars of the cube
            std::array<double, simd_t::size()> disc_helper;
            size_t simd_i = 0;
            for(size_t i_line = q_i%q_inx; i_line < q_inx; i_line++, simd_i++)
                disc_helper[simd_i] = disc[d * disc_offset + d_i + simd_i];
            size_t offset = (inx_large - q_inx);
            // calculate index offset to the second bar
            if constexpr (q_inx2 % simd_t::size() != 0) {
              if ((q_i + simd_i)%q_inx2 == 0) {
                offset += (inx_large - q_inx) * inx_large;
              }
            } 
            for(; simd_i < simd_t::size(); simd_i++)
                disc_helper[simd_i] = disc[d * disc_offset + d_i + simd_i + offset];
            disc_val.copy_from(disc_helper.data(),
                SIMD_NAMESPACE::element_aligned_tag{});
        }
        const simd_t mask_helper1 =
            disc_val *
            SIMD_NAMESPACE::min(SIMD_NAMESPACE::abs(u_plus_di), SIMD_NAMESPACE::abs(u_minus_di));
        simd_mask_t criterias = (mask_helper1 < (SIMD_NAMESPACE::abs(dif)));
        if (SIMD_NAMESPACE::any_of(criterias)) {
            const simd_t mask_helper2 = SIMD_NAMESPACE::min(SIMD_NAMESPACE::abs(u_plus_di),
                                          SIMD_NAMESPACE::abs(u_minus_di)) /
                SIMD_NAMESPACE::max(SIMD_NAMESPACE::abs(u_plus_di), SIMD_NAMESPACE::abs(u_minus_di));
            criterias = (simd_t(eps2) < mask_helper2) && criterias;
            if (SIMD_NAMESPACE::any_of(criterias)) {
                const auto d2p = (1.0 / 6.0) * (u_plus_2di + u_zero - 2.0 * u_plus_di);
                const auto d2m = (1.0 / 6.0) * (u_zero + u_minus_2di - 2.0 * u_minus_di);
                criterias = criterias && ((d2p * d2m) < simd_t(0.0));
                if (SIMD_NAMESPACE::any_of(criterias)) {
                    simd_t eta = 0.0;
                    const simd_mask_t eta_mask =
                        (eps *
                            SIMD_NAMESPACE::min(SIMD_NAMESPACE::abs(u_plus_di),
                                SIMD_NAMESPACE::abs(u_minus_di))) < SIMD_NAMESPACE::abs(dif);
                    eta = SIMD_NAMESPACE::choose(eta_mask, -(d2p - d2m) / dif, eta);
                    eta = SIMD_NAMESPACE::max(simd_t(0.0),
                        SIMD_NAMESPACE::min(simd_t(eta1) * (eta - simd_t(eta2)), simd_t(1.0)));
                    criterias = criterias && (simd_t(0.0) < eta);
                    auto ul = u_minus_di +
                        0.5 *
                            minmod_theta_simd(
                                u_zero - u_minus_di, u_minus_di - u_minus_2di, simd_t(2.0));
                    auto ur = u_plus_di -
                        0.5 *
                            minmod_theta_simd(u_plus_2di - u_plus_di,
                                u_plus_di - u_zero, simd_t(2.0));
                    current_q_results = current_q_results +
                        SIMD_NAMESPACE::choose(
                            criterias, eta * (ur - current_q_results), simd_t(0.0));
                    current_q_results_flipped = current_q_results_flipped +
                        SIMD_NAMESPACE::choose(
                            criterias, eta * (ul - current_q_results_flipped), simd_t(0.0));
                    /* } */
                }
            }
        }
    }
    if (!smooth) {
        make_monotone_simd<simd_t, simd_mask_t>(current_q_results, u_zero,
            current_q_results_flipped);
    }
    // TODO compatibility when moving to KOKKOS HPX backend? Might overwrite stuff from other tasks using AVX512...
    // Hotfix variant 1: element-wise adding in case of not when_all...
    // Hotfix variant 2: Pick task sizes that have more padding (aka 2D border with 2*qinx)
    /* current_q_results = SIMD_NAMESPACE::choose(mask, current_q_results, old_results); */
    current_q_results.copy_to(combined_q + start_index + q_i, SIMD_NAMESPACE::element_aligned_tag{});
    /* current_q_results_flipped = SIMD_NAMESPACE::choose(mask, current_q_results_flipped, old_results_flipped); */
    current_q_results_flipped.copy_to(
        combined_q + start_index_flipped + q_i, SIMD_NAMESPACE::element_aligned_tag{});
}

// Phase 1 and 2
template <typename simd_t, typename simd_mask_t>
CUDA_GLOBAL_METHOD inline void cell_reconstruct_inner_loop_p1_simd(const size_t nf_,
    const int angmom_index_, const int* __restrict__ smooth_field_,
    const int* __restrict__ disc_detect_, double* __restrict__ combined_q,
    const double* __restrict__ combined_u, double* __restrict__ AM, const
    double dx, const double* __restrict__ cdiscs, const int i, const int q_i,
    const int ndir, const int nangmom, const int slice_id) {
    /* const int q_slice_offset = (nf_ * 27 * q_inx3 + 128) * slice_id; */
    /* const int u_slice_offset = (nf_ * H_N3 + 128) * slice_id; */
    /* const int am_slice_offset = (NDIM * q_inx3 + 128) * slice_id; */
    /* const int disc_slice_offset = (ndir / 2 * H_N3 + 128) * slice_id; */ 

    int l_start;
    int s_start;
    if (angmom_index_ > -1) {
        s_start = angmom_index_;
        l_start = angmom_index_ + NDIM;
    } else {
        s_start = lx_i;
        l_start = lx_i;
    }
    if (angmom_index_ > -1) {
        for (int f = 0; f < s_start; f++) {
            for (int d = 0; d < ndir; d++) {
                if (d < ndir / 2) {
                    cell_reconstruct_ppm_simd<simd_t, simd_mask_t>(combined_q, combined_u,
                        smooth_field_[f], disc_detect_[f], cdiscs, d,
                        f, i, q_i, i);
                }
            }
        }
        for (int f = s_start; f < l_start; f++) {
            for (int d = 0; d < ndir; d++) {
                if (d < ndir / 2) {
                    cell_reconstruct_ppm_simd<simd_t, simd_mask_t>(combined_q, combined_u, true, false,
                        cdiscs, d, f, i, q_i, i);
                }
            }
        }
        for (int f = l_start; f < l_start + nangmom; f++) {
            for (int d = 0; d < ndir; d++) {
                cell_reconstruct_minmod_simd<simd_t, simd_mask_t>(
                    combined_q, combined_u, d, f, i, q_i);
            }
        }
        for (int f = l_start + nangmom; f < nf_; f++) {
            for (int d = 0; d < ndir; d++) {
                if (d < ndir / 2) {
                    cell_reconstruct_ppm_simd<simd_t, simd_mask_t>(combined_q, combined_u,
                        smooth_field_[f], disc_detect_[f], cdiscs, d,
                        f, i, q_i, i);
                }
            }
        }
    } else {
        for (int f = 0; f < nf_; f++) {
            if (f < lx_i || f > lx_i + nangmom) {
                for (int d = 0; d < ndir; d++) {
                    if (d < ndir / 2) {
                      cell_reconstruct_ppm_simd<simd_t, simd_mask_t>(combined_q, combined_u,
                          smooth_field_[f], disc_detect_[f], cdiscs,
                          d, f, i, q_i, i);
                    }
                }
            } else {
                for (int d = 0; d < ndir; d++) {
                    cell_reconstruct_minmod_simd<simd_t, simd_mask_t>(
                        combined_q, combined_u, d, f, i, q_i);
                }
            }
        }
    }

    for (int d = 0; d < ndir; d++) {
        if (d != ndir / 2 && angmom_index_ > -1) {
            const int start_index_rho = d * q_dir_offset;

            // n m q Levi Civita
            // 0 1 2 -> 1
            simd_t results0 =
                simd_t(AM + q_i, SIMD_NAMESPACE::element_aligned_tag{}) -
                vw[d] * 1.0 * 0.5 * xloc[d][1] *
                    simd_t(combined_q + (sx_i + 2) * q_face_offset + d *
                        q_dir_offset + q_i,
                        SIMD_NAMESPACE::element_aligned_tag{}) *
                    simd_t(combined_q + start_index_rho + q_i,
                        SIMD_NAMESPACE::element_aligned_tag{}) *
                    dx;

            // n m q Levi Civita
            // 0 2 1 -> -1
            results0 = results0 -
                vw[d] * (-1.0) * 0.5 * xloc[d][2] *
                    simd_t(combined_q + (sx_i + 1) * q_face_offset + d * q_dir_offset + q_i,
                        SIMD_NAMESPACE::element_aligned_tag{}) *
                    simd_t(combined_q + start_index_rho + q_i,
                        SIMD_NAMESPACE::element_aligned_tag{}) *
                    dx;
            // copy results 0 back
            results0.copy_to(AM + q_i,
                SIMD_NAMESPACE::element_aligned_tag{});

            // n m q Levi Civita
            // 1 0 2 -> -1
            simd_t results1 = simd_t(AM + am_offset + q_i,
                                  SIMD_NAMESPACE::element_aligned_tag{}) -
                vw[d] * (-1.0) * 0.5 * xloc[d][0] *
                    simd_t(combined_q + (sx_i + 2) * q_face_offset + d * q_dir_offset + q_i,
                        SIMD_NAMESPACE::element_aligned_tag{}) *
                    simd_t(combined_q + start_index_rho + q_i,
                        SIMD_NAMESPACE::element_aligned_tag{}) *
                    dx;

            // n m q Levi Civita 1 2 0 -> 1
            results1 -= vw[d] * (1.0) * 0.5 * xloc[d][2] *
                simd_t(combined_q + (sx_i + 0) * q_face_offset + d * q_dir_offset + q_i,
                    SIMD_NAMESPACE::element_aligned_tag{}) *
                simd_t(combined_q + start_index_rho + q_i,
                    SIMD_NAMESPACE::element_aligned_tag{}) *
                dx;
            // copy results 1 back
            results1.copy_to(
                AM + am_offset + q_i, SIMD_NAMESPACE::element_aligned_tag{});

            // n m q Levi Civita
            // 2 0 1 -> 1
            simd_t results2 = simd_t(AM + 2 * am_offset + q_i,
                                  SIMD_NAMESPACE::element_aligned_tag{}) -
                vw[d] * (1.0) * 0.5 * xloc[d][0] *
                    simd_t(combined_q + (sx_i + 1) * q_face_offset + d * q_dir_offset + q_i,
                        SIMD_NAMESPACE::element_aligned_tag{}) *
                    simd_t(combined_q + start_index_rho + q_i,
                        SIMD_NAMESPACE::element_aligned_tag{}) *
                    dx;

            // n m q Levi Civita 2 1 0 -> -1
            results2 -= vw[d] * (-1.0) * 0.5 * xloc[d][1] *
                simd_t(combined_q + (sx_i + 0) * q_face_offset + d * q_dir_offset + q_i,
                    SIMD_NAMESPACE::element_aligned_tag{}) *
                simd_t(combined_q + start_index_rho + q_i,
                    SIMD_NAMESPACE::element_aligned_tag{}) *
                dx;
            // copy results 2 back
            results2.copy_to(AM + 2 * am_offset + q_i,
                SIMD_NAMESPACE::element_aligned_tag{});
        }
    }
}

template <typename simd_t, typename simd_mask_t>
CUDA_GLOBAL_METHOD inline void cell_reconstruct_inner_loop_p2_simd(const safe_real omega,
    const int angmom_index_, double* __restrict__ combined_q, const double* __restrict__ combined_x,
    const double* __restrict__ combined_u, const double* __restrict__ AM, const double dx,
    const int d, const int i, const int q_i, const int ndir, const int nangmom,
    const int n_species_, const int nf_, const int slice_id) {

    if (d < ndir / 2 && angmom_index_ > -1) {
        const auto di = dir[d];

        for (int q = 0; q < NDIM; q++) {
            const auto f = sx_i + q;
            const int start_index_f = f * q_face_offset + d * q_dir_offset;
            const int start_index_flipped = f * q_face_offset + flip(d) * q_dir_offset;
            const int start_index_zero = 0 * q_face_offset + d * q_dir_offset;
            const int start_index_zero_flipped = 0 * q_face_offset + flip(d) *
              q_dir_offset;

            const simd_t rho_r(combined_q + start_index_zero + q_i,
                SIMD_NAMESPACE::element_aligned_tag{});
            const simd_t rho_l(combined_q + start_index_zero_flipped + q_i,
                SIMD_NAMESPACE::element_aligned_tag{});

            /* const simd_t ur(combined_u + f * u_face_offset + i + u_slice_offset + di, */
            /*     SIMD_NAMESPACE::element_aligned_tag{}); */
            /* const simd_t u0(combined_u + f * u_face_offset + i + u_slice_offset, */
            /*     SIMD_NAMESPACE::element_aligned_tag{}); */
            /* const simd_t ul(combined_u + f * u_face_offset + i + u_slice_offset - di, */
            /*     SIMD_NAMESPACE::element_aligned_tag{}); */
            simd_t ur;
            simd_t u0;
            simd_t ul;
            /* As combined_u and combined_q are differently indexed (i and q_i respectively),
             * we need to take when loading an entire simd lane of u values as those might
             * across 2 bars in the cube and are thus not necessarily consecutive in memory.
             * Thus we first check if the values are consecutive in memory - if yes we load
             * them immediately, if not we load the values manually from the first and
             * second bar in the else branch (element-wise unfortunately) */
            if (q_i%q_inx + simd_t::size() - 1 < q_inx) { 
                // values are all in the first line/bar and can simply be loaded
                ur.copy_from(combined_u + f * u_face_offset + i + di,
                    SIMD_NAMESPACE::element_aligned_tag{});
                u0.copy_from(combined_u + f * u_face_offset + i,
                    SIMD_NAMESPACE::element_aligned_tag{});
                ul.copy_from(combined_u + f * u_face_offset + i - di,
                    SIMD_NAMESPACE::element_aligned_tag{});
            } else {
                // TODO std::simd should have a specialization for partial loads
                // which would allow us to skip this inefficient implementation of element-wise copies
                std::array<double, simd_t::size()> ur_helper;
                std::array<double, simd_t::size()> u0_helper;
                std::array<double, simd_t::size()> ul_helper;
                size_t simd_i = 0;
                // load from first bar
                for(size_t i_line = q_i%q_inx; i_line < q_inx; i_line++, simd_i++) {
                  ur_helper[simd_i] = combined_u[f * u_face_offset + i + di + simd_i];
                  u0_helper[simd_i] = combined_u[f * u_face_offset + i + simd_i];
                  ul_helper[simd_i] = combined_u[f * u_face_offset + i - di + simd_i];
                }
                // calculate indexing offset to check where the second line/bar is starting
                size_t offset = (inx_large - q_inx);
                if constexpr (q_inx2 % simd_t::size() != 0) {
                  if ((q_i + simd_i)%q_inx2 == 0) {
                    offset += (inx_large - q_inx) * inx_large;
                  }
                } 
                // Load relevant values from second line/bar 
                for(; simd_i < simd_t::size(); simd_i++) {
                    ur_helper[simd_i] =
                        combined_u[f * u_face_offset + i + di + simd_i + offset];
                    u0_helper[simd_i] =
                        combined_u[f * u_face_offset + i + simd_i + offset];
                    ul_helper[simd_i] =
                        combined_u[f * u_face_offset + i - di + simd_i + offset];
                }
                // Copy from tmp helpers into working buffers
                ur.copy_from(ur_helper.data(),
                    SIMD_NAMESPACE::element_aligned_tag{});
                u0.copy_from(u0_helper.data(),
                    SIMD_NAMESPACE::element_aligned_tag{});
                ul.copy_from(ul_helper.data(),
                    SIMD_NAMESPACE::element_aligned_tag{});
            }

            simd_t qr(combined_q + start_index_f + q_i,
                SIMD_NAMESPACE::element_aligned_tag{});
            simd_t ql(combined_q + start_index_flipped + q_i,
                SIMD_NAMESPACE::element_aligned_tag{});

            const auto b0 = qr - ql;
            auto b = b0;
            if (q == 0) {
                // n m q Levi Civita 1 2 0 -> 1
                b += 12.0 *
                    simd_t(AM + 1 * am_offset + q_i,
                        SIMD_NAMESPACE::element_aligned_tag{}) *
                    1.0 * xloc[d][2] / (dx * (rho_l + rho_r));
                // n m q Levi Civita 2 1 0 -> -1
                b -= 12.0 *
                    simd_t(AM + 2 * am_offset + q_i,
                        SIMD_NAMESPACE::element_aligned_tag{}) *
                    1.0 * xloc[d][1] / (dx * (rho_l + rho_r));
            } else if (q == 1) {
                // n m q Levi Civita 0 2 1 -> -1
                b -= 12.0 *
                    simd_t(AM + 0 * am_offset + q_i,
                        SIMD_NAMESPACE::element_aligned_tag{}) *
                    1.0 * xloc[d][2] / (dx * (rho_l + rho_r));
                // n m q Levi Civita 2 0 1 -> 1
                b += 12.0 *
                    simd_t(AM + 2 * am_offset + q_i,
                        SIMD_NAMESPACE::element_aligned_tag{}) *
                    1.0 * xloc[d][0] / (dx * (rho_l + rho_r));
            } else {
                // n m q Levi Civita 0 1 2 -> 1
                b += 12.0 *
                    simd_t(AM + 0 * am_offset + q_i,
                        SIMD_NAMESPACE::element_aligned_tag{}) *
                    1.0 * xloc[d][1] / (dx * (rho_l + rho_r));
                // n m q Levi Civita 1 0 2 -> -1
                b -= 12.0 *
                    simd_t(AM + 1 * am_offset + q_i,
                        SIMD_NAMESPACE::element_aligned_tag{}) *
                    1.0 * xloc[d][0] / (dx * (rho_l + rho_r));
            }

            const simd_mask_t blim_mask = simd_t(0.0) < ((ur - u0) * (u0 - ul));
            const simd_t blim = SIMD_NAMESPACE::choose(blim_mask, b0, simd_t(0.0));
            b = minmod_simd(blim, b);
            qr += 0.5 * (b - b0);
            ql -= 0.5 * (b - b0);

            // TODO should be possible to combine the masks and get rid of some
            // of the statements
            const simd_mask_t u_mask1 = (u0 < ur) && (ul < u0);
            const simd_mask_t u_mask2 = (ur < u0) && (u0 < ul);
            const simd_mask_t qr_larger_mask = (ur < qr) && u_mask1;
            const simd_mask_t ql_smaller_mask = (ql < ul) && u_mask1;
            ql = SIMD_NAMESPACE::choose(qr_larger_mask, ql - (qr - ur), ql);
            qr = SIMD_NAMESPACE::choose(qr_larger_mask, ur, qr);
            qr = SIMD_NAMESPACE::choose(ql_smaller_mask, qr - (ql - ul), qr);
            ql = SIMD_NAMESPACE::choose(ql_smaller_mask, ul, ql);
            const simd_mask_t qr_smaller_mask = (qr < ur) && u_mask2;
            const simd_mask_t ql_larger_mask = (ul < ql) && u_mask2;
            ql = SIMD_NAMESPACE::choose(qr_smaller_mask, ql - (qr - ur), ql);
            qr = SIMD_NAMESPACE::choose(qr_smaller_mask, ur, qr);
            qr = SIMD_NAMESPACE::choose(ql_larger_mask, qr - (ql - ul), qr);
            ql = SIMD_NAMESPACE::choose(ql_larger_mask, ul, ql);

            make_monotone_simd<simd_t, simd_mask_t>(qr, u0, ql);
            // Write back results
            qr.copy_to(combined_q + start_index_f + q_i,
                SIMD_NAMESPACE::element_aligned_tag{});
            ql.copy_to(combined_q + start_index_flipped + q_i,
                SIMD_NAMESPACE::element_aligned_tag{});
        }
    }

    // Phase 3 - post-reconstruct
    const int start_index_rho = rho_i * q_face_offset + d * q_dir_offset;
    if (d != ndir / 2) {
        const int start_index_sx = sx_i * q_face_offset + d * q_dir_offset;
        const int start_index_sy = sy_i * q_face_offset + d * q_dir_offset;

        simd_t q_sx(combined_q + start_index_sx + q_i,
            SIMD_NAMESPACE::element_aligned_tag{});
        simd_t q_sy(combined_q + start_index_sy + q_i,
            SIMD_NAMESPACE::element_aligned_tag{});
        q_sx = 
            q_sx -
                omega *
                    (simd_t(combined_x + 1 * q_inx3 + q_i,
                         SIMD_NAMESPACE::element_aligned_tag{}) +
                        0.5 * xloc[d][1] * dx);
        q_sy = 
            q_sy +
                omega *
                    (simd_t(combined_x + q_i,
                         SIMD_NAMESPACE::element_aligned_tag{}) +
                        0.5 * xloc[d][0] * dx);
        q_sx.copy_to(combined_q + start_index_sx + q_i,
            SIMD_NAMESPACE::element_aligned_tag{});
        q_sy.copy_to(combined_q + start_index_sy + q_i,
            SIMD_NAMESPACE::element_aligned_tag{});

        const simd_t rho(combined_q + start_index_rho + q_i,
            SIMD_NAMESPACE::element_aligned_tag{});

        // n m q Levi Civita
        // 0 1 2 -> 1
        const double xloc_tmp1 = 0.5 * xloc[d][1] * dx;
        const simd_t q_lx_val0(combined_q + (lx_i + 0) * q_face_offset + d * q_dir_offset +
                q_i,
            SIMD_NAMESPACE::element_aligned_tag{});
        simd_t result0 = q_lx_val0 +
            (1.0) *
                (simd_t(combined_x + q_inx3 + q_i,
                     SIMD_NAMESPACE::element_aligned_tag{}) +
                    xloc_tmp1) *
                simd_t(combined_q + (sx_i + 2) * q_face_offset + d * q_dir_offset + q_i,
                    SIMD_NAMESPACE::element_aligned_tag{});

        // n m q Levi Civita
        // 0 2 1 -> -1
        const double xloc_tmp2 = 0.5 * xloc[d][2] * dx;
        result0 = result0 +
            (-1.0) *
                (simd_t(combined_x + 2 * q_inx3 + q_i,
                     SIMD_NAMESPACE::element_aligned_tag{}) +
                    xloc_tmp2) *
                simd_t(combined_q + (sx_i + 1) * q_face_offset + d * q_dir_offset + q_i,
                    SIMD_NAMESPACE::element_aligned_tag{});
        const simd_t q_lx_result = result0;
        q_lx_result.copy_to(combined_q + (lx_i + 0) * q_face_offset + d * q_dir_offset +
                q_i,
            SIMD_NAMESPACE::element_aligned_tag{});

        // n m q Levi Civita
        // 1 0 2 -> -1
        const double xloc_tmp0 = 0.5 * xloc[d][0] * dx;
        const simd_t q_lx_val1(combined_q + (lx_i + 1) * q_face_offset + d * q_dir_offset +
                q_i,
            SIMD_NAMESPACE::element_aligned_tag{});
        simd_t result1 = q_lx_val1 +
            (-1.0) *
                (simd_t(combined_x + q_i,
                     SIMD_NAMESPACE::element_aligned_tag{}) +
                    xloc_tmp0) *
                simd_t(combined_q + (sx_i + 2) * q_face_offset + d * q_dir_offset + q_i,
                    SIMD_NAMESPACE::element_aligned_tag{});

        // n m q Levi Civita
        // 1 2 0 -> 1
        result1 = result1 +
            (1.0) *
                (simd_t(combined_x + 2 * q_inx3 + q_i,
                     SIMD_NAMESPACE::element_aligned_tag{}) +
                    xloc_tmp2) *
                simd_t(combined_q + (sx_i + 0) * q_face_offset + d * q_dir_offset + q_i,
                    SIMD_NAMESPACE::element_aligned_tag{});
        const simd_t q_ly_result = result1;
        q_ly_result.copy_to(combined_q + (lx_i + 1) * q_face_offset + d * q_dir_offset +
                q_i,
            SIMD_NAMESPACE::element_aligned_tag{});

        // n m q Levi Civita
        // 2 0 1 -> 1
        const simd_t q_lx_val2(combined_q + (lx_i + 2) * q_face_offset + d * q_dir_offset +
                q_i,
            SIMD_NAMESPACE::element_aligned_tag{});
        auto result2 = q_lx_val2 +
            (1.0) *
                (simd_t(combined_x + q_i,
                     SIMD_NAMESPACE::element_aligned_tag{}) +
                    xloc_tmp0) *
                simd_t(combined_q + (sx_i + 1) * q_face_offset + d * q_dir_offset + q_i,
                    SIMD_NAMESPACE::element_aligned_tag{});

        // n m q Levi Civita
        // 2 1 0 -> -1
        result2 = result2 + (-1.0) *
            (simd_t(combined_x + q_inx3 + q_i,
                 SIMD_NAMESPACE::element_aligned_tag{}) +
                xloc_tmp1) *
            simd_t(combined_q + (sx_i + 0) * q_face_offset + d * q_dir_offset + q_i,
                SIMD_NAMESPACE::element_aligned_tag{});
        const simd_t q_lz_result = result2;
        q_lz_result.copy_to(combined_q + (lx_i + 2) * q_face_offset + d * q_dir_offset +
                q_i,
            SIMD_NAMESPACE::element_aligned_tag{});

        const int start_index_egas = egas_i * q_face_offset + d * q_dir_offset;
        const int start_index_pot = pot_i * q_face_offset + d * q_dir_offset;
        for (int n = 0; n < nangmom; n++) {
            const int start_index_lx_n = (lx_i + n) * q_face_offset + d * q_dir_offset;
            simd_t current_lx_n(combined_q + start_index_lx_n + q_i,
                SIMD_NAMESPACE::element_aligned_tag{});
            current_lx_n = current_lx_n * rho;
            current_lx_n.copy_to(combined_q + start_index_lx_n + q_i,
                SIMD_NAMESPACE::element_aligned_tag{});
        }

        for (int dim = 0; dim < NDIM; dim++) {
            const int start_index_sx_d = (sx_i + dim) * q_face_offset + d * q_dir_offset;
            simd_t v(combined_q + start_index_sx_d + q_i,
                SIMD_NAMESPACE::element_aligned_tag{});
            const simd_t current_egas(combined_q + start_index_egas + q_i,
                SIMD_NAMESPACE::element_aligned_tag{});
            simd_t egas_result =
                current_egas + 0.5 * v * v * rho;
            egas_result.copy_to(combined_q + start_index_egas + q_i,
                SIMD_NAMESPACE::element_aligned_tag{});
            v = v * rho;
            v.copy_to(combined_q + start_index_sx_d + q_i,
                SIMD_NAMESPACE::element_aligned_tag{});
        }

        simd_t pot_result(combined_q + start_index_pot + q_i,
            SIMD_NAMESPACE::element_aligned_tag{});
        pot_result = pot_result * rho;
        pot_result.copy_to(combined_q + start_index_pot + q_i,
            SIMD_NAMESPACE::element_aligned_tag{});
        simd_t w = 0.0;
        for (int si = 0; si < n_species_; si++) {
            const int start_index_sp_i = (spc_i + si) * q_face_offset + d * q_dir_offset;
            simd_t current_sp_field(combined_q + start_index_sp_i + q_i,
                SIMD_NAMESPACE::element_aligned_tag{});
            w = w + current_sp_field;
            current_sp_field =
                current_sp_field * rho;
            current_sp_field.copy_to(combined_q + start_index_sp_i + q_i,
                SIMD_NAMESPACE::element_aligned_tag{});
        }
        w = 1.0 / w;
        for (int si = 0; si < n_species_; si++) {
            const int start_index_sp_i = (spc_i + si) * q_face_offset + d * q_dir_offset;
            simd_t current_sp_field(combined_q + start_index_sp_i + q_i,
                SIMD_NAMESPACE::element_aligned_tag{});
            current_sp_field =
                current_sp_field * w;
            current_sp_field.copy_to(combined_q + start_index_sp_i + q_i,
                SIMD_NAMESPACE::element_aligned_tag{});
        }
    }
}
