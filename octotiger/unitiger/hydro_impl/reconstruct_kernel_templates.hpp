#pragma once
#if defined(__clang__)
constexpr int number_dirs = 27;
constexpr int q_inx = INX + 2;
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
CUDA_GLOBAL_METHOD const int number_dirs = 27;
CUDA_GLOBAL_METHOD const int q_inx = INX + 2;
CUDA_GLOBAL_METHOD const int q_inx3 = q_inx * q_inx * q_inx;
CUDA_GLOBAL_METHOD const int q_face_offset = number_dirs * q_inx3;
CUDA_GLOBAL_METHOD const int u_face_offset = H_N3;
CUDA_GLOBAL_METHOD const int x_offset = H_N3;
CUDA_GLOBAL_METHOD const int q_dir_offset = q_inx3;
CUDA_GLOBAL_METHOD const int am_offset = q_inx3;
CUDA_GLOBAL_METHOD const int HR_DNX = H_NX * H_NX;
CUDA_GLOBAL_METHOD const int HR_DNY = H_NX;
CUDA_GLOBAL_METHOD const int HR_DNZ = 1;
CUDA_GLOBAL_METHOD const int HR_DN0 = 0;
//CUDA_GLOBAL_METHOD const int NDIR = 27;
CUDA_GLOBAL_METHOD const int disc_offset = H_NX * H_NX * H_NX;
CUDA_GLOBAL_METHOD const int dir[27] = {
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
CUDA_GLOBAL_METHOD const safe_real vw[27] = {
    /**/ 1. / 216., 4. / 216., 1. / 216., 4. / 216., 16. / 216., 4. / 216., 1. / 216., 4. / 216.,
    1. / 216.,
    /****/ 4. / 216., 16. / 216., 4. / 216., 16. / 216., 64. / 216., 16. / 216., 4. / 216.,
    16. / 216., 4. / 216.,
    /****/ 1. / 216., 4. / 216., 1. / 216., 4. / 216., 16. / 216., 4. / 216., 1. / 216., 4. / 216.,
    1. / 216.};

CUDA_GLOBAL_METHOD const int xloc[27][3] = {
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

// TODO Replace those wrappers
template <typename T>
CUDA_GLOBAL_METHOD inline T copysign_wrapper_cuda(const T& tmp1, const T& tmp2) {
    return std::copysign(tmp1, tmp2);
}
template <typename T>
CUDA_GLOBAL_METHOD inline T abs_wrapper_cuda(const T& tmp1) {
    return std::abs(tmp1);
}
template <typename T>
CUDA_GLOBAL_METHOD inline T minmod_wrapper_cuda(const T& a, const T& b) {
    return (copysign_wrapper_cuda<T>(0.5, a) + copysign_wrapper_cuda<T>(0.5, b)) *
        min(abs_wrapper_cuda<T>(a), abs_wrapper_cuda<T>(b));
}
template <typename T>
CUDA_GLOBAL_METHOD inline T minmod_theta_wrapper_cuda(const T& a, const T& b, const T& c) {
    return minmod_wrapper_cuda<T>(c * minmod_wrapper_cuda<T>(a, b), 0.5 * (a + b));
}

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
CUDA_GLOBAL_METHOD inline void make_monotone(double& ql, double q0, double& qr) {
    const double tmp1 = qr - ql;
    const double tmp2 = qr + ql;

    if (bool(qr < q0) != bool(q0 < ql)) {
        qr = ql = q0;
        return;
    }
    const double tmp3 = tmp1 * tmp1 / 6.0;
    const double tmp4 = tmp1 * (q0 - 0.5 * tmp2);
    const double eps = 1.0e-12;
    if (tmp4 > tmp3) {
        ql = (3.0 * q0 - 2.0 * qr);
    } else if (-tmp3 > tmp4) {
        qr = (3.0 * q0 - 2.0 * ql);
    }
}
CUDA_GLOBAL_METHOD inline double minmod_cuda(double a, double b) {
    return (copysign(0.5, a) + copysign(0.5, b)) * std::min(std::abs(a), abs(b));
}
CUDA_GLOBAL_METHOD inline double minmod_cuda_theta(double a, double b, double c) {
    return minmod_cuda(c * minmod_cuda(a, b), 0.5 * (a + b));
}
template <typename container_t, typename const_container_t>
CUDA_GLOBAL_METHOD inline void cell_reconstruct_minmod_cuda(container_t &combined_q,
    const_container_t &combined_u_face, int d, int f, int i, int q_i) {
    const auto di = dir[d];
    const int start_index = f * q_face_offset + d * q_dir_offset;
    combined_q[q_i + start_index] = combined_u_face[i] +
        0.5 *
            minmod_cuda(combined_u_face[i + di] - combined_u_face[i],
                combined_u_face[i] - combined_u_face[i - di]);
}

template <typename container_t, typename const_container_t>
CUDA_GLOBAL_METHOD inline void cell_find_contact_discs_phase1(container_t &P,
    const_container_t &combined_u, const double A_, const double B_, const double fgamma_,
    const double de_switch_1, const unsigned int x, const unsigned int y,
    const unsigned int z) {
    const int i = (x + 1) * 14 * 14 + (y + 1) * 14 + (z + 1);

    const auto rho = combined_u[rho_i * u_face_offset + i];
    const auto rhoinv = 1.0 / rho;
    double hdeg = 0.0, pdeg = 0.0, edeg = 0.0;

    if (A_ != 0.0) {
        const auto x = std::pow(rho / B_, 1.0 / 3.0);
        hdeg = 8.0 * A_ / B_ * (std::sqrt(x * x + 1.0) - 1.0);
        pdeg = deg_pres(x, A_);
        edeg = rho * hdeg - pdeg;
    }

    safe_real ek = 0.0;
    for (int dim = 0; dim < NDIM; dim++) {
        ek += combined_u[(sx_i + dim) * u_face_offset + i] *
            combined_u[(sx_i + dim) * u_face_offset + i] * rhoinv * 0.5;
    }

    auto ein = combined_u[egas_i * u_face_offset + i] - ek - edeg;
    if (ein < de_switch_1 * combined_u[egas_i * u_face_offset + i]) {
        ein = pow(combined_u[tau_i * u_face_offset + i], fgamma_);
    }
    P[i] = (fgamma_ - 1.0) * ein + pdeg;
}

template <typename container_t, typename const_container_t>
CUDA_GLOBAL_METHOD inline void cell_find_contact_discs_phase2(
    container_t &disc, const_container_t &P, const double fgamma_,
    const int ndir, const unsigned int x, const unsigned int y, const unsigned int z) {
    const int disc_offset = 14 * 14 * 14;
    const double K0 = 0.1;
    const int i = (x + 2) * 14 * 14 + (y + 2) * 14 + (z + 2);
    for (int d = 0; d < ndir / 2; d++) {
        const auto di = dir[d];
        const double P_r = P[i + di];
        const double P_l = P[i - di];
        const double tmp1 = fgamma_ * K0;
        const double tmp2 = abs(P_r - P_l) / std::min(std::abs(P_r), abs(P_l));
        disc[d * disc_offset + i] = tmp2 / tmp1;
    }
}

template <typename container_t, typename const_container_t>
CUDA_GLOBAL_METHOD inline void cell_hydro_pre_recon(const_container_t& X, safe_real omega, bool angmom,
        container_t &u, const int nf, const int n_species_, const unsigned int x, const unsigned int y, const unsigned int z) {
    const int i = (x) * 14 * 14 + (y) * 14 + (z);
    const auto rho = u[rho_i * u_face_offset + i];
    const auto rhoinv = 1.0 / rho;
    for (int dim = 0; dim < NDIM; dim++) {
        auto& s = u[(sx_i + dim) * u_face_offset + i];
        u[egas_i * u_face_offset + i] -= 0.5 * s * s * rhoinv;
        s *= rhoinv;
    }
    for (int si = 0; si < n_species_; si++) {
        u[(spc_i + si) * u_face_offset + i] *= rhoinv;
    }
    u[pot_i * u_face_offset + i] *= rhoinv;

    u[(lx_i + 0) * u_face_offset + i] *= rhoinv;
    u[(lx_i + 1) * u_face_offset + i] *= rhoinv;
    u[(lx_i + 2) * u_face_offset + i] *= rhoinv;

    // Levi civita n m q -> lc
    // Levi civita 0 1 2 -> 1
    u[(lx_i + 0) * u_face_offset + i] -=
        1.0 * X[1 * x_offset + i] * u[(sx_i + 2) * u_face_offset + i];
    // Levi civita n m q -> lc
    // Levi civita 0 2 1 -> -1
    u[(lx_i + 0) * u_face_offset + i] -=
        -1.0 * X[2 * x_offset + i] * u[(sx_i + 1) * u_face_offset + i];
    // Levi civita n m q -> lc
    // Levi civita 1 0 2 -> -1
    u[(lx_i + 1) * u_face_offset + i] -=
        -1.0 * X[i] * u[(sx_i + 2) * u_face_offset + i];
    // Levi civita n m q -> lc
    // Levi civita 1 2 0 -> 1
    u[(lx_i + 1) * u_face_offset + i] -=
        1.0 * X[2 * x_offset + i] * u[(sx_i + 0) * u_face_offset + i];
    // Levi civita n m q -> lc
    // Levi civita 2 0 1 -> 1
    u[(lx_i + 2) * u_face_offset + i] -=
        1.0 * X[i] * u[(sx_i + 1) * u_face_offset + i];
    // Levi civita n m q -> lc
    // Levi civita 2 1 0 -> -1
    u[(lx_i + 2) * u_face_offset + i] -=
        -1.0 * X[1 * x_offset + i] * u[(sx_i + 0) * u_face_offset + i];

    u[sx_i * u_face_offset + i] += omega * X[1 * x_offset + i];
    u[sy_i * u_face_offset + i] -= omega * X[0 * x_offset + i];
}

template <typename container_t, typename const_container_t>
CUDA_GLOBAL_METHOD inline void cell_reconstruct_ppm(container_t &combined_q,
    const const_container_t &combined_u_face, bool smooth, bool disc_detect,
    const_container_t &disc, const int d, const int f, int i, int q_i) {
    // const vc_type zindices = vc_type::IndexesFromZero() + 1;
    // static thread_local auto D1 = std::vector<safe_real>(geo.H_N3, 0.0);
    const auto di = dir[d];
    const auto flipped_di = flip(d);

    const int start_index = f * q_face_offset + d * q_dir_offset;
    const int start_index_flipped = f * q_face_offset + flipped_di * q_dir_offset;

    const double u_plus_2di = combined_u_face[i + 2 * di];
    const double u_plus_di = combined_u_face[i + di];
    const double u_zero = combined_u_face[i];
    const double u_minus_di = combined_u_face[i - di];
    const double u_minus_2di = combined_u_face[i - 2 * di];

    const double diff_u_plus = u_plus_di - u_zero;
    const double diff_u_2plus = u_plus_2di - u_plus_di;

    const double diff_u_minus = u_zero - u_minus_di;
    const double diff_u_2minus = u_minus_di - u_minus_2di;
    const double d1 = minmod_theta_wrapper_cuda<double>(diff_u_plus, diff_u_minus, 2.0);
    const double d1_plus = minmod_theta_wrapper_cuda<double>(diff_u_2plus, diff_u_plus, 2.0);
    const double d1_minus = minmod_theta_wrapper_cuda<double>(diff_u_minus, diff_u_2minus, 2.0);

    double results = 0.5 * (u_zero + u_plus_di) + (1.0 / 6.0) * (d1 - d1_plus);
    double results_flipped = 0.5 * (u_minus_di + u_zero) + (1.0 / 6.0) * (d1_minus - d1);

    combined_q[start_index + q_i] = results;
    combined_q[start_index_flipped + q_i] = results_flipped;

    if (disc_detect) {
        constexpr auto eps = 0.01;
        constexpr auto eps2 = 0.001;
        constexpr auto eta1 = 20.0;
        constexpr auto eta2 = 0.05;
        const auto di = dir[d];
        const auto& up = combined_u_face[i + di];
        const auto& u0 = combined_u_face[i];
        const auto& um = combined_u_face[i - di];
        const auto dif = up - um;
        if (std::abs(dif) > disc[d * disc_offset + i] * std::min(std::abs(up), std::abs(um))) {
            if (std::min(std::abs(up), std::abs(um)) / std::max(std::abs(up), std::abs(um)) >
                eps2) {
                const auto d2p = (1.0 / 6.0) *
                    (combined_u_face[i + 2 * di] + u0 - 2.0 * combined_u_face[i + di]);
                const auto d2m = (1.0 / 6.0) *
                    (u0 + combined_u_face[i - 2 * di] - 2.0 * combined_u_face[i - di]);
                if (d2p * d2m < 0.0) {
                    double eta = 0.0;
                    if (std::abs(dif) > eps * std::min(std::abs(up), std::abs(um))) {
                        eta = -(d2p - d2m) / dif;
                    }
                    eta = std::max(0.0, std::min(eta1 * (eta - eta2), 1.0));
                    if (eta > 0.0) {
                        auto ul = um +
                            0.5 *
                                minmod_cuda_theta(
                                    combined_u_face[i] - um, um - combined_u_face[i - 2 * di], 2.0);
                        auto ur = up -
                            0.5 *
                                minmod_cuda_theta(
                                    combined_u_face[i + 2 * di] - up, up - combined_u_face[i], 2.0);
                        // auto& qp = q[d][i];
                        // auto& qm = q[geo.flip(d)][i];
                        auto& qp = combined_q[start_index + q_i];
                        auto& qm = combined_q[start_index_flipped + q_i];
                        qp += eta * (ur - qp);
                        qm += eta * (ul - qm);
                    }
                }
            }
        }
    }
    if (!smooth) {
        make_monotone(combined_q[start_index + q_i], combined_u_face[i],
            combined_q[start_index_flipped + q_i]);
        // auto& qp = q[geo.flip(d)][i];
        // auto& qm = q[d][i];
        // make_monotone(qm, combined_u_face[i], qp);
    }
}

// Phase 1 and 2
template <typename container_t, typename const_container_t, typename const_int_container_t>
CUDA_GLOBAL_METHOD inline void cell_reconstruct_inner_loop_p1(const size_t nf_, const int angmom_index_,
    const_int_container_t& smooth_field_, const_int_container_t& disc_detect_,
    container_t& combined_q, container_t& combined_u, container_t& AM,
    const double dx, const_container_t &cdiscs, const int d, const int i, const int q_i,
    const int ndir, const int nangmom) {
    int l_start;
    int s_start;
    if (angmom_index_ > -1) {
        s_start = angmom_index_;
        l_start = angmom_index_ + NDIM;
    } else {
        s_start = lx_i;
        l_start = lx_i;
    }
    if (d < ndir / 2) {
        for (int f = 0; f < s_start; f++) {
            // const double * __restrict__ combined_u_face = combined_u + u_face_offset * f;
            const const_container_t combined_u_face = combined_u + u_face_offset * f;
            cell_reconstruct_ppm(combined_q, combined_u_face,
                smooth_field_[f], disc_detect_[f], cdiscs, d, f, i, q_i);
        }
        for (int f = s_start; f < l_start; f++) {
            // const double * __restrict__ combined_u_face = combined_u + u_face_offset * f;
            const const_container_t combined_u_face = combined_u + u_face_offset * f;
            cell_reconstruct_ppm(
                combined_q, combined_u_face, true, false, cdiscs, d, f, i, q_i);
        }
    }
    for (int f = l_start; f < l_start + nangmom; f++) {
        // const double * __restrict__ combined_u_face = combined_u + u_face_offset * f;
        const const_container_t combined_u_face = combined_u + u_face_offset * f;
        cell_reconstruct_minmod_cuda(combined_q, combined_u_face, d, f, i, q_i);
    }
    if (d < ndir / 2) {
        for (int f = l_start + nangmom; f < nf_; f++) {
            // const double * __restrict__ combined_u_face = combined_u + u_face_offset * f;
            const const_container_t combined_u_face = combined_u + u_face_offset * f;
            cell_reconstruct_ppm(combined_q, combined_u_face,
                smooth_field_[f], disc_detect_[f], cdiscs, d, f, i, q_i);
        }
    }

    if (d != ndir / 2 && angmom_index_ > -1) {
        const int start_index_rho = d * q_dir_offset;

        // n m q Levi Civita
        // 0 1 2 -> 1
        double results0 = AM[q_i] -
            vw[d] * 1.0 * 0.5 * xloc[d][1] *
                combined_q[(sx_i + 2) * q_face_offset + d * q_dir_offset + q_i] *
                combined_q[start_index_rho + q_i] * dx;

        // n m q Levi Civita
        // 0 2 1 -> -1
        results0 -= vw[d] * (-1.0) * 0.5 * xloc[d][2] *
            combined_q[(sx_i + 1) * q_face_offset + d * q_dir_offset + q_i] *
            combined_q[start_index_rho + q_i] * dx;
        AM[q_i] = results0;

        // n m q Levi Civita
        // 1 0 2 -> -1
        double results1 = AM[am_offset + q_i] -
            vw[d] * (-1.0) * 0.5 * xloc[d][0] *
                combined_q[(sx_i + 2) * q_face_offset + d * q_dir_offset + q_i] *
                combined_q[start_index_rho + q_i] * dx;

        // n m q Levi Civita
        // 1 2 0 -> 1
        results1 -= vw[d] * (1.0) * 0.5 * xloc[d][2] *
            combined_q[(sx_i + 0) * q_face_offset + d * q_dir_offset + q_i] *
            combined_q[start_index_rho + q_i] * dx;
        AM[am_offset + q_i] = results1;

        // n m q Levi Civita
        // 2 0 1 -> 1
        double results2 = AM[2 * am_offset + q_i] -
            vw[d] * (1.0) * 0.5 * xloc[d][0] *
                combined_q[(sx_i + 1) * q_face_offset + d * q_dir_offset + q_i] *
                combined_q[start_index_rho + q_i] * dx;

        // n m q Levi Civita
        // 2 1 0 -> -1
        results2 -= vw[d] * (-1.0) * 0.5 * xloc[d][1] *
            combined_q[(sx_i + 0) * q_face_offset + d * q_dir_offset + q_i] *
            combined_q[start_index_rho + q_i] * dx;
        AM[2 * am_offset + q_i] = results2;
    }
}

template <typename container_t, typename const_container_t>
CUDA_GLOBAL_METHOD inline void cell_reconstruct_inner_loop_p2(const safe_real omega, const int angmom_index_,
    container_t& combined_q, const_container_t& combined_x,
    container_t& combined_u, const_container_t& AM, const double dx, const int d,
    const int i, const int q_i, const int ndir, const int nangmom, const int n_species_) {
    if (d < ndir / 2 && angmom_index_ > -1) {
        const auto di = dir[d];

        for (int q = 0; q < NDIM; q++) {
            const auto f = sx_i + q;
            const int start_index_f = f * q_face_offset + d * q_dir_offset;
            const int start_index_flipped = f * q_face_offset + flip(d) * q_dir_offset;
            const int start_index_zero = 0 * q_face_offset + d * q_dir_offset;
            const int start_index_zero_flipped = 0 * q_face_offset + flip(d) * q_dir_offset;
            // const auto& rho_r = Q[0][d][i];
            // const auto& rho_l = Q[0][geo.flip(d)][i];
            const auto& rho_r = combined_q[start_index_zero + q_i];
            const auto& rho_l = combined_q[start_index_zero_flipped + q_i];
            auto& qr = combined_q[start_index_f + q_i];
            auto& ql = combined_q[start_index_flipped + q_i];
            // auto& qr = Q[f][d][i];
            // auto& ql = Q[f][geo.flip(d)][i];
            const auto& ur = combined_u[f * u_face_offset + i + di];
            const auto& u0 = combined_u[f * u_face_offset + i];
            const auto& ul = combined_u[f * u_face_offset + i - di];
            const auto b0 = qr - ql;
            auto b = b0;
            if (q == 0) {
                // n m q Levi Civita
                // 1 2 0 -> 1
                b += 12.0 * AM[1 * am_offset + q_i] * 1.0 * xloc[d][2] / (dx * (rho_l + rho_r));
                // n m q Levi Civita
                // 2 1 0 -> -1
                b -= 12.0 * AM[2 * am_offset + q_i] * 1.0 * xloc[d][1] / (dx * (rho_l + rho_r));
            } else if (q == 1) {
                // n m q Levi Civita
                // 0 2 1 -> -1
                b -= 12.0 * AM[0 * am_offset + q_i] * 1.0 * xloc[d][2] / (dx * (rho_l + rho_r));
                // n m q Levi Civita
                // 2 0 1 -> 1
                b += 12.0 * AM[2 * am_offset + q_i] * 1.0 * xloc[d][0] / (dx * (rho_l + rho_r));
            } else {
                // n m q Levi Civita
                // 0 1 2 -> 1
                b += 12.0 * AM[0 * am_offset + q_i] * 1.0 * xloc[d][1] / (dx * (rho_l + rho_r));
                // n m q Levi Civita
                // 1 0 2 -> -1
                b -= 12.0 * AM[1 * am_offset + q_i] * 1.0 * xloc[d][0] / (dx * (rho_l + rho_r));
            }
            double blim;
            if ((ur - u0) * (u0 - ul) <= 0.0) {
                blim = 0.0;
            } else {
                blim = b0;
            }
            b = minmod_cuda(blim, b);
            qr += 0.5 * (b - b0);
            ql -= 0.5 * (b - b0);
            if (ur > u0 && u0 > ul) {
                if (qr > ur) {
                    ql -= (qr - ur);
                    qr = ur;
                } else if (ql < ul) {
                    qr -= (ql - ul);
                    ql = ul;
                }
            } else if (ur < u0 && u0 < ul) {
                if (qr < ur) {
                    ql -= (qr - ur);
                    qr = ur;
                } else if (ql > ul) {
                    qr -= (ql - ul);
                    ql = ul;
                }
            }
            make_monotone(qr, u0, ql);
        }
    }

    // Phase 3 - post-reconstrut
    const int start_index_rho = rho_i * q_face_offset + d * q_dir_offset;
    if (d != ndir / 2) {
        const int start_index_sx = sx_i * q_face_offset + d * q_dir_offset;
        const int start_index_sy = sy_i * q_face_offset + d * q_dir_offset;
        combined_q[start_index_sx + q_i] -=
            omega * (combined_x[1 * q_inx3 + q_i] + 0.5 * xloc[d][1] * dx);
        combined_q[start_index_sy + q_i] += omega * (combined_x[q_i] + 0.5 * xloc[d][0] * dx);
        // Q[sx_i][d][i] -= omega * (X[1][i] + 0.5 * xloc[d][1] * dx);
        // Q[sy_i][d][i] += omega * (X[0][i] + 0.5 * xloc[d][0] * dx);
        const double rho = combined_q[start_index_rho + q_i];

        // n m q Levi Civita
        // 0 1 2 -> 1
        const double xloc_tmp1 = 0.5 * xloc[d][1] * dx;
        const double q_lx_val0 = combined_q[(lx_i + 0) * q_face_offset + d * q_dir_offset + q_i];
        double result0 = q_lx_val0 +
            (1.0) * (combined_x[q_inx3 + q_i] + xloc_tmp1) *
                combined_q[(sx_i + 2) * q_face_offset + d * q_dir_offset + q_i];

        // n m q Levi Civita
        // 0 2 1 -> -1
        const double xloc_tmp2 = 0.5 * xloc[d][2] * dx;
        result0 += (-1.0) * (combined_x[2 * q_inx3 + q_i] + xloc_tmp2) *
            combined_q[(sx_i + 1) * q_face_offset + d * q_dir_offset + q_i];
        combined_q[(lx_i + 0) * q_face_offset + d * q_dir_offset + q_i] = result0;

        // n m q Levi Civita
        // 1 0 2 -> -1
        const double xloc_tmp0 = 0.5 * xloc[d][0] * dx;
        const double q_lx_val1 = combined_q[(lx_i + 1) * q_face_offset + d * q_dir_offset + q_i];
        double result1 = q_lx_val1 +
            (-1.0) * (combined_x[q_i] + xloc_tmp0) *
                combined_q[(sx_i + 2) * q_face_offset + d * q_dir_offset + q_i];

        // n m q Levi Civita
        // 1 2 0 -> 1
        result1 += (1.0) * (combined_x[2 * q_inx3 + q_i] + xloc_tmp2) *
            combined_q[(sx_i + 0) * q_face_offset + d * q_dir_offset + q_i];
        combined_q[(lx_i + 1) * q_face_offset + d * q_dir_offset + q_i] = result1;

        // n m q Levi Civita
        // 2 0 1 -> 1
        const double q_lx_val2 = combined_q[(lx_i + 2) * q_face_offset + d * q_dir_offset + q_i];
        auto result2 = q_lx_val2 +
            (1.0) * (combined_x[q_i] + xloc_tmp0) *
                combined_q[(sx_i + 1) * q_face_offset + d * q_dir_offset + q_i];

        // n m q Levi Civita
        // 2 1 0 -> -1
        result2 += (-1.0) * (combined_x[q_inx3 + q_i] + xloc_tmp1) *
            combined_q[(sx_i + 0) * q_face_offset + d * q_dir_offset + q_i];
        combined_q[(lx_i + 2) * q_face_offset + d * q_dir_offset + q_i] = result2;
        const int start_index_egas = egas_i * q_face_offset + d * q_dir_offset;
        const int start_index_pot = pot_i * q_face_offset + d * q_dir_offset;
        for (int n = 0; n < nangmom; n++) {
            const int start_index_lx_n = (lx_i + n) * q_face_offset + d * q_dir_offset;
            combined_q[start_index_lx_n + q_i] *= rho;
        }
        for (int dim = 0; dim < NDIM; dim++) {
            const int start_index_sx_d = (sx_i + dim) * q_face_offset + d * q_dir_offset;
            auto& v = combined_q[start_index_sx_d + q_i];
            combined_q[start_index_egas + q_i] += 0.5 * v * v * rho;
            v *= rho;
        }
        combined_q[start_index_pot + q_i] *= rho;
        safe_real w = 0.0;
        for (int si = 0; si < n_species_; si++) {
            const int start_index_sp_i = (spc_i + si) * q_face_offset + d * q_dir_offset;
            w += combined_q[start_index_sp_i + q_i];
            combined_q[start_index_sp_i + q_i] *= rho;
        }
        w = 1.0 / w;
        for (int si = 0; si < n_species_; si++) {
            const int start_index_sp_i = (spc_i + si) * q_face_offset + d * q_dir_offset;
            combined_q[start_index_sp_i + q_i] *= w;
        }
    }
}