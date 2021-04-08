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