#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp"

__device__ const int number_faces = 15;
__device__ const int number_dirs = 27;
__device__ const int q_inx = INX + 2;
__device__ const int q_inx3 = q_inx * q_inx * q_inx;
__device__ const int q_face_offset = number_dirs * q_inx3;
__device__ const int u_face_offset = H_N3;
__device__ const int q_dir_offset = q_inx3;
__device__ const int am_offset = q_inx3;

__device__ inline int to_q_index(const int j, const int k, const int l) {
    return j * q_inx * q_inx + k * q_inx + l;
}

__device__ const int HR_DNX = H_NX * H_NX;
__device__ const int HR_DNY = H_NX;
__device__ const int HR_DNZ = 1;
__device__ const int HR_DN0 = 0;
__device__ const int NDIR = 27;
__device__ const int disc_offset = H_NX * H_NX * H_NX;

__device__ const safe_real vw[27] = {
    /**/ 1. / 216., 4. / 216., 1. / 216., 4. / 216., 16. / 216., 4. / 216., 1. / 216., 4. / 216.,
    1. / 216.,
    /****/ 4. / 216., 16. / 216., 4. / 216., 16. / 216., 64. / 216., 16. / 216., 4. / 216.,
    16. / 216., 4. / 216.,
    /****/ 1. / 216., 4. / 216., 1. / 216., 4. / 216., 16. / 216., 4. / 216., 1. / 216., 4. / 216.,
    1. / 216.};

__device__ const int xloc[27][3] = {
    /**/ {-1, -1, -1}, {+0, -1, -1}, {+1, -1, -1},
    /**/ {-1, +0, -1}, {+0, +0, -1}, {1, +0, -1},
    /**/ {-1, +1, -1}, {+0, +1, -1}, {+1, +1, -1},
    /**/ {-1, -1, +0}, {+0, -1, +0}, {+1, -1, +0},
    /**/ {-1, +0, +0}, {+0, +0, +0}, {+1, +0, +0},
    /**/ {-1, +1, +0}, {+0, +1, +0}, {+1, +1, +0},
    /**/ {-1, -1, +1}, {+0, -1, +1}, {+1, -1, +1},
    /**/ {-1, +0, +1}, {+0, +0, +1}, {+1, +0, +1},
    /**/ {-1, +1, +1}, {+0, +1, +1}, {+1, +1, +1}};
__device__ const int dir[27] = {
/**/-HR_DNX - HR_DNY - HR_DNZ, +HR_DN0 - HR_DNY - HR_DNZ, +HR_DNX - HR_DNY - HR_DNZ,/**/
/**/-HR_DNX + HR_DN0 - HR_DNZ, +HR_DN0 + HR_DN0 - HR_DNZ, +HR_DNX + HR_DN0 - HR_DNZ,/**/
/**/-HR_DNX + HR_DNY - HR_DNZ, +HR_DN0 + HR_DNY - HR_DNZ, +HR_DNX + HR_DNY - HR_DNZ,/**/
/**/-HR_DNX - HR_DNY + HR_DN0, +HR_DN0 - HR_DNY + HR_DN0, +HR_DNX - HR_DNY + HR_DN0,/**/
/**/-HR_DNX + HR_DN0 + HR_DN0, +HR_DN0 + HR_DN0 + HR_DN0, +HR_DNX + HR_DN0 + HR_DN0,/**/
/**/-HR_DNX + HR_DNY + HR_DN0, +HR_DN0 + HR_DNY + HR_DN0, +HR_DNX + HR_DNY + HR_DN0,/**/
/**/-HR_DNX - HR_DNY + HR_DNZ, +HR_DN0 - HR_DNY + HR_DNZ, +HR_DNX - HR_DNY + HR_DNZ,/**/
/**/-HR_DNX + HR_DN0 + HR_DNZ, +HR_DN0 + HR_DN0 + HR_DNZ, +HR_DNX + HR_DN0 + HR_DNZ,/**/
/**/-HR_DNX + HR_DNY + HR_DNZ, +HR_DN0 + HR_DNY + HR_DNZ, +HR_DNX + HR_DNY + HR_DNZ/**/

};
__device__ inline int flip(const int d) {
  return NDIR - 1 - d;
}

__device__ inline void make_monotone(double &ql, double q0, double &qr) {
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


__device__ inline double minmod_cuda(double a, double b) {
	return (copysign(0.5, a) + copysign(0.5, b)) * min(std::abs(a), abs(b));
}

__device__ inline double minmod_cuda_theta(double a, double b, double c) {
	return minmod_cuda(c * minmod_cuda(a, b), 0.5 * (a + b));
}
// template <>
// inline void store_value<vc_type>(const double* __restrict__ data, const size_t index, const
// vc_type &value) {
//    return value.store(data + index);
//}

__device__ inline void reconstruct_minmod_cuda(double* __restrict__ combined_q,
    const double* __restrict__ combined_u_face, int d, int f, int i, int q_i) {
    const auto di = dir[d];
    const int start_index = f * q_face_offset + d * q_dir_offset;
    combined_q[q_i + start_index] = combined_u_face[i] +
        0.5 *
            minmod_cuda(combined_u_face[i + di] - combined_u_face[i],
                combined_u_face[i] - combined_u_face[i - di]);
}

__device__ inline void reconstruct_ppm_experimental(double* __restrict__ combined_q,
    const double* __restrict__ combined_u_face, bool smooth, bool disc_detect,
    const double* __restrict__ disc, const int d, const int f, int i, int q_i) {
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
    const double d1 = minmod_theta_wrapper<double>(diff_u_plus, diff_u_minus, 2.0);
    const double d1_plus = minmod_theta_wrapper<double>(diff_u_2plus, diff_u_plus, 2.0);
    const double d1_minus = minmod_theta_wrapper<double>(diff_u_minus, diff_u_2minus, 2.0);

    double results = 0.5 * (u_zero + u_plus_di) + (1.0 / 6.0) * (d1 - d1_plus);
    double results_flipped = 0.5 * (u_minus_di + u_zero) + (1.0 / 6.0) * (d1_minus - d1);

    combined_q[start_index + q_i] = results;
    ;
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
__device__ inline void reconstruct_inner_loop_p1(const size_t nf_, const int angmom_index_,
    const int* __restrict__ smooth_field_, const int* __restrict__ disc_detect_ ,
    double* __restrict__ combined_q, const double* __restrict__ combined_u, double* __restrict__ AM,
    const double dx, const double* __restrict__ cdiscs, const int d, const int i,
    const int q_i, const int ndir, const int nangmom) {
    const int sx_i = angmom_index_;
    const int zx_i = sx_i + NDIM;
    if (d < ndir / 2) {
        for (int f = 0; f < angmom_index_; f++) {
            reconstruct_ppm_experimental(combined_q, combined_u + u_face_offset * f,
                smooth_field_[f], disc_detect_[f], cdiscs, d, f, i, q_i);
        }
        for (int f = sx_i; f < sx_i + NDIM; f++) {
            reconstruct_ppm_experimental(
                combined_q, combined_u + u_face_offset * f, true, false, cdiscs, d, f, i, q_i);
        }
    }
    for (int f = zx_i; f < zx_i + nangmom; f++) {
        reconstruct_minmod_cuda(combined_q, combined_u + u_face_offset * f, d, f, i, q_i);
    }
    if (d < ndir / 2) {
        for (int f = angmom_index_ + nangmom + NDIM; f < nf_; f++) {
            reconstruct_ppm_experimental(combined_q, combined_u + u_face_offset * f,
                smooth_field_[f], disc_detect_[f], cdiscs, d, f, i, q_i);
        }
    }

    if (d != ndir / 2) {
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

__device__ inline void reconstruct_inner_loop_p2(const safe_real omega, double* __restrict__ combined_q,
    double* __restrict__ combined_x, double* __restrict__ combined_u, double* __restrict__ AM,
    const double dx, const int d, const int i, const int q_i, const int ndir, const int nangmom, const int n_species_) {
    if (d < ndir / 2) {
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

    // Phase 3
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

__global__ void
__launch_bounds__(128, 1)
reconstruct_cuda_kernel(const double omega, const int nf_, const int angmom_index_,
    int* __restrict__ smooth_field_, int* __restrict__ disc_detect_ ,
    double* __restrict__ combined_q, double* __restrict__ combined_x,
    double* __restrict__ combined_u, double* __restrict__ AM, const double dx,
    const double* __restrict__ cdiscs, const int n_species_, const int ndir, const int nangmom) {
  const int sx_i = angmom_index_;
  const int zx_i = sx_i + NDIM;

  const int q_i = (blockIdx.z * 2 + threadIdx.x) * 64 + (threadIdx.y) * 8 + (threadIdx.z);
  const int i = ((q_i / 100) + 2) * 14 * 14 + (((q_i % 100) / 10 ) + 2) * 14 + (((q_i % 100) % 10) + 2);
  if (q_i < 1000) {
    for (int n = 0; n < nangmom; n++) {
      AM[n * am_offset + q_i] =
      combined_u[(zx_i + n) * u_face_offset + i] * combined_u[i];
    }
    for (int d = 0; d < ndir; d++) {
      reconstruct_inner_loop_p1(nf_, angmom_index_, smooth_field_, disc_detect_,
      combined_q, combined_u, AM, dx, cdiscs, d, i, q_i, ndir, nangmom);
    }
    // Phase 2
    for (int d = 0; d < ndir; d++) {
      reconstruct_inner_loop_p2(omega, combined_q, combined_x, combined_u, AM, dx, d, i, q_i, ndir, nangmom, n_species_);
    }
  }
}


void launch_reconstruct_cuda(
    stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
    double omega, int nf_, int angmom_index_,
    int* smooth_field_, int* disc_detect_ ,
    double* combined_q, double* combined_x,
    double* combined_u, double* AM, double dx,
    double* cdiscs, int n_species_) {
    static const cell_geometry<NDIM, INX> geo;

    // Current implementation limitations of this kernel - can be resolved but that takes more work
    assert(angmom_index_ > -1);
    assert(NDIM > 2);
    assert(nf_ == 15);
    assert(geo.NDIR == 27);
    assert(INX == 8);

    // TODO Set parameters
    dim3 const grid_spec(1, 1, 8);
    dim3 const threads_per_block(2, 8, 8);
    int ndir = geo.NDIR;
    int nangmom = geo.NANGMOM;
    void* args[] = {&omega, &nf_, &angmom_index_, &(smooth_field_), &(disc_detect_), &(combined_q),
      &(combined_x), &(combined_u), &(AM), &dx, &(cdiscs), &n_species_, &ndir, &nangmom};
    // TODO Launch kernel
    executor.post(
    cudaLaunchKernel<decltype(reconstruct_cuda_kernel)>,
    reconstruct_cuda_kernel, grid_spec, threads_per_block, args, 0);


    // TODO Move q back (for now...)
}

