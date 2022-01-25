#include <hpx/config/compiler_specific.hpp> 
#ifndef HPX_COMPUTE_DEVICE_CODE

// Deprecated

// TODO(daissgr)
// Remove entire file once SIMD vectorization of reconstruct works in the reconstruct kokkos kernel

#ifdef __x86_64__ // currently only works on x86
#ifdef OCTOTIGER_HAVE_VC
#include "octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp"

#pragma GCC push_options
#pragma GCC optimize("unroll-loops")

#include <Vc/Vc>
#include <Vc/common/mask.h>
#include <Vc/vector.h>

using vc_type = Vc::Vector<double, Vc::VectorAbi::Avx>;
using mask_type = vc_type::mask_type;
using index_type = Vc::Vector<int, Vc::VectorAbi::Avx>;

template <>
CUDA_GLOBAL_METHOD inline vc_type copysign_wrapper<vc_type>(const vc_type& tmp1, const vc_type& tmp2) {
    return Vc::copysign(tmp1, tmp2);
}
template <>
CUDA_GLOBAL_METHOD inline vc_type abs_wrapper<vc_type>(const vc_type& tmp1) {
    return Vc::abs(tmp1);
}
template <>
CUDA_GLOBAL_METHOD inline vc_type minmod_wrapper<vc_type>(const vc_type& a, const vc_type& b) {
    return (copysign_wrapper<vc_type>(0.5, a) + copysign_wrapper<vc_type>(0.5, b)) *
        min_wrapper<vc_type>(abs_wrapper<vc_type>(a), abs_wrapper<vc_type>(b));
}
template <>
CUDA_GLOBAL_METHOD inline vc_type minmod_theta_wrapper<vc_type>(const vc_type& a, const vc_type& b, const vc_type& c) {
    return minmod_wrapper<vc_type>(c * minmod_wrapper<vc_type>(a, b), 0.5 * (a + b));
}
template <>
CUDA_GLOBAL_METHOD inline vc_type load_value<vc_type>(const double* __restrict__ data, const size_t index) {
    return vc_type(data + index);
}

inline void make_monotone_wrapper(double* __restrict__ qld, const double* __restrict__ q0d,
    double* __restrict__ qrd, const mask_type& mask) {
    vc_type ql(qld);
    vc_type qr(qrd);
    const vc_type q0(q0d);
    const vc_type tmp1 = qr - ql;
    const vc_type tmp2 = qr + ql;

    const mask_type mask1_tmp1 = mask_type(qr < q0);
    const mask_type mask1_tmp2 = mask_type(q0 < ql);
    const mask_type mask1 = (mask1_tmp1 ^ mask1_tmp2);
    const vc_type tmp3 = tmp1 * tmp1 / 6.0;
    const vc_type tmp4 = tmp1 * (q0 - 0.5 * tmp2);
    const mask_type mask2 = mask_type(tmp4 > tmp3);
    const mask_type mask3 = !mask_type(tmp4 > tmp3) && mask_type(-tmp3 > tmp4);
    Vc::where(mask2, ql) = (3.0 * q0 - 2.0 * qr);
    Vc::where(mask3, qr) = (3.0 * q0 - 2.0 * ql);
    Vc::where(mask1, qr) = q0;
    Vc::where(mask1, ql) = q0;
    Vc::where(!mask, ql) = vc_type(qld);
    Vc::where(!mask, qr) = vc_type(qrd);
    ql.store(qld);
    qr.store(qrd);
}

//constexpr int number_faces = 15;
constexpr int number_dirs = 27;
constexpr int q_inx = INX + 2;
constexpr int q_inx3 = q_inx * q_inx * q_inx;
constexpr int q_face_offset = number_dirs * q_inx3;
constexpr int u_face_offset = H_N3;
constexpr int x_offset = H_N3;
constexpr int q_dir_offset = q_inx3;

inline int to_q_index(const int j, const int k, const int l) {
    return j * q_inx * q_inx + k * q_inx + l;
}
// template <>
// inline void store_value<vc_type>(const double* __restrict__ data, const size_t index, const
// vc_type &value) {
//    return value.store(data + index);
//}

void reconstruct_minmod_cpu(
    double* __restrict__ combined_q, const double* __restrict__ combined_u_face, int d, int f) {
    static const cell_geometry<NDIM, INX> geo;
    static constexpr auto dir = geo.direction();
    const auto di = dir[d];
    const int start_index = f * q_face_offset + d * q_dir_offset;
    for (int j = 0; j < geo.H_NX_XM4; j++) {
        for (int k = 0; k < geo.H_NX_YM4; k++) {
            for (int l = 0; l < geo.H_NX_ZM4; l++) {
                const int i = geo.to_index(j + 2, k + 2, l + 2);
                const int q_i = to_q_index(j, k, l) + start_index;
                combined_q[q_i] = combined_u_face[i] +
                    0.5 *
                        minmod(combined_u_face[i + di] - combined_u_face[i],
                            combined_u_face[i] - combined_u_face[i - di]);
            }
        }
    }
}

void reconstruct_ppm_cpu(double* __restrict__ combined_q,
    const double* __restrict__ combined_u_face, bool smooth, bool disc_detect,
    const std::vector<std::vector<double>>& disc, const int d, const int f) {
    static const cell_geometry<NDIM, INX> geo;
    static constexpr auto dir = geo.direction();
    const auto di = dir[d];
    const auto flipped_di = geo.flip(d);
    const int start_index = f * q_face_offset + d * q_dir_offset;
    const int start_index_flipped = f * q_face_offset + flipped_di * q_dir_offset;
    const vc_type zindices = vc_type::IndexesFromZero();
    for (int j = 0; j < geo.H_NX_XM4; j++) {
        for (int k = 0; k < geo.H_NX_YM4; k++) {
            for (int l = 0; l < geo.H_NX_ZM4; l += vc_type::size()) {
                const int border = geo.H_NX_ZM4 - l;
                const mask_type mask = (zindices < border);
                if (Vc::none_of(mask))
                    continue;
                const int i = geo.to_index(j + 2, k + 2, l + 2);
                const int q_i = to_q_index(j, k, l);

                const vc_type u_plus_2di(combined_u_face + i + 2 * di);
                const vc_type u_plus_di(combined_u_face + i + di);
                const vc_type u_zero(combined_u_face + i);
                const vc_type u_minus_di(combined_u_face + i - di);
                const vc_type u_minus_2di(combined_u_face + i - 2 * di);

                const vc_type diff_u_plus = u_plus_di - u_zero;
                const vc_type diff_u_2plus = u_plus_2di - u_plus_di;

                const vc_type diff_u_minus = u_zero - u_minus_di;
                const vc_type diff_u_2minus = u_minus_di - u_minus_2di;
                const vc_type d1 = minmod_theta_wrapper<vc_type>(diff_u_plus, diff_u_minus, 2.0);
                const vc_type d1_plus =
                    minmod_theta_wrapper<vc_type>(diff_u_2plus, diff_u_plus, 2.0);
                const vc_type d1_minus =
                    minmod_theta_wrapper<vc_type>(diff_u_minus, diff_u_2minus, 2.0);

                vc_type results = 0.5 * (u_zero + u_plus_di) + (1.0 / 6.0) * (d1 - d1_plus);
                vc_type results_flipped =
                    0.5 * (u_minus_di + u_zero) + (1.0 / 6.0) * (d1_minus - d1);

                Vc::where(!mask, results) = vc_type(combined_q + start_index + q_i);
                results.store(combined_q + start_index + q_i);
                Vc::where(!mask, results_flipped) = vc_type(combined_q + start_index_flipped + q_i);
                results_flipped.store(combined_q + start_index_flipped + q_i);
            }
        }
    }

    /*if (experiment == 1) {
        for (int j = 0; j < geo.H_NX_XM2; j++) {
            for (int k = 0; k < geo.H_NX_YM2; k++) {
#pragma ivdep
                for (int l = 0; l < geo.H_NX_ZM2; l++) {
                    const int i = geo.to_index(j + 1, k + 1, l + 1);
                    for (int gi = 0; gi < geo.group_count(); gi++) {
                        safe_real sum = 0.0;
                        for (int n = 0; n < geo.group_size(gi); n++) {
                            const auto pair = geo.group_pair(gi, n);
                            sum += q[pair.second][i + pair.first];
                        }
                        sum /= safe_real(geo.group_size(gi));
                        for (int n = 0; n < geo.group_size(gi); n++) {
                            const auto pair = geo.group_pair(gi, n);
                            q[pair.second][i + pair.first] = sum;
                        }
                    }
                }
            }
        }
        for (int d = 0; d < geo.NDIR; d++) {
            const auto di = dir[d];
            for (int j = 0; j < geo.H_NX_XM2; j++) {
                for (int k = 0; k < geo.H_NX_YM2; k++) {
#pragma ivdep
                    for (int l = 0; l < geo.H_NX_ZM2; l++) {
                        const int i = geo.to_index(j + 1, k + 1, l + 1);
                        const auto mx = std::max(u[i + di], u[i]);
                        const auto mn = std::min(u[i + di], u[i]);
                        q[d][i] = std::min(mx, q[d][i]);
                        q[d][i] = std::max(mn, q[d][i]);
                    }
                }
            }
        }
    } */
    if (disc_detect) {
        constexpr auto eps = 0.01;
        constexpr auto eps2 = 0.001;
        constexpr auto eta1 = 20.0;
        constexpr auto eta2 = 0.05;
        const auto di = dir[d];
        for (int j = 0; j < geo.H_NX_XM4; j++) {
            for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
                for (int l = 0; l < geo.H_NX_ZM4; l++) {
                    const int i = geo.to_index(j + 2, k + 2, l + 2);
                    const auto& up = combined_u_face[i + di];
                    const auto& u0 = combined_u_face[i];
                    const auto& um = combined_u_face[i - di];
                    const auto dif = up - um;
                    if (std::abs(dif) > disc[d][i] * std::min(std::abs(up), std::abs(um))) {
                        if (std::min(std::abs(up), std::abs(um)) /
                                std::max(std::abs(up), std::abs(um)) >
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
                                    const int q_i = to_q_index(j, k, l);
                                    auto ul = um +
                                        0.5 *
                                            minmod_theta(combined_u_face[i] - um,
                                                um - combined_u_face[i - 2 * di], 2.0);
                                    auto ur = up -
                                        0.5 *
                                            minmod_theta(combined_u_face[i + 2 * di] - up,
                                                up - combined_u_face[i], 2.0);
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
            }
        }
    }
    if (!smooth) {
        const vc_type zindices = vc_type::IndexesFromZero();
        for (int j = 0; j < geo.H_NX_XM4; j++) {
            for (int k = 0; k < geo.H_NX_YM4; k++) {
                for (int l = 0; l < geo.H_NX_ZM4; l += vc_type::size()) {
                    const int i = geo.to_index(j + 2, k + 2, l + 2);
                    const int q_i = to_q_index(j, k, l);
                    const mask_type mask = (zindices + l < geo.H_NX_ZM4);
                    make_monotone_wrapper(combined_q + start_index + q_i, combined_u_face + i,
                        combined_q + start_index_flipped + q_i, mask);
                    // auto& qp = q[geo.flip(d)][i];
                    // auto& qm = q[d][i];
                    // make_monotone(qm, combined_u_face[i], qp);
                }
            }
        }
    }
}

void reconstruct_cpu_kernel(const safe_real omega, const size_t nf_, const int angmom_index_,
    const std::vector<bool>& smooth_field_, const std::vector<bool>& disc_detect_,
    double* __restrict__ combined_q, double* __restrict__ combined_x,
    double* __restrict__ combined_u, const double dx,
    const std::vector<std::vector<safe_real>>& cdiscs) {
    static const cell_geometry<NDIM, INX> geo;
    static thread_local std::vector<std::vector<safe_real>> AM(
        geo.NANGMOM, std::vector<safe_real>(geo.H_N3));

    static constexpr auto xloc = geo.xloc();
    static constexpr auto levi_civita = geo.levi_civita();
    static constexpr auto vw = geo.volume_weight();
    static constexpr auto dir = geo.direction();
    const int n_species_ = physics<NDIM>::get_n_species();

    // Current implementation limitations of this kernel - can be resolved but that takes more work
    assert(angmom_index_ > -1);
    assert(NDIM > 2);
//    assert(nf == 15); // is not required anymore
    assert(geo.NDIR == 27);
    assert(INX == 8);
    // TODO Make kernel work with a wider range of parameters

    const int sx_i = angmom_index_;
    const int zx_i = sx_i + NDIM;

    for (int n = 0; n < geo.NANGMOM; n++) {
        for (int j = 0; j < geo.H_NX_XM4; j++) {
            for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
                for (int l = 0; l < geo.H_NX_ZM4; l++) {
                    const int i = geo.to_index(j + 2, k + 2, l + 2);
                    AM[n][i] = combined_u[(zx_i + n) * u_face_offset + i] * combined_u[i];
                }
            }
        }
    }

    for (int d = 0; d < geo.NDIR; d++) {
        if (d < geo.NDIR / 2) {
            for (int f = 0; f < angmom_index_; f++) {
                reconstruct_ppm_cpu(combined_q, combined_u + u_face_offset * f, smooth_field_[f],
                    disc_detect_[f], cdiscs, d, f);
            }
            for (int f = sx_i; f < sx_i + NDIM; f++) {
                reconstruct_ppm_cpu(
                    combined_q, combined_u + u_face_offset * f, true, false, cdiscs, d, f);
            }
        }
        for (int f = zx_i; f < zx_i + geo.NANGMOM; f++) {
            reconstruct_minmod_cpu(combined_q, combined_u + u_face_offset * f, d, f);
        }
        if (d < geo.NDIR / 2) {
            for (int f = angmom_index_ + geo.NANGMOM + NDIM; f < nf_; f++) {
                reconstruct_ppm_cpu(combined_q, combined_u + u_face_offset * f, smooth_field_[f],
                    disc_detect_[f], cdiscs, d, f);
            }
        }

        if (d != geo.NDIR / 2) {
            const int start_index_rho = d * q_dir_offset;
            const vc_type zindices = vc_type::IndexesFromZero();
            for (int j = 0; j < geo.H_NX_XM4; j++) {
                for (int k = 0; k < geo.H_NX_YM4; k++) {
                    for (int l = 0; l < geo.H_NX_ZM4; l += vc_type::size()) {
                        const int border = geo.H_NX_ZM4 - l;
                        const mask_type mask = (zindices < border);
                        if (Vc::none_of(mask))
                            continue;
                        const int i = geo.to_index(j + 2, k + 2, l + 2);
                        const int q_i = to_q_index(j, k, l);

                        // n m q Levi Civita
                        // 0 1 2 -> 1
                        vc_type results0 = vc_type(AM[0].data() + i) -
                            vw[d] * 1.0 * 0.5 * xloc[d][1] *
                                vc_type(combined_q + (sx_i + 2) * q_face_offset + d * q_dir_offset +
                                    q_i) *
                                vc_type(combined_q + start_index_rho + q_i) * dx;

                        // n m q Levi Civita
                        // 0 2 1 -> -1
                        results0 -= vw[d] * (-1.0) * 0.5 * xloc[d][2] *
                            vc_type(
                                combined_q + (sx_i + 1) * q_face_offset + d * q_dir_offset + q_i) *
                            vc_type(combined_q + start_index_rho + q_i) * dx;
                        Vc::where(!mask, results0) = vc_type(AM[0].data() + i);
                        results0.store(AM[0].data() + i);

                        // n m q Levi Civita
                        // 1 0 2 -> -1
                        vc_type results1 = vc_type(AM[1].data() + i) -
                            vw[d] * (-1.0) * 0.5 * xloc[d][0] *
                                vc_type(combined_q + (sx_i + 2) * q_face_offset + d * q_dir_offset +
                                    q_i) *
                                vc_type(combined_q + start_index_rho + q_i) * dx;

                        // n m q Levi Civita
                        // 1 2 0 -> 1
                        results1 -= vw[d] * (1.0) * 0.5 * xloc[d][2] *
                            vc_type(
                                combined_q + (sx_i + 0) * q_face_offset + d * q_dir_offset + q_i) *
                            vc_type(combined_q + start_index_rho + q_i) * dx;
                        Vc::where(!mask, results1) = vc_type(AM[1].data() + i);
                        results1.store(AM[1].data() + i);

                        // n m q Levi Civita
                        // 2 0 1 -> 1
                        vc_type results2 = vc_type(AM[2].data() + i) -
                            vw[d] * (1.0) * 0.5 * xloc[d][0] *
                                vc_type(combined_q + (sx_i + 1) * q_face_offset + d * q_dir_offset +
                                    q_i) *
                                vc_type(combined_q + start_index_rho + q_i) * dx;

                        // n m q Levi Civita
                        // 2 1 0 -> -1
                        results2 -= vw[d] * (-1.0) * 0.5 * xloc[d][1] *
                            vc_type(
                                combined_q + (sx_i + 0) * q_face_offset + d * q_dir_offset + q_i) *
                            vc_type(combined_q + start_index_rho + q_i) * dx;
                        Vc::where(!mask, results1) = vc_type(AM[2].data() + i);
                        results2.store(AM[2].data() + i);
                    }
                }
            }
        }
    }
    for (int d = 0; d < geo.NDIR / 2; d++) {
        const auto di = dir[d];

        for (int q = 0; q < NDIM; q++) {
            const auto f = sx_i + q;
            const int start_index_f = f * q_face_offset + d * q_dir_offset;
            const int start_index_flipped = f * q_face_offset + geo.flip(d) * q_dir_offset;
            const int start_index_zero = 0 * q_face_offset + d * q_dir_offset;
            const int start_index_zero_flipped = 0 * q_face_offset + geo.flip(d) * q_dir_offset;
            for (int j = 0; j < geo.H_NX_XM4; j++) {
                for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
                    for (int l = 0; l < geo.H_NX_ZM4; l++) {
                        const int i = geo.to_index(j + 2, k + 2, l + 2);
                        const int q_i = to_q_index(j, k, l);
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
                        for (int n = 0; n < geo.NANGMOM; n++) {
                            for (int m = 0; m < NDIM; m++) {
                                const auto lc = levi_civita[n][m][q];
                                b += 12.0 * AM[n][i] * lc * xloc[d][m] / (dx * (rho_l + rho_r));
                            }
                        }
                        double blim;
                        if ((ur - u0) * (u0 - ul) <= 0.0) {
                            blim = 0.0;
                        } else {
                            blim = b0;
                        }
                        b = minmod(blim, b);
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
            }
        }
    }
    //    for (int d = 0; d < geo.NDIR; d++) {
    // }

    // post_recon_experimental(Q, X, omega, angmom_index_ != -1);

    for (int d = 0; d < geo.NDIR; d++) {
        const int start_index_rho = rho_i * q_face_offset + d * q_dir_offset;
        if (d != geo.NDIR / 2) {
            const int start_index_sx = sx_i * q_face_offset + d * q_dir_offset;
            const int start_index_sy = sy_i * q_face_offset + d * q_dir_offset;
            for (int j = 0; j < geo.H_NX_XM4; j++) {
                for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
                    for (int l = 0; l < geo.H_NX_ZM4; l++) {
                        const int q_i = to_q_index(j, k, l);
                        combined_q[start_index_sx + q_i] -=
                            omega * (combined_x[1 * q_inx3 + q_i] + 0.5 * xloc[d][1] * dx);
                        combined_q[start_index_sy + q_i] +=
                            omega * (combined_x[q_i] + 0.5 * xloc[d][0] * dx);
                        // Q[sx_i][d][i] -= omega * (X[1][i] + 0.5 * xloc[d][1] * dx);
                        // Q[sy_i][d][i] += omega * (X[0][i] + 0.5 * xloc[d][0] * dx);
                    }
                }
            }
            const vc_type zindices = vc_type::IndexesFromZero();
            for (int j = 0; j < geo.H_NX_XM4; j++) {
                for (int k = 0; k < geo.H_NX_YM4; k++) {
                    for (int l = 0; l < geo.H_NX_ZM4; l += vc_type::size()) {
                        const int border = geo.H_NX_ZM4 - l;
                        const mask_type mask = (zindices < border);
                        if (Vc::none_of(mask))
                            continue;

                        const int i = geo.to_index(j + 2, k + 2, l + 2);
                        const int q_i = to_q_index(j, k, l);
                        const vc_type rho(combined_q + start_index_rho + q_i);

                        // n m q Levi Civita
                        // 0 1 2 -> 1
                        const vc_type xloc_tmp1 = vc_type(0.5 * xloc[d][1] * dx);
                        const vc_type q_lx_val0(
                            combined_q + (lx_i + 0) * q_face_offset + d * q_dir_offset + q_i);
                        auto result0 = q_lx_val0 +
                            (1.0) * (vc_type(combined_x + q_inx3 + q_i) + xloc_tmp1) *
                                vc_type(combined_q + (sx_i + 2) * q_face_offset + d * q_dir_offset +
                                    q_i);

                        // n m q Levi Civita
                        // 0 2 1 -> -1
                        const vc_type xloc_tmp2 = vc_type(0.5 * xloc[d][2] * dx);
                        result0 += (-1.0) * (vc_type(combined_x + 2 * q_inx3 + q_i) + xloc_tmp2) *
                            vc_type(
                                combined_q + (sx_i + 1) * q_face_offset + d * q_dir_offset + q_i);
                        Vc::where(!mask, result0) = vc_type(
                            combined_q + (lx_i + 0) * q_face_offset + d * q_dir_offset + q_i);
                        result0.store(
                            combined_q + (lx_i + 0) * q_face_offset + d * q_dir_offset + q_i);

                        // n m q Levi Civita
                        // 1 0 2 -> -1
                        const vc_type xloc_tmp0 = vc_type(0.5 * xloc[d][0] * dx);
                        const vc_type q_lx_val1(
                            combined_q + (lx_i + 1) * q_face_offset + d * q_dir_offset + q_i);
                        auto result1 = q_lx_val1 +
                            (-1.0) * (vc_type(combined_x + q_i) + xloc_tmp0) *
                                vc_type(combined_q + (sx_i + 2) * q_face_offset + d * q_dir_offset +
                                    q_i);

                        // n m q Levi Civita
                        // 1 2 0 -> 1
                        result1 += (1.0) * (vc_type(combined_x + 2 * q_inx3 + q_i) + xloc_tmp2) *
                            vc_type(
                                combined_q + (sx_i + 0) * q_face_offset + d * q_dir_offset + q_i);
                        Vc::where(!mask, result1) = vc_type(
                            combined_q + (lx_i + 1) * q_face_offset + d * q_dir_offset + q_i);
                        result1.store(
                            combined_q + (lx_i + 1) * q_face_offset + d * q_dir_offset + q_i);

                        // n m q Levi Civita
                        // 2 0 1 -> 1
                        const vc_type q_lx_val2(
                            combined_q + (lx_i + 2) * q_face_offset + d * q_dir_offset + q_i);
                        auto result2 = q_lx_val2 +
                            (1.0) * (vc_type(combined_x + q_i) + xloc_tmp0) *
                                vc_type(combined_q + (sx_i + 1) * q_face_offset + d * q_dir_offset +
                                    q_i);

                        // n m q Levi Civita
                        // 2 1 0 -> -1
                        result2 += (-1.0) * (vc_type(combined_x + q_inx3 + q_i) + xloc_tmp1) *
                            vc_type(
                                combined_q + (sx_i + 0) * q_face_offset + d * q_dir_offset + q_i);
                        Vc::where(!mask, result2) = vc_type(
                            combined_q + (lx_i + 2) * q_face_offset + d * q_dir_offset + q_i);
                        result2.store(
                            combined_q + (lx_i + 2) * q_face_offset + d * q_dir_offset + q_i);
                    }
                }
            }
            const int start_index_egas = egas_i * q_face_offset + d * q_dir_offset;
            const int start_index_pot = pot_i * q_face_offset + d * q_dir_offset;
            for (int j = 0; j < geo.H_NX_XM4; j++) {
                for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
                    for (int l = 0; l < geo.H_NX_ZM4; l++) {
                        const int q_i = to_q_index(j, k, l);
                        const auto rho = combined_q[start_index_rho + q_i];
                        for (int n = 0; n < geo.NANGMOM; n++) {
                            const int start_index_lx_n =
                                (lx_i + n) * q_face_offset + d * q_dir_offset;
                            combined_q[start_index_lx_n + q_i] *= rho;
                        }
                        for (int dim = 0; dim < NDIM; dim++) {
                            const int start_index_sx_d =
                                (sx_i + dim) * q_face_offset + d * q_dir_offset;
                            auto& v = combined_q[start_index_sx_d + q_i];
                            combined_q[start_index_egas + q_i] += 0.5 * v * v * rho;
                            v *= rho;
                        }
                        combined_q[start_index_pot + q_i] *= rho;
                        safe_real w = 0.0;
                        for (int si = 0; si < n_species_; si++) {
                            const int start_index_sp_i =
                                (spc_i + si) * q_face_offset + d * q_dir_offset;
                            w += combined_q[start_index_sp_i + q_i];
                            combined_q[start_index_sp_i + q_i] *= rho;
                        }
                        w = 1.0 / w;
                        for (int si = 0; si < n_species_; si++) {
                            const int start_index_sp_i =
                                (spc_i + si) * q_face_offset + d * q_dir_offset;
                            combined_q[start_index_sp_i + q_i] *= w;
                        }
                    }
                }
            }
        }
    }
    return;
}

void hydro_pre_recon_cpu_kernel(const double* __restrict__ X, safe_real omega,
    bool angmom, double* __restrict__ combined_u, const int nf, const int n_species_) {
    static const cell_geometry<NDIM, INX> geo;

    for (int j = 0; j < H_NX; j++) { // == H_NX == 14
        for (int k = 0; k < H_NX; k++) {
#pragma ivdep
            for (int l = 0; l < H_NX; l++) {
                const int i = geo.to_index(j, k, l);
                // const auto rho = V[rho_i][i];
                const auto rho = combined_u[rho_i * u_face_offset + i];
                const auto rhoinv = 1.0 / rho;
                for (int dim = 0; dim < NDIM; dim++) {
                    auto& s = combined_u[(sx_i + dim) * u_face_offset + i];
                    combined_u[egas_i * u_face_offset + i] -= 0.5 * s * s * rhoinv;
                    s *= rhoinv;
                }
                for (int si = 0; si < n_species_; si++) {
                    combined_u[(spc_i + si) * u_face_offset + i] *= rhoinv;
                }
                combined_u[pot_i * u_face_offset + i] *= rhoinv;

                //const auto rho = combined_u[rho_i * u_face_offset + i];
                //const auto rhoinv = 1.0 / rho;
                combined_u[(lx_i + 0) * u_face_offset + i] *= rhoinv;
                combined_u[(lx_i + 1) * u_face_offset + i] *= rhoinv;
                combined_u[(lx_i + 2) * u_face_offset + i] *= rhoinv;

                // Levi civita n m q -> lc
                // Levi civita 0 1 2 -> 1
                combined_u[(lx_i + 0) * u_face_offset + i] -=
                    1.0 * X[1 * x_offset + i] * combined_u[(sx_i + 2) * u_face_offset + i];
                // Levi civita n m q -> lc
                // Levi civita 0 2 1 -> -1
                combined_u[(lx_i + 0) * u_face_offset + i] -=
                    -1.0 * X[2 * x_offset + i] * combined_u[(sx_i + 1) * u_face_offset + i];
                // Levi civita n m q -> lc
                // Levi civita 1 0 2 -> -1
                combined_u[(lx_i + 1) * u_face_offset + i] -=
                    -1.0 * X[i] * combined_u[(sx_i + 2) * u_face_offset + i];
                // Levi civita n m q -> lc
                // Levi civita 1 2 0 -> 1
                combined_u[(lx_i + 1) * u_face_offset + i] -=
                    1.0 * X[2 * x_offset + i] * combined_u[(sx_i + 0) * u_face_offset + i];
                // Levi civita n m q -> lc
                // Levi civita 2 0 1 -> 1
                combined_u[(lx_i + 2) * u_face_offset + i] -=
                    1.0 * X[i] * combined_u[(sx_i + 1) * u_face_offset + i];
                // Levi civita n m q -> lc
                // Levi civita 2 1 0 -> -1
                combined_u[(lx_i + 2) * u_face_offset + i] -=
                    -1.0 * X[1 * x_offset + i] * combined_u[(sx_i + 0) * u_face_offset + i];

                // combined_u[(lx_i + n) * u_face_offset + i] -=
                //    lc * X[m][i] * combined_u[(sx_i + q) * u_face_offset + i];
                combined_u[sx_i * u_face_offset + i] += omega * X[1 * x_offset + i];
                combined_u[sy_i * u_face_offset + i] -= omega * X[0 * x_offset + i];
            }
        }
    }
}

inline double deg_pres(const double x, const double A_) {
    double p;
    if (x < 0.001) {
        p = 1.6 * A_ * std::pow(x, 5);
    } else {
        p = A_ * (x * (2 * x * x - 3) * std::sqrt(x * x + 1) + 3 * asinh(x));
    }
    return p;
}

void convert_find_contact_discs(const double* __restrict__ combined_u, double* __restrict__ disc,
    const double A_, const double B_, const double fgamma_, const double de_switch_1) {
    static const cell_geometry<NDIM, INX> geo;
    auto dir = geo.direction();
    // static thread_local std::vector<std::vector<safe_real>> disc(geo.NDIR / 2,
    // std::vector<double>(geo.H_N3));
    constexpr int disc_offset = geo.H_N3;
    static thread_local std::vector<safe_real> P(H_N3);
    for (int j = 0; j < geo.H_NX_XM2; j++) {
        for (int k = 0; k < geo.H_NX_YM2; k++) {
#pragma ivdep
            for (int l = 0; l < geo.H_NX_ZM2; l++) {
                const int i = geo.to_index(j + 1, k + 1, l + 1);
                const auto rho = combined_u[rho_i * u_face_offset + i];
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
                    ek += combined_u[(sx_i + dim) * u_face_offset + i] *
                        combined_u[(sx_i + dim) * u_face_offset + i] * rhoinv * 0.5;
                }
                auto ein = combined_u[egas_i * u_face_offset + i] - ek - edeg;
                if (ein < de_switch_1 * combined_u[egas_i * u_face_offset + i]) {
                    //	printf( "%e\n", U[tau_i][i]);
                    ein = pow(combined_u[tau_i * u_face_offset + i], fgamma_);
                }
                P[i] = (fgamma_ - 1.0) * ein + pdeg;
            }
        }
    }
    for (int d = 0; d < geo.NDIR / 2; d++) {
        const auto di = dir[d];
        for (int j = 0; j < geo.H_NX_XM4; j++) {
            for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
                for (int l = 0; l < geo.H_NX_ZM4; l++) {
                    constexpr auto K0 = 0.1;
                    const int i = geo.to_index(j + 2, k + 2, l + 2);
                    const auto P_r = P[i + di];
                    const auto P_l = P[i - di];
                    const auto tmp1 = fgamma_ * K0;
                    const auto tmp2 = std::abs(P_r - P_l) / std::min(std::abs(P_r), std::abs(P_l));
                    disc[d * disc_offset + i] = tmp2 / tmp1;
                }
            }
        }
    }
}
#pragma GCC pop_options
#endif
#endif

#endif