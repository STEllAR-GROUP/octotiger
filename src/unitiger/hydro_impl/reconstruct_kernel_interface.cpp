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
inline vc_type copysign_wrapper<vc_type>(const vc_type& tmp1, const vc_type& tmp2) {
    return Vc::copysign(tmp1, tmp2);
}
template <>
inline vc_type abs_wrapper<vc_type>(const vc_type& tmp1) {
    return Vc::abs(tmp1);
}
template <>
inline vc_type minmod_wrapper<vc_type>(const vc_type& a, const vc_type& b) {
    return (copysign_wrapper<vc_type>(0.5, a) + copysign_wrapper<vc_type>(0.5, b)) *
        min_wrapper<vc_type>(abs_wrapper<vc_type>(a), abs_wrapper<vc_type>(b));
}
template <>
inline vc_type minmod_theta_wrapper<vc_type>(const vc_type& a, const vc_type& b, const vc_type& c) {
    return minmod_wrapper<vc_type>(c * minmod_wrapper<vc_type>(a, b), 0.5 * (a + b));
}
template <>
inline vc_type load_value<vc_type>(const double* __restrict__ data, const size_t index) {
    return vc_type(data + index);
}
// template <>
// inline void store_value<vc_type>(const double* __restrict__ data, const size_t index, const
// vc_type &value) {
//    return value.store(data + index);
//}

void reconstruct_minmod(std::vector<std::vector<safe_real>>& q, const std::vector<safe_real>& u) {
    static const cell_geometry<NDIM, INX> geo;
    static constexpr auto dir = geo.direction();
    for (int d = 0; d < geo.NDIR; d++) {
        const auto di = dir[d];
        for (int j = 0; j < geo.H_NX_XM2; j++) {
            for (int k = 0; k < geo.H_NX_YM2; k++) {
                for (int l = 0; l < geo.H_NX_ZM2; l++) {
                    const int i = geo.to_index(j + 1, k + 1, l + 1);
                    q[d][i] = u[i] + 0.5 * minmod(u[i + di] - u[i], u[i] - u[i - di]);
                }
            }
        }
    }
}

void reconstruct_ppm_experimental(std::vector<std::vector<safe_real>>& q,
    const std::vector<safe_real>& u, bool smooth, bool disc_detect,
    const std::vector<std::vector<double>>& disc) {
    PROFILE();

    static const cell_geometry<NDIM, INX> geo;
    static constexpr auto dir = geo.direction();
    // const vc_type zindices = vc_type::IndexesFromZero() + 1;
    for (int d = 0; d < geo.NDIR / 2; d++) {
        const auto di = dir[d];
        const auto flipped_di = geo.flip(d);
        for (int j = 0; j < geo.H_NX_XM2; j++) {
            for (int k = 0; k < geo.H_NX_YM2; k++) {
                for (int l = 0; l < geo.H_NX_ZM2; l += vc_type::size()) {
                    const int i = geo.to_index(j + 1, k + 1, l + 1);
                    const vc_type u_plus_di(u.data() + i + di);
                    const vc_type u_zero(u.data() + i);
                    const vc_type diff_u = u_plus_di - u_zero;

                    const vc_type u_minus_di(u.data() + i - di);
                    const vc_type d1 =
                        minmod_theta_wrapper<vc_type>(diff_u, u_zero - u_minus_di, 2.0);

                    const vc_type u_plus_2di(u.data() + i + 2 * di);
                    const vc_type d1_di =
                        minmod_theta_wrapper<vc_type>(u_plus_2di - u_plus_di, diff_u, 2.0);
                    const vc_type results = 0.5 * (u_zero + u_plus_di) + (1.0 / 6.0) * (d1 - d1_di);
                    results.store(q[d].data() + i);
                    results.store(q[flipped_di].data() + di + i);
                }
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
        for (int d = 0; d < geo.NDIR / 2; d++) {
            const auto di = dir[d];
            for (int j = 0; j < geo.H_NX_XM4; j++) {
                for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
                    for (int l = 0; l < geo.H_NX_ZM4; l++) {
                        const int i = geo.to_index(j + 2, k + 2, l + 2);
                        const auto& up = u[i + di];
                        const auto& u0 = u[i];
                        const auto& um = u[i - di];
                        const auto dif = up - um;
                        if (std::abs(dif) > disc[d][i] * std::min(std::abs(up), std::abs(um))) {
                            if (std::min(std::abs(up), std::abs(um)) /
                                    std::max(std::abs(up), std::abs(um)) >
                                eps2) {
                                const auto d2p =
                                    (1.0 / 6.0) * (u[i + 2 * di] + u0 - 2.0 * u[i + di]);
                                const auto d2m =
                                    (1.0 / 6.0) * (u0 + u[i - 2 * di] - 2.0 * u[i - di]);
                                if (d2p * d2m < 0.0) {
                                    double eta = 0.0;
                                    if (std::abs(dif) >
                                        eps * std::min(std::abs(up), std::abs(um))) {
                                        eta = -(d2p - d2m) / dif;
                                    }
                                    eta = std::max(0.0, std::min(eta1 * (eta - eta2), 1.0));
                                    if (eta > 0.0) {
                                        auto ul = um +
                                            0.5 * minmod_theta(u[i] - um, um - u[i - 2 * di], 2.0);
                                        auto ur = up -
                                            0.5 * minmod_theta(u[i + 2 * di] - up, up - u[i], 2.0);
                                        auto& qp = q[d][i];
                                        auto& qm = q[geo.flip(d)][i];
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
    }
    if (!smooth) {
        for (int d = 0; d < geo.NDIR / 2; d++) {
            for (int j = 0; j < geo.H_NX_XM4; j++) {
                for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
                    for (int l = 0; l < geo.H_NX_ZM4; l++) {
                        const int i = geo.to_index(j + 2, k + 2, l + 2);
                        auto& qp = q[geo.flip(d)][i];
                        auto& qm = q[d][i];
                        make_monotone(qm, u[i], qp);
                    }
                }
            }
        }
    }
}

const hydro::recon_type<NDIM>& reconstruct_experimental(const hydro::state_type& U_,
    const hydro::x_type& X, safe_real omega, const size_t nf_, const int angmom_index_,
    const std::vector<bool>& smooth_field_, const std::vector<bool>& disc_detect_) {
    static const cell_geometry<NDIM, INX> geo;
    static thread_local std::vector<std::vector<safe_real>> AM(
        geo.NANGMOM, std::vector<safe_real>(geo.H_N3));
    static thread_local std::vector<std::vector<std::vector<safe_real>>> Q(
        nf_, std::vector<std::vector<safe_real>>(geo.NDIR, std::vector<safe_real>(geo.H_N3)));

    static constexpr auto xloc = geo.xloc();
    static constexpr auto levi_civita = geo.levi_civita();
    static constexpr auto vw = geo.volume_weight();
    static constexpr auto dir = geo.direction();

    const auto dx = X[0][geo.H_DNX] - X[0][0];
    const auto& U = physics<NDIM>::pre_recon<INX>(U_, X, omega, angmom_index_ != -1);
    const auto& cdiscs = physics<NDIM>::find_contact_discs<INX>(U_);
    if (angmom_index_ == -1 || NDIM == 1) {
        for (int f = 0; f < nf_; f++) {
            if (f < lx_i || f > lx_i + geo.NANGMOM || NDIM == 1) {
                reconstruct_ppm_experimental(Q[f], U[f], smooth_field_[f], disc_detect_[f], cdiscs);
            } else {
                reconstruct_minmod(Q[f], U[f]);
            }
        }

    } else {
        for (int f = 0; f < angmom_index_; f++) {
            reconstruct_ppm_experimental(Q[f], U[f], smooth_field_[f], disc_detect_[f], cdiscs);
        }

        int sx_i = angmom_index_;
        int zx_i = sx_i + NDIM;

        for (int f = sx_i; f < sx_i + NDIM; f++) {
            reconstruct_ppm_experimental(Q[f], U[f], true, false, cdiscs);
        }
        for (int f = zx_i; f < zx_i + geo.NANGMOM; f++) {
            reconstruct_minmod(Q[f], U[f]);
        }

        for (int n = 0; n < geo.NANGMOM; n++) {
            for (int j = 0; j < geo.H_NX_XM4; j++) {
                for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
                    for (int l = 0; l < geo.H_NX_ZM4; l++) {
                        const int i = geo.to_index(j + 2, k + 2, l + 2);
                        AM[n][i] = U[zx_i + n][i] * U[0][i];
                    }
                }
            }
            const vc_type zindices = vc_type::IndexesFromZero();
            for (int m = 0; m < NDIM; m++) {
                for (int q = 0; q < NDIM; q++) {
                    const auto lc = levi_civita[n][m][q];
                    if (lc != 0) {
                        for (int d = 0; d < geo.NDIR; d++) {
                            if (d != geo.NDIR / 2) {
                                for (int j = 0; j < geo.H_NX_XM4; j++) {
                                    for (int k = 0; k < geo.H_NX_YM4; k++) {
                                        //for (int l = 0; l < geo.H_NX_ZM4; l += 1) {
                                        // const int i = geo.to_index(j + 2, k + 2, l + 2);
                                        for (int l = 0; l < geo.H_NX_ZM4; l += vc_type::size()) {
                                            const int i = geo.to_index(j + 2, k + 2, l + 2);
                                            /*const int border = geo.H_NX_ZM4 - l;
                                            const mask_type mask = (zindices < border);
                                            if (Vc::none_of(mask))
                                                continue;
                                            vc_type results = vc_type(AM[n].data() + i);
                                            Vc::where(mask, results) = vc_type(AM[n].data() + i) -
                                                vw[d] * lc * 0.5 * xloc[d][m] *
                                                    vc_type(Q[sx_i + q][d].data() + i) *
                                                    vc_type(Q[0][d].data() + i) * dx;*/
                                            const vc_type results = vc_type(AM[n].data() + i) -
                                                vw[d] * lc * 0.5 * xloc[d][m] *
                                                    vc_type(Q[sx_i + q][d].data() + i) *
                                                    vc_type(Q[0][d].data() + i) * dx;
                                            results.store(AM[n].data() + i);
                                            //AM[n][i] -= vw[d] * lc * 0.5 * xloc[d][m] *
                                             //   Q[sx_i + q][d][i] * Q[0][d][i] * dx;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for (int q = 0; q < NDIM; q++) {
            const auto f = sx_i + q;
            for (int d = 0; d < geo.NDIR / 2; d++) {
                const auto di = dir[d];
                for (int j = 0; j < geo.H_NX_XM4; j++) {
                    for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
                        for (int l = 0; l < geo.H_NX_ZM4; l++) {
                            const int i = geo.to_index(j + 2, k + 2, l + 2);
                            const auto& rho_r = Q[0][d][i];
                            const auto& rho_l = Q[0][geo.flip(d)][i];
                            auto& qr = Q[f][d][i];
                            auto& ql = Q[f][geo.flip(d)][i];
                            const auto& ur = U[f][i + di];
                            const auto& u0 = U[f][i];
                            const auto& ul = U[f][i - di];
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
        for (int f = angmom_index_ + geo.NANGMOM + NDIM; f < nf_; f++) {
            reconstruct_ppm_experimental(Q[f], U[f], smooth_field_[f], disc_detect_[f], cdiscs);
        }
    }

#ifdef TVD_TEST
    {
        PROFILE();
        /**** ENSURE TVD TEST***/
        for (int f = 0; f < nf_; f++) {
            if (!smooth_field_[f]) {
                for (int d = 0; d < geo.NDIR / 2; d++) {
                    for (int j = 0; j < geo.H_NX_XM4; j++) {
                        for (int k = 0; k < geo.H_NX_YM4; k++) {
#pragma ivdep
                            for (int l = 0; l < geo.H_NX_ZM4; l++) {
                                const int i = geo.to_index(j + 2, k + 2, l + 2);
                                const auto up = U[f][i + dir[d]];
                                const auto u0 = U[f][i];
                                const auto um = U[f][i - dir[d]];
                                const auto qp = Q[f][d][i];
                                const auto qm = Q[f][geo.flip(d)][i];
                                auto norm =
                                    std::max(std::abs(u0), std::max(std::abs(up), std::abs(um)));
                                norm *= norm;
                                if ((qp - qm) * (up - um) < -1.0e-12 * norm) {
                                    printf("TVD fail 1 %e\n", (qp - qm) * (up - um) / norm);
                                    abort();
                                }
                                //								if (!PPM_test(qp, u0, qm)) {
                                //									printf("TVD fail 4\n");
                                //									abort();
                                //								}
                            }
                        }
                    }
                }
                for (int d = 0; d < geo.NDIR; d++) {
                    if (d != geo.NDIR / 2) {
                        for (int j = 0; j < geo.H_NX_XM6; j++) {
                            for (int k = 0; k < geo.H_NX_YM6; k++) {
#pragma ivdep
                                for (int l = 0; l < geo.H_NX_ZM6; l++) {
                                    const int i = geo.to_index(j + 3, k + 3, l + 3);
                                    const auto ur = U[f][i + dir[d]];
                                    const auto ul = U[f][i];
                                    const auto ql = Q[f][geo.flip(d)][i + dir[d]];
                                    const auto qr = Q[f][d][i];
                                    auto norm = std::max(std::abs(ur), std::abs(ul));
                                    norm *= norm;
                                    if ((qr - ul) * (ur - qr) < -1.0e-12 * norm) {
                                        printf("TVD fail 3 %e\n", (qr - ul) * (ur - qr) / norm);
                                        abort();
                                    }
                                    if ((ql - ul) * (ur - ql) < -1.0e-12 * norm) {
                                        printf("TVD fail 5 %e\n", (ql - ul) * (ur - ql) / norm);
                                        abort();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

#endif
    physics<NDIM>::post_recon<INX>(Q, X, omega, angmom_index_ != -1);

    return Q;
}
#pragma GCC pop_options
