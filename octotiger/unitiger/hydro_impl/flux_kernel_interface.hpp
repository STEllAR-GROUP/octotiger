#include <array>
#include <vector>

#include "octotiger/hydro_defs.hpp"
#include "octotiger/unitiger/hydro.hpp"
#include "octotiger/unitiger/safe_real.hpp"

#include <boost/container/vector.hpp>    // to get non-specialized vector<bool>

timestep_t flux_kernel_interface(const hydro::recon_type<NDIM>& Q, hydro::flux_type& F,
    hydro::x_type& X, safe_real omega, const size_t nf_);

timestep_t flux_cpu_kernel(const hydro::recon_type<NDIM>& Q, hydro::flux_type& F, hydro::x_type& X,
    safe_real omega, const size_t nf_);

timestep_t flux_unified_cpu_kernel(const hydro::recon_type<NDIM>& Q, hydro::flux_type& F,
    hydro::x_type& X, safe_real omega, const size_t nf_);

// helpers for using vectortype specialization functions
template <typename double_t, typename cond_t>
inline void select_wrapper(
    double_t& target, const cond_t cond, const double_t& tmp1, const double_t& tmp2) {
    target = cond ? tmp1 : tmp2;
}
template <typename T>
inline T max_wrapper(const T& tmp1, const T& tmp2) {
    return std::max(tmp1, tmp2);
}
template <typename T>
inline T min_wrapper(const T& tmp1, const T& tmp2) {
    return std::min(tmp1, tmp2);
}
template <typename T>
inline T sqrt_wrapper(const T& tmp1) {
    return std::sqrt(tmp1);
}
template <typename T>
inline T pow_wrapper(const T& tmp1, const double& tmp2) {
    return std::pow(tmp1, tmp2);
}
template <typename T>
inline T asin_wrapper(const T& tmp1) {
    return std::asin(tmp1);
}
template <typename T>
inline bool skippable(const T& tmp1) {
    return !tmp1;
}

boost::container::vector<bool> create_masks();

#pragma GCC push_options
#pragma GCC optimize("unroll-loops")

template <typename double_t>
inline double_t inner_flux_loop(const double omega, const size_t nf_, const double A_,
    const double B_, const double_t* __restrict__ UR, const double_t* __restrict__ UL,
    double_t* __restrict__ this_flux,
    const double_t* __restrict__ x, const double_t* __restrict__ vg, double_t& ap, double_t& am,
    const size_t dim, const size_t d, const double dx) {
    double_t amr, apr, aml, apl;
    double_t this_ap, this_am;    // tmps
    double_t p, v, v0, c;

    auto rho = UR[rho_i];
    auto rhoinv = (1.) / rho;
    double_t hdeg = static_cast<double_t>(0.0), pdeg = static_cast<double_t>(0.0),
             edeg = static_cast<double_t>(0.0), dpdeg_drho = static_cast<double_t>(0.0);

    // all workitems choose the same path
    if (A_ != 0.0) {
        const auto Binv = 1.0 / B_;
        const auto x = pow_wrapper(rho * Binv, 1.0 / 3.0);
        const auto x_sqr = x * x;
        const auto x_sqr_sqrt = sqrt_wrapper(x_sqr + 1.0);
        const auto x_pow_5 = x_sqr * x_sqr * x;
        hdeg = 8.0 * A_ * Binv * (x_sqr_sqrt - 1.0);

        const double_t edeg_tmp1 = rho * hdeg - pdeg;
        const double_t edeg_tmp2 = 2.4 * A_ * x_pow_5;
        const double_t pdeg_tmp1 = A_ * (x * (2 * x_sqr - 3) * x_sqr_sqrt + 3 * asin_wrapper(x));
        const double_t pdeg_tmp2 = 1.6 * A_ * x_pow_5;
        select_wrapper(edeg, (x > 0.001), edeg_tmp1, edeg_tmp2);
        select_wrapper(pdeg, (x > 0.001), pdeg_tmp1, pdeg_tmp2);

        dpdeg_drho = 8.0 / 3.0 * A_ * Binv * x_sqr / x_sqr_sqrt;
    }
    double_t ek = 0.0;
    for (int dim = 0; dim < NDIM; dim++) {
        ek += UR[sx_i + dim] * UR[sx_i + dim] * rhoinv * 0.5;
    }
    const auto ein1_tmp2 = UR[egas_i] - ek - edeg;
    const auto ein1_mask = (ein1_tmp2 < (physics<NDIM>::de_switch_1 * UR[egas_i]));
    double_t ein;
    if (!skippable(ein1_mask)) {
        const auto ein1_tmp1 = pow_wrapper(UR[tau_i], physics<NDIM>::fgamma_);
        select_wrapper(ein, ein1_mask, ein1_tmp1, ein1_tmp2);
    } else {
        ein = ein1_tmp2;
    }
    double_t dp_drho = dpdeg_drho + (physics<NDIM>::fgamma_ - 1.0) * ein * rhoinv;
    double_t dp_deps = (physics<NDIM>::fgamma_ - 1.0) * rho;
    v0 = UR[sx_i + dim] * rhoinv;
    p = (physics<NDIM>::fgamma_ - 1.0) * ein + pdeg;
    c = sqrt_wrapper(p * rhoinv * rhoinv * dp_deps + dp_drho);
    v = v0 - vg[dim];
    amr = v - c;
    apr = v + c;

    rho = UL[rho_i];
    rhoinv = (1.) / rho;
    hdeg = static_cast<double_t>(0.0);
    pdeg = static_cast<double_t>(0.0);
    edeg = static_cast<double_t>(0.0);
    dpdeg_drho = static_cast<double_t>(0.0);

    // all workitems choose the same path
    if (A_ != 0.0) {
        const auto Binv = 1.0 / B_;
        const auto x = pow_wrapper(rho * Binv, 1.0 / 3.0);
        const auto x_sqr = x * x;
        const auto x_sqr_sqrt = sqrt_wrapper(x_sqr + 1.0);
        const auto x_pow_5 = x_sqr * x_sqr * x;
        hdeg = 8.0 * A_ * Binv * (x_sqr_sqrt - 1.0);
        const double_t edeg_tmp1 = rho * hdeg - pdeg;
        const double_t edeg_tmp2 = 2.4 * A_ * x_pow_5;
        const double_t pdeg_tmp1 = A_ * (x * (2 * x_sqr - 3) * x_sqr_sqrt + 3 * asin_wrapper(x));
        const double_t pdeg_tmp2 = 1.6 * A_ * x_pow_5;
        select_wrapper(edeg, (x > 0.001), edeg_tmp1, edeg_tmp2);
        select_wrapper(pdeg, (x > 0.001), pdeg_tmp1, pdeg_tmp2);
        dpdeg_drho = 8.0 / 3.0 * A_ * Binv * x_sqr / x_sqr_sqrt;
    }
    ek = 0.0;
    for (int dim = 0; dim < NDIM; dim++) {
        ek += UL[sx_i + dim] * UL[sx_i + dim] * rhoinv * 0.5;
    }
    const auto ein2_tmp2 = UL[egas_i] - ek - edeg;
    const auto ein2_mask = (ein2_tmp2 < (physics<NDIM>::de_switch_1 * UL[egas_i]));
    if (!skippable(ein2_mask)) {
        const auto ein2_tmp1 = pow_wrapper(UL[tau_i], physics<NDIM>::fgamma_);
        select_wrapper(ein, ein2_mask, ein2_tmp1, ein2_tmp2);
    } else {
        ein = ein2_tmp2;
    }
    const auto dp_drho2 = dpdeg_drho + (physics<NDIM>::fgamma_ - 1.0) * ein * rhoinv;
    const auto dp_deps2 = (physics<NDIM>::fgamma_ - 1.0) * rho;
    const auto v02 = UL[sx_i + dim] * rhoinv;
    const auto p2 = (physics<NDIM>::fgamma_ - 1.0) * ein + pdeg;
    const auto c2 = sqrt_wrapper(p2 * rhoinv * rhoinv * dp_deps2 + dp_drho2);
    const auto v2 = v02 - vg[dim];
    aml = v2 - c2;
    apl = v2 + c2;

    this_ap = max_wrapper(max_wrapper(apr, apl), double_t(0.0));
    this_am = min_wrapper(min_wrapper(amr, aml), double_t(0.0));
    const auto amp_mask = (this_ap - this_am == 0.0);
#pragma unroll
    for (int f = 0; f < nf_; f++) {
        double_t fr = v * UR[f];
        double_t fl = v2 * UL[f];

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

        if (!skippable(amp_mask)) {
            const double_t flux_tmp1 =
                (this_ap * fl - this_am * fr + this_ap * this_am * (UR[f] - UL[f])) /
                (this_ap - this_am);
            const double_t flux_tmp2 = (fl + fr) / 2.0;
            select_wrapper(this_flux[f], amp_mask, flux_tmp2, flux_tmp1);
        } else {
            this_flux[f] = (this_ap * fl - this_am * fr + this_ap * this_am * (UR[f] - UL[f])) /
                (this_ap - this_am);
        }
    }

    am = min_wrapper(am, this_am);
    ap = max_wrapper(ap, this_ap);
    return max_wrapper(ap, double_t(-am));
}

#pragma GCC pop_options
