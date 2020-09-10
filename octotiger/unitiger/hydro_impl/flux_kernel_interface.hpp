#include <array>
#include <vector>

#include "octotiger/hydro_defs.hpp"
#include "octotiger/unitiger/hydro.hpp"
#include "octotiger/unitiger/safe_real.hpp"

timestep_t flux_kernel_interface(const hydro::recon_type<NDIM>& Q, hydro::flux_type& F,
    hydro::x_type& X, safe_real omega, const size_t nf_);

timestep_t flux_cpu_kernel(const hydro::recon_type<NDIM>& Q, hydro::flux_type& F, hydro::x_type& X,
    safe_real omega, const size_t nf_);

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

template <typename double_t>
inline double_t inner_flux_loop(const double omega, const size_t nf_, const double A_,
    const double B_, const std::vector<double_t>& UR, const std::vector<double_t>& UL,
    std::vector<double_t>& FR, std::vector<double_t>& FL, std::vector<double_t>& this_flux,
    const std::array<double_t, NDIM> x, const std::array<double_t, NDIM>& vg, double_t& ap,
    double_t& am, const size_t dim, const size_t d, const size_t i, const cell_geometry<3, 8> geo,
    const double dx) {
    thread_local constexpr auto xloc = geo.xloc();
    thread_local constexpr auto levi_civita = geo.levi_civita();
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
        // std::cout << "select x0001 1" << std::endl;
        select_wrapper(edeg, (x > 0.001), edeg_tmp1, edeg_tmp2);
        select_wrapper(pdeg, (x > 0.001), pdeg_tmp1, pdeg_tmp2);

        dpdeg_drho = 8.0 / 3.0 * A_ * Binv * x_sqr / x_sqr_sqrt;
        std::cin.get();
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
    // std::cout << "v0 vg" << v0 << " " << vg[dim] << std::endl;
    // std::cout << "c v" << c << " " << v << std::endl;
    amr = v - c;
    apr = v + c;
#pragma unroll
    for (int f = 0; f < nf_; f++) {
        FR[f] = v * UR[f];
    }
    FR[sx_i + dim] += p;
    FR[egas_i] += v0 * p;
    for (int n = 0; n < geo.NANGMOM; n++) {
        for (int m = 0; m < NDIM; m++) {
            FR[lx_i + n] += levi_civita[n][m][dim] * x[m] * p;
        }
    }

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
        // std::cout << "select x0001 1" << std::endl;
        select_wrapper(edeg, (x > 0.001), edeg_tmp1, edeg_tmp2);
        select_wrapper(pdeg, (x > 0.001), pdeg_tmp1, pdeg_tmp2);
        dpdeg_drho = 8.0 / 3.0 * A_ * Binv * x_sqr / x_sqr_sqrt;
        std::cin.get();
    }
    ek = 0.0;
    for (int dim = 0; dim < NDIM; dim++) {
        ek += UL[sx_i + dim] * UL[sx_i + dim] * rhoinv * 0.5;
    }
    const auto ein2_tmp2 = UL[egas_i] - ek - edeg;
    const auto ein2_mask = (ein2_tmp2 < (physics<NDIM>::de_switch_1 * UL[egas_i]));
    // if (ein < physics<NDIM>::de_switch_1 * UL[egas_i]) {
    //    ein = pow_wrapper(UL[tau_i], physics<NDIM>::fgamma_);
    //}
    if (!skippable(ein2_mask)) {
        const auto ein2_tmp1 = pow_wrapper(UL[tau_i], physics<NDIM>::fgamma_);
        select_wrapper(
            ein, ein2_mask, ein2_tmp1, ein2_tmp2);
    } else {
        ein = ein2_tmp2;
    }
    dp_drho = dpdeg_drho + (physics<NDIM>::fgamma_ - 1.0) * ein * rhoinv;
    dp_deps = (physics<NDIM>::fgamma_ - 1.0) * rho;
    v0 = UL[sx_i + dim] * rhoinv;
    p = (physics<NDIM>::fgamma_ - 1.0) * ein + pdeg;
    c = sqrt_wrapper(p * rhoinv * rhoinv * dp_deps + dp_drho);
    v = v0 - vg[dim];
    aml = v - c;
    apl = v + c;
#pragma unroll
    for (int f = 0; f < nf_; f++) {
        FL[f] = v * UL[f];
    }
    FL[sx_i + dim] += p;
    FL[egas_i] += v0 * p;
    for (int n = 0; n < geo.NANGMOM; n++) {
        for (int m = 0; m < NDIM; m++) {
            FL[lx_i + n] += levi_civita[n][m][dim] * x[m] * p;
        }
    }
    // std::cout << apr << "apr---apl" << apl << std::endl;
    // std::cout << amr << "amr---aml" << aml << std::endl;
    this_ap = max_wrapper(max_wrapper(apr, apl), double_t(0.0));
    this_am = min_wrapper(min_wrapper(amr, aml), double_t(0.0));
    // std::cout << this_ap << "ap---am" << this_am << std::endl;
    for (int f = 0; f < nf_; f++) {
        const double_t flux_tmp1 =
            (this_ap * FL[f] - this_am * FR[f] + this_ap * this_am * (UR[f] - UL[f])) /
            (this_ap - this_am);
        const double_t flux_tmp2 = (FL[f] + FR[f]) / 2.0;
        select_wrapper(this_flux[f], (this_ap - this_am != 0.0), flux_tmp1, flux_tmp2);
    }
    // TODO This loooks wrong! ... Does it?
    am = min_wrapper(am, this_am);
    ap = max_wrapper(ap, this_ap);
    return max_wrapper(ap, double_t(-am));
}
