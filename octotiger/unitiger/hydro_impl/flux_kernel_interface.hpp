#include <array>
#include <vector>

#include "octotiger/hydro_defs.hpp"
#include "octotiger/unitiger/hydro.hpp"
#include "octotiger/unitiger/safe_real.hpp"

timestep_t flux_kernel_interface(const hydro::recon_type<NDIM>& Q, hydro::flux_type& F,
    hydro::x_type& X, safe_real omega, const size_t nf_);



// helpers for using vectortype specialization functions
template<typename cond_t, typename double_t>
inline double select_wrapper(const cond_t &cond, const double_t &tmp1, const double_t &tmp2) {
  return cond ? tmp1 : tmp2;
}
template<typename T>
inline T max_wrapper(const T& tmp1, const T& tmp2) {
  return std::max(tmp1, tmp2);
}
template<typename T>
inline T min_wrapper(const T& tmp1, const T& tmp2) {
  return std::min(tmp1, tmp2);
}

template <typename double_t>
inline double_t inner_flux_loop(hydro::x_type& X, double omega, const size_t nf_,
    const double_t A_, const double_t B_, std::vector<double_t>& UR, std::vector<double_t>& UL,
    std::vector<double_t>& FR, std::vector<double_t>& FL, std::vector<double_t>& this_flux,
    std::array<double_t, NDIM> x, std::array<double_t, NDIM>& vg, double_t& ap, double_t& am,
    size_t dim, size_t d, size_t i, const cell_geometry<3, 8> geo, const double dx) {
    thread_local constexpr auto xloc = geo.xloc();
    thread_local constexpr auto levi_civita = geo.levi_civita();
    double_t amr, apr, aml, apl;
    double_t this_ap, this_am;    // tmps
    double_t p, v, v0, c;

    for (int dim = 0; dim < NDIM; dim++) {
        x[dim] = X[dim][i] + 0.5 * xloc[d][dim] * dx;
    }
    vg[0] = -omega * (X[1][i] + 0.5 * xloc[d][1] * dx);
    vg[1] = +omega * (X[0][i] + 0.5 * xloc[d][0] * dx);
    vg[2] = 0.0;
    auto rho = UR[rho_i];
    auto rhoinv = (1.) / rho;
    double_t hdeg = 0.0, pdeg = 0.0, edeg = 0.0, dpdeg_drho = 0.0;

    // all workitems choose the same path
    if (A_ != 0.0) {
        const auto Binv = 1.0 / B_;
        const auto x = std::pow(rho * Binv, 1.0 / 3.0);
        const auto x_sqr = x * x;
        const auto x_sqr_sqrt = std::sqrt(x_sqr + 1.0);
        const auto x_pow_5 = x_sqr * x_sqr * x;
        hdeg = 8.0 * A_ * Binv * (x_sqr_sqrt - 1.0);

        const double_t edeg_tmp1 = rho * hdeg - pdeg;
        const double_t edeg_tmp2 = 2.4 * A_ * x_pow_5;
        const double_t pdeg_tmp1 = A_ * (x * (2 * x_sqr - 3) * x_sqr_sqrt + 3 * asinh(x));
        const double_t pdeg_tmp2 = 1.6 * A_ * x_pow_5;
        edeg = select_wrapper((x > 0.001), edeg_tmp1, edeg_tmp2);
        pdeg = select_wrapper((x > 0.001), pdeg_tmp1, pdeg_tmp2);

        dpdeg_drho = 8.0 / 3.0 * A_ * Binv * x_sqr / x_sqr_sqrt;
    }
    double_t ek = 0.0;
    for (int dim = 0; dim < NDIM; dim++) {
        ek += UR[sx_i + dim] * UR[sx_i + dim] * rhoinv * 0.5;
    }
    auto ein = UR[egas_i] - ek - edeg;
    if (ein < physics<NDIM>::de_switch_1 * UR[egas_i]) {
        ein = pow(UR[tau_i], physics<NDIM>::fgamma_);
    }
    double_t dp_drho = dpdeg_drho + (physics<NDIM>::fgamma_ - 1.0) * ein * rhoinv;
    double_t dp_deps = (physics<NDIM>::fgamma_ - 1.0) * rho;
    v0 = UR[sx_i + dim] * rhoinv;
    p = (physics<NDIM>::fgamma_ - 1.0) * ein + pdeg;
    c = std::sqrt(p * rhoinv * rhoinv * dp_deps + dp_drho);
    v = v0 - vg[dim];
    amr = v - c;
    apr = v + c;
#pragma ivdep
    for (int f = 0; f < nf_; f++) {
        FR[f] = v * UR[f];
    }
    FR[sx_i + dim] += p;
    FR[egas_i] += v0 * p;
    for (int n = 0; n < geo.NANGMOM; n++) {
#pragma ivdep
        for (int m = 0; m < NDIM; m++) {
            FR[lx_i + n] += levi_civita[n][m][dim] * x[m] * p;
        }
    }

    rho = UL[rho_i];
    rhoinv = (1.) / rho;
    hdeg = 0.0, pdeg = 0.0, edeg = 0.0, dpdeg_drho = 0.0;

    // all workitems choose the same path
    if (A_ != 0.0) {
        const auto Binv = 1.0 / B_;
        const auto x = std::pow(rho * Binv, 1.0 / 3.0);
        const auto x_sqr = x * x;
        const auto x_sqr_sqrt = std::sqrt(x_sqr + 1.0);
        const auto x_pow_5 = x_sqr * x_sqr * x;
        hdeg = 8.0 * A_ * Binv * (x_sqr_sqrt - 1.0);
        const double_t edeg_tmp1 = rho * hdeg - pdeg;
        const double_t edeg_tmp2 = 2.4 * A_ * x_pow_5;
        const double_t pdeg_tmp1 = A_ * (x * (2 * x_sqr - 3) * x_sqr_sqrt + 3 * asinh(x));
        const double_t pdeg_tmp2 = 1.6 * A_ * x_pow_5;
        edeg = select_wrapper((x > 0.001), edeg_tmp1, edeg_tmp2);
        pdeg = select_wrapper((x > 0.001), pdeg_tmp1, pdeg_tmp2);
        dpdeg_drho = 8.0 / 3.0 * A_ * Binv * x_sqr / x_sqr_sqrt;
    }
    ek = 0.0;
    for (int dim = 0; dim < NDIM; dim++) {
        ek += UL[sx_i + dim] * UL[sx_i + dim] * rhoinv * 0.5;
    }
    ein = UL[egas_i] - ek - edeg;
    if (ein < physics<NDIM>::de_switch_1 * UL[egas_i]) {
        ein = pow(UL[tau_i], physics<NDIM>::fgamma_);
    }
    dp_drho = dpdeg_drho + (physics<NDIM>::fgamma_ - 1.0) * ein * rhoinv;
    dp_deps = (physics<NDIM>::fgamma_ - 1.0) * rho;
    v0 = UL[sx_i + dim] * rhoinv;
    p = (physics<NDIM>::fgamma_ - 1.0) * ein + pdeg;
    c = std::sqrt(p * rhoinv * rhoinv * dp_deps + dp_drho);
    v = v0 - vg[dim];
    aml = v - c;
    apl = v + c;
#pragma ivdep
    for (int f = 0; f < nf_; f++) {
        FL[f] = v * UL[f];
    }
    FL[sx_i + dim] += p;
    FL[egas_i] += v0 * p;
    for (int n = 0; n < geo.NANGMOM; n++) {
#pragma ivdep
        for (int m = 0; m < NDIM; m++) {
            FL[lx_i + n] += levi_civita[n][m][dim] * x[m] * p;
        }
    }
    this_ap = max_wrapper(max_wrapper(apr, apl), double_t(0.0));
    this_am = min_wrapper(min_wrapper(amr, aml), double_t(0.0));
    for (int f = 0; f < nf_; f++) {
        const double_t flux_tmp1 =
            (this_ap * FL[f] - this_am * FR[f] + this_ap * this_am * (UR[f] - UL[f])) /
            (this_ap - this_am);
        const double_t flux_tmp2 = (FL[f] + FR[f]) / 2.0;
        this_flux[f] = (this_ap - this_am != 0) ? flux_tmp1 : flux_tmp2;
    }
    am = min_wrapper(am, this_am);
    ap = max_wrapper(ap, this_ap);
    return max_wrapper(ap, double_t(-am));
}
