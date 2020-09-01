//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER____FLUX____HPP123
#define OCTOTIGER____FLUX____HPP123

#include "octotiger/unitiger/physics.hpp"
#include "octotiger/unitiger/physics_impl.hpp"

template<int NDIM, int INX, class PHYS>
timestep_t hydro_computer<NDIM, INX, PHYS>::flux(const hydro::state_type &U, const hydro::recon_type<NDIM> &Q, hydro::flux_type &F, hydro::x_type &X,
		safe_real omega) {

	PROFILE();
	// input Q, X
	// output F

	timestep_t ts;
	ts.a = 0.0;
	// bunch of tmp containers
	static thread_local std::vector<safe_real> UR(nf_), UL(nf_), this_flux(nf_);

    // bunch of small helpers
	static const cell_geometry<NDIM, INX> geo;
	static constexpr auto faces = geo.face_pts();
	static constexpr auto weights = geo.face_weight();
	static constexpr auto xloc = geo.xloc();
	static constexpr auto levi_civita = geo.levi_civita();

	const auto dx = X[0][geo.H_DNX] - X[0][0];

	for (int dim = 0; dim < NDIM; dim++) {

		const auto indices = geo.get_indexes(3, geo.face_pts()[dim][0]);

		// zero-initialize F
		for (int f = 0; f < nf_; f++) {
#pragma ivdep
			for (const auto &i : indices) {
				F[dim][f][i] = 0.0;
			}
		}

		for (const auto &i : indices) {
			safe_real ap = 0.0, am = 0.0;
			safe_real this_ap, this_am;
			for (int fi = 0; fi < geo.NFACEDIR; fi++) { // 9
				const auto d = faces[dim][fi];
				// why store this?
				for (int f = 0; f < nf_; f++) { 
					UR[f] = Q[f][d][i];// not cache efficient at all - cacheline is going to be dismissed
					UL[f] = Q[f][geo::flip_dim(d, dim)][i - geo.H_DN[dim]];
				}
				std::array < safe_real, NDIM > x;
				std::array < safe_real, NDIM > vg;
				for (int dim = 0; dim < NDIM; dim++) {
					x[dim] = X[dim][i] + 0.5 * xloc[d][dim] * dx;
				}
				if HOST_CONSTEXPR (NDIM > 1) {
					vg[0] = -omega * (X[1][i] + 0.5 * xloc[d][1] * dx);
					vg[1] = +omega * (X[0][i] + 0.5 * xloc[d][0] * dx);
					if HOST_CONSTEXPR (NDIM == 3) {
						vg[2] = 0.0;
					}
				} else {
					vg[0] = 0.0;
				}

				safe_real amr, apr, aml, apl;
				static thread_local std::vector<safe_real> FR(nf_), FL(nf_);

				PHYS::template physical_flux<INX>(UR, FR, dim, amr, apr, x, vg);
				PHYS::template physical_flux<INX>(UL, FL, dim, aml, apl, x, vg);
				this_ap = std::max(std::max(apr, apl), safe_real(0.0));
				this_am = std::min(std::min(amr, aml), safe_real(0.0));
#pragma ivdep
				for (int f = 0; f < nf_; f++) {
					// this isn't vectorized
					if (this_ap - this_am != 0.0) { 
						this_flux[f] = (this_ap * FL[f] - this_am * FR[f] + this_ap * this_am * (UR[f] - UL[f])) / (this_ap - this_am);
					} else {
						this_flux[f] = (FL[f] + FR[f]) / 2.0;
					}
				}
				am = std::min(am, this_am);
				ap = std::max(ap, this_ap);
#pragma ivdep
				for (int f = 0; f < nf_; f++) {
					// field update from flux
					F[dim][f][i] += weights[fi] * this_flux[f];
				}
			}
			const auto this_amax = std::max(ap, safe_real(-am));
			if (this_amax > ts.a) {
				ts.a = this_amax;
				ts.x = X[0][i];
				ts.y = X[1][i];
				ts.z = X[2][i];
				ts.ur = UL;
				ts.ul = UR;
				ts.dim = dim;
			}
		}
	}
	return ts;
}
template<int NDIM, int INX, class PHYS>
timestep_t hydro_computer<NDIM, INX, PHYS>::flux_experimental(const hydro::state_type &U, const hydro::recon_type<NDIM> &Q, hydro::flux_type &F, hydro::x_type &X,
		safe_real omega) {

	PROFILE();
	// input Q, X
	// output F

	timestep_t ts;
	ts.a = 0.0;
	// bunch of tmp containers
	static thread_local std::vector<double> UR(nf_), UL(nf_), this_flux(nf_);

    // bunch of small helpers
	static const cell_geometry<NDIM, INX> geo;
	static constexpr auto faces = geo.face_pts();
	static constexpr auto weights = geo.face_weight();
	static constexpr auto xloc = geo.xloc();
	static constexpr auto levi_civita = geo.levi_civita();
	double p, v, v0, c;
	const auto A_ = physics<NDIM>::A_;
	const auto B_ = physics<NDIM>::B_;

	const auto dx = X[0][geo.H_DNX] - X[0][0];

	for (int dim = 0; dim < NDIM; dim++) {

		const auto indices = geo.get_indexes(3, geo.face_pts()[dim][0]);

		// zero-initialize F
		for (int f = 0; f < nf_; f++) {
#pragma ivdep
			for (const auto &i : indices) {
				F[dim][f][i] = 0.0;
			}
		}

        for (int fi = 0; fi < geo.NFACEDIR; fi++) {    // 9
            safe_real ap = 0.0, am = 0.0; // final am ap for this i
			safe_real this_ap, this_am; //tmps
			safe_real this_amax;
            for (const auto& i : indices) {
                const auto d = faces[dim][fi];

				std::array < safe_real, NDIM > x;
				std::array < safe_real, NDIM > vg;
				for (int dim = 0; dim < NDIM; dim++) {
					x[dim] = X[dim][i] + 0.5 * xloc[d][dim] * dx;
				}
				vg[0] = -omega * (X[1][i] + 0.5 * xloc[d][1] * dx);
				vg[1] = +omega * (X[0][i] + 0.5 * xloc[d][0] * dx);
				vg[2] = 0.0;

				// why store this?
				const auto flipped_dim = geo::flip_dim(d, dim);
				for (int f = 0; f < nf_; f++) { 
					UR[f] = Q[f][d][i];// not cache efficient at all - cacheline is going to be dismissed
					UL[f] = Q[f][flipped_dim][i - geo.H_DN[dim]];
				}

				safe_real amr, apr, aml, apl;
				static thread_local std::vector<safe_real> FR(nf_), FL(nf_);

                auto rho = UR[rho_i];
                auto rhoinv = (1.) / rho;
                double hdeg = 0.0, pdeg = 0.0, edeg = 0.0, dpdeg_drho = 0.0;

                // all workitems choose the same path
                if (A_ != 0.0) {
                    const auto Binv = 1.0 / B_;
                    const auto x = std::pow(rho * Binv, 1.0 / 3.0);
                    const auto x_sqr = x * x;
                    const auto x_sqr_sqrt = std::sqrt(x_sqr + 1.0);
                    const auto x_pow_5 = x_sqr * x_sqr * x;
                    hdeg = 8.0 * A_ * Binv * (x_sqr_sqrt - 1.0);
                    if (x < 0.001) {
                        pdeg = 1.6 * A_ * x_pow_5;
                    } else {
                        pdeg = A_ * (x * (2 * x_sqr - 3) * x_sqr_sqrt + 3 * asinh(x));
                    }
                    if (x > 0.001) {
                        edeg = rho * hdeg - pdeg;
                    } else {
                        edeg = 2.4 * A_ * x_pow_5;
                    }
                    dpdeg_drho = 8.0 / 3.0 * A_ * Binv * x_sqr / x_sqr_sqrt;
                }
                double ek = 0.0;
                for (int dim = 0; dim < NDIM; dim++) {
                    ek += UR[sx_i + dim] * UR[sx_i + dim] * rhoinv * 0.5;
                }
                auto ein = UR[egas_i] - ek - edeg;
                if (ein < physics<NDIM>::de_switch_1 * UR[egas_i]) {
                    ein = pow(UR[tau_i], physics<NDIM>::fgamma_);
                }
                double dp_drho = dpdeg_drho + (physics<NDIM>::fgamma_ - 1.0) * ein * rhoinv;
                double dp_deps = (physics<NDIM>::fgamma_ - 1.0) * rho;
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
                    if (x < 0.001) {
                        pdeg = 1.6 * A_ * x_pow_5;
                    } else {
                        pdeg = A_ * (x * (2 * x_sqr - 3) * x_sqr_sqrt + 3 * asinh(x));
                    }
                    if (x > 0.001) {
                        edeg = rho * hdeg - pdeg;
                    } else {
                        edeg = 2.4 * A_ * x_pow_5;
                    }
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

                this_ap = std::max(std::max(apr, apl), safe_real(0.0));
				this_am = std::min(std::min(amr, aml), safe_real(0.0));
                if (this_ap - this_am != 0.0) {
#pragma ivdep
                    for (int f = 0; f < nf_; f++) {
                        this_flux[f] = (this_ap * FL[f] - this_am * FR[f] +
                                           this_ap * this_am * (UR[f] - UL[f])) /
                            (this_ap - this_am);
                    }
                } else {
#pragma ivdep
                    for (int f = 0; f < nf_; f++) {
                        this_flux[f] = (FL[f] + FR[f]) / 2.0;
                    }
                }
                am = std::min(am, this_am);
				ap = std::max(ap, this_ap);
				this_amax = std::max(ap, safe_real(-am));
				if (this_amax > ts.a) {
					ts.a = this_amax;
					ts.x = X[0][i];
					ts.y = X[1][i];
					ts.z = X[2][i];
					ts.ur = UL;
					ts.ul = UR;
					ts.dim = dim;
				}
#pragma ivdep
				for (int f = 0; f < nf_; f++) {
					// field update from flux
					F[dim][f][i] += weights[fi] * this_flux[f];
				}
            }
        }
    }
	return ts;
}
#endif
