//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <cassert>

template<int NDIM>
class roe {

	static constexpr auto HALF = safe_real(0.5);
	static constexpr auto ONE = safe_real(1);
	static constexpr int rho_i = physics<NDIM>::rho_i;
	static constexpr int sx_i = physics<NDIM>::sx_i;
	static constexpr int sy_i = physics<NDIM>::sy_i;
	static constexpr int sz_i = physics<NDIM>::sz_i;
	static constexpr int egas_i = physics<NDIM>::egas_i;
	static constexpr int vx_i = sx_i;
	static constexpr int vy_i = sy_i;
	static constexpr int vz_i = sz_i;
	static constexpr int h_i = egas_i;
	static constexpr auto fgamma = physics<NDIM>::fgamma;
	static constexpr int pot_i = physics<NDIM>::pot_i;
	static constexpr int tau_i = physics<NDIM>::tau_i;
	static constexpr int con_i = physics<NDIM>::rho_i;
	static constexpr int acl_i = physics<NDIM>::sx_i;
	static constexpr int acr_i = physics<NDIM>::sy_i;
	static constexpr int sh1_i = physics<NDIM>::sz_i;
	static constexpr int sh2_i = physics<NDIM>::egas_i;
	static constexpr int nf = physics<NDIM>::field_count();

	oct::vector<oct::vector<safe_real>> roe_primitives(oct::vector<oct::vector<safe_real>>& U) {
		const std::size_t sz = U[0].size();
		oct::vector<oct::vector<safe_real>> V(nf);
		for (int f = 0; f != nf; ++f) {
			V[f].resize(sz);
		}
		for (std::size_t iii = 0; iii != sz; ++iii) {
			const safe_real rho = U[rho_i][iii];
			const safe_real vx = U[sx_i][iii] / rho;
			const safe_real vy = U[sy_i][iii] / rho;
			const safe_real vz = U[sz_i][iii] / rho;
			const safe_real egas = U[egas_i][iii];
			const safe_real tau = U[tau_i][iii];
			const safe_real pot = U[pot_i][iii];
			const safe_real ek = HALF * rho * (vx * vx + vy * vy + vz * vz);
			safe_real ei = egas - ek;
			if (ei < 0.001 * egas) {
				ei = std::pow(tau, fgamma);
			}
			assert(ei > safe_real(0));
			assert(rho > safe_real(0));
			const safe_real p = (fgamma - ONE) * ei;
			const safe_real h = (egas + p) / rho;
			V[rho_i][iii] = rho;
			V[vx_i][iii] = vx;
			V[vy_i][iii] = vy;
			V[vz_i][iii] = vz;
			V[h_i][iii] = h;
			V[tau_i][iii] = tau;
			V[pot_i][iii] = pot;
		}
		return V;
	}

	oct::vector<oct::vector<safe_real>> roe_averages(const oct::vector<oct::vector<safe_real>>& VL,
			const oct::vector<oct::vector<safe_real>>& VR) {
		const std::size_t sz = VR[0].size();
		oct::vector<oct::vector<safe_real>> V_(nf);
		for (int f = 0; f != nf; ++f) {
			V_[f].resize(sz);
		}
		for (std::size_t iii = 0; iii != sz; ++iii) {
			const safe_real wr = std::sqrt(VR[rho_i][iii]);
			const safe_real wl = std::sqrt(VL[rho_i][iii]);
			const safe_real w0 = wr + wl;
			V_[rho_i][iii] = wr * wl;
			V_[vx_i][iii] = (wr * VR[vx_i][iii] + wl * VL[vx_i][iii]) / w0;
			V_[vy_i][iii] = (wr * VR[vy_i][iii] + wl * VL[vy_i][iii]) / w0;
			V_[vz_i][iii] = (wr * VR[vz_i][iii] + wl * VL[vz_i][iii]) / w0;
			V_[h_i][iii] = (wr * VR[h_i][iii] + wl * VL[h_i][iii]) / w0;
			V_[tau_i][iii] = (wr * VR[tau_i][iii] + wl * VL[tau_i][iii]) / w0;
			V_[pot_i][iii] = (wr * VR[pot_i][iii] + wl * VL[pot_i][iii]) / w0;
		}
		return V_;
	}

	safe_real roe_fluxes(oct::vector<oct::vector<safe_real>>& F, oct::vector<oct::vector<safe_real>>& UL,
			oct::vector<oct::vector<safe_real>>& UR, int dimension) {
		const std::size_t sz = UL[0].size();

		auto phi0 = [](safe_real lambda, safe_real delta) {
			if( std::abs(lambda) < delta) {
				return (lambda*lambda + delta*delta)/(safe_real(2.0)*delta);
			} else {
				return std::abs(lambda);
			}
		};

		const int u_i = vx_i + dimension;
		const int v_i = vx_i + (dimension == 0 ? 1 : 0);
		const int w_i = vx_i + (dimension == 2 ? 1 : 2);

		const auto VL = roe_primitives(UL);
		const auto VR = roe_primitives(UR);
		const auto V_ = roe_averages(VL, VR);

		safe_real max_lambda = safe_real(0);

		for (std::size_t iii = 0; iii != sz; ++iii) {

			const safe_real rho_r = VR[rho_i][iii];
			const safe_real u_r = VR[u_i][iii];
			const safe_real v_r = VR[v_i][iii];
			const safe_real w_r = VR[w_i][iii];
			const safe_real h_r = VR[h_i][iii];
			const safe_real tau_r = VR[tau_i][iii];
			const safe_real pot_r = VR[pot_i][iii];
			const safe_real p_r = std::max(
					(fgamma - safe_real(1)) / fgamma * rho_r * (h_r - HALF * (u_r * u_r + v_r * v_r + w_r * w_r)), 1.0e-5);
			assert(rho_r > safe_real(0));
			assert(p_r > safe_real(0));
			const safe_real rho_l = VL[rho_i][iii];
			const safe_real u_l = VL[u_i][iii];
			const safe_real v_l = VL[v_i][iii];
			const safe_real w_l = VL[w_i][iii];
			const safe_real h_l = VL[h_i][iii];
			const safe_real tau_l = VL[tau_i][iii];
			const safe_real pot_l = VL[pot_i][iii];
			const safe_real p_l = std::max(
					(fgamma - safe_real(1)) / fgamma * rho_l * (h_l - HALF * (u_l * u_l + v_l * v_l + w_l * w_l)), 1.0e-5);
			assert(rho_l > safe_real(0));
			assert(p_l > safe_real(0));
			const safe_real rho = V_[rho_i][iii];
			const safe_real u = V_[u_i][iii];
			const safe_real v = V_[v_i][iii];
			const safe_real w = V_[w_i][iii];
			const safe_real h = V_[h_i][iii];
			const safe_real p = std::max((fgamma - safe_real(1)) / fgamma * rho * (h - HALF * (u * u + v * v + w * w)),
					1.0e-5);
			const safe_real tau = V_[tau_i][iii];
			const safe_real pot = V_[pot_i][iii];
			assert(rho > safe_real(0));
			const safe_real c = std::sqrt(fgamma * p / rho);
			assert(p > safe_real(0));
			const safe_real drho = rho_r - rho_l;
			const safe_real du = u_r - u_l;
			const safe_real dv = v_r - v_l;
			const safe_real dw = w_r - w_l;
			const safe_real dp = p_r - p_l;
			const safe_real dtaus = tau_r / rho_r - tau_l / rho_l;
			const safe_real dpots = pot_r / rho_r - pot_l / rho_l;
			const safe_real lambda_con = phi0(u, du);
			const safe_real& lambda_sh1 = lambda_con;
			const safe_real& lambda_sh2 = lambda_con;
			const safe_real& lambda_tau = lambda_con;
			const safe_real& lambda_pot = lambda_con;
			const safe_real lambda_acl = phi0(u - c, du);
			const safe_real lambda_acr = phi0(u + c, du);
			const safe_real f_con = (drho - dp / (c * c)) * lambda_con;
			const safe_real f_acr = (du + dp / (rho * c)) * lambda_acr;
			const safe_real f_acl = (du - dp / (rho * c)) * lambda_acl;
			const safe_real f_sh1 = dv * lambda_sh1;
			const safe_real f_sh2 = dw * lambda_sh2;
			const safe_real f_taus = dtaus * lambda_tau;
			const safe_real f_pots = dpots * lambda_pot;
			const safe_real cinv = c;
			const safe_real f_roe_rho = f_con + HALF * rho * cinv * (f_acr - f_acl);
			const safe_real f_roe_u = u * f_con + HALF * rho * cinv * (f_acr * (u + c) - f_acl * (u - c));
			const safe_real f_roe_v = v * f_roe_rho + rho * f_sh1;
			const safe_real f_roe_w = w * f_roe_rho + rho * f_sh2;
			const safe_real f_roe_egas = f_con * HALF * (u * u + v * v + w * w) + rho * (v * f_sh1 + w * f_sh2)
					+ HALF * rho * cinv * (f_acr * (h + u * c) - f_acl * (h - u * c));
			const safe_real f_roe_tau = (tau / rho) * f_roe_rho + rho * f_taus;
			const safe_real f_roe_pot = (pot / rho) * f_roe_rho + rho * f_pots;
			F[rho_i][iii] = HALF * (rho_r * u_r + rho_l * u_l - f_roe_rho);
			F[u_i][iii] = HALF * (rho_r * u_r * u_r + p_r + rho_l * u_l * u_l + p_l - f_roe_u);
			F[v_i][iii] = HALF * (rho_r * u_r * v_r + rho_l * u_l * v_l - f_roe_v);
			F[w_i][iii] = HALF * (rho_r * u_r * w_r + rho_l * u_l * w_l - f_roe_w);
			F[egas_i][iii] = HALF * (rho_r * u_r * h_r + rho_l * u_l * h_l - f_roe_egas);
			F[tau_i][iii] = HALF * (tau_r * u_r + tau_l * u_l - f_roe_tau);
			F[pot_i][iii] = HALF * (pot_r * u_r + pot_l * u_l - f_roe_pot);
			const safe_real this_max_lambda = std::max(std::max(lambda_acl, lambda_acr), lambda_con);
			max_lambda = std::max(max_lambda, this_max_lambda);
		}

		return max_lambda;
	}
};
