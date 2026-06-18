//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/radiation/rad_grid.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/math/AutoDiff.hpp"
#include "octotiger/math/Matrix.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/radiation/implicit.hpp"
#include "octotiger/radiation/opacities.hpp"
#include "octotiger/roe.hpp"
#include "octotiger/space_vector.hpp"

#include <hpx/include/future.hpp>

#include "octotiger/unitiger/radiation/radiation_physics_impl.hpp"
#include <cmath>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
// ΑαΒβΔδΕεΦφΓγΗηΙιΚκΛλΜμΝνΟοΠπΡρΣσςΤτΥυΧχΨψΩωΖζΘθΞξ

void node_server::compute_radiation(Real dt, Real omega) {
	constexpr auto maxSubstepCount = std::numeric_limits<int>::max();
	constexpr auto gam = 1_R - inv(std::numbers::sqrt2_v<Real>);
	static thread_local FpeGuard fpeGuard{};
	auto const c = physcon().c;
	auto const hydroOn = opts().hydro;
	auto const cflFactor = opts().cfl;
	auto const scale = opts().xscale;
	auto &rgrid = *rad_grid_ptr;
	auto &hydro = grid_ptr->data();
	rgrid.set_dx(grid_ptr->get_dx());
	rgrid.set_X(grid_ptr->get_X());
	auto const minDx = 2_R * scale / Real(INX << opts().max_level);
	auto const substepCount = std::max(int(std::ceil(c * dt * inv(cflFactor * minDx))), 1);
	if (substepCount > maxSubstepCount) throw std::runtime_error(print2string("Number of substeps greater than %i.\n", maxSubstepCount));
	auto const bounds = [&]() {
		all_rad_bounds();
		if (hydroOn) {
			all_hydro_bounds();
			rgrid.computeMaterialProperties(hydro);
		}
	};
	bounds();
	auto const substep1 = [this, &rgrid, &hydro, &bounds](Real dt) {
		rgrid.store();
		rgrid.computeFlux();
		GET(exchange_rad_flux_corrections());
		rgrid.applyFlux(1_R, dt);
		auto const source = rgrid.computeSource(hydro, dt);
		rgrid.applySource(hydro, source, dt);
		bounds();
	};
	auto const substep2 = [this, &rgrid, &hydro, &bounds](Real dt) {
		rgrid.store();
		rgrid.computeFlux();
		GET(exchange_rad_flux_corrections());
		rgrid.applyFlux(1_R, dt);
		auto dRdt_src = rgrid.computeSource(hydro, dt);
		rgrid.applySource(hydro, dRdt_src, dt);
		bounds();
		rgrid.computeFlux();
		GET(exchange_rad_flux_corrections());
		rgrid.applySource(hydro, dRdt_src, -dt);
		rgrid.applyFlux(0.5_R, dt);
		dRdt_src = rgrid.computeSource(hydro, dt);
		rgrid.applySource(hydro, dRdt_src, dt);
		bounds();
	};
	for (int i = 0; i < substepCount; i++) {
		substep2(dt / substepCount);
	}
}

void rad_grid::computeFlux() {
	using std::abs;
	using std::copysign;
	using std::max;
	using std::min;
	constexpr bool useLF = false;
	constexpr bool useHLL = true;
	static_assert(useLF != useHLL);
	static_assert(useLF || useHLL);
	FpeGuard fpeGuard{};
	auto const minmod_theta = [](Real a, Real b) {
		auto const minmod = [](Real a, Real b) {
			return (copysign(0.5_R, a) + copysign(0.5_R, b)) * min(abs(a), abs(b));
		};
		constexpr Real theta = 1.3_R;
		return minmod(0.5_R * (a + b), theta * minmod(a, b));
	};
	auto const c = physcon().c;
	auto const c2 = sqr(c);
	auto const ic = inv(c);
	std::vector<Real> E(RAD_N3);
	std::vector<Real> tau(RAD_N3);
	std::vector<Real> H(RAD_N3);
	std::vector<Real> Hp(RAD_N3);
	std::vector<Real> Hm(RAD_N3);
	std::vector<Vector<Real, NDIM>> F(RAD_N3);
	std::vector<Vector<Real, NDIM>> beta(RAD_N3);
	std::vector<Vector<Real, NDIM>> beta_p(RAD_N3);
	std::vector<Vector<Real, NDIM>> beta_m(RAD_N3);
	for (auto i = 0; i < RAD_N3; i++) {
		E[i] = U[er_i][i];
		for (auto d = 0; d < NDIM; d++) {
			F[i][d] = U[fx_i + d][i] * ic;
		}
	}
	for (auto i = 0; i < RAD_N3; i++) {
		ASSERT_POSITIVE(E[i]);
		auto const F2 = F[i].dot(F[i]);
		auto const E2 = E[i] * E[i];
		ASSERT_RANGE(0_R, F2, E2);
		H[i] = (1_R / 3_R) * (2_R * E[i] + sqrt(4_R * E2 - 3_R * F2));
		ASSERT_POSITIVE(H[i]);
		auto const iH = inv(H[i]);
		beta[i] = F[i] * iH;
		ASSERT_POSITIVE(chi[i]);
		tau[i] = chi[i] * dx;
	}
	auto const fluxes = [](Real const &H, Vector<Real, NDIM> const &beta, int k) {
		auto const fE = H * beta[k];
		auto fF = H * beta * beta[k];
		fF[k] += 0.25_R * (1_R - beta.dot(beta)) * H;
		return std::pair<Real, Vector<Real, NDIM>>(fE, fF);
	};
	for (auto dir = 0; dir < NDIM; dir++) {
		auto const dn = R_DN[dir];
		for (auto i = dn; i < RAD_N3 - dn; i++) {
			auto const dH = minmod_theta(H[i + dn] - H[i], H[i] - H[i - dn]);
			Hm[i] = Hp[i] = H[i];
			Hp[i] += 0.5_R * dH;
			Hm[i] -= 0.5_R * dH;
			for (auto d = 0; d < NDIM; d++) {
				auto const dbeta = minmod_theta(beta[i + dn][d] - beta[i][d], beta[i][d] - beta[i - dn][d]);
				beta_p[i][d] = beta_m[i][d] = beta[i][d];
				beta_p[i][d] += 0.5_R * dbeta;
				beta_m[i][d] -= 0.5_R * dbeta;
			}
			ASSERT_RANGE(0_R, beta_p[i].dot(beta_p[i]), 1_R);
			ASSERT_RANGE(0_R, beta_m[i].dot(beta_m[i]), 1_R);
		}
		for (auto i = 2 * dn; i < RAD_N3 - dn; i++) {
			Real fE;
			Vector<Real, NDIM> fF;
			auto const Hr = Hm[i];
			auto const Hl = Hp[i - dn];
			auto const beta_r = beta_m[i];
			auto const beta_l = beta_p[i - dn];
			auto const tau_r = tau[i];
			auto const tau_l = tau[i - dn];
			auto const beta_2r = beta_r.dot(beta_r);
			auto const beta_2l = beta_l.dot(beta_l);
			auto const [fEl, fFl] = fluxes(Hl, beta_l, dir);
			auto const [fEr, fFr] = fluxes(Hr, beta_r, dir);
			auto const Fr = beta_r * Hr;
			auto const Fl = beta_l * Hl;
			auto const Er = (0.75_R + 0.25_R * beta_2r) * Hr;
			auto const El = (0.75_R + 0.25_R * beta_2l) * Hl;
			auto const Xr = sqrt((1_R - beta_2r) * (3_R - beta_2r - 2_R * sqr(beta_r[dir])));
			auto const Xl = sqrt((1_R - beta_2l) * (3_R - beta_2l - 2_R * sqr(beta_l[dir])));
			auto const λ_pr = (2_R * beta_r[dir] + Xr) / (3_R - beta_2r);
			auto const λ_pl = (2_R * beta_l[dir] + Xl) / (3_R - beta_2l);
			auto const λ_mr = (2_R * beta_r[dir] - Xr) / (3_R - beta_2r);
			auto const λ_ml = (2_R * beta_l[dir] - Xl) / (3_R - beta_2l);
			auto const λ_diff = 4_R / (3_R * max(tau_r, tau_l));
			if constexpr (useLF) {
				auto const λ_r = max(0_R, max(λ_pr, λ_pl));
				auto const λ_l = min(0_R, min(λ_mr, λ_ml));
				auto const λ_max = min(λ_diff, max(+λ_r, -λ_l));
				fE = 0.5_R * ((fEl + fEr) - λ_max * (Er - El));
				fF = 0.5_R * ((fFl + fFr) - λ_max * (Fr - Fl));
			} else if constexpr (useHLL) {
				auto const λ_r = max(0_R, min(+λ_diff, max(λ_pr, λ_pl)));
				auto const λ_l = min(0_R, max(-λ_diff, min(λ_mr, λ_ml)));
				auto const iλ = inv(λ_r - λ_l);
				fE = (λ_r * fEl - λ_l * fEr + (λ_r * λ_l * (Er - El))) * iλ;
				fF = (λ_r * fFl - λ_l * fFr + (λ_r * λ_l * (Fr - Fl))) * iλ;
			}
			flux[dir][er_i][i] = c * fE;
			for (auto d = 0; d < NDIM; d++) {
				flux[dir][fx_i + d][i] = c2 * fF[d];
			}
		}
	}
}

void rad_grid::applyFlux(Real β, Real dt) {
	constexpr auto lb = RAD_BW;
	constexpr auto ub = RAD_NX - RAD_BW;
	constexpr auto di = Vector<int, NDIM>({DX, DY, DZ});
	auto const βm1 = β - 1_R;
	auto const λ = dt / dx;
	for (int field = 0; field < NRF; field++) {
		Vector<std::span<Real const>, NDIM> f;
		for (int d = 0; d < NDIM; d++) {
			f[d] = std::span(flux[d][field].begin(), flux[d][field].end());
		}
		auto const &u0 = U0[field];
		auto &u = U[field];
		for (int j = lb; j < ub; j++) {
			for (int k = lb; k < ub; k++) {
				for (int l = lb; l < ub; l++) {
					auto const i = rindex(j, k, l);
					Real du = 0_R;
					for (int d = 0; d < NDIM; d++) {
						du -= λ * (f[d][i + di[d]] - f[d][i]);
					}
					u[i] += βm1 * (u[i] - u0[i]) + β * du;
				}
			}
		}
	}
}

std::vector<RadiationSource> rad_grid::computeSource(HydroState &hydro, Real dt) {
	PROFILE()
	auto const cellCompute = [](Real E0, Vector<Real, NDIM> F0, Real τ0, Vector<Real, NDIM> S0, Real ρ, Real μ, Real κ, Real χ, Real dt) {
		RadiationSource source;
		auto const c = physcon().c;
		auto const ic = inv(c);
		auto const c2 = sqr(c);
		auto const kB = physcon().kb;
		auto const aR = physcon().sigma * 4_R * ic;
		auto const m = physcon().mh;
		auto const Γ = grid::get_fgamma();
		auto const ic2 = sqr(ic);
		auto const Rg = m * inv((Γ - 1_R) * kB);

		auto const iρ = inv(ρ);
		auto const e0 = std::pow(τ0, Γ);
		auto const Eg0 = e0 + 0.5_R * iρ * S0.dot(S0);
		auto const β = iρ * ic * S0;
		auto const σ = χ - κ;
		auto const A = aR * sqr(sqr(iρ * μ * Rg));
		auto const tol2 = sqr(e0 + E0) * eps_R;
		Real dx;
		Real x = 0_R;
		Vector<Real, NDIM> F;
		bool &converged = source.converged;
		auto &cnt = source.iters;
		do {
			auto const E = E0 + x;
			auto const e = e0 - x;
			auto const B = A * sqr(sqr(e));
			auto const dBdx = -4_R * A * e * sqr(e);
			F = (F0 + dt * c2 * (β * (σ * E + κ * B))) * inv(1_R + dt * c * χ);
			auto const dFdx = dt * c2 * β * (σ + κ * dBdx) * inv(1_R + dt * c * χ);
			auto const f = x + dt * (c * κ * (E - B) + (σ - κ) * F.dot(β));
			auto const dfdx = 1_R + dt * (c * κ * (1_R - dBdx) + (σ - κ) * dFdx.dot(β));
			dx = -f * inv(dfdx);
			x += dx;
			cnt++;
			converged = (sqr(dx) < tol2);
		} while (!converged && (cnt < 20));
		auto const E = E0 + x;
		auto const e = e0 - x;
		auto const τ = pow(e, inv(Γ));
		auto const S = S0 + ic2 * (F0 - F);
		auto const Eg = e + 0.5_R * iρ * S.dot(S);
		source.dEr_dt = (E - E0) * inv(dt);
		source.dEg_dt = (Eg - Eg0) * inv(dt);
		source.dFr_dt = (F - F0) * inv(dt);
		source.dS_dt = (S - S0) * inv(dt);
		source.dtau_dt = (τ - τ0) * inv(dt);
		return source;
	};

	const int off = H_BW - RAD_BW;
	auto const fgamma = grid::get_fgamma();
	Opacities const opacity{};

	std::vector<RadiationSource> sources;
	sources.resize(RAD_NX * RAD_NX * RAD_NX);

	for (int xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
		for (int yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
			for (int zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
				const int ir = rindex(xi, yi, zi);
				const int ih = hindex(xi + off, yi + off, zi + off);
				Vector<Real, NDIM> S0, Fphys0;
				Real rho = 0_R;
				for (auto k = 0; k < opts().n_species; k++) {
					rho += hydro[spc_i + k][ih];
				}
				for (auto d = 0; d < NDIM; d++) {
					Fphys0[d] = U[fx_i + d][ir];
					S0[d] = hydro[sx_i + d][ih];
				}
				sources[ir] = cellCompute(U[er_i][ir], Fphys0, hydro[tau_i][ih], S0, rho, mmw[ir], kappa[ir], chi[ir], dt);
			}
		}
	}

	return sources;
}

void rad_grid::applySource(HydroState &hydro, std::vector<RadiationSource> const &sources, Real dt) {
	PROFILE()

	const int off = H_BW - RAD_BW;

	for (int xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
		for (int yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
			for (int zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
				const int ir = rindex(xi, yi, zi);
				const int ih = hindex(xi + off, yi + off, zi + off);

				auto const &source = sources[ir];

				U[er_i][ir] += dt * source.dEr_dt;

				for (auto d = 0; d < NDIM; d++) {
					U[fx_i + d][ir] += dt * source.dFr_dt[d];
				}
				if (opts().hydro) {
					hydro[egas_i][ih] += dt * source.dEg_dt;
					hydro[tau_i][ih] += dt * source.dtau_dt;
					for (auto d = 0; d < NDIM; d++) {
						hydro[sx_i + d][ih] += dt * source.dS_dt[d];
					}
				}
			}
		}
	}
}

void rad_grid::computeMaterialProperties(const std::vector<std::vector<Real>> &hydro) {
	auto const dualEnergySwitch = opts().dual_energy_sw1;
	auto const speciesCount = opts().n_species;
	auto const &atomicMasses = opts().atomic_mass;
	auto const &atomicNumbers = opts().atomic_number;
	auto const gamma = grid::get_fgamma();
	mmw.resize(RAD_N3);
	chi.resize(RAD_N3);
	kappa.resize(RAD_N3);
	Opacities const opacity{};
	for (int i = 0; i != RAD_NX; ++i) {
		for (int j = 0; j != RAD_NX; ++j) {
			for (int k = 0; k != RAD_NX; ++k) {
				const int d = H_BW - RAD_BW;
				const int ir = rindex(i, j, k);
				const int ih = hindex(i + d, j + d, k + d);
				Real immw = 0_R;
				for (int i = 0; i != speciesCount; ++i) {
					auto const X = hydro[spc_i + i][ih];
					auto const A = atomicMasses[i];
					auto const Z = atomicNumbers[i];
					immw += X * (Z + 1_R) / A;
				}
				mmw[ir] = 1.0 / immw;
				auto const rho = hydro[rho_i][ih];
				ASSERT_POSITIVE(rho);
				auto const s = Vector<Real, NDIM>({hydro[sx_i][ih], hydro[sy_i][ih], hydro[sz_i][ih]});
				auto const e = hydro[egas_i][ih];
				auto const ek = s.dot(s) / (2_R * rho);
				auto const tau = hydro[tau_i][ih];
				auto ei = e - ek;
				ei = (ei < e * dualEnergySwitch) ? std::pow(tau, gamma) : ei;
				kappa[ir] = opacity.absorption(hydro[rho_i][ih], ei, mmw[ir]);
				chi[ir] = opacity.extinction(hydro[rho_i][ih], ei, mmw[ir]);
			}
		}
	}
}

std::unordered_map<std::string, int> rad_grid::str_to_index;
std::unordered_map<int, std::string> rad_grid::index_to_str;

void rad_grid::static_init() {
	str_to_index["er"] = er_i;
	str_to_index["fx"] = fx_i;
	str_to_index["fy"] = fy_i;
	str_to_index["fz"] = fz_i;
	for (const auto &s : str_to_index) {
		index_to_str[s.second] = s.first;
	}
}

std::vector<std::string> rad_grid::get_field_names() {
	std::vector<std::string> rc;
	for (auto i : str_to_index) {
		rc.push_back(i.first);
	}
	return rc;
}

void rad_grid::set(const std::string name, Real *data) {
	auto iter = str_to_index.find(name);
	Real eunit = opts().problem == MARSHAK ? 1 : opts().code_to_g / std::pow(opts().code_to_s, 2) / opts().code_to_cm;
	Real funit = opts().problem == MARSHAK ? 1 : eunit * opts().code_to_cm / opts().code_to_s;
	if (iter != str_to_index.end()) {
		int f = iter->second;
		int jjj = 0;
		for (int i = 0; i < INX; i++) {
			for (int j = 0; j < INX; j++) {
				for (int k = 0; k < INX; k++) {
					const int iii = rindex(k + RAD_BW, j + RAD_BW, i + RAD_BW);
					data[jjj] /= f == er_i ? eunit : funit;
					U[f][iii] = data[jjj];
					jjj++;
				}
			}
		}
	}
}

std::vector<silo_var_t> rad_grid::var_data() const {
	std::vector<silo_var_t> s;
	Real eunit = opts().problem == MARSHAK ? 1 : opts().code_to_g / std::pow(opts().code_to_s, 2) / opts().code_to_cm;
	Real funit = opts().problem == MARSHAK ? 1 : eunit * opts().code_to_cm / opts().code_to_s;
	for (auto l : str_to_index) {
		const int f = l.second;
		std::string this_name = l.first;
		int jjj = 0;
		silo_var_t this_s(this_name);
		for (int i = 0; i < INX; i++) {
			for (int j = 0; j < INX; j++) {
				for (int k = 0; k < INX; k++) {
					const int iii = rindex(k + RAD_BW, j + RAD_BW, i + RAD_BW);
					this_s(jjj) = U[f][iii];
					this_s(jjj) *= f == er_i ? eunit : funit;
					this_s.set_range(this_s(jjj));
					jjj++;
				}
			}
		}
		s.push_back(std::move(this_s));
	}
	return std::move(s);
}

constexpr auto _0 = Real(0);
constexpr auto _1 = Real(1);
constexpr auto _2 = Real(2);
constexpr auto _3 = Real(3);
constexpr auto _4 = Real(4);
constexpr auto _5 = Real(5);

using set_rad_grid_action_type = node_server::set_rad_grid_action;
HPX_REGISTER_ACTION(set_rad_grid_action_type);

hpx::future<void> node_client::set_rad_grid(std::vector<Real> &&g /*, std::vector<Real>&& o*/) const {
	return hpx::async<typename node_server::set_rad_grid_action>(get_unmanaged_gid(), g /*, o*/);
}

void node_server::set_rad_grid(const std::vector<Real> &data /*, std::vector<Real>&& outflows*/) {
	rad_grid_ptr->set_prolong(data /*, std::move(outflows)*/);
}

using send_rad_boundary_action_type = node_server::send_rad_boundary_action;
HPX_REGISTER_ACTION(send_rad_boundary_action_type);

using send_rad_flux_correct_action_type = node_server::send_rad_flux_correct_action;
HPX_REGISTER_ACTION(send_rad_flux_correct_action_type);

void node_client::send_rad_flux_correct(std::vector<Real> &&data, const geo::face &face, const geo::octant &ci) const {
	hpx::apply<typename node_server::send_rad_flux_correct_action>(get_unmanaged_gid(), std::move(data), face, ci);
}

void node_server::recv_rad_flux_correct(std::vector<Real> &&data, const geo::face &face, const geo::octant &ci) {
	const geo::quadrant index(ci, face.get_dimension());
	niece_rad_channels[face][index].set_value(std::move(data));
}

void node_client::send_rad_boundary(std::vector<Real> &&data, const geo::direction &dir, std::size_t cycle) const {
	hpx::apply<typename node_server::send_rad_boundary_action>(get_gid(), std::move(data), dir, cycle);
}

void node_server::recv_rad_boundary(std::vector<Real> &&bdata, const geo::direction &dir, std::size_t cycle) {
	sibling_rad_type tmp;
	tmp.data = std::move(bdata);
	tmp.direction = dir;
	sibling_rad_channels[dir].set_value(std::move(tmp), cycle);
}

using send_rad_children_action_type = node_server::send_rad_children_action;
HPX_REGISTER_ACTION(send_rad_children_action_type);

void node_server::recv_rad_children(std::vector<Real> &&data, const geo::octant &ci, std::size_t cycle) {
	child_rad_channels[ci].set_value(std::move(data), cycle);
}

#include <fenv.h>

void node_client::send_rad_children(std::vector<Real> &&data, const geo::octant &ci, std::size_t cycle) const {
	hpx::apply<typename node_server::send_rad_children_action>(get_unmanaged_gid(), std::move(data), ci, cycle);
}

void rad_grid::set_dx(Real _dx) {
	dx = _dx;
}

void rad_grid::set_X(const std::vector<std::vector<Real>> &x) {
	X.resize(NDIM);
	for (int d = 0; d != NDIM; ++d) {
		X[d].resize(RAD_N3);
		for (int xi = 0; xi != RAD_NX; ++xi) {
			for (int yi = 0; yi != RAD_NX; ++yi) {
				for (int zi = 0; zi != RAD_NX; ++zi) {
					const auto D = H_BW - RAD_BW;
					const int iiir = rindex(xi, yi, zi);
					const int iiih = hindex(xi + D, yi + D, zi + D);
					//		printf( "%i %i %i %i %i %i \n", d, iiir, xi, yi, zi, iiih);
					X[d][iiir] = x[d][iiih];
				}
			}
		}
	}
}

Real rad_grid::hydro_signal_speed(const std::vector<Real> &egas, const std::vector<Real> &tau, const std::vector<Real> &sx,
								  const std::vector<Real> &sy, const std::vector<Real> &sz, const std::vector<Real> &rho) {
	Real a = 0.0;
	const Real fgamma = grid::get_fgamma();
	Opacities const opacity{};
	for (int xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
		for (int yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
			for (int zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
				const int D = H_BW - RAD_BW;
				const int iiir = rindex(xi, yi, zi);
				const int iiih = hindex(xi + D, yi + D, zi + D);
				const Real rhoinv = INVERSE(rho[iiih]);
				Real vx = sx[iiih] * rhoinv;
				Real vy = sy[iiih] * rhoinv;
				Real vz = sz[iiih] * rhoinv;
				Real e0 = egas[iiih];
				e0 -= 0.5 * vx * vx * rho[iiih];
				e0 -= 0.5 * vy * vy * rho[iiih];
				e0 -= 0.5 * vz * vz * rho[iiih];
				if (opts().eos == WD) {
					e0 -= ztwd_energy(rho[iiih]);
				}
				if (e0 < egas[iiih] * 0.001) {
					e0 = std::pow(tau[iiih], fgamma);
				}
				Real this_a = (4.0 / 9.0) * U[er_i][iiir] * rhoinv;
				//		printf( "%e %e %e %e\n",rho[iiih], e0, mmw[iiir],dx );
				const Real cons = opacity.extinction(rho[iiih], e0, mmw[iiir]) * dx;
				if (cons < 32.0) {
					this_a *= std::max(1.0 - std::exp(-cons), 0.0);
				}
				a = std::max(this_a, a);
			}
		}
	}
	return SQRT(a);
}

template <class T>
T minmod(T a, T b) {
	return (std::copysign(0.5, a) + std::copysign(0.5, b)) * std::min(std::abs(a), std::abs(b));
}

void rad_grid::allocate() {
	rad_grid::dx = dx;
	U.resize(NRF);
	Ushad.resize(NRF);
	flux.resize(NDIM);
	for (int d = 0; d < NDIM; d++) {
		flux[d].resize(NRF);
	}
	for (int f = 0; f != NRF; ++f) {
		U0[f].resize(RAD_N3);
		U[f].resize(RAD_N3);
		Ushad[f].resize(RAD_N3);
		for (int d = 0; d != NDIM; ++d) {
			flux[d][f].resize(RAD_N3);
		}
	}
}

void rad_grid::store() {
	for (int f = 0; f != NRF; ++f) {
		for (int i = 0; i != RAD_N3; ++i) {
			U0[f][i] = U[f][i];
		}
	}
}

void rad_grid::restore() {
	for (int f = 0; f != NRF; ++f) {
		for (int i = 0; i != RAD_N3; ++i) {
			U[f][i] = U0[f][i];
		}
	}
}

void rad_grid::sanity_check() {
	for (int xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
		for (int yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
			for (int zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
				const int iiir = rindex(xi, yi, zi);
				if (U[er_i][iiir] <= 0.0) {
					printf("INSANE\n");
					//		printf("%e %i %i %i\n", U[er_i][iiir], xi, yi, zi);
					abort();
				}
			}
		}
	}
}

void rad_grid::change_units(Real m, Real l, Real t, Real k) {
	const Real l2 = l * l;
	const Real t2 = t * t;
	const Real t2inv = 1.0 * INVERSE(t2);
	const Real tinv = 1.0 * INVERSE(t);
	const Real l3 = l2 * l;
	const Real l3inv = 1.0 * INVERSE(l3);
	for (int i = 0; i != RAD_N3; ++i) {
		U[er_i][i] *= (m * l2 * t2inv) * l3inv;
		U[fx_i][i] *= tinv * (m * t2inv);
		U[fy_i][i] *= tinv * (m * t2inv);
		U[fz_i][i] *= tinv * (m * t2inv);
	}
}

void rad_grid::set_physical_boundaries(geo::face face, Real t) {
	for (int i = 0; i != RAD_NX; ++i) {
		for (int j = 0; j != RAD_NX; ++j) {
			for (int k = 0; k != RAD_BW; ++k) {
				int iii1, iii0;
				switch (face) {
				case 0:
					iii1 = rindex(k, i, j);
					iii0 = rindex(RAD_BW, i, j);
					break;
				case 1:
					iii1 = rindex(RAD_NX - 1 - k, i, j);
					iii0 = rindex(RAD_NX - 1 - RAD_BW, i, j);
					break;
				case 2:
					iii1 = rindex(i, k, j);
					iii0 = rindex(i, RAD_BW, j);
					break;
				case 3:
					iii1 = rindex(i, RAD_NX - 1 - k, j);
					iii0 = rindex(i, RAD_NX - 1 - RAD_BW, j);
					break;
				case 4:
					iii1 = rindex(i, j, k);
					iii0 = rindex(i, j, RAD_BW);
					break;
				case 5:
				default:
					iii1 = rindex(i, j, RAD_NX - 1 - k);
					iii0 = rindex(i, j, RAD_NX - 1 - RAD_BW);
				}
				for (int f = 0; f != NRF; ++f) {
					U[f][iii1] = U[f][iii0];
				}
				switch (face) {
				case 0:
					if (opts().problem == MARSHAK) {
						if (t > 0) {
							auto u = marshak_wave_analytic(-opts().xscale, 0, 0, t);
							U[fx_i][iii1] = u[opts().n_fields + fx_i];
							U[er_i][iii1] = std::max(u[opts().n_fields + er_i], 1.0e-10);
						} else {
							U[fx_i][iii1] = 0.0;
							U[er_i][iii1] = 1.0e-10;
						}
					} else {
						U[fx_i][iii1] = std::min(U[fx_i][iii1], 0.0);
					}
					break;
				case 1:
					U[fx_i][iii1] = std::max(U[fx_i][iii1], 0.0);
					break;
				case 2:
					U[fy_i][iii1] = std::min(U[fy_i][iii1], 0.0);
					break;
				case 3:
					U[fy_i][iii1] = std::max(U[fy_i][iii1], 0.0);
					break;
				case 4:
					U[fz_i][iii1] = std::min(U[fz_i][iii1], 0.0);
					break;
				case 5:
					U[fz_i][iii1] = std::max(U[fz_i][iii1], 0.0);
					break;
				}
			}
		}
	}
}

hpx::future<void> node_server::exchange_rad_flux_corrections() {
	const geo::octant ci = my_location.get_child_index();
	constexpr auto full_set = geo::face::full_set();
	for (auto &f : full_set) {
		const auto face_dim = f.get_dimension();
		auto const &this_aunt = aunts[f];
		if (!this_aunt.empty()) {
			std::array<int, NDIM> lb, ub;
			lb[XDIM] = lb[YDIM] = lb[ZDIM] = RAD_BW;
			ub[XDIM] = ub[YDIM] = ub[ZDIM] = INX + RAD_BW;
			if (f.get_side() == geo::MINUS) {
				lb[face_dim] = RAD_BW;
			} else {
				lb[face_dim] = INX + RAD_BW;
			}
			ub[face_dim] = lb[face_dim] + 1;
			auto data = rad_grid_ptr->get_flux_restrict(lb, ub, face_dim);
			this_aunt.send_rad_flux_correct(std::move(data), f.flip(), ci);
		}
	}

	constexpr int size = geo::face::count() * geo::quadrant::count();
	std::array<future<void>, size> futs;
	for (auto &f : futs) {
		f = hpx::make_ready_future();
	}
	int index = 0;
	for (auto const &f : geo::face::full_set()) {
		if (this->nieces[f] == +1) {
			for (auto const &quadrant : geo::quadrant::full_set()) {
				futs[index++] =
					niece_rad_channels[f][quadrant].get_future().then([this, f, quadrant](hpx::future<std::vector<Real>> &&fdata) -> void {
						const auto face_dim = f.get_dimension();
						std::array<int, NDIM> lb, ub;
						switch (face_dim) {
						case XDIM:
							lb[XDIM] = (f.get_side() == geo::MINUS ? 0 : INX) + RAD_BW;
							lb[YDIM] = quadrant.get_side(0) * (INX / 2) + RAD_BW;
							lb[ZDIM] = quadrant.get_side(1) * (INX / 2) + RAD_BW;
							ub[XDIM] = lb[XDIM] + 1;
							ub[YDIM] = lb[YDIM] + (INX / 2);
							ub[ZDIM] = lb[ZDIM] + (INX / 2);
							break;
						case YDIM:
							lb[XDIM] = quadrant.get_side(0) * (INX / 2) + RAD_BW;
							lb[YDIM] = (f.get_side() == geo::MINUS ? 0 : INX) + RAD_BW;
							lb[ZDIM] = quadrant.get_side(1) * (INX / 2) + RAD_BW;
							ub[XDIM] = lb[XDIM] + (INX / 2);
							ub[YDIM] = lb[YDIM] + 1;
							ub[ZDIM] = lb[ZDIM] + (INX / 2);
							break;
						case ZDIM:
						default:
							lb[XDIM] = quadrant.get_side(0) * (INX / 2) + RAD_BW;
							lb[YDIM] = quadrant.get_side(1) * (INX / 2) + RAD_BW;
							lb[ZDIM] = (f.get_side() == geo::MINUS ? 0 : INX) + RAD_BW;
							ub[XDIM] = lb[XDIM] + (INX / 2);
							ub[YDIM] = lb[YDIM] + (INX / 2);
							ub[ZDIM] = lb[ZDIM] + 1;
							break;
						}
						rad_grid_ptr->set_flux_restrict(GET(fdata), lb, ub, face_dim);
					});
			}
		}
	}
	return hpx::when_all(std::move(futs)).then([](future<decltype(futs)> fout) {
		auto fin = GET(fout);
		for (auto &f : fin) {
			GET(f);
		}
	});
}

void rad_grid::set_flux_restrict(const std::vector<Real> &data, const std::array<int, NDIM> &lb, const std::array<int, NDIM> &ub,
								 const geo::dimension &dim) {
	int index = 0;
	for (int field = 0; field != NRF; ++field) {
		for (int i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (int j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (int k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					const int iii = rindex(i, j, k);
					flux[dim][field][iii] = data[index];
					++index;
				}
			}
		}
	}
}

std::vector<Real> rad_grid::get_flux_restrict(const std::array<int, NDIM> &lb, const std::array<int, NDIM> &ub,
											  const geo::dimension &dim) const {
	std::vector<Real> data;
	int size = 1;
	for (auto &dim : geo::dimension::full_set()) {
		size *= (ub[dim] - lb[dim]);
	}
	size /= (NCHILD / 2);
	size *= NRF;
	data.reserve(size);
	const int stride1 = (dim == XDIM) ? (RAD_NX) : (RAD_NX) * (RAD_NX);
	const int stride2 = (dim == ZDIM) ? (RAD_NX) : 1;
	for (int field = 0; field != NRF; ++field) {
		for (int i = lb[XDIM]; i < ub[XDIM]; i += 2) {
			for (int j = lb[YDIM]; j < ub[YDIM]; j += 2) {
				for (int k = lb[ZDIM]; k < ub[ZDIM]; k += 2) {
					const int i00 = rindex(i, j, k);
					const int i10 = i00 + stride1;
					const int i01 = i00 + stride2;
					const int i11 = i00 + stride1 + stride2;
					Real value = ZERO;
					value += flux[dim][field][i00];
					value += flux[dim][field][i10];
					value += flux[dim][field][i01];
					value += flux[dim][field][i11];
					value /= Real(4);
					data.push_back(value);
				}
			}
		}
	}
	return data;
}

void node_server::all_rad_bounds() {
	//	if( my_location.level() == 0 ) printf( "\nbounds 1\n");
	GET(exchange_interlevel_rad_data());
	//	if( my_location.level() == 0 ) printf( "\nbounds 2\n");
	collect_radiation_bounds();
	//	if( my_location.level() == 0 ) printf( "\nbounds 3\n");
	send_rad_amr_bounds();
	//	if( my_location.level() == 0 ) printf( "\nbounds 4\n");
	rcycle++;
}

hpx::future<void> node_server::exchange_interlevel_rad_data() {
	hpx::future<void> f = hpx::make_ready_future();
	int ci = my_location.get_child_index();

	if (is_refined) {
		for (auto const &ci : geo::octant::full_set()) {
			auto data = GET(child_rad_channels[ci].get_future(rcycle));
			rad_grid_ptr->set_restrict(data, ci);
		}
	}
	if (my_location.level() > 0) {
		auto data = rad_grid_ptr->get_restrict();
		parent.send_rad_children(std::move(data), ci, rcycle);
	}
	return hpx::make_ready_future();
}

void node_server::collect_radiation_bounds() {
	rad_grid_ptr->clear_amr();
	for (auto const &dir : geo::direction::full_set()) {
		if (!neighbors[dir].empty()) {
			const int width = H_BW;
			auto bdata = rad_grid_ptr->get_boundary(dir);
			neighbors[dir].send_rad_boundary(std::move(bdata), dir.flip(), rcycle);
		}
	}

	std::array<future<void>, geo::direction::count()> results;
	int index = 0;
	for (auto const &dir : geo::direction::full_set()) {
		if (!(neighbors[dir].empty() && my_location.level() == 0)) {
			results[index++] = sibling_rad_channels[dir].get_future(rcycle).then(
				/*hpx::util::annotated_function(*/ [this, dir](future<sibling_rad_type> &&f) -> void {
					auto &&tmp = GET(f);
					if (!neighbors[dir].empty()) {
						rad_grid_ptr->set_boundary(tmp.data, tmp.direction);
					} else {
						rad_grid_ptr->set_rad_amr_boundary(tmp.data, tmp.direction);
					}
				} /*, "node_server::collect_rad_boundaries::set_rad_boundary")*/);
		}
	}
	while (index < geo::direction::count()) {
		results[index++] = hpx::make_ready_future();
	}
	//	wait_all_and_propagate_exceptions(std::move(results));
	for (auto &f : results) {
		GET(f);
	}
	rad_grid_ptr->complete_rad_amr_boundary();
	if (!opts().periodic) {
		for (auto &face : geo::face::full_set()) {
			if (my_location.is_physical_boundary(face)) {
				rad_grid_ptr->set_physical_boundaries(face, current_time);
			}
		}
	}
}

void rad_grid::initialize_erad(const std::vector<Real> rho, const std::vector<Real> tau) {
	const Real fgamma = grid::get_fgamma();
	for (int xi = 0; xi != RAD_NX; ++xi) {
		for (int yi = 0; yi != RAD_NX; ++yi) {
			for (int zi = 0; zi != RAD_NX; ++zi) {
				const auto D = H_BW - RAD_BW;
				const int iiir = rindex(xi, yi, zi);
				const int iiih = hindex(xi + D, yi + D, zi + D);
				const Real ei = POWER(tau[iiih], fgamma);
				//	U[er_i][iiir] = B_p((double) rho[iiih], (double) ei, (double) mmw[iiir]) * (4.0
				//* M_PI / physcon().c); 	U[fx_i][iiir] = U[fy_i][iiir] = U[fz_i][iiir] = 0.0;
			}
		}
	}
}

rad_grid::rad_grid(Real _dx) :
	dx(_dx), is_coarse(RAD_N3), has_coarse(RAD_N3) {
	allocate();
}

rad_grid::rad_grid() :
	is_coarse(RAD_N3), has_coarse(RAD_N3) {
	allocate();
}

void rad_grid::set_boundary(const std::vector<Real> &data, const geo::direction &dir) {
	std::array<int, NDIM> lb, ub;
	get_boundary_size(lb, ub, dir, OUTER, INX, RAD_BW);
	int iter = 0;

	for (int field = 0; field != NRF; ++field) {
		auto &Ufield = U[field];
		for (int i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (int j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (int k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					Ufield[rindex(i, j, k)] = data[iter];
					++iter;
				}
			}
		}
	}
}

std::vector<Real> rad_grid::get_boundary(const geo::direction &dir) {
	std::array<int, NDIM> lb, ub;
	std::vector<Real> data;
	int size = NRF * get_boundary_size(lb, ub, dir, INNER, INX, RAD_BW);
	data.resize(size);
	int iter = 0;

	for (int field = 0; field != NRF; ++field) {
		auto &Ufield = U[field];
		for (int i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (int j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (int k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					data[iter] = Ufield[rindex(i, j, k)];
					++iter;
				}
			}
		}
	}

	return data;
}

void rad_grid::set_field(Real v, int f, int i, int j, int k) {
	U[f][rindex(i, j, k)] = v;
}

Real rad_grid::get_field(int f, int i, int j, int k) const {
	return U[f][rindex(i, j, k)];
}

void rad_grid::set_prolong(const std::vector<Real> &data) {
	int index = 0;
	for (int f = 0; f != NRF; ++f) {
		for (int i = RAD_BW; i != RAD_NX - RAD_BW; ++i) {
			for (int j = RAD_BW; j != RAD_NX - RAD_BW; ++j) {
				for (int k = RAD_BW; k != RAD_NX - RAD_BW; ++k) {
					const int iii = rindex(i, j, k);
					U[f][iii] = data[index];
					++index;
				}
			}
		}
	}
}

std::vector<Real> rad_grid::get_prolong(const std::array<int, NDIM> &lb, const std::array<int, NDIM> &ub) {
	std::vector<Real> data;
	int size = NRF;
	for (int dim = 0; dim != NDIM; ++dim) {
		size *= (ub[dim] - lb[dim]);
	}
	auto lb0 = lb;
	auto ub0 = ub;
	for (int d = 0; d != NDIM; ++d) {
		lb0[d] /= 2;
		ub0[d] /= 2;
	}

	for (int f = 0; f != NRF; ++f) {
		for (int i = lb[XDIM]; i != ub[XDIM]; ++i) {
			for (int j = lb[YDIM]; j != ub[YDIM]; ++j) {
				for (int k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
					const int iii = rindex(i / 2, j / 2, k / 2);
					Real value = U[f][iii];
					data.push_back(value);
				}
			}
		}
	}
	return data;
}

std::vector<Real> rad_grid::get_restrict() const {
	std::vector<Real> data;
	for (int f = 0; f != NRF; ++f) {
		for (int i = RAD_BW; i < RAD_NX - RAD_BW; i += 2) {
			for (int j = RAD_BW; j < RAD_NX - RAD_BW; j += 2) {
				for (int k = RAD_BW; k < RAD_NX - RAD_BW; k += 2) {
					const int iii = rindex(i, j, k);
					Real v = ZERO;
					for (int x = 0; x != 2; ++x) {
						for (int y = 0; y != 2; ++y) {
							for (int z = 0; z != 2; ++z) {
								const int jjj = iii + x * RAD_NX * RAD_NX + y * RAD_NX + z;
								v += U[f][jjj];
							}
						}
					}
					v /= Real(NCHILD);
					data.push_back(v);
				}
			}
		}
	}
	return data;
}

void rad_grid::set_restrict(const std::vector<Real> &data, const geo::octant &octant) {
	int index = 0;
	const int i0 = octant.get_side(XDIM) * (INX / 2);
	const int j0 = octant.get_side(YDIM) * (INX / 2);
	const int k0 = octant.get_side(ZDIM) * (INX / 2);
	for (int f = 0; f != NRF; ++f) {
		for (int i = RAD_BW; i != RAD_NX / 2; ++i) {
			for (int j = RAD_BW; j != RAD_NX / 2; ++j) {
				for (int k = RAD_BW; k != RAD_NX / 2; ++k) {
					const int iii = rindex(i + i0, j + j0, k + k0);
					U[f][iii] = data[index];
					++index;
					if (index > int(data.size())) {
						printf("rad_grid::set_restrict error %i %i\n", int(index), int(data.size()));
					}
				}
			}
		}
	}
};

void node_server::send_rad_amr_bounds() {
	if (is_refined) {
		constexpr auto full_set = geo::octant::full_set();
		for (auto &ci : full_set) {
			const auto &flags = amr_flags[ci];
			for (auto &dir : geo::direction::full_set()) {
				if (flags[dir]) {
					std::array<int, NDIM> lb, ub;
					std::vector<Real> data;
					get_boundary_size(lb, ub, dir, OUTER, INX / 2, H_BW);
					for (int dim = 0; dim != NDIM; ++dim) {
						lb[dim] = std::max(lb[dim] - 1, 0);
						ub[dim] = std::min(ub[dim] + 1, (int)HS_NX);
						lb[dim] = lb[dim] + ci.get_side(dim) * (INX / 2);
						ub[dim] = ub[dim] + ci.get_side(dim) * (INX / 2);
					}
					data = rad_grid_ptr->get_subset(lb, ub);
					children[ci].send_rad_amr_boundary(std::move(data), dir, rcycle);
				}
			}
		}
	}
}

using erad_init_action_type = node_server::erad_init_action;
HPX_REGISTER_ACTION(erad_init_action_type);

hpx::future<void> node_client::erad_init() const {
	return hpx::async<typename node_server::erad_init_action>(get_unmanaged_gid());
}

void node_server::erad_init() {
	std::array<hpx::future<void>, NCHILD> futs;
	int index = 0;
	if (is_refined) {
		for (auto &child : children) {
			futs[index++] = child.erad_init();
		}
	}
	grid_ptr->rad_init();
	if (is_refined) {
		hpx::wait_all(futs);
	}
}

void rad_grid::clear_amr() {
	std::fill(is_coarse.begin(), is_coarse.end(), 0);
	std::fill(has_coarse.begin(), has_coarse.end(), 0);
}

void rad_grid::set_rad_amr_boundary(const std::vector<Real> &data, const geo::direction &dir) {
	PROFILE();

	std::array<int, NDIM> lb, ub;
	int l = 0;
	get_boundary_size(lb, ub, dir, OUTER, INX / 2, H_BW);
	for (int i = lb[0]; i < ub[0]; i++) {
		for (int j = lb[1]; j < ub[1]; j++) {
			for (int k = lb[2]; k < ub[2]; k++) {
				is_coarse[hSindex(i, j, k)]++;
				assert(i < H_BW || i >= HS_NX - H_BW || j < H_BW || j >= HS_NX - H_BW || k < H_BW || k >= HS_NX - H_BW);
			}
		}
	}

	for (int dim = 0; dim < NDIM; dim++) {
		lb[dim] = std::max(lb[dim] - 1, int(0));
		ub[dim] = std::min(ub[dim] + 1, int(HS_NX));
	}

	for (int f = 0; f < NRF; f++) {
		for (int i = lb[0]; i < ub[0]; i++) {
			for (int j = lb[1]; j < ub[1]; j++) {
				for (int k = lb[2]; k < ub[2]; k++) {
					has_coarse[hSindex(i, j, k)]++;
					Ushad[f][hSindex(i, j, k)] = data[l++];
				}
			}
		}
	}
	assert(l == data.size());
}

void rad_grid::complete_rad_amr_boundary() {
	PROFILE();

	using oct_array = std::array<std::array<std::array<double, 2>, 2>, 2>;
	static thread_local std::vector<std::vector<oct_array>> Uf(NRF, std::vector<oct_array>(HS_N3));

	std::array<double, NDIM> xmin;
	for (int dim = 0; dim < NDIM; dim++) {
		xmin[dim] = X[dim][0];
	}

	const auto limiter = [](double a, double b) {
		return minmod_theta(a, b, 64. / 37.);
	};

	for (int f = 0; f < NRF; f++) {
		for (int i0 = 1; i0 < HS_NX - 1; i0++) {
			for (int j0 = 1; j0 < HS_NX - 1; j0++) {
				for (int k0 = 1; k0 < HS_NX - 1; k0++) {
					const int iii0 = hSindex(i0, j0, k0);
					if (is_coarse[iii0]) {
						for (int ir = 0; ir < 2; ir++) {
							for (int jr = 0; jr < 2; jr++) {
								for (int kr = 0; kr < 2; kr++) {
									const auto is = ir % 2 ? +1 : -1;
									const auto js = jr % 2 ? +1 : -1;
									const auto ks = kr % 2 ? +1 : -1;
									const auto &u0 = Ushad[f][iii0];
									const auto &uc = Ushad[f];
									const auto s_x = limiter(uc[iii0 + is * HS_DNX] - u0, u0 - uc[iii0 - is * HS_DNX]);
									const auto s_y = limiter(uc[iii0 + js * HS_DNY] - u0, u0 - uc[iii0 - js * HS_DNY]);
									const auto s_z = limiter(uc[iii0 + ks * HS_DNZ] - u0, u0 - uc[iii0 - ks * HS_DNZ]);
									const auto s_xy =
										limiter(uc[iii0 + is * HS_DNX + js * HS_DNY] - u0, u0 - uc[iii0 - is * HS_DNX - js * HS_DNY]);
									const auto s_xz =
										limiter(uc[iii0 + is * HS_DNX + ks * HS_DNZ] - u0, u0 - uc[iii0 - is * HS_DNX - ks * HS_DNZ]);
									const auto s_yz =
										limiter(uc[iii0 + js * HS_DNY + ks * HS_DNZ] - u0, u0 - uc[iii0 - js * HS_DNY - ks * HS_DNZ]);
									const auto s_xyz = limiter(uc[iii0 + is * HS_DNX + js * HS_DNY + ks * HS_DNZ] - u0,
															   u0 - uc[iii0 - is * HS_DNX - js * HS_DNY - ks * HS_DNZ]);
									auto &uf = Uf[f][iii0][ir][jr][kr];
									uf = u0;
									uf += (9.0 / 64.0) * (s_x + s_y + s_z);
									uf += (3.0 / 64.0) * (s_xy + s_yz + s_xz);
									uf += (1.0 / 64.0) * s_xyz;
								}
							}
						}
					}
				}
			}
		}
	}

	for (int f = 0; f < NRF; f++) {
		for (int i = 0; i < H_NX; i++) {
			for (int j = 0; j < H_NX; j++) {
				for (int k = 0; k < H_NX; k++) {
					const int i0 = (i + H_BW) / 2;
					const int j0 = (j + H_BW) / 2;
					const int k0 = (k + H_BW) / 2;
					const int iii0 = hSindex(i0, j0, k0);
					const int iiir = hindex(i, j, k);
					if (is_coarse[iii0]) {
						int ir, jr, kr;
						if constexpr (H_BW % 2 == 0) {
							ir = i % 2;
							jr = j % 2;
							kr = k % 2;
						} else {
							ir = 1 - (i % 2);
							jr = 1 - (j % 2);
							kr = 1 - (k % 2);
						}
						U[f][iiir] = Uf[f][iii0][ir][jr][kr];
					}
				}
			}
		}
	}
}

std::vector<Real> rad_grid::get_subset(const std::array<int, NDIM> &lb, const std::array<int, NDIM> &ub) {
	PROFILE();
	std::vector<Real> data;
	for (int f = 0; f < NRF; f++) {
		for (int i = lb[0]; i < ub[0]; i++) {
			for (int j = lb[1]; j < ub[1]; j++) {
				for (int k = lb[2]; k < ub[2]; k++) {
					data.push_back(U[f][hindex(i, j, k)]);
				}
			}
		}
	}
	return std::move(data);
}

using send_rad_amr_boundary_action_type = node_server::send_rad_amr_boundary_action;
HPX_REGISTER_ACTION(send_rad_amr_boundary_action_type);

void node_server::recv_rad_amr_boundary(std::vector<Real> &&bdata, const geo::direction &dir, std::size_t cycle) {
	sibling_rad_type tmp;
	tmp.data = std::move(bdata);
	tmp.direction = dir;
	sibling_rad_channels[dir].set_value(std::move(tmp), cycle);
}

void node_client::send_rad_amr_boundary(std::vector<Real> &&data, const geo::direction &dir, std::size_t cycle) const {
	hpx::apply<typename node_server::send_rad_amr_boundary_action>(get_unmanaged_gid(), std::move(data), dir, cycle);
}

#endif
