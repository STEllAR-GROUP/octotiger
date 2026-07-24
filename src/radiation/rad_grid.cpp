//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/radiation/rad_grid.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/astrolib/AutoDiff.hpp"
#include "octotiger/astrolib/Box.hpp"
#include "octotiger/astrolib/Matrix.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/radiation/opacities.hpp"
#include "octotiger/roe.hpp"
#include "octotiger/space_vector.hpp"

#include <hpx/include/future.hpp>

#include "octotiger/unitiger/radiation/radiation_physics_impl.hpp"
#include <cmath>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
RadiationStateVector radiationExternalSource(std::vector<std::vector<Real>> x, Real t);
void radiationApplyExternalSource(RadiationStateVector &Ur, RadiationStateVector const &dUdt, Real dt);
std::pair<std::vector<Real>, std::vector<Real>> radiationOpacities(GasStateVector const &U);
// Α Β Γ Δ Ε Ζ Η Θ Ι Κ Λ Μ Ν Ξ Ο Π Ρ Σ Τ Υ Φ Χ Ψ Ω
// α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ ς τ υ φ χ ψ ω
// ₐ ₑ ₕ ᵢ ⱼ ₖ ₗ ₘ ₙ ₒ ₚ ᵣ ₛ ₜ ᵤ ᵥ ₓ
// ᵃ ᵇ ᶜ ᵈ ᵉ ᶠ ᵍ ʰ ⁱ ʲ ᵏ ˡ ᵐ ⁿ ᵒ ᵖ ʳ ˢ ᵗ ᵘ ᵛ ʷ ˣ ʸ ᶻ
// ᴬ ᴮ ᴰ ᴱ ᴳ ᴴ ᴵ ᴶ ᴷ ᴸ ᴹ ᴺ ᴼ ᴾ ᴿ ᵀ ᵁ ⱽ ᵂ
// ᵅ ᵝ ᵞ ᵟ ᵋ ᶿ ᶥ ᶲ ᵡ
// ∞ ∂ ∇ ∆ ∑ ∏ ∫ √ ≈ ≠ ≤ ≥ ± × · → ← ↔ ħ ℏ Å °	⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹ ⁻ ⁼ ⁽ ⁾
// ₊ ₋ ₌ ₍ ₎

void node_server::compute_radiation(Real timestepSize) {
	try {
		static thread_local FpeGuard fpeGuard{};
		constexpr auto maxSubstepCount = std::numeric_limits<int>::max();
		auto const evolveGas = opts().hydro;
		auto &radGrid = *rad_grid_ptr;
		auto &gasGrid = *grid_ptr;
		radGrid.set_dx(grid_ptr->get_dx());
		radGrid.set_X(grid_ptr->get_X());
		auto const substepCount = radiationSubstepCount(timestepSize);
		if (substepCount > maxSubstepCount)
			throw std::runtime_error(print2string("Number of substeps greater than %i.\n", maxSubstepCount));
		auto [Ur0, Ur] = radGrid.getStateReferences();
		auto [Ug0, Ug] = gasGrid.getStateReferences();
		auto const dt = timestepSize / Real(substepCount);
		auto t = current_time;
		for (int i = 0; i < substepCount; i++) {
			Ur0 = Ur;
			Ug0 = Ug;
			for (int rk = 0; rk < 1; rk++) {
				auto const [κ, χ] = radiationOpacities(Ug);
				radiationTransportFluxes(radGrid.flux, Ur, χ, dx);
				exchange_rad_flux_corrections().get();
				radiationApplyFluxes(Ur, radGrid.flux, dt / dx);
				all_rad_bounds(t + rk * dt);
				//				auto S = radiationExternalSource(radGrid.get_X(), t);
				//				radiationApplyExternalSource(Ur, S, dt);
				auto const S = radiationImplicitSource(Ur, Ug, dt);
				radiationApplySource(Ur, Ug, S, dt);
				//				Ug = Ug0;
			}
			//			Ur = 0.5_R * (Ur0 + Ur);
			//			Ug = 0.5_R * (Ug0 + Ug);
			t += dt;
		}
	} catch (std::exception const &e) {
		std::stringstream os;
		os << "level = " << my_location.level() << std::endl;
		os << "xloc = (" << my_location[0] << ", " << my_location[1] << ", " << my_location[2] << ")" << std::endl;
		os << e.what() << std::endl;
		std::cerr << os.str();
		throw;
	}
}

std::pair<std::vector<Real>, std::vector<Real>> radiationOpacities(GasStateVector const &U) {
	std::vector<Real> χ(H_N3), κ(H_N3);
	forEach(GasGrid::exterior::box, [&](auto I) {
		auto const idx = GasGrid::exterior::box.flatten(I);
		auto const gas = gasConservedToPrimitive(U.get(idx));
		auto const ρ = gas.ρ;
		auto const T = gasTemperature(gas);
		κ[idx] = ρ * opacityAbsorption(ρ, T);
		χ[idx] = ρ * opacityScattering(ρ, T) + κ[idx];
	});
	return std::pair(κ, χ);
}

auto const ghostCount(auto idx) {
	int cnt = 0;
	for (int d = 0; d < NDIM; d++) {
		if (idx[d] < 0) cnt++;
		else if (idx[d] >= INX) cnt++;
	}
	return cnt;
}

int radiationSubstepCount(Real dt) {
	using namespace std;
	auto const c = physcon().c;
	auto const cflFactor = 0.4_R / Real(NDIM);
	return max(int(ceil(c * dt * inv(cflFactor * minimumCellWidth()))), 1);
}

void radiationApplyExternalSource(RadiationStateVector &U, RadiationStateVector const &dUdt, Real dt) {
	auto const Γ = opts().gas_gamma;
	auto const c = physcon().c;
	auto const c2 = sqr(c);
	for (int k = 0; k <= NDIM; k++) {
		auto const &dUk = dUdt[k];
		forEach(RadGrid::exterior::box, [&](auto idx) {
			auto const ir = RadGrid::exterior::box.flatten(idx);
			U[k][ir] += dUk[ir] * dt;
		});
	}
}

RadiationStateVector radiationExternalSource(std::vector<std::vector<Real>> r, Real t) {
	auto const problemType = opts().problem;
	using Source = std::function<std::vector<Real>(Real, Real, Real, Real)>;
	Source S{};
	RadiationStateVector dU;
	switch (problemType) {
	case RADIATION_EQUILIBRIUM_SPHERE:
		S = static_cast<Source>(radiationSourceEquilibriumSphere);
		break;
	default:
		S = nullptr;
		break;
	}
	if (S) {
		constexpr auto interiorBox = Box<NDIM>(INX);
		constexpr auto extBox = interiorBox.pad(RAD_BW);
		for (auto &u : dU) {
			u.resize(RAD_N3, 0_R);
		}
		auto const dx = (r[0][1] - r[0][0]) * sqrt(0.15_R);
		auto const wt_ = std::array<Real, 3>{5_R / 18_R, 8_R / 18_R, 5_R / 18_R};
		auto const *wt = &wt_[1];
		forEach(extBox, [&](auto idx) {
			auto const ir = extBox.flatten(idx);
			auto const x = r[0][ir];
			auto const y = r[1][ir];
			auto const z = r[2][ir];
			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {
					for (int k = -1; k <= 1; k++) {
						auto const du = S(x + i * dx, y + j * dx, z + k * dx, t);
						for (int d = 0; d <= NDIM; d++) {
							dU[d][ir] += du[d] * wt[i] * wt[j] * wt[k];
						}
					}
				}
			}
		});
	}
	return dU;
}

// auto quarticSolve(auto a, auto b, auto c) {
//	using std::abs;
//	using std::max;
//	using std::min;
//	using std::pow;
//	using std::sqrt;
//	constexpr int maxIter = 20;
//	a = expectNonNegative(a);
//	b = expectNonNegative(b);
//	c = expectPositive(c);
//	if (a == 0) return c / b;
//	if (b == 0) return sqrt(sqrt(c / a));
//	auto const hi = min(sqrt(sqrt(c / a)), c / b);
//	auto const lo = c / (b + a * hi * sqr(hi));
//	auto x = sqrt(lo * hi);
//	for (int n = 0; n < maxIter; n++) {
//		auto const f = a * pow(x, 4) + b * x - c;
//		auto const dfdx = 4_R * a * pow(x, 3) + b;
//		auto const dx = -f / dfdx;
//		auto const err = abs(dx) / max(x, x + dx);
//		x += dx;
//		if (err < 2_R * eps_R) return x;
//	}
//	throw std::runtime_error(print2string("quarticSolve failed to converge for a = %e b = %e c = %e\n", a, b, c));
//	return 0_R;
//};

// RadiationStateVector radiationImplicitSource2(RadiationStateVector const &Ur, GasStateVector const &Ug, Real dt) {
//	auto const c = physcon().c;
//	auto const kB = physcon().kb;
//	auto const amu = physcon().mh;
//	auto const aR = 4_R * physcon().sigma / physcon().c;
//	auto const Γ = opts().gas_gamma;
//	auto const ic = 1_R / c;
//	auto const c2 = sqr(c);
//	RadiationStateVector dU{};
//	auto const tol = std::sqrt(eps_R);
//	forEach(interiorBox, [&](auto I) {
//		auto const radIdx = exteriorRadBox.flatten(I);
//		auto const gasIdx = GasGrid::exterior::box.flatten(I);
//		auto const vg = gasConservedToPrimitive(Ug.get(gasIdx));
//		auto const U0 = Ur.get(radIdx);
//		auto const iρc2 = 1_R / expectPositive(vg.ρ * c2);
//		auto const κ = vg.ρ * opacityAbsorption(vg.ρ, vg.T);
//		auto const χ = vg.ρ * opacityTotal(vg.ρ, vg.T);
//		auto const β = vg.v * ic;
//		auto const e = vg.cv * vg.ρ * vg.T;
//		auto const [E, F] = radiationLab2Comoving(U0, β).split();
//		auto const Bo = aR * sqr(sqr(vg.T));
//		auto const λa = c * κ * dt;
//		auto const λt = c * χ * dt;
//		auto const qa = λa * Bo;
//		auto const qb = e * (1_R + λa);
//		auto const qc = λa * (E + e) + e;
//		auto const x = quarticSolve(qa, qb, qc);
//		auto const dE = (1_R - expectNonNegative(x)) * e;
//		auto const dF = -λt / (1_R + λt) * F;
//		auto const U1 = radiationComoving2Lab(ConservedRadiationState(E + dE, F + dF), β);
//		dU.set(radIdx, (U1 - U0) / dt);
//	});
//	return dU;
// }

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
	assert(false);
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

// ΑαΒβΔδΕεΦφΓγΗηΙιΚκΛλΜμΝνΟοΠπΡρΣσςΤτΥυΧχΨψΩωΖζΘθΞξ
Real radiationHydroSignalSpeed(RadiationStateVector const &Ur, GasStateVector const &Ug, Real dx) {
	// ₐ ₑ ₕ ᵢ ⱼ ₖ ₗ ₘ ₙ ₒ ₚ ᵣ ₛ ₜ ᵤ ᵥ ₓ
	using std::exp;
	using std::sqrt;
	static auto const τ_max = log(huge_R);
	static auto const kb = physcon().kb;
	static auto const Γ = opts().gas_gamma;
	static auto const τₒ = sqrt(eps_R);
	auto λ2max = 0_R;
	for (int xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
		for (int yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
			for (int zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
				const int D = H_BW - RAD_BW;
				const int ir = rindex(xi, yi, zi);
				const int ig = hindex(xi, yi, zi);
				auto const [ρ, v, ε, k] = Ug.get(ig).primitiveVariables();
				auto const p = (Γ - 1_R) * ρ * ε;
				auto const iρ = 1_R / expectPositive(ρ);
				auto const χ = opacityTotal(ρ, T);
				auto const E = expectPositive(Ur[er_i][ir]);
				auto const τ = std::min(ρ * χ * dx, τ_max);
				auto const α = (τ > τₒ) ? (1_R - exp(-τ)) : (τ * (1_R + τ));
				auto const λ2 = Γ * p * iρ + α * (4_R / 9_R) * E * iρ;
				λ2max = std::max(λ2max, λ2);
			}
		}
	}
	return sqrt(λ2max);
}

void rad_grid::allocate() {
}

void rad_grid::store() {
	for (int f = 0; f <= NDIM; ++f) {
		for (int i = 0; i != RAD_N3; ++i) {
			U0[f][i] = U[f][i];
		}
	}
}

void rad_grid::restore() {
	for (int f = 0; f <= NDIM; ++f) {
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
	using std::max;
	using std::min;
	auto const hydroCount = opts().n_fields;
	auto const dim = face.get_dimension();
	auto const side = face.get_side();
	Vector<int, NDIM> lb({0, 0, 0});
	Vector<int, NDIM> ub({RAD_NX, RAD_NX, RAD_NX});
	lb[dim] = (side == geo::MINUS) ? 0 : (RAD_NX - RAD_BW);
	ub[dim] = (side == geo::MINUS) ? RAD_BW : RAD_NX;
	const auto analytic = get_analytic();
	for (int l = lb[ZDIM]; l != ub[ZDIM]; l++) {
		for (int k = lb[YDIM]; k != ub[YDIM]; k++) {
			for (int j = lb[XDIM]; j != ub[XDIM]; j++) {
				Vector<int, NDIM> idx({j, k, l});
				const auto i = rindex(idx[0], idx[1], idx[2]);
				if (analytic != nullptr) {
					const auto u = analytic(X[XDIM][i], X[YDIM][i], X[ZDIM][i], t);
					for (integer f = 0; f <= NDIM; f++) {
						U[f][i] = u[f + hydroCount];
					}
				} else {
					auto idx0 = idx;
					if (opts().reflect_bc) {
						idx0[dim] = (side == geo::MINUS) ? (2 * RAD_BW - idx[dim] - 1) : (2 * (RAD_NX - RAD_BW) - idx[dim] - 1);
					} else {
						idx0[dim] = (side == geo::MINUS) ? RAD_BW : RAD_NX - RAD_BW - 1;
					}
					const auto i0 = rindex(idx0[0], idx0[1], idx0[2]);
					for (int field = 0; field <= NDIM; field++) {
						bool const normal = (field == fx_i + dim);
						auto &u = U[field][i];
						u = U[field][i0];
						if (normal) {
							if (opts().reflect_bc) {
								u *= -1_R;
							} else if (!opts().inflow_bc) {
								u = (side == geo::PLUS) ? max(u, 0_R) : min(u, 0_R);
							}
						}
					}
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

void node_server::all_rad_bounds(Real t) {
	//	if( my_location.level() == 0 ) printf( "\nbounds 1\n");
	GET(exchange_interlevel_rad_data());
	//	if( my_location.level() == 0 ) printf( "\nbounds 2\n");
	collect_radiation_bounds(t);
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

void node_server::collect_radiation_bounds(Real time) {
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
				rad_grid_ptr->set_physical_boundaries(face, time);
			}
		}
	}
}

void rad_grid::initialize_erad(const std::vector<Real> rho, const std::vector<Real> tau) {
	//	const Real fgamma = opts().gas_gamma;
	//	for (int xi = 0; xi != RAD_NX; ++xi) {
	//		for (int yi = 0; yi != RAD_NX; ++yi) {
	//			for (int zi = 0; zi != RAD_NX; ++zi) {
	//				const auto D = H_BW - RAD_BW;
	//				const int iiir = rindex(xi, yi, zi);
	//				const int iiih = hindex(xi + D, yi + D, zi + D);
	//				const Real ei = POWER(tau[iiih], fgamma);
	//				//	U[er_i][iiir] = B_p((double) rho[iiih], (double) ei, (double) mmw[iiir]) * (4.0
	//				//* M_PI / physcon().c); 	U[fx_i][iiir] = U[fy_i][iiir] = U[fz_i][iiir] = 0.0;
	//			}
	//		}
	//	}
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

	for (int f = 0; f <= NDIM; f++) {
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
		return minmod(a, b, 64. / 37.);
	};

	for (int f = 0; f <= NDIM; f++) {
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

	for (int f = 0; f <= NDIM; f++) {
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
	for (int f = 0; f <= NDIM; f++) {
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
