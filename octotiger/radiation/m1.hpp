// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include "octotiger/math/Debug.hpp"
#include "octotiger/math/Real.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace radiation_m1 {

// Method references (equation numbers below refer to these papers):
// Hanawa & Audit, JQSRT 145, 9--16 (2014), "Reformulation of the M1 model
// of radiative transfer", doi:10.1016/j.jqsrt.2014.04.014.
// Skinner & Ostriker, ApJS 206, 21 (2013), "A Two-moment Radiation
// Hydrodynamics Module in Athena Using a Time-explicit Godunov Method",
// doi:10.1088/0067-0049/206/2/21.

// Conserved state: (E, Fx, Fy, Fz), with physical (code-unit) fluxes.
// Primitive state: (E, Fx/(c E), Fy/(c E), Fz/(c E)).
using state = std::array<Real, 4>;
using vector = std::array<Real, 3>;
inline constexpr Real reconstruction_theta = 1.3;
// Allow accumulated roundoff after conservative cancellation near the streaming cone.
inline constexpr Real roundoff = 4096 * std::numeric_limits<Real>::epsilon();

inline Real dot(const vector& a, const vector& b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline void check_state(const state& u, Real c) {
#ifndef NDEBUG
	(void) expectPositive(expectFinite(c));
	(void) expectNonNegative(expectFinite(u[0]));
	for (int d = 1; d < 4; ++d) expectFinite(u[d]);
	if (u[0] == 0) {
		for (int d = 1; d < 4; ++d) expectRange(0_R, u[d], 0_R);
	} else {
		const vector f{u[1] / c / u[0], u[2] / c / u[0], u[3] / c / u[0]};
		(void) expectRange(0_R, dot(f, f), 1_R + roundoff);
	}
#else
	(void) u;
	(void) c;
#endif
}

inline state primitive(const state& u, Real c) {
	check_state(u, c);
	if (u[0] == 0) return {};
	state q{u[0], u[1] / c / u[0], u[2] / c / u[0], u[3] / c / u[0]};
	const Real f2 = q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
	// Only roundoff can exceed one for an admissible cell average.
	if (f2 > 1) {
		const Real scale = 1 / std::sqrt(f2);
		for (int d = 1; d < 4; ++d) q[d] *= scale;
	}
	return q;
}

inline Real minmod_theta(Real left, Real right, Real theta = reconstruction_theta) {
	if ((left > 0 && right > 0) || (left < 0 && right < 0)) {
		return std::copysign(std::min({theta * std::abs(left),
			theta * std::abs(right), 0.5 * std::abs(left + right)}), left);
	}
	return 0;
}

// One pair per coordinate direction: six face-centre states per cell.
inline std::pair<state, state> reconstruct(const state& qm, const state& q,
	const state& qp, Real c) {
	state slope;
	for (int f = 0; f < 4; ++f) {
		slope[f] = 0.5 * minmod_theta(q[f] - qm[f], qp[f] - q[f]);
	}
	const vector f{q[1], q[2], q[3]};
	const vector df{slope[1], slope[2], slope[3]};
	const Real a = dot(df, df);
	const Real b = std::abs(dot(f, df));
	const Real room = std::max(Real(0), 1 - dot(f, f));
	// A common vector limiter keeps BOTH faces inside the reduced-flux ball.
	// Clipping components individually would not enforce |F| <= c E.
	if (a + 2 * b > room) {
		const Real denominator = std::sqrt(b * b + a * room) + b;
		const Real scale = denominator > 0 ? room / denominator : 0;
		for (int d = 1; d < 4; ++d) slope[d] *= scale;
	}
	state minus, plus;
	minus[0] = expectNonNegative(q[0] - slope[0]);
	plus[0] = expectNonNegative(q[0] + slope[0]);
	for (int d = 1; d < 4; ++d) {
		minus[d] = c * minus[0] * (q[d] - slope[d]);
		plus[d] = c * plus[0] * (q[d] + slope[d]);
	}
	check_state(minus, c);
	check_state(plus, c);
	return {minus, plus};
}

struct closure {
	Real H, pressure;
	vector beta;
	Real beta2;
};

// Hanawa & Audit (2014), equations (14), (16), (17), (19), (20).
inline closure close(const state& u, Real c) {
	const state q = primitive(u, c);
	const Real f2 = std::min(Real(1), q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
	const Real H_over_E = (2 + std::sqrt(4 - 3 * f2)) / 3;
	const vector beta{q[1] / H_over_E, q[2] / H_over_E, q[3] / H_over_E};
	const Real b2 = std::min(Real(1), dot(beta, beta));
	const Real H = u[0] * H_over_E;
	return {H, 0.25 * (1 - b2) * H, beta, b2};
}

struct flux_state {
	state flux;
	Real minus, plus;
};

inline flux_state physical_flux(const state& u, int normal, Real c, Real grid_velocity = 0) {
	(void) expectRange(0_R, Real(normal), 2_R);
	const auto m = close(u, c);
	const Real bn = m.beta[normal];
	Real transverse2 = 0;
	for (int d = 0; d < 3; ++d) {
		if (d != normal) transverse2 += m.beta[d] * m.beta[d];
	}
	// Equations (51), (53); this form avoids cancellation in the radicand.
	const Real one_minus_b2 = 1 - m.beta2;
	const Real root = std::sqrt(one_minus_b2 * (3 * one_minus_b2 + 2 * transverse2));
	flux_state result;
	result.minus = c * (2 * bn - root) / (3 - m.beta2) - grid_velocity;
	result.plus = c * (2 * bn + root) / (3 - m.beta2) - grid_velocity;
	result.flux[0] = u[normal + 1] - grid_velocity * u[0];
	for (int d = 0; d < 3; ++d) {
		const Real Pnd = m.H * bn * m.beta[d] + (d == normal ? m.pressure : 0);
		result.flux[d + 1] = c * c * Pnd - grid_velocity * u[d + 1];
	}
	return result;
}

// S&O (2013), section 3.3, equation (39), with their c_hat set to c.
// Use the extreme Hanawa-Audit characteristics from BOTH reconstructed states.
inline state hll(const state& left, const state& right, int normal, Real c,
	Real grid_velocity = 0) {
	const auto l = physical_flux(left, normal, c, grid_velocity);
	const auto r = physical_flux(right, normal, c, grid_velocity);
	const Real sm = std::min(l.minus, r.minus);
	const Real sp = std::max(l.plus, r.plus);
	// Also handles coincident zero-speed waves without a 0/0 denominator.
	if (sm >= 0) return l.flux;
	if (sp <= 0) return r.flux;
	const Real wr = sp / (sp - sm);
	const Real wl = -sm / (sp - sm);
	const Real viscosity = sm * wr;
	state result;
	for (int f = 0; f < 4; ++f) {
		result[f] = wr * l.flux[f] + wl * r.flux[f] + viscosity * (right[f] - left[f]);
	}
	return result;
}

// An intentionally restrictive multidimensional CFL for PLM + forward Euler;
// this safety factor is an implementation choice, not a coefficient from either paper.
// The grid
// speed is the maximum |v_grid| component on this block (including its faces).
inline Real transport_timestep(Real dx, Real c, Real cfl, Real grid_speed = 0) {
	(void) expectPositive(expectFinite(dx));
	(void) expectPositive(expectFinite(c));
	(void) expectPositive(expectFinite(cfl));
	(void) expectNonNegative(expectFinite(grid_speed));
	return std::min(cfl, Real(0.25)) * dx / (3 * (c + grid_speed));
}

struct energies {
	Real gas, radiation;
};

// Skinner & Ostriker (2013), (44)--(49), theta=1 and c_hat=c.
// Opacities are frozen for this first-order step. log_alpha describes B=alpha e^4.
// The Marshak test instead has B=e. The scaled polynomial has coefficients
// <=1, avoiding quartic overflow and cancellation in a weak radiation field.
inline energies thermal_exchange(Real e, Real E, Real eta, Real log_alpha,
	bool marshak = false) {
	(void) expectNonNegative(expectFinite(E));
	(void) expectNonNegative(expectFinite(eta));
	if (eta == 0) return {expectNonNegative(e), E};
	const Real inverse = 1 / (1 + eta);
	const Real weight = eta * inverse;
	const Real rhs = expectNonNegative(expectFinite(e + weight * E));
	if (marshak) {
		const Real gas = rhs / (1 + weight);
		return {gas, inverse * E + weight * gas};
	}
	if (rhs == 0) return {0, inverse * E};
	const Real log_coefficient = std::log(weight) + log_alpha;
	const Real log_rhs = std::log(rhs);
	const Real quartic_log_scale = (log_rhs - log_coefficient) / 4;
	const bool linear_bound = log_rhs <= quartic_log_scale;
	const Real scale = linear_bound ? rhs : std::exp(quartic_log_scale);
	// Set the active bounding coefficient to exactly one: exp(log(rhs)) can
	// round below rhs and otherwise put a nearly linear root outside [0,1].
	const Real linear = linear_bound ? 1 : scale / rhs;
	const Real quartic = linear_bound ? std::exp(std::min(Real(0), log_coefficient + 3 * log_rhs)) : 1;
	Real lo = 0, hi = 1, y = 1;
	for (int iteration = 0; iteration < 80; ++iteration) {
		const Real y2 = y * y;
		const Real residual = linear * y + quartic * y2 * y2 - 1;
		if (std::abs(residual) <= 16 * std::numeric_limits<Real>::epsilon()) {
			return {scale * y, inverse * E + rhs * quartic * y2 * y2};
		}
		if (residual > 0) hi = y;
		else lo = y;
		const Real next = y - residual / (linear + 4 * quartic * y2 * y);
		y = next > lo && next < hi ? next : 0.5 * (lo + hi);
	}
	throw std::runtime_error("Radiation thermal exchange did not converge");
}

struct coupled_state {
	state radiation;
	vector momentum;
	Real gas_energy, internal_energy;
};

// S&O (8), (42), (49): explicit velocity terms, backward-Euler flux damping
// and thermal exchange. chi_a and chi_t are extinction coefficients [1/length],
// NOT mass opacities. For distinct energy and flux means, (5a) gives 2 chi_a-chi_t;
// equal opacities recover (8d). We retain (E I+P).v from (8e), rather than the
// optional isotropic approximation (18). Total gas+radiation energy and
// momentum (m+F/c^2) receive exactly opposite exchange increments.
// As in S&O section 3.4, the velocity terms must be non-stiff. A light-crossing
// CFL alone is not a general stability guarantee in the dynamic-diffusion limit.
inline coupled_state couple(const state& u, const vector& momentum, Real gas_energy,
	Real internal_energy, Real rho, Real chi_a, Real chi_t, Real dt, Real c,
	Real log_alpha, bool marshak = false) {
	check_state(u, c);
	(void) expectPositive(expectFinite(rho));
	(void) expectNonNegative(expectFinite(internal_energy));
	(void) expectNonNegative(expectFinite(chi_a));
	(void) expectNonNegative(expectFinite(chi_t));
	(void) expectNonNegative(expectFinite(dt));
	if (dt == 0 || (chi_a == 0 && chi_t == 0)) {
		return {u, momentum, gas_energy, internal_energy};
	}
	const vector velocity{momentum[0] / rho, momentum[1] / rho, momentum[2] / rho};
	const vector F{u[1], u[2], u[3]};
	const auto m = close(u, c);
	const Real eta = expectNonNegative(expectFinite(dt * c * chi_t));
	const Real inverse = 1 / (1 + eta);
	const Real weight = eta * inverse;
	const Real beta_v = dot(m.beta, velocity);
	coupled_state result;
	Real kinetic_change = 0;
	for (int d = 0; d < 3; ++d) {
		const Real advective_flux = (u[0] + m.pressure) * velocity[d] + m.H * m.beta[d] * beta_v;
		result.radiation[d + 1] = inverse * F[d] + weight * advective_flux;
		const Real dm = ((F[d] - result.radiation[d + 1]) / c) / c;
		result.momentum[d] = momentum[d] + dm;
		kinetic_change += dm * (velocity[d] + 0.5 * dm / rho);
	}
	const Real explicit_work = dt * (2 * chi_a - chi_t) * dot(velocity, F) / c;
	const Real E_star = u[0] + explicit_work;
	const Real e_star = internal_energy - explicit_work - kinetic_change;
	// No clipping of a failed explicit update: it would destroy conservation.
	if (!(E_star >= 0) || !(e_star + (dt * c * chi_a / (1 + dt * c * chi_a)) * E_star >= 0)) {
		throw std::runtime_error("Radiation explicit source step exhausted the available energy; decrease the shared timestep");
	}
	const auto energy = thermal_exchange(e_star, E_star, dt * c * chi_a, log_alpha, marshak);
	result.radiation[0] = energy.radiation;
	result.gas_energy = gas_energy + (u[0] - energy.radiation);
	result.internal_energy = energy.gas;
	check_state(result.radiation, c);
	(void) expectNonNegative(expectFinite(result.internal_energy));
	(void) expectFinite(result.gas_energy);
	return result;
}

} // namespace radiation_m1
