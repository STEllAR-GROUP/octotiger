// Copyright (c) 2026 AUTHORS. Distributed under the Boost Software License, Version 1.0.
#pragma once
#include "octotiger/math/Debug.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace radiationTests {
using State = std::array<double, 4>; // E, physical Fx,Fy,Fz; never F/c.
using Point = std::array<double, 3>;

inline auto dot(const Point& a, const Point& b) {
	FpeGuard fpeGuard{};
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}


inline constexpr double pi = std::numbers::pi;

struct Parameters {
	double length = 2, c = 1, chi = 5, width = .2;
	double background = 1, amplitude = .01, time = .2, luminosity = .01;
	void validate() const {
		FpeGuard fpeGuard{};
		for (double v : {length, c, chi, width, background, amplitude, time, luminosity})
			if (!std::isfinite(v)) throw std::runtime_error("Nonfinite radiation test parameter");
		if (!(length > 0 && c > 0 && chi >= 0 && width > 0 && width <= length / 4 && background > 0 && amplitude > 0 && time > 0 &&
			luminosity > 0))
			throw std::runtime_error("Invalid radiation test parameters");
	}
};

inline double sinc(double x) {
	FpeGuard fpeGuard{};
	if (std::abs(x) < 1e-4) {
		double x2 = x * x;
		return 1 - x2 / 6 + x2 * x2 / 120;
	}
	return std::sin(x) / x;
}
// Exact M1 free-streaming solution, F=c E n. Integer mode components make
// the wave periodic on EACH Cartesian side; normalizing k before constructing
// its phase (as in the supplied draft) would spoil that periodicity.
inline State streamingWave(Point x, double t, double dx, Parameters const &p) {
	FpeGuard fpeGuard{};
	Point const mode{2, -3, 1};
	auto const amplitude = std::sqrt(dot(mode, mode));
	double phase = -p.c * amplitude * t, average = 1;
	for (int d = 0; d < 3; ++d) {
		phase += mode[d] * x[d];
		average *= sinc(pi * mode[d] * dx / p.length);
	}
	double const E = 1 + .5 * average * std::cos(2 * pi * phase / p.length);
	return {E, p.c * E * 2 / std::sqrt(14.), -p.c * E * 3 / std::sqrt(14.), p.c * E / std::sqrt(14.)};
}
// Integral of the periodic half-filled square wave; evaluating at cell edges
// gives the exact finite-volume average even while a discontinuity crosses a cell.
inline double frontIntegral(double x, double length) {
	FpeGuard fpeGuard{};
	double const cycles = std::floor((x + length / 2) / length);
	double const remainder = x + length / 2 - cycles * length;
	return cycles * length / 2 + std::min(remainder, length / 2);
}
inline State streamingFront(Point x, double t, double dx, Parameters const &p) {
	FpeGuard fpeGuard{};
	double const a = x[0] - p.c * t - dx / 2, b = a + dx;
	double const fraction = std::clamp((frontIntegral(b, p.length) - frontIntegral(a, p.length)) / dx, 0., 1.);
	double const E = 1e-10 + (1 - 1e-10) * fraction;
	return {E, p.c * E, 0, 0};
}
// Gaussian light bulb, with q=L exp(-r^2/w^2)/(pi^(3/2) w^3).
// In the Eddington/diffusion limit (Pij=E delta_ij/3), solve
// div F=q and F=-c grad E/(3 chi). The Gaussian source and the reference
// match exactly, unlike mixing a Gaussian source with a uniform solid bulb.
// This is a limiting-M1 test, not an exact nonlinear M1 equilibrium.
inline State equilibriumSphere(Point x, Parameters const &p) {
	FpeGuard fpeGuard{};
	double const r = std::hypot(x[0], x[1], x[2]), s = r / p.width, s2 = s * s;
	double potential, radialCoefficient;
	if (s < 1e-3) {
		potential = 2 / (std::sqrt(pi) * p.width) * (1 - s2 / 3 + s2 * s2 / 10);
		radialCoefficient = 4 / (std::sqrt(pi) * std::pow(p.width, 3)) * (1. / 3 - s2 / 5 + s2 * s2 / 14);
	} else {
		potential = std::erf(s) / r;
		radialCoefficient = (std::erf(s) - 2 * s * std::exp(-s2) / std::sqrt(pi)) / (r * r * r);
	}
	double const f = p.luminosity / (4 * pi) * radialCoefficient;
	return {p.background + 3 * p.chi * p.luminosity / (4 * pi * p.c) * potential, f * x[0], f * x[1], f * x[2]};
}
// Tensor-product 4-point Gauss integration of the smooth spherical reference.
// The front and wave above instead have exact cell integrals.
inline State sphereAverage(Point x, double dx, Parameters const &p) {
	FpeGuard fpeGuard{};
	constexpr double nodes[]{-.8611363115940526, -.3399810435848563, .3399810435848563, .8611363115940526};
	constexpr double weights[]{.3478548451374538, .6521451548625461, .6521451548625461, .3478548451374538};
	State result{};
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			for (int k = 0; k < 4; ++k) {
				auto const u = equilibriumSphere({x[0] + dx / 2 * nodes[i], x[1] + dx / 2 * nodes[j], x[2] + dx / 2 * nodes[k]}, p);
				for (int f = 0; f < 4; ++f)
					result[f] += weights[i] * weights[j] * weights[k] / 8 * u[f];
			}
	return result;
}
inline double bulbSourceAverage(Point x, double dx, Parameters const &p) {
	FpeGuard fpeGuard{};
	double fraction = 1;
	for (double v : x)
		fraction *= .5 * (std::erf((v + dx / 2) / p.width) - std::erf((v - dx / 2) / p.width));
	return p.luminosity * fraction / (dx * dx * dx);
}
// Fixed, externally supported scattering medium: the same damping algebra as
// Skinner & Ostriker (2013), (42b)--(43), https://arxiv.org/abs/1306.0010:
// ∂ₜF = -ĉ ρ κ₀ F, Fⁿ⁺¹ = Fⁿ[1 - (1 - θ)ĉ ρ κ₀ Δt]/[1 + θĉ ρ κ₀ Δt].
// Use θ = 1, ĉ = c, with prescribed χ replacing their ρ κ₀, plus an explicit
// bulb source. Their paper neglects scattering; this test reuses the damping
// formula for a prescribed scattering medium. State stores physical F.
// Gas recoil/heating is deliberately absent in these linear transport problems.
inline State mediumSource(State u, double dt, Parameters const &p, double source = 0) {
	FpeGuard fpeGuard{};
	u[0] += dt * source;
	for (int d = 1; d < 4; ++d)
		u[d] /= 1 + dt * p.c * p.chi;
	return u;
}
// Fourier mode propagator for E_tt + c chi E_t + (c^2 k^2/3)E=0,
// with E_t(0)=0. Return A and B where E=A E0, E_t=-v^2 k^2 B E0.
// This expression handles overdamped, oscillatory and critically damped modes
// without exp(-alpha*t)*cosh(lambda*t) overflow or cancellation at k=0.
inline std::array<double, 2> telegraphMode(double k2, double t, double c, double chi) {
	FpeGuard fpeGuard{};
	double const alpha = c * chi / 2, omega2 = c * c * k2 / 3, disc = alpha * alpha - omega2;
	if (k2 == 0) return {1, alpha == 0 ? t : -std::expm1(-2 * alpha * t) / (2 * alpha)};
	if (disc > 0) {
		double const lambda = std::sqrt(disc), z = lambda * t;
		if (z < 1e-4) {
			double const z2 = z * z, e = std::exp(-alpha * t);
			double const B = e * t * (1 + z2 / 6 + z2 * z2 / 120);
			return {e * (1 + z2 / 2 + z2 * z2 / 24) + alpha * B, B};
		}
		double const slow = omega2 / (alpha + lambda);
		double const e1 = std::exp(-slow * t), e2 = std::exp(-(alpha + lambda) * t);
		double const B = e1 * (-std::expm1(-2 * lambda * t)) / (2 * lambda);
		return {.5 * (e1 + e2) + alpha * B, B};
	}
	double const z = std::sqrt(-disc) * t, e = std::exp(-alpha * t), B = e * t * sinc(z);
	return {e * std::cos(z) + alpha * B, B};
}
} // namespace radiationTests
