/*
 * radiationWave.cpp
 *
 *  Created on: Jun 22, 2026
 *      Author: dmarce1
 */

#include <hpx/hpx_init.hpp>

#include "octotiger/defs.hpp"
#include "octotiger/gas/GasEoS.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/hpxfft/hpxfft.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/radiation/RadiationEoS.hpp"

// Steady-state diffusion equilibrium ("light bulb") test
// ∂ₜE + ∇·F = 0
// Equilibrium: ∂ₜE = 0
// Therefore: ∇·F = 0
// Spherical symmetry: (1/r²) ∂ᵣ(r²Fᵣ) = 0
// Integrating: r²Fᵣ = const
// Define luminosity: L = 4πr²Fᵣ
// Therefore: Fᵣ(r) = L/(4πr²)
// Diffusion limit: Fᵣ = -(c/3χ) ∂ᵣE
// Substituting: ∂ᵣE = -(3χL)/(4πcr²)
// Integrating from R to r:
// E(r) = E(R) + (3χL)/(4πc)(1/r - 1/R)
//      = E(R) + (3χL)/(4πcR)(R/r - 1)
// Flux vector: F⃗(r) = (L/(4πr²)) r̂
// Finite-radius bulb:  ≤ r ≤ R
// Uniform volumetric source: q = L/((4/3)πrb³)
// Inside bulb: ∇·F = q
// Spherical symmetry: (1/r²) ∂ᵣ(r²Fᵣ) = q
// Integrating: Fᵣ(r) = qr/3 = Lr/(4πrb³)
// Diffusion relation: Fᵣ = -(c/3χ) ∂ᵣE
// Therefore: ∂ᵣE = -(χq/c)r
// Integrating: E(r) = E(rb) + (χq/(2c))(rb² - r²)
// Result: E(r) is finite at r = 0.
//
//

// Α Β Γ Δ Ε Ζ Η Θ Ι Κ Λ Μ Ν Ξ Ο Π Ρ Σ Τ Υ Φ Χ Ψ Ω
// α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ ς τ υ φ χ ψ ω
// ₐ ₑ ₕ ᵢ ⱼ ₖ ₗ ₘ ₙ ₒ ₚ ᵣ ₛ ₜ ᵤ ᵥ ₓ
// ᵃ ᵇ ᶜ ᵈ ᵉ ᶠ ᵍ ʰ ⁱ ʲ ᵏ ˡ ᵐ ⁿ ᵒ ᵖ ʳ ˢ ᵗ ᵘ ᵛ ʷ ˣ ʸ ᶻ
// ᴬ ᴮ ᴰ ᴱ ᴳ ᴴ ᴵ ᴶ ᴷ ᴸ ᴹ ᴺ ᴼ ᴾ ᴿ ᵀ ᵁ ⱽ ᵂ
// ᵅ ᵝ ᵞ ᵟ ᵋ ᶿ ᶥ ᶲ ᵡ
// ∞ ∂ ∇ ∆ ∑ ∏ ∫ √ ≈ ≠ ≤ ≥ ± × · → ← ↔ ħ ℏ Å °	⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹
static auto const L_cgs = 1e38_R;
static auto const R_cgs = 1e13_R;
static auto const massDensity = 1.0e-9_R;
static auto const bulbCellRadius = 1_R;
// static auto const bulbCellVolume = []() {
//	int const imax = std::ceil(bulbCellRadius);
//	int cnt = 0;
//	for (int i = -imax; i < imax; i++) {
//		for (int j = -imax; j < imax; j++) {
//			for (int k = -imax; k < imax; k++) {
//				auto const r2 = sqr(Real(i) + 0.5_R) + sqr(Real(j) + 0.5_R) + sqr(Real(k) + 0.5_R);
//				auto const r = std::sqrt(r2);
//				if (r < bulbCellRadius) cnt++;
//			}
//		}
//	}
//	return cnt;
// }();

std::vector<Real> radiationSourceEquilibriumSphere(Real x_, Real y_, Real z_, Real) {
	using std::exp;
	using std::max;
	using std::sqrt;
	FpeGuard fpeGuard{};
	auto const convert = CgsToCode{};
	auto const rb = bulbCellRadius * minimumCellWidth();
	auto const c0 = 1_R / cube(rb * sqrt(pi_R));
	auto const L = convert.power(L_cgs);
	auto const R = convert.length(R_cgs);
	std::vector<Real> S(NDIM + 1, 0_R);
	auto const r = Vector<Real, NDIM>({x_, y_, z_});
	auto const x = r / rb;
	auto const x2 = x.dot(x);
	S[er_i] = c0 * L * exp(-x2);
	return S;
}

std::vector<Real> analyticEquilibriumSphere(Real x_, Real y_, Real z_, Real) {
	using std::max;
	using std::pow;
	FpeGuard fpeGuard{};
	auto const convert = CgsToCode{};
	auto const L = convert.power(L_cgs);
	auto const R = convert.length(R_cgs);
	auto const rb = bulbCellRadius * minimumCellWidth();
	auto const R2 = R * R;
	auto const rb2 = rb * rb;
	auto const c = physcon().c;
	auto const kB = physcon().kb;
	auto const amu = physcon().mh;
	auto const Γ = opts().gas_gamma;
	auto const Z = opts().atomic_number[0];
	auto const A = opts().atomic_mass[0];
	auto const μ = A / (1_R + Z);
	auto const r = Vector<Real, NDIM>({x_, y_, z_});
	auto const r2 = r.dot(r);
	auto const r1 = sqrt(r2);
	auto const rHat = normalize(r);
	Vector<Real, NDIM> F;
	Real E;
	auto const ρ = convert.massDensity(r1 < R ? massDensity : 1e-20_R * massDensity);
	auto const EatR = L / (4_R * pi_R * c * R2);
	auto const χ = ρ * convert.opacity(opts().sigma0 + opts().kappa0);
	auto const Ehat = EatR + (3_R * χ * L) / (4_R * pi_R * c) * (1_R / rb - 1_R / R);
	if (r1 < rb) {
		E = Ehat + (3_R * χ * L) / (8_R * pi_R * c * rb) * (1_R - r2 / rb2);
		F = L * r1 / (4_R * pi_R * c * rb * rb * rb) * rHat;
	} else if (r1 < R) {
		E = EatR + (3_R * χ * L) / (4_R * pi_R * c) * (1_R / r1 - 1_R / R);
		F = L / (4_R * pi_R * c * r2) * rHat;
	} else {
		E = L / (4_R * pi_R * c * r2);
		F = E * almostOne * rHat;
	}
	auto const F1 = abs(F);
	if (F1 > E) {
		printf("F = %e * c * E\n", F1 / E);
	}
	assert(F1 < E);
	auto const T = radiationTemperature(E);
	auto const ε = (kB * T) / ((Γ - 1_R) * amu * μ);
	{
		std::vector<Real> G(opts().n_fields, 0_R);
		std::vector<Real> R(NRF, 0_R);
		G[rho_i] = ρ;
		G[tau_i] = gasEnergy2Entropy(ρ, ρ * ε);
		G[egas_i] = ρ * ε;
		G[spc_i] = G[rho_i];
		R[er_i] = E;
		for (int d = 0; d < NDIM; d++) {
			R[fx_i + d] = F[d];
		}
		G.insert(G.end(), R.begin(), R.end());
		return G;
	}
}

std::vector<Real> testEquilibriumSphere(Real x, Real y, Real z, Real dx) {
	return analyticEquilibriumSphere(x, y, z, 0_R);
}
