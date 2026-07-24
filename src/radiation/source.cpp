/*
 * source.cpp
 *
 *  Created on: Jul 10, 2026
 *      Author: dmarce1
 */

#include "octotiger/math/AutoDiff.hpp"
#include "octotiger/radiation/rad_grid.hpp"
// Α Β Γ Δ Ε Ζ Η Θ Ι Κ Λ Μ Ν Ξ Ο Π Ρ Σ Τ Υ Φ Χ Ψ Ω
// α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ ς τ υ φ χ ψ ω
// ₐ ₑ ₕ ᵢ ⱼ ₖ ₗ ₘ ₙ ₒ ₚ ᵣ ₛ ₜ ᵤ ᵥ ₓ
// ᵃ ᵇ ᶜ ᵈ ᵉ ᶠ ᵍ ʰ ⁱ ʲ ᵏ ˡ ᵐ ⁿ ᵒ ᵖ ʳ ˢ ᵗ ᵘ ᵛ ʷ ˣ ʸ ᶻ
// ᴬ ᴮ ᴰ ᴱ ᴳ ᴴ ᴵ ᴶ ᴷ ᴸ ᴹ ᴺ ᴼ ᴾ ᴿ ᵀ ᵁ ⱽ ᵂ
// ᵅ ᵝ ᵞ ᵟ ᵋ ᶿ ᶥ ᶲ ᵡ
// ∞ ∂ ∇ ∆ ∑ ∏ ∫ √ ≈ ≠ ≤ ≥ ± × · → ← ↔ ħ ℏ Å °	⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹ ⁻ ⁼ ⁽ ⁾
// ₊ ₋ ₌ ₍ ₎
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
//

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
//	forEach(RadGrid::interior::box, [&](auto I) {
//		auto const radIdx = RadGrid::exterior::box.flatten(I);
//		auto const gasIdx = GasGrid::exterior::box.flatten(I);
//		auto const Urad0 = Ur.get(radIdx);
//		auto const gasPrim = gasConservedToPrimitive(Ug.get(gasIdx));
//		auto const& [ρ, p, v, _] = gasPrim;
//		auto const iρc2 = 1_R / (ρ * c2);
//		auto const T = gasTemperature(gasPrim);
//		auto const κ = opacityAbsorption(ρ, T);
//		auto const χ = opacityTotal(ρ, T);
//		auto const β = v * ic;
//		auto const e = p / (Γ - 1_R);
//		auto const [Eₒ, Fₒ] = radiationLab2Comoving(Urad0, β);
//		auto const Bₚ = aR * sqr(sqr(T));
//		auto const ρcκdt = ρ * c * κ * dt;
//		auto const ρcχdt = ρ * c * χ * dt;
//		auto const qa = ρcκdt * Bₚ;
//		auto const qb = e * (1_R + ρcκdt);
//		auto const qc = ρcκdt * (Eₒ + e) + e;
//		auto const x = quarticSolve(qa, qb, qc);
//		auto const dEₒ = (1_R - expectNonNegative(x)) * e;
//		auto const dFₒ = -ρcχdt / (1_R + ρcχdt) * Fₒ;
//		auto const Urad1 = radiationComoving2Lab(ConservedRadiationState(Eₒ + dEₒ, Fₒ + dFₒ), β);
//		dU.set(radIdx, (Urad1 - Urad0) / dt);
//	});
//	return dU;
// }

auto const stressEnergyTensor(auto Er, auto const &F) {
	using Type = std::remove_cvref_t<decltype(Er)>;
	auto const δ = identityMatrix<Type, NDIM>();
	auto const iEr = 1_R / Er;
	auto const f = F * iEr;
	auto const f2 = expectRange(0_R, f.dot(f), 1_R);
	auto const ξ = (1_R / 3_R) * (5_R - 2_R * sqrt(4_R - 3_R * f2));
	auto const Dd = 0.5_R * (1_R - ξ);
	auto const Ds = 0.5_R * (3_R * ξ - 1_R);
	auto const Ds_if2 = (f2.value() > eps_R) ? (Ds / f2) : (0.5_R - 0.1875_R * f2);
	auto const P = Er * (Dd * δ + Ds_if2 * outerProduct(f, f));
	Matrix<Type, NDIM + 1> 𝒫;
	𝒫[0][0] = Er;
	for (int n = 0; n < NDIM; n++) {
		𝒫[n + 1][0] = 𝒫[0][n + 1] = F[n];
		for (int m = n; m < NDIM; m++) {
			𝒫[n + 1][m + 1] = 𝒫[m + 1][n + 1] = P[n][m];
		}
	}
	return 𝒫;
};

template <int Oβ = 2>
auto const lorentzBoost(auto const β) {
	using Type = std::remove_cvref_t<decltype(β[0])>;
	auto const δ = identityMatrix<Type, NDIM + 1>();
	if constexpr (Oβ == 0) {
		return std::pair(δ, δ);
	} else {
		auto Λ = δ;
		auto iΛ = δ;
		for (int n = 0, ν = 1; n < NDIM; n++, ν++) {
			auto const &βn = β[n];
			Λ[0][ν] = Λ[ν][0] = -βn;
			iΛ[0][ν] = iΛ[ν][0] = βn;
		}
		if constexpr (Oβ == 2) {
			Λ[0][0] += 0.5_R * β.dot(β);
			for (int n = 0, ν = 1; n < NDIM; n++, ν++) {
				for (int m = 0, μ = 1; m < NDIM; m++, μ++) {
					auto const dΛ = 0.5_R * β[n] * β[m];
					Λ[ν][μ] += dΛ;
					iΛ[ν][μ] += dΛ;
				}
			}
		}
		return std::pair(Λ, iΛ);
	}
};

RadiationStateVector radiationImplicitSource(RadiationStateVector const &Ur, GasStateVector const &Ug, Real dt) {
	constexpr unsigned Oβ = 2;
	using std::abs;
	using std::min;
	constexpr int maxIters = 40;
	constexpr auto dEmax = 0.05_R;
	constexpr auto tol2 = eps_R;
	constexpr auto I4 = identityMatrix<Real, NDIM + 1>();
	auto const c = physcon().c;
	auto const ic = 1_R / c;
	auto const c2 = sqr(c);
	auto const ic2 = sqr(ic);
	auto const kB = physcon().kb;
	auto const mₕ = physcon().mh;
	auto const aᵣ = 4_R * physcon().sigma * ic;
	auto const Γ = opts().gas_gamma;
	auto const idt = 1_R / dt;
	RadiationStateVector dUrad{};
	GasStateVector dUgas{};
	using Auto = AutoDiff<Real, NDIM + 1>;
	forEach(RadGrid::exterior::box, [&](auto I) {
		auto const idxR = RadGrid::exterior::box.flatten(I);
		auto const idxG = GasGrid::exterior::box.flatten(I);
		auto const Ug0 = Ug.get(idxG);
		auto const Ur0 = Ur.get(idxR);
		auto const gasPrim0 = Ug0.toPrimitive();
		auto const &ρ = Ug0.ρ;
		auto const &T = gasPrim0.T;
		auto const ρc = ρ * c;
		auto const iρ = 1_R / expectPositive(ρ);
		auto const iρc = iρ * ic;
		auto const iρc2 = iρc * ic;
		Auto const Er0 = Ur0.E;
		Auto const Eg0 = Ug0.E;
		Auto const E0 = Er0 + Eg0;
		Vector<Auto, NDIM> const F0 = Ur0.F;
		Vector<Auto, NDIM> const β0 = Ug0.S * iρc;
		RadiationState<Real> x = 0_R;
		auto const ε0 = max(0_R, Eg0 * iρ - 0.5_R * c2 * β0.dot(β0));
		auto const B0 = aᵣ * sqr(sqr(T));
		auto err2 = inf_R;
		for (int iter = 0; iter < maxIters; iter++) {
			auto const dE = Auto::genVar(x[0], 0);
			auto const dF = Vector<Auto, NDIM>({Auto::genVar(x[1], 1), Auto::genVar(x[2], 2), Auto::genVar(x[3], 3)});
			auto const Er = expectPositive(Er0 + dE);
			auto const Eg = expectPositive(Eg0 - dE);
			auto const F = F0 + dF;
			auto const β = β0 - dF * iρc2;
			auto const β2 = expectRange(0_R, β.dot(β), 9_R / 64_R);
			auto const ε = max(0_R, Eg * iρ - 0.5_R * c2 * β2);
			auto const κ = opacityAbsorption(ρ, T);
			auto const χ = opacityTotal(ρ, T);
			auto const Bₚ = B0 * sqr(sqr(ε / ε0));
			auto const [Λ, iΛ] = lorentzBoost<Oβ>(β);
			auto const 𝒫 = stressEnergyTensor(Er, F);
			auto const [Eₒ, Fₒ] = split<1, NDIM, Auto>(Λ[0] * 𝒫 * Λ);
			auto const Gₒ = ρc * concatenate(κ * (Eₒ - Bₚ), χ * Fₒ);
			auto const G = iΛ * Gₒ;
			Vector<Real, NDIM + 1> f;
			Matrix<Real, NDIM + 1> dfdx;
			for (int k = 0; k <= NDIM; k++) {
				f[k] = x[k] + dt * G[k].value();
				for (int j = 0; j <= NDIM; j++) {
					dfdx[k][j] = dt * G[k].derivative(j);
				}
				dfdx[k][k] += 1_R;
			}
			Vector<Real, NDIM + 1> const dx = -inv(dfdx) * f;
			x += dx;
			err2 = dx.dot(dx) / sqr(E0.value());
			if (err2 <= tol2) break;
		}
		if (err2 > tol2) throw std::runtime_error(print2string("radiationImplicitSource failed for idx = %i, err2 = %e\n", idxR, err2));
		dUrad.set(idxR, x * idt);
	});
	return dUrad;
}

void radiationApplySource(RadiationStateVector &Ur, GasStateVector &Ug, RadiationStateVector const &dUdt, Real dt) {
	using std::max;
	using std::pow;
	FpeGuard fpeGuard{};
	auto const Γ = opts().gas_gamma;
	auto const c = physcon().c;
	auto const ic = 1_R / c;
	forEach(RadGrid::exterior::box, [&](auto idx) {
		auto const ir = RadGrid::exterior::box.flatten(idx);
		auto const ig = GasGrid::exterior::box.flatten(idx);
		auto radCon = Ur.get(ir);
		auto gasCon = Ug.get(ig);
		auto &[E, F] = radCon;
		auto &[ρ, Eg, τ, S, _] = gasCon;
		auto const iρ = 1_R / expectPositive(ρ);
		auto const dE = dUdt[er_i][ir] * dt;
		auto const dF = Vector<Real, NDIM>({dUdt[fx_i][ir], dUdt[fy_i][ir], dUdt[fz_i][ir]}) * dt;
		auto const de = -dE + iρ * ic * dF.dot(S - 0.5_R * ic * dF);
		E += dE;
		F += dF;
		Eg -= dE;
		S -= dF * ic;
		//		if (pow(τ, Γ) + de < 0_R)
		//			throw std::runtime_error(print2string("radiationApplySource failed for idx = %i, τ^Γ  = %e de = %e\n", ir, pow(τ, Γ),
		// de));
		auto const ei0 = gasEntropy2Energy(ρ, τ);
		auto const ei1 = expectPositive(ei0 + de);
		τ = gasEnergy2Entropy(ρ, ei1);
		τ = gasEntropyUpdate(gasCon);
		Ug.set(ig, gasCon);
		Ur.set(ir, radCon);
	});
}

// GasStateVector radiationRadDiff2GasDiff(RadiationStateVector const &Ur, GasStateVector const &Ug, RadiationStateVector const &dUr_dt,
//										Real dt) {
//	GasStateVector dUg_dt{};
//	using std::pow;
//	FpeGuard fpeGuard{};
//	auto const Γ = opts().gas_gamma;
//	auto const c = physcon().c;
//	auto const ic = 1_R / c;
//	forEach(RadGrid::interior::box, [&](auto idx) {
//		auto const ir = RadGrid::exterior::box.flatten(idx);
//		auto const ig = GasGrid::exterior::box.flatten(idx);
//		auto gas1 = Ug.get(ig);
//		auto const gas0 = gas1;
//		auto &[ρ, Eg, τ, S, _] = gas1;
//		auto const iρ = 1_R / expectPositive(ρ);
//		auto const dE = dUr_dt[er_i][ir] * dt;
//		auto const dF = Vector<Real, NDIM>({dUr_dt[fx_i][ir], dUr_dt[fy_i][ir], dUr_dt[fz_i][ir]}) * dt;
//		Eg -= dE;
//		S -= dF * ic;
//		auto const de = -dE + iρ * ic * dF.dot(S - 0.5_R * ic * dF);
//		τ = pow(expectPositive(pow(τ, Γ) + de), 1_R / Γ);
//		dUg_dt.set(ig, (gas1 - gas0) / dt);
//	});
//	return dUg_dt;
// }
//
// void radiationApply(RadiationStateVector const &Ur0, RadiationStateVector &Ur, RadiationStateVector const &dUr_dt, Real β, Real dt) {
//	FpeGuard fpeGuard{};
//	forEach(RadGrid::interior::box, [&](auto idx) {
//		auto const ir = RadGrid::exterior::box.flatten(idx);
//		for (int f = 0; f <= NDIM; f++) {
//			Ur[f][ir] = (1_R - β) * Ur0[f][ir] + β * (Ur0[f][ir] + dUr_dt[f][ir] * dt);
//		}
//	});
// }
//
// void radiationApply(GasStateVector const &Ug0, GasStateVector &Ug, GasStateVector const &dUg_dt, Real β, Real dt) {
//	FpeGuard fpeGuard{};
//	static auto const fieldCount = opts().n_fields;
//	forEach(RadGrid::interior::box, [&](auto idx) {
//		auto const ir = RadGrid::exterior::box.flatten(idx);
//		for (int f = 0; f < fieldCount; f++) {
//			Ug[f][ir] = (1_R - β) * Ug0[f][ir] + β * (Ug0[f][ir] + dUg_dt[f][ir] * dt);
//		}
//	});
// }
