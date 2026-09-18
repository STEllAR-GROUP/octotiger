// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#pragma once

#include "octotiger/math/Debug.hpp"
#include "octotiger/math/Real.hpp"
#include "octotiger/math/Vector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

// Method references (equation numbers below refer to these papers):
// Hanawa & Audit, JQSRT 145, 9--16 (2014), "Reformulation of the M1 model
// of radiative transfer", https://doi.org/10.1016/j.jqsrt.2014.04.014.
// Skinner & Ostriker, ApJS 206, 21 (2013), "A Two-moment Radiation
// Hydrodynamics Module in Athena Using a Time-explicit Godunov Method",
// https://doi.org/10.1088/0067-0049/206/2/21; https://arxiv.org/abs/1306.0010.
// S&O write radiation energy as ℰ and use physical F. Here E = ℰ and
// Q = F/c, so every ConservedState component has energy-density units.
// We use the full light speed (their ĉ = c); subgrid storage still holds F.
//
// dimensionCount counts transported flux components. The closure retains the
// physical three-dimensional angular moments (P = E I / 3 at isotropy), also
// when a one- or two-dimensional spatial problem is being solved.
// A namespace-like template container: operations on a state belong to that state.
template <typename Type = Real, int dimensionCount = 3>
class RadiationM1 {
	static_assert(dimensionCount > 0);

public:
	RadiationM1() = delete;

	using Scalar = Type;
	using SpatialVector = Vector<Type, dimensionCount>;
	using StateVector = Vector<Type, dimensionCount + 1>;
	static constexpr int fieldCount = dimensionCount + 1;
	static constexpr Type reconstructionTheta = Type(1.3L);
	// Allow accumulated roundoff after conservative cancellation near streaming.
	static constexpr Type roundoff = Type(4096) * std::numeric_limits<Type>::epsilon();

	class Primitives;
	struct Closure;
	struct FluxState;
	struct CoupledState;

	// Hanawa & Audit (42): U = (E, Hβ) = (E, Q), Q = Fphysical/c.
	class ConservedState : public StateVector {
	public:
		using StateVector::StateVector;
		constexpr ConservedState() = default;
		constexpr ConservedState(StateVector const &state) : StateVector(state) {
		}
		constexpr Type &energyDensity() {
			return (*this)[0];
		}
		constexpr Type energyDensity() const {
			return (*this)[0];
		}
		// All components have energy-density units; this returns Fphysical / c.
		constexpr SpatialVector normalizedFlux() const {
			return this->template split<1>().second;
		}
		Primitives toPrimitives() const;
		Closure closure() const;
		CoupledState couple(SpatialVector const &momentum, Type gasEnergy, Type internalEnergy,
			Type rho, Type chiA, Type chiT, Type dt, Type c, Type logAlpha, bool marshak = false) const;
		FluxState physicalFlux(int normal, Type c, Type gridVelocity = 0) const;
		void checkState() const {
#ifndef NDEBUG
			FpeGuard fpeGuard{};
			auto const &u = *this;
			(void) nonNegative(u[0]);
			if (u[0] == 0) {
				for (int d = 1; d <= dimensionCount; ++d) {
					if (u[d] != 0) throw std::runtime_error("Radiation vacuum has nonzero flux");
				}
			} else {
				SpatialVector f;
				for (int d = 0; d < dimensionCount; ++d) f[d] = u[d + 1] / u[0];
				if (!(f.dot(f) <= Type(1) + roundoff)) {
					throw std::runtime_error("Normalized radiation flux exceeds E");
				}
			}
#endif
		}
	};

	// Hanawa & Audit (11), (14): H = E + P, βᵢ = Fᵢ/(cH) = Qᵢ/H.
	// P is the isotropic pressure component; H is not comoving radiation energy.
	// Primitives are (H, β), with |β| ≤ 1.
	class Primitives : public StateVector {
	public:
		using StateVector::StateVector;
		constexpr Primitives() = default;
		constexpr Primitives(StateVector const &state) : StateVector(state) {
		}
		constexpr Type &enthalpy() {
			return (*this)[0];
		}
		constexpr Type enthalpy() const {
			return (*this)[0];
		}
		constexpr SpatialVector beta() const {
			return this->template split<1>().second;
		}
		Type pressure() const {
			FpeGuard fpeGuard{};
			// Hanawa & Audit (20): P = (1 - β²)H/4, β² = Σᵢ βᵢ².
			auto const b = beta();
			return Type(0.25) * enthalpy() * std::max(Type(0), Type(1) - b.dot(b));
		}
		void checkState() const {
#ifndef NDEBUG
			FpeGuard fpeGuard{};
			(void) nonNegative(enthalpy());
			auto const b = beta();
			if (!(b.dot(b) <= Type(1) + roundoff)) {
				throw std::runtime_error("Radiation primitive beta lies outside the unit ball");
			}
#endif
		}
		Closure closure() const;
		FluxState physicalFlux(int normal, Type c, Type gridVelocity = 0) const;
		ConservedState toConserved() const {
			FpeGuard fpeGuard{};
			return conservedFromPrimitives(canonicalPrimitives(*this));
		}

	};

	// Hanawa & Audit (41)--(43): ∂ₜU + c ∂ₙF = 0, U = (E, Hβ).
	// Our ConservedFlux includes c: ℱₙ(U) = c F = (c Qₙ, c Pₙⱼ).
	// This also follows from S&O (20)--(21), ĉ = c, after rescaling Fphysical/c.
	// physicalFlux means the PDE flux of U; it does not convert Q back to F.
	class ConservedFlux : public StateVector {
	public:
		using StateVector::StateVector;
		constexpr ConservedFlux() = default;
		constexpr ConservedFlux(StateVector const &flux) : StateVector(flux) {
		}
	};

	struct Closure {
		Type H, pressure;
		SpatialVector beta;
		Type beta2;
	};

	struct FluxState {
		ConservedFlux flux;
		Type minus, plus;
	};

	struct Energies {
		Type gas, radiation;
	};

	struct CoupledState {
		ConservedState radiation;
		SpatialVector momentum;
		Type gasEnergy, internalEnergy;
	};

	static Type minmodTheta(Type left, Type right, Type theta = reconstructionTheta) {
		FpeGuard fpeGuard{};
		if ((left > 0 && right > 0) || (left < 0 && right < 0)) {
			return std::copysign(std::min({theta * std::abs(left), theta * std::abs(right),
				Type(0.5) * std::abs(left + right)}), left);
		}
		return 0;
	}

	// One face pair per direction. Reconstruct H and beta directly; the
	// conserved-state wrapper below converts the result back to (E,Q). The scalar
	// limiter keeps H nonnegative, and the common
	// vector limiter keeps both reconstructed beta vectors inside the unit ball.
	static std::pair<Primitives, Primitives> reconstructPrimitives(Primitives const &qm,
		Primitives const &q, Primitives const &qp) {
		FpeGuard fpeGuard{};
		q.checkState();
		StateVector slope;
		for (int f = 0; f < fieldCount; ++f) {
			slope[f] = Type(0.5) * minmodTheta(q[f] - qm[f], qp[f] - q[f]);
		}
		auto const beta = q.beta();
		auto const dBeta = slope.template split<1>().second;
		auto const a = dBeta.dot(dBeta);
		auto const b = std::abs(beta.dot(dBeta));
		auto const room = std::max(Type(0), Type(1) - beta.dot(beta));
		if (a + Type(2) * b > room) {
			auto const denominator = std::sqrt(b * b + a * room) + b;
			auto const scale = denominator > 0 ? room / denominator : Type(0);
			for (int d = 1; d <= dimensionCount; ++d) slope[d] *= scale;
		}
		Primitives const minus = q - slope;
		Primitives const plus = q + slope;
		return {canonicalPrimitives(minus), canonicalPrimitives(plus)};
	}

	static std::pair<ConservedState, ConservedState> reconstruct(Primitives const &qm,
		Primitives const &q, Primitives const &qp) {
		FpeGuard fpeGuard{};
		auto const [minus, plus] = reconstructPrimitives(qm, q, qp);
		return {minus.toConserved(), plus.toConserved()};
	}

	// S&O (2013), (38)--(39), with ĉ = c; formula in hllFromFluxes below.
	// Use the extreme Hanawa-Audit characteristics from both reconstructed states.
	static ConservedFlux hll(ConservedState const &left, ConservedState const &right,
		int normal, Type c, Type gridVelocity = 0) {
		FpeGuard fpeGuard{};
		auto const l = left.physicalFlux(normal, c, gridVelocity);
		auto const r = right.physicalFlux(normal, c, gridVelocity);
		return hllFromFluxes(left, right, l, r);
	}

	static ConservedFlux hll(Primitives const &left, Primitives const &right,
		int normal, Type c, Type gridVelocity = 0) {
		FpeGuard fpeGuard{};
		auto const qLeft = canonicalPrimitives(left);
		auto const qRight = canonicalPrimitives(right);
		auto const uLeft = conservedFromPrimitives(qLeft);
		auto const uRight = conservedFromPrimitives(qRight);
		auto const l = fluxFromClosure(uLeft, closureFromPrimitives(qLeft), normal, c, gridVelocity);
		auto const r = fluxFromClosure(uRight, closureFromPrimitives(qRight), normal, c, gridVelocity);
		return hllFromFluxes(uLeft, uRight, l, r);
	}

	// Restrictive multidimensional CFL for PLM + forward Euler; the factor 1/4
	// is an implementation choice, not a coefficient from either paper.
	// gridSpeed is the maximum |vGrid| component on this block and its faces.
	static Type transportTimestep(Type dx, Type c, Type cfl, Type gridSpeed = 0) {
		FpeGuard fpeGuard{};
		(void) positive(dx);
		(void) positive(c);
		(void) positive(cfl);
		(void) nonNegative(gridSpeed);
		return std::min(cfl, Type(0.25)) * dx / (Type(dimensionCount) * (c + gridSpeed));
	}

	// S&O (44a,b), with ĉ = c and χₐ = ρ κ₀:
	//   ∂ₜe = -c χₐ(αe⁴ - ℰ),  ∂ₜℰ = c χₐ(αe⁴ - ℰ),
	//   α = aᵣ[(γ - 1)μ/(ρ k_B)]⁴,  ℰ_eq = αe⁴ = aᵣT⁴.
	// Here logAlpha = ln(α); μ is the mean particle mass. Freeze opacities
	// for the step and use θ = 1 in (47)--(49) (backward Euler).
	// The Marshak test substitutes ℰ_eq = e for this ideal-gas relation.
	static Energies thermalExchange(Type e, Type E, Type eta, Type logAlpha,
		bool marshak = false) {
		FpeGuard fpeGuard{};
		(void) nonNegative(E);
		(void) nonNegative(eta);
		if (eta == 0) return {nonNegative(e), E};
		// Rearrangement of S&O (49): x + w αx⁴ = r, with x = eⁿ⁺¹,
		// η = c χₐ Δt, w = η/(1 + η), r = eⁿ + w ℰⁿ.
		// From (47)--(48), ℰⁿ⁺¹ = ℰⁿ/(1 + η) + w αx⁴.
		auto const inverse = Type(1) / (Type(1) + eta);
		auto const weight = eta * inverse;
		auto const rhs = nonNegative(e + weight * E);
		if (marshak) {
			auto const gas = rhs / (Type(1) + weight);
			return {gas, inverse * E + weight * gas};
		}
		if (rhs == 0) return {0, inverse * E};
		auto const logCoefficient = std::log(weight) + logAlpha;
		auto const logRhs = std::log(rhs);
		auto const quarticLogScale = (logRhs - logCoefficient) / Type(4);
		auto const linearBound = logRhs <= quarticLogScale;
		auto const scale = linearBound ? rhs : std::exp(quarticLogScale);
		// Implementation scaling of that quartic: x = s y, s = min(r, (r/(wα))¼),
		// so (s/r)y + (wαs⁴/r)y⁴ = 1, with both coefficients ≤ 1.
		// The active bounding coefficient must be exactly one: exp(log(rhs))
		// can round below rhs and put a nearly linear root outside [0,1].
		auto const linear = linearBound ? Type(1) : scale / rhs;
		auto const quartic = linearBound ? std::exp(std::min(Type(0), logCoefficient + Type(3) * logRhs)) : Type(1);
		Type lo = 0, hi = 1, y = 1;
		for (int iteration = 0; iteration < 80; ++iteration) {
			auto const y2 = y * y;
			auto const residual = linear * y + quartic * y2 * y2 - Type(1);
			if (std::abs(residual) <= Type(16) * std::numeric_limits<Type>::epsilon()) {
				return {scale * y, inverse * E + rhs * quartic * y2 * y2};
			}
			if (residual > 0) hi = y;
			else lo = y;
			auto const next = y - residual / (linear + Type(4) * quartic * y2 * y);
			y = next > lo && next < hi ? next : Type(0.5) * (lo + hi);
		}
		throw std::runtime_error("Radiation thermal exchange did not converge");
	}

private:
	static Primitives canonicalPrimitives(Primitives q) {
		FpeGuard fpeGuard{};
		q.checkState();
		// Vacuum has no preferred direction; use the same isotropic speeds
		// whether a face enters the solver as conserved or primitive variables.
		if (q.enthalpy() == 0) return {};
		auto const beta = q.beta();
		auto const beta2 = beta.dot(beta);
		if (beta2 > 1) {
			auto const scale = Type(1) / std::sqrt(beta2);
			for (int d = 1; d <= dimensionCount; ++d) q[d] *= scale;
		}
		return q;
	}

	// Both helpers consume the same already-canonical primitive state, so the
	// HLL conserved jump and physical flux use identical beta components.
	static ConservedState conservedFromPrimitives(Primitives const &q) {
		FpeGuard fpeGuard{};
		auto const beta = q.beta();
		auto const beta2 = std::min(Type(1), beta.dot(beta));
		// Hanawa & Audit (19), (14): E = (3 + β²)H/4, Qᵢ = Hβᵢ.
		auto const energy = q.enthalpy() * (Type(0.25) * (Type(3) + beta2));
		ConservedState const result = concatenate(energy, beta * q.enthalpy());
		result.checkState();
		return result;
	}

	static Closure closureFromPrimitives(Primitives const &q) {
		FpeGuard fpeGuard{};
		// Hanawa & Audit (17), (20): Pᵢⱼ = Hβᵢβⱼ + Pδᵢⱼ,
		// P = (1 - β²)H/4. Store the factors; form a pressure row only as needed.
		auto const beta = q.beta();
		auto const beta2 = std::min(Type(1), beta.dot(beta));
		return {q.enthalpy(), Type(0.25) * (Type(1) - beta2) * q.enthalpy(), beta, beta2};
	}

	static FluxState fluxFromClosure(ConservedState const &u, Closure const &m,
		int normal, Type c, Type gridVelocity) {
		FpeGuard fpeGuard{};
		(void) positive(c);
		if (normal < 0 || normal >= dimensionCount) {
			throw std::out_of_range("Radiation flux normal is outside the spatial dimensions");
		}
		auto const bn = m.beta[normal];
		Type transverse2 = 0;
		for (int d = 0; d < dimensionCount; ++d) {
			if (d != normal) transverse2 += m.beta[d] * m.beta[d];
		}
		// Hanawa & Audit (51), (53), with their z replaced by this face normal n:
		// λ₁,₄ = c[2βₙ ± √((1 - β²)(3 - β² - 2βₙ²))]/(3 - β²).
		// Their λ₁ is plus, λ₄ is minus. Evaluate the equivalent radicand
		// (1 - β²)[3(1 - β²) + 2β⊥²], β⊥² = Σⱼ≠ₙ βⱼ², to avoid cancellation.
		auto const oneMinusBeta2 = Type(1) - m.beta2;
		auto const root = std::sqrt(oneMinusBeta2 * (Type(3) * oneMinusBeta2 + Type(2) * transverse2));
		FluxState result;
		// Moving-grid transformation of the published stationary-grid system:
		// λ_grid = λ - v_grid,n, ℱ_grid = ℱₙ - v_grid,n U.
		result.minus = c * (Type(2) * bn - root) / (Type(3) - m.beta2) - gridVelocity;
		result.plus = c * (Type(2) * bn + root) / (Type(3) - m.beta2) - gridVelocity;
		result.flux[0] = c * u[normal + 1] - gridVelocity * u[0];
		for (int d = 0; d < dimensionCount; ++d) {
			// Hanawa & Audit (17): Pₙⱼ = Hβₙβⱼ + Pδₙⱼ; ℱₙ from (41)--(43).
			auto const pressureNd = m.H * bn * m.beta[d] + (d == normal ? m.pressure : Type(0));
			result.flux[d + 1] = c * pressureNd - gridVelocity * u[d + 1];
		}
		return result;
	}

	static ConservedFlux hllFromFluxes(ConservedState const &left, ConservedState const &right,
		FluxState const &l, FluxState const &r) {
		FpeGuard fpeGuard{};
		// S&O (39): ℱ_HLL = [Sᴿ⁺ℱᴸ - Sᴸ⁻ℱᴿ + Sᴿ⁺Sᴸ⁻(Uᴿ - Uᴸ)]
		//                         / (Sᴿ⁺ - Sᴸ⁻),
		// Sᴸ = min(λ_minᴸ, λ_minᴿ), Sᴿ = max(λ_maxᴸ, λ_maxᴿ),
		// Sᴸ⁻ = min(Sᴸ, 0), Sᴿ⁺ = max(Sᴿ, 0). The upwind cases (38)
		// are handled first; the mixed case below uses equivalent weights.
		auto const sm = std::min(l.minus, r.minus);
		auto const sp = std::max(l.plus, r.plus);
		// Also handles coincident zero-speed waves without a 0/0 denominator.
		if (sm >= 0) return l.flux;
		if (sp <= 0) return r.flux;
		auto const wr = sp / (sp - sm);
		auto const wl = -sm / (sp - sm);
		auto const viscosity = sm * wr;
		return wr * l.flux + wl * r.flux + viscosity * (right - left);
	}

	// Scalar checks avoid legacy printf-based diagnostics, which assume double
	// arguments and are not valid for a long-double instantiation.

	static Type positive(Type value) {
#ifndef NDEBUG
		FpeGuard fpeGuard{};
		if (!(value > 0)) throw std::runtime_error("Expected positive radiation value");
#endif
		return value;
	}
	static Type nonNegative(Type value) {
#ifndef NDEBUG
		FpeGuard fpeGuard{};
		if (!(value >= 0)) throw std::runtime_error("Expected nonnegative radiation value");
#endif
		return value;
	}
};

template <typename Type, int dimensionCount>
auto RadiationM1<Type, dimensionCount>::ConservedState::toPrimitives() const -> Primitives {
	FpeGuard fpeGuard{};
	checkState();
	if (energyDensity() == 0) return {};
	SpatialVector f;
	for (int d = 0; d < dimensionCount; ++d) f[d] = (*this)[d + 1] / energyDensity();
	auto f2 = f.dot(f);
	// An admissible conserved state can exceed |Q|/E=1 only by roundoff.
	if (f2 > 1) {
		f *= Type(1) / std::sqrt(f2);
		f2 = Type(1);
	}
	// Hanawa & Audit (8), (16): fᵢ = Qᵢ/E, βᵢ = 3fᵢ/[2 + √(4 - 3f²)].
	// Rearranging (14) gives H/E = [2 + √(4 - 3f²)]/3, finite also at f = 0.
	auto const hOverE = (Type(2) + std::sqrt(Type(4) - Type(3) * f2)) / Type(3);
	return concatenate(energyDensity() * hOverE, f / hOverE);
}

template <typename Type, int dimensionCount>
auto RadiationM1<Type, dimensionCount>::ConservedState::physicalFlux(int normal, Type c,
	Type gridVelocity) const -> FluxState {
	FpeGuard fpeGuard{};
	return fluxFromClosure(*this, closure(), normal, c, gridVelocity);
}

template <typename Type, int dimensionCount>
auto RadiationM1<Type, dimensionCount>::Primitives::physicalFlux(int normal, Type c,
	Type gridVelocity) const -> FluxState {
	FpeGuard fpeGuard{};
	// The reconstructed primitive state supplies the closure directly.
	auto const q = canonicalPrimitives(*this);
	return fluxFromClosure(conservedFromPrimitives(q), closureFromPrimitives(q),
		normal, c, gridVelocity);
}

template <typename Type, int dimensionCount>
auto RadiationM1<Type, dimensionCount>::ConservedState::closure() const -> Closure {
	FpeGuard fpeGuard{};
	return closureFromPrimitives(toPrimitives());
}

template <typename Type, int dimensionCount>
auto RadiationM1<Type, dimensionCount>::Primitives::closure() const -> Closure {
	FpeGuard fpeGuard{};
	return closureFromPrimitives(canonicalPrimitives(*this));
}

// S&O (5a), through first order in gas v/c, with κ₀E = κ₀P:
//   ∂ₜℰ = c χₐ(aᵣT⁴ - ℰ) + (2χₐ - χₜ) v·Q.
// Here χₐ = ρ κ₀P (chiA), χₜ = ρ κ₀F (chiT), both inverse lengths.
// S&O (8e), generalized from their equal means to χₜ, supplies
//   ∂ₜQ = -c χₜ Q + χₜ(ℰ I + P)·v.
// As in (8e), this uses aᵣT⁴ ≈ ℰ in the velocity-dependent momentum terms;
// it is not the full unequal-opacity expression (5b). We keep the full
// pressure tensor rather than the isotropic substitution (18).
// Gas v/c here is distinct from the radiation primitive β. The velocity
// terms must be non-stiff; a light-crossing CFL alone does not guarantee this.
template <typename Type, int dimensionCount>
auto RadiationM1<Type, dimensionCount>::ConservedState::couple(SpatialVector const &momentum,
	Type gasEnergy, Type internalEnergy, Type rho, Type chiA, Type chiT, Type dt,
	Type c, Type logAlpha, bool marshak) const -> CoupledState {
	FpeGuard fpeGuard{};
	auto const &u = *this;
	checkState();
	(void) positive(c);
	(void) positive(rho);
	(void) nonNegative(internalEnergy);
	(void) nonNegative(chiA);
	(void) nonNegative(chiT);
	(void) nonNegative(dt);
	if (dt == 0 || (chiA == 0 && chiT == 0)) {
		return {u, momentum, gasEnergy, internalEnergy};
	}
	auto const velocity = momentum / rho;
	auto const Q = u.normalizedFlux();
	auto const m = closure();
	auto const eta = nonNegative(dt * c * chiT);
	auto const inverse = Type(1) / (Type(1) + eta);
	auto const weight = eta * inverse;
	auto const betaV = m.beta.dot(velocity);
	CoupledState result;
	Type kineticChange = 0;
	// Backward-Euler damping (S&O 43, θ = 1), with the (8e) velocity source
	// evaluated at the old state: Qⁿ⁺¹ = [Qⁿ + Δt χₜ(ℰ I + P)·v]/(1 + c χₜ Δt).
	// Opposite gas increments conserve m + Q/c (S&O 8b,e with ĉ = c).
	for (int d = 0; d < dimensionCount; ++d) {
		auto const advectiveFlux = ((u[0] + m.pressure) * velocity[d] + m.H * m.beta[d] * betaV) / c;
		result.radiation[d + 1] = inverse * Q[d] + weight * advectiveFlux;
		auto const dm = (Q[d] - result.radiation[d + 1]) / c;
		result.momentum[d] = momentum[d] + dm;
		kineticChange += dm * (velocity[d] + Type(0.5) * dm / rho);
	}
	auto const explicitWork = dt * (Type(2) * chiA - chiT) * velocity.dot(Q);
	auto const eRadiationStar = u[0] + explicitWork;
	auto const eGasStar = internalEnergy - explicitWork - kineticChange;
	// No clipping of a failed explicit update: that would destroy conservation.
	if (!(eRadiationStar >= 0) || !(eGasStar + (dt * c * chiA / (Type(1) + dt * c * chiA)) * eRadiationStar >= 0)) {
		throw std::runtime_error("Radiation explicit source step exhausted the available energy; decrease the shared timestep");
	}
	auto const energy = thermalExchange(eGasStar, eRadiationStar, dt * c * chiA, logAlpha, marshak);
	result.radiation[0] = energy.radiation;
	result.gasEnergy = gasEnergy + (u[0] - energy.radiation);
	result.internalEnergy = energy.gas;
	result.radiation.checkState();
	(void) nonNegative(result.internalEnergy);
	return result;
}
