#pragma once

#include "octotiger/defs.hpp"
#include "octotiger/gas/GasEoS.hpp"
#include "octotiger/math/Arithmetic.hpp"
#include "octotiger/math/Math.hpp"
#include "octotiger/math/Matrix.hpp"
#include "octotiger/math/Real.hpp"
#include "octotiger/math/Vector.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"

#include <cmath>

#define MAX_SPECIES_COUNT (OCTOTIGER_MAX_NUMBER_FIELDS - spc_i)
// Α Β Γ Δ Ε Ζ Η Θ Ι Κ Λ Μ Ν Ξ Ο Π Ρ Σ Τ Υ Φ Χ Ψ Ω
// α β γ δ ε ζ θ ι κ λ η μ ν ξ ο π ρ σ ς τ υ φ χ ψ ω
// ₐ ₑ ₕ ᵢ ⱼ ₖ ₗ ₘ ₙ ₒ ₚ ᵣ ₛ ₜ ᵤ ᵥ ₓ
// ᵃ ᵇ ᶜ ᵈ ᵉ ᶠ ᵍ ʰ ⁱ ʲ ᵏ ˡ ᵐ ⁿ ᵒ ᵖ ʳ ˢ ᵗ ᵘ ᵛ ʷ ˣ ʸ ᶻ
// ᴬ ᴮ ᴰ ᴱ ᴳ ᴴ ᴵ ᴶ ᴷ ᴸ ᴹ ᴺ ᴼ ᴾ ᴿ ᵀ ᵁ ⱽ ᵂ
// ᵅ ᵝ ᵞ ᵟ ᵋ ᶿ ᶥ ᶲ ᵡ
template <typename T, int fieldCount>
struct Eigensystem {
	Vector<T, fieldCount> λ;
	Matrix<T, fieldCount> L;
	Matrix<T, fieldCount> R;
};

enum class RiemannSolverType { LLF, HLL, HLLC, Roe };

template <typename T, int dimCount>
struct PrimitiveGasState {
	T ρ;
	Vector<T, dimCount> v;
	T ε;
	T κ;
};

template <typename T, int dimCount>
struct ConservedGasState {
	T D;
	Vector<T, dimCount> S;
	T E;
	T K;
};

template <typename T, int dimCount>
using ConservedGasFlux = ConservedGasState<T, dimCount>;

template <typename T>
auto pressure(T ρ, T ε) {
	auto const &Γ = opts().gas_gamma;
	auto const p = (Γ - T(1)) * ρ * ε;
	return expectPositive(p);
}

template <typename T>
auto entropy2energy(T ρ, T κ) {
	auto const &Γ = opts().gas_gamma;
	auto const p = κ * pow(ρ, Γ);
	auto const ε = p / ((Γ - T(1)) * ρ);
	return expectPositive(ε);
}

template <typename T>
auto energy2entropy(T ρ, T ε) {
	auto const &Γ = opts().gas_gamma;
	auto const p = pressure(ρ, ε);
	auto const κ = p * pow(ρ, -Γ);
	return expectPositive(κ);
}

template <typename T, int dimCount>
auto con2prim(ConservedGasState<T, dimCount> const &U) {
	static auto const &δ = opts().dual_energy_sw1;
	PrimitiveGasState<T, dimCount> V;
	auto const &[D, S, E, K] = U;
	auto &[ρ, v, ε, κ] = V;
	ρ = expectPositive(D);
	v = S / ρ;
	κ = expectPositive(K) / ρ;
	auto const ek = T(0.5) * v.dot(S);
	auto const ei = U.E - ek;
	ε = (ei > δ * E) ? (ei / ρ) : entropy2energy(ρ, κ);
	return ε;
}

template <typename T, int dimCount>
auto prim2con(PrimitiveGasState<T, dimCount> const &V) {
	static auto const &δ = opts().dual_energy_sw2;
	ConservedGasState<T, dimCount> U;
	auto const &[ρ, v, ε, κ] = V;
	auto &[D, S, E, K] = U;
	D = expectPositive(ρ);
	S = ρ * v;
	auto const ek = T(0.5) * v.dot(S);
	auto const ei = ρ * ε;
	E = ei + ek;
	K = ρ * ((ei > δ * E) ? energy2entropy(ρ, ε) : κ);
	return U;
}

template <typename T, int dimCount>
auto flux(PrimitiveGasState<T, dimCount> const &V, int k) {
	auto const δk = Vector<T, dimCount>::unit(k);
	auto const [ρ, v, ε, κ] = V;
	auto const &u = v[k];
	auto const p = pressure(ρ, ε);
	auto const h = ε + p / ρ;
	auto const v2 = v.dot(v);
	return ConservedGasFlux<T, dimCount>{ρ * u, ρ * v * u + δk * p, ρ * (h + v2) * u, ρ * κ * u};
}

template <typename T, int dimCount>
auto flux(ConservedGasState<T, dimCount> const &U, int k) {
	return flux(con2prim(U), k);
}

template <typename T, int dimCount>
auto eigensystem(PrimitiveGasState<T, dimCount> V, int k) {
	using namespace std;
	auto const fieldCount = dimCount + 3;
	auto const transverseCount = dimCount - 1;
	Eigensystem<T, fieldCount> Λ{};
	auto const zero = Vector<T, transverseCount>(T(0));
	auto const delta = identityMatrix<T, transverseCount>();
	auto const &Γ = opts().gas_gamma;
	swap(V.v[0], V.v[k]);
	auto const& [ρ, v, ε, κ] = V;
	auto const [u, w] = split<1, transverseCount>(v);
	auto const q = T(0.5) * v.dot(v);
	auto const p = pressure(ρ, ε);
	auto const iρ = T(1) / expectPositive(ρ);
	auto const h = q + ε + p * iρ;
	auto const a = sqrt((Γ - T(1)) * (h - q));
	auto const acoustic1 = Vector<T, fieldCount>{T(1), u,    w, h + u * a, κ};
	auto const contact = Vector<T, fieldCount>{T(1), u, v, w, q, κ};
	auto const acoustic2 = Vector<T, fieldCount>{T(0), a, zero, u * a, T(0)};
	int i = 0;
	Λ.R[i++] = acoustic1 - acoustic2;
	Λ.R[i++] = contact;
	Λ.R[i++] = {T(0), zero, T(1)};
	for (int t = 0; t < transverseCount; t++) {
		Λ.R[i++] = {T(0), delta[t], w[t]};
	}
	Λ.R[i++] = acoustic1 + acoustic2;
	Λ.R = transpose(Λ.R);
	swap(Λ.R[sx_i], Λ.R[sx_i + k]);
	Λ.L = inv(Λ.R);
	Λ.λ = Vector<T, fieldCount>{T(0)};
	Λ.λ.front() -= a;
	Λ.λ.back() += a;
	return Λ;
}

template <typename Type>
struct GasState : Vector<Type, NDIM + 3> {
	static constexpr auto fieldCount = NDIM + 3;
	using base_type = Vector<Type, fieldCount>;

public:
	auto &massDensity() {
		return (*this)[rho_i];
	}
	auto &energyDensity() {
		return (*this)[egas_i];
	}
	auto &entropyTracer() {
		return (*this)[tau_i];
	}
	auto &momentumDensity() {
		auto &s = (*this)[sx_i];
		return Vector<Type, NDIM, Type *>(&s);
	}
	auto massDensity() const {
		return expectPositive((*this)[rho_i]);
	}
	auto energyDensity() const {
		return (*this)[egas_i];
	}
	auto entropyTracer() const {
		return expectPositive((*this)[tau_i]);
	}
	auto conservedVariables() const {
		return std::tuple(massDensity(), momentumDensity(), energyDensity(), entropyTracer());
	}
	auto primitiveVariables() const {
		using namespace std;
		auto const Γ = adiabaticIndex();
		auto const δ = dualEnergySwitch().first;
		auto const [D, S, E, ρK] = conservedVariables();
		auto const &ρ = D;
		auto const iρ = 1_R / ρ;
		auto const v = S * iρ;
		auto const ek = 0.5_R * v.dot(S);
		auto const K = ρK * iρ;
		auto const ei = (ek < E * (1_R - δ)) ? (E - ek) : (K * pow(ρ, Γ) / (Γ - 1_R));
		auto const ε = iρ * ei;
		return PrimitiveState{ρ, v, ε, K};
	}
	auto eigenvalues(int n) const {
		using namespace std;
		auto const [ρ, v, ε, _] = primitiveVariables();
		return eigenvalues(ρ, v, pressure(ρ, ε), n);
	}
	auto eigensystem(int n) const {
		using namespace std;
		auto const Γ = adiabaticIndex();
		Eigensystem<Type, fieldCount> Λ;
		auto &R = Λ.R;
		auto const i = (n + 1) % NDIM;
		auto const j = (n + 2) % NDIM;
		auto const [D, S, E, ρK] = std::tuple(massDensity(), momentumDensity(), energyDensity(), entropyTracer());
		auto const [ρ, q, ε, K] = primitiveVariables();
		auto const [u, v, w] = std::tie(q[n], q[i], q[j]);
		auto const p = pressure(ρ, ε);
		auto const a = soundSpeed(ρ, ε);
		auto const iρ = 1_R / ρ;
		auto const h = iρ * (E + p);
		auto const q2 = sqr(q);
		constexpr auto zero = Type(0_R), half = Type(0.5_R), one = Type(1_R); // clang-format off
		R[rho_i] =    {one,        one,       zero, zero, zero, one      };
		R[sx_i + n] = {u - a,      u,         zero, zero, zero, u + a    };
		R[sx_i + i] = {v,          v,         one,  zero, zero, v,       };
		R[sx_i + j] = {w,          w,         zero, one,  zero, w,       };
		R[egas_i] =   {h - u * a,  half * q2, v,    w,    zero, h + u * a};
		R[tau_i] =    {K,          K,         zero, zero, one,  K         }; // clang-format on
		Λ.λ = eigenvalues(ρ, q, ε, n);
		Λ.L = inverse(R);
		return Λ;
	}
	auto flux(int n) const {
		static auto const δ = Matrix<Type, NDIM>::identity();
		auto const [D, S, E, K] = std::tuple(massDensity(), momentumDensity(), energyDensity(), entropyTracer());
		auto const [ρ, v, ε, k] = primitiveVariables();
		auto const p = pressure(ρ, ε);
		return GasState<Type>{D * v[n], S * v[n] + p * δ[n], (E + p) * v[n], K * v[n]};
	}
	template <RiemannSolverType solver = RiemannSolverType::HLL>
	friend auto riemannSolver(GasState<Type> const &UL, GasState<Type> const &UR, int n) {
		using namespace std;
		auto const Γ = adiabaticIndex();
		auto const FL = UL.flux(n);
		auto const FR = UR.flux(n);
		if (solver == RiemannSolverType::Roe) {
			auto const [ρL, vL, εL, kL] = UL.primitiveVariables();
			auto const [ρR, vR, εR, kR] = UR.primitiveVariables();
			auto const pR = pressure(ρR, εR);
			auto const pL = pressure(ρL, εL);
			auto const hR = εR + pR / ρR + 0.5_R * sqr(vR);
			auto const hL = εL + pL / ρL + 0.5_R * sqr(vL);
			auto const wR = sqrt(ρR);
			auto const wL = sqrt(ρL);
			auto const ρ = wR * wL;
			auto const h = (wR * hR + wL * hL) / (wR + wL);
			auto const k = (wR * kR + wL * kL) / (wR + wL);
			auto const v = (wR * vR + wL * vL) / (wR + wL);
			auto const p = ρ * ((Γ - 1_R) / Γ) * (h - 0.5_R * sqr(v));
			auto const U = GasState<Type>{ρ, ρ * v, ρ * h - p, ρ * k};
			auto [λ, L, R] = U.eigensystem(n);
			std::transform(λ.begin(), λ.end(), λ.begin(), abs);
			return 0.5_R * (FL + FR) - 0.5_R * R * λ * L * (UR - UL);
		} else if (solver == RiemannSolverType::LLF) {
			auto const λL = UL.eigenvalues(n);
			auto const λR = UR.eigenvalues(n);
			auto const sL = min(min(λL.front(), λR.front()), 0_R);
			auto const sR = max(max(λL.back(), λR.back()), 0_R);
			auto const a = max(-sL, +sR);
			return 0.5_R * (FL + FR) - 0.5_R * a * (UR - UL);
		} else if (solver == RiemannSolverType::HLL) {
			auto const λL = UL.eigenvalues(n);
			auto const λR = UR.eigenvalues(n);
			auto const sL = min(min(λL.front(), λR.front()), 0_R);
			auto const sR = max(max(λL.back(), λR.back()), 0_R);
			return (sR * FL - sL * FR + sL * sR * (UR - UL)) / (sR - sL);
		} else if (solver == RiemannSolverType::HLLC) {
			auto const i = (n + 1) % NDIM;
			auto const j = (n + 2) % NDIM;
			auto const C0 = (Γ + 1_R) / (2_R * Γ);
			auto const &EL = UL.energyDensity();
			auto const &ER = UR.energyDensity();
			auto const [ρL, vL, εL, kL] = UL.primitiveVariables();
			auto const [ρR, vR, εR, kR] = UR.primitiveVariables();
			auto const pL = pressure(ρL, εL);
			auto const pR = pressure(ρR, εR);
			auto const aL = sqrt(Γ * pL / ρL);
			auto const aR = sqrt(Γ * pR / ρR);
			auto const pStar = max(Type(0), Type(0.5_R) * (pL + pR) - Type(0.125_R) * (vR[n] - vL[n]) * (ρL + ρR) * (aL + aR));
			auto const qL = sqrt(1_R + C0 * max(0_R, pStar / pL - 1_R));
			auto const qR = sqrt(1_R + C0 * max(0_R, pStar / pR - 1_R));
			auto const sL = min(0_R, vL[n] - aL * qL);
			auto const sR = max(0_R, vR[n] + aR * qR);
			auto const num = pR - pL + ρL * vL[n] * (sL - vL[n]) - ρR * vR[n] * (sR - vR[n]);
			auto const den = ρL * (sL - vL[n]) - ρR * (sR - vR[n]);
			auto const sStar = num / den;
			bool const flag = sStar >= 0_R;
			auto const &ρK = flag ? ρL : ρR;
			auto const &pK = flag ? pL : pR;
			auto const &sK = flag ? sL : sR;
			auto const &EK = flag ? EL : ER;
			auto const &uK = flag ? vL[n] : vR[n];
			auto const &vK = flag ? vL[i] : vR[i];
			auto const &wK = flag ? vL[j] : vR[j];
			auto const &kK = flag ? kL : kR;
			auto const &FK = flag ? FL : FR;
			auto const &UK = flag ? UL : UR;
			auto const ρStar = ρK * (sK - uK) / (sK - sStar);
			GasState<Type> Vstar;
			Vstar[rho_i] = 1_R;
			Vstar[sx_i + n] = sStar;
			Vstar[sx_i + i] = vK;
			Vstar[sx_i + j] = wK;
			Vstar[egas_i] = EK / ρK + (sStar - uK) * (sStar + pK / (ρK * (sK - uK)));
			Vstar[tau_i] = kK;
			auto const Ustar = ρStar * Vstar;
			return FK + sK * (Ustar - UK);
		}
		throw std::runtime_error("Unknown gas Riemann solver type");
	}
	static auto pressure(Type ρ, Type ε) {
		auto const Γ = adiabaticIndex();
		return (Γ - 1_R) * ρ * ε;
	}
	static auto soundSpeed(Type ρ, Type ε) {
		auto const Γ = adiabaticIndex();
		using namespace std;
		auto const c2 = expectPositive((Γ - 1_R) * Γ * ε);
		return sqrt(c2);
	}
	static auto eigenvalues(Type ρ, Vector<Type, NDIM> const &v, Type ε, int n) {
		using namespace std;
		auto const a = soundSpeed(ρ, ε);
		auto const &u = v[n];
		return Vector<Type, fieldCount>{u - a, u, u, u, u, u + a};
	}

	static auto const &adiabaticIndex() {
		static auto const Γ = opts().gas_gamma;
		return Γ;
	}
	static auto const &dualEnergySwitch() {
		static auto const switches = std::pair(opts().dual_energy_sw1, opts().dual_energy_sw2);
		return switches;
	}
};

template <typename Type, int speciesCount>
struct GasComposition : public Vector<Real, speciesCount> {
	static constexpr auto fieldCount = MAX_SPECIES_COUNT;
	using base_type = Vector<Real, fieldCount>;
	Type massDensity(int s) const {
		return base_type::operator[](s);
	}
	Type massDensity() const {
		return std::accumulate(base_type::begin(), base_type::end(), 0_R);
	}
	Type numberDensity(int s) const {
		auto const μ = meanMolecularWeight(s);
		auto const m = atomicMassUnit();
		auto const ρ = massDensity(s);
		if (μ == 0_R) return Type(0);
		return ρ / (m * μ);
	}
	Type numberDensity() const {
		Type n = 0_R;
		for (int s = 0; s < speciesCount; s++) {
			n += numberDensity(s);
		}
		return n;
	}
	Type meanMolecularWeight() const {
		Type ρiμ = 0_R;
		Type ρ = 0_R;
		for (int j = 0; j < speciesCount; j++) {
			auto const μj = meanMolecularWeight(j);
			auto const ρj = massDensity(j);
			ρ += ρj;
			ρiμ += ρj / μj;
		}
		return ρ / ρiμ;
	}
	static Type meanMolecularWeight(int s) {
		auto const [A, Z, m] = std::tuple(atomicMass(s), atomicNumber(s), atomicMassUnit());
		return A / (Z + 1_R);
	}
	static Type atomicMass(int s) {
		static auto const A = opts().atomic_number;
		return (s < opts().n_species) ? A[s] : 0_R;
	}
	static Type atomicNumber(int s) {
		static auto const Z = opts().atomic_mass;
		return (s < opts().n_species) ? Z[s] : 0_R;
	}
	static Type &atomicMassUnit() {
		static auto const amu = physcon().mh;
		return amu;
	}
};

// static Scalar const &kB;
// static Scalar const &amu;
// static Vector<Scalar, MAX_SPECIES_COUNT> const A;
// static Vector<Scalar, MAX_SPECIES_COUNT> const Z;
//
// template <typename Scalar>
// Vector<Scalar, MAX_SPECIES_COUNT> const GasState<Scalar>::A = []() {
//	Vector<Scalar, MAX_SPECIES_COUNT> A;
//	auto const &a = opts().atomic_mass;
//	for (int s = 0; s < opts().n_species; s++) {
//		A[s] = a[s];
//	}
//	for (int s = opts().n_species; s < MAX_SPECIES_COUNT; s++) {
//		A[s] = 1_R;
//	}
// }();
//
// template <typename Scalar>
// Vector<Scalar, MAX_SPECIES_COUNT> const GasState<Scalar>::Z = []() {
//	Vector<Scalar, MAX_SPECIES_COUNT> Z;
//	auto const &z = opts().atomic_number;
//	for (int s = 0; s < opts().n_species; s++) {
//		Z[s] = z[s];
//	}
//	for (int s = opts().n_species; s < MAX_SPECIES_COUNT; s++) {
//		Z[s] = 1_R;
//	}
// }();

// template <typename Scalar>
// Scalar const &GasState<Scalar>::amu = physcon().mh;
//
// template <typename Scalar>
// Scalar const &GasState<Scalar>::kB = physcon().kb;

// template <typename Scalar>
// GasState<Scalar> PrimitiveGasState<Scalar>::toConserved() const {
//	GasState<Scalar> cons;
//	NUMBER_CONSTANTS();
//	GAS_CONSTANTS();
//	ACCESS_MEMBERS(GASCON_MEMBERS, cons);
//	D = expectPositive(ρ);
//	S = ρ * v;
//	E = ρ * ε + half * S.dot(v);
//	auto const n = nₛ.sum();
//	auto const e = ((ρ * ε > dualEnergySwitch2 * E) ? (ρ * ε) : kb * n * T / (Γ - one));
//	K = gasEnergy2Entropy(ρ, e);
//	for (int s = 0; s < opts().n_species; s++) {
//		Dₛ[s] = expectNonNegative(ρ * nₛ[s] * (mh * A[s]) / (1_R + Z[s]));
//	}
//	return cons;
// }

// template <typename Scalar>
// PrimitiveGasState<Scalar> GasState<Scalar>::toPrimitive() const {
//	PrimitiveGasState<Scalar> prims;
//	NUMBER_CONSTANTS();
//	GAS_CONSTANTS();
//	ACCESS_MEMBERS(GASPRIM_MEMBERS, prims);
//	ρ = D;
//	auto const iρ = one / expectPositive(ρ);
//	v = iρ * S;
//	auto const e = E - half * v.dot(S);
//	ε = iρ * ((e > dualEnergySwitch1 * E) ? e : gasEntropy2Energy(D, K));
//	for (int s = 0; s < opts().n_species; s++) {
//		nₛ[s] = expectNonNegative(D * (mh * A[s]) / (1_R + Z[s])) * iρ;
//	}
//	for (int s = opts().n_species; s < MAX_SPECIES_COUNT; s++) {
//		nₛ[s] = 0_R;
//	}
//	auto const n = nₛ.sum();
//	auto const p = ;
//	T = p / (kb * n);
//	return prims;
// }

//
// ThermodynamicState GasState::thermodynamics() const {
//	constexpr auto zero = Real(0.0_R), half = Real(0.5_R), one = Real(1.0_R);
//	static auto const [A, Z, Γ, kb, mh, δ, _] = GasConstants{};
//	auto const ρ = expectPositive(D.sum());
//	auto const iρ = one / ρ;
//	auto iμ = zero;
//	for (int s = 0; s < opts().n_species; ++s) {
//		iμ += expectNonNegative(D[s] * (1_R + Z[s]) / A[s]);
//	}
//	auto const μ = ρ / expectPositive(iμ);
//	auto const e = E - half * iρ * S.dot(S);
//	auto const ε = iρ * ((e > δ * E) ? e : gasEntropy2Energy(D, K));
//	auto const icv = ((Γ - one) * mh * μ) / kb;
//	auto const T = ε * icv;
//	return ThermodynamicState{ρ, μ, T};
// }

//
// GasState PrimitiveGasState::toConserved() const {
//	constexpr auto zero = Real(0.0_R), half = Real(0.5_R), one = Real(1.0_R);
//	static auto const [A, Z, Γ, kb, mh, _, δ] = GasConstants{};
//	auto const in = one / expectPositive(n);
//	auto ρ = zero;
//	for (int s = 0; s < opts().n_species; ++s) {
//		auto const dρ = expectNonNegative(n * X[s] * (mh * A[s]) / (1_R + Z[s]));
//		ρ += dρ;
//	}
//	auto const ei = kb * n * T / (Γ - one);
//	auto const S = ρ * v;
//	auto const v2 = v.dot(v);
//	auto const E = ei + half * ρ * v2;
//	auto const K = ((ei > δ * E) ? pow(ei, one / Γ) : ρ * s);
//	auto const ρₛ = ρ * X;
//	return GasState{ρ, E, K, S, ρₛ};
// }

struct GasStateVector {
	using storage_type = std::vector<std::vector<Real>>;
	using const_iterator = storage_type::const_iterator;
	using const_reference = storage_type::const_reference;
	using iterator = storage_type::iterator;
	using reference = storage_type::reference;
	using size_type = storage_type::size_type;

	GasStateVector() :
		v_{OCTOTIGER_MAX_NUMBER_FIELDS} {
		for (auto &field : v_) {
			field.resize(H_N3);
		}
	}
	GasStateVector(GasStateVector const &) = default;
	GasStateVector(GasStateVector &&) noexcept = default;
	GasStateVector &operator=(GasStateVector const &) = default;
	GasStateVector &operator=(GasStateVector &&) noexcept = default;

	iterator begin() noexcept {
		return v_.begin();
	}
	const_iterator begin() const noexcept {
		return v_.begin();
	}
	iterator end() noexcept {
		return v_.end();
	}
	const_iterator end() const noexcept {
		return v_.end();
	}
	reference operator[](size_type n) {
		return v_[n];
	}
	const_reference operator[](size_type n) const {
		return v_[n];
	}
	size_type size() const noexcept {
		return v_.size();
	}
	operator storage_type &() {
		return v_;
	}
	operator storage_type const &() const {
		return v_;
	}
	void serialize(auto &arc, unsigned) {
		arc & v_;
	}
	DEFINE_ARRAY_ARITHMETIC(GasStateVector, Real)
	auto get(int i) const {
		GasState<Real> U{};
		for (unsigned n = 0; n < v_.size(); n++) {
			U[n] = v_[n][i];
		}
		return U;
	}
	void set(int i, GasState<Real> const &U) {
		for (unsigned n = 0; n < v_.size(); n++) {
			v_[n][i] = U[n];
		}
	}

private:
	storage_type v_;
};
