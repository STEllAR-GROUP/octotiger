/*
 * radiationEoS.hpp
 *
 *  Created on: Jun 25, 2026
 *      Author: dmarce1
 */

#ifndef OCTOTIGER_RADIATION_RADIATIONEOS_HPP_
#define OCTOTIGER_RADIATION_RADIATIONEOS_HPP_

#include "octotiger/defs.hpp"
#include "octotiger/gas/GasEoS.hpp"
#include "octotiger/math/Box.hpp"
#include "octotiger/math/Math.hpp"
#include "octotiger/math/Matrix.hpp"
#include "octotiger/math/Real.hpp"
#include "octotiger/math/Vector.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"

#include <algorithm>
#include <ranges>
#include <span>

inline constexpr int er_i = 0;
inline constexpr int fx_i = 1;
inline constexpr int fy_i = 2;
inline constexpr int fz_i = 3;
// Α Β Γ Δ Ε Ζ Η Θ Ι Κ Λ Μ Ν Ξ Ο Π Ρ Σ Τ Υ Φ Χ Ψ Ω
// α β γ δ ε ζ θ ι κ λ η μ ν ξ ο π ρ σ ς τ υ φ χ ψ ω
// ₐ ₑ ₕ ᵢ ⱼ ₖ ₗ ₘ ₙ ₒ ₚ ᵣ ₛ ₜ ᵤ ᵥ ₓ
// ᵃ ᵇ ᶜ ᵈ ᵉ ᶠ ᵍ ʰ ⁱ ʲ ᵏ ˡ ᵐ ⁿ ᵒ ᵖ ʳ ˢ ᵗ ᵘ ᵛ ʷ ˣ ʸ ᶻ
// ᴬ ᴮ ᴰ ᴱ ᴳ ᴴ ᴵ ᴶ ᴷ ᴸ ᴹ ᴺ ᴼ ᴾ ᴿ ᵀ ᵁ ⱽ ᵂ
// ᵅ ᵝ ᵞ ᵟ ᵋ ᶿ ᶥ ᶲ ᵡ

template <typename Type>
class RadiationState : public Vector<Type, NDIM + 1> {
	static constexpr auto zero = Type(0_R);
	static constexpr auto one = Type(1_R);
	static constexpr auto two = Type(2_R);
	static constexpr auto three = Type(3_R);
	static constexpr auto four = Type(4_R);
	static constexpr auto five = Type(5_R);
	static constexpr auto half = one / two;
	static constexpr auto third = one / three;
	static constexpr auto quarter = one / four;

public:
	using base_type = Vector<Type, NDIM + 1>;
	using SpaceVector = Vector<Type, NDIM>;
	using SpaceTimeVector = Vector<Type, NDIM + 1>;
	using SpaceMatrix = Matrix<Type, NDIM>;
	using SpaceTimeMatrix = Matrix<Type, NDIM + 1>;
	auto conservedFlux() {
		return Vector<Type, NDIM, Type *>(&(*this)[fx_i]);
	}
	auto &energyDensity() {
		return (*this)[er_i];
	}
	auto conservedFlux() const {
		auto const &v = (*this)[fx_i];
		return Vector<Type, NDIM, Type const *>(&v);
	}
	auto energyDensity() const {
		return expectPositive((*this)[er_i]);
	}
	auto conservedFlux(int i) const {
		return (*this)[fx_i + i];
	}
	auto reducedFlux() const {
		return conservedFlux() / energyDensity();
	}
	auto enthalpy() const {
		auto const f = reducedFlux();
		auto const χ = third * (five - two * sqrt(expectPositive(four - three * f.dot(f))));
		return energyDensity() * (three - χ) * half;
	}
	auto conservedVariables() const {
		return std::pair{energyDensity(), conservedFlux()};
	}
	auto primitiveVariables() const {
		auto const H = enthalpy();
		auto const β = reducedFlux() / expectPositive(H);
		return std::pair<Type, SpaceVector>{H, β};
	}
	auto pressureTensor() const {
		auto const [H, β] = primitiveVariables();
		auto const β2 = expectRange(zero, β.dot(β), one);
		return (quarter * (one - β2) * δ + outerProduct(β, β)) * H;
	}
	auto temperature() const {
		using std::pow;
		static auto const σ = physcon().sigma;
		static auto const c = physcon().c;
		static auto const a = four * σ / c;
		auto const E = energyDensity();
		return pow(E / a, quarter);
	}
	auto stressEnergyTensor() const {
		SpaceTimeMatrix T;
		auto const [E, F, P] = std::pair(energyDensity(), conservedFlux(), pressureTensor());
		T[0] = concatenate(E, F);
		for (int n = 0; n < NDIM; n++) {
			T[n + 1] = concatenate(F[n], P[n]);
		}
		return T;
	}
	auto eigenvalues(int k) const {
		auto const [H, β] = primitiveVariables();
		return eigenvalues(H, β, k);
	}
	auto flux(int k) const {
		auto const [H, β] = primitiveVariables();
		return flux(H, β, k);
	}
	auto eigensystem(int k) const {
		auto const [_, β] = primitiveVariables();
		return eigensystem(β, k);
	}
	template <RiemannSolverType solver = RiemannSolverType::HLL>
	friend auto riemannSolver(RadiationState<Type> const &leftState, RadiationState<Type> const &rightState, int k,
							  Type opticalDepth = 0_R) {
		using namespace std;
		auto const &τ = opticalDepth;
		auto const &[UL, UR] = std::pair(leftState, rightState);
		auto const &[EL, FL] = UL;
		auto const &[ER, FR] = UR;
		auto const [HL, βL] = UL.primitiveVariables();
		auto const [HR, βR] = UR.primitiveVariables();
		auto const fL = flux(HL, βL, k);
		auto const fR = flux(HR, βR, k);
		if constexpr (solver == RiemannSolverType::Roe) {
			auto const wL = sqrt(HL);
			auto const wR = sqrt(HR);
			auto const β = (wL * βL + wR * βR) / (wL + wR);
			auto [λ, L, R] = eigensystem(β, k);
			std::transform(λ.begin(), λ.end(), λ.begin(), abs);
			return (half * (fR + fL) - half * R * λ * L * (UR - UL));
		}
		auto const λap = two / (two + three * τ);
		auto const λL = UL.eigenvalues(HL, βL, k);
		auto const λR = UR.eigenvalues(HR, βR, k);
		auto const &[λ1L, λ1R] = std::tuple(λL.back(), λR.back());
		auto const &[λ4L, λ4R] = std::tuple(λL.front(), λR.front());
		auto const sR = min(+λap, max(zero, max(λ1R, λ1L)));
		auto const sL = max(-λap, min(zero, min(λ4R, λ4L)));
		if constexpr (solver == RiemannSolverType::LLF) {
			auto const a = max(abs(sL), abs(sR));
			return half * (fL + fR) - half * a * (UR - UL);
		} else if constexpr (solver == RiemannSolverType::HLL) {
			return (sR * fL - sL * fR - sL * sR * (UR - UL)) / (sR - sL);
		} else if constexpr (solver == RiemannSolverType::HLLC) {
			auto const βL2 = expectRange(zero, βL.dot(βL), one);
			auto const βR2 = expectRange(zero, βR.dot(βR), one);
			auto const ΠL = quarter * (one - βL2) * HL;
			auto const ΠR = quarter * (one - βR2) * HR;
			auto const AL = sL * EL - FL[k];
			auto const AR = sR * ER - FR[k];
			auto const BL = (sL - βL[k]) * FL[k] - ΠL;
			auto const BR = (sR - βR[k]) * FR[k] - ΠR;
			auto const a = AL * sR - AR * sL;
			auto const b = -AL - BL * sR + AR + BR * sL;
			auto const c = BL - BR;
			auto const disc = max(zero, b * b - four * a * c);
			auto const λ0 = -two * c / copysign(abs(b) + sqrt(disc), b);
			auto const left = λ0 > zero;
			auto const Π0 = left ? (AL * λ0 - BL) / (one - λ0 * sL) : (AR * λ0 - BR) / (one - λ0 * sR);
			auto const F = left ? (FL * (sL - βL[k]) + δ[k] * (Π0 - ΠL)) / (sL - λ0) : (FR * (sR - βR[k]) + δ[k] * (Π0 - ΠR)) / (sR - λ0);
			return cocatenate(F[k], λ0 * F + Π0 * δ[k]);
		} else {
			throw("Unknown RiemannSolverType");
		}
	}
	static auto flux(Type H, SpaceVector const &β, int k) {
		auto const β2 = expectRange(zero, β.dot(β), one);
		auto const &βk = β[k];
		return H * concatenate(βk, βk * β + δ[k] * quarter * (one - β2));
	}
	static auto eigenvalues(SpaceVector const &β, int k) {
		SpaceTimeVector λ{};
		auto const βz = β[k];
		auto const β2 = expectRange(zero, β.dot(β), one);
		auto const tmp0 = one / (three - β2);
		auto const tmp1 = sqrt(expectPositive((one - β2) * (three - β2 - two * βz)));
		λ[0] = (two * βz - tmp1) * tmp0;
		λ[1] = λ[2] = βz;
		λ[3] = (two * βz + tmp1) * tmp0;
		return λ;
	}
	static auto eigensystem(SpaceVector const &β, int k) {
		using namespace std;
		constexpr auto t = 0;
		Eigensystem<Type, NDIM + 1> Λ;
		Vector<Type, NDIM + 1> Rt;
		Matrix<Type, NDIM + 1, NDIM> Rxyz;
		auto const i = (k + 1) % NDIM;
		auto const j = (k + 2) % NDIM;
		Λ.λ = eigenvalues(β, k);
		auto const &λ1 = Λ.λ.front();
		auto const &λ4 = Λ.λ.back();
		auto const λ12 = sqr(λ1);
		auto const λ42 = sqr(λ4);
		auto const βx = β[i];
		auto const βy = β[j];
		auto const βz = β[k];
		auto const βt2 = sqr(βx) + sqr(βy);
		auto const βz2 = sqr(βz);
		auto const β2 = expectRange(zero, βt2 + βz2, one);
		auto const flag = βt2 > eps_R * β2; // clang-format off
		auto const r = 
			{ λ12 - two * βz * λ1 + one, flag ? βt2              : zero, flag ?  zero : zero,  λ42 - two * βz * λ4 + one};
		auto const R = { 
			{ βx * (one - λ12)         , flag ? βx * (one - βz2) : one , flag ?  βy   : zero,  βx * (one - λ42)         },
			{ βy * (one - λ12)         , flag ? βy * (one - βz2) : zero, flag ? -βx   : one ,  βy * (one - λ42)         },
			{-βz * λ12 + two * λ1 - βz , flag ? βz * βt2         : zero, flag ?  zero : zero, -βz * λ42 + two * λ4 - βz }
		}; // clang-format on
		Λ.R = concatenate(half * (three * r + β.dot(R)), R + β * r);
		Λ.L = inverse(Λ.R);
		return Λ;
	}

private:
	static Matrix<Type, NDIM> const δ;
};

template <typename Type>
Matrix<Type, NDIM> const RadiationState<Type>::δ = identityMatrix<Type, NDIM>();

struct RadiationStateVector {
	using storage_type = std::array<std::vector<Real>, (NDIM + 1)>;
	using const_iterator = storage_type::const_iterator;
	using const_reference = storage_type::const_reference;
	using iterator = storage_type::iterator;
	using reference = storage_type::reference;
	using size_type = storage_type::size_type;

	RadiationStateVector() :
		v_{} {
		for (auto &field : v_) {
			field.resize(RAD_N3);
		}
	}
	RadiationStateVector(RadiationStateVector const &) = default;
	RadiationStateVector(RadiationStateVector &&) noexcept = default;
	RadiationStateVector &operator=(RadiationStateVector const &) = default;
	RadiationStateVector &operator=(RadiationStateVector &&) noexcept = default;

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
	auto get(int i) const {
		RadiationState<Real> U;
		for (int d = 0; d <= NDIM; ++d) {
			U[d] = v_[d][i];
		}
		return U;
	}
	void set(int i, RadiationState<Real> const &U) {
		for (int d = 0; d <= NDIM; ++d) {
			v_[d][i] = U[d];
		}
	}
	DEFINE_ARRAY_ARITHMETIC(RadiationStateVector, Real);

private:
	storage_type v_;
};

auto opacityAbsorption(auto ρ, auto T) {
	static CgsToCode const convert{};
	static auto const n = opts().kappa_rho_exp;
	static auto const m = opts().kappa_T_exp;
	static auto const κₒ = convert.inverseLength(opts().kappa0);
	static auto const Tₒ = convert.temperature();
	static auto const ρₒ = convert.massDensity();
	return κₒ * pow(ρ / ρₒ, n) * pow(T / Tₒ, m);
}

auto opacityScattering(auto ρ, auto T) {
	static CgsToCode const convert{};
	static auto const n = opts().sigma_rho_exp;
	static auto const m = opts().sigma_T_exp;
	static auto const σₒ = convert.inverseLength(opts().sigma0);
	static auto const Tₒ = convert.temperature();
	static auto const ρₒ = convert.massDensity();
	return σₒ * pow(ρ / ρₒ, n) * pow(T / Tₒ, m);
}

auto opacityTotal(auto ρ, auto T) {
	return opacityAbsorption(ρ, T) + opacityScattering(ρ, T);
}

inline std::string radiationAbsorptionExpression(std::string const &rho = "rho", std::string const &T = "T") {
	return hpx::util::format("({:e}) * ({})^{:e} * ({})^{:e}", opts().kappa0, rho, opts().kappa_rho_exp, T, opts().kappa_T_exp);
}

inline std::string radiationScatteringExpression(std::string const &rho = "rho", std::string const &T = "T") {
	return hpx::util::format("({:e}) * ({})^{:e} * ({})^{:e}", opts().sigma0, rho, opts().sigma_rho_exp, T, opts().sigma_T_exp);
}

inline std::string radiationExtinctionExpression(std::string const &rho = "rho", std::string const &T = "T") {
	return "(" + radiationAbsorptionExpression(rho, T) + ") + (" + radiationScatteringExpression(rho, T) + ")";
}

#endif /* OCTOTIGER_RADIATION_RADIATIONEOS_HPP_ */
