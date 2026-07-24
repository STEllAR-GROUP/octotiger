/*
 * transport.cpp
 *
 *  Created on: Jul 10, 2026
 *      Author: dmarce1
 */

#include "octotiger/radiation/rad_grid.hpp"


void radiationTransportFluxes(RadiationFluxVector &flux, RadiationStateVector const &U, std::vector<Real> const &χ, Real dx) {
	FpeGuard fpeGuard{};
	using std::abs;
	using std::max;
	using std::min;
	using std::sqrt;
	auto const c = physcon().c;
	auto const c2 = sqr(c);
	auto const ic = inv(c);
	auto const dU_dx = radiationModalReconstruction(U);
	for (int k = 0; k < NDIM; k++) {
		auto const dnRad = RadGrid::stride[k];
		auto const dnGas = GasGrid::stride[k];
		forEach(RadGrid::interior::box.pad(k, std::pair(0, 1)), [&](auto I) {
			auto const ir = RadGrid::exterior::box.flatten(I);
			auto const ig = GasGrid::exterior::box.flatten(I);
			auto const Uᵣ = ConservedRadiationState(U.get(ir) - 0.5_R * dU_dx[k].get(ir));
			auto const Uₗ = ConservedRadiationState(U.get(ir - dnRad) + 0.5_R * dU_dx[k].get(ir - dnRad));
			auto const Vᵣ = Uᵣ.toPrimitive();
			auto const Vₗ = Uₗ.toPrimitive();
			auto const &βᵣ = Vᵣ.β;
			auto const &βₗ = Vₗ.β;
			auto const &βkᵣ = βᵣ[k];
			auto const &βkₗ = βₗ[k];
			auto const β2ᵣ = expectRange(0_R, βᵣ.dot(βᵣ), 1_R);
			auto const β2ₗ = expectRange(0_R, βₗ.dot(βₗ), 1_R);
			auto const Xᵣ = 0.5_R * sqrt((1_R - β2ᵣ) * (3_R - β2ᵣ - 2_R * sqr(βkᵣ)));
			auto const Xₗ = 0.5_R * sqrt((1_R - β2ₗ) * (3_R - β2ₗ - 2_R * sqr(βkₗ)));
			auto const ξᵣ = 2_R / (3_R - β2ᵣ);
			auto const ξₗ = 2_R / (3_R - β2ₗ);
			auto const τᵣ = χ[ig] * dx;
			auto const τₗ = χ[ig - dnGas] * dx;
			auto const τ = 0.5_R * (τᵣ + τₗ);
			auto const λₐ = 1_R / (1_R + 1.5_R * τ);
			auto const αᵣ = min(λₐ, ξᵣ * Xᵣ);
			auto const αₗ = min(λₐ, ξₗ * Xₗ);
			auto const υᵣ = ξᵣ * βkᵣ;
			auto const υₗ = ξₗ * βkₗ;
			auto const Fᵣ = radiationConservedFlux(Vᵣ, k);
			auto const Fₗ = radiationConservedFlux(Vₗ, k);

			ConservedRadiationState<Real> F{};
			switch (riemannSolver) {
			case RiemannSolver::LxF: {
				auto λ = expectRange(-1_R, max(abs(υᵣ) + αᵣ, abs(υₗ) + αₗ), +1_R);
				F = c * 0.5_R * (Fₗ + Fᵣ - λ * (Uᵣ - Uₗ));
			} break;
			case RiemannSolver::HLL: {
				auto λᵣ = max(υᵣ + αᵣ, υₗ + αₗ);
				auto λₗ = min(υᵣ - αᵣ, υₗ - αₗ);
				λᵣ = expectRange(-0_R, max(0_R, λᵣ), +1_R);
				λₗ = expectRange(-1_R, min(0_R, λₗ), +0_R);
				auto const iλ = 1_R / expectPositive(λᵣ - λₗ);
				F = c * (λᵣ * Fₗ - λₗ * Fᵣ + λᵣ * λₗ * (Uᵣ - Uₗ)) * iλ;
			} break;
			};

			flux[k].set(ir, ConservedRadiationState<Real>(F));
		});
	}
}
// F = c * (-1/(3*τ))*(Eᵣ - Eₗ) - 0.5_R * λₐ * (Eᵣ - Eₗ));

void radiationApplyFluxes(RadiationStateVector &U, RadiationFluxVector const &F, Real h) {
	forEach(RadGrid::interior::box, [&](auto idx) {
		auto const ir = RadGrid::exterior::box.flatten(idx);
		for (int f = 0; f <= NDIM; f++) {
			for (int d = 0; d < NDIM; d++) {
				auto const Fp = F[d][f][ir + RadGrid::stride[d]];
				auto const Fm = F[d][f][ir];
				U[f][ir] -= (Fp - Fm) * h;
			}
		}
	});
}
