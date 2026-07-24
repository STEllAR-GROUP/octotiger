/*
 * reconstruct.hpp
 *
 *  Created on: Jul 10, 2026
 *      Author: dmarce1
 */

#include "octotiger/radiation/rad_grid.hpp"

std::array<RadiationStateVector, NDIM> radiationModalReconstruction(RadiationStateVector const &U) {
	using std::min;
	using std::sqrt;
	std::array<RadiationStateVector, NDIM> dUdx;
	std::array<RadiationStateVector, NDIM> dVdx;
	forEach(RadGrid::interior::box.pad(1), [&](auto I) {
		auto const idx = RadGrid::exterior::box.flatten(I);
		for (int dim = 0; dim < NDIM; dim++) {
			auto const &stride = RadGrid::stride[dim];
			auto const Up = U.get(idx + stride);
			auto const U0 = U.get(idx);
			auto const Um = U.get(idx - stride);
			auto const dUp = Up - U0;
			auto const dUm = U0 - Um;
			auto const Λ = U0.eigensystem(dim);
			auto const dVp = Λ.L * dUp;
			auto const dVm = Λ.L * dUm;
			auto const dV = vanleer(dVp, dVm);
			auto const dU = Λ.R * dV;
			dUdx[dim].set(idx, dU);
		}
//		auto θ = E0 / std::max(E0, dEmax);
//		for (int ci = 0; ci < (1 << NDIM); ci++) {
//			std::bitset<NDIM> bits(ci);
//			Vector<Real, NDIM> dFtot = 0_R;
//			Real dEtot = 0_R;
//			for (int d = 0; d < NDIM; d++) {
//				auto const sign = bits[d] ? 1_R : -1_R;
//				dEtot += sign * dE[d];
//				dFtot += sign * dF[d];
//			}
//			auto const a = (sqr(dEtot) - dFtot.dot(dFtot));
//			auto const b = expectNonNegative(E0) * dEtot - F0.dot(dFtot);
//			auto const c = sqr(E0) - F0.dot(F0);
//			auto const d2 = std::max(0_R, sqr(b) - a * c);
//			auto const d = sqrt(d2);
//			if (sqr(a) > sqr(eps_R) * (sqr(b) + sqr(c))) {
//				auto const ia = 1_R / a;
//				auto const θ1 = -2_R * (b - d) * ia;
//				auto const θ2 = -2_R * (b + d) * ia;
//				if (θ1 >= 0_R && θ1 < θ) θ = θ1;
//				if (θ2 >= 0_R && θ2 < θ) θ = θ2;
//			} else if (b < 0_R) {
//				θ = min(θ, -c / b);
//			}
//		}
//		θ *= almostOne;
	});
	return dUdx;
}
