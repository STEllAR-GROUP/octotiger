//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/defs.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/matrix.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/radiation/implicit.hpp"
#include "octotiger/radiation/kernel_interface.hpp"
#include "octotiger/radiation/opacities.hpp"
#include "octotiger/radiation/rad_grid.hpp"
#include "octotiger/real.hpp"
#include "octotiger/roe.hpp"
#include "octotiger/space_vector.hpp"

#include <hpx/include/future.hpp>

#include <fenv.h>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "octotiger/unitiger/radiation/radiation_physics_impl.hpp"
#include "octotiger/matrix.hpp"

#if !defined(HPX_COMPUTE_DEVICE_CODE)

using real = double;

#define CHECK_FLUX( er, fx, fy, fz) if( ((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c) > 1 ) {printf( "flux exceded %s %i %e fx %e fy %e fz %e er %e\n", __FILE__, __LINE__, sqrt(((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c)), fx, fy, fz, er*physcon().c); abort();}
#define CHECK_FLUX1( er, fx, fy, fz, rho) if( ((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c) > 1 ) {printf( "flux exceded %s %i %e fx %e fy %e fz %e er %e rho %e\n", __FILE__, __LINE__, sqrt(((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c)), fx, fy, fz, er*physcon().c, rho); abort();}
#define BAD_FLUX( er, fx, fy, fz, rho) bool(((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c) > 1 )

std::unordered_map<std::string, int> rad_grid::str_to_index;
std::unordered_map<int, std::string> rad_grid::index_to_str;

void node_server::compute_radiation(Real dt, Real omega) {
	static constexpr Real zero = Real(0), third = Real(1.0 / 3.0), half = Real(0.5), one = Real(1), two = Real(2), three = Real(3), four = Real(4), five = Real(
			5);
	const bool root = bool(my_location.level() == 0);
	auto rgrid = rad_grid_ptr;
	rgrid->set_dx(grid_ptr->get_dx());
	rgrid->set_X(grid_ptr->get_X());
	rgrid->compute_mmw(grid_ptr->U);
	const Real min_dx = Real(TWO * grid::get_scaling_factor() / real(INX << opts().max_level));
	const Real c = Real(physcon().c);
	const Real max_dt = Real(min_dx / c / 6.0);
	const integer Nsubstep = std::max(int(std::ceil(dt / max_dt)), 1);
	const Real sub_dt = dt / Real(Nsubstep);
	if (root) {
		std::cout << "Radiation Transport with " << std::to_string(Nsubstep) << " substeps\n";
	}
	all_rad_bounds();
	const Real gamma = Real(1.0 - 1.0 / sqrt(2.0));
	const Real invGamma = one / gamma;
	using rad_store_t = decltype(rgrid->U);
	using gas_store_t = decltype(grid_ptr->U);
	auto &Ur = rgrid->U;
	auto &Ug = grid_ptr->U;
	rad_store_t Ur0, F1, F2, Gr1, Gr2;
	gas_store_t Ug0, Gg1, Gg2;
	for (integer i = 0; i != Nsubstep; ++i) {
		if (root) {
			std::cout << "i = " << std::to_string(i) << "\n";
		}
		Ur0 = Ur;
		Ug0 = Ug;
		Gr1 = Ur;
		Gg1 = Ug;
		rgrid->implicit_source(grid_ptr->U, gamma * sub_dt);
		for (int f = 0; f < NRF; f++) {
			Gr1[f].resize(RAD_N3);
			for (int n = 0; n < RAD_N3; n++) {
				Gr1[f][n] = (Ur[f][n] - Gr1[f][n]) * invGamma;
			}
		}
		for (int f = 0; f < opts().n_fields; f++) {
			Gg1[f].resize(H_N3);
			for (int n = 0; n < H_N3; n++) {
				Gg1[f][n] = (Ug[f][n] - Gg1[f][n]) * invGamma;
			}
		}
		all_rad_bounds();

		F1 = Ur;
		rgrid->compute_flux(omega);
		GET(exchange_rad_flux_corrections());
		rgrid->advance(sub_dt);
		for (int f = 0; f < NRF; f++) {
			for (int n = 0; n < RAD_N3; n++) {
				F1[f][n] = Ur[f][n] - F1[f][n];
			}
		}
		for (int f = 0; f < NRF; f++) {
			for (int n = 0; n < RAD_N3; n++) {
				Ur[f][n] += (one - two * gamma) * Gr1[f][n];
			}
		}
		for (int f = 0; f < opts().n_fields; f++) {
			for (int n = 0; n < H_N3; n++) {
				Ug[f][n] += (one - two * gamma) * Gg1[f][n];
			}
		}
		rgrid->implicit_source(grid_ptr->U, gamma * sub_dt);
		Gr2 = Ur;
		Gg2 = Ug;
		for (int f = 0; f < NRF; f++) {
			for (int n = 0; n < RAD_N3; n++) {
				Gr2[f][n] = (Ur[f][n] - Gr2[f][n]) * invGamma;
			}
		}
		for (int f = 0; f < opts().n_fields; f++) {
			for (int n = 0; n < H_N3; n++) {
				Gg2[f][n] = (Ug[f][n] - Gg2[f][n]) * invGamma;
			}
		}
		all_rad_bounds();

		F2 = Ur;
		rgrid->compute_flux(omega);
		GET(exchange_rad_flux_corrections());
		rgrid->advance(sub_dt);
		for (int f = 0; f < NRF; f++) {
			for (int n = 0; n < RAD_N3; n++) {
				F2[f][n] = Ur[f][n] - F2[f][n];
			}
		}
		all_rad_bounds();

		for (int f = 0; f < NRF; f++) {
			for (int n = 0; n < RAD_N3; n++) {
				Ur[f][n] = Ur0[f][n] + half * (F1[f][n] + F2[f][n]) + half * (Gr1[f][n] + Gr2[f][n]);
			}
		}
		for (int f = 0; f < opts().n_fields; f++) {
			for (int n = 0; n < H_N3; n++) {
				Ug[f][n] = Ug0[f][n] + half * (Gg1[f][n] + Gg2[f][n]);
			}
		}
		all_rad_bounds();
	}
	if (my_location.level() == 0) {
		std::cout << "\rradiation done\n";
	}
}

const auto computeEddingtonTensor = [](Real const &rEnergy, std::array<Real, NDIM> rFlux) {
	static constexpr Real zero = Real(0), third = Real(1.0 / 3.0), half = Real(0.5), one = Real(1), two = Real(2), three = Real(3), four = Real(4), five = Real(
			5);
	constexpr Real tiny = Real::tiny();
	std::array<std::array<Real, NDIM>, NDIM> eddingtonTensor;
	std::array<Real, NDIM> fluxUnitVector;
	Real const lightSpeed = Real(physcon().c);
	Real const invLightSpeed = one / lightSpeed;
	Real const invEnergy = one / rEnergy;
	assert(rEnergy > zero);
	Real fluxNormSquared = zero;
	for (int d = 0; d < NDIM; d++) {
		rFlux[d] *= invLightSpeed * invEnergy;
	}
	for (int d = 0; d < NDIM; d++) {
		fluxNormSquared += sqr(rFlux[d]);
	}
	assert(zero <= fluxNormSquared);
	assert(fluxNormSquared <= one);
	Real const invFluxNorm = sqrt(fluxNormSquared) / (fluxNormSquared + tiny);
	for (int d = 0; d < NDIM; d++) {
		fluxUnitVector[d] = rFlux[d] * invFluxNorm;
	}
	Real const eddingtonFactorNum = three + four * fluxNormSquared;
	Real const eddingtonFactorDen = five + two * sqrt(four - three * fluxNormSquared);
	Real const eddingtonFactor = eddingtonFactorNum / eddingtonFactorDen;
	Real const diffusionCoefficient = half * (one - eddingtonFactor);
	Real const streamingCoefficient = half * (three * eddingtonFactor - one);
	for (int l = 0; l < NDIM; l++) {
		eddingtonTensor[l][l] = diffusionCoefficient + streamingCoefficient * sqr(fluxUnitVector[l]);
		for (int m = 0; m < l; m++) {
			eddingtonTensor[m][l] = eddingtonTensor[l][m] = fluxUnitVector[l] * fluxUnitVector[m] * streamingCoefficient;
		}
	}
	return eddingtonTensor;
};

struct GasMixture {
	real meanMolecularWeight;
	real hydrogenFraction;
	real heliumFraction;
	real metalFraction;
};

GasMixture meanMolecularWeight(specie_state_t<real> const &speciesDensity) {
	integer const nSpecies = opts().n_species;
	auto const &atomicNumber = opts().atomic_number;
	auto const &atomicMass = opts().atomic_mass;
	static auto const invAtomicMass = [nSpecies](auto atomicMass) {
		for (int i = 0; i < nSpecies; i++) {
			atomicMass[i] = real(1) / atomicMass[i];
		}
		return atomicMass;
	}(atomicMass);
	real totalMass = real(0);
	real totalNumber = real(0);
	real hydrogenMass = real(0);
	real heliumMass = real(0);
	real metalMass = real(0);
	for (int i = 0; i < nSpecies; i++) {
		real const mass = speciesDensity[i];
		integer const thisAtomicNumber = atomicNumber[i];
		totalNumber += mass * invAtomicMass[i] * (thisAtomicNumber + real(1));
		totalMass += mass;
		if (thisAtomicNumber == 1) {
			hydrogenMass += mass;
		} else if (thisAtomicNumber == 2) {
			heliumMass += mass;
		} else if (thisAtomicNumber >= 3) {
			metalMass += mass;
		}
	}
	GasMixture result;
	real const invTotalMass = real(1) / totalMass;
	result.meanMolecularWeight = totalMass / totalNumber;
	result.hydrogenFraction = hydrogenMass * invTotalMass;
	result.heliumFraction = heliumMass * invTotalMass;
	result.metalFraction = metalMass * invTotalMass;
	return result;
}

void solve_implicit(safe_real rho, safe_real mu, safe_real &Ei, std::array<safe_real, NDIM> &beta, safe_real &Er, std::array<safe_real, NDIM> &F) {

}

inline real delta(int i, int k) {
	return real(i == k);
}

std::array<std::array<Real, NRF>, NRF> matrixInverse(const std::array<std::array<Real, NRF>, NRF> &m) {
	static constexpr Real one = Real(1);
	Real const det = m[0][0]
			* (m[1][1] * (m[2][2] * m[3][3] - m[2][3] * m[3][2]) - m[1][2] * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
					+ m[1][3] * (m[2][1] * m[3][2] - m[2][2] * m[3][1]))
			- m[0][1]
					* (m[1][0] * (m[2][2] * m[3][3] - m[2][3] * m[3][2]) - m[1][2] * (m[2][0] * m[3][3] - m[2][3] * m[3][0])
							+ m[1][3] * (m[2][0] * m[3][2] - m[2][2] * m[3][0]))
			+ m[0][2]
					* (m[1][0] * (m[2][1] * m[3][3] - m[2][3] * m[3][1]) - m[1][1] * (m[2][0] * m[3][3] - m[2][3] * m[3][0])
							+ m[1][3] * (m[2][0] * m[3][1] - m[2][1] * m[3][0]))
			- m[0][3]
					* (m[1][0] * (m[2][1] * m[3][2] - m[2][2] * m[3][1]) - m[1][1] * (m[2][0] * m[3][2] - m[2][2] * m[3][0])
							+ m[1][2] * (m[2][0] * m[3][1] - m[2][1] * m[3][0]));

	Real const inv_det = one / det;
	std::array<std::array<Real, NRF>, NRF> result { { { inv_det
			* (m[1][1] * (m[2][2] * m[3][3] - m[2][3] * m[3][2]) - m[1][2] * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
					+ m[1][3] * (m[2][1] * m[3][2] - m[2][2] * m[3][1])), -inv_det
			* (m[0][1] * (m[2][2] * m[3][3] - m[2][3] * m[3][2]) - m[0][2] * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
					+ m[0][3] * (m[2][1] * m[3][2] - m[2][2] * m[3][1])), inv_det
			* (m[0][1] * (m[1][2] * m[3][3] - m[1][3] * m[3][2]) - m[0][2] * (m[1][1] * m[3][3] - m[1][3] * m[3][1])
					+ m[0][3] * (m[1][1] * m[3][2] - m[1][2] * m[3][1])), -inv_det
			* (m[0][1] * (m[1][2] * m[2][3] - m[1][3] * m[2][2]) - m[0][2] * (m[1][1] * m[2][3] - m[1][3] * m[2][1])
					+ m[0][3] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])) }, { -inv_det
			* (m[1][0] * (m[2][2] * m[3][3] - m[2][3] * m[3][2]) - m[1][2] * (m[2][0] * m[3][3] - m[2][3] * m[3][0])
					+ m[1][3] * (m[2][0] * m[3][2] - m[2][2] * m[3][0])), inv_det
			* (m[0][0] * (m[2][2] * m[3][3] - m[2][3] * m[3][2]) - m[0][2] * (m[2][0] * m[3][3] - m[2][3] * m[3][0])
					+ m[0][3] * (m[2][0] * m[3][2] - m[2][2] * m[3][0])), -inv_det
			* (m[0][0] * (m[1][2] * m[3][3] - m[1][3] * m[3][2]) - m[0][2] * (m[1][0] * m[3][3] - m[1][3] * m[3][0])
					+ m[0][3] * (m[1][0] * m[3][2] - m[1][2] * m[3][0])), inv_det
			* (m[0][0] * (m[1][2] * m[2][3] - m[1][3] * m[2][2]) - m[0][2] * (m[1][0] * m[2][3] - m[1][3] * m[2][0])
					+ m[0][3] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])) }, { inv_det
			* (m[1][0] * (m[2][1] * m[3][3] - m[2][3] * m[3][1]) - m[1][1] * (m[2][0] * m[3][3] - m[2][3] * m[3][0])
					+ m[1][3] * (m[2][0] * m[3][1] - m[2][1] * m[3][0])), -inv_det
			* (m[0][0] * (m[2][1] * m[3][3] - m[2][3] * m[3][1]) - m[0][1] * (m[2][0] * m[3][3] - m[2][3] * m[3][0])
					+ m[0][3] * (m[2][0] * m[3][1] - m[2][1] * m[3][0])), inv_det
			* (m[0][0] * (m[1][1] * m[3][3] - m[1][3] * m[3][1]) - m[0][1] * (m[1][0] * m[3][3] - m[1][3] * m[3][0])
					+ m[0][3] * (m[1][0] * m[3][1] - m[1][1] * m[3][0])), -inv_det
			* (m[0][0] * (m[1][1] * m[2][3] - m[1][3] * m[2][1]) - m[0][1] * (m[1][0] * m[2][3] - m[1][3] * m[2][0])
					+ m[0][3] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])) } } };
	return result;
}

struct testImplicitRadiation {
	static constexpr Real zero = Real(0), third = Real(1.0 / 3.0), half = Real(0.5), one = Real(1), two = Real(2), three = Real(3), four = Real(4), twelve =
			Real(12);
	testImplicitRadiation(Real dt_, Real Er0_, std::array<Real, NDIM> F0_, Real Eg0_, std::array<Real, NDIM> Beta0_, Real rho_, Real mu_, Real kappa_,
			Real Chi_, Real gamma_) {
		mAMU = Real(physcon().mh);
		kB = Real(physcon().kb);
		aR = Real(4.0 * physcon().sigma / physcon().c);
		c = Real(physcon().c);
		dt = dt_;
		Er0 = Er0_;
		Eg0 = Eg0_;
		rho = rho_;
		mu = mu_;
		kappa = kappa_;
		Chi = Chi_;
		gamma = gamma_;
		F0 = F0_;
		Beta0 = Beta0_;
	}
	auto operator()(std::array<Real, NRF> U) const {
		auto Er = U[NDIM];
		std::array<Real, NDIM> F;
		for (int n = 0; n < NDIM; n++) {
			F[n] = U[n];
		}
		static std::array<std::array<Real, NDIM>, NDIM> const I = { { { one, zero, zero }, { zero, one, zero }, { zero, zero, one } } };
		static auto const cHat = c;
		std::array<Real, NDIM> Beta;
		for (int n = 0; n < NDIM; n++) {
			Beta[n] = Beta0[n] + (F0[n] - F[n]) / (rho * c);
		}
		Real Ek = zero;
		std::array<Real, NDIM> dEk_dF;
		for (int n = 0; n < NDIM; n++) {
			dEk_dF[n] = -c * Beta[n];
		}
		Real const Eg = Eg0 + Er0 - Er;
		Real const dEg_dEr = -one;
		Real const eint = Eg - Ek;
		std::array<Real, NDIM> deint_dF;
		for (int n = 0; n < NDIM; n++) {
			deint_dF[n] = -dEk_dF[n];
		}
		Real const deint_dEr = -one;
		Real F2 = zero;
		for (int n = 0; n < NDIM; n++) {
			F2 += sqr(F[n]);
		}
		Real const absF = sqrt(F2);
		Real const iCv = (mu * mAMU) / ((gamma - one) * kB * rho);
		Real const T = eint * iCv;
		Real const dT_dEr = -iCv;
		std::array<Real, NDIM> dT_dF;
		for (int n = 0; n < NDIM; n++) {
			dT_dF[n] = deint_dF[n] * iCv;
		}
		Real const T2 = sqr(T);
		Real const T4 = sqr(T2);
		std::array<std::array<Real, NDIM>, NDIM> dBeta_dF;
		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m < NDIM; m++) {
				dBeta_dF[n][m] = -I[n][m] / (rho * c);
			}
		}
		std::array<Real, NDIM> N;
		for (int n = 0; n < NDIM; n++) {
			N[n] = F[n] / (absF + Real::tiny());
		}
		Real const T3 = T * T2;
		std::array<std::array<Real, NDIM>, NDIM> dN_dF;
		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m < NDIM; m++) {
				dN_dF[n][m] = (I[n][m] - N[n] * N[m]) / (absF + Real::tiny());
			}
		}
		std::array<Real, NDIM> df_dF;
		Real const f = absF / Er;
		Real const df_dEr = -f / Er;
		for (int n = 0; n < NDIM; n++) {
			df_dF[n] = N[n] / Er;
		}
		if (f > one || f < zero) {
			std::cout << "f out of range: " << to_string(f) << "\n";
			abort();
		}
		Real const Xi = three - two * sqrt(four - f * three * f);
		Real const dXi_df = twelve * f / (three - Xi);
		Real const dXi_dEr = dXi_df * df_dEr;
		std::array<Real, NDIM> dXi_dF;
		for (int n = 0; n < NDIM; n++) {
			dXi_dF[n] = dXi_df * df_dF[n];
		}
		std::array<std::array<Real, NDIM>, NDIM> D;
		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m < NDIM; m++) {
				D[n][m] = half * ((Xi + one) * third * I[n][m] + (one - Xi) * N[n] * N[m]);
			}
		}
		std::array<std::array<Real, NDIM>, NDIM> dD_dEr;
		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m < NDIM; m++) {
				dD_dEr[n][m] = half * dXi_dEr * (third * I[n][m] - N[n] * N[m]);
			}
		}
		std::array<std::array<std::array<Real, NDIM>, NDIM>, NDIM> dD_dF;
		for (int l = 0; l < NDIM; l++) {
			for (int n = 0; n < NDIM; n++) {
				for (int m = 0; m < NDIM; m++) {
					dD_dF[l][n][m] = half * (dXi_dF[l] * (third * I[n][m] - N[n] * N[m]) + (one - Xi) * (dN_dF[n][l] * N[m] + N[n] * dN_dF[m][l]));
				}
			}
		}
		std::array<std::array<Real, NDIM>, NDIM> P;
		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m < NDIM; m++) {
				P[n][m] = Er * D[n][m];
			}
		}
		std::array<std::array<Real, NDIM>, NDIM> dP_dEr;
		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m < NDIM; m++) {
				dP_dEr[n][m] = Er * dD_dEr[n][m] + D[n][m];
			}
		}
		std::array<std::array<std::array<Real, NDIM>, NDIM>, NDIM> dP_dF;
		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m < NDIM; m++) {
				for (int l = 0; l < NDIM; l++) {
					dP_dF[l][n][m] = Er * dD_dF[l][n][m];
				}
			}
		}
		Real gk = kappa * (Er - aR * T4);
		for (int k = 0; k < NDIM; k++) {
			gk -= two * kappa * Beta[k] * F[k];
		}
		Real const dgk_dEr = kappa * (one - four * aR * T3 * dT_dEr);
		std::array<Real, NDIM> dgk_dF;
		for (int n = 0; n < NDIM; n++) {
			dgk_dF[n] = -kappa * (two * Beta[n] + three * aR * T3 * dT_dF[n]);
			for (int k = 0; k < NDIM; k++) {
				dgk_dF[n] -= two * kappa * F[k] * dBeta_dF[k][n];
			}
		}
		std::array<Real, NDIM> Gx;
		for (int n = 0; n < NDIM; n++) {
			Gx[n] = Chi * (F[n] - Beta[n] * Er);
			for (int k = 0; k < NDIM; k++) {
				Gx[n] -= Chi * P[n][k] * Beta[k];
			}
		}
		std::array<Real, NDIM> dGx_dEr;
		for (int n = 0; n < NDIM; n++) {
			dGx_dEr[n] = -Chi * Beta[n];
			for (int k = 0; k < NDIM; k++) {
				dGx_dEr[n] -= Chi * dP_dEr[n][k] * Beta[k];
			}
		}
		std::array<std::array<Real, NDIM>, NDIM> dGx_dF;
		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m < NDIM; m++) {
				dGx_dF[n][m] = Chi * (I[n][m] - dBeta_dF[n][m] * Er);
				for (int k = 0; k < NDIM; k++) {
					dGx_dF[n][m] -= Chi * dP_dF[m][n][k] * Beta[k];
					dGx_dF[n][m] -= Chi * P[n][k] * dBeta_dF[k][m];
				}
			}
		}
		Real gx = zero;
		for (int k = 0; k < NDIM; k++) {
			gx += Gx[k] * Beta[k];
		}
		Real dgx_dEr = zero;
		for (int k = 0; k < NDIM; k++) {
			dgx_dEr += dGx_dEr[k] * Beta[k];
		}
		std::array<Real, NDIM> dgx_dF;
		for (int n = 0; n < NDIM; n++) {
			dgx_dF[n] = zero;
			for (int k = 0; k < NDIM; k++) {
				dgx_dF[n] = dGx_dF[k][n] * Beta[k] + dBeta_dF[k][n] * Gx[k];
			}
		}
		std::array<Real, NDIM> Gk;
		for (int n = 0; n < NDIM; n++) {
			Gk[n] = gk * Beta[n];
		}
		std::array<Real, NDIM> dGk_dEr;
		for (int n = 0; n < NDIM; n++) {
			dGk_dEr[n] = dgk_dEr * Beta[n];
		}
		std::array<std::array<Real, NDIM>, NDIM> dGk_dF;
		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m < NDIM; m++) {
				dGk_dF[n][m] = Beta[n] * dgk_dF[m] + gk * dBeta_dF[n][m];
			}
		}
		Real const hr = Er - Er0 + dt * cHat * (gk + gx);
		Real const dhr_dEr = one + dt * cHat * (dgk_dEr + dgx_dEr);
		std::array<Real, NDIM> dhr_dF;
		for (int n = 0; n < NDIM; n++) {
			dhr_dF[n] = dt * cHat * (dgk_dF[n] + dgx_dF[n]);
		}
		std::array<Real, NDIM> Hr;
		for (int n = 0; n < NDIM; n++) {
			Hr[n] = F[n] - F0[n] + dt * (Gk[n] + Gx[n]);
		}
		std::array<Real, NDIM> dHr_dEr;
		for (int n = 0; n < NDIM; n++) {
			dHr_dEr[n] = dt * cHat * (dGk_dEr[n] + dGx_dEr[n]);
		}
		std::array<std::array<Real, NRF>, NRF> dHr_dF;
		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m < NDIM; m++) {
				dHr_dF[n][m] = I[n][m] + dt * cHat * (dGk_dF[n][m] + dGx_dF[n][m]);
			}
		}

		std::pair<std::array<Real, NRF>, std::array<std::array<Real, NRF>, NRF>> rc;
		std::array<Real, NRF> &F4 = rc.first;
		std::array<std::array<Real, NRF>, NRF> &dF4 = rc.second;
		for (int k = 0; k < NDIM; k++) {
			F4[k] = Hr[k];
			for (int n = 0; n < NDIM; n++) {
				dF4[k][n] = dHr_dF[k][n];
			}
			dF4[NDIM][k] = dhr_dF[k];
			dF4[k][NDIM] = dHr_dEr[k];
		}
		F4[NDIM] = hr;
		dF4[NDIM][NDIM] = dhr_dEr;
		return rc;
	}
private:
	Real dt;
	Real Er0;
	Real Eg0;
	Real rho;
	Real mu;
	Real kappa;
	Real Chi;
	Real gamma;
	std::array<Real, NDIM> F0;
	std::array<Real, NDIM> Beta0;
	Real mAMU;
	Real kB;
	Real aR;
	Real c;
};

void solveImplicitRadiation(Real &Er, std::array<Real, NDIM> &F, Real &Eg, std::array<Real, NDIM> &Mg, Real rho, Real mu, Real kappa, Real Chi, Real gamma,
		Real dt) {
	/*	static constexpr Constants<Real> pc;
	 * 	Real c;
	 *
	 mAMU = pc.m;
	 kB = pc.kB;
	 aR = pc.aR;
	 c = pc.c;
	 */
	static constexpr Real zero = Real(0);
	Real const c = Real(1);
	std::array<Real, NDIM> Beta0;
	for (int n = 0; n < NDIM; n++) {
		Beta0[n] = Mg[n] / (rho * c);
	}
	auto const Eg0 = Eg;
	testImplicitRadiation test(dt, Er, F, Eg0, Beta0, rho, mu, kappa, Chi, gamma);
	std::array<Real, NRF> x;
	for (int n = 0; n < NDIM; n++) {
		x[n] = F[n] / c;
	}
	x[NDIM] = Er;
	Real toler = Real(1e-9);
	Real err;
	do {
		std::array<Real, NRF> dx;
		auto const f_and_dfdx = test(x);
		auto const &f = f_and_dfdx.first;
		auto const &dfdx = f_and_dfdx.second;
		auto const inv_dfdx = matrixInverse(dfdx);
		err = zero;
		for (int n = 0; n < NRF; n++) {
			dx[n] = zero;
			for (int m = 0; m < NRF; m++) {
				dx[n] -= inv_dfdx[n][m] * f[m];
			}
			err += sqr(dx[n]);
		}
		err = sqrt(err);
		for (int n = 0; n < NRF; n++) {
			x[n] += Real(0.5) * dx[n];
		}
	} while (err > toler);
	for (int n = 0; n < NDIM; n++) {
		Mg[n] = rho * c * Beta0[n] + F[n] - x[n];
		F[n] = c * x[n];
	}
	Eg = Eg0 + Er - x[NDIM];
	Er = x[NDIM];
}

void rad_grid::implicit_source(std::vector<std::vector<real>> &hydroVars, Real dt) {
	PROFILE()
	constexpr real toler = 1000.0 * std::numeric_limits<real>::epsilon();
	constexpr Real zero = Real(0.0);
	constexpr Real half = Real(0.5);
	constexpr real alpha = 0.5;
	constexpr real tiny = std::numeric_limits<real>::min();
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);
	const real sigma = physcon().sigma;
	const real mh = physcon().mh;
	const real kb = physcon().kb;
	const real gam = grid::get_fgamma();
	const real c = physcon().c;
	const real ar = 4.0 * sigma * INVERSE(c);
	const real inv_c = INVERSE(physcon().c);
	const real inv_gam = INVERSE(gam);
	const real c_hat = c * opts().clight_reduced;
	const real inv_cc_hat = INVERSE(c * c_hat);
	const real inv_c2 = INVERSE(c * c);
	for (integer xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
		for (integer yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
			for (integer zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
				const integer DI = H_BW - RAD_BW;
				const integer ir = rindex(xi, yi, zi);
				const integer ih = hindex(xi + DI, yi + DI, zi + DI);
				std::array<Real, NDIM> F;
				std::array<Real, NDIM> Mg;
				Real Ek = zero;
				Real Er = U[er_i][ir];
				Real Eg = Real(hydroVars[egas_i][ir]);
				const Real rho(hydroVars[rho_i][ih]);
				for (int n = 0; n < NDIM; n++) {
					F[n] = U[fx_i + n][ir];
					Mg[n] = Real(hydroVars[sx_i + n][ih]);
					Ek += half * sqr(Real(Mg[n])) / rho;
				}
				const real egas = hydroVars[egas_i][ih];
				const Real mu(mmw[ih]);
				const Real Xspc(X_spc[ih]);
				const Real Zspc(Z_spc[ih]);
				const Real kappa = kappa_p(rho, Real(Mg[NDIM]) - Ek, mu, Xspc, Zspc);
				const Real Chi = kappa_R(rho, Real(Mg[NDIM]) - Ek, mu, Xspc, Zspc);
				solveImplicitRadiation(Er, F, Eg, Mg, rho, mu, kappa, Chi, Real(gam), dt);
				U[er_i][ir] = Er;
				hydroVars[egas_i][ir] = Eg;
				for (int n = 0; n < NDIM; n++) {
					U[fx_i + n][ir] = F[n];
					hydroVars[sx_i + n][ih] = Mg[n];
				}
			}
		}
	}
}

template<typename T>
int sign(T const &x) {
	if (x < T(0)) {
		return -1;
	} else if (x > T(0)) {
		return +1;
	} else {
		return 0;
	}
}

std::pair<Real, Real> radiationEigenvalues(Real f, Real mu) {
	std::pair<Real, Real> rc;
	Real constexpr third = Real(1. / 3.), one = Real(1), two = Real(2), three = Real(3), four = Real(4);
	Real const f2 = f * f;
	Real const mu2 = mu * mu;
	Real const a = sqrt(four - three * f2);
	Real const b = a * a - a;
	Real const c = two - f2 - a;
	Real const d = two * (third * b + mu2 * c);
	Real const e = one / a;
	Real const g = mu * f;
	rc.first = e * (g - d);
	rc.second = e * (g + d);
	return rc;
}

void rad_grid::compute_flux(real omega) {
	PROFILE()
	static Real constexpr zero = Real(0), half = Real(0.5), one = Real(1);
	static Real constexpr eps = Real::epsilon();

	Real const c = Real(physcon().c);
	Real const c2 = sqr(c);
	Real const cInv = Real(1.0) / c;
	std::array<std::vector<Real>, NRF> dUdx;
	for (auto &du : dUdx) {
		du.resize(RAD_N3);
	}
	int lb[NDIM] = { RAD_INTERIOR_BEGIN, RAD_INTERIOR_BEGIN, RAD_INTERIOR_BEGIN };
	int ub[NDIM] = { RAD_INTERIOR_END, RAD_INTERIOR_END, RAD_INTERIOR_END };
	int &xlb = lb[XDIM];
	int &ylb = lb[YDIM];
	int &zlb = lb[ZDIM];
	int &xub = ub[XDIM];
	int &yub = ub[YDIM];
	int &zub = ub[ZDIM];
	int DN = 1;
	for (int dim = ZDIM; dim >= XDIM; dim--) {
		lb[dim]--;
		ub[dim]++;
		for (integer xi = xlb; xi != xub; ++xi) {
			for (integer yi = ylb; yi != yub; ++yi) {
				for (integer zi = zlb; zi != zub; ++zi) {
					int const i0 = rindex(xi, yi, zi);
					Real const um = U[er_i][i0 - DN];
					Real const u0 = U[er_i][i0];
					Real const up = U[er_i][i0 + DN];
					Real const pSlope = up - u0;
					Real const mSlope = u0 - um;
					Real const sign = copysign(Real(0.25), pSlope) + copysign(Real(0.25), mSlope);
					dUdx[er_i][i0] = sign * min(abs(pSlope), abs(mSlope));
					for (int field = fx_i; field <= fz_i; field++) {
						Real const rEnergyM = U[er_i][i0 - DN];
						Real const rEnergy0 = U[er_i][i0];
						Real const rEnergyP = U[er_i][i0 + DN];
						Real um = U[field][i0 - DN] * cInv * INVERSE(rEnergyM);
						Real u0 = U[field][i0] * cInv * INVERSE(rEnergy0);
						Real up = U[field][i0 + DN] * cInv * INVERSE(rEnergyP);
						assert(one - abs(up) >= -eps);
						assert(one - abs(u0) >= -eps);
						assert(one - abs(um) >= -eps);
						um = copysign(min(one - eps, abs(um)), um);
						u0 = copysign(min(one - eps, abs(u0)), u0);
						up = copysign(min(one - eps, abs(up)), up);
						Real const pSlope = up - u0;
						Real const mSlope = u0 - um;
						Real const sign = copysign(Real(0.25), pSlope) + copysign(Real(0.25), mSlope);
						dUdx[field][i0] = sign * min(abs(pSlope), abs(mSlope));
					}
				}
			}
		}
		lb[dim]++;
		for (integer xi = xlb; xi != xub; ++xi) {
			for (integer yi = ylb; yi != yub; ++yi) {
				for (integer zi = zlb; zi != zub; ++zi) {
					int const rI = rindex(xi, yi, zi);
					int const lI = rI - DN;
					std::array<Real, NDIM> rFluxFlow;
					std::array<Real, NDIM> lFluxFlow;
					std::array<Real, NDIM> rUnitVector;
					std::array<Real, NDIM> lUnitVector;
					Real const rEnergySlope = dUdx[er_i][rI];
					Real const lEnergySlope = dUdx[er_i][lI];
					Real const rEnergy = U[er_i][rI] - rEnergySlope;
					Real const lEnergy = U[er_i][lI] + lEnergySlope;
					std::array<Real, NDIM> const rFluxSlope = { dUdx[fx_i][rI], dUdx[fy_i][rI], dUdx[fz_i][rI] };
					std::array<Real, NDIM> const lFluxSlope = { dUdx[fx_i][lI], dUdx[fy_i][lI], dUdx[fz_i][lI] };
					std::array<Real, NDIM> rFlux = { U[fx_i][rI], U[fy_i][rI], U[fz_i][rI] };
					std::array<Real, NDIM> lFlux = { U[fx_i][lI], U[fy_i][lI], U[fz_i][lI] };
					for (int d = 0; d < NDIM; d++) {
						rFlux[d] -= rFluxSlope[d] * c * rEnergy;
						lFlux[d] += lFluxSlope[d] * c * lEnergy;
					}
					Real const rEnergyInv = one / rEnergy;
					Real const lEnergyInv = one / lEnergy;
					Real rNormSquared = zero;
					Real lNormSquared = zero;
					for (int d = 0; d < NDIM; d++) {
						lUnitVector[d] = lFlux[d] * cInv * lEnergyInv;
						rUnitVector[d] = rFlux[d] * cInv * rEnergyInv;
					}
					for (int d = 0; d < NDIM; d++) {
						rNormSquared += sqr(rUnitVector[d]);
						lNormSquared += sqr(lUnitVector[d]);
					}
					assert(Real(0) <= lNormSquared);
					assert(Real(0) <= rNormSquared);
					assert(lNormSquared <= one);
					assert(rNormSquared <= one);
					Real const lNormInv = sqrt(lNormSquared) / (lNormSquared + Real::tiny());
					Real const rNormInv = sqrt(rNormSquared) / (rNormSquared + Real::tiny());
					for (int d = 0; d < NDIM; d++) {
						lUnitVector[d] = lFlux[d] * lNormInv;
						rUnitVector[d] = rFlux[d] * rNormInv;
					}
					Real const rEddingtonNum = Real(3) + Real(4) * rNormSquared;
					Real const rEddingtonDen = Real(5) + sqrt(Real(16) - Real(12) * rNormSquared);
					Real const rEddington = rEddingtonNum / rEddingtonDen;
					Real const lEddingtonNum = Real(3) + Real(4) * lNormSquared;
					Real const lEddingtonDen = Real(5) + sqrt(Real(16) - Real(12) * lNormSquared);
					Real const lEddington = lEddingtonNum / lEddingtonDen;
					auto const rLambda = radiationEigenvalues(rEddington, rUnitVector[dim]);
					auto const lLambda = radiationEigenvalues(lEddington, lUnitVector[dim]);
					Real const sL = c * min(min(lLambda.first, lLambda.second), min(rLambda.first, rLambda.second));
					Real const sR = c * max(max(lLambda.first, lLambda.second), max(rLambda.first, rLambda.second));
					Real const sDen = one / (sR - sL);
					Real const wL = +half * sR * sDen;
					Real const wR = -half * sL * sDen;
					Real const wLR = -half * sL * sR * sDen;
					auto const rEddingtonTensor = computeEddingtonTensor(rEnergy, rFlux);
					auto const lEddingtonTensor = computeEddingtonTensor(lEnergy, lFlux);
					flux[dim][er_i][rI] = wL * lFlux[dim] + wR * rFlux[dim] - wLR * (rEnergy - lEnergy);
					for (int d = 0; d < NDIM; d++) {
						flux[dim][fx_i + d][rI] = wL * lEddingtonTensor[dim][d] + wR * rEddingtonTensor[dim][d] - wLR * (rFlux[d] - lFlux[d]);
					}
				}
			}
		}
		ub[dim]--;
		DN *= RAD_NX;
	}
}

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

void rad_grid::set(const std::string name, real *data) {
	auto iter = str_to_index.find(name);
	real eunit = opts().problem == MARSHAK ? 1 : opts().code_to_g / std::pow(opts().code_to_s, 2) / opts().code_to_cm;
	real funit = opts().problem == MARSHAK ? 1 : eunit * opts().code_to_cm / opts().code_to_s;
	if (iter != str_to_index.end()) {
		int f = iter->second;
		int jjj = 0;
		for (int i = 0; i < INX; i++) {
			for (int j = 0; j < INX; j++) {
				for (int k = 0; k < INX; k++) {
					const int iii = rindex(k + RAD_BW, j + RAD_BW, i + RAD_BW);
					data[jjj] /= f == er_i ? eunit : funit;
					U[f][iii] = Real(data[jjj]);
					jjj++;
				}
			}
		}
	}

}

std::vector<silo_var_t> rad_grid::var_data() const {
	std::vector<silo_var_t> s;
	real eunit = opts().problem == MARSHAK ? 1 : opts().code_to_g / std::pow(opts().code_to_s, 2) / opts().code_to_cm;
	real funit = opts().problem == MARSHAK ? 1 : eunit * opts().code_to_cm / opts().code_to_s;
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

constexpr auto _0 = real(0);
constexpr auto _1 = real(1);
constexpr auto _2 = real(2);
constexpr auto _3 = real(3);
constexpr auto _4 = real(4);
constexpr auto _5 = real(5);

using set_rad_grid_action_type = node_server::set_rad_grid_action;
HPX_REGISTER_ACTION (set_rad_grid_action_type);

hpx::future<void> node_client::set_rad_grid(std::vector<real> &&g, std::vector<real> &&o) const {
	return hpx::async<typename node_server::set_rad_grid_action>(get_unmanaged_gid(), g, o);
}

void node_server::set_rad_grid(const std::vector<real> &data, const std::vector<real> &outflows) {
	rad_grid_ptr->set_prolong(data, outflows);
}

using send_rad_boundary_action_type = node_server::send_rad_boundary_action;
HPX_REGISTER_ACTION (send_rad_boundary_action_type);

using send_rad_flux_correct_action_type = node_server::send_rad_flux_correct_action;
HPX_REGISTER_ACTION (send_rad_flux_correct_action_type);

void node_client::send_rad_flux_correct(std::vector<real> &&data, const geo::face &face, const geo::octant &ci) const {
	hpx::apply<typename node_server::send_rad_flux_correct_action>(get_unmanaged_gid(), std::move(data), face, ci);
}

void node_server::recv_rad_flux_correct(std::vector<real> &&data, const geo::face &face, const geo::octant &ci) {
	const geo::quadrant index(ci, face.get_dimension());
	niece_rad_channels[face][index].set_value(std::move(data));
}

void node_client::send_rad_boundary(std::vector<real> &&data, const geo::direction &dir, std::size_t cycle) const {
	hpx::apply<typename node_server::send_rad_boundary_action>(get_gid(), std::move(data), dir, cycle);
}

void node_server::recv_rad_boundary(std::vector<real> &&bdata, const geo::direction &dir, std::size_t cycle) {
	sibling_rad_type tmp;
	tmp.data = std::move(bdata);
	tmp.direction = dir;
	sibling_rad_channels[dir].set_value(std::move(tmp), cycle);
}

using send_rad_children_action_type = node_server::send_rad_children_action;
HPX_REGISTER_ACTION (send_rad_children_action_type);

void node_server::recv_rad_children(std::vector<real> &&data, const geo::octant &ci, std::size_t cycle) {
	child_rad_channels[ci].set_value(std::move(data), cycle);
}

#include <fenv.h>

void node_client::send_rad_children(std::vector<real> &&data, const geo::octant &ci, std::size_t cycle) const {
	hpx::apply<typename node_server::send_rad_children_action>(get_unmanaged_gid(), std::move(data), ci, cycle);
}

#define ASSERT(a) if (!a) { \
   printf( "Error in %s on line %i\n", __FILE__, __LINE__); \
   abort(); \
   }

//Er_np1 + c * dt * kap * Er_np1 - 4.0 * dt * kap * sigma * ((gam - 1)*(mh*mmw * (Ei_n - Er_np1 + Er_n) / (rho * kb))^4;

#include <iostream>
#include <cmath>
#include <stdexcept>

void rad_grid::set_dx(real _dx) {
	dx = _dx;
}

void rad_grid::set_X(const std::vector<std::vector<real>> &x) {
	X.resize(NDIM);
	for (integer d = 0; d != NDIM; ++d) {
		X[d].resize(RAD_N3);
		for (integer xi = 0; xi != RAD_NX; ++xi) {
			for (integer yi = 0; yi != RAD_NX; ++yi) {
				for (integer zi = 0; zi != RAD_NX; ++zi) {
					const auto D = H_BW - RAD_BW;
					const integer ir = rindex(xi, yi, zi);
					const integer ih = hindex(xi + D, yi + D, zi + D);
					X[d][ir] = x[d][ih];
				}
			}
		}
	}
}

real rad_grid::hydro_signal_speed(const std::vector<real> &egas_, const std::vector<real> &tau_, const std::vector<real> &sx_, const std::vector<real> &sy_,
		const std::vector<real> &sz_, const std::vector<real> &rho_) {
	real a = 0.0;
	const real fgamma = grid::get_fgamma();
	const integer D = H_BW - RAD_BW;
	for (integer xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
		for (integer yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
			for (integer zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
				const integer ir = rindex(xi, yi, zi);
				const integer ih = hindex(xi + D, yi + D, zi + D);
				const real sx = sx_[ih];
				const real sy = sy_[ih];
				const real sz = sz_[ih];
				const real egas = egas_[ih];
				const real Erad = U[er_i][ir];
				const real Fx = U[fx_i][ir];
				const real Fy = U[fy_i][ir];
				const real Fz = U[fz_i][ir];
				const real rho = rho_[ih];
				CHECK_FLUX1(Erad, Fx, Fy, Fz, rho);
				const real inv_rho = INVERSE(rho);
				const real vx = sx * inv_rho;
				const real vy = sy * inv_rho;
				const real vz = sz * inv_rho;
				const real eint = std::max(egas - 0.5 * rho * (vx * vx + vy * vy + vz * vz), 0.0);
				const real kap = kappa_R(rho, eint, mmw[ih], X_spc[ih], Z_spc[ih]);
				real cs2 = (4.0 / 9.0) * Erad * inv_rho;
				const real cons = kap * dx;
				if (cons < 32.0) {
					cs2 *= 1.0 - std::exp(-std::min(32.0, cons));
				}
				a = std::max(cs2, a);
			}
		}
	}
	return SQRT(a);
}

void rad_grid::compute_mmw(const std::vector<std::vector<safe_real>> &U) {
	mmw.resize(RAD_N3);
	X_spc.resize(RAD_N3);
	Z_spc.resize(RAD_N3);
	for (integer i = 0; i != RAD_NX; ++i) {
		for (integer j = 0; j != RAD_NX; ++j) {
			for (integer k = 0; k != RAD_NX; ++k) {
				const integer d = H_BW - RAD_BW;
				const integer iiir = rindex(i, j, k);
				const integer iiih = hindex(i + d, j + d, k + d);
				specie_state_t<real> spc;
				for (integer si = 0; si != opts().n_species; ++si) {
					spc[si] = U[spc_i + si][iiih];
				}
				mean_ion_weight(spc, mmw[iiir], X_spc[iiir], Z_spc[iiir]);
			}
		}
	}

}

template<class T>
T minmod(T a, T b) {
	return (std::copysign(0.5, a) + std::copysign(0.5, b)) * std::min(std::abs(a), std::abs(b));
}

void rad_grid::allocate() {
	rad_grid::dx = dx;
	U.resize(NRF);
	Ushad.resize(NRF);
	flux.resize(NDIM);
	for (int d = 0; d < NDIM; d++) {
		flux[d].resize(NRF);
	}
	for (integer f = 0; f != NRF; ++f) {
		U[f].resize(RAD_N3);
		Ushad[f].resize(RAD_N3);
		for (integer d = 0; d != NDIM; ++d) {
			flux[d][f].resize(RAD_N3);
		}
	}
}

void rad_grid::sanity_check() {
	for (integer xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
		for (integer yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
			for (integer zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
				const integer iiir = rindex(xi, yi, zi);
				if (U[er_i][iiir] <= 0.0) {
					printf("INSANE\n");
					//		printf("%e %i %i %i\n", U[er_i][iiir], xi, yi, zi);
					abort();
				}
			}
		}
	}
}

void rad_grid::change_units(real m, real l, real t, real k) {
	const real l2 = l * l;
	const real t2 = t * t;
	const real t2inv = 1.0 * INVERSE(t2);
	const real tinv = 1.0 * INVERSE(t);
	const real l3 = l2 * l;
	const real l3inv = 1.0 * INVERSE(l3);
	for (integer i = 0; i != RAD_N3; ++i) {
		U[er_i][i] *= Real((m * l2 * t2inv) * l3inv);
		U[fx_i][i] *= Real(tinv * (m * t2inv));
		U[fy_i][i] *= Real(tinv * (m * t2inv));
		U[fz_i][i] *= Real(tinv * (m * t2inv));
	}
}

void rad_grid::advance(Real dt) {
	const Real l = dt / Real(dx);
	const integer D[3] = { DX, DY, DZ };
	for (integer xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
		for (integer yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
			for (integer zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
				const integer iii = rindex(xi, yi, zi);
				for (integer f = 0; f != NRF; ++f) {
					for (integer d = 0; d != NDIM; ++d) {
						U[f][iii] -= l * Real(flux[d][f][iii + D[d]] - flux[d][f][iii]);
					}
				}
			}
		}
	}
}

void rad_grid::set_physical_boundaries(geo::face face, real t) {
	for (integer i = 0; i != RAD_NX; ++i) {
		for (integer j = 0; j != RAD_NX; ++j) {
			for (integer k = 0; k != RAD_BW; ++k) {
				integer iii1, iii0;
				switch (face) {
				case 0:
					iii1 = rindex(k, i, j);
					iii0 = rindex(RAD_BW, i, j);
					break;
				case 1:
					iii1 = rindex(RAD_NX - 1 - k, i, j);
					iii0 = rindex(RAD_NX - 1 - RAD_BW, i, j);
					break;
				case 2:
					iii1 = rindex(i, k, j);
					iii0 = rindex(i, RAD_BW, j);
					break;
				case 3:
					iii1 = rindex(i, RAD_NX - 1 - k, j);
					iii0 = rindex(i, RAD_NX - 1 - RAD_BW, j);
					break;
				case 4:
					iii1 = rindex(i, j, k);
					iii0 = rindex(i, j, RAD_BW);
					break;
				case 5:
				default:
					iii1 = rindex(i, j, RAD_NX - 1 - k);
					iii0 = rindex(i, j, RAD_NX - 1 - RAD_BW);
				}
				for (integer f = 0; f != NRF; ++f) {
					U[f][iii1] = U[f][iii0];
				}
				switch (face) {
				case 0:
					if (opts().problem == MARSHAK) {
						if (t > 0) {
							auto u = marshak_wave_analytic(-opts().xscale, 0, 0, t);
							U[fx_i][iii1] = Real(u[opts().n_fields + fx_i]);
							U[er_i][iii1] = Real(std::max(u[opts().n_fields + er_i], 1.0e-10));
						} else {
							U[fx_i][iii1] = Real(0.0);
							U[er_i][iii1] = Real(1.0e-10);
						}
					} else {
						U[fx_i][iii1] = min(U[fx_i][iii1], Real(0.0));
					}
					break;
				case 1:
					U[fx_i][iii1] = max(U[fx_i][iii1], Real(0.0));
					break;
				case 2:
					U[fy_i][iii1] = min(U[fy_i][iii1], Real(0.0));
					break;
				case 3:
					U[fy_i][iii1] = max(U[fy_i][iii1], Real(0.0));
					break;
				case 4:
					U[fz_i][iii1] = min(U[fz_i][iii1], Real(0.0));
					break;
				case 5:
					U[fz_i][iii1] = max(U[fz_i][iii1], Real(0.0));
					break;
				}
				if (opts().problem == RADIATION_COUPLING) {
					U[fz_i][iii1] = U[fy_i][iii1] = U[fx_i][iii1] = Real(0.0);
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
			std::array<integer, NDIM> lb, ub;
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

	constexpr integer size = geo::face::count() * geo::quadrant::count();
	std::array<future<void>, size> futs;
	for (auto &f : futs) {
		f = hpx::make_ready_future();
	}
	integer index = 0;
	for (auto const &f : geo::face::full_set()) {
		if (this->nieces[f] == +1) {
			for (auto const &quadrant : geo::quadrant::full_set()) {
				futs[index++] = niece_rad_channels[f][quadrant].get_future().then([this, f, quadrant](hpx::future<std::vector<real> > &&fdata) -> void {
					const auto face_dim = f.get_dimension();
					std::array<integer, NDIM> lb, ub;
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

void rad_grid::set_flux_restrict(const std::vector<real> &data, const std::array<integer, NDIM> &lb, const std::array<integer, NDIM> &ub,
		const geo::dimension &dim) {

	integer index = 0;
	for (integer field = 0; field != NRF; ++field) {
		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					const integer iii = rindex(i, j, k);
					flux[dim][field][iii] = data[index];
					++index;
				}
			}
		}
	}
}

std::vector<real> rad_grid::get_flux_restrict(const std::array<integer, NDIM> &lb, const std::array<integer, NDIM> &ub, const geo::dimension &dim) const {

	std::vector<real> data;
	integer size = 1;
	for (auto &dim : geo::dimension::full_set()) {
		size *= (ub[dim] - lb[dim]);
	}
	size /= (NCHILD / 2);
	size *= NRF;
	data.reserve(size);
	const integer stride1 = (dim == XDIM) ? (RAD_NX) : (RAD_NX) * (RAD_NX);
	const integer stride2 = (dim == ZDIM) ? (RAD_NX) : 1;
	for (integer field = 0; field != NRF; ++field) {
		for (integer i = lb[XDIM]; i < ub[XDIM]; i += 2) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; j += 2) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; k += 2) {
					const integer i00 = rindex(i, j, k);
					const integer i10 = i00 + stride1;
					const integer i01 = i00 + stride2;
					const integer i11 = i00 + stride1 + stride2;
					real value = ZERO;
					value += flux[dim][field][i00];
					value += flux[dim][field][i10];
					value += flux[dim][field][i01];
					value += flux[dim][field][i11];
					value /= real(4);
					data.push_back(value);
				}
			}
		}
	}
	return data;
}

void node_server::all_rad_bounds() {
//	if( my_location.level() == 0 ) printf( "\nbounds 1\n");
	GET(exchange_interlevel_rad_data());
//	if( my_location.level() == 0 ) printf( "\nbounds 2\n");
	collect_radiation_bounds();
//	if( my_location.level() == 0 ) printf( "\nbounds 3\n");
	send_rad_amr_bounds();
//	if( my_location.level() == 0 ) printf( "\nbounds 4\n");
	rcycle++;
}

hpx::future<void> node_server::exchange_interlevel_rad_data() {

	hpx::future<void> f = hpx::make_ready_future();
	integer ci = my_location.get_child_index();

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

void node_server::collect_radiation_bounds() {

	rad_grid_ptr->clear_amr();
	for (auto const &dir : geo::direction::full_set()) {
		if (!neighbors[dir].empty()) {
			const integer width = H_BW;
			auto bdata = rad_grid_ptr->get_boundary(dir);
			neighbors[dir].send_rad_boundary(std::move(bdata), dir.flip(), rcycle);
		}
	}

	std::array<future<void>, geo::direction::count()> results;
	integer index = 0;
	for (auto const &dir : geo::direction::full_set()) {
		if (!(neighbors[dir].empty() && my_location.level() == 0)) {
			results[index++] = sibling_rad_channels[dir].get_future(rcycle).then(
			/*hpx::util::annotated_function(*/[this, dir](future<sibling_rad_type> &&f) -> void {
				auto &&tmp = GET(f);
				if (!neighbors[dir].empty()) {
					rad_grid_ptr->set_boundary(tmp.data, tmp.direction);
				} else {
					rad_grid_ptr->set_rad_amr_boundary(tmp.data, tmp.direction);

				}
			}/*, "node_server::collect_rad_boundaries::set_rad_boundary")*/);
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
	for (auto &face : geo::face::full_set()) {
		if (my_location.is_physical_boundary(face)) {
			rad_grid_ptr->set_physical_boundaries(face, current_time);
		}
	}

}

void rad_grid::initialize_erad(const std::vector<safe_real> rho, const std::vector<safe_real> tau) {
	const real fgamma = grid::get_fgamma();
	for (integer xi = 0; xi != RAD_NX; ++xi) {
		for (integer yi = 0; yi != RAD_NX; ++yi) {
			for (integer zi = 0; zi != RAD_NX; ++zi) {
				const auto D = H_BW - RAD_BW;
				const integer iiir = rindex(xi, yi, zi);
				const integer iiih = hindex(xi + D, yi + D, zi + D);
				const real ei = POWER(tau[iiih], fgamma);
				if (opts().problem == STAR || opts().problem == ROTATING_STAR || opts().problem == RADIATION_COUPLING) {
					U[er_i][iiir] = Real(B_p((double) rho[iiih], (double) ei, (double) mmw[iiir]) * (4.0 * M_PI / physcon().c));
					U[fx_i][iiir] = U[fy_i][iiir] = U[fz_i][iiir] = Real(0.0);
				}
			}
		}
	}
}

rad_grid::rad_grid(real _dx) :
		dx(_dx), is_coarse(RAD_N3), has_coarse(RAD_N3) {
	allocate();
}

rad_grid::rad_grid() :
		is_coarse(RAD_N3), has_coarse(RAD_N3), U_out(NRF, ZERO), U_out0(NRF, ZERO) {
	allocate();
}

void rad_grid::set_boundary(const std::vector<real> &data, const geo::direction &dir) {

	std::array<integer, NDIM> lb, ub;
	get_boundary_size(lb, ub, dir, OUTER, INX, RAD_BW);
	integer iter = 0;

	for (integer field = 0; field != NRF; ++field) {
		auto &Ufield = U[field];
		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					Ufield[rindex(i, j, k)] = Real(data[iter]);
					++iter;
				}
			}
		}
	}
}

std::vector<real> rad_grid::get_boundary(const geo::direction &dir) {

	std::array<integer, NDIM> lb, ub;
	std::vector<real> data;
	integer size = NRF * get_boundary_size(lb, ub, dir, INNER, INX, RAD_BW);
	data.resize(size);
	integer iter = 0;

	for (integer field = 0; field != NRF; ++field) {
		auto &Ufield = U[field];
		for (integer i = lb[XDIM]; i < ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j < ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k < ub[ZDIM]; ++k) {
					data[iter] = Ufield[rindex(i, j, k)];
					++iter;
				}
			}
		}
	}

	return data;
}

void rad_grid::set_field(real v, integer f, integer i, integer j, integer k) {
	U[f][rindex(i, j, k)] = Real(v);
}

real rad_grid::get_field(integer f, integer i, integer j, integer k) const {
	return U[f][rindex(i, j, k)];
}

void rad_grid::set_prolong(const std::vector<real> &data, const std::vector<real> &out) {
	integer index = 0;
	U_out = std::move(out);
	for (integer f = 0; f != NRF; ++f) {
		for (integer i = RAD_BW; i != RAD_NX - RAD_BW; ++i) {
			for (integer j = RAD_BW; j != RAD_NX - RAD_BW; ++j) {
				for (integer k = RAD_BW; k != RAD_NX - RAD_BW; ++k) {
					const integer iii = rindex(i, j, k);
					U[f][iii] = Real(data[index]);
					++index;
				}
			}
		}
	}
}

std::vector<real> rad_grid::get_prolong(const std::array<integer, NDIM> &lb, const std::array<integer, NDIM> &ub) {
	std::vector<real> data;
	integer size = NRF;
	for (integer dim = 0; dim != NDIM; ++dim) {
		size *= (ub[dim] - lb[dim]);
	}
	auto lb0 = lb;
	auto ub0 = ub;
	for (integer d = 0; d != NDIM; ++d) {
		lb0[d] /= 2;
		ub0[d] /= 2;
	}

	for (integer f = 0; f != NRF; ++f) {
		for (integer i = lb[XDIM]; i != ub[XDIM]; ++i) {
			for (integer j = lb[YDIM]; j != ub[YDIM]; ++j) {
				for (integer k = lb[ZDIM]; k != ub[ZDIM]; ++k) {
					const integer iii = rindex(i / 2, j / 2, k / 2);
					real value = U[f][iii];
					data.push_back(value);
				}
			}
		}
	}
	return data;
}

std::vector<real> rad_grid::get_restrict() const {
	std::vector<real> data;
	for (integer f = 0; f != NRF; ++f) {
		for (integer i = RAD_BW; i < RAD_NX - RAD_BW; i += 2) {
			for (integer j = RAD_BW; j < RAD_NX - RAD_BW; j += 2) {
				for (integer k = RAD_BW; k < RAD_NX - RAD_BW; k += 2) {
					const integer iii = rindex(i, j, k);
					real v = ZERO;
					for (integer x = 0; x != 2; ++x) {
						for (integer y = 0; y != 2; ++y) {
							for (integer z = 0; z != 2; ++z) {
								const integer jjj = iii + x * RAD_NX * RAD_NX + y * RAD_NX + z;
								v += U[f][jjj];
							}
						}
					}
					v /= real(NCHILD);
					data.push_back(v);
				}
			}
		}
	}
	return data;
}

void rad_grid::set_restrict(const std::vector<real> &data, const geo::octant &octant) {
	integer index = 0;
	const integer i0 = octant.get_side(XDIM) * (INX / 2);
	const integer j0 = octant.get_side(YDIM) * (INX / 2);
	const integer k0 = octant.get_side(ZDIM) * (INX / 2);
	for (integer f = 0; f != NRF; ++f) {
		for (integer i = RAD_BW; i != RAD_NX / 2; ++i) {
			for (integer j = RAD_BW; j != RAD_NX / 2; ++j) {
				for (integer k = RAD_BW; k != RAD_NX / 2; ++k) {
					const integer iii = rindex(i + i0, j + j0, k + k0);
					U[f][iii] = Real(data[index]);
					++index;
					if (index > int(data.size())) {
						printf("rad_grid::set_restrict error %i %i\n", int(index), int(data.size()));
					}
				}
			}
		}
	}

}
;

void node_server::send_rad_amr_bounds() {
	if (is_refined) {
		constexpr auto full_set = geo::octant::full_set();
		for (auto &ci : full_set) {
			const auto &flags = amr_flags[ci];
			for (auto &dir : geo::direction::full_set()) {
				if (flags[dir]) {
					std::array<integer, NDIM> lb, ub;
					std::vector<real> data;
					get_boundary_size(lb, ub, dir, OUTER, INX / 2, H_BW);
					for (integer dim = 0; dim != NDIM; ++dim) {
						lb[dim] = std::max(lb[dim] - 1, integer(0));
						ub[dim] = std::min(ub[dim] + 1, integer(HS_NX));
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
HPX_REGISTER_ACTION (erad_init_action_type);

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

void rad_grid::set_rad_amr_boundary(const std::vector<real> &data, const geo::direction &dir) {
	PROFILE();

	std::array<integer, NDIM> lb, ub;
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
		lb[dim] = std::max(lb[dim] - 1, integer(0));
		ub[dim] = std::min(ub[dim] + 1, integer(HS_NX));
	}

	for (int f = 0; f < NRF; f++) {
		for (int i = lb[0]; i < ub[0]; i++) {
			for (int j = lb[1]; j < ub[1]; j++) {
				for (int k = lb[2]; k < ub[2]; k++) {
					has_coarse[hSindex(i, j, k)]++;
					Ushad[f][hSindex(i, j, k)] = Real(data[l++]);
				}
			}
		}
	}
	assert(l == data.size());
}

std::vector<std::pair<std::string, real>> rad_grid::get_outflows() const {
	std::vector<std::pair<std::string, real>> rc;
	rc.reserve(NRF);
	rc.push_back(std::make_pair("er", U_out[0]));
	rc.push_back(std::make_pair("fx", U_out[1]));
	rc.push_back(std::make_pair("fy", U_out[2]));
	rc.push_back(std::make_pair("fz", U_out[3]));
	return rc;
}

void rad_grid::set_outflows(std::vector<std::pair<std::string, real>> &&u) {
	for (const auto &p : u) {
		set_outflow(p);
	}
}

void rad_grid::set_outflow(const std::pair<std::string, real> &p) {
	if (p.first == "er") {
		U_out[0] = p.second;
	} else if (p.first == "fx") {
		U_out[1] = p.second;
	} else if (p.first == "fy") {
		U_out[2] = p.second;
	} else if (p.first == "fz") {
		U_out[3] = p.second;
	} else {
		assert(false);
	}
}

void rad_grid::complete_rad_amr_boundary() {
	PROFILE();

	using oct_array = std::array<std::array<std::array<Real, 2>, 2>, 2>;
	static thread_local std::vector<std::vector<oct_array>> Uf(NRF, std::vector<oct_array>(HS_N3));

	const auto limiter = [](Real a, Real b) {
		return minmod_theta(a, b, 64. / 37.);
	};

	for (int f = 0; f < NRF; f++) {

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
									const auto s_xy = limiter(uc[iii0 + is * HS_DNX + js * HS_DNY] - u0, u0 - uc[iii0 - is * HS_DNX - js * HS_DNY]);
									const auto s_xz = limiter(uc[iii0 + is * HS_DNX + ks * HS_DNZ] - u0, u0 - uc[iii0 - is * HS_DNX - ks * HS_DNZ]);
									const auto s_yz = limiter(uc[iii0 + js * HS_DNY + ks * HS_DNZ] - u0, u0 - uc[iii0 - js * HS_DNY - ks * HS_DNZ]);
									const auto s_xyz = limiter(uc[iii0 + is * HS_DNX + js * HS_DNY + ks * HS_DNZ] - u0,
											u0 - uc[iii0 - is * HS_DNX - js * HS_DNY - ks * HS_DNZ]);
									auto &uf = Uf[f][iii0][ir][jr][kr];
									uf = u0;
									uf += Real(9.0 / 64.0) * Real(s_x + s_y + s_z);
									uf += Real(3.0 / 64.0) * Real(s_xy + s_yz + s_xz);
									uf += Real(1.0 / 64.0) * Real(s_xyz);
								}
							}
						}
					}
				}
			}
		}
	}

	for (int f = 0; f < NRF; f++) {
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

std::vector<real> rad_grid::get_subset(const std::array<integer, NDIM> &lb, const std::array<integer, NDIM> &ub) {
	PROFILE();
	std::vector<real> data;
	for (int f = 0; f < NRF; f++) {
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

using send_rad_amr_boundary_action_type = node_server:: send_rad_amr_boundary_action;
HPX_REGISTER_ACTION (send_rad_amr_boundary_action_type);

void node_server::recv_rad_amr_boundary(std::vector<real> &&bdata, const geo::direction &dir, std::size_t cycle) {
	sibling_rad_type tmp;
	tmp.data = std::move(bdata);
	tmp.direction = dir;
	sibling_rad_channels[dir].set_value(std::move(tmp), cycle);
}

void node_client::send_rad_amr_boundary(std::vector<real> &&data, const geo::direction &dir, std::size_t cycle) const {
	hpx::apply<typename node_server::send_rad_amr_boundary_action>(get_unmanaged_gid(), std::move(data), dir, cycle);
}

#endif
