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
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>
#include "octotiger/unitiger/radiation/radiation_physics_impl.hpp"
#include "octotiger/matrix.hpp"

#include <experimental/simd>


#if !defined(HPX_COMPUTE_DEVICE_CODE)

//#define SANITY_CHECK(ptr)
#define SANITY_CHECK1(ptr, lev) (ptr)->sanity_check(__FILE__, __LINE__, (lev))
#define SANITY_CHECK(ptr) (ptr)->sanity_check(__FILE__, __LINE__, -1)

using real = double;

#define CHECK_FLUX( er, fx, fy, fz) if( ((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c) > 1 ) {printf( "flux exceded %s %i %e fx %e fy %e fz %e er %e\n", __FILE__, __LINE__, sqrt(((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c)), fx, fy, fz, er*physcon().c); abort();}
#define CHECK_FLUX1( er, fx, fy, fz, rho) if( ((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c) > 1 ) {printf( "flux exceded %s %i %e fx %e fy %e fz %e er %e rho %e\n", __FILE__, __LINE__, sqrt(((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c)), fx, fy, fz, er*physcon().c, rho); abort();}
#define BAD_FLUX( er, fx, fy, fz, rho) bool(((fx)*(fx)+(fy)*(fy)+(fz)*(fz))/(er*er*physcon().c*physcon().c) > 1 )

std::unordered_map<std::string, int> rad_grid::str_to_index;
std::unordered_map<int, std::string> rad_grid::index_to_str;

void node_server::compute_radiation(Real dt, Real omega) {
	static constexpr Real zero = Real(0), third = Real(1.0 / 3.0), half = Real(0.5), one = Real(1), two = Real(2), three = Real(3), four = Real(4), five = Real(
			5);
	const int level = my_location.level();
	const bool root = bool(level == 0);
	auto rgrid = rad_grid_ptr;
	SANITY_CHECK1(rgrid, level);
	rgrid->set_dx(grid_ptr->get_dx());
	rgrid->set_X(grid_ptr->get_X());
	rgrid->compute_mmw(grid_ptr->U);
	const Real min_dx = Real(TWO * grid::get_scaling_factor() / real(INX << opts().max_level));
	const Real c = Real(physcon().c);
	const Real max_dt = third * Real(min_dx / c);
	const integer Nsubstep = std::max(int(std::ceil(dt / max_dt)), 1);
	const Real sub_dt = dt / Real(Nsubstep);
	if (root) {
		std::cout << "Radiation Transport with " << std::to_string(Nsubstep) << " substeps\n";
	}
	all_rad_bounds();
	using rad_store_t = decltype(rgrid->U);
	using gas_store_t = decltype(grid_ptr->U);
	auto &Ur = rgrid->U;
	auto &Ug = grid_ptr->U;
	const integer N2 = (Nsubstep + 1) / 2;
	all_rad_bounds();
	rgrid->implicit_source(Ug, dt);
	all_hydro_bounds();
	for (integer i = 0; i != Nsubstep; ++i) {
		all_rad_bounds();
		SANITY_CHECK1(rgrid, level);
		auto const Ur0 = Ur;
		if (root) {
			std::cout << "               \rsubstep = " << std::to_string(i);
			std::fflush(stdout);
		}
		SANITY_CHECK1(rgrid, level);
		rgrid->compute_flux(omega);
		GET(exchange_rad_flux_corrections());
		rgrid->advance(sub_dt);
		all_rad_bounds();
		SANITY_CHECK1(rgrid, level);
		rgrid->compute_flux(omega);
		GET(exchange_rad_flux_corrections());
		rgrid->advance(sub_dt);
		for (int f = 0; f < NRF; f++) {
			auto &ur0 = Ur0[f];
			auto &ur = Ur[f];
			for (int i = 0; i < RAD_N3; i++) {
				ur[i] += half * (ur0[i] - ur[i]);
			}
		}
		all_rad_bounds();
		SANITY_CHECK1(rgrid, level);
	}
	if (root) {
		std::cout << "\rradiation done\n";
	}
	SANITY_CHECK1(rgrid, level);
}

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

#include <array>
#include <stdexcept>

using Vector3 = std::array<Real, NDIM>;
using Matrix3x3 = std::array<std::array<Real, NDIM>, NDIM>;
using Matrix4x4 = std::array<std::array<Real, NRF>, NRF>;

Real determinant3x3(Real a00, Real a01, Real a02, Real a10, Real a11, Real a12, Real a20, Real a21, Real a22) {
	return a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);
}

Matrix3x3 findRotationMatrix(Vector3 const &v, Vector3 const &a) {
	constexpr Real zero(0);
	constexpr Real one(1);
	Vector3 u;
	Matrix3x3 A;
	u[0] = +a[1] * v[2] - a[2] * v[1];
	u[1] = -a[0] * v[2] + a[2] * v[0];
	u[2] = +a[0] * v[1] - a[1] * v[0];
	Real c = sqrt(sqr(u[0]) + sqr(u[1]) + sqr(u[2]));
	if (c) {
		for (int d = 0; d < NDIM; d++) {
			u[d] /= c;
		}
		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m < NDIM; m++) {
				A[n][m] += u[n] * u[m];
			}
			A[n][n] -= one;
		}
	} else {
		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m < NDIM; m++) {
				A[n][m] = zero;
			}
			A[n][n] = one;
		}
	}
	return A;
}

Matrix3x3 matrixInverse(const Matrix3x3 &M) {
	Real constexpr one(1);
	Matrix3x3 A;
	A[0][0] = +M[1][1] * M[2][2] - M[1][2] * M[2][1];
	A[1][0] = -M[1][0] * M[2][2] + M[1][2] * M[2][0];
	A[2][0] = +M[1][0] * M[2][1] - M[1][1] * M[2][0];
	A[0][1] = -M[0][1] * M[2][2] + M[0][2] * M[2][1];
	A[1][1] = +M[0][0] * M[2][2] - M[0][2] * M[2][0];
	A[2][1] = -M[0][0] * M[2][1] + M[0][1] * M[2][0];
	A[0][2] = -M[0][1] * M[1][2] + M[0][2] * M[1][1];
	A[1][2] = +M[0][0] * M[1][2] - M[0][2] * M[1][0];
	A[2][2] = -M[0][0] * M[1][1] + M[0][1] * M[1][0];
	Real const det = M[0][0] * A[0][0] + M[1][0] * A[1][0] + M[2][0] * A[2][0];
	Real const detInv = one / det;
	for (int n = 0; n < NDIM; n++) {
		for (int m = 0; m < NDIM; m++) {
			A[n][m] *= detInv;
		}
	}
	return A;
}

Matrix4x4 matrixInverse(const Matrix4x4 &m) {
	Real det = m[0][0] * determinant3x3(m[1][1], m[1][2], m[1][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], m[3][3])
			- m[0][1] * determinant3x3(m[1][0], m[1][2], m[1][3], m[2][0], m[2][2], m[2][3], m[3][0], m[3][2], m[3][3])
			+ m[0][2] * determinant3x3(m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], m[2][3], m[3][0], m[3][1], m[3][3])
			- m[0][3] * determinant3x3(m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], m[3][2]);

	if (det == Real(0)) {
		throw std::runtime_error("Matrix is singular and cannot be inverted.");
	}
	Real inv_det = Real(1) / det;
	Matrix4x4 cofactors = { { { { determinant3x3(m[1][1], m[1][2], m[1][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], m[3][3]), -determinant3x3(m[1][0],
			m[1][2], m[1][3], m[2][0], m[2][2], m[2][3], m[3][0], m[3][2], m[3][3]), determinant3x3(m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], m[2][3],
			m[3][0], m[3][1], m[3][3]), -determinant3x3(m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], m[3][2]) } }, { {
			-determinant3x3(m[0][1], m[0][2], m[0][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], m[3][3]), determinant3x3(m[0][0], m[0][2], m[0][3], m[2][0],
					m[2][2], m[2][3], m[3][0], m[3][2], m[3][3]), -determinant3x3(m[0][0], m[0][1], m[0][3], m[2][0], m[2][1], m[2][3], m[3][0], m[3][1],
					m[3][3]), determinant3x3(m[0][0], m[0][1], m[0][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], m[3][2]) } }, { { determinant3x3(m[0][1],
			m[0][2], m[0][3], m[1][1], m[1][2], m[1][3], m[3][1], m[3][2], m[3][3]), -determinant3x3(m[0][0], m[0][2], m[0][3], m[1][0], m[1][2], m[1][3],
			m[3][0], m[3][2], m[3][3]), determinant3x3(m[0][0], m[0][1], m[0][3], m[1][0], m[1][1], m[1][3], m[3][0], m[3][1], m[3][3]), -determinant3x3(
			m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[3][0], m[3][1], m[3][2]) } }, { { -determinant3x3(m[0][1], m[0][2], m[0][3], m[1][1],
			m[1][2], m[1][3], m[2][1], m[2][2], m[2][3]), determinant3x3(m[0][0], m[0][2], m[0][3], m[1][0], m[1][2], m[1][3], m[2][0], m[2][2], m[2][3]),
			-determinant3x3(m[0][0], m[0][1], m[0][3], m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], m[2][3]), determinant3x3(m[0][0], m[0][1], m[0][2], m[1][0],
					m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]) } } } };
	Matrix4x4 inverse;
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			inverse[i][j] = inv_det * cofactors[i][j];
		}
	}
	return inverse;
}

struct testImplicitRadiation {
	static constexpr Real zero = Real(0), sixth = Real(1.0 / 6.0), third = Real(1.0 / 3.0), half = Real(0.5), one = Real(1), three_halves = Real(1.5), two =
			Real(2), three = Real(3), four = Real(4), five = Real(5), twelve = Real(12);
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
			Beta[n] = Beta0[n] - (F[n] - F0[n]) / (rho * c);
		}
		Real Ek = zero;
		std::array<Real, NDIM> dEk_dF;
		for (int n = 0; n < NDIM; n++) {
			Ek += half * rho * sqr(c * Beta[n]);
			dEk_dF[n] = -Beta[n];
		}
		Real const Eg = Eg0 + Er0 - Er;
		Real const dEg_dEr = -one;
		Real Ei = max(Eg - Ek, zero);
		if (Ei < zero) {
			std::cout << "Er = " << std::to_string(Er) << "\n";
			std::cout << "Eg = " << std::to_string(Eg) << "\n";
			std::cout << "Ei = " << std::to_string(Ei) << "\n";
			std::cout << "Fx = " << std::to_string(F[0]) << "\n";
			std::cout << "Fy = " << std::to_string(F[1]) << "\n";
			std::cout << "Fz = " << std::to_string(F[2]) << "\n";
			std::cout << "Bx = " << std::to_string(Beta[0]) << "\n";
			std::cout << "By = " << std::to_string(Beta[1]) << "\n";
			std::cout << "Bz = " << std::to_string(Beta[2]) << "\n";
			std::cout << "Er0 = " << std::to_string(Er0) << "\n";
			std::cout << "Eg0 = " << std::to_string(Eg0) << "\n";
			std::cout << "F0x = " << std::to_string(F0[0]) << "\n";
			std::cout << "F0y = " << std::to_string(F0[1]) << "\n";
			std::cout << "F0z = " << std::to_string(F0[2]) << "\n";
			std::cout << "B0x = " << std::to_string(Beta0[0]) << "\n";
			std::cout << "B0y = " << std::to_string(Beta0[1]) << "\n";
			std::cout << "B0z = " << std::to_string(Beta0[2]) << "\n";
			throw;
		}
		std::array<Real, NDIM> dEi_dF;
		for (int n = 0; n < NDIM; n++) {
			dEi_dF[n] = -dEk_dF[n];
		}
		Real const dEi_dEr = -one;
		Real const iCv = (mu * mAMU) / ((gamma - one) * kB * rho);
		Real const T = Ei * iCv;
		Real const dT_dEr = -iCv;
		std::array<Real, NDIM> dT_dF;
		for (int n = 0; n < NDIM; n++) {
			dT_dF[n] = dEi_dF[n] * iCv;
		}
		Real const T2 = sqr(T);
		Real const T3 = T * T2;
		Real const T4 = sqr(T2);
		std::array<std::array<Real, NDIM>, NDIM> dBeta_dF;
		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m < NDIM; m++) {
				dBeta_dF[n][m] = -I[n][m] / (rho * c);
			}
		}
		Real hr = Er - Er0 + dt * cHat * kappa * (Er - aR * T4);
		Real dhr_dEr = one + dt * cHat * kappa * (one - four * aR * T3 * dT_dEr);
		for (int d = 0; d < NDIM; d++) {
			hr += -dt * cHat * Chi * Beta[d] * F[d];
		}
		std::array<Real, NDIM> dhr_dF;
		for (int n = 0; n < NDIM; n++) {
			dhr_dF[n] = -dt * cHat * Chi * (Beta[n] - F[n] / (rho * c));
		}
		std::array<Real, NDIM> Hr;
		for (int n = 0; n < NDIM; n++) {
			Hr[n] = F[n] - F0[n] + dt * cHat * Chi * (F[n] - four * third * Er * Beta[n]);
		}
		std::array<Real, NDIM> dHr_dEr;
		for (int n = 0; n < NDIM; n++) {
			dHr_dEr[n] = four * third * dt * cHat * Chi;
		}
		std::array<std::array<Real, NRF>, NRF> dHr_dF;
		for (int n = 0; n < NDIM; n++) {
			for (int m = 0; m < NDIM; m++) {
				dHr_dF[n][m] = I[n][m] * (one + dt * cHat * Chi) - dt * four * third * Er * dBeta_dF[n][m];
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
	static constexpr Real zero = Real(0), one = Real(1), two = Real(2);
	Real const c = Real(physcon().c);
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
	int const maxIterations = 1 << 12;
	int numIterations = 0;
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
		Real theta = min(one, Real(0.1) * x[NDIM] / abs(dx[NDIM] + Real::tiny()));
		for (int n = 0; n < NRF; n++) {
//			printf( "%i %e\n", n, dx[n]);
			x[n] += Real(0.9) * dx[n];
		}
		numIterations++;
		if (numIterations >= maxIterations) {
			printf("Maximum iterations exceeded for implicit radiation solve\n");
			abort();
		}
	} while (err > toler);
	for (int n = 0; n < NDIM; n++) {
		Mg[n] = rho * c * Beta0[n] + F[n] - x[n];
		F[n] = c * x[n];
		Real const thisBeta = Mg[n] / (rho * c * c);
		if (abs(thisBeta) > one) {
			printf("%e\n", (double) thisBeta);
		}
	}
	Eg = Eg0 + Er - x[NDIM];
	Er = x[NDIM];
}

Real eta(Real x, Real sigma) {
	constexpr Real zero = Real(0.0), half = Real(0.5), one = Real(1);
	x = max(x, Real(1e-100));
	Real argument = log(x) / sigma;
	return half * (one + erf(argument));
}

void rad_grid::implicit_source(std::vector<std::vector<real>> &hydroVars, Real dt) {
PROFILE()
																										SANITY_CHECK(this);
	constexpr real toler = 1000.0 * std::numeric_limits<real>::epsilon();
	constexpr Real zero = Real(0.0), half = Real(0.5), one = Real(1);
	constexpr real alpha = 0.5;
	constexpr real tiny = std::numeric_limits<real>::min();
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);
	Real const sigma(physcon().sigma);
	Real const mh(physcon().mh);
	Real const kb(physcon().kb);
	Real const gam(grid::get_fgamma());
	Real const c(physcon().c);
	Real const ar(4.0 * sigma * INVERSE(c));
	Real const inv_c(INVERSE(physcon().c));
	Real const inv_gam(INVERSE(gam));
	Real const c_hat(c * opts().clight_reduced);
	Real const inv_cc_hat = one / (c * c_hat);
	Real const inv_c2 = one / (c * c);
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
				Real const rho(hydroVars[rho_i][ih]);
				for (int n = 0; n < NDIM; n++) {
					F[n] = U[fx_i + n][ir];
					Mg[n] = Real(hydroVars[sx_i + n][ih]);
					Ek += half * sqr(Real(Mg[n])) / rho;
				}
				Real const Ei0 = Eg - Ek;
				Real const mu(mmw[ih]);
				Real const Xspc(X_spc[ih]);
				Real const Zspc(Z_spc[ih]);
				Real const kappa = kappa_p(rho, Real(Mg[NDIM]) - Ek, mu, Xspc, Zspc);
				Real const Etot = Er + Eg;
				Real const Chi = eta(Eg / (Etot * Real(0.001)), Real(0.2)) * kappa_R(rho, Real(Mg[NDIM]) - Ek, mu, Xspc, Zspc);
				solveImplicitRadiation(Er, F, Eg, Mg, rho, mu, kappa, Chi, Real(gam), dt);
				//		solveImplicitRadiation(Er, F, Eg, Mg, rho, mu, zero, zero, Real(gam), dt);

				Real f = zero;
				for (int n = 0; n < NDIM; n++) {
					f += sqr(F[n]);
				}
				f = sqrt(f) / Er;
				if (f > one) {
					printf("f = sqrt(f) / Er = %e\n", f);
					printf("U[er_i][ir] = %e\n", U[er_i][ir]);
					printf("U[fx_i][ir] = %e\n", U[fx_i][ir]);
					printf("U[fy_i][ir] = %e\n", U[fy_i][ir]);
					printf("U[fz_i][ir] = %e\n", U[fz_i][ir]);
					printf("hydroVars[egas_i][ih] = %e\n", hydroVars[egas_i][ih]);
					printf("hydroVars[sx_i][ih] = %e\n", hydroVars[sx_i][ih]);
					printf("hydroVars[sy_i][ih] = %e\n", hydroVars[sy_i][ih]);
					printf("hydroVars[sz_i][ih] = %e\n", hydroVars[sz_i][ih]);
					abort();
				}

				U[er_i][ir] = Er;
				hydroVars[egas_i][ih] = Eg;
				Ek = zero;
				for (int n = 0; n < NDIM; n++) {
					U[fx_i + n][ir] = F[n];
					hydroVars[sx_i + n][ih] = Mg[n];
					Ek += half * sqr(Real(Mg[n])) / rho;
				}
				Real const Ei1 = Eg - Ek;
				hydroVars[tau_i][ih] = pow(Ei1, one / gam);
				Real f2 = zero;
				for (int n = 0; n < NDIM; n++) {
					f2 += sqr(U[fx_i + n][ir]);
				}
				f = sqrt(f2) / U[er_i][ir];
				if (f > one) {
					printf("f  =    %e\n", f);
					abort();
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

inline Real vanleer(Real a, Real b) {
	static Real constexpr zero = Real(0), two = Real(2);
	Real const ab = a + b;
	return two * max(a * b, zero) * ab / (ab * ab + Real::tiny());
}

inline Real minmod(Real a, Real b) {
	static Real constexpr half = Real(0.5);
	return (copysign(half, a) + copysign(half, b)) * min(abs(a), abs(b));
}

void rad_grid::compute_flux(real omega) {
PROFILE()
									SANITY_CHECK(this);
	static Real constexpr zero = Real(0), half = Real(0.5), one = Real(1), two = Real(2), one_third = Real(1.0 / 3.0), two_thirds = Real(2.0 / 3.0), three =
			Real(3), four = Real(4), five = Real(5);
	static Real constexpr eps = Real::epsilon();
	static Real constexpr f0 = Real(0.999);

	Real const c = Real(physcon().c);
	Real const c2 = sqr(c);
	Real const cInv = Real(1.0) / c;
	std::array<std::vector<Real>, NRF> Up;
	std::array<std::vector<Real>, NRF> Um;
	for (auto &du : Up) {
		du.resize(RAD_N3, one);
	}
	for (auto &du : Um) {
		du.resize(RAD_N3, one);
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
		ub[dim]++;
		lb[dim]--;
		for (int iii = 0; iii < RAD_N3; iii++) {
			for (int d = 0; d < NDIM; d++) {
				U[fx_i + d][iii] /= U[er_i][iii];
			}
		}
		for (integer xi = xlb; xi != xub; ++xi) {
			for (integer yi = ylb; yi != yub; ++yi) {
				for (integer zi = zlb; zi != zub; ++zi) {
					std::array<Real, NDIM> f0, df0;
					int const iii = rindex(xi, yi, zi);
					Real const em = U[er_i][iii - DN];
					Real const e0 = U[er_i][iii];
					Real const ep = U[er_i][iii + DN];
					Real const dep = ep - e0;
					Real const dem = e0 - em;
					Real const de0 = minmod(dep, dem);
					Real f2p = zero, f2m = zero, f2 = zero;
					for (int d = 0; d < NDIM; d++) {
						Real const fm = U[fx_i + d][iii - DN];
						Real const fp = U[fx_i + d][iii + DN];
						f0[d] = U[fx_i + d][iii];
						Real const dfp = fp - f0[d];
						Real const dfm = f0[d] - fm;
						df0[d] = minmod(dfp, dfm);
//						f2m += sqr(fm);
//						f2p += sqr(fp);
//						f2 += sqr(f0[d]);
					}
//					Real fp = sqrt(f2p) / ep;
//					Real fm = sqrt(f2m) / em;
//					Real const f = sqrt(f2) / e0;
//					Real const w0(0.999);
//					Real const w1 = one - w0;
//					Real const am = sqr(min(one, max(w0 * f + w1 * fm, w1 * f + w0 * fm)));
//					Real const ap = sqr(min(one, max(w0 * f + w1 * fp, w1 * f + w0 * fp)));
//					Real Am = -am * sqr(e0);
//					Real Bm = -am * e0 * de0;
//					Real Cm = -am * sqr(half * de0);
//					Real Ap = -ap * sqr(e0);
//					Real Bp = -ap * e0 * de0;
//					Real Cp = -ap * sqr(half * de0);
//					for (int d = 0; d < NDIM; d++) {
//						Ap += sqr(f0[d]);
//						Am += sqr(f0[d]);
//						Bp += f0[d] * df0[d];
//						Bm -= f0[d] * df0[d];
//						Cp += sqr(half * df0[d]);
//						Cm += sqr(half * df0[d]);
//					}
//					Real const m0 = -Bm / (two * Cm + Real::tiny());
//					Real const p0 = -Bp / (two * Cp + Real::tiny());
//					Real const m1 = sqrt(sqr(Bm) - four * Am * Cm) / (two * Cm + Real::tiny());
//					Real const p1 = sqrt(sqr(Bp) - four * Ap * Cp) / (two * Cp + Real::tiny());
//					Real const alpha1 = min(one, min(max(p0 + p1, p0 - p1), max(m0 + m1, m0 - m1)));
//					Real const alpha2 = max(zero, max(min(p0 + p1, p0 - p1), min(m0 + m1, m0 - m1)));
//					Real const alpha = min(alpha1, alpha2);
					//					Up[er_i][iii] = e0 + half * alpha * de0;
					//					Um[er_i][iii] = e0 - half * alpha * de0;
					Up[er_i][iii] = e0 + half * de0;
					Um[er_i][iii] = e0 - half * de0;
					for (int d = 0; d < NDIM; d++) {
						Up[fx_i + d][iii] = f0[d] + half * df0[d];
						Um[fx_i + d][iii] = f0[d] - half * df0[d];
						f2p += sqr(Up[fx_i + d][iii]);
						f2m += sqr(Um[fx_i + d][iii]);
					}
					Real const fp = sqrt(f2p);
					Real const fm = sqrt(f2m);
					if ((fp > one) || (fm > one)) {
						printf("\n fp, fm, %e %e\n", fp, fm);
						printf("%e %e %e %e\n", e0, f0[0], f0[1], f0[2]);
						printf("%e %e %e %e\n", de0, df0[0], df0[1], df0[2]);
//						printf("%e %e\n", p0 + p1, p0 - p1);
//						printf("%e %e\n", m0 + m1, m0 - m1);
						abort();
					}
				}
			}
		}
		for (int iii = 0; iii < H_N3; iii++) {
			for (int d = 0; d < NDIM; d++) {
				U[fx_i + d][iii] *= U[er_i][iii];
				Up[fx_i + d][iii] *= Up[er_i][iii];
				Um[fx_i + d][iii] *= Um[er_i][iii];
			}
		}
		lb[dim]++;
		Real const ti = Real::tiny();
		for (integer xi = xlb; xi != xub; ++xi) {
			for (integer yi = ylb; yi != yub; ++yi) {
				for (integer zi = zlb; zi != zub; ++zi) {
					int const rI = rindex(xi, yi, zi);
					int const lI = rI - DN;
					const Real ER = Um[er_i][rI];
					const Real EL = Up[er_i][lI];
					const std::array<Real, NDIM> FR = { Um[fx_i][rI], Um[fy_i][rI], Um[fz_i][rI] };
					const std::array<Real, NDIM> FL = { Up[fx_i][lI], Up[fy_i][lI], Up[fz_i][lI] };
					Real const ERinv = one / ER;
					Real const ELinv = one / EL;
					Real f2L = zero, f2R = zero;
					Real F2L = zero, F2R = zero;
					for (int d = 0; d < NDIM; d++) {
						F2R += sqr(FR[d]);
						F2L += sqr(FL[d]);
						f2R += sqr(FR[d] * cInv * ERinv);
						f2L += sqr(FL[d] * cInv * ELinv);
					}
					assert(Real(0) <= f2L);
					assert(Real(0) <= f2R);
					if ((f2L > one) || (f2R > one)) {
						std::cout << "f2L = " << f2L << " f2R = " << f2R << "\n";
						std::cout << "EL = " << EL << " ER = " << ER << "\n";
						for (int d = 0; d < NDIM; d++) {
							std::cout << "FL[d] = " << FL[d] << " FR[d] = " << FR[d] << "\n";
						}
						fflush(stdout);
						abort();
					}
					std::array<Real, NDIM> muL, muR;
					std::array<std::array<Real, NDIM>, NDIM> PL, PR;
					Real const absFL = sqrt(F2L);
					Real const absFR = sqrt(F2R);
					Real const fR = sqrt(f2R);
					Real const fL = sqrt(f2L);
					for (int d = 0; d < NDIM; d++) {
						muL[d] = FL[d] / (absFL + ti);
						muR[d] = FR[d] / (absFR + ti);
					}
					Real const XiL = one_third * (five - two * sqrt(four - three * f2L));
					Real const XiR = one_third * (five - two * sqrt(four - three * f2R));
					Real const difCoR = half * (one - XiR);
					Real const difCoL = half * (one - XiL);
					Real const strCoR = half * (three * XiR - one);
					Real const strCoL = half * (three * XiL - one);
					for (int l = 0; l < NDIM; l++) {
						for (int m = 0; m < NDIM; m++) {
							PR[l][m] = ER * (strCoR * muR[l] * muR[m] + Real(l == m) * difCoR);
							PL[l][m] = EL * (strCoL * muL[l] * muL[m] + Real(l == m) * difCoL);
						}
					}
					Real const denR = sqrt(four - three * f2R);
					Real const denL = sqrt(four - three * f2L);
					Real const part1R = muR[dim] * fR;
					Real const part1L = muL[dim] * fL;
					Real const part2R = sqrt(two_thirds * (four - three * f2R - denR) + two * sqr(muR[dim]) * (two - f2R - denR));
					Real const part2L = sqrt(two_thirds * (four - three * f2L - denL) + two * sqr(muL[dim]) * (two - f2L - denL));
					Real const l1 = (part1R - part2R) / denR;
					Real const l2 = (part1R + part2R) / denR;
					Real const l3 = (part1L - part2L) / denL;
					Real const l4 = (part1L + part2L) / denL;
					Real const sL = c * min(min(min(l1, l2), min(l3, l4)), zero);
					Real const sR = c * max(max(max(l1, l2), max(l3, l4)), zero);
					flux[dim][er_i][rI] = (sR * FL[dim] - sL * FR[dim] + sR * sL * (ER - EL)) / (sR - sL);
					for (int d = 0; d < NDIM; d++) {
						flux[dim][fx_i + d][rI] = (sR * PL[dim][d] - sL * PR[dim][d] + sR * sL * (FR[d] - FL[d])) / (sR - sL);
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
	bool const flag = opts().problem == MARSHAK || opts().output_code_units;
	real eunit = flag ? 1 : opts().code_to_g / std::pow(opts().code_to_s, 2) / opts().code_to_cm;
	real funit = flag ? 1 : eunit * opts().code_to_cm / opts().code_to_s;
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

void rad_grid::sanity_check(const char *filename, int line, int level) {
	Real const c = Real(physcon().c);
	bool flag = false;
	for (integer xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
		for (integer yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
			for (integer zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
				const integer iii = rindex(xi, yi, zi);
				Real F2 = Real(0);
				for (int d = 0; d < NDIM; d++) {
					F2 += sqr(U[fx_i + d][iii]);
				}
				Real const E = U[er_i][iii];
				Real const F = sqrt(F2);
				if (F > c * E) {
					printf("level = %i: %i %e %i %e %i %e F = %e E = %e f = %e\n", level, xi, X[0][xi], yi, X[1][yi], zi, X[2][zi], F, c * E, F / (c * E));
					flag = true;
				}
			}
		}
	}
	if (flag) {
		printf("error at %s : %i\n", filename, line);
		fflush(stdout);
		abort();
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
	for (integer d = 0; d != NDIM; ++d) {
		for (integer f = 0; f != NRF; ++f) {
			for (integer xi = RAD_BW; xi != RAD_NX - RAD_BW; ++xi) {
				for (integer yi = RAD_BW; yi != RAD_NX - RAD_BW; ++yi) {
					for (integer zi = RAD_BW; zi != RAD_NX - RAD_BW; ++zi) {
						const integer iii = rindex(xi, yi, zi);
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
	return;
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
