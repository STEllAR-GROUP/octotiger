#include <string>
#include <array>
#include <numeric>
#include <cmath>
#include <assert.h>
#include <valarray>
#include <functional>
#include "../space_vector.hpp"

#include "../physcon.hpp"

#include <functional>
#define _3DIM 3
using function_type = std::function<real(real)>;

using mat_function_type = std::function<real(real, real, real)>;
#include "/home/dmarce1/local/vc/include/Vc/Vc"

template<class T, int N>
class vectorN: public std::valarray<T> {
public:
	template<class V>
	vectorN(const V& arg) :
			std::valarray<T>(arg) {
	}
	vectorN() :
			std::valarray<T>(N) {
	}
	T dot(const vectorN& other) const {
		return ((*this) * other).sum();
	}
};

inline real dot( const space_vector& a, space_vector& b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void implicit_radiation_step(real& E0, real& e0, space_vector& F0, space_vector& u0, real rho, real mmw, real kp, real kr,
		real dt) {

	const real c = physcon.c;
	const real cinv = 1.0 / c;
	const real c2 = c * c;

	space_vector F1, u1;
	real e1;



	const auto f = [&](real this_E) {
		F1 = (3*c*F0*rho + 4*this_E*dt*kr*(F0 + c2*rho*u0))/(4*this_E*dt*kr + 3*c*(1 + c*dt*kr)*rho);
		u1 = u0 + (F0 - F1)/(rho*c2);
		e1 = std::max(e0 - ((this_E - E0) + 0.5 * rho * (dot(u1,u1) - dot(u0,u0))),0.0);
		constexpr real gm1 = 2.0 / 3.0;
		const real B = B_p(rho,e1,mmw);
		const real f = this_E - E0 + dt * (c * kp * (this_E - B) - (2 * kp - kr) * dot(u1,F1) * cinv);
		return f;
	};

	const real ke = 0.5 * rho * dot(u0,u0);
	real minE = 0.0;
	real maxE = E0 + e0 + ke;
	real error;
	do {
		real fmax, fmid;
		const real midE = 0.5 * (maxE + minE);
		fmax = f(maxE);
		fmid = f(midE);
		const real score = fmax * fmid;
		if (score < 0.0) {
			minE = midE;
			error = (maxE - minE) / midE;
		} else if (score > 0.0) {
			maxE = midE;
			error = (maxE - minE) / midE;
		} else {
			error = 0.0;
		}
	} while (error > 1.0e-6);

	e0 = e1;
	E0 = 0.5 * (maxE + minE);
	F0 = F1;
	u0 = u1;
}

void implicit_radiation_step_2nd_order(real& E0, real& e0, space_vector& F0, space_vector& u0, real rho, real mmw, real kp,
		real kr, real dt) {

	const auto step = [rho, mmw, kp, kr](real E, real e, space_vector F, space_vector u, real dt ) {
		real E1 = E;
		space_vector F1 = F;
		implicit_radiation_step(E1, e, F1, u, rho, mmw, kp, kr, dt);
		return std::make_pair<real,space_vector>((E1 - E)/dt, (F1-F)/dt);

	};

	constexpr int N = 2;
	const real gam = 1.0 + std::sqrt(2) / 2.0;
	const real A[N][N] = { { gam, 0 }, { 1 - gam, gam } };
	const real B[N] = { 1 - gam, gam };
	real kdE[N];
	space_vector kdF[N];
	const real c2 = physcon.c * physcon.c;
	real E, e;
	space_vector F, u;
	for (int i = 0; i < N; i++) {
		E = E0;
		F = F0;
		u = u0;
		e = e0;
		for (int j = 0; j < i; j++) {
			const real dE = (A[i][j] * kdE[j]) * dt;
			const space_vector dF = (A[i][j] * kdF[j]) * dt;
			E += dE;
			F += dF;
		}
		u -= (F - F0) / (rho * c2);
		e -= (E - E0) + 0.5 * rho * (dot(u,u) - dot(u0,u0));
		const auto tmp = step(E, e, F, u, A[i][i] * dt);
		kdE[i] = tmp.first;
		kdF[i] = tmp.second;
	}
	E = E0;
	F = F0;
	u = u0;
	for (int i = 0; i < N; i++) {
		E0 += B[i] * kdE[i] * dt;
		F0 += B[i] * kdF[i] * dt;
	}
	u0 -= (F0 - F) / (rho * c2);
	e0 -= (E0 - E) + 0.5 * rho * (dot(u0,u0) - dot(u,u));
}

void implicit_radiation_step_2nd_order(real& E0, real& e0, space_vector& F0, space_vector& u0, real rho, real mmw, real dt) {

	// Take a single backward Euler step to get opacities at end of timestep to average with start of timestep
	real E1 = E0;
	real e1 = e0;
	space_vector F1 = F0;
	space_vector u1 = u0;
	real kp = kappa_p(rho, e0, mmw);
	real kr = kappa_R(rho, e0, mmw);
	implicit_radiation_step(E1, e1, F1, u1, rho, mmw, kp, kr, dt);
	kp += 0.5 * (kappa_p(rho, e1, mmw) - kp);
	kr += 0.5 * (kappa_R(rho, e1, mmw) - kr);

	//Now take 2nd order step

	implicit_radiation_step(E0, e0, F0, u0, rho, mmw, kp, kr, dt);

}

