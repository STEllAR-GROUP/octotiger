#include "octotiger/radiation/implicit.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/radiation/opacities.hpp"
#include "octotiger/space_vector.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <numeric>
#include <string>


std::pair<real, space_vector> implicit_radiation_step(real E0, real& e0, space_vector F0,
		space_vector u0, real rho, real mmw, real X, real Z, real dt) {

	const real c = physcon().c;
	real kp = kappa_p(rho, e0, mmw, X, Z);
	real kr = kappa_R(rho, e0, mmw, X, Z);
	const real rhoc2 = rho * c * c;

	E0 /= rhoc2;
	F0 = F0 / (rhoc2 * c);
	e0 /= rhoc2;
	u0 = u0 / c;
	kp *= dt * c;
	kr *= dt * c;

	const auto B = [rho,mmw,c,rhoc2](real e) {
		return (4.0 * M_PI / c) * B_p(rho, e*rhoc2, mmw) / rhoc2;
	};

	auto E = E0;
	auto eg_t = e0 + 0.5 * (u0[0]*u0[0]+u0[1]*u0[1]+u0[2]*u0[2]);
	auto F = F0;
	auto u = u0;
	real ei;
	const auto eg_t0 = eg_t;

	real u2_0 = 0.0;
	real F2_0 = 0.0;
	for (int d = 0; d < NDIM; d++) {
		u2_0 += u[d] * u[d];
		F2_0 += F[d] * F[d];
	}
//	printf( "%e %e\n", (double) u2_0, (double) (F2_0/E/E));
	const auto test = [&](real de) {
		E = E0 + de;
		real u2 = 0.0;
		real udotF = 0.0;
		for( int d = 0; d < NDIM; d++) {
			const auto num = F0[d] + (4.0/3.0)*kr*E*(u0[d]+F0[d]);
			const auto den = 1.0 + kr*(1.0+(4.0/3.0)*E);
			const auto deninv = 1.0 / den;
			F[d] = num * deninv;
			u[d] = u0[d] + F0[d] - F[d];
			u2 += u[d] * u[d];
			udotF += F[d] * u[d];
		}
		ei = std::max(eg_t0 - E + E0 - 0.5*u2,real(0.0));
		const real b = B(ei);
		real f = E - E0;
		f += (kp * (E - b) + (kr - 2.0 * kp) * udotF);
		eg_t = eg_t0 + E0 - E;
		return f;
	};

	real de_max, de_min, de_mid;

	de_max = +eg_t0;
	de_min = -E0;
	int N = 50;
	real f_min = 0, f_mid = 0, error = 0;
	de_mid = 0.0;
	error = std::abs(test(0.0)) * INVERSE (E + eg_t);
		const real toler = 1.0e-9;
		int i;
	for (i = 0; i < N; i++) {
		f_min = test(de_min);
		f_mid = test(de_mid);

		if (f_min * f_mid < 0.0) {
			de_max = de_mid;
		} else if (f_min * f_mid > 0.0) {
			de_min = de_mid;
		} else {
			break;
		}
		error = std::max(std::abs(f_mid), std::abs(f_min)) / (E + eg_t);
		if( error < toler ) {
			break;
		}
		de_mid = 0.5 * (de_min + de_max);
	}
	if( i == 50 ) {
		printf( "Implicit radiation solver failed to converge\n");
		abort();
	}
	ei = eg_t - 0.5 * (u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
	e0 = ei * rhoc2;
	const auto dtinv = 1.0 / dt;

	return std::make_pair(real((E - E0) * dtinv * rhoc2), ((F - F0) * dtinv * rhoc2 * c));
}

