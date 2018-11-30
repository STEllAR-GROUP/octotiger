#include <hpx/include/run_as.hpp>

extern "C" {
#include <quadmath.h>

#include <stdlib.h>
#include <math.h>


typedef __float128 sfloat;
void sed_1d_(const sfloat* t, const int* N, const sfloat* xpos, const sfloat* eblast, const sfloat* omega, const sfloat* geom, const sfloat* rho0, const sfloat* vel0, const sfloat* ener0, const sfloat* pres0, const sfloat* cs0, const sfloat* gam0, sfloat* den, sfloat* ener, sfloat* pres, sfloat* vel,sfloat *cs);


void sedov_solution( double t, int N, const double* xpos, double E, double rho, double* dout, double* eout, double* vout ) {
	sfloat *xpos0, *pout, *csout, *dout0, *eout0, *vout0;
	sfloat vel0, gam0, pres0, cs0, omega, geom, ener0, t0, E0, rho0;
	int i;
	t0 = (sfloat) t;
	E0 = (sfloat) E;
	rho0 = (sfloat) rho;
	eout0 = (sfloat*) malloc( sizeof( sfloat ) * N );
	dout0 = (sfloat*) malloc( sizeof( sfloat ) * N );
	vout0 = (sfloat*) malloc( sizeof( sfloat ) * N );
	xpos0 = (sfloat*) malloc( sizeof( sfloat ) * N );
	pout = (sfloat*) malloc( sizeof( sfloat ) * N );
	csout = (sfloat*) malloc(sizeof( sfloat ) * N );
			vel0 = 0.0;
			omega = 0.0;
			geom = 3.0;
			ener0 = 0.0;
			gam0 = 5.0/3.0;
			pres0 = (gam0 - 1.0)*rho0*ener0;
			cs0 = sqrtq( gam0 * pres0 / rho0 );
			for( i = 0; i < N; i++ ) {
				xpos0[i] = (sfloat) xpos[i];
			}
			sed_1d_( &t0, &N, xpos0, &E0, &omega, &geom, &rho0, &vel0, &ener0, &pres0, &cs0, &gam0, dout0, eout0, pout, vout0, csout );
			free( xpos0 );
			free( pout );
			free( csout );
			for( i = 0; i < N; i++ ) {
				dout[i] = (double) dout0[i];
				eout[i] = (double) eout0[i];
				vout[i] = (double) vout0[i];
			}
			free( dout0 );
			free( vout0 );
			free( eout0 );
		}
	}

#include <vector>
#include <stdio.h>

#ifndef TESTME
#include "defs.hpp"
#include "grid.hpp"
#endif



class sedov_analytic {
	int N;
	static constexpr int bw = 2;

	double rmax, dr;
	std::vector<double> eout;
	std::vector<double> dout;
	std::vector<double> vout;
public:
	std::vector<double> state_at(double x, double y, double z) {
#ifndef TESTME
		std::vector<double> u(opts().n_fields,0.0);
#endif
		int i[4];
		const double r = std::sqrt(x * x + y * y + z * z);
		i[1] = (r + bw * dr) / dr;
		i[0] = i[1] - 1;
		i[2] = i[1] + 1;
		i[3] = i[1] + 2;
		double r0 = (r - (i[1]-bw)*dr)/dr;
	//	printf( "%e %i\n", r0,  i[1]);
		const auto interp = [&r0,&i](const std::vector<double>& data) {
			double sum = 0.0;
			sum += (-0.5 * data[i[0]] + 1.5 * data[i[1]] - 1.5 * data[i[2]] + 0.5 * data[i[3]]) * r0 * r0 * r0;
			sum += (+1.0 * data[i[0]] - 2.5 * data[i[1]] + 2.0 * data[i[2]] - 0.5 * data[i[3]]) * r0 * r0;
			sum += (-0.5 * data[i[0]]                   +  0.5 * data[i[2]]) * r0;
			sum += data[i[1]];
			return sum;
		};
		double d = std::max(interp(dout),1.0e-20);
		double v = interp(vout);
		double e = std::max(interp(eout), 1.0e-20);
#ifdef TESTME
		printf( "%e %e %e %e\n", r, d, v, e);
		return std::vector<double>();
#else
		u[rho_i] = d;
		u[spc_i] = d;
		u[sx_i] = v * d * x / r;
		u[sy_i] = v * d * y / r;
		u[sz_i] = v * d * z / r;
		u[egas_i] = d * v * v * 0.5 + e * d;
		u[tau_i] = std::pow(e * d, 3.0 / 5.0);
		u[zx_i] = 0.0;
		u[zy_i] = 0.0;
		u[zz_i] = 0.0;
//		printf( "%e\n", std::pow(e * d, 3.0 / 5.0));
		return std::move(u);
#endif
	}
	sedov_analytic(double t) {
		N = 1000 + bw;
		rmax = 2.0;
		dr = rmax / (N-bw);
		std::vector<double> xpos(N);
		for (int i = 0; i < N; i++) {
			xpos[i] = (i + 0.5 - bw) * dr;
		}
		eout.resize(N);
		vout.resize(N);
		dout.resize(N);
		hpx::threads::run_as_os_thread([&]() {
			sedov_solution(t, N - bw, xpos.data() + bw, 1.0, 1.0, dout.data() + bw, eout.data() + bw, vout.data() + bw);
		}).get();
		dout[0] = dout[3];
		dout[1] = dout[2];
		eout[0] = eout[3];
		eout[1] = eout[2];
		vout[0] = -vout[3];
		vout[1] = -vout[2];
#ifdef TESTME
		for( double r = 0; r < 1.0; r += 1.0e-3 ) {
			state_at(r,0.0,0.0);
		}
#endif	
	}
};

constexpr int sedov_analytic::bw;

#ifdef TESTME
int main() {
	sedov_analytic test(1.0e-12);
}
#endif



std::vector<real> blast_wave(real x, real y, real z, real dx) {
	const real fgamma = grid::get_fgamma();
	std::vector<real> u(opts().n_fields, real(0));
	u[spc_i] = u[rho_i] = 1.0e-3;
	const real a = std::sqrt(10.0) * std::min(dx, 0.1);
	real r = std::sqrt(x * x + y * y + z * z);
	u[egas_i] = std::max(1.0e-10, exp(-r * r / a / a)) / 100.0;
	u[tau_i] = std::pow(u[egas_i], ONE / fgamma);
	return u;
}


#ifndef TESTME
//std::vector<double> blast_wave(double x, double y, double z, double dx) {
//	static sedov_analytic state(1.0e-3);
//	return state.state_at(x,y,z);
//}
#endif







