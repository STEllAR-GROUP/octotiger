
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

	static constexpr double coeff[4][4] = { { -0.06250000000, 0.5625000000, -0.06250000000, 0.5625000000 }, { 0.04166666667, -1.125000000,
			-0.04166666667, 1.125000000 }, { 0.2500000000, -0.2500000000, 0.2500000000, -0.2500000000 }, { -0.1666666667,
			0.5000000000, 0.1666666667, -0.5000000000 } };
	double rmax, dr;
	std::vector<double> eout;
	std::vector<double> dout;
	std::vector<double> vout;
public:
	std::vector<double> state_at(double x, double y, double z) {
		std::vector<double> u(0.0);
		int i[4];
		const double r = std::sqrt(x * x + y * y + z * z);
		i[1] = (r + 2 * dr) / dr;
		i[0] = i[1] - 1;
		i[2] = i[1] + 1;
		i[3] = i[1] + 2;
		double r0 = r - (i[1]+2)*dr;
		const auto interp = [&coeff,&r0](const std::vector<double>& data) {
			double rn = 1.0;
			double sum = 0.0;
			for( int n = 0; n < 4; n++) {
				for( int m = 0; m < 4; m++) {
					sum += rn * coeff[n][m] * data[m];
				}
				rn *= r0;
			}
			return sum;
		};
		double d = interp(dout);
		double v = interp(vout);
		double e = interp(eout);
#ifdef TESTME
		printf( "%e %e %e %e\n", r, d, v, e);
#else
		u[rho_i] = d;
		u[spc_i] = d;
		u[sx_i] = v * d * x / r;
		u[sy_i] = v * d * y / r;
		u[sz_i] = v * d * z / r;
		u[egas_i] = d * v * v * 0.5 + e * d;
		u[tau_i] = std::pow(e * d, 1.0 / grid::get_fgamma());
		u[zx_i] = 0.0;
		u[zy_i] = 0.0;
		u[zz_i] = 0.0;
#endif
		return std::move(u);
	}
	sedov_analytic(double t) {
		N = 100 + bw;
		rmax = 2.0;
		dr = rmax / (N-bw);
		std::vector<double> xpos(N);
		for (int i = 0; i < N; i++) {
			xpos[i] = (i + 0.5 - bw) * dr;
		}
		eout.resize(N);
		vout.resize(N);
		dout.resize(N);
		sedov_solution(t, N - bw, xpos.data() + bw, 1.0, 1.0, dout.data() + bw, eout.data() + bw, vout.data() + bw);
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
constexpr double sedov_analytic::coeff[4][4];

#ifdef TESTME
int main() {
	sedov_analytic test(1.0);
}
#endif
