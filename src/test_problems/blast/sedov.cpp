extern "C" {

#include <stdlib.h>
#include <math.h>




typedef long double sfloat;
void sed_1d_(const sfloat* t, const int* N, const sfloat* xpos, const sfloat* eblast, const sfloat* omega,  const sfloat*  geom, const sfloat* rho0, const sfloat* vel0, const sfloat* ener0, const sfloat* pres0, const sfloat* cs0, const sfloat* gam0, sfloat* den, sfloat* ener, sfloat* pres, sfloat* vel,sfloat *cs);




void sedov_solution( double t, int N, const double* xpos, double E, double rho,  double* dout, double* eout, double* vout ) {
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
	csout = (sfloat*) malloc( sizeof( sfloat ) * N );
	vel0 = 0.0;
	omega = 0.0;
	geom = 3.0;
	ener0 = 0.0;
	gam0 = 5.0/3.0;
	pres0 = (gam0 - 1.0)*rho0*ener0;
	cs0 = sqrt( gam0 * pres0 / rho0 );
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
