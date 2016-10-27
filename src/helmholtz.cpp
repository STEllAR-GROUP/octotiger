#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "helmholtz.hpp"
#include "const.h"

#ifdef HELMHOLTZ

#define IMAX 1081
#define JMAX 401

#define HELMHOLTZ_TMIN (1.0e+3)
#define HELMHOLTZ_TMAX (1.0e+13)

static double cm_to_code = 1.0, g_to_code = 1.0, s_to_code = 1.0, K_to_code = 1.0;
static double erg_to_code = 1.0, dyne_to_code = 1.0;

double tlo, thi, tstp, tstpi, dlo, dhi, dstp, dstpi;
double d[IMAX], t[JMAX];
double f[IMAX][JMAX], fd[IMAX][JMAX], ft[IMAX][JMAX], fdd[IMAX][JMAX], ftt[IMAX][JMAX], fdt[IMAX][JMAX],
		fddt[IMAX][JMAX], fdtt[IMAX][JMAX], fddtt[IMAX][JMAX];
double dpdf[IMAX][JMAX], dpdfd[IMAX][JMAX], dpdft[IMAX][JMAX], dpdfdt[IMAX][JMAX];
double ef[IMAX][JMAX], efd[IMAX][JMAX], eft[IMAX][JMAX], efdt[IMAX][JMAX];
double xf[IMAX][JMAX], xfd[IMAX][JMAX], xft[IMAX][JMAX], xfdt[IMAX][JMAX];
double dt_sav[JMAX], dt2_sav[JMAX], dti_sav[JMAX], dt2i_sav[JMAX], dt3i_sav[JMAX], dd_sav[IMAX], dd2_sav[IMAX],
		ddi_sav[IMAX], dd2i_sav[IMAX], dd3i_sav[IMAX];
double tion_lo, tion_hi, tion_stp, tion_stpi, dion_lo, dion_hi, dion_stp, dion_stpi;
double dion[IMAX], tion[JMAX];
double fion[IMAX][JMAX], fiond[IMAX][JMAX], fiont[IMAX][JMAX], fiondd[IMAX][JMAX], fiontt[IMAX][JMAX],
		fiondt[IMAX][JMAX], fionddt[IMAX][JMAX], fiondtt[IMAX][JMAX], fionddtt[IMAX][JMAX];
double dpiondf[IMAX][JMAX], dpiondfd[IMAX][JMAX], dpiondft[IMAX][JMAX], dpiondfdt[IMAX][JMAX];
double efion[IMAX][JMAX], efiond[IMAX][JMAX], efiont[IMAX][JMAX], efiondt[IMAX][JMAX];
double xfion[IMAX][JMAX], xfiond[IMAX][JMAX], xfiont[IMAX][JMAX], xfiondt[IMAX][JMAX];
double dt_sav_ion[JMAX], dt2_sav_ion[JMAX], dti_sav_ion[JMAX], dt2i_sav_ion[JMAX], dt3i_sav_ion[JMAX], dd_sav_ion[IMAX],
		dd2_sav_ion[IMAX], ddi_sav_ion[IMAX], dd2i_sav_ion[IMAX], dd3i_sav_ion[JMAX];

#define psi0(z)   ((z)*(z)*(z) * ( (z) * (-6.0*(z) + 15.0) -10.0) + 1.0)
#define dpsi0(z)  ((z)*(z) * ( (z) * (-30.0*(z) + 60.0) - 30.0))
#define ddpsi0(z) ((z)* ( (z)*( -120.0*(z) + 180.0) -60.0))
#define psi1(z)   ((z)* ( (z)*(z) * ( (z) * (-3.0*(z) + 8.0) - 6.0) + 1.0))
#define dpsi1(z)  ((z)*(z) * ( (z) * (-15.0*(z) + 32.0) - 18.0) +1.0)
#define ddpsi1(z) ((z) * ((z) * (-60.0*(z) + 96.0) -36.0))
#define psi2(z)   (0.5*(z)*(z)*( (z)* ( (z) * (-(z) + 3.0) - 3.0) + 1.0))
#define dpsi2(z)  (0.5*(z)*( (z)*((z)*(-5.0*(z) + 12.0) - 9.0) + 2.0))
#define ddpsi2(z) (0.5*((z)*( (z) * (-20.0*(z) + 36.0) - 18.0) + 2.0))
#define h5(i,j,w0t,w1t,w2t,w0mt,w1mt,w2mt,w0d,w1d,w2d,w0md,w1md,w2md) \
             fi[0]  *w0d*w0t   + fi[1]  *w0md*w0t \
           + fi[2]  *w0d*w0mt  + fi[3]  *w0md*w0mt \
           + fi[4]  *w0d*w1t   + fi[5]  *w0md*w1t \
           + fi[6]  *w0d*w1mt  + fi[7]  *w0md*w1mt \
           + fi[8]  *w0d*w2t   + fi[9] *w0md*w2t \
           + fi[10] *w0d*w2mt  + fi[11] *w0md*w2mt \
           + fi[12] *w1d*w0t   + fi[13] *w1md*w0t \
           + fi[14] *w1d*w0mt  + fi[15] *w1md*w0mt \
           + fi[16] *w2d*w0t   + fi[17] *w2md*w0t \
           + fi[18] *w2d*w0mt  + fi[19] *w2md*w0mt \
           + fi[20] *w1d*w1t   + fi[21] *w1md*w1t \
           + fi[22] *w1d*w1mt  + fi[23] *w1md*w1mt \
           + fi[24] *w2d*w1t   + fi[25] *w2md*w1t \
           + fi[26] *w2d*w1mt  + fi[27] *w2md*w1mt \
           + fi[28] *w1d*w2t   + fi[29] *w1md*w2t \
           + fi[30] *w1d*w2mt  + fi[31] *w1md*w2mt \
           + fi[32] *w2d*w2t   + fi[33] *w2md*w2t \
           + fi[34] *w2d*w2mt  + fi[35] *w2md*w2mt
#define xpsi0(z)  ((z) * (z) * (2.0*(z) - 3.0) + 1.0)
#define xdpsi0(z) ((z) * (6.0*(z) - 6.0))
#define xpsi1(z)  ((z) * ( (z) * ((z) - 2.0) + 1.0))
#define xdpsi1(z) ((z) * (3.0*(z) - 4.0) + 1.0)
#define h3(i,j,w0t,w1t,w0mt,w1mt,w0d,w1d,w0md,w1md)  \
             fi[0]  *w0d*w0t   +  fi[1]  *w0md*w0t \
           + fi[2]  *w0d*w0mt  +  fi[3]  *w0md*w0mt \
           + fi[4]  *w0d*w1t   +  fi[5]  *w0md*w1t \
           + fi[6]  *w0d*w1mt  +  fi[7]  *w0md*w1mt \
           + fi[8]  *w1d*w0t   +  fi[9] *w1md*w0t \
           + fi[10] *w1d*w0mt  +  fi[11] *w1md*w0mt \
           + fi[12] *w1d*w1t   +  fi[13] *w1md*w1t \
           + fi[14] *w1d*w1mt  +  fi[15] *w1md*w1mt

static double max(double a, double b) {
	return a > b ? a : b;
}

static double min(double a, double b) {
	return a < b ? a : b;
}

__attribute__((constructor)) void read_helm_table() {
	int i, j;
	double tsav, dsav, dth, dt2, dti, dt2i, dt3i, dd, dd2, ddi, dd2i, dd3i;
	FILE* fp;
	fp = fopen("helmholtz.table.dat", "rt");

	tlo = 3.0;
	thi = 13.0;
	tstp = (thi - tlo) / double(JMAX - 1);
	tstpi = 1.0 / tstp;
	dlo = -12.0;
	dhi = 15.0;
	dstp = (dhi - dlo) / double(IMAX - 1);
	dstpi = 1.0 / dstp;

// read the helmholtz free energy and its derivatives
	for (j = 0; j < JMAX; j++) {
		tsav = tlo + j * tstp;
		t[j] = pow(10.0, tsav);
		for (i = 0; i < IMAX; i++) {
			dsav = dlo + i * dstp;
			d[i] = pow(10.0, dsav);
			fscanf(fp, "%le %le %le %le %le %le %le %le %le\n", f[i] + j, fd[i] + j, ft[i] + j, fdd[i] + j, ftt[i] + j,
					fdt[i] + j, fddt[i] + j, fdtt[i] + j, fddtt[i] + j);
			
	//		fddtt[i][j] = fddt[i][j] = fdtt[i][j] = 0.0;
		}
	}

	printf( "Read helm table\n");
	fclose(fp);

// construct the temperature and density deltas and their inverses
	for (j = 0; j < JMAX - 1; j++) {
		dth = t[j + 1] - t[j];
		dt2 = dth * dth;
		dti = 1.0 / dth;
		dt2i = 1.0 / dt2;
		dt3i = dt2i * dti;
		dt_sav[j] = dth;
		dt2_sav[j] = dt2;
		dti_sav[j] = dti;
		dt2i_sav[j] = dt2i;
		dt3i_sav[j] = dt3i;
	}
	for (i = 0; i < IMAX - 1; i++) {
		dd = d[i + 1] - d[i];
		dd2 = dd * dd;
		ddi = 1.0 / dd;
		dd2i = 1.0 / dd2;
		dd3i = dd2i * ddi;
		dd_sav[i] = dd;
		dd2_sav[i] = dd2;
		ddi_sav[i] = ddi;
		dd2i_sav[i] = dd2i;
		dd3i_sav[i] = dd3i;
	}
}

void helmholtz_set_cgs_units(double cm, double g, double s, double K) {
//	printf("Setting cgs units cm = %e g = %e s = %e K = %e\n", cm, g, s, K);
	cm_to_code = cm;
	g_to_code = g;
	s_to_code = s;
	K_to_code = K;
	erg_to_code = g * cm * cm / s / s;
	dyne_to_code = g * cm / s / s;
}

static void eos_to_code(eos_t* eos) {
	eos->p /= dyne_to_code / pow(cm_to_code, 2);
	eos->e /= erg_to_code / g_to_code;
	eos->T /= K_to_code;
	eos->cs /= cm_to_code / s_to_code;
	eos->cv /= erg_to_code / g_to_code / K_to_code;
	eos->rho /= g_to_code / pow(cm_to_code, 3);
}

static void eos_from_code(eos_t* eos) {
	eos->p *= dyne_to_code / pow(cm_to_code, 2);
	eos->e *= erg_to_code / g_to_code;
	eos->T *= K_to_code;
	eos->cs *= cm_to_code / s_to_code;
	eos->cv *= erg_to_code / g_to_code / K_to_code;
	eos->rho *= g_to_code / pow(cm_to_code, 3);
}
void helmholtz_eos(eos_t* eos) {

	int iat, jat;
	double den, temp, abar, zbar, ytot1, ye, x, deni, xni, dxnidd, dpepdt, dpepdd, deepdt, dsepdt, dpiondd, dpiondt,
			deiondt, kt, pion, eion, pele, eele, sele, dpgasdd, dpgasdt, free, df_d, df_t, df_tt, df_dt, df_dd, xt, xd, mxt,
			mxd, si0t, si1t, si2t, si0mt, si1mt, si2mt, si0d, si1d, si2d, si0md, si1md, si2md, dsi0t, dsi1t, dsi2t,
			dsi0mt, dsi1mt, dsi2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md, ddsi0t, ddsi1t, ddsi2t, ddsi0mt,
			ddsi1mt, ddsi2mt, din, fi[36], ddsi0d , ddsi1d, ddsi2d, ddsi0md, ddsi1md, ddsi2md;


	eos_from_code(eos);

	den = eos->rho;
	temp = max(eos->T, HELMHOLTZ_TMIN);
	abar = eos->abar;
	zbar = eos->zbar;

	ytot1 = 1.0 / abar;
	ye = max(1.0d - 16, ytot1 * zbar);

// initialize
	deni = 1.0 / den;
	kt = kerg * temp;

// ion section:
	xni = avo * ytot1 * den;
	dxnidd = avo * ytot1;

	pion = xni * kt;
	dpiondd = dxnidd * kt;
	dpiondt = xni * kerg;

	eion = 1.5 * pion * deni;
	deiondt = 1.5 * dpiondt * deni;

// sackur-tetrode equation for the ion entropy of
// a single ideal gas characterized by abar

// electron-positron section:

// enter the table with ye*den
	din = ye * den;

// bomb proof the input
	if ((temp > t[JMAX - 1]) || (temp < t[0]) || (din > d[IMAX - 1]) || (din < d[0])) {
		printf("Out of range temp = %e den = %e\n", temp, den);
		abort();
	}

// hash locate this temperature and density
	jat = int((log10(temp) - tlo) * tstpi) + 1;
	jat = max(1, min(jat, JMAX - 1));
	iat = int((log10(din) - dlo) * dstpi) + 1;
	iat = max(1, min(iat, IMAX - 1));
	--jat;
	--iat;

// access the table locations only once
	fi[0] = f[iat][jat];
	fi[1] = f[iat + 1][jat];
	fi[2] = f[iat][jat + 1];
	fi[3] = f[iat + 1][jat + 1];
	fi[4] = ft[iat][jat];
	fi[5] = ft[iat + 1][jat];
	fi[6] = ft[iat][jat + 1];
	fi[7] = ft[iat + 1][jat + 1];
	fi[8] = ftt[iat][jat];
	fi[9] = ftt[iat + 1][jat];
	fi[10] = ftt[iat][jat + 1];
	fi[11] = ftt[iat + 1][jat + 1];
	fi[12] = fd[iat][jat];
	fi[13] = fd[iat + 1][jat];
	fi[14] = fd[iat][jat + 1];
	fi[15] = fd[iat + 1][jat + 1];
	fi[16] = fdd[iat][jat];
	fi[17] = fdd[iat + 1][jat];
	fi[18] = fdd[iat][jat + 1];
	fi[19] = fdd[iat + 1][jat + 1];
	fi[20] = fdt[iat][jat];
	fi[21] = fdt[iat + 1][jat];
	fi[22] = fdt[iat][jat + 1];
	fi[23] = fdt[iat + 1][jat + 1];
	fi[24] = fddt[iat][jat];
	fi[25] = fddt[iat + 1][jat];
	fi[26] = fddt[iat][jat + 1];
	fi[27] = fddt[iat + 1][jat + 1];
	fi[28] = fdtt[iat][jat];
	fi[29] = fdtt[iat + 1][jat];
	fi[30] = fdtt[iat][jat + 1];
	fi[31] = fdtt[iat + 1][jat + 1];
	fi[32] = fddtt[iat][jat];
	fi[33] = fddtt[iat + 1][jat];
	fi[34] = fddtt[iat][jat + 1];
	fi[35] = fddtt[iat + 1][jat + 1];

// various differences
	xt = max((temp - t[jat]) * dti_sav[jat], 0.0);
	xd = max((din - d[iat]) * ddi_sav[iat], 0.0);
	mxt = 1.0 - xt;
	mxd = 1.0 - xd;

// the six density and six temperature basis functions
	si0t = psi0(xt);
	si1t = psi1(xt) * dt_sav[jat];
	si2t = psi2(xt) * dt2_sav[jat];

	si0mt = psi0(mxt);
	si1mt = -psi1(mxt) * dt_sav[jat];
	si2mt = psi2(mxt) * dt2_sav[jat];

	si0d = psi0(xd);
	si1d = psi1(xd) * dd_sav[iat];
	si2d = psi2(xd) * dd2_sav[iat];

	si0md = psi0(mxd);
	si1md = -psi1(mxd) * dd_sav[iat];
	si2md = psi2(mxd) * dd2_sav[iat];

// derivatives of the weight functions
	dsi0t = dpsi0(xt) * dti_sav[jat];
	dsi1t = dpsi1(xt);
	dsi2t = dpsi2(xt) * dt_sav[jat];

	dsi0mt = -dpsi0(mxt) * dti_sav[jat];
	dsi1mt = dpsi1(mxt);
	dsi2mt = -dpsi2(mxt) * dt_sav[jat];

	dsi0d = dpsi0(xd) * ddi_sav[iat];
	dsi1d = dpsi1(xd);
	dsi2d = dpsi2(xd) * dd_sav[iat];

	dsi0md = -dpsi0(mxd) * ddi_sav[iat];
	dsi1md = dpsi1(mxd);
	dsi2md = -dpsi2(mxd) * dd_sav[iat];

// second derivatives of the weight functions
	ddsi0t = ddpsi0(xt) * dt2i_sav[jat];
	ddsi1t = ddpsi1(xt) * dti_sav[jat];
	ddsi2t = ddpsi2(xt);

	ddsi0mt = ddpsi0(mxt) * dt2i_sav[jat];
	ddsi1mt = -ddpsi1(mxt) * dti_sav[jat];
	ddsi2mt = ddpsi2(mxt);

        ddsi0d =   ddpsi0(xd)*dd2i_sav[iat];
        ddsi1d =   ddpsi1(xd)*ddi_sav[iat];
        ddsi2d =   ddpsi2(xd);

        ddsi0md =  ddpsi0(mxd)*dd2i_sav[iat];
        ddsi1md = -ddpsi1(mxd)*ddi_sav[iat];
        ddsi2md =  ddpsi2(mxd);


// the free energy
	free = h5(iat,jat,
			si0t, si1t, si2t, si0mt, si1mt, si2mt,
			si0d, si1d, si2d, si0md, si1md, si2md);

// derivative with respect to density
	df_d = h5(iat,jat,
			si0t, si1t, si2t, si0mt, si1mt, si2mt,
			dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md);

// derivative with respect to temperature
	df_t = h5(iat,jat,
			dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt,
			si0d, si1d, si2d, si0md, si1md, si2md);

// derivative with respect to density**2
        df_dd = h5(iat,jat,
               si0t,   si1t,   si2t,   si0mt,   si1mt,   si2mt,
               ddsi0d, ddsi1d, ddsi2d, ddsi0md, ddsi1md, ddsi2md);


// derivative with respect to temperature**2
	df_tt = h5(iat,jat,
			ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt,
			si0d, si1d, si2d, si0md, si1md, si2md);

// derivative with respect to temperature and density
	df_dt = h5(iat,jat,
			dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt,
			dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md);

// now get the pressure derivative with density, chemical potential, and
// electron positron number densities
// get the interpolation weight functions
	si0t = xpsi0(xt);
	si1t = xpsi1(xt) * dt_sav[jat];

	si0mt = xpsi0(mxt);
	si1mt = -xpsi1(mxt) * dt_sav[jat];

	si0d = xpsi0(xd);
	si1d = xpsi1(xd) * dd_sav[iat];

	si0md = xpsi0(mxd);
	si1md = -xpsi1(mxd) * dd_sav[iat];

// derivatives of weight functions
	dsi0t = xdpsi0(xt) * dti_sav[jat];
	dsi1t = xdpsi1(xt);

	dsi0mt = -xdpsi0(mxt) * dti_sav[jat];
	dsi1mt = xdpsi1(mxt);

	dsi0d = xdpsi0(xd) * ddi_sav[iat];
	dsi1d = xdpsi1(xd);

	dsi0md = -xdpsi0(mxd) * ddi_sav[iat];
	dsi1md = xdpsi1(mxd);

// the desired electron-positron thermodynamic quantities

	x = din * din;
	pele = x * df_d;
        dpepdd  = ye * (x * df_dd + 2.0 * din * df_d);
	dpepdt = x * df_dt;

	x = ye * ye;
	sele = -df_t * ye;
	dsepdt = -df_tt * ye;

	eele = ye * free + temp * sele;
	deepdt = temp * dsepdt;

	eos->p = pele + pion;
	eos->e = eele + eion;
	eos->cv = deiondt + deepdt;
	dpgasdd = dpiondd + dpepdd;
	dpgasdt = dpiondt + dpepdt;
	x = (eos->p * deni * deni) * (dpgasdt / eos->cv) + dpgasdd;
	if( eos->cv < 0.0 ) {
	//	printf( "%e %e \n", eos->rho, eos->T );
	//	abort();
	}
	if (x < 0.0 ) {
	//	printf("cs is negative rho=%e T=%e p=%e dpgasdd=%e dpgasdt=%e cv=%e E=%e\n", eos->rho, eos->T, eos->p, dpgasdd, dpgasdt, eos->cv, eos->e );
//		abort();
		eos->cs = 0.0;
	} else {
		eos->cs = sqrt(x);
	}
	eos_to_code(eos);

}

void helmholtz_ztwd(double* p_ptr, double* e_ptr, double zbar, double rho) {
	const double A = 6.023e+22;
	const double B = 9.7595e+5 * zbar;
	double x;
	x = pow(rho / B, 1.0 / 3.0);
	if (x > 0.1) {
		*p_ptr = A * (x * (2.0 * x * x - 3.0) * sqrt(x * x + 1.0) + 3.0 * log(x + sqrt(x * x + 1.0)));
	} else {
		*p_ptr = A * 1.6 * pow(x, 5);
	}
	*e_ptr = A * (8.0 * pow(x, 3) * (sqrt(x * x + 1.0) - 1.0)) - *p_ptr / rho;
}

void helmholtz_compute_T(eos_t* eos) {
	double e0, f, dT, T0, tmin, tmax;
	int iters;
	int sign, last_sign;
	T0 = eos->T;
	e0 = eos->e;
	eos->T = HELMHOLTZ_TMIN;
	helmholtz_eos(eos);
	eos->T = T0;
	if (eos->e >= e0) {
		eos->T = HELMHOLTZ_TMIN;
	} else {
		//	tmax = HELMHOLTZ_TMAX;
//	tmin = HELMHOLTZ_TMIN;
	if (eos->e > e0) {
			printf("helmholtz_compute_T failed: Energy is too low, e is %e and needs to be %e\n", e0, eos->e);
			abort();
	} else {
		iters = 0;
		do {
			//	last_sign = f > 0.0 ? 1 : -1;
			T0 = eos->T;
			helmholtz_eos(eos);
			f = (eos->e - e0);
			dT = -f / eos->cv;
			if (iters > 90) {
				printf("%i %e %e %e %e %e\n", iters, f / e0, eos->T, eos->cv, eos->rho, dT);
			}
			//		sign = f > 0.0 ? 1 : -1;
			//			if (iters > 90) {
			//		printf("%10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %i %i\n", f / e0, eos->T, eos->rho, eos->e, dT,
			//				tmin, tmax, sign, last_sign);
			//		}
			/*			if (last_sign != sign && iters > 0) {
			 if (dT > 0.0) {
			 tmin = min(T0, eos->T);
			 } else {
			 tmax = max(T0, eos->T);
			 }
			 }
			 dT *= 0.9999;*/
			eos->T += dT;
			if (eos->T < 1.0) {
				eos->T = 1.0;
				break;
			}
			/*	if (iters > 0) {*/
			//		 eos->T = max(eos->T, (T0 + tmin) / 2.0);
			//	eos->T = min(eos->T, (T0 + HELMHOLTZ_TMAX) / 2.0);
			//	 }
			//		 T0 = eos->T;
			iters++;
		} while (fabs(f / e0) >= 1.0e-9 && iters < 100);
		if (iters >= 100) {
			printf("Helmholtz NR failed to coverge\n");
			abort();
		}
	}
  }
}
#endif

