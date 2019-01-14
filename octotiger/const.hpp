#ifndef __CONST___H
#define __CONST___H

const double pi = 3.1415926535897932384e0, eulercon = 0.577215664901532861e0, a2rad = pi / 180.0e0, rad2a = 180.0e0
		/ pi;

// physical constants
const double g = 6.6742867e-8, h = 6.6260689633e-27, hbar = 0.5e0 * h / pi, qe = 4.8032042712e-10,
		avo = 6.0221417930e23, clight = 2.99792458e10, kerg = 1.380650424e-16, ev2erg = 1.60217648740e-12, kev = kerg
				/ ev2erg, amu = 1.66053878283e-24, mn = 1.67492721184e-24, mp = 1.67262163783e-24,
		me = 9.1093821545e-28, rbohr = hbar * hbar / (me * qe * qe), fine = qe * qe / (hbar * clight), hion =
				13.605698140e0;

const double ssol = 5.67051e-5, asol = 4.0e0 * ssol / clight, weinlam = h * clight / (kerg * 4.965114232e0), weinfre =
		2.821439372e0 * kerg / h, rhonuc = 2.342e14;

// astronomical constants
const double msol = 1.9892e33, rsol = 6.95997e10, lsol = 3.8268e33, mearth = 5.9764e27, rearth = 6.37e8, ly =
		9.460528e17, pc = 3.261633e0 * ly, au = 1.495978921e13, secyer = 3.1558149984e7;

#endif

