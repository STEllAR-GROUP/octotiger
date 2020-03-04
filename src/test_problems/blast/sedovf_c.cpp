//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Original Fortran source: http://cococubed.asu.edu/research_pages/sedov.shtml



/* sedov3.f -- translated by f2c (version 20160102).
 You must link the resulting object file with libf2c:
 on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm
 or, if you install libf2c.a in a standard place, with -lf2c -lm
 -- in that order, at the end of the command line, as ina
 cc *.o -lf2c -lm
 Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

 http://www.netlib.org/f2c/libf2c.zip
 */

#include <cmath>
#include <vector>
#include <unordered_map>
#include <memory>
//#include "f2c.h"
#include <memory>
#if !defined(OCTOTIGER_HAVE_BOOST_MULTIPRECISION)
#include <quadmath.h>
using sed_real = __float128;

sed_real _exp(sed_real a) {
    return expq(a);
}

sed_real pow_dd(sed_real *a, sed_real *b) {
    return powq(*a, *b);
}


sed_real d_sign(sed_real *a, sed_real * b) {
    return copysignq(*a, *b);
}
#else
#include <boost/multiprecision/cpp_bin_float.hpp>
using sed_real = boost::multiprecision::cpp_bin_float_quad;

sed_real _exp(sed_real a) {
    return boost::multiprecision::exp(a);
}

sed_real pow_dd(sed_real *a, sed_real *b) {
    return boost::multiprecision::pow(*a, *b);
}


sed_real d_sign(sed_real *a, sed_real * b) {
    if ((*a > static_cast<sed_real>(0.) && (*b > static_cast<sed_real>(0.))) ||
        (*a < static_cast<sed_real>(0.) && (*b < static_cast<sed_real>(0.))))
    {
        return *a;
    }
    return *b;
}
#endif

/* Subroutine */int sed_1d__(sed_real *time, int *nstep,
        sed_real * xpos, sed_real *eblast, sed_real *omega_in__,
        sed_real * xgeom_in__, sed_real *rho0, sed_real *vel0,
        sed_real *ener0, sed_real *pres0, sed_real *cs0, sed_real *gam0,
        sed_real *den, sed_real *ener, sed_real *pres, sed_real *vel,
        sed_real *cs);
/* Common Block Declarations */

struct {
    sed_real gamma, gamm1, gamp1, gpogm, xgeom, xg2, rwant, r2, a0, a1, a2, a3,
            a4, a5, a_val__, b_val__, c_val__, d_val__, e_val__, omega, vv,
            xlam_want__, vwant, rvv;
    bool lsingular, lstandard, lvacuum, lomega2, lomega3;
} slap_;

#define slap_1 slap_

struct {
    sed_real gam_int__;
} cmidp_;

#define cmidp_1 cmidp_

/* Table of constant values */

using D_fp = sed_real (*)(sed_real*);
using S_fp = int (*)(
    D_fp, sed_real*, sed_real*, sed_real*, int*);    //Subroutine
using U_fp = int (*)();                              // Unknown procedure type

static sed_real c_b52 = 2.;
static sed_real c_b53 = 1e-10;
static sed_real c_b79 = 0.f;
static sed_real c_b80 = 1e-30;
static int c__3 = 3;
static int c__5 = 5;
static sed_real efun01_(sed_real *v);
static sed_real efun02_(sed_real *v);
static sed_real sed_v_find__(sed_real *v);
static sed_real sed_r_find__(sed_real *r__);
/* Subroutine */int sedov_funcs__(sed_real *v, sed_real *l_fun__, sed_real *dlamdv,
        sed_real *f_fun__, sed_real *g_fun__, sed_real *h_fun__);
/* Subroutine */static int midpnt_(D_fp func, sed_real *a, sed_real *b, sed_real *s,
        int *n);
/* Subroutine */static int midpowl_(D_fp funk, sed_real *aa, sed_real *bb,
        sed_real *s, int *n);
/* Subroutine */static int midpowl2_(D_fp funk, sed_real *aa, sed_real *bb,
        sed_real *s, int *n);
/* Subroutine */static int qromo_(D_fp func, sed_real *a, sed_real *b, sed_real *eps,
        sed_real *ss, S_fp choose);
/* Subroutine */static int polint_(sed_real *xa, sed_real *ya, int *n,
        sed_real *x, sed_real *y, sed_real *dy);
static sed_real zeroin_(sed_real *ax, sed_real *bx, D_fp f, sed_real *tol);


int pow_ii(int * a, int * b) {
    return std::pow(*a, *b);
}


/* Subroutine */int sed_1d__(sed_real *time, int *nstep, sed_real * xpos,
        sed_real *eblast, sed_real *omega_in__, sed_real * xgeom_in__, sed_real *rho0,
        sed_real *vel0, sed_real *ener0, sed_real *pres0, sed_real *cs0, sed_real *gam0,
        sed_real *den, sed_real *ener, sed_real *pres, sed_real *vel, sed_real *cs) {

    /* System generated locals */
    int i__1;
    sed_real d__1, d__2, d__3;

    /* Builtin functions */

    /* Local variables */
    static int i__;
    static sed_real p2, v0, u2, v2, us;
    static sed_real vat, rho1, rho2;
    static sed_real vmin, eval1, eval2, alpha, f_fun__;
    static sed_real g_fun__, h_fun__, l_fun__;
    static sed_real vstar, denom2, denom3, dlamdv;
    sed_real zeroin_(sed_real *ax, sed_real *bx, D_fp f, sed_real *tol);

    /* ..this routine produces 1d solutions for a sedov blast wave propagating */
    /* ..through a density gradient rho = rho**(-omega) */
    /* ..in planar, cylindrical or spherical geometry */
    /* ..for the standard, singular and vaccum cases. */
    /* ..standard case: a nonzero solution extends from the shock to the origin, */
    /* ..               where the pressure is finite. */
    /* ..singular case: a nonzero solution extends from the shock to the origin, */
    /* ..               where the pressure vanishes. */
    /* ..vacuum case  : a nonzero solution extends from the shock to a boundary point, */
    /* ..               where the density vanishes making the pressure meaningless. */
    /* ..input: */
    /* ..time     = temporal point where solution is desired seconds */
    /* ..xpos(i)  = spatial points where solution is desired cm */
    /* ..eblast   = energy of blast erg */
    /* ..rho0     = ambient density g/cm**3    rho = rho0 * r**(-omega_in) */
    /* ..omegain  = density power law _exponent rho = rho0 * r**(-omega_in) */
    /* ..vel0     = ambient material speed cm/s */
    /* ..pres0    = ambient pressure erg/cm**3 */
    /* ..cs0      = ambient sound speed cm/s */
    /* ..gam0   = gamma law equation of state */
    /* ..xgeom_in = geometry factor, 3=spherical, 2=cylindircal, 1=planar */
    /* ..for efficiency reasons (doing the energy integrals only once), */
    /* ..this routine returns the solution for an array of spatial points */
    /* ..at the desired time point. */
    /* ..output: */
    /* ..den(i)  = density  g/cm**3 */
    /* ..ener(i) = specific internal energy erg/g */
    /* ..pres(i) = presssure erg/cm**3 */
    /* ..vel(i)  = velocity cm/s */
    /* ..cs(i)   = sound speed cm/s */
    /* ..this routine is based upon two papers: */
    /* .."evaluation of the sedov-von neumann-taylor blast wave solution" */
    /* ..jim kamm, la-ur-00-6055 */
    /* .."the sedov self-similiar point blast solutions in nonuniform media" */
    /* ..david book, shock waves, 4, 1, 1994 */
    /* ..although the ordinary differential equations are analytic, */
    /* ..the sedov _expressions appear to become singular for various */
    /* ..combinations of parameters and at the lower limits of the integration */
    /* ..range. all these singularies are removable and done so by this routine. */
    /* ..these routines are written in sed_real*8 precision because the */
    /* ..sed_real*8 implementations simply run out of precision "near" the origin */
    /* ..in the standard case or the transition region in the vacuum case. */
    /* ..declare the pass */
    /* ..local variables */
    /* ..eps controls the integration accuracy, don't get too greedy or the number */
    /* ..of function evaluations required kills. */
    /* ..eps2 controls the root find accuracy */
    /* ..osmall controls the size of transition regions */
    /* ..common block communication */
    /* ..common block communication with the integration stepper */
    /* ..popular formats */
    /* ..initialize the solution */
    /* Parameter adjustments */
    --cs;
    --vel;
    --pres;
    --ener;
    --den;
    --xpos;

    /* Function Body */
    /* L87: */
    /* L88: */
    i__1 = *nstep;
    for (i__ = 1; i__ <= i__1; ++i__) {
        den[i__] = 0.f;
        vel[i__] = 0.f;
        pres[i__] = 0.f;
        ener[i__] = 0.f;
        cs[i__] = 0.f;
    }
    /* ..return on unphysical cases */
    /* ..infinite mass */
    if (*omega_in__ >= *xgeom_in__) {
        return 0;
    }
    /* ..transfer the pass to common block and create some frequent combinations */
    slap_1.gamma = *gam0;
    slap_1.gamm1 = slap_1.gamma - 1.f;
    slap_1.gamp1 = slap_1.gamma + 1.f;
    slap_1.gpogm = slap_1.gamp1 / slap_1.gamm1;
    slap_1.xgeom = *xgeom_in__;
    slap_1.omega = *omega_in__;
    slap_1.xg2 = slap_1.xgeom + 2.f - slap_1.omega;
    denom2 = slap_1.gamm1 * 2.f + slap_1.xgeom - slap_1.gamma * slap_1.omega;
    denom3 = slap_1.xgeom * (2.f - slap_1.gamma) - slap_1.omega;
    /* ..post shock location v2 and location of singular point vstar */
    /* ..kamm equation 18 and 19 */
    v2 = 4.f / (slap_1.xg2 * slap_1.gamp1);
    vstar = 2.f / (slap_1.gamm1 * slap_1.xgeom + 2.f);
    /* ..set two bools that determines the type of solution */
    slap_1.lstandard = false;
    slap_1.lsingular = false;
    slap_1.lvacuum = false;
    if ((d__1 = v2 - vstar, fabs(static_cast<double>(d__1))) <= 1e-4) {
        slap_1.lsingular = true;
    } else if (v2 < vstar - 1e-4) {
        slap_1.lstandard = true;
    } else if (v2 > vstar + 1e-4) {
        slap_1.lvacuum = true;
    }
    /* ..two apparent singularies, book's notation for omega2 and omega3 */
    slap_1.lomega2 = false;
    slap_1.lomega3 = false;
    if (fabs(static_cast<double>(denom2)) <= 1e-4) {
        slap_1.lomega2 = true;
        denom2 = 1e-8f;
    } else if (fabs(static_cast<double>(denom3)) <= 1e-4) {
        slap_1.lomega3 = true;
        denom3 = 1e-8f;
    }
    /* ..various _exponents, kamm equations 42-47 */
    /* ..in terms of book's notation: */
    /* ..a0=beta6 a1=beta1  a2=-beta2 a3=beta3 a4=beta4 a5=-beta5 */
    slap_1.a0 = 2.f / slap_1.xg2;
    slap_1.a2 = -slap_1.gamm1 / denom2;
    slap_1.a1 = slap_1.xg2 * slap_1.gamma / (slap_1.xgeom * slap_1.gamm1 + 2.f)
            * ((slap_1.xgeom * (2.f - slap_1.gamma) - slap_1.omega) * 2.f
                    / (slap_1.gamma * slap_1.xg2 * slap_1.xg2) - slap_1.a2);
    slap_1.a3 = (slap_1.xgeom - slap_1.omega) / denom2;
    slap_1.a4 = slap_1.xg2 * (slap_1.xgeom - slap_1.omega) * slap_1.a1 / denom3;
    slap_1.a5 = (slap_1.omega * slap_1.gamp1 - slap_1.xgeom * 2.f) / denom3;
    /* ..frequent combinations, kamm equations 33-37 */
    slap_1.a_val__ = slap_1.xg2 * .25f * slap_1.gamp1;
    slap_1.b_val__ = slap_1.gpogm;
    slap_1.c_val__ = slap_1.xg2 * .5f * slap_1.gamma;
    slap_1.d_val__ = slap_1.xg2 * slap_1.gamp1
            / (slap_1.xg2 * slap_1.gamp1
                    - (slap_1.xgeom * slap_1.gamm1 + 2.f) * 2.f);
    slap_1.e_val__ = (slap_1.xgeom * slap_1.gamm1 + 2.f) * .5f;
    /* ..evaluate the energy integrals */
    /* ..the singular case can be done by hand; save some cpu cycles */
    /* ..kamm equations 80, 81, and 85 */
    if (slap_1.lsingular) {
        /* Computing 2nd power */
        d__1 = slap_1.gamm1 * slap_1.xgeom + 2.f;
        eval2 = slap_1.gamp1 / (slap_1.xgeom * (d__1 * d__1));
        eval1 = 2.f / slap_1.gamm1 * eval2;
        /* Computing 2nd power */
        d__1 = slap_1.gamm1 * slap_1.xgeom + 2.f;
        alpha = slap_1.gpogm * pow_dd(&c_b52, &slap_1.xgeom)
                / (slap_1.xgeom * (d__1 * d__1));
        if (static_cast<int>(slap_1.xgeom) != 1) {
            alpha *= 3.1415926535897932384626433832795029;
        }
        /* ..for the standard or vacuum cases */
        /* ..v0 = post-shock origin v0 and vv = vacuum boundary vv */
        /* ..set the radius corespondin to vv to zero for now */
        /* ..kamm equations 18, and 28. */
    } else {
        v0 = 2.f / (slap_1.xg2 * slap_1.gamma);
        slap_1.vv = 2.f / slap_1.xg2;
        slap_1.rvv = 0.;
        if (slap_1.lstandard) {
            vmin = v0;
        }
        if (slap_1.lvacuum) {
            vmin = slap_1.vv;
        }
        /* ..the first energy integral */
        /* ..in the standard case the term (c_val*v - 1) might be singular at v=vmin */
        if (slap_1.lstandard) {
            cmidp_1.gam_int__ = slap_1.a3 - slap_1.a2 * slap_1.xg2 - 1.f;
            if (cmidp_1.gam_int__ >= 0.) {
                qromo_(static_cast<D_fp>(efun01_), &vmin, &v2, &c_b53, &eval1,
                        static_cast<S_fp>(midpnt_));
            } else {
                cmidp_1.gam_int__ = fabs(static_cast<double>(cmidp_1.gam_int__));
                qromo_(static_cast<D_fp>(efun01_), &vmin, &v2, &c_b53, &eval1,
                        static_cast<S_fp>(midpowl_));
            }
            /* ..in the vacuum case the term (1 - c_val/gamma*v) might be singular at v=vmin */
        } else if (slap_1.lvacuum) {
            cmidp_1.gam_int__ = slap_1.a5;
            if (cmidp_1.gam_int__ >= 0.) {
                qromo_(static_cast<D_fp>(efun01_), &vmin, &v2, &c_b53, &eval1,
                        static_cast<S_fp>(midpnt_));
            } else {
                cmidp_1.gam_int__ = fabs(static_cast<double>(cmidp_1.gam_int__));
                qromo_(static_cast<D_fp>(efun01_), &vmin, &v2, &c_b53, &eval1,
                        static_cast<S_fp>(midpowl2_));
            }
        }
        /* ..the second energy integral */
        /* ..in the standard case the term (c_val*v - 1) might be singular at v=vmin */
        if (slap_1.lstandard) {
            cmidp_1.gam_int__ = slap_1.a3 - slap_1.a2 * slap_1.xg2 - 2.f;
            if (cmidp_1.gam_int__ >= 0.) {
                qromo_(static_cast<D_fp>(efun02_), &vmin, &v2, &c_b53, &eval2,
                        static_cast<S_fp>(midpnt_));
            } else {
                cmidp_1.gam_int__ = fabs(static_cast<double>(cmidp_1.gam_int__));
                qromo_(static_cast<D_fp>(efun02_), &vmin, &v2, &c_b53, &eval2,
                        static_cast<S_fp>(midpowl_));
            }
            /* ..in the vacuum case the term (1 - c_val/gamma*v) might be singular at v=vmin */
        } else if (slap_1.lvacuum) {
            cmidp_1.gam_int__ = slap_1.a5;
            if (cmidp_1.gam_int__ >= 0.) {
                qromo_(static_cast<D_fp>(efun02_), &vmin, &v2, &c_b53, &eval2,
                        static_cast<S_fp>(midpnt_));
            } else {
                cmidp_1.gam_int__ = fabs(static_cast<double>(cmidp_1.gam_int__));
                qromo_(static_cast<D_fp>(efun02_), &vmin, &v2, &c_b53, &eval2,
                        static_cast<S_fp>(midpowl2_));
            }
        }
        /* ..kamm equations 57 and 58 for alpha, in a slightly different form. */
        if (static_cast<int>(slap_1.xgeom) == 1) {
            alpha = eval1 * .5f + eval2 / slap_1.gamm1;
        } else {
            alpha = (slap_1.xgeom - 1.f) * 3.1415926535897932384626433832795029
                    * (eval1 + eval2 * 2.f / slap_1.gamm1);
        }
    }
    /* ..write what we have for the energy integrals */
    if (true) {
//		s_wsfe(&io___42);
//		do_fio(&c__1, "xgeom =", (ftnlen) 7);
//		do_fio(&c__1, (char *) &slap_1.xgeom, (ftnlen) sizeof(sed_real));
//		do_fio(&c__1, "eblast=", (ftnlen) 7);
//		do_fio(&c__1, (char *) &(*eblast), (ftnlen) sizeof(sed_real));
//		do_fio(&c__1, "omega =", (ftnlen) 7);
//		do_fio(&c__1, (char *) &slap_1.omega, (ftnlen) sizeof(sed_real));
//		do_fio(&c__1, "alpha =", (ftnlen) 7);
//		do_fio(&c__1, (char *) &alpha, (ftnlen) sizeof(sed_real));
//		do_fio(&c__1, "j1    =", (ftnlen) 7);
//		do_fio(&c__1, (char *) &eval1, (ftnlen) sizeof(sed_real));
//		do_fio(&c__1, "j2    =", (ftnlen) 7);
//		do_fio(&c__1, (char *) &eval2, (ftnlen) sizeof(sed_real));
//		e_wsfe();
    }
    /*      write(6,87) omega,alpha */
    /* ..immediate post-shock values */
    /* ..kamm page 14 or equations 14, 16, 5, 13 */
    /* ..r2 = shock position, u2 = shock speed, rho1 = pre-shock density, */
    /* ..u2 = post-shock material speed, rho2 = post-shock density, */
    /* ..p2 = post-shock pressure, e2 = post-shoock specific internal energy, */
    /* ..and cs2 = post-shock sound speed */
    d__1 = *eblast / (alpha * *rho0);
    d__2 = 1.f / slap_1.xg2;
    d__3 = 2.f / slap_1.xg2;
    slap_1.r2 = pow_dd(&d__1, &d__2) * pow_dd(time, &d__3);
    us = 2.f / slap_1.xg2 * slap_1.r2 / *time;
    d__1 = -slap_1.omega;
    rho1 = *rho0 * pow_dd(&slap_1.r2, &d__1);
    u2 = us * 2.f / slap_1.gamp1;
    rho2 = slap_1.gpogm * rho1;
    /* Computing 2nd power */
    d__1 = us;
    p2 = rho1 * 2.f * (d__1 * d__1) / slap_1.gamp1;
//	e2 = p2 / (slap_1.gamm1 * rho2);
//	cs2 = sqrt(slap_1.gamma * p2 / rho2);
    /* ..find the radius corresponding to vv */
    if (slap_1.lvacuum) {
        slap_1.vwant = slap_1.vv;
        slap_1.rvv = zeroin_(&c_b79, &slap_1.r2, static_cast<D_fp>(sed_r_find__), &c_b80);
    }
//	if (slap_1.lstandard) {
//		s_wsfe(&io___50);
//		do_fio(&c__1, "r2    =", (ftnlen) 7);
//		do_fio(&c__1, (char *) &slap_1.r2, (ftnlen) sizeof(sed_real));
//		do_fio(&c__1, "rho2  =", (ftnlen) 7);
//		do_fio(&c__1, (char *) &rho2, (ftnlen) sizeof(sed_real));
//		do_fio(&c__1, "u2    =", (ftnlen) 7);
//		do_fio(&c__1, (char *) &u2, (ftnlen) sizeof(sed_real));
//		do_fio(&c__1, "e2    =", (ftnlen) 7);
//		do_fio(&c__1, (char *) &e2, (ftnlen) sizeof(sed_real));
//		do_fio(&c__1, "p2    =", (ftnlen) 7);
//		do_fio(&c__1, (char *) &p2, (ftnlen) sizeof(sed_real));
//		do_fio(&c__1, "cs2   =", (ftnlen) 7);
//		do_fio(&c__1, (char *) &cs2, (ftnlen) sizeof(sed_real));
//		e_wsfe();
//	}
//	if (slap_1.lvacuum) {
//		s_wsfe(&io___51);
//		do_fio(&c__1, "rv    =", (ftnlen) 7);
//		do_fio(&c__1, (char *) &slap_1.rvv, (ftnlen) sizeof(sed_real));
//		do_fio(&c__1, "r2    =", (ftnlen) 7);
//		do_fio(&c__1, (char *) &slap_1.r2, (ftnlen) sizeof(sed_real));
//		do_fio(&c__1, "rho2  =", (ftnlen) 7);
//		do_fio(&c__1, (char *) &rho2, (ftnlen) sizeof(sed_real));
//		do_fio(&c__1, "u2    =", (ftnlen) 7);
//		do_fio(&c__1, (char *) &u2, (ftnlen) sizeof(sed_real));
//		do_fio(&c__1, "e2    =", (ftnlen) 7);
//		do_fio(&c__1, (char *) &e2, (ftnlen) sizeof(sed_real));
//		do_fio(&c__1, "p2    =", (ftnlen) 7);
//		do_fio(&c__1, (char *) &p2, (ftnlen) sizeof(sed_real));
//		do_fio(&c__1, "cs2   =", (ftnlen) 7);
//		do_fio(&c__1, (char *) &cs2, (ftnlen) sizeof(sed_real));
//		e_wsfe();
//	}
    /* ..now start the loop over spatial positions */
    i__1 = *nstep;
    for (i__ = 1; i__ <= i__1; ++i__) {
        slap_1.rwant = xpos[i__];
        /* ..if we are upstream from the shock front */
        if (slap_1.rwant > slap_1.r2) {
            d__1 = -slap_1.omega;
            den[i__] = *rho0 * pow_dd(&slap_1.rwant, &d__1);
            vel[i__] = *vel0;
            pres[i__] = *pres0;
            ener[i__] = *ener0;
            cs[i__] = *cs0;
            /* ..if we are between the origin and the shock front */
            /* ..find the correct similarity value for this radius in the standard or vacuum cases */
        } else {
            if (slap_1.lstandard) {
                d__1 = v0 * .9f;
                vat = zeroin_(&d__1, &v2, static_cast<D_fp>(sed_v_find__), &c_b80);
            } else if (slap_1.lvacuum) {
                d__1 = slap_1.vv * 1.2f;
                vat = zeroin_(&v2, &d__1, static_cast<D_fp>(sed_v_find__), &c_b80);
            }
            /* ..the physical solution */
            sedov_funcs__(&vat, &l_fun__, &dlamdv, &f_fun__, &g_fun__,
                    &h_fun__);
            den[i__] = rho2 * g_fun__;
            vel[i__] = u2 * f_fun__;
            pres[i__] = p2 * h_fun__;
            ener[i__] = 0.f;
            cs[i__] = 0.f;
            if (den[i__] != 0.f) {
                ener[i__] = pres[i__] / (slap_1.gamm1 * den[i__]);
                cs[i__] = sqrt(static_cast<double>(slap_1.gamma * pres[i__] / den[i__]));
            }
        }
        /* ..end of loop over positions */
    }
    return 0;
} /* sed_1d__ */

sed_real efun01_(sed_real *v) {
    /* System generated locals */
    sed_real ret_val, d__1, d__2;
    ret_val = 0;

    /* Builtin functions */
    sed_real pow_dd(sed_real *, sed_real *);

    /* Local variables */
    static sed_real f_fun__, g_fun__, h_fun__, l_fun__, dlamdv;

    /* ..evaluates the first energy integrand, kamm equations 67 and 10. */
    /* ..the (c_val*v - 1) term might be singular at v=vmin in the standard case. */
    /* ..the (1 - c_val/gamma * v) term might be singular at v=vmin in the vacuum case. */
    /* ..due care should be taken for these removable singularities by the integrator. */
    /* ..declare the pass */
    /* ..common block communication */
    /* ..local variables */
    /* ..go */
    sedov_funcs__(v, &l_fun__, &dlamdv, &f_fun__, &g_fun__, &h_fun__);
    d__1 = slap_1.xgeom + 1.f;
    /* Computing 2nd power */
    d__2 = *v;
    ret_val = dlamdv * pow_dd(&l_fun__, &d__1) * slap_1.gpogm * g_fun__
            * (d__2 * d__2);
    return ret_val;
} /* efun01_ */

sed_real efun02_(sed_real *v) {
    /* System generated locals */
    sed_real ret_val, d__1;
    ret_val = 0;

    /* Builtin functions */
    sed_real pow_dd(sed_real *, sed_real *);

    /* Local variables */
    static sed_real z__;
    static sed_real f_fun__, g_fun__, h_fun__, l_fun__, dlamdv;

    /* ..evaluates the second energy integrand, kamm equations 68 and 11. */
    /* ..the (c_val*v - 1) term might be singular at v=vmin in the standard case. */
    /* ..the (1 - c_val/gamma * v) term might be singular at v=vmin in the vacuum case. */
    /* ..due care should be taken for these removable singularities by the integrator. */
    /* ..declare the pass */
    /* ..common block communication */
    /* ..local variables */
    /* ..go */
    sedov_funcs__(v, &l_fun__, &dlamdv, &f_fun__, &g_fun__, &h_fun__);
    /* Computing 2nd power */
    d__1 = slap_1.xgeom + 2.f - slap_1.omega;
    z__ = 8.f / (d__1 * d__1 * slap_1.gamp1);
    d__1 = slap_1.xgeom - 1.f;
    ret_val = dlamdv * pow_dd(&l_fun__, &d__1) * h_fun__ * z__;
    return ret_val;
} /* efun02_ */

sed_real sed_v_find__(sed_real *v) {
    /* System generated locals */
    sed_real ret_val;
    ret_val = 0;

    /* Local variables */
    static sed_real f_fun__, g_fun__, h_fun__, l_fun__, dlamdv;

    /* ..given corresponding physical distances, find the similarity variable v */
    /* ..kamm equation 38 as a root find */
    /* ..declare the pass */
    /* ..common block communication */
    /* ..local variables */
    sedov_funcs__(v, &l_fun__, &dlamdv, &f_fun__, &g_fun__, &h_fun__);
    ret_val = slap_1.r2 * l_fun__ - slap_1.rwant;
    return ret_val;
} /* sed_v_find__ */

sed_real sed_r_find__(sed_real *r__) {
    /* System generated locals */
    sed_real ret_val;
    ret_val = 0;

    /* Local variables */
    static sed_real f_fun__, g_fun__, h_fun__, l_fun__, dlamdv;

    /* ..given the similarity variable v, find the corresponding physical distance */
    /* ..kamm equation 38 as a root find */
    /* ..declare the pass */
    /* ..common block communication */
    /* ..local variables */
    sedov_funcs__(&slap_1.vwant, &l_fun__, &dlamdv, &f_fun__, &g_fun__,
            &h_fun__);
    ret_val = slap_1.r2 * l_fun__ - *r__;
    return ret_val;
} /* sed_r_find__ */

/* Subroutine */int sedov_funcs__(sed_real *v, sed_real *l_fun__, sed_real *dlamdv,
        sed_real *f_fun__, sed_real *g_fun__, sed_real *h_fun__) {
    /* System generated locals */
    sed_real d__1, d__2, d__3;

    /* Builtin functions */
//	sed_real pow_dd(sed_real *, sed_real *), _exp(sed_real);

    /* Local variables */
    static sed_real y, z__, c2, c6, x1, x2, x3, x4, pp1, pp2, pp3, pp4, cbag,
            ebag, beta0, dx1dv, dx2dv, dx3dv, dx4dv, dpp2dv;

    /* ..given the similarity variable v, returns functions */
    /* ..lambda, f, g, and h and the derivative of lambda with v dlamdv */
    /* ..although the ordinary differential equations are analytic, */
    /* ..the sedov _expressions appear to become singular for various */
    /* ..combinations of parameters and at the lower limits of the integration */
    /* ..range. all these singularies are removable and done so by this routine. */
    /* ..declare the pass */
    /* ..common block communication */
    /* ..local variables */
    /* ..frequent combinations and their derivative with v */
    /* ..kamm equation 29-32, x4 a bit different to save a divide */
    /* ..x1 is book's F */
    x1 = slap_1.a_val__ * *v;
    dx1dv = slap_1.a_val__;
    /* Computing MAX */
    d__1 = 1e-30, d__2 = slap_1.c_val__ * *v - 1.f;
    cbag = fmax(static_cast<double>(d__1), static_cast<double>(d__2));
    x2 = slap_1.b_val__ * cbag;
    dx2dv = slap_1.b_val__ * slap_1.c_val__;
    ebag = 1.f - slap_1.e_val__ * *v;
    x3 = slap_1.d_val__ * ebag;
    dx3dv = -slap_1.d_val__ * slap_1.e_val__;
    x4 = slap_1.b_val__ * (1.f - slap_1.xg2 * .5f * *v);
    dx4dv = -slap_1.b_val__ * .5f * slap_1.xg2;
    /* ..transition region between standard and vacuum cases */
    /* ..kamm page 15 or equations 88-92 */
    /* ..lambda = l_fun is book's zeta */
    /* ..f_fun is books V, g_fun is book's D, h_fun is book's P */
    if (slap_1.lsingular) {
        *l_fun__ = slap_1.rwant / slap_1.r2;
        *dlamdv = 0.f;
        *f_fun__ = *l_fun__;
        d__1 = slap_1.xgeom - 2.f;
        *g_fun__ = pow_dd(l_fun__, &d__1);
        *h_fun__ = pow_dd(l_fun__, &slap_1.xgeom);
        /* ..for the vacuum case in the hole */
    } else if (slap_1.lvacuum && slap_1.rwant < slap_1.rvv) {
        *l_fun__ = 0.f;
        *dlamdv = 0.f;
        *f_fun__ = 0.f;
        *g_fun__ = 0.f;
        *h_fun__ = 0.f;
        /* ..omega = omega2 = (2*(gamma -1) + xgeom)/gamma case, denom2 = 0 */
        /* ..book _expressions 20-22 */
    } else if (slap_1.lomega2) {
        beta0 = 1.f / (slap_1.e_val__ * 2.f);
        pp1 = slap_1.gamm1 * beta0;
        c6 = slap_1.gamp1 * .5f;
        c2 = c6 / slap_1.gamma;
        y = 1.f / (x1 - c2);
        z__ = (1.f - x1) * y;
        pp2 = slap_1.gamp1 * beta0 * z__;
        dpp2dv = -slap_1.gamp1 * beta0 * dx1dv * y * (z__ + 1.f);
        pp3 = (4.f - slap_1.xgeom - slap_1.gamma * 2.f) * beta0;
        pp4 = -slap_1.xgeom * slap_1.gamma * beta0;
        d__1 = -slap_1.a0;
        *l_fun__ = pow_dd(&x1, &d__1) * pow_dd(&x2, &pp1) * _exp(pp2);
        *dlamdv = (-slap_1.a0 * dx1dv / x1 + pp1 * dx2dv / x2 + dpp2dv)
                * *l_fun__;
        *f_fun__ = x1 * *l_fun__;
        d__1 = slap_1.a0 * slap_1.omega;
        *g_fun__ = pow_dd(&x1, &d__1) * pow_dd(&x2, &pp3) * pow_dd(&x4, &
        slap_1.a5) * _exp(pp2 * -2.f);
        d__1 = slap_1.a0 * slap_1.xgeom;
        d__2 = slap_1.a5 + 1.f;
        *h_fun__ = pow_dd(&x1, &d__1) * pow_dd(&x2, &pp4) * pow_dd(&x4, &d__2);
        /* ..omega = omega3 = xgeom*(2 - gamma) case, denom3 = 0 */
        /* ..book _expressions 23-25 */
    } else if (slap_1.lomega3) {
        beta0 = 1.f / (slap_1.e_val__ * 2.f);
        pp1 = slap_1.a3 + slap_1.omega * slap_1.a2;
        pp2 = 1.f - beta0 * 4.f;
        c6 = slap_1.gamp1 * .5f;
        pp3 = -slap_1.xgeom * slap_1.gamma * slap_1.gamp1 * beta0 * (1.f - x1)
                / (c6 - x1);
        pp4 = (slap_1.xgeom * slap_1.gamm1 - slap_1.gamma) * 2.f * beta0;
        d__1 = -slap_1.a0;
        d__2 = -slap_1.a2;
        d__3 = -slap_1.a1;
        *l_fun__ = pow_dd(&x1, &d__1) * pow_dd(&x2, &d__2) * pow_dd(&x4, &d__3);
        *dlamdv = -(slap_1.a0 * dx1dv / x1 + slap_1.a2 * dx2dv / x2 +
        slap_1.a1 * dx4dv / x4) * *l_fun__;
        *f_fun__ = x1 * *l_fun__;
        d__1 = slap_1.a0 * slap_1.omega;
        *g_fun__ = pow_dd(&x1, &d__1) * pow_dd(&x2, &pp1) * pow_dd(&x4, &pp2)
                * _exp(pp3);
        d__1 = slap_1.a0 * slap_1.xgeom;
        *h_fun__ = pow_dd(&x1, &d__1) * pow_dd(&x4, &pp4) * _exp(pp3);
        /* ..for the standard or vacuum case not in the hole */
        /* ..kamm equations 38-41 */
    } else {
        d__1 = -slap_1.a0;
        d__2 = -slap_1.a2;
        d__3 = -slap_1.a1;
        *l_fun__ = pow_dd(&x1, &d__1) * pow_dd(&x2, &d__2) * pow_dd(&x3, &d__3);
        *dlamdv = -(slap_1.a0 * dx1dv / x1 + slap_1.a2 * dx2dv / x2 +
        slap_1.a1 * dx3dv / x3) * *l_fun__;
        *f_fun__ = x1 * *l_fun__;
        d__1 = slap_1.a0 * slap_1.omega;
        d__2 = slap_1.a3 + slap_1.a2 * slap_1.omega;
        d__3 = slap_1.a4 + slap_1.a1 * slap_1.omega;
        *g_fun__ = pow_dd(&x1, &d__1) * pow_dd(&x2, &d__2) * pow_dd(&x3, &d__3)
                * pow_dd(&x4, &slap_1.a5);
        d__1 = slap_1.a0 * slap_1.xgeom;
        d__2 = slap_1.a4 + slap_1.a1 * (slap_1.omega - 2.f);
        d__3 = slap_1.a5 + 1.f;
        *h_fun__ = pow_dd(&x1, &d__1) * pow_dd(&x3, &d__2) * pow_dd(&x4, &d__3);
    }
    return 0;
} /* sedov_funcs__ */

/* Subroutine */int midpnt_(D_fp func, sed_real *a, sed_real *b, sed_real *s,
        int *n) {
    /* System generated locals */
    int i__1;
    sed_real d__1;

    /* Builtin functions */
    int pow_ii(int *, int *);

    /* Local variables */
    static int j;
    static sed_real x;
    static int it;
    static sed_real del, tnm, sum, ddel;

    /* ..this routine computes the n'th stage of refinement of an extended midpoint */
    /* ..rule. func is input as the name of the function to be integrated between */
    /* ..limits a and b. when called with n=1, the routine returns as s the crudest */
    /* ..estimate of the integralof func from a to b. subsequent calls with n=2,3... */
    /* ..improve the accuracy of s by adding 2/3*3**(n-1) addtional interior points. */
    /* ..declare */
    if (*n == 1) {
        d__1 = (*a + *b) * .5f;
        *s = (*b - *a) * (*func)(&d__1);
    } else {
        i__1 = *n - 2;
        it = pow_ii(&c__3, &i__1);
        tnm = static_cast<sed_real>(it);
        del = (*b - *a) / (tnm * 3.f);
        ddel = del + del;
        x = *a + del * .5f;
        sum = 0.f;
        i__1 = it;
        for (j = 1; j <= i__1; ++j) {
            sum += (*func)(&x);
            x += ddel;
            sum += (*func)(&x);
            x += del;
        }
        *s = (*s + (*b - *a) * sum / tnm) / 3.f;
    }
    return 0;
} /* midpnt_ */

/* Subroutine */int midpowl_(D_fp funk, sed_real *aa, sed_real *bb, sed_real *s,
        int *n) {
    /* System generated locals */
    int i__1;
    sed_real d__1, d__2, d__3, d__4;

    /* Builtin functions */
    sed_real pow_dd(sed_real *, sed_real *);
    int pow_ii(int *, int *);

    /* Local variables */
    static sed_real a, b;
    static int j;
    static sed_real x;
    static int it;
    static sed_real del, tnm, sum, ddel;

    /* ..this routine is an exact replacement for midpnt, except that it allows for */
    /* ..an integrable power-law singularity of the form (x - a)**(-gam_int) */
    /* ..at the lower limit aa for 0 < gam_int < 1. */
    /* ..declare */
    /* ..common block communication */
    /* ..a little conversion, recipe equation 4.4.3 */
    d__1 = *bb - *aa;
    d__2 = 1.f - cmidp_1.gam_int__;
    b = pow_dd(&d__1, &d__2);
    a = 0.f;
    /* ..now exactly as midpnt */
    if (*n == 1) {
        d__1 = (a + b) * .5f;
        d__2 = cmidp_1.gam_int__ / (1.f - cmidp_1.gam_int__);
        d__4 = 1.f / (1.f - cmidp_1.gam_int__);
        d__3 = pow_dd(&d__1, &d__4) + *aa;
        *s = (b - a)
                * (1.f / (1.f - cmidp_1.gam_int__) * pow_dd(&d__1, &d__2)
                        * (*funk)(&d__3));
    } else {
        i__1 = *n - 2;
        it = pow_ii(&c__3, &i__1);
        tnm = static_cast<sed_real>(it);
        del = (b - a) / (tnm * 3.f);
        ddel = del + del;
        x = a + del * .5f;
        sum = 0.f;
        i__1 = it;
        for (j = 1; j <= i__1; ++j) {
            d__1 = cmidp_1.gam_int__ / (1.f - cmidp_1.gam_int__);
            d__3 = 1.f / (1.f - cmidp_1.gam_int__);
            d__2 = pow_dd(&x, &d__3) + *aa;
            sum += 1.f / (1.f - cmidp_1.gam_int__) * pow_dd(&x, &d__1)
                    * (*funk)(&d__2);
            x += ddel;
            d__1 = cmidp_1.gam_int__ / (1.f - cmidp_1.gam_int__);
            d__3 = 1.f / (1.f - cmidp_1.gam_int__);
            d__2 = pow_dd(&x, &d__3) + *aa;
            sum += 1.f / (1.f - cmidp_1.gam_int__) * pow_dd(&x, &d__1)
                    * (*funk)(&d__2);
            x += del;
        }
        *s = (*s + (b - a) * sum / tnm) / 3.f;
    }
    return 0;
} /* midpowl_ */

/* Subroutine */int midpowl2_(D_fp funk, sed_real *aa, sed_real *bb, sed_real *s,
        int *n) {
    /* System generated locals */
    int i__1;
    sed_real d__1, d__2, d__3, d__4;

    /* Builtin functions */

    /* Local variables */
    static sed_real a, b;
    static int j;
    static sed_real x;
    static int it;
    static sed_real del, tnm, sum, ddel;

    /* ..this routine is an exact replacement for midpnt, except that it allows for */
    /* ..an integrable power-law singularity of the form (a - x)**(-gam_int) */
    /* ..at the lower limit aa for 0 < gam_int < 1. */
    /* ..declare */
    /* ..common block communication */
    /* ..a little conversion, modulo recipe equation 4.4.3 */
    d__1 = *aa - *bb;
    d__2 = 1.f - cmidp_1.gam_int__;
    b = pow_dd(&d__1, &d__2);
    a = 0.f;
    /* ..now exactly as midpnt */
    if (*n == 1) {
        d__1 = (a + b) * .5f;
        d__2 = cmidp_1.gam_int__ / (1.f - cmidp_1.gam_int__);
        d__4 = 1.f / (1.f - cmidp_1.gam_int__);
        d__3 = *aa - pow_dd(&d__1, &d__4);
        *s = (b - a)
                * (1.f / (cmidp_1.gam_int__ - 1.f) * pow_dd(&d__1, &d__2)
                        * (*funk)(&d__3));
    } else {
        i__1 = *n - 2;
        it = pow_ii(&c__3, &i__1);
        tnm = static_cast<sed_real>(it);
        del = (b - a) / (tnm * 3.f);
        ddel = del + del;
        x = a + del * .5f;
        sum = 0.f;
        i__1 = it;
        for (j = 1; j <= i__1; ++j) {
            d__1 = cmidp_1.gam_int__ / (1.f - cmidp_1.gam_int__);
            d__3 = 1.f / (1.f - cmidp_1.gam_int__);
            d__2 = *aa - pow_dd(&x, &d__3);
            sum += 1.f / (cmidp_1.gam_int__ - 1.f) * pow_dd(&x, &d__1)
                    * (*funk)(&d__2);
            x += ddel;
            d__1 = cmidp_1.gam_int__ / (1.f - cmidp_1.gam_int__);
            d__3 = 1.f / (1.f - cmidp_1.gam_int__);
            d__2 = *aa - pow_dd(&x, &d__3);
            sum += 1.f / (cmidp_1.gam_int__ - 1.f) * pow_dd(&x, &d__1)
                    * (*funk)(&d__2);
            x += del;
        }
        *s = (*s + (b - a) * sum / tnm) / 3.f;
    }
    return 0;
} /* midpowl2_ */

/* Subroutine */int qromo_(D_fp func, sed_real *a, sed_real *b, sed_real *eps,
        sed_real *ss, S_fp choose) {
    /* Builtin functions */

    /* Local variables */
    static sed_real h__[15];
    static int j;
    static sed_real s[15], dss;

    /* Fortran I/O blocks */

    /* ..this routine returns as s the integral of the function func from a to b */
    /* ..with fractional accuracy eps. *//* ..jmax limits the number of steps; nsteps = 3**(jmax-1) */
    /* ..integration is done via romberg algorithm. */
    /* ..it is assumed the call to choose triples the number of steps on each call */
    /* ..and that its error series contains only even powers of the number of steps. */
    /* ..the external choose may be any of the above drivers, i.e midpnt,midinf... */
    /* ..declare */
    h__[0] = 1.f;
    for (j = 1; j <= 14; ++j) {
        (*choose)(static_cast<D_fp>(func), a, b, &s[j - 1], &j);
        if (j >= 5) {
            polint_(&h__[j - 5], &s[j - 5], &c__5, &c_b79, ss, &dss);
            if (fabs(static_cast<double>(dss)) <= *eps * fabs(static_cast<double>(*ss))) {
                return 0;
            }
        }
        s[j] = s[j - 1];
        h__[j] = h__[j - 1] / 9.f;
    }
    return 0;
} /* qromo_ */

/* Subroutine */int polint_(sed_real *xa, sed_real *ya, int *n, sed_real *x,
        sed_real *y, sed_real *dy) {
    /* System generated locals */
    int i__1, i__2;
    sed_real d__1;


    /* Local variables */
    static sed_real c__[20], d__[20];
    static int i__, m;
    static sed_real w, ho, hp;
    static int ns;
    static sed_real dif, den, dift;

    /* ..given arrays xa and ya of length n and a value x, this routine returns a */
    /* ..value y and an error estimate dy. if p(x) is the polynomial of degree n-1 */
    /* ..such that ya = p(xa) ya then the returned value is y = p(x) */
    /* ..declare */
    /* ..find the index ns of the closest table entry; initialize the c and d tables */
    /* Parameter adjustments */
    --ya;
    --xa;

    /* Function Body */
    ns = 1;
    dif = (d__1 = *x - xa[1], fabs(static_cast<double>(d__1)));
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        dift = (d__1 = *x - xa[i__], fabs(static_cast<double>(d__1)));
        if (dift < dif) {
            ns = i__;
            dif = dift;
        }
        c__[i__ - 1] = ya[i__];
        d__[i__ - 1] = ya[i__];
    }
    /* ..first guess for y */
    *y = ya[ns];
    /* ..for each column of the table, loop over the c's and d's and update them */
    --ns;
    i__1 = *n - 1;
    for (m = 1; m <= i__1; ++m) {
        i__2 = *n - m;
        for (i__ = 1; i__ <= i__2; ++i__) {
            ho = xa[i__] - *x;
            hp = xa[i__ + m] - *x;
            w = c__[i__] - d__[i__ - 1];
            den = ho - hp;
            if (den == 0.f) {
    //			s_stop(" 2 xa entries are the same in polint", (ftnlen) 36);
            }
            den = w / den;
            d__[i__ - 1] = hp * den;
            c__[i__ - 1] = ho * den;
        }
        /* ..after each column is completed, decide which correction c or d, to add */
        /* ..to the accumulating value of y, that is, which path to take in the table */
        /* ..by forking up or down. ns is updated as we go to keep track of where we */
        /* ..are. the last dy added is the error indicator. */
        if (ns << 1 < *n - m) {
            *dy = c__[ns];
        } else {
            *dy = d__[ns - 1];
            --ns;
        }
        *y += *dy;
    }
    return 0;
} /* polint_ */

sed_real zeroin_(sed_real *ax, sed_real *bx, D_fp f, sed_real *tol) {
    /* System generated locals */
    sed_real ret_val, d__1;
    ret_val = 0;

    /* Local variables */
    static sed_real a, b, c__, d__, e, p, q, r__, s, fa, fb, fc, xm, eps, tol1;

    /* ----------------------------------------------------------------------- */

    /* This subroutine solves for a zero of the function  f(x)  in the */
    /* interval ax,bx. */

    /*  input.. */

    /*  ax     left endpoint of initial interval */
    /*  bx     right endpoint of initial interval */
    /*  f      function subprogram which evaluates f(x) for any x in */
    /*         the interval  ax,bx */
    /*  tol    desired length of the interval of uncertainty of the */
    /*         final result ( .ge. 0.0) */

    /*  output.. */

    /*  zeroin abcissa approximating a zero of  f  in the interval ax,bx */

    /*      it is assumed  that   f(ax)   and   f(bx)   have  opposite  signs */
    /*  without  a  check.  zeroin  returns a zero  x  in the given interval */
    /*  ax,bx  to within a tolerance  4*macheps*fabs(x) + tol, where macheps */
    /*  is the relative machine precision. */
    /*      this function subprogram is a slightly  modified  translation  of */
    /*  the algol 60 procedure  zero  given in  richard brent, algorithms for */
    /*  minimization without derivatives, prentice - hall, inc. (1973). */

    /* ----------------------------------------------------------------------- */
    /* .... call list variables */

    /* ---------------------------------------------------------------------- */

    /*  compute eps, the relative machine precision */

    eps = 1.f;
    L10: eps /= 2.f;
    tol1 = eps + 1.f;
    if (tol1 > 1.f) {
        goto L10;
    }

    /* initialization */

    a = *ax;
    b = *bx;
    fa = (*f)(&a);
    fb = (*f)(&b);

    /* begin step */

    L20: c__ = a;
    fc = fa;
    d__ = b - a;
    e = d__;
    L30: if (fabs(static_cast<double>(fc)) >= fabs(static_cast<double>(fb))) {
        goto L40;
    }
    a = b;
    b = c__;
    c__ = a;
    fa = fb;
    fb = fc;
    fc = fa;

    /* convergence test */

    L40: tol1 = eps * 2.f * fabs(static_cast<double>(b)) + *tol * .5f;
    xm = (c__ - b) * .5f;
    if (fabs(static_cast<double>(xm)) <= tol1) {
        goto L90;
    }
    if (fb == 0.f) {
        goto L90;
    }

    /* is bisection necessary? */

    if (fabs(static_cast<double>(e)) < tol1) {
        goto L70;
    }
    if (fabs(static_cast<double>(fa)) <= fabs(static_cast<double>(fb))) {
        goto L70;
    }

    /* is quadratic interpolation possible? */

    if (a != c__) {
        goto L50;
    }

    /* linear interpolation */

    s = fb / fa;
    p = xm * 2.f * s;
    q = 1.f - s;
    goto L60;

    /* inverse quadratic interpolation */

    L50: q = fa / fc;
    r__ = fb / fc;
    s = fb / fa;
    p = s * (xm * 2.f * q * (q - r__) - (b - a) * (r__ - 1.f));
    q = (q - 1.f) * (r__ - 1.f) * (s - 1.f);

    /* adjust signs */

    L60: if (p > 0.f) {
        q = -q;
    }
    p = fabs(static_cast<double>(p));

    /* is interpolation acceptable? */

    if (p * 2.f >= xm * 3.f * q - (d__1 = tol1 * q, fabs(static_cast<double>(d__1)))) {
        goto L70;
    }
    if (p >= (d__1 = e * .5f * q, fabs(static_cast<double>(d__1)))) {
        goto L70;
    }
    e = d__;
    d__ = p / q;
    goto L80;

    /* bisection */

    L70: d__ = xm;
    e = d__;

    /* complete step */

    L80: a = b;
    fa = fb;
    if (fabs(static_cast<double>(d__)) > tol1) {
        b += d__;
    }
    if (fabs(static_cast<double>(d__)) <= tol1) {
        b += d_sign(&tol1, &xm);
    }
    fb = (*f)(&b);
    if (fb * (fc / fabs(static_cast<double>(fc))) > 0.f) {
        goto L20;
    }
    goto L30;

    /* done */

    L90: ret_val = b;
    return ret_val;
} /* zeroin_ */



#include <functional>
#include <mutex>

#ifndef NO_HPX
#include <hpx/lcos/local/spinlock.hpp>
using mutex_type = hpx::lcos::local::spinlock;
#else
#include <unordered_map>
#include <memory>
#include <cassert>
using mutex_type = std::mutex;
#endif

namespace sedov {

void solution(double time, double r, double rmax, double& d, double& v, double& p, int ndim) {
	int nstep = 10000;
	constexpr int bw = 2;
	using function_type = std::function<void(double,double&,double&,double&)>;
	using map_type = std::unordered_map<double,std::shared_ptr<function_type>>;

	static map_type map;
	static mutex_type mutex;


	sed_real rho0 = 1.0;
	sed_real vel0 = 0.0;
	sed_real ener0 = 0.0;
	sed_real pres0 = 0.0;
	sed_real cs0 = 0.0;
	sed_real gamma = 7.0/5.0;
	sed_real omega = 0.0;
	sed_real eblast = 1.0;
	sed_real xgeom = sed_real(ndim);

	std::vector<sed_real> xpos(nstep+2*bw);
	std::vector<sed_real> den(nstep+2*bw);
	std::vector<sed_real> ener(nstep+2*bw);
	std::vector<sed_real> pres(nstep+2*bw);
	std::vector<sed_real> vel(nstep+2*bw);
	std::vector<sed_real> cs(nstep+2*bw);

	std::vector<double> den1(nstep+2*bw);
	std::vector<double> pres1(nstep+2*bw);
	std::vector<double> vel1(nstep+2*bw);

	std::shared_ptr<function_type> ptr;

	for( int i = 0; i < nstep + 2*bw; i++) {
		xpos[i] = (i - bw + 0.5)*rmax/(nstep);
	}
	nstep += bw;

	std::unique_lock<mutex_type> lock(mutex);
	auto iter = map.find(time);
	if (iter == map.end()) {
		sed_real sed_time = time;
		printf( "Computing sedov solution\n");
		sed_1d__(&sed_time, &nstep, xpos.data() + bw, &eblast, &omega, &xgeom, &rho0,
				&vel0, &ener0, &pres0, &cs0, &gamma, den.data() + bw, ener.data() + bw,
				pres.data() + bw, vel.data() + bw, cs.data() + bw);

		xpos[0] = -xpos[3];
		den[0] = den[3];
		ener[0] = ener[3];
		pres[0] = pres[3];
		vel[0] = -vel[3];
		cs[0] = cs[3];

		xpos[1] = -xpos[2];
		den[1] = den[2];
		ener[1] = ener[2];
		pres[1] = pres[2];
		vel[1] = -vel[2];
		cs[1] = cs[2];

#if defined(OCTOTIGER_HAVE_BOOST_MULTIPRECISION)
		std::transform(den.begin(), den.end(), den1.begin(),
			[](sed_real v) { return v.convert_to<double>(); });
		std::transform(vel.begin(), vel.end(), vel1.begin(),
			[](sed_real v) { return v.convert_to<double>(); });
		std::transform(pres.begin(), pres.end(), pres1.begin(),
			[](sed_real v) { return v.convert_to<double>(); });
#else
		std::copy(den.begin(), den.end(), den1.begin());
		std::copy(vel.begin(), vel.end(), vel1.begin());
		std::copy(pres.begin(), pres.end(), pres1.begin());
#endif

		function_type func = [nstep,rmax,den1,pres1,vel1,bw](double r, double& d, double& v, double & p) {
			double dr = rmax / (nstep);
			std::array<int,4> i;
			i[1] = (r + (bw - 0.5)*dr) / dr;
			i[0] = i[1] - 1;
			i[2] = i[1] + 1;
			i[3] = i[1] + 2;
			double r0 = (r - (i[1]-bw + 0.5)*dr)/dr;
	//		printf( "%i %e\n", i[0], r, dr );
			assert( i[0] >= 0 );
			assert( i[3] < int(vel1.size()));
			const auto interp = [r0,i](const std::vector<double>& data) {
				double sum = 0.0;
				sum += (-0.5 * data[i[0]] + 1.5 * data[i[1]] - 1.5 * data[i[2]] + 0.5 * data[i[3]]) * r0 * r0 * r0;
				sum += (+1.0 * data[i[0]] - 2.5 * data[i[1]] + 2.0 * data[i[2]] - 0.5 * data[i[3]]) * r0 * r0;
				sum += (-0.5 * data[i[0]]                   +  0.5 * data[i[2]]) * r0;
				sum += data[i[1]];
				return sum;
			};

			d = interp(den1);
			v = interp(vel1);
			p = interp(pres1);

		};

		ptr = std::make_shared<function_type>(std::move(func));
		map[time] = ptr;
		lock.unlock();
	} else {
		lock.unlock();
		ptr = iter->second;
	}

	const auto& func = *(ptr);

	func(r, d, v, p);
}

}
