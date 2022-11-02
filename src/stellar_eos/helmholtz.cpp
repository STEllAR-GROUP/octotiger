#include "octotiger/stellar_eos/helmholtz_eos.hpp"

#include <algorithm>
#include <cstdio>
#include <cmath>

double helm_eos::pressure_from_energy(double rho, double ene, double abar, double zbar) {
	eos_t eos;
	eos.rho = rho;
	eos.abar = abar;
	eos.zbar = zbar;
	eos.e = ene / rho;
	compute_T(&eos);
	return eos.p;
}

std::pair<double, double> helm_eos::pressure_and_soundspeed(double rho, double ene, double abar, double zbar) {
	std::pair<double, double> rc;
	eos_t eos;
	eos.rho = rho;
	eos.abar = abar;
	eos.zbar = zbar;
	eos.e = ene / rho;
	compute_T(&eos);
	rc.first = eos.p;
	rc.second = eos.cs;
	return rc;
}

double helm_eos::T_from_energy(double rho, double ene, double abar, double zbar) {
	eos_t eos;
	eos.rho = rho;
	eos.abar = abar;
	eos.zbar = zbar;
	eos.e = ene / rho;
	compute_T(&eos);
	return eos.T;
}

helm_eos::helm_eos() {
	read_helm_table();
}

void helm_eos::read_helm_table() {
	int i, j;
	double tsav, dsav, dth, dt2, dti, dt2i, dt3i, dd, dd2, ddi, dd2i, dd3i;
	FILE *fp;
	fp = fopen("helmholtz.table.dat", "rt");
	if (fp == NULL) {
		printf("Unable to load helmholtz.table.dat\n");
		abort();
	}

// read the helmholtz free energy and its derivatives
	for (j = 0; j < JMAX; j++) {
		tsav = tlo + j * tstp;
		t[j] = pow(10.0, tsav);
		for (i = 0; i < IMAX; i++) {
			dsav = dlo + i * dstp;
			d[i] = pow(10.0, dsav);
			fscanf(fp, "%le %le %le %le %le %le %le %le %le\n", &f[i][j], &fd[i][j], &ft[i][j], &fdd[i][j], &ftt[i][j], &fdt[i][j], &fddt[i][j], &fdtt[i][j],
					&fddtt[i][j]);

			//		fddtt[i][j] = fddt[i][j] = fdtt[i][j] = 0.0;
		}
	}

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

void helm_eos::set_units(double g, double cm, double s) {
//	printf("Setting cgs units cm = %e g = %e s = %e K = %e\n", cm, g, s, K);
	stellar_eos::set_units(g, cm, s);
	erg_to_code = g * cm * cm / s / s;
	dyne_to_code = g * cm / s / s;
}

void helm_eos::eos_to_code(eos_t *eos) {

	eos->p /= dyne_to_code / pow(cm_to_code, 2);
	eos->e /= erg_to_code / g_to_code;
	eos->cs /= cm_to_code / s_to_code;
	eos->cv /= erg_to_code / g_to_code;
	eos->rho /= g_to_code / pow(cm_to_code, 3);
}

void helm_eos::eos_from_code(eos_t *eos) {

	eos->p *= dyne_to_code / pow(cm_to_code, 2);
	eos->e *= erg_to_code / g_to_code;
	eos->cs *= cm_to_code / s_to_code;
	eos->cv *= erg_to_code / g_to_code;
	eos->rho *= g_to_code / pow(cm_to_code, 3);
}

void helm_eos::helmholtz_eos(eos_t *eos) {

	/* This code was translated from helmholtz.f90 on Timmes page cococubed.com */

	constexpr auto kerg = 1.380650424e-16;
	constexpr auto avo = 6.0221417930e23;
	std::array<double, 36> fi;
//	int iat, jat;
//	double den, temp, abar, zbar, ytot1, ye, x, deni, xni, dxnidd, dpepdt, dpepdd, deepdt, dsepdt, dpiondd, dpiondt, deiondt, kt, pion, eion, pele, eele, sele,
//			dpgasdd, dpgasdt, free, df_d, df_t, df_tt, df_dt, df_dd, xt, xd, mxt, mxd;

	static const auto psi0 = [](double z) {
		return (z * z * z * (z * (-6.0 * z + 15.0) - 10.0) + 1.0);
	};

	static const auto dpsi0 = [](double z) {
		return (z * z * (z * (-30.0 * z + 60.0) - 30.0));
	};

	static const auto ddpsi0 = [](double z) {
		return (z * (z * (-120.0 * z + 180.0) - 60.0));
	};

	static const auto psi1 = [](double z) {
		return (z * (z * z * (z * (-3.0 * z + 8.0) - 6.0) + 1.0));
	};

	static const auto dpsi1 = [](double z) {
		return (z * z * (z * (-15.0 * z + 32.0) - 18.0) + 1.0);
	};

	static const auto ddpsi1 = [](double z) {
		return (z * (z * (-60.0 * z + 96.0) - 36.0));
	};

	static const auto psi2 = [](double z) {
		return (0.5 * z * z * (z * (z * (-z + 3.0) - 3.0) + 1.0));
	};

	static const auto dpsi2 = [](double z) {
		return (0.5 * z * (z * (z * (-5.0 * z + 12.0) - 9.0) + 2.0));
	};

	static const auto ddpsi2 = [](double z) {
		return (0.5 * (z * (z * (-20.0 * z + 36.0) - 18.0) + 2.0));
	};

	const auto h5 = [&fi](const double w0t, const double w1t, const double w2t, const double w0mt, const double w1mt, const double w2mt,
			const double w0d, const double w1d, const double w2d, const double w0md, const double w1md, const double w2md) {
		return fi[0] * w0d * w0t + fi[1] * w0md * w0t + fi[2] * w0d * w0mt + fi[3] * w0md * w0mt + fi[4] * w0d * w1t + fi[5] * w0md * w1t + fi[6] * w0d * w1mt
				+ fi[7] * w0md * w1mt + fi[8] * w0d * w2t + fi[9] * w0md * w2t + fi[10] * w0d * w2mt + fi[11] * w0md * w2mt + fi[12] * w1d * w0t
				+ fi[13] * w1md * w0t + fi[14] * w1d * w0mt + fi[15] * w1md * w0mt + fi[16] * w2d * w0t + fi[17] * w2md * w0t + fi[18] * w2d * w0mt
				+ fi[19] * w2md * w0mt + fi[20] * w1d * w1t + fi[21] * w1md * w1t + fi[22] * w1d * w1mt + fi[23] * w1md * w1mt + fi[24] * w2d * w1t
				+ fi[25] * w2md * w1t + fi[26] * w2d * w1mt + fi[27] * w2md * w1mt + fi[28] * w1d * w2t + fi[29] * w1md * w2t + fi[30] * w1d * w2mt
				+ fi[31] * w1md * w2mt + fi[32] * w2d * w2t + fi[33] * w2md * w2t + fi[34] * w2d * w2mt + fi[35] * w2md * w2mt;
	};

	eos_from_code(eos);

	const auto &den = eos->rho;
	const auto temp = std::max(eos->T, 3e3);
	const auto &abar = eos->abar;
	const auto &zbar = eos->zbar;

	const auto ytot1 = 1.0 / abar;
	const auto ye = std::max(1.0e-16, ytot1 * zbar);

// initialize
	const auto deni = 1.0 / den;
	const auto kt = kerg * temp;

// ion section:
	const auto xni = avo * ytot1 * den;
	const auto dxnidd = avo * ytot1;

	const auto pion = xni * kt;
	const auto dpiondd = dxnidd * kt;
	const auto dpiondt = xni * kerg;

	const auto eion = 1.5 * pion * deni;
	const auto deiondt = 1.5 * dpiondt * deni;

	const auto din = ye * den;

	auto jat = int((log10(temp) - tlo) * tstpi) + 1;
	jat = std::max(1, std::min(jat, JMAX - 1));
	auto iat = int((log10(din) - dlo) * dstpi) + 1;
	iat = std::max(1, std::min(iat, IMAX - 1));
	--jat;
	--iat;
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

	const auto xt = std::max((temp - t[jat]) * dti_sav[jat], 0.0);
	const auto xd = std::max((din - d[iat]) * ddi_sav[iat], 0.0);
	const auto mxt = 1.0 - xt;
	const auto mxd = 1.0 - xd;

	const auto si0t = psi0(xt);
	const auto si1t = psi1(xt) * dt_sav[jat];
	const auto si2t = psi2(xt) * dt2_sav[jat];
	const auto si0mt = psi0(mxt);
	const auto si1mt = -psi1(mxt) * dt_sav[jat];
	const auto si2mt = psi2(mxt) * dt2_sav[jat];
	const auto dsi0t = dpsi0(xt) * dti_sav[jat];
	const auto dsi1t = dpsi1(xt);
	const auto dsi2t = dpsi2(xt) * dt_sav[jat];
	const auto dsi0mt = -dpsi0(mxt) * dti_sav[jat];
	const auto dsi1mt = dpsi1(mxt);
	const auto dsi2mt = -dpsi2(mxt) * dt_sav[jat];
	const auto si0d = psi0(xd);
	const auto si1d = psi1(xd) * dd_sav[iat];
	const auto si2d = psi2(xd) * dd2_sav[iat];
	const auto si0md = psi0(mxd);
	const auto si1md = -psi1(mxd) * dd_sav[iat];
	const auto si2md = psi2(mxd) * dd2_sav[iat];
	const auto ddsi0t = ddpsi0(xt) * dt2i_sav[jat];
	const auto ddsi1t = ddpsi1(xt) * dti_sav[jat];
	const auto ddsi2t = ddpsi2(xt);
	const auto ddsi0mt = ddpsi0(mxt) * dt2i_sav[jat];
	const auto ddsi1mt = -ddpsi1(mxt) * dti_sav[jat];
	const auto ddsi2mt = ddpsi2(mxt);
	const auto dsi0d = dpsi0(xd) * ddi_sav[iat];
	const auto dsi1d = dpsi1(xd);
	const auto dsi2d = dpsi2(xd) * dd_sav[iat];
	const auto dsi0md = -dpsi0(mxd) * ddi_sav[iat];
	const auto dsi1md = dpsi1(mxd);
	const auto dsi2md = -dpsi2(mxd) * dd_sav[iat];
	const auto ddsi0d = ddpsi0(xd) * dd2i_sav[iat];
	const auto ddsi1d = ddpsi1(xd) * ddi_sav[iat];
	const auto ddsi2d = ddpsi2(xd);
	const auto ddsi0md = ddpsi0(mxd) * dd2i_sav[iat];
	const auto ddsi1md = -ddpsi1(mxd) * ddi_sav[iat];
	const auto ddsi2md = ddpsi2(mxd);

	const auto df_d = h5(si0t, si1t, si2t, si0mt, si1mt, si2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md);
	const auto df_dd = h5(si0t, si1t, si2t, si0mt, si1mt, si2mt, ddsi0d, ddsi1d, ddsi2d, ddsi0md, ddsi1md, ddsi2md);
	const auto df_dt = h5(dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md);
	auto x = din * din;
	const auto pele = x * df_d;
	const auto dpepdd = ye * (x * df_dd + 2.0 * din * df_d);
	const auto dpepdt = x * df_dt;
	const auto free = h5(si0t, si1t, si2t, si0mt, si1mt, si2mt, si0d, si1d, si2d, si0md, si1md, si2md);
	const auto df_t = h5(dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, si0d, si1d, si2d, si0md, si1md, si2md);
	const auto df_tt = h5(ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt, si0d, si1d, si2d, si0md, si1md, si2md);
	x = ye * ye;
	const auto sele = -df_t * ye;
	const auto dsepdt = -df_tt * ye;
	const auto eele = ye * free + temp * sele;
	const auto deepdt = temp * dsepdt;
	eos->p = pele + pion;
	eos->e = eele + eion;
	eos->cv = deiondt + deepdt;
	const auto dpgasdd = dpiondd + dpepdd;
	const auto dpgasdt = dpiondt + dpepdt;
	x = (eos->p * deni * deni) * (dpgasdt / eos->cv) + dpgasdd;
	if (x < 0.0) {
		eos->cs = 0.0;
	} else {
		eos->cs = sqrt(x);
	}
	eos_to_code(eos);

}

void helm_eos::compute_T(eos_t *eos) {

	const double table_tmin = 3000;
	const double e0 = eos->e;
	eos->T = table_tmin;
	int iters = 0;
	helmholtz_eos(eos);
	iters++;
	double ed = eos->e * eos->rho;
	double pd = eos->p;
	double cs = eos->cs;
	const auto c0 = amu / kb;
	eos->T = (2.0 / 3.0) * (e0 - ed / eos->rho) * eos->abar / (eos->zbar + 1) * c0 + table_tmin;
	double tmin = table_tmin;
	double tmax = 1e+15;
	if (eos->T <= table_tmin) {
		eos->p = pd;
		eos->cs = cs;
	} else {
		double f;
		do {
			helmholtz_eos(eos);
			f = eos->e - e0;
			if (f < 0) {
				tmin = eos->T;
			} else {
				tmax = eos->T;
			}
			auto dT = -f / eos->cv;
			if (iters > 90) {
				printf("%i %e %e %e %e %e\n", iters, f / e0, eos->T, eos->cv, eos->rho, dT);
			}
			double oldT = eos->T;
			double newT = eos->T + dT;
			constexpr auto a = 0.9;
			newT = std::min(newT, (1 - a) * oldT + a * tmax);
			newT = std::max(newT, (1 - a) * oldT + a * tmin);
			eos->T = newT;
			iters++;
			if (iters > 1000) {
				printf("Helmholtz solver failed to converge\n");
				abort();
			}
		} while (std::abs(f / e0) >= 1.0e-6);
	}
}

