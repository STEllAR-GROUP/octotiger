#include "bibi.hpp"

real toler = 1.0e-6;

#define PRINT( a ) printf( #a " = %e\n", a )

real bibi_polytrope::K_C() {
	return P0 * pow(rho_C, -1.0 - 1.0 / n_C);
}
real bibi_polytrope::K_E() {
	return P0 * pow(rho_E, -1.0 - 1.0 / n_E);
}

bibi_polytrope::bibi_polytrope(real m, real r, real nc, real ne, real ac, real ae) {
//	ne = nc = 1.0;
	//ae = ac;
	n_E = ne;
	n_C = nc;
	theta_E = pow(ae, 1.0 / n_E);
	theta_C = pow(ac, 1.0 / n_C);
	rho0 = 1.0;
	P0 = 1.0;
	//solve();
	solve_for_mass_and_radius(m, r);
	rho_C = ac * rho0;
	rho_E = ae * rho0;

	print();/*
	 PRINT(mass);
	 PRINT(radius);
	 PRINT(theta_C);
	 PRINT(theta_E);
	 PRINT(P0);
	 PRINT(rho0);
	 PRINT(rho_C);
	 PRINT(rho_E);
	 PRINT(K_C());
	 PRINT(K_E());
	 abort();*/

}

void bibi_polytrope::solve_for_mass(real target_mass) {
	real lo = 1.0e-10;
	real hi = 1.0e+10;
	do {
		const real mid = sqrt(lo * hi);
		P0 = mid;
		solve();
		if (mass > target_mass) {
			hi = mid;
		} else {
			lo = mid;
		}
		//		printf( "%e %e %e %e\n", lo, mid, hi, mass);
	} while (fabs(log(lo) - log(hi)) > toler);
}

void bibi_polytrope::solve_for_radius(real target_radius) {
	real lo = 1.0e-10;
	real hi = 1.0e+10;
	while (fabs(log(lo) - log(hi)) > toler) {
		const real mid = sqrt(lo * hi);
		rho0 = mid;
		solve();
		if (radius < target_radius) {
			hi = mid;
		} else {
			lo = mid;
		}
	}
}

void bibi_polytrope::solve_for_mass_and_radius(real tmass, real tradius) {
	do {
		solve_for_mass(tmass);
		solve_for_radius(tradius);
		//	printf("%e %e %e %e \n", mass, tmass, radius, tradius);
	} while (fabs(log(tmass) - log(mass)) > toler);
}

void bibi_polytrope::solve() {
	real m = 0.0;
	real alpha = sqrt(P0 / 4.0 / M_PI / rho0 / rho0);
	if (pow(theta_C, n_C) < pow(theta_E, n_E)) {
		printf("Density jump wrong\n");
		abort();
	}
	const real dx = 1.0e-2;

	real x = 0.0, y = 1.0 - theta_C + theta_E, z = 0.0;
	bool bump = false;
	while (y > 0.0) {
		//		printf("%e %e %e %e\n", x, y, z, y_to_rho(y));
		const real dy1 = dy_dx(x, y, z) * dx;
		const real dz1 = dz_dx(x, y, z) * dx;
		const real dm1 = dm_dx(x, y, z) * dx;
		const real dy2 = dy_dx(x + dx, y + dy1, z + dz1) * dx;
		const real dz2 = dz_dx(x + dx, y + dy1, z + dz1) * dx;
		const real dm2 = dm_dx(x + dx, y + dy1, z + dz1) * dx;
		x += dx;
		y += (dy1 + dy2) / 2.0;
		z += (dz1 + dz2) / 2.0;
		m += (dm1 + dm2) / 2.0;
		if (!bump && y < theta_E) {
			const real c1 = pow(theta_E, 1.0 + n_E) / (1.0 + n_E);
			const real c2 = pow(theta_C, 1.0 + n_C) / (1.0 + n_C);
			z *= (c1 / c2);
			bump = true;
		}
	}
	mass = m;
	radius = alpha * x;
}

bool bibi_polytrope::solve_at(real r, real& den, real& pre, real& menc) {
	if (r >= radius) {
		den = 0.0;
		pre = 0.0;
		return false;
	} else if (r == 0.0) {
		menc = 0.0;
	}
	real alpha = sqrt(P0 / 4.0 / M_PI / rho0 / rho0);
	const real xmax = r / alpha;
	real dx = std::min(radius / 128.0 / alpha, xmax);
	menc = 0.0;
	real x = 0.0, y = 1.0 - theta_C + theta_E, z = 0.0;
	bool bump = false;
	while (y > 0.0 && x < xmax) {
		const real dy1 = dy_dx(x, y, z) * dx;
		const real dz1 = dz_dx(x, y, z) * dx;
		const real dm1 = dm_dx(x, y, z) * dx;
		const real dy2 = dy_dx(x + dx, y + dy1, z + dz1) * dx;
		const real dz2 = dz_dx(x + dx, y + dy1, z + dz1) * dx;
		const real dm2 = dm_dx(x + dx, y + dy1, z + dz1) * dx;
		x += dx;
		y += (dy1 + dy2) / 2.0;
		z += (dz1 + dz2) / 2.0;
		menc += (dm1 + dm2) / 2.0;
		dx = std::min(dx, xmax - x);
		if (!bump && y < theta_E) {
			const real c1 = pow(theta_E, 1.0 + n_E) / (1.0 + n_E);
			const real c2 = pow(theta_C, 1.0 + n_C) / (1.0 + n_C);
			z *= (c1 / c2);
			bump = true;
		}
	}
	den = y_to_rho(y);
	if (den < rho_E) {
//			pre = K_E() * pow(den, 1.0 + 1.0 / n_E);
		pre = P0 * pow(den / rho_E, 1.0 + 1.0 / n_E);
	} else if (den > rho_C) {
		//	pre = K_C() * pow(den, 1.0 + 1.0 / n_C);
		pre = P0 * pow(den / rho_C, 1.0 + 1.0 / n_C);
	} else {
		pre = P0;
	}
	return y > theta_E;
}

void bibi_polytrope::print() {
	const int N = 1000;
	FILE* fp = fopen("bibi.dat", "wt");
	for (int i = 0; i != N; ++i) {
		const real dr = (1.0 / N) * radius;
		const real r = (i + 0.5) * dr;
		real den, pre, menc, dpdr;
		real denr, prer, mencr;
		real denl, prel, mencl;
		solve_at(r, den, pre, menc);
		solve_at(r - dr / 2.0, denl, prel, mencl);
		solve_at(r + dr / 2.0, denr, prer, mencr);
		dpdr = (prer - prel) / dr;
		const real fg = den * menc / (r * r);

		fprintf(fp, "%e %e %e %e %e\n", r, den, pre, dpdr, fg);
	}
	fclose(fp);
}
