#include "octotiger/stellar_eos/segretain_eos.hpp"
#include <algorithm>
#include <cmath>
;

void segretain_eos::set_units(double g, double cm, double s) {
	stellar_eos::set_units(g, cm, s);
	const auto c = 2.99792458e10 * s / cm;
	const auto me = 9.1093897e-28 / g;
	const auto h = 6.6260755e-27 * s / (g * cm * cm);
	B = 8.0 * M_PI * amu / 3.0 * std::pow(me * c / h, 3);
	A = M_PI * std::pow(me * c, 4) * c / 3 / std::pow(h, 3);
//	abort();
}

std::pair<double, double> segretain_eos::ztwd_pressure_and_energy(double rho, double abar, double zbar) {
	std::pair<double, double> rc;
	const auto mu = abar / zbar;
	const auto x = std::pow(rho / B / mu, 1.0 / 3.0);
	double Pdeg;
	double Edeg;
	if (x < 0.01) {
		const auto x2 = x * x;
		const auto x3 = x2 * x;
		const auto x5 = x3 * x2;
		const auto x7 = x2 * x5;
		const auto x9 = x2 * x7;
		const auto x11 = x2 * x9;
		Pdeg = A * (1.6 * x5 - (4.0 / 7.0) * x7 + (1.0 / 3.0) * x9 - (5.0 / 22.0) * x11);
		Edeg = A * (2.4 * x5 - (3.0 / 7.0) * x7 + (1.0 / 6.0) * x9 - (15.0 / 176.0) * x11);
	} else {
		Pdeg = A * (x * (2 * x * x - 3) * std::sqrt(x * x + 1) + 3 * asinh(x));
		const auto hdeg = 8 * A / (mu * B) * (std::sqrt(1 + x * x) - 1);
		Edeg = rho * hdeg - Pdeg;
	}
	rc.first = Pdeg;
	rc.second = Edeg;
	return rc;
}

double segretain_eos::pressure_from_energy(double rho, double e, double abar, double zbar) {
	const auto tmp = ztwd_pressure_and_energy(rho, abar, zbar);
	const double Pd = tmp.first;
	const double Ed = tmp.second;
	e = std::max(e, Ed);
//	printf( "%e %e %e\n", Pdeg, Edeg, e);
	return Pd + (fgamma - 1.0) * (e - Ed);
}

std::pair<double, double> segretain_eos::pressure_and_soundspeed(double rho, double e, double abar, double zbar) {
	std::pair<double, double> rc;
	const auto mu = abar / zbar;
	const auto tmp = ztwd_pressure_and_energy(rho, abar, zbar);
	const double Pd = tmp.first;
	const double Ed = tmp.second;
	e = std::max(e, Ed);
	const auto x = std::pow(rho / B / mu, 1.0 / 3.0);
	const auto dPddx = 8.0 * A * std::pow(x, 4) / std::sqrt(x * x + 1);
	double dEddx;
	const auto x2 = x * x;
	if (x < 0.01) {
		const auto x4 = x2 * x2;
		const auto x6 = x2 * x4;
		const auto x8 = x2 * x6;
		const auto x10 = x2 * x8;
		dEddx = A * (12.0 * x4 - 3.0 * x6 + 1.5 * x8 - (15.0 / 16.0) * x10);
	} else {
		dEddx = 24.0 * A * x * x * (std::sqrt(x2 + 1) - 1);
	}
	const auto dxdrho = 1.0 / (3 * B * mu * x2);
	const auto dPddrho = dPddx * dxdrho;
	const auto dEddrho = dEddx * dxdrho;
	const auto P = Pd + (fgamma - 1.0) * (e - Ed);
	auto dPdrho = dPddrho + (fgamma - 1) * (e / rho - dEddrho);
	const auto dPdeps = (fgamma - 1) * rho;
	const auto cs2 = dPdrho + P / (rho * rho) * dPdeps;
//	if (cs2 < 0.0) {
//		printf("%e %e %e %e %e %e %e\n", cs2, x, e, Ed, dPdrho, abar, zbar);
//	}
	rc.first = P;
	rc.second = std::sqrt(std::max(cs2, 0.0));
	return rc;
}

double segretain_eos::T_from_energy(double rho, double e, double abar, double zbar) {
	const auto tmp = ztwd_pressure_and_energy(rho, abar, zbar);
	const double Edeg = tmp.second;
	e = std::max(e, Edeg);
	return (fgamma - 1) * (e - Edeg) * abar * amu / (rho * kb * (zbar + 1));
}

