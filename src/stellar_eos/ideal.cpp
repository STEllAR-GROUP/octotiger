#include "octotiger/stellar_eos/ideal_eos.hpp"

#include <cmath>

double ideal_eos::pressure_from_energy(double rho, double ene, double abar, double zbar) {
	return (fgamma - 1.0) * ene;
}

std::pair<double, double> ideal_eos::pressure_and_soundspeed(double rho, double ene, double abar, double zbar) {
	std::pair<double, double> rc;
	rc.first = pressure_from_energy(rho, ene, abar, zbar);
	rc.second = std::sqrt(rc.first * fgamma / rho);
	return rc;

}

double ideal_eos::T_from_energy(double rho, double ene, double abar, double zbar) {
	return (fgamma - 1) * ene * amu * abar / (rho * kb * (zbar + 1));
}
