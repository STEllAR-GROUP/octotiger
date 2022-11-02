#pragma once

#include "octotiger/stellar_eos/stellar_eos.hpp"

class ideal_eos: public stellar_eos {
public:
	double pressure_from_energy(double rho, double ene, double abar, double zbar);
	std::pair<double, double> pressure_and_soundspeed(double rho, double ene, double abar, double zbar);
	double T_from_energy(double rho, double ene, double abar, double zbar);
};
