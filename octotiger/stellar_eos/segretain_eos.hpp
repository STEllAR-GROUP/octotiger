#pragma once

#include "octotiger/stellar_eos/stellar_eos.hpp"

class segretain_eos: public stellar_eos {
	double A;
	double B;
	std::pair<double, double> ztwd_pressure_and_energy(double rho, double abar, double zbar);
public:
	double pressure_from_energy(double rho, double ene, double abar, double zbar);
	std::pair<double, double> pressure_and_soundspeed(double rho, double ene, double abar, double zbar);
	double T_from_energy(double rho, double ene, double abar, double zbar);
	virtual void set_units(double g, double cm, double s);
};
