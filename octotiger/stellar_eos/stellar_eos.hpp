/*
 * stellar_eos.hpp
 *
 *  Created on: May 3, 2020
 *      Author: dmarce1
 */

#ifndef OCTOTIGER_STELLAR_EOS_HPP_
#define OCTOTIGER_STELLAR_EOS_HPP_

#include <array>
#include <utility>

class stellar_eos {
protected:
	double g_to_code;
	double s_to_code;
	double cm_to_code;
	double amu;
	double kb;
	double fgamma;
public:
	stellar_eos();
	virtual void set_units(double g, double cm, double s);
	void set_fgamma(double fg);
	virtual double pressure_from_energy(double rho, double ene, double abar, double zbar) = 0;
	virtual std::pair<double, double> pressure_and_soundspeed(double rho, double ene, double abar, double zbar) = 0;
	virtual double T_from_energy(double rho, double ene, double abar, double zbar) = 0;
	virtual double kappa_R(double rho, double T, double X, double Z);
	virtual double kappa_p(double rho, double T, double X, double Z);
	virtual double B_p(double rho, double T);
};

#endif /* OCTOTIGER_STELLAR_EOS_HPP_ */
