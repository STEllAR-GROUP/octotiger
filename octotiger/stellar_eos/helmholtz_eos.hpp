#pragma once


#include "octotiger/stellar_eos/stellar_eos.hpp"

class helm_eos: public stellar_eos {
private:

	typedef struct {
		double p, e, cs, cv, abar, zbar, rho, T, s;
	} eos_t;

	double erg_to_code, dyne_to_code;

	static constexpr int IMAX = 1081;
	static constexpr int JMAX = 401;
	static constexpr double tlo = 3.0;
	static constexpr double thi = 13.0;
	static constexpr double tstp = (thi - tlo) / double(JMAX - 1);
	static constexpr double tstpi = 1.0 / tstp;
	static constexpr double dlo = -12.0;
	static constexpr double dhi = 15.0;
	static constexpr double dstp = (dhi - dlo) / double(IMAX - 1);
	static constexpr double dstpi = 1.0 / dstp;

	std::array<double, IMAX> d, dd_sav, dd2_sav, ddi_sav, dd2i_sav, dd3i_sav;
	std::array<double, JMAX> t, dt_sav, dt2_sav, dti_sav, dt2i_sav, dt3i_sav;
	std::array<std::array<double, JMAX>, IMAX> f, fd, ft, fdd, ftt, fdt, fddt, fdtt, fddtt;

	void helmholtz_eos(eos_t*);
	void compute_T(eos_t*);
	void read_helm_table();

	void eos_to_code(eos_t *eos);
	void eos_from_code(eos_t *eos);

public:
	helm_eos();
	double pressure_from_energy(double rho, double ene, double abar, double zbar);
	std::pair<double, double> pressure_and_soundspeed(double rho, double ene, double abar, double zbar);
	double T_from_energy(double rho, double ene, double abar, double zbar);
	virtual void set_units(double g, double cm, double s);
};
