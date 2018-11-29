#include "libeos.hpp"

void eos::set_eos_type(eos_type type, double fgamma) {

	type_ = type;
	fgamma_ = fgamma;

}

void eos::set_units(double cm, double g, double s, double K) {

	cm_ = cm;
	g_ = g;
	s_ = s;
	K_ = K;

	// pressure is g / (cm s^2)
	A_ = Acgs_ / (g_ / (cm_ * s_ * s_));
	// density is g / cm^3
	B_ = Bcgs_ / (g_ / (cm_ * cm_ * cm_));

}

eos::eos_type eos::type_ = IDEAL;
double eos::fgamma_ = 5.0 / 3.0;
double eos::cm_ = 1.0;
double eos::g_ = 1.0;
double eos::s_ = 1.0;
double eos::K_ = 1.0;
const double eos::Acgs_ = 6.00228e+22;
const double eos::Bcgs_ = 2.0 * 9.81011e+5;
double eos::A_ = eos::Acgs_;
double eos::B_ = eos::Bcgs_;
