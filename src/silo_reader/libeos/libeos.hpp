#ifndef __LIBEOS__H
#define __LIBEOS__H

#include <iostream>
#include <utility>

struct eos {
	enum eos_type {
		IDEAL, WD
	};
public:
	static void set_eos_type(eos_type type, double fgamma = 5.0 / 3.0);
	static void set_units(double cm, double g, double s, double K);

	template<class T>
	static T pressure(const T& d, const T& e);

	template<class T>
	static T pressure_de(const T& d, const T& e, const T& tau, const T& ek);

	template<class T>
	static T pressure_de(const T& d, const T& e, const T& tau, const T& sx, const T& sy, const T& sz);

private:
	static const double Acgs_;
	static const double Bcgs_;
	static eos_type type_;
	static double fgamma_;
	static double cm_;
	static double g_;
	static double s_;
	static double K_;
	static double A_;
	static double B_;
};

template<class T>
T eos::pressure(const T& d, const T& e) {
	T p;

	if (type_ == IDEAL) {
		p = (fgamma_ - T(1)) * e;
	} else if (type_ == WD) {
		const T x = pow(d / B_, T(1.0 / 3.0));
		p = A_ * (x * (T(2) * x * x - T(3)) * sqrt(x * x + T(1)) + T(3) * asinh(x)) + (fgamma_ - T(1)) * e;
	} else {
		std::cout << "libeos : unknown eos\n";
		throw;
	}

	return std::move(p);
}

template<class T>
T eos::pressure_de(const T& d, const T& egas, const T& tau, const T& ek) {

	T ein = egas - ek;
	if (ein < T(0.001) * egas) {
		ein = pow(tau, fgamma_);
	}

	return pressure(d, ein);

}

template<class T>
T eos::pressure_de(const T& d, const T& egas, const T& tau, const T& sx, const T& sy, const T& sz) {
	const T ek = (sx * sx + sy * sy + sz * sz) / (T(2) * d);
	return pressure_de(d, egas, tau, ek);
}

#endif
