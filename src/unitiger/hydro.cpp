//============================================================================
// Name        : hydro.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "../../octotiger/octotiger/unitiger/unitiger.hpp"
#include "../../octotiger/octotiger/unitiger/hydro.hpp"

#include <functional>

constexpr double vol_weight1d[3] = { 1. / 6., 2. / 3., 1. / 6. };
constexpr double vol_weight2d[9] = { 1. / 36., 1. / 9., 1. / 36., 1. / 9., 4. / 9., 1. / 9., 1. / 36., 1. / 9., 1. / 36. };
constexpr double vol_weight3d[27] = { 1. / 216., 1. / 54., 1. / 216., 1. / 54., 2. / 27., 1. / 54., 1. / 216., 1. / 54., 1. / 216., 1. / 54., 2. / 27., 1.
		/ 54., 2. / 27., 8. / 27., 2. / 27., 1. / 54., 2. / 27., 1. / 54., 1. / 216., 1. / 54., 1. / 216., 1. / 54., 2. / 27., 1. / 54., 1. / 216., 1. / 54., 1.
		/ 216. };

constexpr double filter1d[3][3] = { { 0., 1., 0. }, { -0.5, 0., 0.5 }, { 0.5, -1., 0.5 } };

constexpr double filter2d[9][9] = { { 0., 0., 0., 0., 1., 0., 0., 0., 0. }, { 0., -0.5, 0., 0., 0., 0., 0., 0.5, 0. },
		{ 0., 0.5, 0., 0., -1., 0., 0., 0.5, 0. }, { 0., 0., 0., -0.5, 0., 0.5, 0., 0., 0. }, { 0.25, 0., -0.25, 0., 0., 0., -0.25, 0., 0.25 }, { -0.25, 0.,
				0.25, 0.5, 0., -0.5, -0.25, 0., 0.25 }, { 0., 0., 0., 0.5, -1., 0.5, 0., 0., 0. }, { -0.25, 0.5, -0.25, 0., 0., 0., 0.25, -0.5, 0.25 }, { 0.25,
				-0.5, 0.25, -0.5, 1., -0.5, 0.25, -0.5, 0.25 } };

constexpr double filter3d[27][27] = { { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0.,
		0., 0., -0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0. }, { 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0.,
		0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.5, 0., 0., 0., 0., 0., 0.5, 0., 0., 0.,
		0., 0., 0., 0., 0., 0., 0. }, { 0., 0.25, 0., 0., 0., 0., 0., -0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.25, 0., 0., 0., 0., 0., 0.25, 0. },
		{ 0., -0.25, 0., 0., 0., 0., 0., 0.25, 0., 0., 0.5, 0., 0., 0., 0., 0., -0.5, 0., 0., -0.25, 0., 0., 0., 0., 0., 0.25, 0. }, { 0., 0., 0., 0., 0., 0.,
				0., 0., 0., 0., 0.5, 0., 0., -1., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., -0.25, 0., 0., 0.5, 0., 0., -0.25, 0., 0., 0.,
				0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0., 0., -0.5, 0., 0., 0.25, 0. }, { 0., 0.25, 0., 0., -0.5, 0., 0., 0.25, 0., 0., -0.5, 0., 0., 1., 0.,
				0., -0.5, 0., 0., 0.25, 0., 0., -0.5, 0., 0., 0.25, 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.5, 0., 0.5, 0., 0., 0., 0., 0.,
				0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0.25, 0., -0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.25, 0., 0.25, 0.,
				0., 0. }, { 0., 0., 0., -0.25, 0., 0.25, 0., 0., 0., 0., 0., 0., 0.5, 0., -0.5, 0., 0., 0., 0., 0., 0., -0.25, 0., 0.25, 0., 0., 0. }, { 0., 0.,
				0., 0., 0., 0., 0., 0., 0., 0.25, 0., -0.25, 0., 0., 0., -0.25, 0., 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { -0.125, 0., 0.125, 0., 0., 0.,
				0.125, 0., -0.125, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.125, 0., -0.125, 0., 0., 0., -0.125, 0., 0.125 }, { 0.125, 0., -0.125, 0., 0., 0.,
				-0.125, 0., 0.125, -0.25, 0., 0.25, 0., 0., 0., 0.25, 0., -0.25, 0.125, 0., -0.125, 0., 0., 0., -0.125, 0., 0.125 }, { 0., 0., 0., 0., 0., 0.,
				0., 0., 0., -0.25, 0., 0.25, 0.5, 0., -0.5, -0.25, 0., 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0.125, 0., -0.125, -0.25, 0., 0.25, 0.125,
				0., -0.125, 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.125, 0., 0.125, 0.25, 0., -0.25, -0.125, 0., 0.125 }, { -0.125, 0., 0.125, 0.25, 0., -0.25,
				-0.125, 0., 0.125, 0.25, 0., -0.25, -0.5, 0., 0.5, 0.25, 0., -0.25, -0.125, 0., 0.125, 0.25, 0., -0.25, -0.125, 0., 0.125 }, { 0., 0., 0., 0.,
				0., 0., 0., 0., 0., 0., 0., 0., 0.5, -1., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., -0.25, 0.5, -0.25, 0., 0., 0.,
				0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25, -0.5, 0.25, 0., 0., 0. }, { 0., 0., 0., 0.25, -0.5, 0.25, 0., 0., 0., 0., 0., 0., -0.5,
				1., -0.5, 0., 0., 0., 0., 0., 0., 0.25, -0.5, 0.25, 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.25, 0.5, -0.25, 0., 0., 0., 0.25,
				-0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0.125, -0.25, 0.125, 0., 0., 0., -0.125, 0.25, -0.125, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
				-0.125, 0.25, -0.125, 0., 0., 0., 0.125, -0.25, 0.125 }, { -0.125, 0.25, -0.125, 0., 0., 0., 0.125, -0.25, 0.125, 0.25, -0.5, 0.25, 0., 0., 0.,
				-0.25, 0.5, -0.25, -0.125, 0.25, -0.125, 0., 0., 0., 0.125, -0.25, 0.125 }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25, -0.5, 0.25, -0.5, 1.,
				-0.5, 0.25, -0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { -0.125, 0.25, -0.125, 0.25, -0.5, 0.25, -0.125, 0.25, -0.125, 0., 0., 0., 0.,
				0., 0., 0., 0., 0., 0.125, -0.25, 0.125, -0.25, 0.5, -0.25, 0.125, -0.25, 0.125 }, { 0.125, -0.25, 0.125, -0.25, 0.5, -0.25, 0.125, -0.25,
				0.125, -0.25, 0.5, -0.25, 0.5, -1., 0.5, -0.25, 0.5, -0.25, 0.125, -0.25, 0.125, -0.25, 0.5, -0.25, 0.125, -0.25, 0.125 } };


inline static double limit_this(double a, double b) {
	return std::copysign(std::min(std::abs(a), std::abs(b)), a);
}

double limiter(double ql, double qr, double ll, double lr, double ul, double u0, double ur) {
	const auto M = std::max(qr, ql);
	const auto m = std::min(qr, ql);
	const auto M_p = std::max(lr, qr);
	const auto M_m = std::max(ll, ql);
	const auto m_p = std::min(lr, qr);
	const auto m_m = std::min(ll, ql);
	double theta = 1.0;
	double tmp;
	if (ul < u0 && u0 < ur) {
		tmp = M - lr;
		if (tmp != 0.0) {
			theta = std::min(theta, (M_p - lr) / tmp);
		}
		tmp = m - ll;
		if (tmp != 0.0) {
			theta = std::min(theta, (m_m - ll) / tmp);
		}
	} else if (ul > u0 && u0 > ur) {
		tmp = m - lr;
		if (tmp != 0.0) {
			theta = std::min(theta, (m_p - lr) / tmp);
		}
		tmp = M - ll;
		if (tmp != 0.0) {
			theta = std::min(theta, (M_m - ll) / tmp);
		}
	}
	assert(theta >= 0.0 && theta <= 1.0);
	return theta;
}

