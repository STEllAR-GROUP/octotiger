#include "../../grid.hpp"
#include <complex>

using complex = std::complex<real>;


constexpr auto one = complex(1,0);
constexpr real eps = 1.0;
constexpr auto root3 = complex(std::sqrt(3),0.0);
constexpr real theta_inc = 1.0;
constexpr real kappa = 1.0;
constexpr real c = 1.0;

complex inverse_laplace(const std::function<complex(const complex&)>& f, real tau) {
	constexpr real dx = 1.0e-3;
	complex result, last_result;
	real x = 0.5 * dx;
	result = std::numeric_limits<real>::max();
	do {
		last_result = result;
		auto sn = complex(1, -x);
		auto sp = complex(1, +x);
		result += std::exp(sp) * f(sp) * dx;
		result += std::exp(sn) * f(sn) * dx;
		x += dx;
	} while (result.real() != last_result.real());
	return result;
}

complex beta(const complex& s) {
	return std::sqrt(s / (s + one) * (one + eps * (s + one)));
}

complex v_func(real x, real tau) {
	auto v = [x]( const complex& s) {
		const auto b = beta(s);
		return root3 * std::exp(-x*b) / (s*(s+one)*(root3+2.0*b));
	};
	return inverse_laplace(v, tau);
}


complex u_func(real x, real tau) {
	auto v = [x]( const complex& s) {
		const auto b = beta(s);
		return root3 * std::exp(-x*b) / (s*(root3+2.0*b));
	};
	return inverse_laplace(v, tau);
}


void solution(real z, real t, real& T_mat, real& T_rad ) {
	const real x = std::sqrt(3) * kappa * z;
	const real tau = eps * kappa / c;
	const auto u = u_func(x, tau);
	const auto v = v_func(x, tau);
	T_mat = std::pow(v.real(),1.0/4.0) * theta_inc;
	T_rad = std::pow(u.real(),1.0/4.0) * theta_inc;
}

std::vector<real> marshak_wave(real x, real y, real z, real dx) {
	std::vector<real> u(opts().n_fields);
	real e;
	if (x > 0) {
		e = u[rho_i] = u[spc_i] = 1.0;
	} else {
		e = u[rho_i] = u[spc_i] = 1.0e-20;
	}
	u[egas_i] = e;
	u[tau_i] = std::pow(e, grid::get_fgamma());
	return u;

}



std::vector<real> marshak_wave_analytic(real x, real y, real z, real t) {

}

#ifdef TESTME
int main() {

}
#endif
