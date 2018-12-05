
#include "../../grid.hpp"

std::vector<real> marshak_wave(real x, real y, real z, real dx) {
	std::vector<real> u(opts().n_fields);
	real e;
	if( x > 0 ) {
		e = u[rho_i] = u[spc_i] = 1.0;
	} else {
		e = u[rho_i] = u[spc_i] = 1.0e-20;
	}
	u[egas_i] = e;
	u[tau_i] = std::pow(e,grid::get_fgamma());
	return u;

}

std::vector<real> marshak_wave_analytic(real x, real y, real z, real t) {


}
