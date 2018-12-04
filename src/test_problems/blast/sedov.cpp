
#include "../../grid.hpp"
#include <mutex>
#include <unordered_map>
#include <functional>
#include <memory>
#include <vector>

extern "C" {
/* Subroutine */int sed_1d__(double *time, int *nstep,
		double * xpos, double *eblast, double *omega_in__,
		double * xgeom_in__, double *rho0, double *vel0,
		double *ener0, double *pres0, double *cs0, double *gam0,
		double *den, double *ener, double *pres, double *vel,
		double *cs);
}

namespace sedov {

void solution(double time, double r, double rmax, double& d, double& v, double& p) {
	int nstep = 10000;
	constexpr int bw = 2;

	using function_type = std::function<void(double,double&,double&,double&)>;
	using map_type = std::unordered_map<double,std::shared_ptr<function_type>>;
	using mutex_type = std::mutex;

	static map_type map;
	static mutex_type mutex;

	std::unique_lock<mutex_type> lock(mutex);

	double rho0 = 1.0;
	double vel0 = 0.0;
	double ener0 = 0.0;
	double pres0 = 0.0;
	double cs0 = 0.0;
	double gamma = grid::get_fgamma();
	double omega = 0.0;
	double eblast = 1.0;
	double xgeom = 3.0;

	std::vector<double> xpos(nstep+2*bw);
	std::vector<double> den(nstep+2*bw);
	std::vector<double> ener(nstep+2*bw);
	std::vector<double> pres(nstep+2*bw);
	std::vector<double> vel(nstep+2*bw);
	std::vector<double> cs(nstep+2*bw);

	std::shared_ptr<function_type> ptr;

	auto iter = map.find(time);
	for( int i = 0; i < nstep + 2*bw; i++) {
		xpos[i] = (i - bw + 0.5)*rmax/(nstep);
	}
	nstep += bw;
	if (iter == map.end()) {


		printf( "HELLo1\n");
		sed_1d__(&time, &nstep, xpos.data() + bw, &eblast, &omega, &xgeom, &rho0,
				&vel0, &ener0, &pres0, &cs0, &gamma, den.data() + bw, ener.data() + bw,
				pres.data() + bw, vel.data() + bw, cs.data() + bw);
		printf( "HELLo\n");

		xpos[0] = -xpos[3];
		den[0] = den[3];
		ener[0] = ener[3];
		pres[0] = pres[3];
		vel[0] = -vel[3];
		cs[0] = cs[3];

		xpos[1] = -xpos[2];
		den[1] = den[2];
		ener[1] = ener[2];
		pres[1] = pres[2];
		vel[1] = -vel[2];
		cs[1] = cs[2];


		function_type func = [nstep,rmax,den,pres,vel](double r, double& d, double& v, double & p) {
			double dr = rmax / (nstep);
			std::array<int,4> i;
			i[1] = (r + bw * dr) / dr;
			i[0] = i[1] - 1;
			i[2] = i[1] + 1;
			i[3] = i[1] + 2;
			double r0 = (r - (i[1]-bw)*dr)/dr;


			const auto interp = [r0,i](const std::vector<double>& data) {
				double sum = 0.0;
				sum += (-0.5 * data[i[0]] + 1.5 * data[i[1]] - 1.5 * data[i[2]] + 0.5 * data[i[3]]) * r0 * r0 * r0;
				sum += (+1.0 * data[i[0]] - 2.5 * data[i[1]] + 2.0 * data[i[2]] - 0.5 * data[i[3]]) * r0 * r0;
				sum += (-0.5 * data[i[0]]                   +  0.5 * data[i[2]]) * r0;
				sum += data[i[1]];
				return sum;
			};

			d = interp(den);
			v = interp(vel);
			p = interp(pres);

		};

		ptr = std::make_shared<function_type>(std::move(func));
		map[time] = ptr;
	} else {
		ptr = iter->second;
	}
	lock.unlock();

	const auto& func = *(ptr);

	func(r, d, v, p);
}

}

constexpr double blast_wave_t0 = 1.0;


std::vector<double> blast_wave_analytic(double x, double y, double z, double t) {
	real r = std::sqrt(x * x + y * y + z * z);
	t += blast_wave_t0;
	double rmax = 2.0 * opts().xscale;
	double d, v, p;
	sedov::solution(t, r, rmax, d, v, p);
	std::vector<double> u(opts().n_fields, 0.0);
	u[rho_i] = u[spc_i] = d;
	real s = d * v;
	u[sx_i] = s * x / r;
	u[sy_i] = s * y / r;
	u[sz_i] = s * z / r;
	real e = p / (grid::get_fgamma() - 1);
	u[egas_i] = e + s * v * 0.5;
	u[tau_i] = std::pow(e, 1 / grid::get_fgamma());
	return u;
}

std::vector<double> blast_wave(double x, double y, double z, double dx) {
	return blast_wave_analytic(x,y,z,0.0);


}
