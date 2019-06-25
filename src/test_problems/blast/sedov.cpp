#include "octotiger/grid.hpp"

#include <hpx/runtime/threads/run_as_os_thread.hpp>

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#if !defined(OCTOTIGER_HAVE_BOOST_MULTIPRECISION)
#include <quadmath.h>
using sed_real = __float128;
#else
#include <boost/multiprecision/cpp_bin_float.hpp>
using sed_real = boost::multiprecision::cpp_bin_float_quad;
#endif




/*extern "C" {*/
/* Subroutine */int sed_1d__(sed_real *time, int *nstep,
		sed_real * xpos, sed_real *eblast, sed_real *omega_in__,
		sed_real * xgeom_in__, sed_real *rho0, sed_real *vel0,
		sed_real *ener0, sed_real *pres0, sed_real *cs0, sed_real *gam0,
		sed_real *den, sed_real *ener, sed_real *pres, sed_real *vel,
		sed_real *cs);
//}

namespace sedov {

void solution(real time, real r, real rmax, real& d, real& v, real& p) {
	int nstep = 10000;
	constexpr int bw = 2;
	using function_type = std::function<void(real,real&,real&,real&)>;
	using map_type = std::unordered_map<real,std::shared_ptr<function_type>>;
	using mutex_type = hpx::lcos::local::spinlock;

	static map_type map;
	static mutex_type mutex;


	sed_real rho0 = 1.0;
	sed_real vel0 = 0.0;
	sed_real ener0 = 0.0;
	sed_real pres0 = 0.0;
	sed_real cs0 = 0.0;
	sed_real gamma = grid::get_fgamma();
	sed_real omega = 0.0;
	sed_real eblast = 1.0;
	sed_real xgeom = 3.0;

	std::vector<sed_real> xpos(nstep+2*bw);
	std::vector<sed_real> den(nstep+2*bw);
	std::vector<sed_real> ener(nstep+2*bw);
	std::vector<sed_real> pres(nstep+2*bw);
	std::vector<sed_real> vel(nstep+2*bw);
	std::vector<sed_real> cs(nstep+2*bw);

	std::vector<real> den1(nstep+2*bw);
	std::vector<real> pres1(nstep+2*bw);
	std::vector<real> vel1(nstep+2*bw);

	std::shared_ptr<function_type> ptr;

	for( int i = 0; i < nstep + 2*bw; i++) {
		xpos[i] = (i - bw + 0.5)*rmax/(nstep);
	}
	nstep += bw;

	std::unique_lock<mutex_type> lock(mutex);
	auto iter = map.find(time);
	if (iter == map.end()) {
		sed_real sed_time = time;
		printf( "Computing sedov solution\n");
		sed_1d__(&sed_time, &nstep, xpos.data() + bw, &eblast, &omega, &xgeom, &rho0,
				&vel0, &ener0, &pres0, &cs0, &gamma, den.data() + bw, ener.data() + bw,
				pres.data() + bw, vel.data() + bw, cs.data() + bw);

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

#if defined(OCTOTIGER_HAVE_BOOST_MULTIPRECISION)
		std::transform(den.begin(), den.end(), den1.begin(),
			[](sed_real v) { return v.convert_to<double>(); });
		std::transform(vel.begin(), vel.end(), vel1.begin(),
			[](sed_real v) { return v.convert_to<double>(); });
		std::transform(pres.begin(), pres.end(), pres1.begin(),
			[](sed_real v) { return v.convert_to<double>(); });
#else
		std::copy(den.begin(), den.end(), den1.begin());
		std::copy(vel.begin(), vel.end(), vel1.begin());
		std::copy(pres.begin(), pres.end(), pres1.begin());
#endif

		function_type func = [nstep,rmax,den1,pres1,vel1,bw](real r, real& d, real& v, real & p) {
			real dr = rmax / (nstep);
			std::array<int,4> i;
			i[1] = (r + (bw - 0.5)*dr) / dr;
			i[0] = i[1] - 1;
			i[2] = i[1] + 1;
			i[3] = i[1] + 2;
			real r0 = (r - (i[1]-bw + 0.5)*dr)/dr;
	//		printf( "%i %e\n", i[0], r, dr );
			assert( i[0] >= 0 );
			assert( i[3] < vel1.size());
			const auto interp = [r0,i](const std::vector<real>& data) {
				real sum = 0.0;
				sum += (-0.5 * data[i[0]] + 1.5 * data[i[1]] - 1.5 * data[i[2]] + 0.5 * data[i[3]]) * r0 * r0 * r0;
				sum += (+1.0 * data[i[0]] - 2.5 * data[i[1]] + 2.0 * data[i[2]] - 0.5 * data[i[3]]) * r0 * r0;
				sum += (-0.5 * data[i[0]]                   +  0.5 * data[i[2]]) * r0;
				sum += data[i[1]];
				return sum;
			};

			d = interp(den1);
			v = interp(vel1);
			p = interp(pres1);

		};

		ptr = std::make_shared<function_type>(std::move(func));
		map[time] = ptr;
		lock.unlock();
	} else {
		lock.unlock();
		ptr = iter->second;
	}

	const auto& func = *(ptr);

	func(r, d, v, p);
}

}
constexpr real blast_wave_t0 = 7e-4;


std::vector<real> blast_wave_analytic(real x, real y, real z, real t) {
	real r = std::sqrt(x * x + y * y + z * z);
	t += blast_wave_t0;
	real rmax = 3.0 * opts().xscale;
	real d, v, p;
	sedov::solution(t, r, rmax, d, v, p);
	std::vector<real> u(opts().n_fields, 0.0);
	u[rho_i] = u[spc_i] = std::max(d,1.0e-20);
	real s = d * v;
	u[sx_i] = s * x / r;
	u[sy_i] = s * y / r;
	u[sz_i] = s * z / r;
	real e = std::max(p / (grid::get_fgamma() - 1),1.0e-20);
	u[egas_i] = e + s * v * 0.5;
	u[tau_i] = std::pow(e, 1 / grid::get_fgamma());
	return u;
}

std::vector<real> blast_wave(real x, real y, real z, real dx) {
	return blast_wave_analytic(x,y,z,0.0);


}
