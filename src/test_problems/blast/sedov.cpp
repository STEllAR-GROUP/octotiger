#include <hpx/include/run_as.hpp>
#include <stdio.h>
#include <stdlib.h>
#include "../../options.hpp"

void compute_sedov(std::vector<double>& den, std::vector<double>& ener, std::vector<double>& vel, double xmax, int N, double eblast, double t ) {

	char str[256];

	char* ptr;
	asprintf( &ptr, "./sedov_fort %i %e 3 0 %e %e 1.6666666 sedov.txt\n", N, eblast, t, xmax );
	printf( ptr );
	system(ptr);
	free(ptr);
	den.resize(N);
	ener.resize(N);
	vel.resize(N);
	FILE* fp = fopen( "sedov.txt", "rt" );
	fgets( str, 256, fp );
	fgets( str, 256, fp );
	for( int i = 0; i < N; i++ ) {
		fgets( str, 256, fp );
		den[i] = atof( str + 21 );
		ener[i] = atof( str + 35 );
		vel[i] = atof( str + 63 );
	}
	fclose(fp);
}


class sedov_analytic {
	int N;
	static constexpr int bw = 2;

	double rmax, dr;
	std::vector<double> eout;
	std::vector<double> dout;
	std::vector<double> vout;
public:
	std::vector<double> state_at(double x, double y, double z) {
#ifndef TESTME
		std::vector<double> u(opts().n_fields,0.0);
#endif
		int i[4];
		const double r = std::sqrt(x * x + y * y + z * z);
		i[1] = (r + bw * dr) / dr;
		i[0] = i[1] - 1;
		i[2] = i[1] + 1;
		i[3] = i[1] + 2;
		double r0 = (r - (i[1]-bw)*dr)/dr;
	//	printf( "%e %i\n", r0,  i[1]);
		const auto interp = [&r0,&i](const std::vector<double>& data) {
			double sum = 0.0;
			sum += (-0.5 * data[i[0]] + 1.5 * data[i[1]] - 1.5 * data[i[2]] + 0.5 * data[i[3]]) * r0 * r0 * r0;
			sum += (+1.0 * data[i[0]] - 2.5 * data[i[1]] + 2.0 * data[i[2]] - 0.5 * data[i[3]]) * r0 * r0;
			sum += (-0.5 * data[i[0]]                   +  0.5 * data[i[2]]) * r0;
			sum += data[i[1]];
			return sum;
		};
		double d = std::max(interp(dout),1.0e-20);
		double v = interp(vout);
		double e = std::max(interp(eout), 1.0e-20);
#ifdef TESTME
		printf( "%e %e %e %e\n", r, d, v, e);
		return std::vector<double>();
#else
		u[rho_i] = d;
		u[spc_i] = d;
		u[sx_i] = v * d * x / r;
		u[sy_i] = v * d * y / r;
		u[sz_i] = v * d * z / r;
		u[egas_i] = d * v * v * 0.5 + e * d;
		u[tau_i] = std::pow(e * d, 3.0 / 5.0);
		u[zx_i] = 0.0;
		u[zy_i] = 0.0;
		u[zz_i] = 0.0;
//		printf( "%e\n", std::pow(e * d, 3.0 / 5.0));
		return std::move(u);
#endif
	}
	sedov_analytic(double t) {
		N = 1000  + bw;
		rmax = 2.0;
		dr = rmax / (N-bw);
		std::vector<double> xpos(N);
		for (int i = 0; i < N; i++) {
			xpos[i] = (i + 0.5 - bw) * dr;
		}
		eout.resize(N);
		vout.resize(N);
		dout.resize(N);

		//TODO insert new call

		hpx::threads::run_as_os_thread([&]() {
//			sedov_solution(t, N - bw, xpos.data() + bw, 1.0, 1.0, dout.data() + bw, eout.data() + bw, vout.data() + bw);
		}).get();
		dout[0] = dout[3];
		dout[1] = dout[2];
		eout[0] = eout[3];
		eout[1] = eout[2];
		vout[0] = -vout[3];
		vout[1] = -vout[2];
#ifdef TESTME
		for( double r = 0; r < 1.0; r += 1.0e-3 ) {
			state_at(r,0.0,0.0);
		}
#endif	
	}
};

constexpr int sedov_analytic::bw;

#ifdef TESTME
int main() {
	sedov_analytic test(1.0e-12);
}
#endif


/*
std::vector<real> blast_wave(real x, real y, real z, real dx) {
	const real fgamma = grid::get_fgamma();
	std::vector<real> u(opts().n_fields, real(0));
	u[spc_i] = u[rho_i] = 1.0e-3;
	const real a = std::sqrt(10.0) * std::min(dx, 0.1);
	real r = std::sqrt(x * x + y * y + z * z);
	u[egas_i] = std::max(1.0e-10, exp(-r * r / a / a)) / 100.0;
	u[tau_i] = std::pow(u[egas_i], ONE / fgamma);
	return u;
}*/


#ifndef TESTME
std::vector<double> blast_wave(double x, double y, double z, double dx) {
	static sedov_analytic state(1.0e-3);
	return state.state_at(x,y,z);
}
#endif







