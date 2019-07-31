#include <fenv.h>

#include "../../octotiger/unitiger/unitiger.hpp"
#include "../../octotiger/unitiger/hydro.hpp"

#include <hpx/hpx_init.hpp>

#define NDIM 2
#define INX 100
#define ORDER 2

#define H_BW 3
#define NF 3 + NDIM
#define H_NX (INX + H_BW)
#define H_N3 std::pow(INX+3,NDIM)
static constexpr double CFL = (0.4 / ORDER / NDIM);

int hpx_main(int, char*[]) {

	hydro_computer<NDIM, INX, ORDER> computer(2);
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);

	std::vector<std::array<double, NDIM>> X(H_N3);
	std::vector<std::vector<std::vector<double>>> F(NDIM, std::vector<std::vector<double>>(NF, std::vector<double>(H_N3, SNAN)));
	std::vector<std::vector<double>> U(NF, std::vector<double>(H_N3, SNAN));
	std::vector<std::vector<double>> U0(NF, std::vector<double>(H_N3, SNAN));

	const double dx = 1.0 / H_NX;

	for (int i = 0; i < H_N3; i++) {
		int k = i;
		int j = 0;
		while (k) {
			X[i][j] = (((k % H_NX) - H_BW) + 0.5) * dx - 0.5;
			k /= H_NX;
			j++;
		}
	}

	for (int i = 0; i < H_N3; i++) {
		for (int f = 0; f < NF; f++) {
			U[f][i] = 0.0;
		}
		double xsum = 0.0;
		double x2 = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			xsum += X[i][dim];
			auto o = dim == 0 ? 0.25 : 0.0;
			x2 += (X[i][dim] - o) * (X[i][dim] - o);
		}
//		if (xsum < 0.5 * NDIM) {
//			U[rho_i][i] = 1.0;
//			U[egas_i][i] = 2.5;
//		} else {
//			U[rho_i][i] = 0.125;
//			U[egas_i][i] = 0.25;
//		}
		U[rho_i][i] = 1.0;
		U[egas_i][i] = 1.0 + 1.0e+6 * std::exp(-x2 * H_NX * H_NX / 4.0);
		U[tau_i][i] = std::pow(U[egas_i][i], 1.0 / FGAMMA);
	}

	double t = 0.0;
	int iter = 0;

	computer.output(U, X, iter++);
	const double omega = 2.0 * M_PI / tmax / 100.0;

	while (t < tmax) {
		U0 = U;
		auto a = computer.hydro_flux(U, F, X, omega);
		double dt = CFL * dx / a;
		dt = std::min(dt, tmax - t + 1.0e-20);
		computer.advance(U0, U, F, dx, dt, 1.0, omega);
		computer.boundaries(U);
		if (ORDER >= 2) {
			computer.boundaries(U);
			computer.hydro_flux(U, F, X, omega);
			computer.advance(U0, U, F, dx, dt, ORDER == 2 ? 0.5 : 0.25, omega);
			if ( ORDER >= 3) {
				computer.boundaries(U);
				computer.hydro_flux(U, F, X, omega);
				computer.advance(U0, U, F, dx, dt, 2.0 / 3.0, omega);
			}
		}
		t += dt;
		computer.boundaries(U);
		computer.update_tau(U);
		computer.boundaries(U);
		computer.output(U, X, iter++);
		printf("%i %e %e\n", iter, t, dt);
	}
	computer.output(U, X, iter++);

	return hpx::finalize();
}

int main(int argc, char *argv[]) {
	printf("Running\n");
	std::vector<std::string> cfg = { "hpx.commandline.allow_unknown=1", // HPX should not complain about unknown command line options
			"hpx.scheduler=local-priority-lifo",       // Use LIFO scheduler by default
			"hpx.parcel.mpi.zero_copy_optimization!=0" // Disable the usage of zero copy optimization for MPI...
			};
	hpx::init(argc, argv, cfg);
}

