#include <fenv.h>

#ifndef NOHPX
//#include <hpx/hpx_init.hpp>
#include "../../octotiger/unitiger/unitiger.hpp"
#include "../../octotiger/unitiger/hydro.hpp"
#include "../../octotiger/unitiger/safe_real.hpp"
#else
#include "../../octotiger/octotiger/unitiger/unitiger.hpp"
#include "safe_real.hpp"
#include "../../octotiger/octotiger/unitiger/hydro.hpp"
#endif


#define NDIM 2
#define INX 128

#define H_BW 3
#define H_NX (INX + 2 * H_BW)
#define H_N3 std::pow(INX+2*H_BW,NDIM)
static constexpr safe_real CFL = (0.4 / NDIM);

int main(int, char*[]) {
//int hpx_main(int, char*[]) {

	hydro_computer<NDIM, INX> computer;
	using comp = decltype(computer);
	//computer.use_angmom_correction(comp::sx_i, 1);
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);

	std::vector<std::vector<std::vector<safe_real>>> F(NDIM, std::vector<std::vector<safe_real>>(computer.nf, std::vector<safe_real>(H_N3)));
	std::vector<std::vector<safe_real>> U(computer.nf, std::vector<safe_real>(H_N3));
	std::vector<std::vector<safe_real>> U0(computer.nf, std::vector<safe_real>(H_N3));
	hydro::x_type<NDIM> X(H_N3);

	const safe_real dx = 1.0 / INX;

	for (int i = 0; i < H_N3; i++) {
		int k = i;
		int j = 0;
		for (int dim = 0; dim < NDIM; dim++) {
			X[i][j] = (((k % H_NX) - H_BW) + 0.5) * dx - 0.5;
			k /= H_NX;
			j++;
		}
	}

	for (int i = 0; i < H_N3; i++) {
		for (int f = 0; f < computer.nf; f++) {
			U[f][i] = 0.0;
		}
		safe_real xsum = 0.0;
		safe_real x2 = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			xsum += X[i][dim];
			auto o = dim == 0 ? 0.0 : 0.0;
			x2 += (X[i][dim] - o) * (X[i][dim] - o);
		}
//		if (xsum < 0) {
//			U[comp::rho_i][i] = 1.0;
//			U[comp::egas_i][i] = 2.5;
//		} else {
//			U[comp::rho_i][i] = 0.125;
//			U[comp::egas_i][i] = 0.25;
//		}
		U[comp::rho_i][i] = 1.0;
		U[comp::egas_i][i] = 1e+6 * std::exp(-x2 * INX * INX / 4.0);
		U[comp::tau_i][i] = POWER(U[comp::egas_i][i], 1.0 / FGAMMA);
	}

	safe_real t = 0.0;
	int iter = 0;

	computer.output(U, X, iter++);
//	const safe_real omega = 2.0 * M_PI / tmax / 4.0;
	const safe_real omega = 0.0;
	while (t < tmax) {
		U0 = U;
		auto q = computer.reconstruct(U, dx);
		auto a = computer.flux(q, F, X, omega);
		safe_real dt = CFL * dx / a;
		dt = std::min(double(dt), tmax - t + 1.0e-20);
		computer.advance(U0, U, F, X, dx, dt, 1.0, omega);
		computer.boundaries(U);
		computer.boundaries(U);
		q = computer.reconstruct(U, dx);
		computer.flux(q, F, X, omega);
		computer.advance(U0, U, F, X, dx, dt, 0.25, omega);
		computer.boundaries(U);
		q = computer.reconstruct(U, dx);
		computer.flux(q, F, X, omega);
		computer.advance(U0, U, F, X, dx, dt, 2.0 / 3.0, omega);
		t += dt;
		computer.boundaries(U);
		computer.post_process(U, dx);
		computer.boundaries(U);
		computer.output(U, X, iter++);
		printf("%i %e %e\n", iter, double(t), double(dt));
	}
	computer.output(U, X, iter++);
#ifdef NOHPX
	return 0;
#else
//	return hpx::finalize();
#endif
}
//
//int main(int argc, char *argv[]) {
//#ifdef NOHPX
//	return hpx_main(argc, argv);
//#else
//	printf("Running\n");
//	std::vector<std::string> cfg = { "hpx.commandline.allow_unknown=1", // HPX should not complain about unknown command line options
//			"hpx.scheduler=local-priority-lifo",       // Use LIFO scheduler by default
//			"hpx.parcel.mpi.zero_copy_optimization!=0" // Disable the usage of zero copy optimization for MPI...
//			};
//	hpx::init(argc, argv, cfg);
//#endif
//}
//
