#include <fenv.h>

//#include <hpx/hpx_init.hpp>
#include "../../octotiger/unitiger/unitiger.hpp"
#include "../../octotiger/unitiger/hydro.hpp"
#include "../../octotiger/unitiger/safe_real.hpp"

#define NDIM 2
#define INX 150

static constexpr double tmax = 1.0e-2;
static constexpr safe_real dt_out = tmax / 250;
static constexpr auto problem = physics<NDIM>::BLAST;


#define H_BW 3
#define H_NX (INX + 2 * H_BW)
#define H_N3 std::pow(INX+2*H_BW,NDIM)
static constexpr safe_real CFL = (0.4 / NDIM);

int main(int, char*[]) {
//int hpx_main(int, char*[]) {

	hydro_computer<NDIM, INX> computer;
	computer.use_angmom_correction(physics<NDIM>::sx_i, 1);
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);

	std::vector<std::vector<std::vector<safe_real>>> F(NDIM, std::vector<std::vector<safe_real>>(physics<NDIM>::nf, std::vector<safe_real>(H_N3)));
	std::vector<std::vector<safe_real>> U(physics<NDIM>::nf, std::vector<safe_real>(H_N3));
	std::vector<std::vector<safe_real>> U0(physics<NDIM>::nf, std::vector<safe_real>(H_N3));
	hydro::x_type<NDIM> X;
	for (int dim = 0; dim < NDIM; dim++) {
		X[dim].resize(H_N3);
	}

	safe_real t = 0.0;
	int iter = 0;

	physics<NDIM> phys;
	computer.set_bc(phys.initialize<INX>(problem, U, X));
	const safe_real dx = X[0][cell_geometry<NDIM,INX>::H_DNX] - X[0][0];
	computer.output(U, X, iter++, 0);
//	const safe_real omega = 2.0 * M_PI / tmax / 4.0;
	const safe_real omega = 0.0;
	printf("omega = %e\n", omega);
	while (t < tmax) {
		U0 = U;
		auto q = computer.reconstruct(U, X, omega);
		auto a = computer.flux(U, q, F, X, omega);
		safe_real dt = CFL * dx / a;
		dt = std::min(double(dt), tmax - t + 1.0e-20);
		computer.advance(U0, U, F, X, dx, dt, 1.0, omega);
		computer.boundaries(U);
		computer.boundaries(U);
		q = computer.reconstruct(U, X, omega);
		computer.flux(U, q, F, X, omega);
		computer.advance(U0, U, F, X, dx, dt, 0.25, omega);
		computer.boundaries(U);
		q = computer.reconstruct(U, X, omega);
		computer.flux(U, q, F, X, omega);
		computer.advance(U0, U, F, X, dx, dt, 2.0 / 3.0, omega);
		t += dt;
		computer.boundaries(U);
		computer.post_process(U, dx);
		computer.boundaries(U);
		if (int(t / dt_out) != int((t - dt) / dt_out))
			computer.output(U, X, iter, t);
		iter++;
		printf("%i %e %e\n", iter, double(t), double(dt));
	}
	physics<NDIM>::analytic_solution<INX>(problem, U, X, t);
	computer.output(U, X, iter++, t);
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
//	std::vector<std::string> cfg = 9{ "hpx.commandline.allow_unknown=1", // HPX should not complain about unknown command line options
//			"hpx.scheduler=local-priority-lifo",       // Use LIFO scheduler by default
//			"hpx.parcel.mpi.zero_copy_optimization!=0" // Disable the usage of zero copy optimization for MPI...
//			};
//	hpx::init(argc, argv, cfg);
//#endif
//}
//
