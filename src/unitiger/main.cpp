//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <fenv.h>
#include <time.h>

//#include <hpx/hpx_init.hpp>

#include "octotiger/unitiger/unitiger.hpp"
#include "octotiger/unitiger/hydro.hpp"
#include "octotiger/unitiger/physics.hpp"
#include "octotiger/unitiger/physics_impl.hpp"
#include "octotiger/unitiger/radiation/radiation_physics.hpp"
#include "octotiger/unitiger/radiation/radiation_physics_impl.hpp"
#include "octotiger/unitiger/safe_real.hpp"
#include "octotiger/unitiger/hydro_impl/reconstruct.hpp"
#include "octotiger/unitiger/hydro_impl/flux.hpp"
#include "octotiger/unitiger/hydro_impl/boundaries.hpp"
#include "octotiger/unitiger/hydro_impl/advance.hpp"
#include "octotiger/unitiger/hydro_impl/output.hpp"

static constexpr double tmax = 1.0;
static constexpr safe_real dt_out = tmax / 100;

#define H_BW 3
#define H_NX (INX + 2 * H_BW)
#if NDIM == 3
#define H_N3 (INX+2*H_BW) * (INX+2*H_BW) * (INX+2*H_BW)
#endif
#if NDIM == 2
#define H_N3 (INX+2*H_BW) * (INX+2*H_BW) 
#endif
#if NDIM == 1
#define H_N3 (INX+2*H_BW)
#endif

template<int NDIM, int INX, class PHYS>
void run_test(typename PHYS::test_type problem, bool with_correction, bool writingForTest);

template<int NDIM, int INX, class PHYS>
void run_test(typename PHYS::test_type problem, bool with_correction, bool writingForTest) {
	static constexpr safe_real CFL = (0.4 / NDIM);
	hydro_computer<NDIM, INX, PHYS> computer;
	if (with_correction) {
		computer.use_angmom_correction(PHYS::get_angmom_index());
	}
	const auto nf = PHYS::field_count();
	computer.use_disc_detect(PHYS::rho_i);
	for (int s = 0; s < 5; s++) {
		computer.use_disc_detect(PHYS::spc_i + s);
	}
	std::vector<std::vector<std::vector<safe_real>>> F(NDIM, std::vector<std::vector<safe_real>>(nf, std::vector<safe_real>(H_N3)));
	std::vector<std::vector<safe_real>> U(nf, std::vector<safe_real>(H_N3));
	std::vector<std::vector<safe_real>> U0(nf, std::vector<safe_real>(H_N3));
	hydro::x_type X(NDIM);
	for (int dim = 0; dim < NDIM; dim++) {
		X[dim].resize(H_N3);
	}

	safe_real t = 0.0;
	int iter = 0;
	int oter = 0;
	bool printEachTimeStep = true;
	std::string type_test_string = PHYS::get_test_type_string(problem);
	hydro::recon_type<NDIM> q;
	PHYS phys;
	computer.set_bc(phys.template initialize<INX>(problem, U, X));
	const safe_real dx = X[0][cell_geometry<NDIM, INX>::H_DNX] - X[0][0];
	computer.output(U, X, oter++, 0);
//	const safe_real omega = 2.0 * M_PI / tmax / 10.0;
	const safe_real omega = 0.0;
	print("omega = %e\n", (double) omega);

	constexpr int RK = 2;

	const auto tstart = time(NULL);
	while (t < tmax) {
		U0 = U;
		safe_real dt;
		q = computer.reconstruct(U, X, omega);
		auto a = computer.flux(U, q, F, X, omega);
		dt = CFL * dx / a.a;
		dt = std::min(double(dt), tmax - t + 1.0e-20);
		computer.advance(U0, U, F, X, dx, dt, 1.0, omega);
		computer.boundaries(U, X);
		if HOST_CONSTEXPR (RK == 3) {
			q = computer.reconstruct(U, X, omega);
			computer.flux(U, q, F, X, omega);
			computer.advance(U0, U, F, X, dx, dt, 0.25, omega);
			computer.boundaries(U, X);
			q = computer.reconstruct(U, X, omega);
			computer.flux(U, q, F, X, omega);
			computer.advance(U0, U, F, X, dx, dt, 2.0 / 3.0, omega);
			computer.boundaries(U, X);
		} else if HOST_CONSTEXPR (RK == 2) {
			q = computer.reconstruct(U, X, omega);
			computer.flux(U, q, F, X, omega);
			computer.advance(U0, U, F, X, dx, dt, 0.5, omega);
			computer.boundaries(U, X);
		}
		computer.post_process(U, X, dx);
		t += dt;
		computer.boundaries(U, X);
		if (int(t / dt_out) != int((t - dt) / dt_out)){
			computer.output(U, X, oter++, t);
		}
		iter++;
		print("%i %e %e\n", iter, double(t), double(dt));
		if (writingForTest) {
			computer.outputU(U, iter, type_test_string);
			computer.outputQ(q, iter, type_test_string);
			computer.outputF(F, iter, type_test_string);
		}
		if (printEachTimeStep) {
			int testU = computer.compareU(U, iter, type_test_string);
			int testQ = computer.compareQ(q, iter, type_test_string);
			int testF = computer.compareF(F, iter, type_test_string);
			if ((testU == -1) or (testQ == -1) or (testF == -1))
				print("Could not test, output files do not exist! Create test files by running unitiger with -C\n");
			if (testU * testQ * testF == 1)
				print("%s tests are OK!\n", type_test_string.c_str());
		}
	}
//      U0 = U;
//      PHYS::template analytic_solution<INX>(problem, U, X, t);
//      computer.output(U, X, iter++, t);
//
//      phys.template pre_recon<INX>(U0, X, omega, with_correction);
//      phys.template pre_recon<INX>(U, X, omega, with_correction);
//      std::vector<safe_real> L1(nf);
//      std::vector<safe_real> L2(nf);
//      std::vector<safe_real> Linf(nf);
//      for (int f = 0; f < nf; f++) {
//              L1[f] = L2[f] = Linf[f];
//              for (int i = 0; i < H_N3; i++) {
//                      L1[f] += std::abs(U0[f][i] - U[f][i]);
//                      L2[f] += std::pow(U0[f][i] - U[f][i], 2);
//                      Linf[f] = std::max((double) Linf[f], std::abs(U0[f][i] - U[f][i]));
//              }
//              L2[f] = sqrt(L2[f]);
//              L1[f] /= INX * INX;
//              L2[f] /= INX * INX;
//      }

//      FILE *fp1 = fopen("L1.dat", "at");
//      FILE *fp2 = fopen("L2.dat", "at");
//      FILE *fpinf = fopen("Linf.dat", "at");
//      fprintf(fp1, "%i ", INX);
//      fprintf(fp2, "%i ", INX);
//      fprintf(fpinf, "%i ", INX);
	const auto tstop = time(NULL);
	FILE *fp = fopen("time.dat", "wt");
	fprintf(fp, "%i %li\n", INX, tstop - tstart);
	fclose(fp);
	if (writingForTest) {
		computer.outputU(U, -1, type_test_string);
		computer.outputQ(q, -1, type_test_string);
		computer.outputF(F, -1, type_test_string);
	}
	int testU = computer.compareU(U, -1, type_test_string);
	int testQ = computer.compareQ(q, -1, type_test_string);
	int testF = computer.compareF(F, -1, type_test_string);
	if ((testU == -1) or (testQ == -1) or (testF == -1))
		print("Could not test, output files do not exist! Create test files by running unitiger with -C\n");
	if (testU * testF * testQ == 1)
		print("Final %s tests are OK!!\n", type_test_string.c_str());
}

int main(int argc, char *argv[]) {
	//feenableexcept(FE_DIVBYZERO);
	//feenableexcept(FE_INVALID);
	//feenableexcept(FE_OVERFLOW);

	bool createTests = false;

	if (argc > 1) {
		std::string input = argv[1];
		if (input == "-C") {
			print("Creating Tests.\n");
			createTests = true;
		}
	}

	srand(123);
	run_test<2, 64, physics<2>>(physics<2>::KEPLER, true, createTests);
//	run_test<2, 100, physics<2>>(physics<2>::KEPLER, true, createTests);
//        run_test<3, 8, physics<3>>(physics<3>::SOD, false, createTests);
//        run_test<2, 50, physics<2>>(physics<2>::BLAST, true, createTests);
//        run_test<2, 50, radiation_physics<2>>(radiation_physics<2>::CONTACT, true, createTests);

	return 0;
}
