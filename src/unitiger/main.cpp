//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <fenv.h>
#include <time.h>

//#include <hpx/hpx_init.hpp>

#include "../../octotiger/unitiger/unitiger.hpp"
#include "../../octotiger/unitiger/hydro.hpp"
#include "../../octotiger/unitiger/safe_real.hpp"
#include <octotiger/unitiger/hydro_impl/reconstruct.hpp>


static constexpr double tmax = 2.49;
static constexpr safe_real dt_out = tmax / 249;

#define H_BW 3
#define H_NX (INX + 2 * H_BW)
#define H_N3 std::pow(INX+2*H_BW,NDIM)

template<int NDIM, int INX>
void run_test(typename physics<NDIM>::test_type problem, bool with_correction);

template<int NDIM, int INX>
void run_test(typename physics<NDIM>::test_type problem, bool with_correction) {
	static constexpr safe_real CFL = (0.4 / NDIM);
	hydro_computer<NDIM, INX> computer;
	if (with_correction) {
		computer.use_angmom_correction(physics<NDIM>::sx_i, 1);
	}
	const auto nf = physics<NDIM>::field_count();
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
	physics<NDIM> phys;
	computer.set_bc(phys.template initialize<INX>(problem, U, X));
	const safe_real dx = X[0][cell_geometry<NDIM, INX>::H_DNX] - X[0][0];
	computer.output(U, X, oter++, 0);
//	const safe_real omega = 2.0 * M_PI / tmax / 10.0;
	const safe_real omega = 0.0;
	printf("omega = %e\n", (double) omega);

	const auto tstart = time(NULL);
	while (t < tmax) {
		U0 = U;
		auto q = computer.reconstruct(U, X, omega);
		auto a = computer.flux(U, q, F, X, omega);
		safe_real dt = CFL * dx / a;
		dt = std::min(double(dt), tmax - t + 1.0e-20);
		computer.advance(U0, U, F, X, dx, dt, 1.0, omega);
		computer.boundaries(U);
		q = computer.reconstruct(U, X, omega);
		computer.flux(U, q, F, X, omega);
		computer.advance(U0, U, F, X, dx, dt, 0.25, omega);
		computer.boundaries(U);
		q = computer.reconstruct(U, X, omega);
		computer.flux(U, q, F, X, omega);
		computer.advance(U0, U, F, X, dx, dt, 2.0 / 3.0, omega);
		computer.boundaries(U);
		computer.post_process(U, dx);
		t += dt;
		computer.boundaries(U);
		if (int(t / dt_out) != int((t - dt) / dt_out))
			computer.output(U, X, oter++, t);
		iter++;
		printf("%i %e %e\n", iter, double(t), double(dt));
	}
	const auto tstop = time(NULL);
	U0 = U;
	physics<NDIM>::template analytic_solution<INX>(problem, U, X, t);
	computer.output(U, X, iter++, t);

	phys.template pre_recon<INX>(U0, X, omega, with_correction);
	phys.template pre_recon<INX>(U, X, omega, with_correction);
	std::vector<safe_real> L1(nf);
	std::vector<safe_real> L2(nf);
	std::vector<safe_real> Linf(nf);
	for (int f = 0; f < nf; f++) {
		L1[f] = L2[f] = Linf[f];
		for (int i = 0; i < H_N3; i++) {
			L1[f] += std::abs(U0[f][i] - U[f][i]);
			L2[f] += std::pow(U0[f][i] - U[f][i], 2);
			Linf[f] = std::max((double) Linf[f], std::abs(U0[f][i] - U[f][i]));
		}
		L2[f] = sqrt(L2[f]);
		L1[f] /= INX * INX;
		L2[f] /= INX * INX;
	}

	FILE *fp1 = fopen("L1.dat", "at");
	FILE *fp2 = fopen("L2.dat", "at");
	FILE *fpinf = fopen("Linf.dat", "at");
	fprintf(fp1, "%i ", INX);
	fprintf(fp2, "%i ", INX);
	fprintf(fpinf, "%i ", INX);
	for (int f = 0; f < nf; f++) {
		fprintf(fp1, "%e ", (double) L1[f]);
		fprintf(fp2, "%e ", (double) L2[f]);
		fprintf(fpinf, "%e ", (double) Linf[f]);
	}
	fprintf(fp1, "\n");
	fprintf(fp2, "\n");
	fprintf(fpinf, "\n");
	fclose(fp1);
	fclose(fp2);
	fclose(fpinf);
	FILE* fp = fopen( "time.dat", "wt");
	fprintf( fp, "%i %li\n", INX, tstop -tstart);
	fclose(fp);
}


int main(int, char*[]) {
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);

	run_test<2, 200>(physics<2>::KH, false);
//	run_test<2, 128>(physics<2>::CONTACT, false);
//	run_test<2, 256>(physics<2>::CONTACT, false);
//	run_test<2, 512>(physics<2>::CONTACT, false);
//	run_test<1, 1024>(physics<1>::CONTACT, false);
	//	run_test<3, 8>(physics<3>::SOD, false);
//	run_test<3, 32>(physics<3>::BLAST, false);
//	run_test<2, 200>(physics<2>::BLAST, true);
//	run_test<2, 160>(physics<2>::BLAST, true);
//	run_test<2, 250>(physics<2>::BLAST, true);
//	run_test<2, 300>(physics<2>::BLAST, true);
//	run_test<2, 350>(physics<2>::BLAST, true);
//	run_test<2, 350>(physics<2>::BLAST, true);
//	run_test<2, 420>(physics<2>::BLAST, true);
//	run_test<2, 500>(physics<2>::BLAST, true);
//	run_test<2, 600>(physics<2>::BLAST, true);
//	run_test<2, 7>(physics<2>::BLAST, true);
//	run_test<2, 840>(physics<2>::BLAST, true);
//	run_test<2, 1000>(physics<2>::BLAST, true);

	return 0;
}
