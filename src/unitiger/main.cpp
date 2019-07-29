
#include <fenv.h>

#include "../../octotiger/unitiger/unitiger.hpp"

#include <hpx/hpx_init.hpp>

void boundaries(std::vector<std::vector<double>> &U) {
	for (int f = 0; f < NF; f++) {
		if (NDIM == 1) {
			for (int i = 0; i < H_BW + 20; i++) {
				U[f][i] = U[f][H_BW];
				U[f][H_NX - 1 - i] = U[f][H_NX - H_BW - 1];
			}
		} else if (NDIM == 2) {

			const auto index = [](int i, int j) {
				return i + H_NX * j;
			};

			for (int i = 0; i < H_BW; i++) {
				for (int j = 0; j < H_NX; j++) {
					U[f][index(i, j)] = U[f][index(H_BW, j)];
					U[f][index(j, i)] = U[f][index(j, H_BW)];
					U[f][index(H_NX - 1 - i, j)] = U[f][index(H_NX - 1 - H_BW, j)];
					U[f][index(j, H_NX - 1 - i)] = U[f][index(j, H_NX - 1 - H_BW)];
				}
			}
		} else {
			const auto index = [](int i, int j, int k) {
				return i + H_NX * j + k * H_NX * H_NX;
			};

			for (int i = 0; i < H_BW; i++) {
				for (int j = 0; j < H_NX; j++) {
					for (int k = 0; k < H_NX; k++) {
						U[f][index(i, j, k)] = U[f][index(H_BW, j, k)];
						U[f][index(j, i, k)] = U[f][index(j, H_BW, k)];
						U[f][index(j, k, i)] = U[f][index(j, k, H_BW)];
						U[f][index(H_NX - 1 - i, j, k)] = U[f][index(H_NX - 1 - H_BW, j, k)];
						U[f][index(j, H_NX - 1 - i, k)] = U[f][index(j, H_NX - 1 - H_BW, k)];
						U[f][index(j, H_NX - 1 - k, i)] = U[f][index(j, k, H_NX - 1 - H_BW)];
					}
				}
			}
		}
	}
}

void advance(const std::vector<std::vector<double>> &U0, std::vector<std::vector<double>> &U, const std::vector<std::vector<std::vector<double>>> &F, double dx,
		double dt, double beta) {
	int stride = 1;
	int H_BW = bound_width();
	for (int dim = 0; dim < NDIM; dim++) {
		for (int f = 0; f < NF; f++) {
			for (int i = 2 * H_BW; i < H_N3 - 2 * H_BW; i++) {
				double u0 = U0[f][i];
				const auto fr = F[dim][f][i + stride];
				const auto fl = F[dim][f][i];
				const auto dudt = -(fr - fl) / dx;
				double u1 = U[f][i] + dudt * dt;
				U[f][i] = u0 * (1.0 - beta) + u1 * beta;
			}
		}
		stride *= H_NX;
	}
}

void output(const std::vector<std::vector<double>> &U, const std::vector<std::array<double, NDIM>> &X, int num) {
	std::string filename = "Y." + std::to_string(num);
	if (NDIM == 1) {
		filename += ".txt";
		FILE *fp = fopen(filename.c_str(), "wt");
		for (int i = 0; i < H_NX; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				fprintf(fp, "%13.6e ", X[i][dim]);
			}
			for (int f = 0; f < NF; f++) {
				fprintf(fp, "%13.6e ", U[f][i]);
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
	} else {
		filename += ".silo";
		auto db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Uni-tiger", DB_PDB);
		const char *coord_names[] = { "x", "y", "z" };
		double coords[NDIM][H_NX + 1];
		for (int i = 0; i < H_NX + 1; i++) {
			const auto x = double(i - H_BW) / H_NX;
			for (int dim = 0; dim < NDIM; dim++) {
				coords[dim][i] = x;
			}
		}
		void *coords_[] = { coords, coords + 1, coords + 2 };
		int dims1[] = { H_NX + 1, H_NX + 1, H_NX + 1 };
		int dims2[] = { H_NX, H_NX, H_NX };
		DBPutQuadmesh(db, "quadmesh", coord_names, coords_, dims1, NDIM, DB_DOUBLE, DB_COLLINEAR, NULL);
		for (int f = 0; f < NF; f++) {
			DBPutQuadvar1(db, field_names[f], "quadmesh", U[f].data(), dims2, NDIM, NULL, 0, DB_DOUBLE, DB_ZONECENT,
			NULL);
		}
		DBClose(db);
	}

}



int hpx_main(int, char*[]) {

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
			X[i][j] = (((k % H_NX) - H_BW) + 0.5) * dx;
			k /= H_NX;
			j++;
		}
	}

	for (int i = 0; i < H_N3; i++) {
		for (int f = 0; f < NF; f++) {
			U[f][i] = 0.0;
		}
		double xsum = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			xsum += X[i][dim];
		}
		if (xsum < 1) {
			U[rho_i][i] = 1.0;
			U[egas_i][i] = 2.5;
		} else {
			U[rho_i][i] = 0.125;
			U[egas_i][i] = 0.25;
		}
	}

	double t = 0.0;
	int iter = 0;
	output(U, X, iter++);
	while (t < tmax) {
		U0 = U;
		auto a = hydro_flux(U, F);
		double dt = 0.4 * dx / a / NDIM;
		advance(U0, U, F, dx, dt, 1.0);
		boundaries(U);
		if (ORDER >= 2) {
			boundaries(U);
			hydro_flux(U, F);
			advance(U0, U, F, dx, dt, 0.5);
		}
		t += dt;
		boundaries(U);
		output(U, X, iter++);
		printf("%e %e\n", t, dt);
	}
	output(U, X, iter++);

	return hpx::finalize();
}



int main(int argc, char* argv[]) {
	printf( "Running\n");
	std::vector<std::string> cfg = { "hpx.commandline.allow_unknown=1", // HPX should not complain about unknown command line options
			"hpx.scheduler=local-priority-lifo",       // Use LIFO scheduler by default
			"hpx.parcel.mpi.zero_copy_optimization!=0" // Disable the usage of zero copy optimization for MPI...
			};
	hpx::init(argc, argv, cfg);
}

