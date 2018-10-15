//============================================================================
// Author      : Dominic Marcello
// Version     :
// Copyright   : Copyright (C) 2017 Dominic Marcello
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <array>
using namespace std;
#include <cmath>
#include <vector>

constexpr int nr = 256;
constexpr int nz = 256;
#define nt (nr/4)
#define zedge 0.6
#define redge 0.9
#define zei int(zedge*nz)
#define rei int(redge*nr)

using array_type = std::array<std::array<double, nz>, nr>;

#include <fenv.h>


int main() {

//	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

	static array_type phi;
	static array_type rho;
	const double dr = 1.0 / nr;
	const double dz = 1.0 / nz;
	double omega, K;
	const double n = 1.5;

	// initialize
	{
		for (int i = 0; i < nr; i++) {
			for (int k = 0; k < nz; k++) {
				phi[i][k] = 0.0;
				rho[i][k] = 0.0;
				const double R = (i + 0.5) / nr;
				const double z = (k + 0.5) / nz;
				if (i == 0 && k == 0) {
					rho[i][k] = 1.0;
				}
			}
		}
	}

	double scf_error = 0.0;
	int scf_iter = 0;
	do {

		// Solve boundaries
		{
			for (int i = 0; i < nr; i++) {
				const double R = (i + 0.5) / nr;
				for (int k = 0; k < nz; k++) {
					const double z = (k + 0.5) / nz;
					if (i == nr - 1 || k == nz - 1) {
						double this_phi = 0.0;
						for (int i0 = 0; i0 < nr - 1; i0++) {
							const double R0 = (i0 + 0.5) / nr;
							const double a = R0 * R0 + R * R;
							const double b = 2.0 * R0 * R;
							for (int k0 = 0; k0 < nz - 1; k0++) {
								if (rho[i0][k0] > 0.0) {
									const double z0 = (k0 + 0.5) / nz;
									const double cp = pow(z - z0, 2);
									const double cm = pow(z + z0, 2);
									for (int j0 = 0; j0 < nt; j0++) {
										constexpr double dphi = M_PI / nt / 2.0;
										const double phi = (j0 + 0.5) * dphi;
										const double b_cos_phi = b * cos(phi);
										const double d = 2.0 * dphi * rho[i0][k0] * R0 * dr * dz;
										this_phi += d / sqrt(cp + a - b_cos_phi);
										this_phi += d / sqrt(cm + a - b_cos_phi);
										this_phi += d / sqrt(cp + a + b_cos_phi);
										this_phi += d / sqrt(cm + a + b_cos_phi);
									}
								}
							}
						}
						phi[i][k] = -this_phi;
					}
				}
			}
		}

		// Solve interior
		{
			int iter = 0;
			double error;
			double toler = 1.0e-12;
			constexpr double dz2 = 1.0 / (nz * nz);
			constexpr double dr2 = 1.0 / (nr * nr);
			const double den0 = 2.0 * (dz2 + dr2);
			static array_type next_phi;
			next_phi = phi;
			do {
				error = 0.0;
				for (int i = 0; i < nr - 1; i++) {
					const double den1 = i != 0 ? 0.0 : -dz2 * (1.0 - 0.5 / (i + 0.5));
					for (int k = 0; k < nz - 1; k++) {
						const double den2 = k != 0 ? 0.0 : -dr2;
						const double num_xp = dz2 * (1.0 + 0.5 / (i + 0.5)) * phi[i + 1][k];
						const double num_xm = i != 0 ? dz2 * (1.0 - 0.5 / (i + 0.5)) * phi[i - 1][k] : 0.0;
						const double num_zp = dr2 * phi[i][k + 1];
						const double num_zm = k != 0 ? dr2 * phi[i][k - 1] : 0.0;
						const double num_den = -4.0 * M_PI * dz2 * dr2 * rho[i][k];
						next_phi[i][k] = (num_xp + num_xm + num_zp + num_zm + num_den) / (den0 + den1 + den2);
						error += std::pow(next_phi[i][k] - phi[i][k], 2);
					}
				}
				phi = next_phi;
				error = std::sqrt(error);
				iter++;
			} while (error > toler);
		}
		// next rho
		double W = 0.0;
		double T = 0.0;
		{
			static array_type next_rho;
			const double Rb = (rei + 0.5) * dr;
			const double Rbinv2 = 1.0 / Rb / Rb;
			const double phi0 = phi[0][zei];
			const double phic = phi[0][0];
			const double o2 = 2.0 * (phi[rei][0] - phi0) * Rbinv2;
//			printf( "%e %e %i %i\n", phi[rei][0], phi0, rei, zei);
			omega = std::sqrt(o2);
			K = (phi0 - phic) / (n + 1);
			for (int i = 0; i < nr - 1; i++) {
				for (int k = 0; k < nz - 1; k++) {
					constexpr double w = 1.0;
					const double R = (i + 0.5) * dr;
					const double R2 = R * R;
					const double new_rho = std::pow(std::max(phi0 - phi[i][k] + 0.5 * R2 * o2, 0.0) / (K * (n + 1.0)), n);
					rho[i][k] = (1.0 - w) * rho[i][k] + w * new_rho;
					const double P = K * std::pow(rho[i][k], 1.0 + 1.0 / n);
					T += (3.0 * P + o2 * R2 * rho[i][k]) * R * dr * dz;
					W += (0.5 * rho[i][k] * phi[i][k]) * R * dr * dz;
				}
			}
		}
		double virial = (T + W) / (T - W);
		printf("%e %e %e %e\n", omega, virial, T, W);

		scf_iter++;
		if (scf_iter > 20) {
			break;
		}
	} while (true);
// output phi
	{
		FILE* fp = fopen("phi.dat", "wt");
		for (int i = 0; i < nr; i++) {
			//		int i = nr - 1;
			for (int k = 0; k < nz; k++) {
				const double R = (i + 0.5) * dr;
				const double z = (k + 0.5) * dz;
				fprintf(fp, "%e %e %e %e\n", R, z, phi[i][k], rho[i][k]);
			}
		}
		fclose(fp);
		int i0, k0;
		fp = fopen("rotating_star.bin", "wb");
		fwrite(&nr, 1, sizeof(std::int64_t), fp);
		fwrite(&nz, 1, sizeof(std::int64_t), fp);
		fwrite(&omega, 1, sizeof(double), fp);
		for (int i = 0; i < 2 * nr; i++) {
			i0 = i - nr >= 0 ? (i - nr) : -i + nr - 1;
			for (int k = 0; k < 2 * nr; k++) {
				k0 = k - nz >= 0 ? (k - nz) : -k + nz - 1;
				const double d = rho[i0][k0];
				const double e = n * K * std::pow(d, 1.0 + 1.0 / n);
				fwrite(&d, 1, sizeof(double), fp);
				fwrite(&e, 1, sizeof(double), fp);
			}
		}
		fclose(fp);
	}

	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

	return 0;
}
