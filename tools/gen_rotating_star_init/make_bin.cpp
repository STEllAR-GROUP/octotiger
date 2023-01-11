//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <array>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

constexpr int nr = 100;
constexpr int nz = 100;

constexpr int nt = nr / 4;
constexpr double zedge = 0.6;
constexpr double redge = 0.9;
constexpr int zei = static_cast<int>(zedge * nz);
constexpr int rei = static_cast<int>(redge * nr);

using namespace std;

using array_type = std::array<std::array<double, nz>, nr>;

int main()
{
    //	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

    static array_type phi;
    static array_type rho;
    const double dr = 1.0 / nr;
    const double dz = 1.0 / nz;
    double omega, K;
    const double n = 1.5;

    // initialize
    {
        for (int i = 0; i < nr; i++)
        {
            for (int k = 0; k < nz; k++)
            {
                phi[i][k] = 0.0;
                rho[i][k] = 0.0;
                //const double R = (i + 0.5) / nr;
                //const double z = (k + 0.5) / nz;
                if (i == 0 && k == 0)
                {
                    rho[i][k] = 1.0;
                }
            }
        }
    }

    //double scf_error = 0.0;
    int scf_iter = 0;
    printf("omega, virial, T, W, K\n");

    do
    {
        // Solve boundaries
        {
            for (int i = 0; i < nr; i++)
            {
                const double R = (i + 0.5) / nr;
                for (int k = 0; k < nz; k++)
                {
                    const double z = (k + 0.5) / nz;
                    if (i == nr - 1 || k == nz - 1)
                    {
                        double this_phi = 0.0;
                        for (int i0 = 0; i0 < nr - 1; i0++)
                        {
                            const double R0 = (i0 + 0.5) / nr;
                            const double a = R0 * R0 + R * R;
                            const double b = 2.0 * R0 * R;
                            for (int k0 = 0; k0 < nz - 1; k0++)
                            {
                                if (rho[i0][k0] > 0.0)
                                {
                                    const double z0 = (k0 + 0.5) / nz;
                                    const double cp = pow(z - z0, 2);
                                    const double cm = pow(z + z0, 2);
                                    for (int j0 = 0; j0 < nt; j0++)
                                    {
                                        constexpr double dphi = M_PI / nt / 2.0;
                                        const double phi = (j0 + 0.5) * dphi;
                                        const double b_cos_phi = b * cos(phi);
                                        const double d = 2.0 * dphi *
                                            rho[i0][k0] * R0 * dr * dz;
                                        this_phi +=
                                            d / sqrt(cp + a - b_cos_phi);
                                        this_phi +=
                                            d / sqrt(cm + a - b_cos_phi);
                                        this_phi +=
                                            d / sqrt(cp + a + b_cos_phi);
                                        this_phi +=
                                            d / sqrt(cm + a + b_cos_phi);
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
            double error;
            double toler = 1.0e-8;
            constexpr double dz2 = 1.0 / (nz * nz);
            constexpr double dr2 = 1.0 / (nr * nr);
            const double den0 = 2.0 * (dz2 + dr2);
            static array_type next_phi;
            next_phi = phi;
            const double rhoc0 = -4.0 * M_PI * dz2 * dr2;
            do
            {
                for (int iter = 0; iter < 16; iter++)
                {
                    for (int i = 0; i < nr - 1; i++)
                    {
                        const double den1 =
                            i != 0 ? 0.0 : -dz2 * (1.0 - 0.5 / (i + 0.5));
                        const double c_xp = dz2 * (1.0 + 0.5 / (i + 0.5));
                        const double c_xm =
                            i != 0 ? dz2 * (1.0 - 0.5 / (i + 0.5)) : 0.0;
                        for (int k = 0; k < nz - 1; k++)
                        {
                            const double den2 = k != 0 ? 0.0 : -dr2;
                            const double num_xp = c_xp * next_phi[i + 1][k];
                            const double num_xm = c_xm * next_phi[std::max(i - 1,0)][k];
                            const double num_zm =
                                (k != 0 ? dr2 : 0.0) * next_phi[i][std::max(k - 1,0)];
                            const double num_zp = dr2 * next_phi[i][k + 1];
                            const double num_den = rhoc0 * rho[i][k];
                            next_phi[i][k] =
                                (num_xp + num_xm + num_zp + num_zm + num_den) /
                                (den0 + den1 + den2);
                        }
                    }
                    if (iter != 15)
                    {
                        phi = next_phi;
                    }
                }
                error = 0.0;
                for (int i = 0; i < nr - 1; i++)
                {
                    for (int k = 0; k < nz - 1; k++)
                    {
                        error += std::pow(next_phi[i][k] - phi[i][k], 2);
                    }
                }
                error = std::sqrt(error);
                phi = next_phi;
            } while (error > toler);
        }
        // next rho
        double W = 0.0;
        double T = 0.0;
        {
            //static array_type next_rho;
            const double Rb = (rei + 0.5) * dr;
            const double Rbinv2 = 1.0 / Rb / Rb;
            const double phi0 = phi[0][zei];
            const double phic = phi[0][0];
            const double o2 = 2.0 * (phi[rei][0] - phi0) * Rbinv2;
            //			printf( "%e %e %i %i\n", phi[rei][0], phi0, rei, zei);
            omega = std::sqrt(o2);
            K = (phi0 - phic) / (n + 1);
            for (int i = 0; i < nr - 1; i++)
            {
                for (int k = 0; k < nz - 1; k++)
                {
                    constexpr double w = 1.0;
                    const double R = (i + 0.5) * dr;
                    const double R2 = R * R;
                    const double new_rho = std::pow(
                        std::max(phi0 - phi[i][k] + 0.5 * R2 * o2, 0.0) /
                            (K * (n + 1.0)),
                        n);
                    rho[i][k] = (1.0 - w) * rho[i][k] + w * new_rho;
                    const double P = K * std::pow(rho[i][k], 1.0 + 1.0 / n);
                    T += (3.0 * P + o2 * R2 * rho[i][k]) * R * dr * dz;
                    W += (0.5 * rho[i][k] * phi[i][k]) * R * dr * dz;
                }
            }
        }
        double virial = (T + W) / (T - W);
        printf("%e %e %e %e %e\n", omega, virial, T, W, K);

        scf_iter++;
        if (scf_iter > 50)
        {
            break;
        }
    } while (true);
    // output phi
    {
        FILE* fp = fopen("phi.dat", "wt");
        for (int i = 0; i < nr; i++)
        {
            //		int i = nr - 1;
            for (int k = 0; k < nz; k++)
            {
                const double R = (i + 0.5) * dr;
                const double z = (k + 0.5) * dz;
                fprintf(fp, "%e %e %e %e\n", R, z, phi[i][k], rho[i][k]);
            }
        }
        fclose(fp);
        int i0, k0;
        fp = fopen("rotating_star.bin", "wb");
        fwrite(&nr, 1, sizeof(decltype(nr)), fp);
        fwrite(&nz, 1, sizeof(decltype(nz)), fp);
        fwrite(&omega, 1, sizeof(double), fp);
        for (int i = 0; i < 2 * nr; i++)
        {
            i0 = i - nr >= 0 ? (i - nr) : -i + nr - 1;
            for (int k = 0; k < 2 * nr; k++)
            {
                k0 = k - nz >= 0 ? (k - nz) : -k + nz - 1;
                const double d = rho[i0][k0];
                const double e = n * K * std::pow(d, 1.0 + 1.0 / n);
                fwrite(&d, 1, sizeof(double), fp);
                fwrite(&e, 1, sizeof(double), fp);
            }
        }
        fclose(fp);
    }

    printf("\rDone!\n");

    return 0;
}
