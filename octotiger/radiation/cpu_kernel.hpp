#if !defined(RADIATION_CPU_KERNEL_HPP_)
#define RADIATION_CPU_KERNEL_HPP_

#include "octotiger/defs.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/radiation/implicit.hpp"
#include "octotiger/radiation/opacities.hpp"
#include "octotiger/real.hpp"
#include "octotiger/roe.hpp"
#include "octotiger/safe_math.hpp"
#include "octotiger/space_vector.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

namespace octotiger { namespace radiation {
    namespace detail {
        template <typename F>
        void abort_if_solver_not_converged(real const eg_t0, real E0,
            F const test, real const E, real const eg_t)
        {
            // Bisection root finding method
            // Indices of max, mid, and min
            real de_max = eg_t0;
            real de_mid = 0.0;
            real de_min = -E0;
            // Values of max, mid, and min
            real f_min = test(de_min);
            real f_mid = test(de_mid);
            // Max iterations
            constexpr std::size_t max_iterations = 50;
            // Errors
            real const error_tolerance = 1.0e-9;

            for (std::size_t i = 0; i < max_iterations; ++i)
            {
                // Root solver has converged if error is smaller that error tolerance
                real const error =
                    std::max(std::abs(f_mid), std::abs(f_min)) / (E + eg_t);
                if (error < error_tolerance)
                {
                    return;
                }

                // If signs don't match, continue search in the lower half
                if ((f_min < 0) != (f_mid < 0))
                {
                    de_max = de_mid;
                    de_mid = 0.5 * (de_min + de_max);
                    f_mid = test(de_mid);
                }
                // Continue search in the upper half
                else
                {
                    de_min = de_mid;
                    de_mid = 0.5 * (de_min + de_max);
                    f_min = f_mid;
                    f_mid = test(de_mid);
                }
            }
            // Error is not smaller that error tolerance after performed iterations. Abort.
            printf("Implicit radiation solver failed to converge\n");
            abort();
        }    // abort_if_solver_not_converged

        std::pair<real, space_vector> implicit_radiation_step(real E0, real& e0,
            space_vector F0, space_vector u0, real rho, real mmw, real X,
            real Z, real dt)
        {
            real const c = physcon().c;
            real kp = kappa_p(rho, e0, mmw, X, Z);
            real kr = kappa_R(rho, e0, mmw, X, Z);
            real const rhoc2 = rho * c * c;

            E0 /= rhoc2;
            F0 = F0 / (rhoc2 * c);
            e0 /= rhoc2;
            u0 = u0 / c;
            kp *= dt * c;
            kr *= dt * c;

            auto const B = [rho, mmw, c, rhoc2](real e) {
                return (4.0 * M_PI / c) * B_p(rho, e * rhoc2, mmw) / rhoc2;
            };

            auto E = E0;
            auto eg_t =
                e0 + 0.5 * (u0[0] * u0[0] + u0[1] * u0[1] + u0[2] * u0[2]);
            auto F = F0;
            auto u = u0;
            real ei;
            auto const eg_t0 = eg_t;

            real u2_0 = 0.0;
            real F2_0 = 0.0;
            for (int d = 0; d < NDIM; d++)
            {
                u2_0 += u[d] * u[d];
                F2_0 += F[d] * F[d];
            }
            // printf( "%e %e\n", (double) u2_0, (double) (F2_0/E/E));
            auto const test = [&](real de) {
                E = E0 + de;
                real u2 = 0.0;
                real udotF = 0.0;
                for (int d = 0; d < NDIM; d++)
                {
                    auto const num =
                        F0[d] + (4.0 / 3.0) * kr * E * (u0[d] + F0[d]);
                    auto const den = 1.0 + kr * (1.0 + (4.0 / 3.0) * E);
                    auto const deninv = 1.0 / den;
                    F[d] = num * deninv;
                    u[d] = u0[d] + F0[d] - F[d];
                    u2 += u[d] * u[d];
                    udotF += F[d] * u[d];
                }
                ei = std::max(eg_t0 - E + E0 - 0.5 * u2, real(0.0));
                real const b = B(ei);
                real f = E - E0;
                f += (kp * (E - b) + (kr - 2.0 * kp) * udotF);
                eg_t = eg_t0 + E0 - E;
                return f;
            };

            abort_if_solver_not_converged(eg_t0, E0, test, E, eg_t);

            ei = eg_t - 0.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
            e0 = ei * rhoc2;
            auto const dtinv = 1.0 / dt;

            return std::make_pair(
                real((E - E0) * dtinv * rhoc2), ((F - F0) * dtinv * rhoc2 * c));
        }    // implicit_radiation_step
    }        // namespace detail

    template <integer er_i, integer fx_i, integer fy_i, integer fz_i>
    void radiation_cpu_kernel(integer const d, std::vector<real> const& rho,
        std::vector<real>& sx, std::vector<real>& sy, std::vector<real>& sz,
        std::vector<real>& egas, std::vector<real>& tau, real const fgamma,
        std::array<std::vector<real>, NRF> U, std::vector<real> mmw,
        std::vector<real> X_spc, std::vector<real> Z_spc, real dt,
        real const clightinv)
    {
        for (integer i = RAD_BW; i != RAD_NX - RAD_BW; ++i)
        {
            for (integer j = RAD_BW; j != RAD_NX - RAD_BW; ++j)
            {
                for (integer k = RAD_BW; k != RAD_NX - RAD_BW; ++k)
                {
                    integer const iiih = hindex(i + d, j + d, k + d);
                    integer const iiir = rindex(i, j, k);
                    real const den = rho[iiih];
                    real const deninv = INVERSE(den);
                    real vx = sx[iiih] * deninv;
                    real vy = sy[iiih] * deninv;
                    real vz = sz[iiih] * deninv;

                    // Compute e0 from dual energy formalism
                    real e0 = egas[iiih];
                    e0 -= 0.5 * vx * vx * den;
                    e0 -= 0.5 * vy * vy * den;
                    e0 -= 0.5 * vz * vz * den;
                    if (opts().eos == WD)
                    {
                        e0 -= ztwd_energy(den);
                    }
                    if (e0 < egas[iiih] * de_switch2)
                    {
                        e0 = std::pow(tau[iiih], fgamma);
                    }
                    real E0 = U[er_i][iiir];
                    space_vector F0;
                    space_vector u0;
                    F0[0] = U[fx_i][iiir];
                    F0[1] = U[fy_i][iiir];
                    F0[2] = U[fz_i][iiir];
                    u0[0] = vx;
                    u0[1] = vy;
                    u0[2] = vz;
                    real E1 = E0;
                    space_vector F1 = F0;
                    space_vector u1 = u0;
                    real e1 = e0;

                    auto const ddt = detail::implicit_radiation_step(E1, e1, F1,
                        u1, den, mmw[iiir], X_spc[iiir], Z_spc[iiir], dt);
                    real const dE_dt = ddt.first;
                    real const dFx_dt = ddt.second[0];
                    real const dFy_dt = ddt.second[1];
                    real const dFz_dt = ddt.second[2];

                    // Accumulate derivatives
                    U[er_i][iiir] += dE_dt * dt;
                    U[fx_i][iiir] += dFx_dt * dt;
                    U[fy_i][iiir] += dFy_dt * dt;
                    U[fz_i][iiir] += dFz_dt * dt;

                    egas[iiih] -= dE_dt * dt;
                    sx[iiih] -= dFx_dt * dt * clightinv * clightinv;
                    sy[iiih] -= dFy_dt * dt * clightinv * clightinv;
                    sz[iiih] -= dFz_dt * dt * clightinv * clightinv;

                    // Find tau with dual energy formalism
                    real e = egas[iiih];
                    e -= 0.5 * sx[iiih] * sx[iiih] * deninv;
                    e -= 0.5 * sy[iiih] * sy[iiih] * deninv;
                    e -= 0.5 * sz[iiih] * sz[iiih] * deninv;
                    if (opts().eos == WD)
                    {
                        e -= ztwd_energy(den);
                    }
                    if (e < de_switch1 * egas[iiih])
                    {
                        e = e1;
                    }
                    if (opts().problem == MARSHAK)
                    {
                        egas[iiih] = e;
                        sx[iiih] = sy[iiih] = sz[iiih] = 0;
                    }
                    if (U[er_i][iiir] <= 0.0)
                    {
                        printf("Er = %e %e %e %e\n", E0, E1, U[er_i][iiir], dt);
                        abort();
                    }
                    e = std::max(e, decltype(e)(0));
                    tau[iiih] = std::pow(e, INVERSE(fgamma));
                    if (U[er_i][iiir] <= 0.0)
                    {
                        printf("2231242!!! %e %e %e \n", E0, U[er_i][iiir],
                            dE_dt * dt);
                        abort();
                    }
                    if (opts().problem == MARSHAK)
                    {
                        sx[iiih] = sy[iiih] = sz[iiih] = 0;
                        egas[iiih] = e;
                    }
                }
            }
        }
    }    // radiation_cpu_kernel
}}       // namespace octotiger::radiation

#endif    // RADIATION_CPU_KERNEL_HPP_
