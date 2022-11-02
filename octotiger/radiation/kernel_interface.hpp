//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "octotiger/defs.hpp"
#include "octotiger/radiation/cpu_kernel.hpp"
#include "octotiger/radiation/cuda_kernel.hpp"
#include "octotiger/real.hpp"

#include <hpx/include/run_as.hpp>

#include <array>
#include <vector>

namespace octotiger { namespace radiation {
#if defined(OCTOTIGER_DUMP_RADIATION_CASES)
    namespace dumper {
        constexpr char const* basepath = OCTOTIGER_DUMP_DIR "/octotiger-radiation-";

        void save_v(std::ostream& os, std::vector<double> const& v)
        {
            std::size_t size = v.size();
            os.write(reinterpret_cast<char*>(&size), sizeof(std::size_t));

            os.write(
                reinterpret_cast<char const*>(&v[0]), size * sizeof(double));
        }

        void save_a(
            std::ostream& os, std::array<std::vector<double>, NRF> const& a)
        {
            std::size_t size = a.size();
            os.write(reinterpret_cast<char*>(&size), sizeof(std::size_t));

            for (auto const& e : a)
            {
                save_v(os, e);
            }
        }

        void save_i(std::ostream& os, std::int64_t const i)
        {
            os.write(reinterpret_cast<char const*>(&i), sizeof(std::int64_t));
        }

        void save_d(std::ostream& os, double const d)
        {
            os.write(reinterpret_cast<char const*>(&d), sizeof(double));
        }

        void save_case_args(std::size_t const index,
            std::int64_t const& opts_eos,
            std::int64_t const& opts_problem,
            double const& opts_dual_energy_sw1,
            double const& opts_dual_energy_sw2,
            double const& physcon_A,
            double const& physcon_B,
            double const& physcon_c,
            double const& fgamma,
            double const& dt,
            double const& clightinv,
            std::int64_t const& er_i,
            std::int64_t const& fx_i,
            std::int64_t const& fy_i,
            std::int64_t const& fz_i,
            std::int64_t const& d,
            std::vector<double> const& sx,
            std::vector<double> const& sy,
            std::vector<double> const& sz,
            std::vector<double> const& egas,
            std::vector<double> const& ei,
            std::array<std::vector<double>, NRF> const& U,
            std::vector<double> const& rho,
            std::vector<double> const& X_spc,
            std::vector<double> const& Z_spc,
            std::vector<double> const& mmw)
        {
            std::string const args_fn = std::string{basepath} +
                std::to_string(index) + std::string{".args"};
            std::ofstream os{args_fn, std::ios::binary};

            if (!os)
            {
                throw std::runtime_error(hpx::util::format(
                    "error: cannot open args file \"{}\".", args_fn));
            }

            save_i(os, opts_eos);
            save_i(os, opts_problem);
            save_d(os, opts_dual_energy_sw1);
            save_d(os, opts_dual_energy_sw2);
            save_d(os, physcon_A);
            save_d(os, physcon_B);
            save_d(os, physcon_c);
            save_d(os, fgamma);
            save_d(os, dt);
            save_d(os, clightinv);
            save_i(os, er_i);
            save_i(os, fx_i);
            save_i(os, fy_i);
            save_i(os, fz_i);
            save_i(os, d);
            save_v(os, sx);
            save_v(os, sy);
            save_v(os, sz);
            save_v(os, egas);
            save_v(os, ei);
            save_a(os, U);
            save_v(os, rho);
            save_v(os, X_spc);
            save_v(os, Z_spc);
            save_v(os, mmw);
        }

        void save_case_outs(std::size_t const index,
            std::vector<double> const& sx,
            std::vector<double> const& sy,
            std::vector<double> const& sz,
            std::vector<double> const& egas,
            std::vector<double> const& ei,
            std::array<std::vector<double>, NRF> const& U)
        {
            std::string const outs_fn = std::string{basepath} +
                std::to_string(index) + std::string{".outs"};
            std::ofstream os{outs_fn, std::ios::binary};

            if (!os)
            {
                throw std::runtime_error(hpx::util::format(
                    "error: cannot open outs file \"{}\".", outs_fn));
            }

            save_v(os, sx);
            save_v(os, sy);
            save_v(os, sz);
            save_v(os, egas);
            save_v(os, ei);
            save_a(os, U);
        }
    }
#endif

    template <integer er_i, integer fx_i, integer fy_i, integer fz_i>
    void radiation_kernel(integer const d,
        std::vector<real> const& rho,
        std::vector<real>& sx,
        std::vector<real>& sy,
        std::vector<real>& sz,
        std::vector<real>& egas,
        std::vector<real>& ei,
        real const fgamma,
        std::vector<std::vector<real>>& U,
        std::vector<real> const& abar,
        std::vector<real> const& zbar,
        std::vector<real> const& X_spc,
        std::vector<real> const& Z_spc,
        real dt,
        real const clightinv,
		stellar_eos* eos)
    {
#if defined(OCTOTIGER_DUMP_RADIATION_CASES)
        static std::atomic_size_t next_index(0);
        std::size_t index = next_index++;

        hpx::threads::run_as_os_thread([&]() {
            dumper::save_case_args(index, opts().eos, opts().problem,
                opts().dual_energy_sw1, opts().dual_energy_sw2, physcon().A,
                physcon().B, physcon().c, fgamma, dt, clightinv, er_i, fx_i,
                fy_i, fz_i, d, sx, sy, sz, egas, ei, U, rho, X_spc, Z_spc,
                mmw);
        }).get();
#endif

        radiation_cpu_kernel<er_i, fx_i, fy_i, fz_i>(d, rho, sx, sy, sz, egas,
            ei, fgamma, U, abar, zbar, X_spc, Z_spc, dt, clightinv, eos);

#if defined(OCTOTIGER_DUMP_RADIATION_CASES)
        hpx::threads::run_as_os_thread([&]() {
            dumper::save_case_outs(index, sx, sy, sz, egas, ei, U);
        }).get();
#endif
    }

}}
