#if !defined(KERNEL_INTERFACE_HPP_)
#define RADIATION_KERNEL_INTERFACE_HPP_

#include "octotiger/defs.hpp"
#include "octotiger/radiation/cpu_kernel.hpp"
#include "octotiger/radiation/cuda_kernel.hpp"
#include "octotiger/real.hpp"

#include <array>
#include <vector>

namespace octotiger { namespace radiation {
    template <integer er_i, integer fx_i, integer fy_i, integer fz_i>
    void radiation_kernel(integer const d, std::vector<real> const& rho,
        std::vector<real>& sx, std::vector<real>& sy, std::vector<real>& sz,
        std::vector<real>& egas, std::vector<real>& tau, real const fgamma,
        std::array<std::vector<real>, NRF> U, std::vector<real> mmw,
        std::vector<real> X_spc, std::vector<real> Z_spc, real dt,
        real const clightinv)
    {
        return radiation_cpu_kernel<er_i, fx_i, fy_i, fz_i>(
            d, rho, sx, sy, sz, egas, tau, fgamma, U, mmw, X_spc, Z_spc, dt,
            clightinv);
    }

}}

#endif    // KERNEL_INTERFACE_HPP_
