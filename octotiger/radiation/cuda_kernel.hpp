//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if OCTOTIGER_HAVE_CUDA && !defined(RADIATION_CUDA_KERNEL_HPP_)
#define RADIATION_CUDA_KERNEL_HPP_

#include "octotiger/defs.hpp"

#include "octotiger/real.hpp"

#include <array>
#include <vector>

namespace octotiger { namespace radiation {
    template <integer er_i, integer fx_i, integer fy_i, integer fz_i>
    void radiation_cuda_kernel(integer const d, std::vector<real> const& rho,
        std::vector<real>& sx, std::vector<real>& sy, std::vector<real>& sz,
        std::vector<real>& egas, std::vector<real>& ein, real const fgamma,
        std::array<std::vector<real>, NRF> U, std::vector<real> mmw,
        std::vector<real> X_spc, std::vector<real> Z_spc, real dt,
        real const clightinv)
    {
        throw std::logic_error{"Not Implemented"};
    }
}}

#endif    // RADIATION_CUDA_KERNEL_HPP_
