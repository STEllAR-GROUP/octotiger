//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if OCTOTIGER_HAVE_CUDA && !defined(RADIATION_CUDA_KERNEL_HPP_)
#define RADIATION_CUDA_KERNEL_HPP_

#include "octotiger/defs.hpp"

#include "octotiger/math/Real.hpp"

#include <array>
#include <vector>

namespace octotiger { namespace radiation {
    template <integer er_i, integer fx_i, integer fy_i, integer fz_i>
    void radiation_cuda_kernel(integer const d, std::vector<Real> const& rho,
        std::vector<Real>& sx, std::vector<Real>& sy, std::vector<Real>& sz,
        std::vector<Real>& egas, std::vector<Real>& tau, Real const fgamma,
        std::array<std::vector<Real>, NRF> U, std::vector<Real> mmw,
        std::vector<Real> X_spc, std::vector<Real> Z_spc, Real dt,
        Real const clightinv)
    {
        throw std::logic_error{"Not Implemented"};
    }
}}

#endif    // RADIATION_CUDA_KERNEL_HPP_
