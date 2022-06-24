//  Copyright (c) 2020-2022 Gregor Daiß
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#pragma once

//#define TVD_TEST

#include <array>
#include <vector>

#include "octotiger/cuda_util/cuda_global_def.hpp"
#include "octotiger/hydro_defs.hpp"
#include "octotiger/unitiger/hydro.hpp"
#include "octotiger/unitiger/safe_real.hpp"

#include "octotiger/unitiger/physics.hpp"
#include "octotiger/unitiger/physics_impl.hpp"

#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp"

#ifdef OCTOTIGER_HAVE_CUDA
#include <cuda_buffer_util.hpp>
#include <cuda_runtime.h>
#include <stream_manager.hpp>
#include "octotiger/cuda_util/cuda_helper.hpp"
#endif

#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
__global__ void discs_phase1(double* __restrict__ P, double* __restrict__ combined_u,
    const double A_, const double B_, const double fgamma_, const double
    de_switch_1, const int nf);
__global__ void discs_phase2(
    double* __restrict__ disc, double* __restrict__ P, const double
    fgamma_, const int ndir);

#include <aggregation_manager.hpp>
using aggregated_executor_t = Aggregated_Executor<hpx::cuda::experimental::cuda_executor>::Executor_Slice;

void launch_reconstruct_cuda(
    aggregated_executor_t& executor,
    double omega, int nf_, int angmom_index_,
    int* smooth_field_, int* disc_detect_ ,
    double* combined_q, double* combined_x,
    double* combined_u, double* AM, double *dx,
    double* cdiscs, int n_species_);
void launch_find_contact_discs_cuda(aggregated_executor_t& executor, double* combined_u, double *device_P, double* disc, double A_, double B_, double fgamma_, double de_switch_1, const int nf);
void launch_hydro_pre_recon_cuda(aggregated_executor_t& executor, 
    double* device_X, double omega, bool angmom, double* device_u, int nf, int n_species_);
#endif
