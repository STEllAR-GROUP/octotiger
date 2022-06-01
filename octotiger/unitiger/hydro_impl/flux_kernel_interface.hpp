#pragma once

#include <array>
#include <vector>

#include "octotiger/cuda_util/cuda_global_def.hpp"
#include "octotiger/hydro_defs.hpp"
#include "octotiger/unitiger/hydro.hpp"
#include "octotiger/unitiger/safe_real.hpp"

#ifdef OCTOTIGER_HAVE_CUDA
#include "octotiger/cuda_util/cuda_helper.hpp"
#include <buffer_manager.hpp>
#include <cuda_buffer_util.hpp>
#include <cuda_runtime.h>
#include <stream_manager.hpp>
#endif
#ifdef OCTOTIGER_HAVE_HIP
#include "octotiger/cuda_util/cuda_helper.hpp"
#include <buffer_manager.hpp>
#include <hip_buffer_util.hpp>
#include <hip/hip_runtime.h>
#include <stream_manager.hpp>
#endif

#include <boost/container/vector.hpp>    // to get non-specialized vector<bool>

timestep_t flux_kernel_interface(const hydro::recon_type<NDIM>& Q, hydro::flux_type& F,
    hydro::x_type& X, safe_real omega, const size_t nf_);

#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
#include <aggregation_manager.hpp>
using aggregated_executor_t = Aggregated_Executor<hpx::cuda::experimental::cuda_executor>::Executor_Slice;
#endif
#if defined(OCTOTIGER_HAVE_CUDA) 
void launch_flux_cuda_kernel_post(aggregated_executor_t& executor,
    dim3 const grid_spec, dim3 const threads_per_block, void *args[]);
#endif
#if defined(OCTOTIGER_HAVE_HIP) 
void launch_flux_hip_kernel_post(
    aggregated_executor_t& executor,
    dim3 const grid_spec, dim3 const threads_per_block, double* device_q, double* device_x,
    double* device_f, double* device_amax, int* device_amax_indices, int* device_amax_d,
    const bool* masks, const double omega, const double *dx, const double A_, const double B_,
    const size_t nf_, const double fgamma, const double de_switch_1, const int number_blocks);
#endif
