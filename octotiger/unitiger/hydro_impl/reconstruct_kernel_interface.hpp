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

template <typename T>
CUDA_GLOBAL_METHOD inline T copysign_wrapper(const T& tmp1, const T& tmp2) {
    return std::copysign(tmp1, tmp2);
}
template <typename T>
CUDA_GLOBAL_METHOD inline T abs_wrapper(const T& tmp1) {
    return std::abs(tmp1);
}
template <typename T>
CUDA_GLOBAL_METHOD inline T minmod_wrapper(const T& a, const T& b) {
    return (copysign_wrapper<T>(0.5, a) + copysign_wrapper<T>(0.5, b)) *
        min_wrapper<T>(abs_wrapper<T>(a), abs_wrapper<T>(b));
}
template <typename T>
CUDA_GLOBAL_METHOD inline T minmod_theta_wrapper(const T& a, const T& b, const T& c) {
    return minmod_wrapper<T>(c * minmod_wrapper<T>(a, b), 0.5 * (a + b));
}
// Vc kernel
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
