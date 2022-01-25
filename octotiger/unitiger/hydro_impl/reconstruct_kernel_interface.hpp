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
void hydro_pre_recon_cpu_kernel(const double* __restrict__ X, safe_real omega,
    bool angmom, double* __restrict__ combined_u, const int nf, const int n_species_);
void reconstruct_experimental(const safe_real omega, const size_t nf_, const int angmom_index_,
    const int* __restrict__ smooth_field_, const int* __restrict__ disc_detect_ ,
    double* __restrict__ combined_q, double* __restrict__ combined_x,
    double* __restrict__ combined_u, double* __restrict__ AM, const double dx,
    const double* __restrict__ cdiscs);
// Vc kernel
#ifdef __x86_64__
void reconstruct_cpu_kernel(const safe_real omega, const size_t nf_, const int angmom_index_,
    const std::vector<bool>& smooth_field_, const std::vector<bool>& disc_detect_,
    double* __restrict__ combined_q, double* __restrict__ combined_x,
    double* __restrict__ combined_u, const double dx,
    const std::vector<std::vector<safe_real>>& cdiscs);
#endif
#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
void launch_reconstruct_cuda(
    stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
    double omega, int nf_, int angmom_index_,
    int* smooth_field_, int* disc_detect_ ,
    double* combined_q, double* combined_x,
    double* combined_u, double* AM, double dx,
    double* cdiscs, int n_species_);
void convert_find_contact_discs(const double* __restrict__ combined_u, double* __restrict__ disc, const double A_, const double B_, const double fgamma_, const double de_switch_1);
void launch_find_contact_discs_cuda(stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor, double* combined_u, double *device_P, double* disc, double A_, double B_, double fgamma_, double de_switch_1);
void launch_hydro_pre_recon_cuda(stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor, 
    double* device_X, double omega, bool angmom, double* device_u, int nf, int n_species_);
#endif
