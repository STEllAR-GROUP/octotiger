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

// helpers for using vectortype specialization functions
template <typename double_t, typename cond_t>
CUDA_GLOBAL_METHOD inline void select_wrapper(
    double_t& target, const cond_t cond, const double_t& tmp1, const double_t& tmp2) {
    target = cond ? tmp1 : tmp2;
}
template <typename T>
CUDA_GLOBAL_METHOD inline T max_wrapper(const T& tmp1, const T& tmp2) {
    return max(tmp1, tmp2);
}
template <typename T>
CUDA_GLOBAL_METHOD inline T min_wrapper(const T& tmp1, const T& tmp2) {
    return min(tmp1, tmp2);
}
template <typename T>
CUDA_GLOBAL_METHOD inline T sqrt_wrapper(const T& tmp1) {
    return std::sqrt(tmp1);
}
template <typename T>
CUDA_GLOBAL_METHOD inline T pow_wrapper(const T& tmp1, const double& tmp2) {
    return std::pow(tmp1, tmp2);
}
template <typename T>
CUDA_GLOBAL_METHOD inline T asinh_wrapper(const T& tmp1) {
    return std::asinh(tmp1);
}
template <typename T>
CUDA_GLOBAL_METHOD inline bool skippable(const T& tmp1) {
    return !tmp1;
}
template <typename T>
CUDA_GLOBAL_METHOD inline T load_value(const double* data, const size_t index) {
    return data[index];
}
template <typename T>
CUDA_GLOBAL_METHOD inline void store_value(
    double* data, const size_t index, const T& value) {
    data[index] = value;
}
template <typename T, typename container_t>
CUDA_GLOBAL_METHOD inline T load_value(const container_t &data, const size_t index) {
    return data[index];
}
template <typename T, typename container_t>
CUDA_GLOBAL_METHOD inline void store_value(
    container_t &data, const size_t index, const T& value) {
    data[index] = value;
}

boost::container::vector<bool> create_masks();
