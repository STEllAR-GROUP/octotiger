#pragma once

#include "octotiger/cuda_util/cuda_global_def.hpp"


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
    return sqrt(tmp1);
}
template <typename T>
CUDA_GLOBAL_METHOD inline T pow_wrapper(const T& tmp1, const double& tmp2) {
    return pow(tmp1, tmp2);
}
template <typename T>
CUDA_GLOBAL_METHOD inline T asinh_wrapper(const T& tmp1) {
    return asinh(tmp1);
}
template <typename T>
CUDA_GLOBAL_METHOD inline T copysign_wrapper(const T& tmp1, const T& tmp2) {
    return copysign(tmp1, tmp2);
}
template <typename T>
CUDA_GLOBAL_METHOD inline T abs_wrapper(const T& tmp1) {
    return abs(tmp1);
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
template <typename T>
CUDA_GLOBAL_METHOD inline T limiter_wrapper(const T& a, const T& b) {
    return minmod_theta_wrapper<T>(a, b, 64. / 37.);
}
template <typename T>
CUDA_GLOBAL_METHOD inline bool skippable(const T& tmp1) {
    return !tmp1;
}
template <typename T>
CUDA_GLOBAL_METHOD inline T load_value(const double* __restrict__ data, const size_t index) {
    return data[index];
}
template <typename T>
CUDA_GLOBAL_METHOD inline void store_value(
    double* __restrict__ data, const size_t index, const T& value) {
    data[index] = value;
}
