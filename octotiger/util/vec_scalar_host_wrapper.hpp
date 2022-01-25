#pragma once

#include "vec_base_wrapper.hpp"

template <>
CUDA_GLOBAL_METHOD inline void select_wrapper<double, bool>(
    double_t& target, const bool cond, const double& tmp1, const double& tmp2) {
    target = cond ? tmp1 : tmp2;
}
template <>
CUDA_GLOBAL_METHOD inline double max_wrapper<double>(const double& tmp1, const double& tmp2) {
    return std::max(tmp1, tmp2);
}
template <>
CUDA_GLOBAL_METHOD inline double min_wrapper<double>(const double& tmp1, const double& tmp2) {
    return std::min(tmp1, tmp2);
}
template <>
CUDA_GLOBAL_METHOD inline double sqrt_wrapper<double>(const double& tmp1) {
    return std::sqrt(tmp1);
}
template <>
CUDA_GLOBAL_METHOD inline double pow_wrapper<double>(const double& tmp1, const double& tmp2) {
    return std::pow(tmp1, tmp2);
}
template <>
CUDA_GLOBAL_METHOD inline double copysign_wrapper<double>(const double& tmp1, const double& tmp2) {
    return std::copysign(tmp1, tmp2);
}
template <>
CUDA_GLOBAL_METHOD inline double abs_wrapper<double>(const double& tmp1) {
    return std::abs(tmp1);
}
template <>
CUDA_GLOBAL_METHOD inline double minmod_wrapper<double>(const double& a, const double& b) {
    return (copysign_wrapper<double>(0.5, a) + copysign_wrapper<double>(0.5, b)) *
        min_wrapper<double>(abs_wrapper<double>(a), abs_wrapper<double>(b));
}
template <>
CUDA_GLOBAL_METHOD inline double minmod_theta_wrapper<double>(const double& a, const double& b, const double& c) {
    return minmod_wrapper<double>(c * minmod_wrapper<double>(a, b), 0.5 * (a + b));
}
template <>
CUDA_GLOBAL_METHOD inline double limiter_wrapper<double>(const double& a, const double& b) {
    return minmod_theta_wrapper<double>(a, b, 64. / 37.);
}
template <>
CUDA_GLOBAL_METHOD inline double asinh_wrapper<double>(const double& tmp1) {
    return std::asinh(tmp1);
}
template <>
CUDA_GLOBAL_METHOD inline bool skippable<double>(const double& tmp1) {
    return !tmp1;
}
template <>
CUDA_GLOBAL_METHOD inline double load_value<double>(const double* __restrict__ data, const size_t index) {
    return data[index];
}
template <>
CUDA_GLOBAL_METHOD inline void store_value<double>(
    double* __restrict__ data, const size_t index, const double& value) {
    data[index] = value;
}

