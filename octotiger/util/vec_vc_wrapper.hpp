#ifdef OCTOTIGER_HAVE_VC
#include "vec_base_wrapper.hpp"

#include <Vc/Vc>
#include <Vc/common/mask.h>
#include <Vc/vector.h>
#include <Vc/SimdArray>

#ifdef __x86_64__
#include <x86intrin.h>
#endif

#ifdef __ppc64__
#include <spe.h>
#endif

//using vc_type = Vc::Vector<double, Vc::VectorAbi::Avx>;
//using mask_type = vc_type::mask_type;
//using index_type = Vc::Vector<int, Vc::VectorAbi::Avx>;
using vc_type = Vc::SimdArray<double, 4>;
using mask_type = vc_type::mask_type;
using index_type = Vc::SimdArray<int, 4>;

template <>
CUDA_GLOBAL_METHOD inline void select_wrapper<vc_type, mask_type>(
    vc_type& target, const mask_type cond, const vc_type& tmp1, const vc_type& tmp2) {
    target = tmp2;
    Vc::where(cond, target) = tmp1;
}

template <>
CUDA_GLOBAL_METHOD inline vc_type max_wrapper<vc_type>(const vc_type& tmp1, const vc_type& tmp2) {
    return Vc::max(tmp1, tmp2);
}
template <>
CUDA_GLOBAL_METHOD inline vc_type min_wrapper<vc_type>(const vc_type& tmp1, const vc_type& tmp2) {
    return Vc::min(tmp1, tmp2);
}
template <>
CUDA_GLOBAL_METHOD inline vc_type sqrt_wrapper<vc_type>(const vc_type& tmp1) {
    return Vc::sqrt(tmp1);
}
template <>
CUDA_GLOBAL_METHOD inline vc_type asinh_wrapper<vc_type>(const vc_type& tmp1) {
    // return Vc::asinh(tmp1);
    vc_type ret = 0.0;
    for (auto vec_i = 0; vec_i < vc_type::size(); vec_i++) {
      ret[vec_i] = std::asinh(tmp1[vec_i]);
    }
    return ret;
}
template <>
CUDA_GLOBAL_METHOD inline vc_type copysign_wrapper<vc_type>(const vc_type& tmp1, const vc_type& tmp2) {
    return Vc::copysign(tmp1, tmp2);
}
template <>
CUDA_GLOBAL_METHOD inline vc_type abs_wrapper<vc_type>(const vc_type& tmp1) {
    return Vc::abs(tmp1);
}
template <>
CUDA_GLOBAL_METHOD inline vc_type minmod_wrapper<vc_type>(const vc_type& a, const vc_type& b) {
    return (copysign_wrapper<vc_type>(0.5, a) + copysign_wrapper<vc_type>(0.5, b)) *
        min_wrapper<vc_type>(abs_wrapper<vc_type>(a), abs_wrapper<vc_type>(b));
}
template <>
CUDA_GLOBAL_METHOD inline vc_type minmod_theta_wrapper<vc_type>(const vc_type& a, const vc_type& b, const vc_type& c) {
    return minmod_wrapper<vc_type>(c * minmod_wrapper<vc_type>(a, b), 0.5 * (a + b));
}
template <>
CUDA_GLOBAL_METHOD inline vc_type limiter_wrapper<vc_type>(const vc_type& a, const vc_type& b) {
    return minmod_theta_wrapper<vc_type>(a, b, 64. / 37.);
}
template <>
CUDA_GLOBAL_METHOD inline vc_type load_value<vc_type>(const double* __restrict__ data, const size_t index) {
    return vc_type(data + index);
}
template <>
CUDA_GLOBAL_METHOD inline void store_value<vc_type>(
    double* __restrict__ data, const size_t index, const vc_type& value) {
  value.store(data + index);
}
template <>
CUDA_GLOBAL_METHOD inline bool skippable<mask_type>(const mask_type& tmp1) {
    return Vc::none_of(tmp1);
}
/// Awful workaround for missing Vc::pow
template <>
CUDA_GLOBAL_METHOD inline vc_type pow_wrapper<vc_type>(const vc_type& tmp1, const double& tmp2) {
    // TODO(daissgr) is this accurate enough?
    return Vc::exp(static_cast<vc_type>(tmp2) * Vc::log(tmp1));
    // vc_type ret = 0.0;
    // for (auto vec_i = 0; vec_i < vc_type::size(); vec_i++) {
    // ret[vec_i] = std::pow(tmp1[vec_i], tmp2);
    //}
    // return ret;
}

#endif
