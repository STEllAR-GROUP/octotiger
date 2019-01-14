#pragma once
// #include "taylor.hpp"

#include <Vc/Vc>

#include <cstdint>

#ifdef __AVX__
//using m2m_vector = typename Vc::datapar<double, Vc::datapar_abi::avx512>;
// for 8-wide (and not 16-wide) integers
//using m2m_int_vector = typename Vc::datapar<int32_t, Vc::datapar_abi::avx>;
// using m2m_int_vector = Vc::datapar<int64_t, Vc::datapar_abi::avx512>;
//#elif defined(Vc_HAVE_AVX)
#endif
#ifdef __AVX2__  // assumes AVX 2
using m2m_vector = Vc::Vector<double, Vc::VectorAbi::Avx>;
using m2m_int_vector = Vc::Vector<int32_t, Vc::VectorAbi::Avx>;
// using m2m_int_vector = typename Vc::datapar<int64_t, Vc::datapar_abi::avx>;
#else                         // falling back to fixed_size types
using m2m_vector = Vc::Vector<double>;
using m2m_int_vector = Vc::Vector<std::int32_t>;
#endif

// using multipole_v = taylor<4, m2m_vector>;
// using expansion_v = taylor<4, m2m_vector>;
