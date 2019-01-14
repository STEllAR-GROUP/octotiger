/*
 * space_vector_gen.hpp
 *
 *  Created on: Jun 3, 2015
 *      Author: dmarce1
 */

#ifndef SPACE_VECTOR_HPP_
#define SPACE_VECTOR_HPP_

#include "octotiger/defs.hpp"

#include <hpx/parallel/traits/vector_pack_type.hpp>
#include <hpx/runtime/serialization/datapar.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>

#include <Vc/Vc>

#include <cmath>

// #if !defined(HPX_HAVE_DATAPAR)

// template<class T = real>
// class space_vector_gen : public std::array<T, NDIM> {
// public:
// 	template<class Archive>
// 	void serialize(Archive& arc, unsigned) {
// 		arc & *(static_cast<std::array<T, NDIM>*>(this));
// 	}
// 	space_vector_gen& operator=(T a) {
// 		for (integer i = 0; i != NDIM; ++i) {
// 			(*this)[i] = a;
// 		}
// 		return *this;
// 	}
// 	space_vector_gen() = default;
// 	space_vector_gen(T a) {
// 		for (integer i = 0; i != NDIM; ++i) {
// 			(*this)[i] = a;
// 		}
// 	}
// 	space_vector_gen& operator=(const space_vector_gen& other) {
// 		for (integer i = 0; i != NDIM; ++i) {
// 			(*this)[i] = other[i];
// 		}
// 		return *this;
// 	}
// 	space_vector_gen& operator+=(const space_vector_gen& other) {
// 		for (integer i = 0; i != NDIM; ++i) {
// 			(*this)[i] += other[i];
// 		}
// 		return *this;
// 	}
// 	space_vector_gen& operator-=(const space_vector_gen& other) {
// 		for (integer i = 0; i != NDIM; ++i) {
// 			(*this)[i] -= other[i];
// 		}
// 		return *this;
// 	}
// 	space_vector_gen& operator*=(T a) {
// 		for (integer i = 0; i != NDIM; ++i) {
// 			(*this)[i] *= a;
// 		}
// 		return *this;
// 	}
// 	space_vector_gen& operator/=(T a) {
// 		for (integer i = 0; i != NDIM; ++i) {
// 			(*this)[i] /= a;
// 		}
// 		return *this;
// 	}
// 	space_vector_gen operator+(const space_vector_gen& other) const {
// 		space_vector_gen me = *this;
// 		me += other;
// 		return me;
// 	}
// 	space_vector_gen operator-(const space_vector_gen& other) const {
// 		space_vector_gen me = *this;
// 		me -= other;
// 		return me;
// 	}
// 	space_vector_gen operator*(T a) const {
// 		auto me = *this;
// 		me *= a;
// 		return me;
// 	}
// 	space_vector_gen operator/(T a) const {
// 		space_vector_gen me = *this;
// 		me /= a;
// 		return me;
// 	}
// 	space_vector_gen operator+() const {
// 		space_vector_gen me = *this;
// 		return me;
// 	}
// 	space_vector_gen operator-() const {
// 		space_vector_gen me = *this;
// 		for (integer i = 0; i != NDIM; ++i) {
// 			me[i] = -me[i];
// 		}
// 		return me;
// 	}
// 	T abs() const {
// 		T sum = ZERO;
// 		for (integer d = 0; d != NDIM; ++d) {
// 			sum += (*this)[d] * (*this)[d];
// 		}
// 		return std::sqrt(sum);
// 	}
// };

// namespace hpx { namespace traits
// {
//     template <class T>
//     struct is_bitwise_serializable<space_vector_gen<T> >
//       : is_bitwise_serializable<typename std::remove_const<T>::type>
//     {};
// }}

// using space_vector = space_vector_gen<real>;

// #else

// #if defined(Vc_HAVE_AVX512F) || defined(Vc_HAVE_AVX)
using space_vector = Vc::Vector<real,Vc::VectorAbi::Avx>;
// #else
// using space_vector = typename hpx::parallel::traits::vector_pack_type<real, 4>::type;
// #endif

// #endif

#endif /* SPACE_VECTOR_HPP_ */
