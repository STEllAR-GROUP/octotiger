
//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#pragma once

#include <array>
#include <vector>
#include "octotiger/print.hpp"

#define CHECK_BOUNDS

#define DO_BOUNDS_CHECK(i) \
		if( i >= this->size() ) { \
			print( "Bounds error - %i is not between 0 and %i\n",  i, this->size()); \
			abort(); \
		}

namespace oct {

#ifdef __CUDACC__
#define CUDA_EXPORT __host__ __device__
#else
#define CUDA_EXPORT
#endif

template<class T, class Alloc = std::allocator<T>>
class vector: public std::vector<T, Alloc> {
	using base_type = std::vector<T, Alloc>;
	using base_type::base_type;
public:
#ifdef CHECK_BOUNDS
	CUDA_EXPORT inline T& operator[](size_t i) {
		DO_BOUNDS_CHECK(i);
		return base_type::operator[](i);
	}
	CUDA_EXPORT inline const T& operator[](size_t i) const {
		DO_BOUNDS_CHECK(i);
		return base_type::operator[](i);
	}
#endif
};

template<class Alloc>
class vector<bool,Alloc>: public std::vector<bool, Alloc> {
	using base_type = std::vector<bool, Alloc>;
	using base_type::base_type;
public:
#ifdef CHECK_BOUNDS
	struct reference {
		std::vector<bool, Alloc>* ptr;
		int index;
		operator bool() const {
			return (*ptr)[index];
		}
		reference& operator=(bool b) const {
			(*ptr)[index] = b;
			return *this;
		}
		const reference& operator=(bool b)  {
			(*ptr)[index] = b;
			return *this;
		}
	};
	CUDA_EXPORT inline reference operator[](size_t i) {
		DO_BOUNDS_CHECK(i);
		reference ref;
		ref.ptr = (std::vector<bool, Alloc>*) this;
		ref.index = i;
		return ref;
	}
	CUDA_EXPORT inline bool operator[](size_t i) const {
		DO_BOUNDS_CHECK(i);
		return base_type::operator[](i);
	}
#endif
};

template<class T, int N>
class array: public std::array<T, N> {
	using base_type = std::array<T, N>;
	using base_type::base_type;
	using type = T;
public:
	operator std::array<T, N>() {
		return  (*((std::array<T,N>*) this));
	}
	operator std::array<T, N>&() {
		return  (*((std::array<T,N>*) this));
	}
#ifdef CHECK_BOUNDS
	CUDA_EXPORT inline T& operator[](size_t i) {
		DO_BOUNDS_CHECK(i);
		return base_type::operator[](i);
	}
	CUDA_EXPORT inline const T operator[](size_t i) const {
		DO_BOUNDS_CHECK(i);
		return base_type::operator[](i);
	}
#endif
};

}
