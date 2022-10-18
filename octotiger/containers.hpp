
//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#pragma once

#include <array>
#include <vector>

#define DO_BOUNDS_CHECK(i) \
		if( i >= this->size() ) { \
			print( "Bounds error - %i is not between 0 and %i\n",  i, this->size()); \
			abort(); \
		}

namespace oct {

template<class T, class Alloc = std::allocator<T>>
class vector: public std::vector<T, Alloc> {
	using base_type = std::vector<T, Alloc>;
	using base_type::base_type;
public:
#ifdef CHECK_BOUNDS
	inline T& operator[](size_t i) {
		DO_BOUNDS_CHECK(i);
		return (*((std::vector<T>*) this))[i];
	}
	inline const T& operator[](size_t i) const {
		DO_BOUNDS_CHECK(i);
		return (*((std::vector<T>*) this))[i];
	}
#endif
};

template<class T, int N>
class array: public std::array<T, N> {
	using base_type = std::array<T, N>;
	using base_type::base_type;
public:
#ifdef CHECK_BOUNDS
	inline T& operator[](size_t i) {
		DO_BOUNDS_CHECK(i);
		return (*((std::array<T,N>*) this))[i];
	}
	inline const T& operator[](size_t i) const {
		DO_BOUNDS_CHECK(i);
		return (*((std::array<T,N>*) this))[i];
	}
#endif
};

}
