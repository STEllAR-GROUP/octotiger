//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config/defines.hpp>

#include <cstdio>
#include <cstdlib>

// Note: This was commented out, because it doesn't actually resolve the alignement issue
// Consider the struct interaction_type (interaction_types.hpp):
// Its third member has to be aligned. But the posix_memalign approach would align the whole
// struct and thereby misalign the vector members.

// #if defined(__MIC__) || defined(__AVX512CD__)
// constexpr std::size_t alignment = 64;
// #else
// constexpr std::size_t alignment = 32;
// #endif

// #if !defined(_MSC_VER)
// #pragma message( "Compiling with custom new/delete" )
// static void* allocate(std::size_t);
// static void deallocate(void*);

// void* operator new(std::size_t n) {
// 	return allocate(n);
// }

// void* operator new[](std::size_t n) {
// 	return allocate(n);
// }

// void operator delete(void* p) {
// 	deallocate(p);
// }

// void operator delete[](void* p) {
// 	deallocate(p);
// }

// static void* allocate(std::size_t n) {
// 	void* ptr;
// 	if ((n >= alignment) && (n % alignment == 0)) {
// 		if (posix_memalign(&ptr, alignment, n) != 0) {
// 			print("posix_memalign failed!\n");
// 			abort();
// 		}
// 	} else {
// 		ptr = (void*) malloc(n);
// 		if (ptr == nullptr) {
// 			print("malloc failed!\n");
// 			abort();
// 		}

// }
// 	return ptr;
// }

// static void deallocate(void* ptr) {
// 	free(ptr);
// }
// #else
// #pragma message( "Compiling without custom new/delete" )
// #endif
