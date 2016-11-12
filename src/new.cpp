/*
 * new.cpp
 *
 *  Created on: Oct 1, 2015
 *      Author: dmarce1
 */

#include <cstdlib>
#include <cstdio>

#ifdef __MIC__
constexpr std::size_t alignment = 64;
#else
constexpr std::size_t alignment = 32;
#endif



static void* allocate(std::size_t);
static void deallocate(void*);

void* operator new(std::size_t n) {
	return allocate(n);
}

void* operator new[](std::size_t n) {
	return allocate(n);
}

void operator delete(void* p) {
	deallocate(p);
}

void operator delete[](void* p) {
	deallocate(p);
}

#if defined(_MSC_VER)
#include <malloc.h>     // _aligned_alloc
#endif

static void* allocate(std::size_t n) {
#if defined(_MSC_VER)
    void* ptr = _aligned_malloc(n, alignment);
    if (ptr == 0) {
        printf("std::_aligned_alloc failed!\n");
        abort();
    }
    return ptr;
#else
	void* ptr;
	if (posix_memalign(&ptr, alignment, n) != 0) {
		printf("posix_memalign failed!\n");
		abort();
	}
	return ptr;
#endif
}

static void deallocate(void* ptr) {
	free(ptr);
}
