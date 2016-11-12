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


#if !defined(_MSC_VER)

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

static void* allocate(std::size_t n) {
	void* ptr;
	if (posix_memalign(&ptr, alignment, n) != 0) {
		printf("posix_memalign failed!\n");
		abort();
	}
	return ptr;
}

static void deallocate(void* ptr) {
	free(ptr);
}

#endif
