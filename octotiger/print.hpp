#pragma once

#include <utility>
#include <cstdio>


template<class ...Args>
inline int print(const char* format, Args&& ... args) {
	const int rc = printf( format, std::forward<Args>(args)...);
	fflush(stdout);
	return rc;
}

template<class ...Args>
inline int error(const char* format, Args&& ... args) {
	const int rc = fprintf( stderr, format, std::forward<Args>(args)...);
	fflush(stderr);
	return rc;
}
