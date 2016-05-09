/*
 * util.hpp
 *
 *  Created on: Apr 20, 2016
 *      Author: dmarce1
 */

#ifndef UTIL_HPP_
#define UTIL_HPP_

#include <stdio.h>

int file_copy(const char* fin, const char* fout);

template<class... Args>
int lprintf( const char* log, const char* str, Args&&...args) {
	FILE* fp = fopen (log, "at");
	if( fp == NULL) {
		return -1;
	}
	fprintf( fp, str, std::forward<Args>(args)...);
	printf( str, std::forward<Args>(args)...);
	fclose(fp);
	return 0;
}


#endif /* UTIL_HPP_ */
