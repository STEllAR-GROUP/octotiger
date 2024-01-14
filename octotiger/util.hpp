//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef UTILAA_HPP_
#define UTILAA_HPP_

#include "octotiger/options.hpp"

#include <hpx/include/threads.hpp>

#include <cstdio>
#include <functional>



/*OCTOTIGER_FORCEINLINE real minmod(real a, real b) {
//	return (std::copysign(HALF, a) + std::copysign(HALF, b)) * std::min(std::abs(a), std::abs(b));
	bool a_is_neg = a < 0;
	bool b_is_neg = b < 0;
	if (a_is_neg != b_is_neg)
		return ZERO;

	real val = std::min(std::abs(a), std::abs(b));
	return a_is_neg ? -val : val;
}*/

OCTOTIGER_FORCEINLINE real minmod_theta(real a, real b, real c, real theta) {
	return minmod(theta * minmod(a, b), c);
}


real LambertW(real z);

inline integer refinement_freq() {
	return  integer(2.0 / opts().cfl + 0.5);
}
int file_copy(const char* fin, const char* fout);


template<class... Args>
int lprint( const char* log, const char* str, Args&&...args) {
    // run output on separate thread
    auto f = hpx::run_as_os_thread([&]() -> int
    {
        if(!opts().disable_output) {
            FILE* fp = fopen (log, "at");
            if (fp == nullptr)
            {
                return -1;
            }
            fprintf( fp, str, std::forward<Args>(args)...);
	        fclose(fp);
	    }
        printf( str, std::forward<Args>(args)...);
	    return 0;
    });
    return f.get();
}


bool find_root(std::function<real(real)>& func, real xmin, real xmax,
		real& root, real toler = 1.0e-10);

inline real  assert_positive(real r, const char* filename, int line) {
	if( r <= 0.0 ) {
		FILE* fp = fopen("assert.log", "at");
		printf( "ASSERT_POSITIVE FAILED\n");
		printf( "file %s line %i\n", filename, line);
		fprintf( fp, "ASSERT_POSITIVE FAILED\n");
		fprintf( fp, "file %s line %i\n", filename, line);
		fclose(fp);
		abort();
	}
	return r;
}

inline void  assert_nonan(real r, const char* filename, int line) {
	if( std::isnan(r) ) {
		FILE* fp = fopen("assert.log", "at");
		printf( "ASSERT_NONAN FAILED\n");
		printf( "file %s line %i\n", filename, line);
		fprintf( fp, "ASSERT_NONAN FAILED\n");
		fprintf( fp, "file %s line %i\n", filename, line);
		fclose(fp);
		abort();
	}
}

#define ASSERT_POSITIVE(r) assert_positive((r), __FILE__, __LINE__)
#define ASSERT_NONAN(r) assert_nonan((r), __FILE__, __LINE__)


#endif /* UTIL_HPP_ */
