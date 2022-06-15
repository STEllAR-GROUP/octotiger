//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "octotiger/defs.hpp"
#include "octotiger/real.hpp"

#include <hpx/include/run_as.hpp>
#include <hpx/include/threads.hpp>

#include <cmath>
#include <cstdio>
#include <functional>

//#include <sse_mathfun.h>
#include <cstdio>

using real = double;


real LambertW(real z) {
	real W;
	if (z >= 0.0) {
		W = z < 1.0 ? z : 1.0 + std::log(z);
		for (int i = 0; i != 7; ++i) {
			const real eW = std::exp(W);
			const real WeW = W * eW;
			const real WeWmz = WeW - z;
			W -= WeWmz / (eW + WeW - 0.5 * ((W + 2.0) * WeWmz) / (W + 1.0));
		}
	} else {
		printf("LambertW not uniquely defined for z <= 0.0\n");
		abort();
	}
	return W;
}

int file_copy(const char* fin, const char* fout) {
    // run output on separate thread
    auto f = hpx::threads::run_as_os_thread([&]()
    {
	    constexpr size_t chunk_size = BUFSIZ;
	    char buffer[chunk_size];
	    FILE* fp_in = fopen(fin, "rb");
	    FILE* fp_out = fopen(fout, "wb");
	    if (fp_in == nullptr) {
		    return 1;
	    }
	    if (fp_out == nullptr) {
		    return 2;
	    }
	    size_t bytes_read;
	    while ((bytes_read = fread(buffer, sizeof(char), chunk_size, fp_in)) != 0) {
		    fwrite(buffer, sizeof(char), bytes_read, fp_out);
	    }
	    fclose(fp_in);
	    fclose(fp_out);
        return 0;
    });
    return f.get();
}

bool find_root(std::function<double(double)>& func, double xmin, double xmax,
		double& root, double toler) {
	double xmid;
	const int max_iter = 200;
	int iter = 0;
	const auto error = [](const double _xmax, const double _xmin) {
		return (_xmax - _xmin) / (std::abs(_xmax) + std::abs(_xmin))*2.0;
	};
	double xmin0 = xmin;
	double xmax0 = xmax;
	while (error(xmax,xmin) > toler) {
		printf("iter num: %d\n", iter); 
		if (iter >= max_iter) {
			printf("exceeded max iterations in find_root!\n");
			return false;
		}
		xmid = (xmax + xmin) / 2.0;
		if (func(xmid) * func(xmax) < 0.0) {
			xmin = xmid;
		} else {
			xmax = xmid;
		}
		iter++;
	}
	root = xmid;
	if( error(root,xmin0) < 10.0*toler || error(xmax0,root) < 10.0*toler ) {
		return false;
	}
	return true;
}

