//  Copyright (c) 2024 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "octotiger/verbose.hpp"
#include "octotiger/options.hpp"

progress::progress(std::string s) {
	on = opts().verbose;
	if (on) {
		str = s;
		this_start = std::chrono::high_resolution_clock::now();
		printf("t=%li BEGIN: %s\n", (std::uint64_t) (time(NULL) - tstart), str.c_str());
		fflush(stdout);
	}
}

progress::progress(bool o, std::string s) {
	on = o && opts().verbose;
	if (on) {
		str = s;
		this_start = std::chrono::high_resolution_clock::now();
		printf("t=%li BEGIN: %s\n", (std::uint64_t) (time(NULL) - tstart), str.c_str());
		fflush(stdout);
	}
}

progress::~progress() {
	if (on) {
		const auto tm = std::chrono::high_resolution_clock::now() - this_start;
		const double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(tm).count() / 1000000.0;
		printf("t=%li END  : %s (%e s elapsed)\n", (std::uint64_t) (time(NULL) - tstart), str.c_str(), elapsed);
		fflush(stdout);
	}
}

std::uint64_t const progress::tstart = time(NULL);
