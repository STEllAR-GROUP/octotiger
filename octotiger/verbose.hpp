//  Copyright (c) 2024 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <string>
#include <cstdint>
#include <ctime>
#include <chrono>


class progress {
	static const std::uint64_t tstart;
	std::chrono::time_point<std::chrono::high_resolution_clock> this_start;
	std::string str;
	bool on;
public:
	progress(std::string s);
	progress(bool, std::string s);
	~progress();
};


#define CPROGRESS(on, ostr) \
		char* str; \
      asprintf(&str, "%s (file: %s, line: %i, function: %s)", ostr, __FILE__, __LINE__, __FUNCTION__); \
      progress p_r_o_g_r_e_s_s(on, str); \
      free(str);

#define PROGRESS(ostr) CPROGRESS(true, ostr)
