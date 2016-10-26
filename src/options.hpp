/*
 * options.hpp
 *
 *  Created on: Nov 13, 2015
 *      Author: dmarce1
 */

#ifndef OPTIONS_HPP_
#define OPTIONS_HPP_

#include <string.h>
#include "defs.hpp"

enum problem_type {
	DWD, SOD, BLAST, NONE, SOLID_SPHERE, STAR
};

class options {

	std::string exe_name;

	bool cmp(const char* str1, const char* str2);
	bool cmp(const std::string str1, const char* str2);
	void show_help();
public:
	integer max_level;
	real xscale;
	real omega;
	problem_type problem;
	std::string restart_filename;
	bool found_restart_file;
	std::string output_filename;
	bool output_only;
	real output_dt;
	real stop_time;
	real contact_fill;
	bool bench;
	real theta;
	bool ang_con;

	template<class Arc>
	void serialize(Arc& arc, unsigned) {
		arc & ang_con;
		arc & stop_time;
		arc & max_level;
		arc & xscale;
		arc & omega;
		arc & restart_filename;
		arc & found_restart_file;
		arc & output_filename;
		arc & output_only;
		arc & output_dt;
		int tmp = problem;
		arc & tmp;
		arc & theta;
		problem = (problem_type)tmp;

	}

/*
	std::size_t loadsave(FILE* fp, std::size_t (*foo)(void*, std::size_t, std::size_t, FILE*)) {
		std::size_t cnt = 0;
		cnt += sizeof(real) * fread(&xscale, sizeof(real), 1, fp);
		cnt += sizeof(problem_type) * fread(&problem, sizeof(problem_type), 1, fp);
		cnt += sizeof(integer) * fread(&max_level, sizeof(integer), 1, fp);
		return cnt;
	}

	std::size_t load(FILE* fp) {
		return loadsave(fp, fread);
	}

	std::size_t save(FILE* fp) {
		return loadsave(fp, [](void* ptr, std::size_t a, std::size_t b, FILE* c ) {
			return fwrite(ptr,a,b,c);
		});
	}*/

	bool process_options(int argc, char* argv[]);
};

#endif /* OPTIONS_HPP_ */
