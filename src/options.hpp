/*
 * options.hpp
 *
 *  Created on: Nov 13, 2015
 *      Author: dmarce1
 */

#ifndef OPTIONS_HPP_
#define OPTIONS_HPP_

#include <string>
#include "defs.hpp"

enum problem_type {
	DWD, SOD, BLAST, NONE, SOLID_SPHERE, STAR, MOVING_STAR
#ifdef RADIATION
	, RADIATION_TEST
#endif
};

enum eos_type {
	IDEAL, WD
};

class options {

	std::string exe_name;

	bool cmp(const char* str1, const char* str2);
	bool cmp(const std::string str1, const char* str2);
	void show_help();
public:
	eos_type eos;
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
    integer stop_step;
	real contact_fill;
	bool bench;
	real theta;
	bool ang_con;
    bool disable_output;

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
        arc & stop_step;
        arc & disable_output;
		arc & theta;
		int tmp = problem;
		arc & tmp;
		problem = (problem_type)tmp;
		tmp = eos;
		arc & tmp;
		eos = (eos_type)tmp;

	}

	bool process_options(int argc, char* argv[]);
};

#endif /* OPTIONS_HPP_ */
