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
	DWD, SOD, RESTART
};


class options {
	problem_type problem;
	std::string restart_filename;
	bool found_restart_file;
	std::string exe_name;
	integer max_level;
	bool cmp(const char* str1, const char* str2);
	bool cmp(const std::string str1, const char* str2);
	void show_help();
public:
	bool process_options(int argc, char* argv[]);
};



#endif /* OPTIONS_HPP_ */
