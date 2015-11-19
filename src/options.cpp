/*
 * options.cpp
 *
 *  Created on: Nov 13, 2015
 *      Author: dmarce1
 */

#include "options.hpp"

#define HELP_OPT "-help"
#define PROBLEM_OPT "-problem"
#define RESTART_OPT "-restart"
#define MAX_LEVEL_OPT "-max_level"

#define PROBLEM_OPT_DWD "dwd"
#define PROBLEM_OPT_SOD "sod"


bool options::cmp(const char* str1, const char* str2) {
	return strncmp(str1, str2, strlen(str2)) == 0;
}

bool options::cmp(const std::string str1, const char* str2) {
	return strncmp(str1.c_str(), str2, strlen(str2)) == 0;
}


void options::show_help() {
	printf("Command line options for Octo-tiger\n");
	printf("-problem=<problem name> - sets up the initial model\n");
	printf("\t\t\t\tDWD - double white dwarf using the internal SCF solver.\n");
	printf("\t\t\t\tSod - Sod shock tube.\n");
	printf("-restart=<file name> - restart from a checkpoint file\n");
	printf("-max_level=<number of refined levels> - set maximum number of refinement levels\n");
	printf("-help - displays this help page\n");
	exit(1);
}

bool options::process_options(int argc, char* argv[])  {
	bool rc;
	rc = true;
	problem = RESTART;
	found_restart_file = false;

	exe_name = std::string(argv[0]);

	for (integer i = 1; i < argc; ++i) {
		integer cnt = strlen(argv[i]);
		for (integer j = 0; j != cnt; ++j) {
			argv[i][j] = tolower(argv[i][j]);
		}
	}

	for (integer i = 0; i < argc; ++i) {
		if (cmp(argv[i], HELP_OPT)) {
			rc = false;
		} else if (cmp(argv[i], PROBLEM_OPT)) {
			std::string prob(argv[i] + strlen(PROBLEM_OPT) + 1);
			if (cmp(prob, PROBLEM_OPT_DWD)) {
				problem = DWD;
			} else if (cmp(prob, PROBLEM_OPT_SOD)) {
				problem = SOD;
			} else {
				printf("The user specified an invalid problem type, \"%s\"\n", prob.c_str());
				rc = false;
			}
		} else if (cmp(argv[i], RESTART_OPT)) {
			restart_filename = std::string(argv[i] + strlen(RESTART_OPT) + 1);
			found_restart_file = true;
		} else if (cmp(argv[i], MAX_LEVEL_OPT)) {
			max_level = atoi(argv[i] + strlen(MAX_LEVEL_OPT) + 1);
		}
	}
	if (!rc) {
		show_help();
	}
	return rc;
}
