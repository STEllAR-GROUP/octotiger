/*
 * options.cpp
 *
 *  Created on: Nov 13, 2015
 *      Author: dmarce1
 */

#include "options.hpp"
#include <math.h>

#define HELP_OPT "-Help"
#define PROBLEM_OPT "-Problem"
#define RESTART_OPT "-Restart"
#define OUTPUT_OPT "-Output"
#define XSCALE_OPT "-Xscale"
#define OMEGA_OPT "-Omega"
#define BENCH_OPT "-Bench"
#define THETA_OPT "-Theta"

#define MAX_LEVEL_OPT "-Max_level"

#define PROBLEM_OPT_DWD "dwd"
#define PROBLEM_OPT_SOD "sod"
#define PROBLEM_OPT_OLD_SCF "old_scf"
#define PROBLEM_OPT_SOLID_SPHERE "solid_sphere"
#define PROBLEM_OPT_STAR "star"

bool options::cmp(const char* str1, const char* str2) {
	return strncmp(str1, str2, strlen(str2)) == 0;
}

bool options::cmp(const std::string str1, const char* str2) {
	return strncmp(str1.c_str(), str2, strlen(str2)) == 0;
}

void options::show_help() {
	printf("Command line options for Octo-tiger\n");
	printf("-Problem=<problem name> - sets up the initial model\n");
	printf("\t\t\t\tDWD - use SCF solver.\n");
	printf("\t\t\t\tSod - Sod shock tube.\n");
	printf("-Restart=<file name> - restart from a checkpoint file\n");
	printf("-Output=<file name> - output restart to silo file and exit\n");
	printf("-Max_level=<number of refined levels> - set maximum number of refinement levels\n");
	printf("-Help - displays this help page\n");
}

bool options::process_options(int argc, char* argv[]) {
	bool rc;
	rc = true;
	theta = 0.35;
	max_level = 5;
	problem = NONE;
	found_restart_file = false;
	output_only = false;
	xscale = 2.0;
	omega = -1.0;
	exe_name = std::string(argv[0]);
	contact_fill = 0.0;
	output_dt = 1.0;
	bench = false;

	for (integer i = 1; i < argc; ++i) {
		if (cmp(argv[i], HELP_OPT)) {
			rc = false;
		} else if (cmp(argv[i], PROBLEM_OPT)) {
			std::string prob(argv[i] + strlen(PROBLEM_OPT) + 1);
			if (cmp(prob, PROBLEM_OPT_DWD)) {
				problem = DWD;
				//	} else if (cmp(prob, PROBLEM_OPT_OLD_SCF)) {
				//		problem = OLD_SCF;
			} else if (cmp(prob, PROBLEM_OPT_SOLID_SPHERE)) {
				problem = SOLID_SPHERE;
			} else if (cmp(prob, PROBLEM_OPT_STAR)) {
				problem = STAR;
			} else if (cmp(prob, PROBLEM_OPT_SOD)) {
				problem = SOD;
			} else {
				printf("The user specified an invalid problem type, \"%s\"\n", prob.c_str());
				rc = false;
			}
		} else if (cmp(argv[i], BENCH_OPT)) {
			bench = true;
		} else if (cmp(argv[i], THETA_OPT)) {
			theta = atof(argv[i] + strlen(THETA_OPT) + 1);
		} else if (cmp(argv[i], RESTART_OPT)) {
			restart_filename = std::string(argv[i] + strlen(RESTART_OPT) + 1);
			found_restart_file = true;
		} else if (cmp(argv[i], OUTPUT_OPT)) {
			output_filename = std::string(argv[i] + strlen(OUTPUT_OPT) + 1);
			output_only = true;
		} else if (cmp(argv[i], MAX_LEVEL_OPT)) {
			max_level = atoi(argv[i] + strlen(MAX_LEVEL_OPT) + 1);
		} else if (cmp(argv[i], XSCALE_OPT)) {
			xscale = atof(argv[i] + strlen(XSCALE_OPT) + 1);
		} else if (cmp(argv[i], OMEGA_OPT)) {
			omega = atof(argv[i] + strlen(OMEGA_OPT) + 1);
			output_dt = (2.0 * M_PI / omega) / 100.0;
		} else {
			printf("Unknown option - %s\n", argv[i]);
			abort();
		}
	}
	if (!rc) {
		show_help();
	}
	theta = std::max(1.0 / 3.0, theta);
	theta = std::min(1.0 / 2.0, theta);
	return rc;
}
