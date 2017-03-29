/*
 * options.cpp
 *
 *  Created on: Nov 13, 2015
 *      Author: dmarce1
 */

#include "defs.hpp"
#include "options.hpp"
#include <math.h>
#include "grid.hpp"

#define HELP_OPT "-Help"
#define PROBLEM_OPT "-Problem"
#define RESTART_OPT "-Restart"
#define OUTPUT_OPT "-Output"
#define NGRIDS_OPT "-Ngrids"
#define REFINEMENT_FLOOR_OPT "-RefinementFloor"
#define DATA_DIR_OPT "-Datadir"
#define ANGCON_OPT "-Angcon"
#define XSCALE_OPT "-Xscale"
#define OMEGA_OPT "-Omega"
#define VOMEGA_OPT "-VariableOmega"
#define ODT_OPT "-Odt"
#define DISABLEOUTPUT_OPT "-Disableoutput"
#define STOPTIME_OPT "-Stoptime"
#define STOPSTEP_OPT "-Stopstep"
#define BENCH_OPT "-Bench"
#define THETA_OPT "-Theta"
#define EOS_OPT "-Eos"
#define PARALLEL_SILO_OPT "-ParallelSilo"
#define SILO_PLANES_ONLY_OPT "-SiloPlanesOnly"

#define MAX_LEVEL_OPT "-Max_level"
#define MAX_RESTART_LEVEL_OPT "-Regrid_level"

#define PROBLEM_OPT_DWD "dwd"
#define PROBLEM_OPT_SOD "sod"
#define PROBLEM_OPT_BLAST "blast"
#define PROBLEM_OPT_OLD_SCF "old_scf"
#define PROBLEM_OPT_SOLID_SPHERE "solid_sphere"
#define PROBLEM_OPT_STAR "star"
#define PROBLEM_OPT_MOVING_STAR "moving_star"
#define PROBLEM_OPT_RADIATION_TEST "radiation_test"

bool options::cmp(const char* str1, const char* str2) {
    return strncmp(str1, str2, strlen(str2)) == 0;
}

bool options::cmp(const std::string str1, const char* str2) {
    return strncmp(str1.c_str(), str2, strlen(str2)) == 0;
}

void options::show_help() {
    printf( "Command line options for Octo-tiger\n\n"
            "-AngCon=<1/0>                         - 1 (default) turns on the angular momentum conservation feature, 0 turns it off.\n"
            "\n"
    		"-Bench                                - Runs for a few time-steps then exit.\n"
            "\n"
            "-Disableoutput                        - Disables SILO and checkpoint output.\n"
            "\n"
            "-Eos=<equation of state>              - Sets the equation of state for pressure and energy\n"
            "                                        ideal        - ideal gas equation of state\n"
            "                                        wd           - white dwarf equation of state\n"
            "\n"
            "-Help                                 - Displays this help page and exits\n"
            "\n"
            "-Max_level=<number of refined levels> - Set maximum level of refinment.\n"
            "-Regrid_level=<number of refined levels> - Number of regrids ater loading restart file.\n"
            "\n"
            "-Odt=<output frequency>               - Specifies the frequency for SILO output in units of simulation time.\n"
            "\n"
            "-Omega=<angular frequency>            - The grid rotates along z axis at x=0 and y=0 with this angular frequency (default 0.0). \n"
            "                                        (Note that for the dwd problem this specifies only the initial value)\n"
            "\n"
            "-Output=<file name>                   - Load restart file, output to SILO, then exit. Used for converting checkpoints to SILO.\n"
            "\n"
            "-Datadir=<directory name>             - Directory where to load input from/write output to (must exist!).\n"
            "\n"
            "-Problem=<problem name>               - Sets up the initial model\n"
            "                                        blast        - Set up and run a Sedov-Taylor blast wave.\n"
            "                                        dwd          - use SCF solver to find an initial model for a double white dwarf (see grid_scf.cpp for options). \n"
            "                                                       Terminates after creating restart file.\n"
            "                                        moving_star  - Set up a single equilibrium star moving across grid at constant velocity, then run.\n"
            "                                        sod          - Set up and run a Sod shock tube.\n"
            "                                        solid_sphere - Set up a constant density solid sphere, solve for gravity, then exit.\n"
            "                                        star         - Set up a single equilibrium star at zero velocity, then run.\n"
            "\n"
            "-Restart=<file name>                  - Restart from a checkpoint file. If this option is not selected, Octo-tiger sets up the\n"
            "                                        initial problem based on the -Problem option\n"
            "\n"
            "-Stopstep=<stop step>                 - The simulation runs until stop step number of steps have executed. The firs step from a restart file\n"
            "                                        is counted as step one regardless of the number of steps prior to the restart (default is infinity).\n"
            "                                        If the simulation time reaches the time specified by Stoptime first, Octotiger exits then.\n"
            "\n"
            "-Stoptime=<stop time>                 - The simulation runs until the simulation time hits the time specified here, or until the maximum\n"
            "                                        number of steps specified by Stopstep. (default is infinity)\n"
            "\n"
            "-VariableOmega=1/0                    - 1 turns on variable omega, 0 off. 0 by default, unless problem=dwd, then 1 by default\n"
            "                                        mean faster FMM execution but a higher solution error.\n"
            "-Theta                                - 'Opening criterion' for FMM (default 0.35). Must be between 1/3 and 1/2, inclusive. Larger values\n"
            "                                        mean faster FMM execution but a higher solution error.\n"
            "-Xscale=<xmax>                        - The domain of the coarsest grid is set to (-xmax,xmax) for each all three dimensions (default 1.0)\n"
            "\n"
            "");
}

bool options::process_options(int argc, char* argv[]) {
    bool rc;
    eos = IDEAL;
    rc = true;
	theta = 0.35;
	parallel_silo = false;
	silo_planes_only = false;
	ngrids = -1;
	max_level = 5;
	refinement_floor = -1.0;
	refinement_floor_specified = false;
	max_restart_level = 0;
    problem = NONE;
    found_restart_file = false;
    output_only = false;
    xscale = 2.0;
    omega = 0.0;
    exe_name = std::string(argv[0]);
    contact_fill = 0.0;
    output_dt = -1;
    bench = false;
    ang_con = true;
    stop_time = std::numeric_limits<real>::max() - 1;
    stop_step = std::numeric_limits<integer>::max() / 10;
    disable_output = false;
    bool vomega_found = false;

    for (integer i = 1; i < argc; ++i) {
        if (cmp(argv[i], HELP_OPT)) {
            rc = false;
        } else if (cmp(argv[i], PROBLEM_OPT)) {
            std::string prob(argv[i] + strlen(PROBLEM_OPT) + 1);
            if (cmp(prob, PROBLEM_OPT_DWD)) {
                problem = DWD;
                //    } else if (cmp(prob, PROBLEM_OPT_OLD_SCF)) {
                //        problem = OLD_SCF;
            } else if (cmp(prob, PROBLEM_OPT_SOLID_SPHERE)) {
                problem = SOLID_SPHERE;
            } else if (cmp(prob, PROBLEM_OPT_STAR)) {
                problem = STAR;
            } else if (cmp(prob, PROBLEM_OPT_MOVING_STAR)) {
                problem = MOVING_STAR;
#ifdef RADIATION
            } else if (cmp(prob, PROBLEM_OPT_RADIATION_TEST)) {
                problem = RADIATION_TEST;
#endif
            } else if (cmp(prob, PROBLEM_OPT_SOD)) {
                problem = SOD;
            } else if (cmp(prob, PROBLEM_OPT_BLAST)) {
                problem = BLAST;
            } else {
                printf("The user specified an invalid problem type, \"%s\"\n", prob.c_str());
                rc = false;
            }
        } else if (cmp(argv[i], BENCH_OPT)) {
            bench = true;
        } else if (cmp(argv[i], EOS_OPT)) {
            const char* str = argv[i] + strlen(EOS_OPT) + 1;
            if (strncmp(str, "ideal", 3) == 0) {
                eos = IDEAL;
            } else if (strncmp(str, "wd", 2) == 0) {
                eos = WD;
            } else {
                printf("Unknown EOS specified - choose ideal or wd.\n");
                abort();
            }
        } else if (cmp(argv[i], THETA_OPT)) {
            theta = atof(argv[i] + strlen(THETA_OPT) + 1);
        } else if (cmp(argv[i], RESTART_OPT)) {
			restart_filename = std::string(argv[i] + strlen(RESTART_OPT) + 1);
			found_restart_file = true;
		} else if (cmp(argv[i], DATA_DIR_OPT)) {
			data_dir = std::string(argv[i] + strlen(DATA_DIR_OPT) + 1);
			data_dir += "/";
		} else if (cmp(argv[i], PARALLEL_SILO_OPT)) {
			parallel_silo = true;
		} else if (cmp(argv[i], SILO_PLANES_ONLY_OPT)) {
			silo_planes_only = true;
		} else if (cmp(argv[i], OUTPUT_OPT)) {
			output_filename = std::string(argv[i] + strlen(OUTPUT_OPT) + 1);
			output_only = true;
		} else if (cmp(argv[i], REFINEMENT_FLOOR_OPT)) {
			refinement_floor = atof(argv[i] + strlen(REFINEMENT_FLOOR_OPT) + 1);
			refinement_floor_specified = true;
		} else if (cmp(argv[i], ANGCON_OPT)) {
			ang_con = atoi(argv[i] + strlen(ANGCON_OPT) + 1) != 0;
		} else if (cmp(argv[i], VOMEGA_OPT)) {
			vomega_found = true;
			vomega = atoi(argv[i] + strlen(VOMEGA_OPT) + 1) != 0;
		} else if (cmp(argv[i], MAX_LEVEL_OPT)) {
			max_level = atoi(argv[i] + strlen(MAX_LEVEL_OPT) + 1);
		} else if (cmp(argv[i], NGRIDS_OPT)) {
			ngrids = atoi(argv[i] + strlen(NGRIDS_OPT) + 1);
        } else if (cmp(argv[i], MAX_RESTART_LEVEL_OPT)) {
            max_restart_level = atoi(argv[i] + strlen(MAX_RESTART_LEVEL_OPT) + 1);
        } else if (cmp(argv[i], XSCALE_OPT)) {
            xscale = atof(argv[i] + strlen(XSCALE_OPT) + 1);
        } else if (cmp(argv[i], OMEGA_OPT)) {
            omega = atof(argv[i] + strlen(OMEGA_OPT) + 1);
        } else if (cmp(argv[i], ODT_OPT)) {
            output_dt = atof(argv[i] + strlen(ODT_OPT) + 1);
        } else if (cmp(argv[i], DISABLEOUTPUT_OPT)) {
            disable_output = true;
        } else if (cmp(argv[i], STOPTIME_OPT)) {
            stop_time = atof(argv[i] + strlen(STOPTIME_OPT) + 1);
        } else if (cmp(argv[i], STOPSTEP_OPT)) {
            stop_step = atoi(argv[i] + strlen(STOPSTEP_OPT) + 1);
        } else {
            printf("Unknown option - %s\n", argv[i]);
            abort();
        }
    }
    if (output_dt < 0) {
        if (omega > 0.0) {
            output_dt = (2.0 * M_PI / omega) / 100.0;
        } else {
            output_dt = 1.0e-2;
        }
    }
    if (!rc) {
        show_help();
    }
    theta = std::max(1.0 / 3.0, theta);
    theta = std::min(1.0 / 2.0, theta);
    if( !vomega_found ) {
    	vomega = (problem == DWD);
    }
    printf( "Variable omega is %s\n", vomega ? "on" : "off");
    return rc;
}

std::vector<hpx::id_type> options::all_localities = {};
