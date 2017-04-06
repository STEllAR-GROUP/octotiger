#include "defs.hpp"

#include "node_server.hpp"
#include "node_client.hpp"
#include "future.hpp"
#include "problem.hpp"
#include "options.hpp"

#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include <fenv.h>
#if !defined(_MSC_VER)
#include <unistd.h>
#else
#include <float.h>
#endif

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/lcos/broadcast.hpp>

options opts;

bool gravity_on = true;
bool hydro_on = true;
HPX_PLAIN_ACTION(grid::set_pivot, set_pivot_action);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(set_pivot_action)
HPX_REGISTER_BROADCAST_ACTION(set_pivot_action)

void compute_ilist();

void initialize(options _opts, std::vector<hpx::id_type> const& localities)
{
    options::all_localities = localities;
	opts = _opts;
    grid::get_omega() = opts.omega;
#if !defined(_MSC_VER)
	feenableexcept (FE_DIVBYZERO);
	feenableexcept (FE_INVALID);
	feenableexcept (FE_OVERFLOW);
#else
    _controlfp(_EM_INEXACT | _EM_DENORMAL | _EM_INVALID, _MCW_EM);
#endif
	grid::set_scaling_factor(opts.xscale);
	grid::set_max_level(opts.max_level);
#ifdef RADIATION
	if (opts.problem == RADIATION_TEST) {
		gravity_on = false;
		set_problem(radiation_test_problem);
		set_refine_test(radiation_test_refine);
	} else
#endif
	if (opts.problem == DWD) {
		set_problem(scf_binary);
		set_refine_test(refine_test);
	} else if (opts.problem == SOD) {
		grid::set_fgamma(7.0 / 5.0);
		gravity_on = false;
		set_problem(sod_shock_tube_init);
		set_refine_test (refine_sod);
		grid::set_analytic_func(sod_shock_tube_analytic);
	} else if (opts.problem == BLAST) {
		grid::set_fgamma(7.0 / 5.0);
		gravity_on = false;
		set_problem (blast_wave);
		set_refine_test (refine_blast);
	} else if (opts.problem == STAR) {
		grid::set_fgamma(5.0 / 3.0);
		set_problem(star);
		set_refine_test(refine_test_bibi);
	} else if (opts.problem == MOVING_STAR) {
		grid::set_fgamma(5.0 / 3.0);
		grid::set_analytic_func(moving_star_analytic);
		set_problem(moving_star);
		set_refine_test(refine_test_bibi);
		/*} else if (opts.problem == OLD_SCF) {
		 set_refine_test(refine_test_bibi);
		 set_problem(init_func_type([=](real a, real b, real c, real dx) {
		 return old_scf(a,b,c,opts.omega,opts.core_thresh_1,opts.core_thresh_2, dx);
		 }));
		 if (!opts.found_restart_file) {
		 if (opts.omega < ZERO) {
		 printf("Must specify omega for bibi polytrope\n");
		 throw;
		 }
		 if (opts.core_thresh_1 < ZERO) {
		 printf("Must specify core_thresh_1 for bibi polytrope\n");
		 throw;
		 }
		 if (opts.core_thresh_2 < ZERO) {
		 printf("Must specify core_thresh_2 for bibi polytrope\n");
		 throw;
		 }
		 }*/
	} else if (opts.problem == SOLID_SPHERE) {
		hydro_on = false;
		set_problem(init_func_type([](real x, real y, real z, real dx) {
			return solid_sphere(x,y,z,dx,0.25);
		}));
	} else {
		printf("No problem specified\n");
		throw;
	}
	node_server::set_gravity(gravity_on);
	node_server::set_hydro(hydro_on);
	compute_ilist();
}

HPX_PLAIN_ACTION(initialize, initialize_action);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(initialize_action)
HPX_REGISTER_BROADCAST_ACTION(initialize_action)

real OMEGA;
void node_server::set_pivot() {
    space_vector pivot = grid_ptr->center_of_mass();
    hpx::lcos::broadcast<set_pivot_action>(options::all_localities, pivot).get();
}

int hpx_main(int argc, char* argv[]) {
    printf("###########################################################\n");
    #if defined(__AVX512F__)
        printf("Compiled for AVX512 SIMD architectures.\n");
    #elif defined(__AVX2__)
        printf("Compiled for AVX2 SIMD architectures.\n");
    #elif defined(__AVX__)
        printf("Compiled for AVX SIMD architectures.\n");
    #elif defined(__SSE2__ )
        printf("Compiled for SSE2 SIMD architectures.\n");
    #else
        printf("Not compiled for a known SIMD architecture.\n");
    #endif
    printf("###########################################################\n");

    printf("Running\n");

    try {
        if (opts.process_options(argc, argv)) {
            auto all_locs = hpx::find_all_localities();
            hpx::lcos::broadcast<initialize_action>(all_locs, opts, all_locs).get();

            node_client root_id = hpx::new_ < node_server > (hpx::find_here());
            node_client root_client(root_id);
            node_server* root = root_client.get_ptr().get();

            int ngrids = 0;
            if (opts.found_restart_file) {
                set_problem(null_problem);
                const std::string fname = opts.restart_filename;
                printf("Loading from %s...\n", fname.c_str());
                if (opts.output_only) {
                    const std::string oname = opts.output_filename;
                    root->load_from_file_and_output(fname, oname, opts.data_dir);
                } else {
                    root->load_from_file(fname, opts.data_dir);
                    ngrids = root->regrid(root_client.get_gid(), ZERO, -1, true);

                    if (opts.max_restart_level > 0)
                    {
                        for (integer l = 0; l < opts.max_restart_level - 1; ++l)
                        {
                            ngrids = root->regrid(root_client.get_gid(), grid::get_omega(), -1, false);
                            printf("---------------Created Level %i---------------\n\n", int(l + 1));
                        }
                    }
                }
                printf("Done. \n");
            } else {
                for (integer l = 0; l < opts.max_level; ++l) {
                    ngrids = root->regrid(root_client.get_gid(), grid::get_omega(), -1, false);
                    printf("---------------Created Level %i---------------\n\n", int(l + 1));
                }
                ngrids = root->regrid(root_client.get_gid(), grid::get_omega(), -1, false);
                printf("---------------Regridded Level %i---------------\n\n", int(opts.max_level));
            }

            if (gravity_on && !opts.output_only) {
                printf("solving gravity------------\n");
                root->solve_gravity(false);
                printf("...done\n");
            }

            if (!opts.output_only) {
                //  set_problem(null_problem);
                hpx::async(&node_server::start_run, root, opts.problem == DWD && !opts.found_restart_file, ngrids).get();
//              root->start_run(opts.problem == DWD && !opts.found_restart_file, ngrids);
            }
            root->report_timing();
        }
    } catch (...) {
        throw;
    }
    printf("Exiting...\n");
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg = {
        "hpx.commandline.allow_unknown=1",         // HPX should not complain about unknown command line options
        "hpx.scheduler=local-priority-lifo",       // use LIFO scheduler by default
        "hpx.parcel.mpi.zero_copy_optimization!=0" // Disable the usage of zero copy optimization for MPI...
    };

    hpx::register_pre_shutdown_function([](){options::all_localities.clear(); });

    hpx::init(argc, argv, cfg);
}
