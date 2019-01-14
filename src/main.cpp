#include "defs.hpp"
#include "./test_problems/rotating_star/rotating_star.hpp"

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

#include "compute_factor.hpp"

#ifdef OCTOTIGER_WITH_CUDA
#include "cuda_util/cuda_helper.hpp"
#endif


void normalize_constants();

void compute_ilist();

void initialize(options _opts, std::vector<hpx::id_type> const& localities) {
	options::all_localities = localities;
	opts() = _opts;
	grid::get_omega() = opts().omega;
#if !defined(_MSC_VER)
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);
#else
	_controlfp(_EM_INEXACT | _EM_DENORMAL | _EM_INVALID, _MCW_EM);
#endif
	grid::set_scaling_factor(opts().xscale);
	grid::set_max_level(opts().max_level);
	if (opts().problem == RADIATION_TEST) {
		assert(opts().radiation);
//		opts().gravity = false;
		set_problem(radiation_test_problem);
		set_refine_test(radiation_test_refine);
	} else if (opts().problem == DWD) {
		opts().n_species=5;
		set_problem(scf_binary);
		set_refine_test(refine_test);
	} else if (opts().problem == SOD) {
		grid::set_fgamma(7.0 / 5.0);
//		opts().gravity = false;
		set_problem(sod_shock_tube_init);
		set_refine_test(refine_sod);
		set_analytic(sod_shock_tube_analytic);
	} else if (opts().problem == BLAST) {
		grid::set_fgamma(7.0 / 5.0);
//		opts().gravity = false;
		set_problem(blast_wave);
		set_refine_test(refine_blast);
		set_analytic(blast_wave_analytic);
	} else if (opts().problem == STAR) {
		grid::set_fgamma(5.0 / 3.0);
		set_problem(star);
		set_refine_test(refine_test_moving_star);
	} else if (opts().problem == ROTATING_STAR) {
		grid::set_fgamma(5.0 / 3.0);
		set_problem(rotating_star);
		set_analytic(rotating_star_a);
		set_refine_test(refine_test_moving_star);
	} else if (opts().problem == MOVING_STAR) {
		grid::set_fgamma(5.0 / 3.0);
//		grid::set_analytic_func(moving_star_analytic);
		set_problem(moving_star);
		set_refine_test(refine_test_moving_star);
	} else if (opts().problem == MARSHAK) {
		grid::set_fgamma(5.0 / 3.0);
		set_analytic(nullptr);
		set_analytic(marshak_wave_analytic);
		set_problem(marshak_wave);
		set_refine_test(refine_test_marshak);
	} else if (opts().problem == SOLID_SPHERE) {
	//	opts().hydro = false;
		set_problem(init_func_type([](real x, real y, real z, real dx) {
			return solid_sphere(x,y,z,dx,0.25);
		}));
	} else {
		printf("No problem specified\n");
		throw;
	}
	compute_ilist();
	compute_factor();

#ifdef OCTOTIGER_WITH_CUDA
	std::cout << "Cuda is enabled! Available cuda targets on this localility: " << std::endl;
	octotiger::util::cuda_helper::print_local_targets();
#endif
	grid::static_init();
	normalize_constants();
#ifdef SILO_UNITS
//	grid::set_unit_conversions();
#endif
}

HPX_PLAIN_ACTION(initialize, initialize_action);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(initialize_action);
HPX_REGISTER_BROADCAST_ACTION(initialize_action);

real OMEGA;
namespace scf_options {
void read_option_file();
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
		if (opts().process_options(argc, argv)) {
			auto all_locs = hpx::find_all_localities();
			hpx::lcos::broadcast<initialize_action>(all_locs, opts(), all_locs).get();

			hpx::id_type root_id = hpx::new_<node_server>(hpx::find_here()).get();
			node_client root_client(root_id);
			node_server* root = root_client.get_ptr().get();

			int ngrids = 0;
			//		printf("1\n");
			if (!opts().restart_filename.empty()) {
				std::cout << "Loading from " << opts().restart_filename << " ...\n";
				load_data_from_silo(opts().restart_filename, root, root_client.get_unmanaged_gid());
				printf( "Regrid\n");
				ngrids = root->regrid(root_client.get_unmanaged_gid(), ZERO, -1, true, false);
				printf("Done. \n");
			} else {
				for (integer l = 0; l < opts().max_level; ++l) {
					ngrids = root->regrid(root_client.get_gid(), grid::get_omega(), -1, false);
					printf("---------------Created Level %i---------------\n\n", int(l + 1));
				}
				ngrids = root->regrid(root_client.get_gid(), grid::get_omega(), -1, false);
				printf("---------------Regridded Level %i---------------\n\n", int(opts().max_level));
			}
			if (opts().gravity) {
				printf("solving gravity------------\n");
				root->solve_gravity(false, false);
				printf("...done\n");
			}
			hpx::async(&node_server::start_run, root, opts().problem == DWD && opts().restart_filename.empty(), ngrids).get();
			root->report_timing();
		}
	} catch (...) {
		throw;
	}
	printf("Exiting...\n");
	return hpx::finalize();
}

int main(int argc, char* argv[]) {
	std::vector<std::string> cfg = { "hpx.commandline.allow_unknown=1", // HPX should not complain about unknown command line options
			"hpx.scheduler=local-priority-lifo",       // use LIFO scheduler by default
			"hpx.parcel.mpi.zero_copy_optimization!=0" // Disable the usage of zero copy optimization for MPI...
			};

	hpx::register_pre_shutdown_function([]() {options::all_localities.clear();});

	hpx::init(argc, argv, cfg);
}
