#include "octotiger/compute_factor.hpp"
#include "octotiger/defs.hpp"
#include "octotiger/future.hpp"
#include "octotiger/node_client.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/problem.hpp"
#include "octotiger/test_problems/rotating_star/rotating_star.hpp"

#ifdef OCTOTIGER_HAVE_CUDA
#include "octotiger/cuda_util/cuda_helper.hpp"
#include "octotiger/cuda_util/cuda_scheduler.hpp"
#include "octotiger/monopole_interactions/cuda_p2p_interaction_interface.hpp"
#include "octotiger/multipole_interactions/cuda_multipole_interaction_interface.hpp"
#endif
#include "octotiger/monopole_interactions/calculate_stencil.hpp"
#include "octotiger/monopole_interactions/p2m_interaction_interface.hpp"
#include "octotiger/monopole_interactions/p2p_interaction_interface.hpp"
#include "octotiger/multipole_interactions/calculate_stencil.hpp"
#include "octotiger/multipole_interactions/multipole_interaction_interface.hpp"

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>
#include <hpx/lcos/broadcast.hpp>

#include <chrono>
#include <cstdio>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <fenv.h>
#if !defined(_MSC_VER)
#include <unistd.h>
#else
#include <float.h>
#endif

namespace scf_options {
void read_option_file();
}

void normalize_constants();

void compute_ilist();

std::size_t init_thread_local_worker(std::size_t desired)
{
    std::size_t current = hpx::get_worker_thread_num();
    if (current == desired)
    {
#ifdef OCTOTIGER_HAVE_CUDA
        // Initialize CUDA/CPU scheduler
        octotiger::fmm::kernel_scheduler::scheduler.init();
#endif

        namespace mono_inter = octotiger::fmm::monopole_interactions;
        using mono_inter_p2p =
            octotiger::fmm::monopole_interactions::p2p_interaction_interface;
        // Initialize stencil and four constants for p2p fmm interactions
        mono_inter_p2p::stencil = mono_inter::calculate_stencil().first;
        mono_inter_p2p::stencil_masks() =
            mono_inter::calculate_stencil_masks(mono_inter_p2p::stencil).first;
        mono_inter_p2p::four = mono_inter::calculate_stencil().second;
        mono_inter_p2p::stencil_four_constants() =
            mono_inter::calculate_stencil_masks(mono_inter_p2p::stencil).second;

        // Initialize stencil for p2m fmm interactions
        mono_inter::p2m_interaction_interface::stencil =
            mono_inter::calculate_stencil().first;

        namespace multi_inter = octotiger::fmm::multipole_interactions;
        using multi_inter_p2p = octotiger::fmm::multipole_interactions::
            multipole_interaction_interface;
        // Initialize stencil for multipole fmm interactions
        multi_inter_p2p::stencil = multi_inter::calculate_stencil();
        multi_inter_p2p::stencil_masks() =
            multi_inter::calculate_stencil_masks(multi_inter_p2p::stencil)
                .first;
        multi_inter_p2p::inner_stencil_masks() =
            multi_inter::calculate_stencil_masks(multi_inter_p2p::stencil)
                .second;

        std::cout << "OS-thread " << current << " on locality "
                  << hpx::get_locality_id()
                  << ": Initialized thread_local memory!\n";
        return desired;
    }
    // NOTE: This might be an issue. Throw an exception and/or make the output
    // a tuple with the second being the error code
    return std::size_t(-1);
}
HPX_PLAIN_ACTION(init_thread_local_worker, init_thread_local_worker_action);

#ifdef OCTOTIGER_HAVE_CUDA
std::array<size_t, 7> sum_counters_worker(std::size_t desired)
{
    std::array<size_t, 7> ret;
    ret[0] = std::size_t(-1);
    ret[1] = 0;
    ret[2] = 0;
    ret[3] = 0;
    ret[4] = 0;
    ret[5] = 0;
    ret[6] = 0;
    std::size_t current = hpx::get_worker_thread_num();
    if (current == desired)
    {
        using cuda_multi_intfc = octotiger::fmm::multipole_interactions::
            cuda_multipole_interaction_interface;
        using cuda_mono_intfc = octotiger::fmm::monopole_interactions::
            cuda_p2p_interaction_interface;

        ret[0] = desired;
        ret[1] = cuda_multi_intfc::cpu_launch_counter;
        ret[2] = cuda_multi_intfc::cuda_launch_counter;

        ret[3] = cuda_mono_intfc::cpu_launch_counter;
        ret[4] = cuda_mono_intfc::cuda_launch_counter;

        ret[5] = cuda_multi_intfc::cpu_launch_counter_non_rho;
        ret[6] = cuda_multi_intfc::cuda_launch_counter_non_rho;

        // std::cout << "OS-thread " << ret[1] << " "
        //           << ret[2] << " " << ret[3]
        //           << " " << ret[4] << std::endl;
    }
    return ret;
}
HPX_PLAIN_ACTION(sum_counters_worker, sum_counters_worker_action);
#endif

void initialize(options _opts, std::vector<hpx::id_type> const& localities) {
	scf_options::read_option_file();

	options::all_localities = localities;
	opts() = _opts;
	grid::get_omega() = opts().omega;
#if !defined(_MSC_VER) && !defined(__APPLE__)
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
#if defined(OCTOTIGER_HAVE_BLAST_TEST)
	} else if (opts().problem == BLAST) {
		grid::set_fgamma(7.0 / 5.0);
//		opts().gravity = false;
		set_problem(blast_wave);
		set_refine_test(refine_blast);
		set_analytic(blast_wave_analytic);
#endif
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
		set_analytic([](real x, real y, real z, real dx) {
			return solid_sphere(x,y,z,dx,0.25);
		});
		set_refine_test(refine_test_center);
		set_problem(init_func_type([](real x, real y, real z, real dx) {
			return solid_sphere(x,y,z,dx,0.25);
		}));
	} else {
		printf("No problem specified\n");
		throw;
	}
	compute_ilist();
	compute_factor();

#ifdef OCTOTIGER_HAVE_CUDA
	std::cout << "CUDA is enabled! Available CUDA targets on this locality: " << std::endl;
	octotiger::util::cuda_helper::print_local_targets();
    octotiger::fmm::kernel_scheduler::init_constants();
#endif
	grid::static_init();
	normalize_constants();
#ifdef SILO_UNITS
//	grid::set_unit_conversions();
#endif
    std::size_t const os_threads = hpx::get_os_thread_count();
    hpx::naming::id_type const here = hpx::find_here();
    std::set<std::size_t> attendance;
    for (std::size_t os_thread = 0; os_thread < os_threads; ++os_thread)
        attendance.insert(os_thread);
    while (!attendance.empty())
    {
        std::vector<hpx::lcos::future<std::size_t> > futures;
        futures.reserve(attendance.size());

        for (std::size_t worker : attendance)
        {
            using action_type = init_thread_local_worker_action;
            futures.push_back(hpx::async<action_type>(here, worker));
        }
        hpx::lcos::local::spinlock mtx;
        hpx::lcos::wait_each(
            hpx::util::unwrapping([&](std::size_t t) {
                if (std::size_t(-1) != t)
                {
                    std::lock_guard<hpx::lcos::local::spinlock> lk(mtx);
                    attendance.erase(t);
                }
            }),
            futures);
    }
}

void analyze_local_launch_counters() {
#ifdef OCTOTIGER_HAVE_CUDA
    std::size_t const os_threads = hpx::get_os_thread_count();
    hpx::naming::id_type const here = hpx::find_here();
    std::set<std::size_t> attendance;
    for (std::size_t os_thread = 0; os_thread < os_threads; ++os_thread)
        attendance.insert(os_thread);
    size_t total_multipole_cpu_launches = 0;
    size_t total_multipole_cuda_launches = 0;
    size_t total_multipole_cpu_launches_non_rho = 0;
    size_t total_multipole_cuda_launches_non_rho = 0;
    size_t total_p2p_cpu_launches = 0;
    size_t total_p2p_cuda_launches = 0;
    while (!attendance.empty())
    {
        std::vector<hpx::lcos::future<std::array<size_t, 7>>> futures;
        futures.reserve(attendance.size());

        for (std::size_t worker : attendance)
        {
            using action_type = sum_counters_worker_action;
            futures.push_back(hpx::async<action_type>(here, worker));
        }
        hpx::lcos::local::spinlock mtx;
        hpx::lcos::wait_each(
            hpx::util::unwrapping([&](std::array<size_t, 7> t) {
                if (std::size_t(-1) != t[0])
                {
                    std::lock_guard<hpx::lcos::local::spinlock> lk(mtx);
                    total_multipole_cpu_launches += t[1];
                    total_multipole_cuda_launches += t[2];
                    total_p2p_cpu_launches += t[3];
                    total_p2p_cuda_launches += t[4];
                    total_multipole_cpu_launches_non_rho += t[5];
                    total_multipole_cuda_launches_non_rho += t[6];
                    attendance.erase(t[0]);
                }
                }),
            futures);
    }
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Total multipole launches: "
              << total_multipole_cpu_launches + total_multipole_cuda_launches << std::endl;
    std::cout << "CPU multipole launches " << total_multipole_cpu_launches << std::endl;
    std::cout << "CUDA multipole launches " << total_multipole_cuda_launches << std::endl;
    if (total_multipole_cpu_launches + total_multipole_cuda_launches > 0) {
        float percentage = static_cast<float>(total_multipole_cuda_launches) /
            (static_cast<float>(total_multipole_cuda_launches) + total_multipole_cpu_launches);
        std::cout << "=> Percentage of multipole on the GPU: " << percentage * 100 << "\n";
    }
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Total non-rho-multipole launches: "
              << total_multipole_cpu_launches_non_rho + total_multipole_cuda_launches_non_rho << std::endl;
    std::cout << "CPU non-rho-multipole launches " << total_multipole_cpu_launches_non_rho << std::endl;
    std::cout << "CUDA non-rho-multipole launches " << total_multipole_cuda_launches_non_rho << std::endl;
    if (total_multipole_cpu_launches_non_rho + total_multipole_cuda_launches_non_rho > 0) {
        float percentage = static_cast<float>(total_multipole_cuda_launches_non_rho) /
            (static_cast<float>(total_multipole_cuda_launches_non_rho) + total_multipole_cpu_launches_non_rho);
        std::cout << "=> Percentage of non-rho-multipole on the GPU: " << percentage * 100 << "\n";
    }
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Total p2p launches: "
              << total_p2p_cpu_launches + total_p2p_cuda_launches << std::endl;
    std::cout << "CPU p2p launches " << total_p2p_cpu_launches << std::endl;
    std::cout << "CUDA p2p launches " << total_p2p_cuda_launches << std::endl;
    if (total_p2p_cpu_launches + total_p2p_cuda_launches > 0) {
        float percentage = static_cast<float>(total_p2p_cuda_launches) /
            (static_cast<float>(total_p2p_cuda_launches) + total_p2p_cpu_launches);
        std::cout << "=> Percentage of p2p on the GPU: " << percentage * 100 << "\n";
    }
#endif
}




HPX_PLAIN_ACTION(initialize, initialize_action);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(initialize_action);
HPX_REGISTER_BROADCAST_ACTION(initialize_action);

real OMEGA;
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

			node_count_type ngrids = {0,0};
			//		printf("1\n");
			if (!opts().restart_filename.empty()) {
				std::cout << "Loading from " << opts().restart_filename << " ...\n";
				load_data_from_silo(opts().restart_filename, root, root_client.get_unmanaged_gid());
				printf( "Re-grid\n");
				ngrids = root->regrid(root_client.get_unmanaged_gid(), ZERO, -1, true, false);
				printf("Done. \n");
			} else {
				for (integer l = 0; l < opts().max_level; ++l) {
					ngrids = root->regrid(root_client.get_gid(), grid::get_omega(), -1, false);
					printf("---------------Created Level %i---------------\n\n", int(l + 1));
				}
				ngrids = root->regrid(root_client.get_gid(), grid::get_omega(), -1, false);
				printf("---------------Re-gridded Level %i---------------\n\n", int(opts().max_level));
			}
			if (opts().gravity) {
				printf("solving gravity------------\n");
				root->solve_gravity(false, false);
				printf("...done\n");
			}
			hpx::async(&node_server::execute_solver, root, opts().problem == DWD && opts().restart_filename.empty(), ngrids).get();
			root->report_timing();
            analyze_local_launch_counters();
		}
	} catch (...) {
		throw;
	}
	printf("Exiting...\n");
	return hpx::finalize();
}

int main(int argc, char* argv[]) {
	std::vector<std::string> cfg = { "hpx.commandline.allow_unknown=1", // HPX should not complain about unknown command line options
			"hpx.scheduler=local-priority-lifo",       // Use LIFO scheduler by default
			"hpx.parcel.mpi.zero_copy_optimization!=0" // Disable the usage of zero copy optimization for MPI...
			};

	hpx::register_pre_shutdown_function([]() {options::all_localities.clear();});

	hpx::init(argc, argv, cfg);
}
