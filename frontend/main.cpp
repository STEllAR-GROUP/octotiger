//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config/compiler_specific.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/hpx_init.hpp>
#ifdef OCTOTIGER_HAVE_KOKKOS
#include <hpx/kokkos.hpp>
#endif

#include "frontend-helper.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cfenv>
#if !defined(_MSC_VER)
#include <unistd.h>
#else
#include <cfloat>
#endif

#include <iostream>

#ifdef OCTOTIGER_HAVE_HIP
#warning "Experimental HIP Build! Do not (yet) use for production runs"
#endif
<<<<<<< HEAD
=======
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
	} else if (opts().problem == ADVECTION) {
		grid::set_fgamma(5.0 / 3.0);
		set_analytic(advection_test_analytic);
		set_problem(advection_test_init);
		set_refine_test(refine_test_moving_star);
	} else if (opts().problem == AMR_TEST) {
		grid::set_fgamma(5.0 / 3.0);
//		grid::set_analytic_func(moving_star_analytic);
		set_problem(amr_test);
		set_refine_test(refine_test_moving_star);
		set_refine_test(refine_test_amr);
	} else if (opts().problem == MARSHAK) {
		grid::set_fgamma(5.0 / 3.0);
		set_analytic(nullptr);
		set_analytic(marshak_wave_analytic);
		set_problem(marshak_wave);
		set_refine_test(refine_test_marshak);
	} else if (opts().problem == RADIATION_DIFFUSION) {
		grid::set_fgamma(5.0 / 3.0);
		set_analytic(nullptr);
//		set_analytic(marshak_wave_analytic);
		set_problem(radiation_diffusion_test_problem);
		set_refine_test(radiation_test_refine);
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
>>>>>>> FLD

int hpx_main(int argc, char* argv[]) {

	    // The ascii logo was created by combining, modifying and extending the ascii arts from:
    // http://ascii.co.uk/art/octopus (Author "jgs")
    // and
    // http://www.ascii-art.de/ascii/t/tiger.txt (Author "fL")
    const char* logo = R"(



          ___       _      _____ _                 
         / _ \  ___| |_ __|_   _(_) __ _  ___ _ __ 
        | | | |/ __| __/ _ \| | | |/ _` |/ _ \ '__|
        | |_| | (__| || (_) | | | | (_| |  __/ |   
         \___/ \___|\__\___/|_| |_|\__, |\___|_|   
      _                            |___/           
     (_)
              _ __..-;''`-
      O   (`/' ` |  \ \ \ \-.
       o /'`\ \   |  \ | \|  \
        /<7' ;  \ \  | ; ||/ `'.
       /  _.-, `,-\,__| ' ' . `'.
       `-`  f/ ;       \        \             ___.--,
           `~-'_.._     |  -~    |     _.---'`__.-( (_.
        __.--'`_.. '.__.\    '--. \_.-' ,.--'`     `""`
       ( ,.--'`   ',__ /./;   ;, '.__.'`    __
       _`) )  .---.__.' / |   |\   \__..--""  """--.,_
      `---' .'.''-._.-'`_./  /\ '.  \ _.-~~~````~~~-._`-.__.'
            | |  .' _.-' |  |  \  \  '.               `~---`
             \ \/ .'     \  \   '. '-._)
              \/ /        \  \    `=.__`~-.
              / /\         `) )    / / `"".`\
        , _.-'.'\ \        / /    ( (     / /
         `--~`   ) )    .-'.'      '.'.  | (
                (/`    ( (`          ) )  '-;
                 `      '-;         (-'





    )";
    std::cout << logo << std::endl;

    // hpx::kokkos::ScopeGuard g(argc, argv);

    // TODO Why are these printfs? Replace by cout
    printf("###########################################################\n");
#if defined(__VSX__)
    printf("Compiled for VSX SIMD architectures.\n");
#elif defined(__AVX512F__)
    printf("Compiled for AVX512 SIMD architectures.\n");
#elif defined(__AVX2__)
    printf("Compiled for AVX2 SIMD architectures.\n");
#elif defined(__AVX__)
    printf("Compiled for AVX SIMD architectures.\n");
#elif defined(__SSE2__)
    printf("Compiled for SSE2 SIMD architectures.\n");
#else
    printf("Not compiled for a known SIMD architecture.\n");
#endif
#if defined(OCTOTIGER_FORCE_SCALAR_KOKKOS_SIMD)
    printf("Note: OCTOTIGER_FORCE_SCALAR_KOKKOS_SIMD is on! Kokkos kernel will not use SIMD!\n");
#endif
#ifdef OCTOTIGER_HAVE_HIP
    printf("WARNING: Experimental HIP Build! Do not (yet) use for production runs!\n");
#endif
    printf("###########################################################\n");

    printf("Running\n");

    start_octotiger(argc, argv);

    return hpx::finalize();
}

int main(int argc, char* argv[]) {
    hpx::init_params p;
    p.cfg = {
        "hpx.commandline.allow_unknown=1",    // HPX should not complain about unknown command line
                                              // options
        "hpx.scheduler=local-priority-lifo",          // Use LIFO scheduler by default
        "hpx.parcel.mpi.zero_copy_optimization!=0"    // Disable the usage of zero copy optimization
                                                      // for MPI...
    };
    register_hpx_functions();
    hpx::init(argc, argv, p);
#ifdef OCTOTIGER_HAVE_HIP
    std::cout << std::endl << "WARNING: Experimental HIP Build! Do not (yet) use for production runs!\n" << std::endl;
#endif
}
#endif
