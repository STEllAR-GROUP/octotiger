//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/resource_partitioner/partitioner.hpp>
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

int hpx_main(int argc, char* argv[]) {
    // The ascii logo was created by combining, modifying and extending the ascii arts from:
    // http://ascii.co.uk/art/octopus (Author "jgs")
    // and
    // http://www.ascii-art.de/ascii/t/tiger.txt (Author "fL")
    const char *logo = R"(



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
#elif defined(__SSE2__ )
	printf("Compiled for SSE2 SIMD architectures.\n");
#else
	printf("Not compiled for a known SIMD architecture.\n");
#endif
#if defined(OCTOTIGER_FORCE_SCALAR_KOKKOS_SIMD) 
	printf("Note: OCTOTIGER_FORCE_SCALAR_KOKKOS_SIMD is on! Kokkos kernel will not use SIMD!\n");
#endif
	printf("###########################################################\n");

	printf("Running\n");

    start_octotiger(argc, argv);

    return hpx::finalize();
}

// -------------------------------------------------------------------------
void init_resource_partitioner_handler(hpx::resource::partitioner& rp, const hpx::program_options::variables_map &vm)
{
    // how many threads are reserved for polling
    int polling_threads = vm["polling-threads"].as<int>();
    const std::string pool_name = "polling";

    if (polling_threads>0)
    {
        // background work will be done by polling pool
        using namespace hpx::threads::policies;
        rp.create_thread_pool(pool_name,
                              hpx::resource::scheduling_policy::shared_priority,
                              scheduler_mode::do_background_work
                              );
        // add N pus to network pool
        int count = 0;
        for (const hpx::resource::numa_domain& d : rp.numa_domains())
        {
            for (const hpx::resource::core& c : d.cores())
            {
                for (const hpx::resource::pu& p : c.pus())
                {
                    if (count < polling_threads)
                    {
                        std::cout << "Added pu " << count++ << " to pool \"" << pool_name << "\"\n";
                        rp.add_resource(p, pool_name);
                    }
                }
            }
        }
        {
            // remove background work flag from the default pool as this will be done by polling pool
            using namespace hpx::threads::policies;
            std::uint32_t deft = scheduler_mode::default_mode;
            std::uint32_t idle = scheduler_mode::enable_idle_backoff;
            std::uint32_t back = scheduler_mode::do_background_work;
            std::uint32_t mode = deft & ~idle & ~back;
            //
            rp.create_thread_pool("default",
                                  hpx::resource::scheduling_policy::unspecified,
                                  hpx::threads::policies::scheduler_mode(mode));
        }
    }
}

// -------------------------------------------------------------------------
int main(int argc, char* argv[]) {

    register_hpx_functions();

    hpx::program_options::options_description desc_cmdline("Options");
    desc_cmdline.add_options()
        ("polling-threads", hpx::program_options::value<int>()->default_value(0),
         "Enable dedicated HPX thread pool for cuda/network polling using N threads");

    std::vector<std::string> cfg = { "hpx.commandline.allow_unknown=1", // HPX should not complain about unknown command line options
        "hpx.scheduler=local-priority-lifo",       // Use LIFO scheduler by default
        "hpx.parcel.mpi.zero_copy_optimization!=0" // Disable the usage of zero copy optimization for MPI...
        };

    // Setup the init parameters
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_cmdline;
    init_args.rp_callback = &init_resource_partitioner_handler;
    init_args.cfg = cfg;
    hpx::init(argc, argv, init_args);
}
