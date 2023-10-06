//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config/compiler_specific.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)

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

#ifdef OCTOTIGER_HAVE_HIP
#warning "Experimental HIP Build! Do not (yet) use for production runs"
#include <hip/hip_runtime.h>
#endif

#ifndef OCTOTIGER_GIT_COMMIT_HASH
#define OCTOTIGER_GIT_COMMIT_HASH "unknown"
#endif
#ifndef OCTOTIGER_GIT_COMMIT_MESSAGE
#define OCTOTIGER_GIT_COMMIT_MESSAGE "unknown"
#endif
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
    std::cout << "GIT COMMIT: " << OCTOTIGER_GIT_COMMIT_HASH << std::endl 
              << "            \""  << OCTOTIGER_GIT_COMMIT_MESSAGE << "\"" << std::endl;
#ifdef OCTOTIGER_GIT_REPO_DIRTY
    std::cout << "\nReproducibility Warning: Octo-Tiger source directory contained uncommitted "
                 "changes during the CMake configuration step! " << std::endl;
#endif
    std::cout << std::endl;

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
#ifdef OCTOTIGER_HAVE_KOKKOS
#if defined(OCTOTIGER_KOKKOS_SIMD_AUTOMATIC_DISCOVERY) 
    printf("Note: Kokkos kernels will try to use this SIMD type!\n");
#elif defined(OCTOTIGER_KOKKOS_SIMD_AVX512)
    printf("Note: Kokkos CPU kernels are manually set to AVX512 as per CMAKE configuration!\n");
#elif defined(OCTOTIGER_KOKKOS_SIMD_AVX)
    printf("Note: Kokkos CPU kernels are manually set to AVX as per CMAKE configuration!\n");
#elif defined(OCTOTIGER_KOKKOS_SIMD_VSX)
    printf("Note: Kokkos CPU kernels are manually set to VSX as per CMAKE configuration!\n");
#elif defined(OCTOTIGER_KOKKOS_SIMD_SVE)
    printf("Note: Kokkos CPU kernels are manually set to SVE as per CMAKE configuration!\n");
#elif defined(OCTOTIGER_KOKKOS_SIMD_NEON)
    printf("Note: Kokkos CPU kernels are manually set to NEON as per CMAKE configuration!\n");
#elif defined(OCTOTIGER_KOKKOS_SIMD_SCALAR)
    printf("Note: Kokkos kernels are manually set to SCALAR as per CMAKE configuration!\n");
    printf("Note: Kokkos kernels are will not use explicit vectorization in this configuration!\n");
#endif
#if defined(OCTOTIGER_HAVE_STD_EXPERIMENTAL_SIMD)
    printf("Using std::experimential::simd SIMD types.\n");
#else
    printf("Using Kokkos SIMD types.\n");
#endif
#endif
#ifdef OCTOTIGER_HAVE_HIP
    printf("WARNING: Experimental HIP Build! Do not (yet) use for production runs!\n");
#endif
#if defined(CPPUDDLE_DEACTIVATE_BUFFER_RECYCLING)
    printf("WARNING: Using build without buffer recycling enabled. This will cause a major degradation of GPU performance !\n");
    printf("         Consider recompiling CPPuddle (and Octo-Tiger) with CPPUDDLE_WITH_BUFFER_RECYCLING=ON !\n");
#endif
#if defined(CPPUDDLE_DEACTIVATE_AGGRESSIVE_ALLOCATORS)
    printf("WARNING: Using build without buffer content recycling enabled. This will cause a slight degradation performance !\n");
    printf("         Consider recompiling CPPuddle (and Octo-Tiger) with CPPUDDLE_WITH_AGGRESSIVE_CONTENT_RECYCLING=ON !\n");
#endif
    printf("###########################################################\n");


    printf("\n###########################################################\n\n");

    printf("Running\n");

    start_octotiger(argc, argv);

    return hpx::finalize();
}

void init_resource_partitioner_handler(hpx::resource::partitioner& rp,
    const hpx::program_options::variables_map &vm) {
    // how many threads are reserved for polling
    int polling_threads = vm["polling-threads"].as<int>();
    const std::string pool_name = "polling";
    if (polling_threads > 0) {
        // background work will be done by polling pool
        using namespace hpx::threads::policies;
        rp.create_thread_pool(pool_name, hpx::resource::scheduling_policy::shared_priority,
            scheduler_mode::do_background_work);
        // add N pus to polling pool
        int count = 0;
        for (const hpx::resource::numa_domain& d : rp.numa_domains()) {
            for (auto it = d.cores().rbegin(); it != d.cores().rend(); it++) {
                for (const hpx::resource::pu& p : (*it).pus()) {
                    if (count < polling_threads) {
                        std::cout << "Added pu " << count++ << " to pool \"" <<
                          pool_name << "\"\n";
                        rp.add_resource(p, pool_name);
                    }
                }
            }
        }

        {
            // remove background work flag from the default pool as this will be done by polling pool
            using namespace hpx::threads::policies;
            auto deft = scheduler_mode::default_;
            auto idle = scheduler_mode::enable_idle_backoff;
            std::uint32_t mode = deft & ~idle; 
            //
            rp.create_thread_pool("default",
                                  hpx::resource::scheduling_policy::unspecified,
                                  hpx::threads::policies::scheduler_mode(mode));
        }
    }
}

int main(int argc, char* argv[]) {

#if defined(OCTOTIGER_HAVE_HIP) || (defined(OCTOTIGER_HAVE_KOKKOS) && defined(KOKKOS_ENABLE_HIP))
    // Touch all AMDGPUs before before starting HPX. This initializes all GPUs before starting HPX
    // which avoids multithreaded initialization later on which makes the driver segfault
    //
    // See bug https://github.com/ROCm-Developer-Tools/HIP/issues/3063
    //
    int numDevices = 0;
    hipGetDeviceCount(&numDevices);
    for (size_t gpu_id = 0; gpu_id < numDevices; gpu_id++) {
      hipSetDevice(gpu_id);
      hipStream_t gpu1;
      hipStreamCreate(&gpu1);
      hipStreamDestroy(gpu1);
      hipDeviceSynchronize();
    }
#endif
#ifdef OCTOTIGER_HAVE_UNBUFFERED_STDOUT
    std::setbuf(stdout, nullptr);
    std::cout << "Set to unbuffered stdout on current process... " << std::endl;
#endif
    std::cerr << "Starting main..." << std::endl;
    std::cerr << "Registering functions ..." << std::endl;
    register_hpx_functions();
    register_cppuddle_allocator_counters();

    hpx::program_options::options_description desc_cmdline("Options");
    desc_cmdline.add_options()
        ("polling-threads", hpx::program_options::value<int>()->default_value(0),
         "Enable dedicated HPX thread pool for cuda/network polling using N threads");
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_cmdline;
    init_args.rp_callback = &init_resource_partitioner_handler;
    init_args.cfg = {
        "hpx.commandline.allow_unknown=1"    // HPX should not complain about unknown command line
    };
    std::cerr << "Starting hpx init ..." << std::endl;
    hpx::init(argc, argv, init_args);
#ifdef OCTOTIGER_HAVE_HIP
    std::cout << std::endl << "WARNING: Experimental HIP Build! Do not (yet) use for production runs!\n" << std::endl;
#endif
#if defined(CPPUDDLE_DEACTIVATE_BUFFER_RECYCLING)
    std::cout << "WARNING: Using build without buffer recycling enabled. " 
              << "This will cause a major degradation of GPU performance !\n";
    std::cout << "         Consider recompiling CPPuddle (and Octo-Tiger) with "
              << "CPPUDDLE_WITH_BUFFER_RECYCLING=ON !\n";
#endif
#if defined(CPPUDDLE_DEACTIVATE_AGGRESSIVE_ALLOCATORS)
    std::cout << "WARNING: Using build without buffer content recycling enabled. "
              << "This will cause a slight degradation performance !\n";
    std::cout << "         Consider recompiling CPPuddle (and Octo-Tiger) with "
              << "CPPUDDLE_WITH_AGGRESSIVE_CONTENT_RECYCLING=ON !\n";
#endif

}
#endif

