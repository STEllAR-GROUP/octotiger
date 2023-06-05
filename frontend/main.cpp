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

#ifndef OCTOTIGER_GIT_COMMIT_HASH
#define OCTOTIGER_GIT_COMMIT_HASH "unknown"
#endif
#ifndef OCTOTIGER_GIT_COMMIT_MESSAGE
#define OCTOTIGER_GIT_COMMIT_MESSAGE "unknown"
#endif

int hpx_main(int argc, char* argv[]) {

#ifdef OCTOTIGER_HAVE_KOKKOS
    // Initialize Kokkos on root
    std::cout << "Initializing Kokkos on Root locality" << std::endl;
    Kokkos::initialize(argc, argv);
#endif
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
    printf("###########################################################\n");

#ifdef OCTOTIGER_HAVE_KOKKOS
    Kokkos::print_configuration(std::cout, true);
#endif

    printf("\n###########################################################\n\n");

    printf("Running\n");

    start_octotiger(argc, argv);

    return hpx::finalize();
}

int main(int argc, char* argv[]) {
#ifdef OCTOTIGER_HAVE_UNBUFFERED_STDOUT
    std::setbuf(stdout, nullptr);
    std::cout << "Set to unbuffered stdout on current process... " << std::endl;
#endif
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
