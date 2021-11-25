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
}
#endif
