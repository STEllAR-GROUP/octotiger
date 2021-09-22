//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)

#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/common_kernel/multiindex.hpp"
#include "octotiger/common_kernel/struct_of_array_data.hpp"
#include "octotiger/config/export_definitions.hpp"
#include "octotiger/cuda_util/cuda_helper.hpp"
#include "octotiger/real.hpp"
#include "octotiger/taylor.hpp"

#include <cstddef>
#include <vector>

namespace octotiger { namespace fmm {
    // Define sizes of CUDA buffers
    constexpr std::size_t local_monopoles_size =
        NUMBER_LOCAL_MONOPOLE_VALUES * sizeof(real);
    constexpr std::size_t local_expansions_size =
        NUMBER_LOCAL_EXPANSION_VALUES * sizeof(real);
    constexpr std::size_t center_of_masses_size =
        NUMBER_MASS_VALUES * sizeof(real);
    constexpr std::size_t potential_expansions_size =
        NUMBER_POT_EXPANSIONS * sizeof(real);
    constexpr std::size_t potential_expansions_small_size =
        NUMBER_POT_EXPANSIONS_SMALL * sizeof(real);
    constexpr std::size_t angular_corrections_size =
        NUMBER_ANG_CORRECTIONS * sizeof(real);
    constexpr std::size_t stencil_size =
        P2P_PADDED_STENCIL_SIZE * sizeof(octotiger::fmm::multiindex<>);
    constexpr std::size_t indicator_size = STENCIL_SIZE * sizeof(real);
    constexpr std::size_t four_constants_size =
        4 * P2P_PADDED_STENCIL_SIZE * sizeof(real);
    constexpr std::size_t full_stencil_size = FULL_STENCIL_SIZE * sizeof(real);

    // Scheduler which decides on what device to launch kernel and what memory to use
    class kernel_scheduler
    {
    public:
        static OCTOTIGER_EXPORT kernel_scheduler& scheduler();

        OCTOTIGER_EXPORT void init();
        static OCTOTIGER_EXPORT void init_constants();

    public:
        kernel_scheduler(kernel_scheduler& other) = delete;
        kernel_scheduler(kernel_scheduler const& other) = delete;
        kernel_scheduler operator=(kernel_scheduler const& other) = delete;
        // Deallocates CUDA memory
        ~kernel_scheduler();

    private:
        // Constructs number of streams indicated by the options
        kernel_scheduler();
    };
}}
#endif
