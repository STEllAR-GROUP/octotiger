//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "frontend-helper.hpp"

#include "octotiger/compute_factor.hpp"
#include "octotiger/defs.hpp"
// #include "octotiger/future.hpp"
#include "octotiger/grid_fmm.hpp"
#include "octotiger/grid_scf.hpp"
#include "octotiger/node_client.hpp"
#include "octotiger/node_server.hpp"
#include "octotiger/options.hpp"
#include "octotiger/physcon.hpp"
#include "octotiger/problem.hpp"
#include "octotiger/test_problems/blast.hpp"
#include "octotiger/test_problems/rotating_star.hpp"
#include "octotiger/unitiger/physics.hpp"
#include "octotiger/unitiger/physics_impl.hpp"

#include "octotiger/test_problems/amr/amr.hpp"

#ifdef OCTOTIGER_HAVE_CUDA
#include <stream_manager.hpp>
#include "octotiger/cuda_util/cuda_helper.hpp"
#include "octotiger/cuda_util/cuda_scheduler.hpp"
#include "octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp"
#include "octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp"

#endif
#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/monopole_interactions/legacy/monopole_interaction_interface.hpp"
#include "octotiger/monopole_interactions/legacy/p2m_interaction_interface.hpp"
#include "octotiger/monopole_interactions/util/calculate_stencil.hpp"
#include "octotiger/multipole_interactions/legacy/multipole_interaction_interface.hpp"
#include "octotiger/multipole_interactions/util/calculate_stencil.hpp"

#include <hpx/collectives/broadcast_direct.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>
#if HPX_VERSION_FULL > 0x010600
// Can't find hpx::find_all_localities() in newer HPX versions without this header
#include <hpx/modules/runtime_distributed.hpp>
#endif

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

#if !defined(HPX_COMPUTE_DEVICE_CODE)


void initialize(options _opts, std::vector<hpx::id_type> const& localities) {
    std::cerr << "Started initialize method" << std::endl;

    scf_options::read_option_file();
    std::cerr << "Finished reading options" << std::endl;

    options::all_localities = localities;
    opts() = _opts;

    init_problem();
    std::cerr << "Finished initializing options" << std::endl;

    static_assert(octotiger::fmm::STENCIL_WIDTH <= INX, R"(
            ERROR: Stencil is too wide for the subgrid size. 
            Please increase either OCTOTIGER_THETA_MINIMUM or OCTOTIGER_WITH_GRIDDIM (see cmake file))");

#ifndef OCTOTIGER_DISABLE_ILIST
    compute_ilist();
    static_assert(INX <= 20, R"(
        ERROR: OCTOTIGER_WITH_GRIDDIM too large to compute the interaction list.
        Consider OCTOTIGER_DISALBE_ILIST=ON (for non-gravity scenarios only) or reduce the OCTOTIGER_WITH_GRIDDIM!)");
#else
    static_assert(INX > 16, R"(
        ERROR: The griddim is too small for OCTOTIGER_DISABLE_ILIST=ON. 
        For griddims <=16, there is no reason to disable the interaction list and it will limit the build to scenarios without gravity.
        This is likely an error! Either increase the OCTOTIGER_WITH_GRIDDIM or drop the OCTOTIGER_DISABLE_ILIST=ON flag!)");
#warning "Interaction list disabled. Distributed runs with gravity not possible using this build!"
#endif

    compute_factor();
    std::cerr << "Finished computing factors" << std::endl;

    init_executors();
    std::cerr << "Finished executor init" << std::endl;

    grid::static_init();
    std::cerr << "Finished static_init" << std::endl;
    normalize_constants();
    std::cerr << "Finished normalizing" << std::endl;
#ifdef SILO_UNITS
//	grid::set_unit_conversions();
#endif
    std::cerr << "Finished initialization" << std::endl;
}

void cleanup_buffers() {
    cleanup_puddle_on_this_locality();
    return;
}
HPX_PLAIN_ACTION(cleanup_buffers, cleanup_buffers_action);

void cleanup() {
    std::vector<hpx::naming::id_type> localities = hpx::find_all_localities();

    std::vector<hpx::lcos::future<void>> futures;
    futures.reserve(localities.size());

    for (hpx::naming::id_type const& node : localities) {
        using action_type = cleanup_buffers_action;
        futures.push_back(hpx::async<action_type>(node));
    }
    auto f = hpx::when_all(futures.begin(), futures.end());
    f.get();
}

HPX_PLAIN_ACTION(initialize, initialize_action);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(initialize_action);
HPX_REGISTER_BROADCAST_ACTION(initialize_action);

void start_octotiger(int argc, char* argv[]) {
    std::cerr << "Start octotiger" << std::endl;
    try {
        std::cerr << "Start processing options" << std::endl;
        if (opts().process_options(argc, argv)) {
            std::cerr << "Finished processing options" << std::endl;
            auto all_locs = hpx::find_all_localities();
            hpx::lcos::broadcast<initialize_action>(all_locs, opts(), all_locs).get();
            std::cerr << "Finished init" << std::endl;

            hpx::id_type root_id = hpx::new_<node_server>(hpx::find_here()).get();
            node_client root_client(root_id);
            node_server* root = root_client.get_ptr().get();
            std::cerr << "Found root" << std::endl;

            node_count_type ngrids;
            //		printf("1\n");
            if (!opts().restart_filename.empty()) {
                std::cerr << "Loading from " << opts().restart_filename << " ...\n";
                load_data_from_silo(opts().restart_filename, root, root_client.get_unmanaged_gid());
                std::cerr << "Re-grid" << std::endl;
                ngrids = root->regrid(root_client.get_unmanaged_gid(), ZERO, -1, true, false);
                std::cerr << "Done!" << std::endl;

                set_AB(physcon().A, physcon().B);

            } else {
                std::cerr << "Starting refinement" << std::endl;
                for (integer l = 0; l < opts().max_level; ++l) {
                    ngrids =
                        root->regrid(root_client.get_gid(), grid::get_omega(), -1, false, false);
                    std::cerr << "---------------Created Level " << int(l + 1)
                              << "---------------\n"
                              << std::endl;
                }
                ngrids = root->regrid(root_client.get_gid(), grid::get_omega(), -1, false, false);
                std::cerr << "---------------Re-gridded Level " << int(opts().max_level)
                          << "---------------\n"
                          << std::endl;
            }
            for (integer l = 0; l < opts().extra_regrid; ++l) {
                std::cerr << "Starting extra regridding step..." << std::endl;
                ngrids = root->regrid(root_client.get_gid(), grid::get_omega(), -1, false, false);
                std::cerr << "Finished extra regridding step..." << std::endl;
            }

            if (opts().gravity && opts().stop_step != 0) {
                std::cerr << "solving gravity------------" << std::endl;
                root->solve_gravity(false, false);
                std::cerr << "...done" << std::endl;
            }
            if (opts().problem != AMR_TEST) {
                std::cerr << "Start execution the solver..." << std::endl;
                hpx::async(&node_server::execute_solver, root,
                    opts().problem == DWD && opts().restart_filename.empty(), ngrids)
                    .get();
                std::cerr << "Finished solver exeuction - Scenario done!" << std::endl;
            } else {
                std::cerr << "Start AMR test..." << std::endl;
                root->enforce_bc();
                auto e = root->amr_error();
                std::cerr << "Finished AMR test" << std::endl;
                printf("AMR Error: %e %e %e\n", e.first, e.second, e.first / e.second);
                output_all(root, "X", 0, true);
            }
            std::cerr << "Start timings report..." << std::endl;
            root->report_timing();
            std::cerr << "Finished timings report!" << std::endl;
            std::cerr << "Start cleanup..." << std::endl;
            cleanup();    // cleanup buffer and executor pools
            std::cerr << "Localities cleanup finished" << std::endl;
        }
    } catch (...) {
        throw;
    }
    printf("Exiting...\n");
    FILE* fp = fopen("profile.txt", "wt");
    profiler_output(fp);
    fclose(fp);
}

void register_hpx_functions(void) {
    hpx::register_startup_function(&node_server::register_counters);
    hpx::register_pre_shutdown_function([]() { options::all_localities.clear(); });
}
#endif
