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
#include "octotiger/monopole_interactions/util/calculate_stencil.hpp"
#include "octotiger/monopole_interactions/legacy/p2m_interaction_interface.hpp"
#include "octotiger/monopole_interactions/legacy/monopole_interaction_interface.hpp"
#include "octotiger/multipole_interactions/util/calculate_stencil.hpp"
#include "octotiger/multipole_interactions/legacy/multipole_interaction_interface.hpp"

#include <hpx/collectives/broadcast_direct.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>

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

std::array<size_t, 7> sum_counters_worker(std::size_t desired) {
    std::array<size_t, 7> ret;
    ret[0] = std::size_t(-1);
    ret[1] = 0;
    ret[2] = 0;
    ret[3] = 0;
    ret[4] = 0;
    ret[5] = 0;
    ret[6] = 0;
    std::size_t current = hpx::get_worker_thread_num();
    if (current == desired) {
        using cuda_multi_intfc =
            octotiger::fmm::multipole_interactions::multipole_interaction_interface;
        using cuda_mono_intfc = octotiger::fmm::monopole_interactions::monopole_interaction_interface;

        ret[0] = desired;
        ret[1] = cuda_multi_intfc::cpu_launch_counter();
        ret[2] = cuda_multi_intfc::cuda_launch_counter();

        ret[3] = cuda_mono_intfc::cpu_launch_counter();
        ret[4] = cuda_mono_intfc::cuda_launch_counter();

        ret[5] = cuda_multi_intfc::cpu_launch_counter_non_rho();
        ret[6] = cuda_multi_intfc::cuda_launch_counter_non_rho();

        // std::cout << "OS-thread " << ret[1] << " "
        //           << ret[2] << " " << ret[3]
        //           << " " << ret[4] << std::endl;
    }
    return ret;
}
HPX_PLAIN_ACTION(sum_counters_worker, sum_counters_worker_action);

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

    // compute_ilist();
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
    return ;
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
    std::cout << "Localities cleanup finished" << std::endl;
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

            hpx::id_type root_id = hpx::new_<node_server>(hpx::find_here()).get();
            node_client root_client(root_id);
            node_server* root = root_client.get_ptr().get();

            node_count_type ngrids;
            //		printf("1\n");
            if (!opts().restart_filename.empty()) {
                std::cout << "Loading from " << opts().restart_filename << " ...\n";
                load_data_from_silo(opts().restart_filename, root, root_client.get_unmanaged_gid());
                printf("Re-grid\n");
                ngrids = root->regrid(root_client.get_unmanaged_gid(), ZERO, -1, true, false);
                printf("Done. \n");

                set_AB(physcon().A, physcon().B);

            } else {
                for (integer l = 0; l < opts().max_level; ++l) {
                    ngrids =
                        root->regrid(root_client.get_gid(), grid::get_omega(), -1, false, false);
                    printf("---------------Created Level %i---------------\n\n", int(l + 1));
                }
                ngrids = root->regrid(root_client.get_gid(), grid::get_omega(), -1, false, false);
                printf(
                    "---------------Re-gridded Level %i---------------\n\n", int(opts().max_level));
            }
            for (integer l = 0; l < opts().extra_regrid; ++l) {
                ngrids = root->regrid(root_client.get_gid(), grid::get_omega(), -1, false, false);
            }

            if (opts().gravity && opts().stop_step != 0) {
                printf("solving gravity------------\n");
                root->solve_gravity(false, false);
                printf("...done\n");
            }
            if (opts().problem != AMR_TEST) {
                hpx::async(&node_server::execute_solver, root,
                    opts().problem == DWD && opts().restart_filename.empty(), ngrids)
                    .get();
            } else {
                root->enforce_bc();
                auto e = root->amr_error();
                printf("AMR Error: %e %e %e\n", e.first, e.second, e.first / e.second);
                output_all(root, "X", 0, true);
            }
            root->report_timing();
            cleanup(); // cleanup buffer and executor pools
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
