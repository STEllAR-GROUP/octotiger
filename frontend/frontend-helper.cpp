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
#include "octotiger/monopole_interactions/legacy/cuda_p2p_interaction_interface.hpp"
#include "octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp"

#endif
#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/monopole_interactions/util/calculate_stencil.hpp"
#include "octotiger/monopole_interactions/legacy/p2m_interaction_interface.hpp"
#include "octotiger/monopole_interactions/legacy/p2p_interaction_interface.hpp"
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

std::size_t init_thread_local_worker(std::size_t desired) {
    std::size_t current = hpx::get_worker_thread_num();
    if (current == desired) {
        init_stencil(current);
        //std::cout << "OS-thread " << current << " on locality " << hpx::get_locality_id()
        //          << ": Initialized thread_local memory!\n";
        return desired;
    }
    // NOTE: This might be an issue. Throw an exception and/or make the output
    // a tuple with the second being the error code
    return std::size_t(-1);
}
HPX_PLAIN_ACTION(init_thread_local_worker, init_thread_local_worker_action);

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
        using cuda_mono_intfc = octotiger::fmm::monopole_interactions::p2p_interaction_interface;

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

    scf_options::read_option_file();

    options::all_localities = localities;
    opts() = _opts;

    init_problem();

    compute_ilist();
    compute_factor();

    init_executors();

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
    while (!attendance.empty()) {
        std::vector<hpx::lcos::future<std::size_t>> futures;
        futures.reserve(attendance.size());

        for (std::size_t worker : attendance) {
            using action_type = init_thread_local_worker_action;
            futures.push_back(hpx::async<action_type>(here, worker));
        }
        hpx::lcos::local::spinlock mtx;
        hpx::lcos::wait_each(hpx::util::unwrapping([&](std::size_t t) {
            if (std::size_t(-1) != t) {
                std::lock_guard<hpx::lcos::local::spinlock> lk(mtx);
                attendance.erase(t);
            }
        }),
            futures);
    }
}

std::array<size_t, 6> analyze_local_launch_counters() {
    std::size_t const os_threads = hpx::get_os_thread_count();
    hpx::naming::id_type const here = hpx::find_here();
    std::set<std::size_t> attendance;
    for (std::size_t os_thread = 0; os_thread < os_threads; ++os_thread)
        attendance.insert(os_thread);

    std::array<size_t, 6> results{0, 0, 0, 0, 0, 0};
    while (!attendance.empty()) {
        std::vector<hpx::lcos::future<std::array<size_t, 7>>> futures;
        futures.reserve(attendance.size());

        for (std::size_t worker : attendance) {
            using action_type = sum_counters_worker_action;
            futures.push_back(hpx::async<action_type>(here, worker));
        }
        hpx::lcos::local::spinlock mtx;
        hpx::lcos::wait_each(hpx::util::unwrapping([&](std::array<size_t, 7> t) {
            if (std::size_t(-1) != t[0]) {
                std::lock_guard<hpx::lcos::local::spinlock> lk(mtx);
                results[0] += t[1];
                results[1] += t[2];
                results[2] += t[3];
                results[3] += t[4];
                results[4] += t[5];
                results[5] += t[6];
                attendance.erase(t[0]);
            }
        }),
            futures);
    }
    cleanup_puddle_on_this_locality();
    return results;
}
HPX_PLAIN_ACTION(analyze_local_launch_counters, analyze_local_launch_counters_action);

void accumulate_distributed_counters() {
    std::vector<hpx::naming::id_type> localities = hpx::find_all_localities();

    std::vector<hpx::lcos::future<std::array<size_t, 6>>> futures;
    futures.reserve(localities.size());

    for (hpx::naming::id_type const& node : localities) {
        using action_type = analyze_local_launch_counters_action;
        futures.push_back(hpx::async<action_type>(node));
    }

    std::array<size_t, 6> results{0, 0, 0, 0, 0, 0};
    hpx::lcos::local::spinlock mtx;
    hpx::lcos::wait_each(hpx::util::unwrapping([&](std::array<size_t, 6> t) {
        std::lock_guard<hpx::lcos::local::spinlock> lk(mtx);
        results[0] += t[0];
        results[1] += t[1];
        results[2] += t[2];
        results[3] += t[3];
        results[4] += t[4];
        results[5] += t[5];
    }),
        futures);
    size_t total_multipole_cpu_launches = results[0];
    size_t total_multipole_cuda_launches = results[1];
    size_t total_p2p_cpu_launches = results[2];
    size_t total_p2p_cuda_launches = results[3];
    size_t total_multipole_cpu_launches_non_rho = results[4];
    size_t total_multipole_cuda_launches_non_rho = results[5];
    std::cout << "========================================" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total multipole launches: "
              << total_multipole_cpu_launches + total_multipole_cuda_launches << std::endl;
    std::cout << "CPU multipole launches: " << total_multipole_cpu_launches << std::endl;
    std::cout << "CUDA multipole launches: " << total_multipole_cuda_launches << std::endl;
    if (total_multipole_cpu_launches + total_multipole_cuda_launches > 0) {
        float percentage = static_cast<float>(total_multipole_cuda_launches) /
            (static_cast<float>(total_multipole_cuda_launches) + total_multipole_cpu_launches);
        std::cout << "=> Percentage of multipole on the GPU: " << percentage * 100 << "\n";
    }
    std::cout << "========================================" << std::endl;
    std::cout << "Total non-rho-multipole launches: "
              << total_multipole_cpu_launches_non_rho + total_multipole_cuda_launches_non_rho
              << std::endl;
    std::cout << "CPU non-rho-multipole launches: " << total_multipole_cpu_launches_non_rho
              << std::endl;
    std::cout << "CUDA non-rho-multipole launches: " << total_multipole_cuda_launches_non_rho
              << std::endl;
    if (total_multipole_cpu_launches_non_rho + total_multipole_cuda_launches_non_rho > 0) {
        float percentage = static_cast<float>(total_multipole_cuda_launches_non_rho) /
            (static_cast<float>(total_multipole_cuda_launches_non_rho) +
                total_multipole_cpu_launches_non_rho);
        std::cout << "=> Percentage of non-rho-multipole on the GPU: " << percentage * 100 << "\n";
    }
    std::cout << "========================================" << std::endl;
    std::cout << "Total p2p launches: " << total_p2p_cpu_launches + total_p2p_cuda_launches
              << std::endl;
    std::cout << "CPU p2p launches: " << total_p2p_cpu_launches << std::endl;
    std::cout << "CUDA p2p launches: " << total_p2p_cuda_launches << std::endl;
    if (total_p2p_cpu_launches + total_p2p_cuda_launches > 0) {
        float percentage = static_cast<float>(total_p2p_cuda_launches) /
            (static_cast<float>(total_p2p_cuda_launches) + total_p2p_cpu_launches);
        std::cout << "=> Percentage of p2p on the GPU: " << percentage * 100 << "\n";
    }
}
HPX_PLAIN_ACTION(initialize, initialize_action);
HPX_REGISTER_BROADCAST_ACTION_DECLARATION(initialize_action);
HPX_REGISTER_BROADCAST_ACTION(initialize_action);

void start_octotiger(int argc, char* argv[]) {
    try {
        if (opts().process_options(argc, argv)) {
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
            accumulate_distributed_counters();
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
