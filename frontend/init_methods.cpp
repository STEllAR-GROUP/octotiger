#include "frontend-helper.hpp"
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

#ifdef OCTOTIGER_HAVE_KOKKOS
#include "octotiger/common_kernel/kokkos_util.hpp"
#include "octotiger/monopole_interactions/kernel/kokkos_kernel.hpp"
#include "octotiger/multipole_interactions/kernel/kokkos_kernel.hpp"
#endif

void cleanup_puddle_on_this_locality(void) {
    // Cleaning up of cuda buffers before the runtime gets shutdown
    recycler::force_cleanup();
    // Shutdown stream manager
    if (opts().cuda_streams_per_gpu > 0) {
#ifdef OCTOTIGER_HAVE_CUDA
      stream_pool::cleanup<hpx::cuda::experimental::cuda_executor, pool_strategy>();

#ifdef OCTOTIGER_HAVE_KOKKOS
      stream_pool::cleanup<hpx::kokkos::cuda_executor, round_robin_pool<hpx::kokkos::cuda_executor>>(
#endif
#endif
          );
    }
    // Disable polling
    if (opts().cuda_polling_executor) {
        std::cout << "Unregistering cuda polling..." << std::endl;
        hpx::cuda::experimental::detail::unregister_polling(hpx::resource::get_thread_pool(0));
    }
}

void init_stencil(std::size_t worker_id) {
    namespace mono_inter = octotiger::fmm::monopole_interactions;
    using mono_inter_p2p = octotiger::fmm::monopole_interactions::p2p_interaction_interface;
    // Initialize stencil and four constants for p2p fmm interactions
    mono_inter_p2p::stencil() = mono_inter::calculate_stencil().first;
    mono_inter_p2p::stencil_masks() =
        mono_inter::calculate_stencil_masks(mono_inter_p2p::stencil()).first;
    mono_inter_p2p::four() = mono_inter::calculate_stencil().second;
    mono_inter_p2p::stencil_four_constants() =
        mono_inter::calculate_stencil_masks(mono_inter_p2p::stencil()).second;

    // Initialize stencil for p2m fmm interactions
    mono_inter::p2m_interaction_interface::stencil() = mono_inter::calculate_stencil().first;

    namespace multi_inter = octotiger::fmm::multipole_interactions;
    using multi_inter_p2p = octotiger::fmm::multipole_interactions::multipole_interaction_interface;
    // Initialize stencil for multipole fmm interactions
    multi_inter_p2p::stencil() = multi_inter::calculate_stencil();
    multi_inter_p2p::stencil_masks() =
        multi_inter::calculate_stencil_masks(multi_inter_p2p::stencil()).first;
    multi_inter_p2p::inner_stencil_masks() =
        multi_inter::calculate_stencil_masks(multi_inter_p2p::stencil()).second;
    // print run informations
    if (worker_id == 0) {
        std::cout << "\nSubgrid side-length is " << INX << std::endl;
        std::cout << "Minimal allowed theta is " << octotiger::fmm::THETA_FLOOR << std::endl;
        std::cout << "Stencil maximal allowed half side-length is " << octotiger::fmm::STENCIL_WIDTH
                  << " (Total length " << 2 * octotiger::fmm::STENCIL_WIDTH + 1 << ")" << std::endl;
        std::cout << "Total number of stencil elements (stencil size): "
                  << mono_inter::calculate_stencil().first.size() << std::endl
                  << std::endl;
    }
    static_assert(octotiger::fmm::STENCIL_WIDTH <= INX, R"(
            ERROR: Stencil is too wide for the subgrid size. 
            Please increase either OCTOTIGER_THETA_MINIMUM or OCTOTIGER_WITH_GRIDDIM (see cmake file))");
}

void init_executors(void) {
#ifdef OCTOTIGER_HAVE_CUDA
    if (opts().cuda_polling_executor) {
        std::cout << "Registering cuda polling..." << std::endl;
        hpx::cuda::experimental::detail::register_polling(hpx::resource::get_thread_pool(0));
    }
#ifdef OCTOTIGER_HAVE_KOKKOS
    std::cout << "KOKKOS/CUDA is enabled!" << std::endl;
    stream_pool::init<hpx::kokkos::cuda_executor, round_robin_pool<hpx::kokkos::cuda_executor>>(
        opts().cuda_streams_per_gpu, hpx::kokkos::execution_space_mode::independent);
    hpx::kokkos::cuda_executor mover{};
    get_device_masks<device_buffer<int>, host_buffer<int>,
        hpx::kokkos::cuda_executor>(mover);
    get_device_masks<device_buffer<int>, host_buffer<int>,
        hpx::kokkos::cuda_executor>(mover, true);
    Kokkos::fence();
#endif
    std::cout << "CUDA is enabled!" << std::endl;
    stream_pool::init<hpx::cuda::experimental::cuda_executor, pool_strategy>(
        opts().cuda_streams_per_gpu, opts().cuda_number_gpus, opts().cuda_polling_executor);
    octotiger::fmm::kernel_scheduler::init_constants();
#endif
}

void init_problem(void) {
    physics<NDIM>::set_n_species(opts().n_species);
    physics<NDIM>::update_n_field();
    grid::get_omega() = opts().omega;
#if !defined(_MSC_VER) && !defined(__APPLE__)
    //feenableexcept(FE_DIVBYZERO);
    //feenableexcept(FE_INVALID);
    //feenableexcept(FE_OVERFLOW);
#else
    _controlfp(_EM_INEXACT | _EM_DENORMAL | _EM_INVALID, _MCW_EM);
#endif
    grid::set_scaling_factor(opts().xscale);
    grid::set_min_level(opts().min_level);
    grid::set_max_level(opts().max_level);
    if (opts().problem == RADIATION_TEST) {
        assert(opts().radiation);
        //		opts().gravity = false;
        set_problem(radiation_test_problem);
        set_refine_test(radiation_test_refine);
    } else if (opts().problem == DWD) {
        opts().n_species = 5;
        set_problem(scf_binary);
        set_refine_test(refine_test);
    } else if (opts().problem == SOD) {
        grid::set_fgamma(opts().sod_gamma);
        // grid::set_fgamma(7.0 / 5.0);
        //		opts().gravity = false;
        set_problem(sod_shock_tube_init);
        set_refine_test(refine_sod);
        set_analytic(sod_shock_tube_analytic);
    } else if (opts().problem == BLAST) {
#if defined(OCTOTIGER_HAVE_BLAST_TEST)
        grid::set_fgamma(7.0 / 5.0);
        //		opts().gravity = false;
        set_problem(blast_wave);
        set_refine_test(refine_blast);
        set_analytic(blast_wave_analytic);
#else
        std::cout << "Error! Octotiger has been compiled without BLAST test support!" << std::endl;
        exit(EXIT_FAILURE);
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
    } else if (opts().problem == SOLID_SPHERE) {
        //	opts().hydro = false;
        set_analytic(
            [](real x, real y, real z, real dx) { return solid_sphere(x, y, z, dx, 0.25); });
        set_refine_test(refine_test_center);
        set_problem(init_func_type(
            [](real x, real y, real z, real dx) { return solid_sphere(x, y, z, dx, 0.25); }));
    } else {
        printf("No problem specified\n");
        throw;
    }
}
