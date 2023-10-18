#include "frontend-helper.hpp"
#ifdef OCTOTIGER_HAVE_KOKKOS
#include "octotiger/common_kernel/kokkos_util.hpp"
#endif
#ifdef OCTOTIGER_HAVE_CUDA
#include "octotiger/cuda_util/cuda_scheduler.hpp"
#include "octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp"
#include "octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp"

#include <cuda_buffer_util.hpp>
#endif
#ifdef OCTOTIGER_HAVE_HIP
#include <hip_buffer_util.hpp>
#endif
#if defined(OCTOTIGER_HAVE_KOKKOS) && defined(KOKKOS_ENABLE_SYCL)
#include <sycl_buffer_util.hpp>
#endif
#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/monopole_interactions/util/calculate_stencil.hpp"
#include "octotiger/monopole_interactions/legacy/p2m_interaction_interface.hpp"
#include "octotiger/monopole_interactions/legacy/monopole_interaction_interface.hpp"
#include "octotiger/multipole_interactions/util/calculate_stencil.hpp"
#include "octotiger/multipole_interactions/legacy/multipole_interaction_interface.hpp"


#include "octotiger/compute_factor.hpp"
#include "octotiger/defs.hpp"
// #include "octotiger/future.hpp"
#include "octotiger/grid_fmm.hpp"
#include "octotiger/grid_scf.hpp"
#include "octotiger/node_client.hpp"
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

#include <iostream>
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
#include "octotiger/monopole_interactions/kernel/kokkos_kernel.hpp"
#include "octotiger/multipole_interactions/kernel/kokkos_kernel.hpp"
#include "octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp"
#endif
#ifdef OCTOTIGER_HAVE_CUDA
#include <stream_manager.hpp>
#include "octotiger/cuda_util/cuda_helper.hpp"
#endif

// In case we build without kokkos we want the cuda futures to default
// to the polling futures! Change to 1 to change to callback futures!
// In kokkos builds this variable comes from hpx-kokkos as it MUST have the same value
// otherwise it might deadlock as we don't poll
#ifndef HPX_KOKKOS_CUDA_FUTURE_TYPE
#define HPX_KOKKOS_CUDA_FUTURE_TYPE 0
#endif

void cleanup_puddle_on_this_locality(void) {
    // Shutdown stream manager
    if (opts().executors_per_gpu > 0) {
#if defined(OCTOTIGER_HAVE_CUDA) 
      stream_pool::cleanup<hpx::cuda::experimental::cuda_executor, pool_strategy>();
#elif defined(OCTOTIGER_HAVE_HIP)  // TODO verify 
      stream_pool::cleanup<hpx::cuda::experimental::cuda_executor, pool_strategy>();
#endif

#if defined(KOKKOS_ENABLE_CUDA)
      stream_pool::cleanup<hpx::kokkos::cuda_executor, round_robin_pool<hpx::kokkos::cuda_executor>>();
#elif defined(KOKKOS_ENABLE_HIP) 
      stream_pool::cleanup<hpx::kokkos::hip_executor, round_robin_pool<hpx::kokkos::hip_executor>>();
#elif defined(KOKKOS_ENABLE_SYCL) 
      stream_pool::cleanup<hpx::kokkos::sycl_executor, round_robin_pool<hpx::kokkos::sycl_executor>>();
#endif
    }
    // Disable polling
#if (defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)) && HPX_KOKKOS_CUDA_FUTURE_TYPE == 0 
    if (opts().polling_threads > 0) {
      std::cout << "Unregistering cuda polling on polling pool... " << std::endl;
      hpx::cuda::experimental::detail::unregister_polling(hpx::resource::get_thread_pool("polling"));
    } else {
        std::cout << "Unregistering cuda polling..." << std::endl;
        hpx::cuda::experimental::detail::unregister_polling(hpx::resource::get_thread_pool(0));
    }
#endif
#if defined(OCTOTIGER_HAVE_KOKKOS) && defined(KOKKOS_ENABLE_SYCL)
    if (opts().polling_threads > 0) {
      std::cout << "Unregistering sycl polling on polling pool... " << std::endl;
      hpx::sycl::experimental::detail::unregister_polling(hpx::resource::get_thread_pool("polling"));
    } else {
      std::cout << "Unregistering sycl polling..." << std::endl;
      hpx::sycl::experimental::detail::unregister_polling(hpx::resource::get_thread_pool(0));
    }
#endif
    // Use finalize functionality. Cleans up all buffers and prevents further use
    recycler::finalize();
#ifdef OCTOTIGER_HAVE_KOKKOS
    stream_pool::cleanup<hpx::kokkos::hpx_executor, round_robin_pool<hpx::kokkos::hpx_executor>>();
    stream_pool::cleanup<hpx::kokkos::serial_executor, round_robin_pool<hpx::kokkos::serial_executor>>();
    Kokkos::finalize();
#endif
    
}

void init_executors(void) {
    std::cout << "Check number of available GPUs..." << std::endl;
    int num_devices = 0; 
#if defined(OCTOTIGER_HAVE_CUDA) || defined(KOKKOS_ENABLE_CUDA) 
    cudaGetDeviceCount(&num_devices);
    std::cout << "Found " << num_devices << " CUDA devices! " << std::endl;
#elif defined(OCTOTIGER_HAVE_HIP) || defined(KOKKOS_ENABLE_HIP) 
    hipGetDeviceCount(&num_devices);
    std::cout << "Found " << num_devices << " HIP devices! " << std::endl;
#endif
    if (num_devices > 0) { // some devices were found
      if (opts().number_gpus > num_devices) {
          std::cerr << "ERROR: Requested " << opts().number_gpus << " GPUs but only "
                    << num_devices << " were found!" << std::endl;
          abort();
      }
      if (opts().number_gpus > recycler::max_number_gpus) {
        std::cerr << "ERROR: Requested " << opts().number_gpus
                  << " GPUs but CPPuddle was built with CPPUDDLE_WITH_MAX_NUMBER_GPUS="
                  << recycler::max_number_gpus << std::endl;
        abort();
      }
    }


    std::cout << "Initialize executors and masks..." << std::endl;
    // Init Kokkos
#ifdef OCTOTIGER_HAVE_KOKKOS
    Kokkos::initialize();
		if (hpx::get_locality_id() == 0)
      Kokkos::print_configuration(std::cout, true);
#ifdef OCTOTIGER_MULTIPOLE_HOST_HPX_EXECUTOR
    std::cout << "Using Kokkos HPX executors for multipole FMM kernels..." << std::endl;
    std::cout << "Number of tasks per KOKKOS multipole kernel: " << OCTOTIGER_KOKKOS_MULTIPOLE_TASKS << std::endl;
#else
    std::cout << "Using Kokkos serial executors for multipole FMM kernels..." << std::endl;
#endif
#ifdef OCTOTIGER_MONOPOLE_HOST_HPX_EXECUTOR
    std::cout << "Using Kokkos HPX executors for monopole FMM kernels..." << std::endl;
    std::cout << "Number of tasks per KOKKOS monopole kernel: " << OCTOTIGER_KOKKOS_MONOPOLE_TASKS << std::endl;
#else
    std::cout << "Using Kokkos serial executors for monopole FMM kernels..." << std::endl;
#endif
#ifdef OCTOTIGER_WITH_HYDRO_HOST_HPX_EXECUTOR
    std::cout << "Using Kokkos HPX executors for hydro kernels..." << std::endl;
    std::cout << "Number of tasks per KOKKOS hydro kernel: " << OCTOTIGER_KOKKOS_HYDRO_TASKS << std::endl;
#else
    std::cout << "Using Kokkos serial executors for hydro kernels..." << std::endl;
#endif
    // initialize stencils in kokkos host memory
    octotiger::fmm::monopole_interactions::get_host_masks<host_buffer<int>>();
    octotiger::fmm::monopole_interactions::get_host_constants<host_buffer<double>>();
    octotiger::fmm::multipole_interactions::get_host_masks<host_buffer<int>>(true);

    get_flux_host_masks<host_buffer<bool>>();
#endif

#if HPX_KOKKOS_CUDA_FUTURE_TYPE == 0
#if (defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP) || defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP))  
    if (opts().polling_threads>0) {
      std::cout << "Registering HPX CUDA polling on polling pool..." << std::endl;
      hpx::cuda::experimental::detail::register_polling(hpx::resource::get_thread_pool("polling"));
    } else {
      std::cout << "Registering HPX CUDA polling..." << std::endl;
      hpx::cuda::experimental::detail::register_polling(hpx::resource::get_thread_pool(0));
    }
    std::cout << "Registered HPX CUDA polling!" << std::endl;
#endif
#endif
#if defined(OCTOTIGER_HAVE_KOKKOS) && defined(KOKKOS_ENABLE_SYCL)
    if (opts().polling_threads>0) {
      std::cout << "Registering HPX SYCL polling on polling pool..." << std::endl;
      hpx::sycl::experimental::detail::register_polling(hpx::resource::get_thread_pool("polling"));
    } else {
      std::cout << "Registering HPX SYCL polling..." << std::endl;
      hpx::sycl::experimental::detail::register_polling(hpx::resource::get_thread_pool(0));
    }
    std::cout << "Registered HPX SYCL polling!" << std::endl;
#if HPX_KOKKOS_SYCL_FUTURE_TYPE == 0 
    std::cout << "Using HPX SYCL futures with polling!" << std::endl;
#endif
#if HPX_KOKKOS_SYCL_FUTURE_TYPE == 1 
    std::cout << "Using HPX SYCL futures with host_tasks!" << std::endl;
#endif
#endif

#if defined(OCTOTIGER_HAVE_KOKKOS)
    std::cout << "Initializing Kokkos host executors..." << std::endl;
    stream_pool::init_all_executor_pools<hpx::kokkos::serial_executor, round_robin_pool<hpx::kokkos::serial_executor>>(
        256, hpx::kokkos::execution_space_mode::independent);
    stream_pool::init_all_executor_pools<hpx::kokkos::hpx_executor, round_robin_pool<hpx::kokkos::hpx_executor>>(
        256, hpx::kokkos::execution_space_mode::independent);
    std::cout << "Initializing Kokkos device executors..." << std::endl;
    std::cout << "CPPuddle config: Max number GPUs: " << recycler::max_number_gpus << " devices!" << std::endl;
    std::cout << "CPPuddle config: Using " << recycler::number_instances << " internal buffer buckets!"
              << std::endl;
#if defined(KOKKOS_ENABLE_CUDA)
    stream_pool::set_device_selector<hpx::kokkos::cuda_executor,
          round_robin_pool<hpx::kokkos::cuda_executor>>([](size_t gpu_id) {
              cudaSetDevice(gpu_id);
              });
    // initialize stencils / executor pool in kokkos device
    for (size_t gpu_id = 0; gpu_id < opts().number_gpus; gpu_id++) {
      stream_pool::init_executor_pool<hpx::kokkos::cuda_executor,
          round_robin_pool<hpx::kokkos::cuda_executor>>(
          gpu_id, opts().executors_per_gpu,
          hpx::kokkos::execution_space_mode::independent);
    }
    std::cout << "KOKKOS/CUDA is enabled!" << std::endl;
#elif defined(KOKKOS_ENABLE_HIP)

    stream_pool::set_device_selector<hpx::kokkos::hip_executor,
          round_robin_pool<hpx::kokkos::hip_executor>>([](size_t gpu_id) {
              hipSetDevice(gpu_id);
              });
    for (size_t gpu_id = 0; gpu_id < opts().number_gpus; gpu_id++) {
      stream_pool::init_executor_pool<hpx::kokkos::hip_executor,
          round_robin_pool<hpx::kokkos::hip_executor>>(
          gpu_id, opts().executors_per_gpu,
          hpx::kokkos::execution_space_mode::independent);
    }
    std::cout << "KOKKOS/HIP is enabled!" << std::endl;
#elif defined(KOKKOS_ENABLE_SYCL)
    for (size_t gpu_id = 0; gpu_id < opts().number_gpus; gpu_id++) {
      stream_pool::init_executor_pool<hpx::kokkos::sycl_executor,
          round_robin_pool<hpx::kokkos::sycl_executor>>(
          gpu_id, opts().executors_per_gpu,
          hpx::kokkos::execution_space_mode::independent);
    }
    std::cout << "KOKKOS/SYCL is enabled!" << std::endl;
#endif
#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    kokkos_device_executor mover{};
    octotiger::fmm::monopole_interactions::get_device_masks<device_buffer<int>, host_buffer<int>,
        kokkos_device_executor>(mover);
    octotiger::fmm::monopole_interactions::get_device_constants<device_buffer<double>,
        host_buffer<double>, kokkos_device_executor>(mover);
    octotiger::fmm::multipole_interactions::get_device_masks<device_buffer<int>, host_buffer<int>,
        kokkos_device_executor>(mover, true);
    get_flux_device_masks<device_buffer<bool>, host_buffer<bool>,
        kokkos_device_executor>(mover);
    Kokkos::fence();
#if HPX_KOKKOS_CUDA_FUTURE_TYPE == 0 
    std::cout << "KOKKOS with polling futures enabled!" << std::endl;
#else
    std::cout << "KOKKOS with callback futures enabled!" << std::endl;
#endif
#endif
#endif

#if defined(OCTOTIGER_HAVE_CUDA)
    std::cout << "CUDA is enabled!" << std::endl;
    stream_pool::set_device_selector<hpx::cuda::experimental::cuda_executor,
          round_robin_pool<hpx::cuda::experimental::cuda_executor>>([](size_t gpu_id) {
              cudaSetDevice(gpu_id);
              });
#if HPX_KOKKOS_CUDA_FUTURE_TYPE == 0 
    std::cout << "CUDA with polling futures enabled!" << std::endl;
    for (size_t gpu_id = 0; gpu_id < opts().number_gpus; gpu_id++) {
      stream_pool::init_executor_pool<hpx::cuda::experimental::cuda_executor, pool_strategy>(gpu_id,
          opts().executors_per_gpu, gpu_id, true);
    }
#else
    std::cout << "CUDA with callback futures enabled!" << std::endl;
    for (size_t gpu_id = 0; gpu_id < opts().number_gpus; gpu_id++) {
      stream_pool::init_executor_pool<hpx::cuda::experimental::cuda_executor, pool_strategy>(gpu_id,
          opts().executors_per_gpu, gpu_id, false);
    }
#endif
    octotiger::fmm::init_fmm_constants();

#endif

#if defined(OCTOTIGER_HAVE_HIP)

    std::cout << "HIP is enabled!" << std::endl;
    stream_pool::set_device_selector<hpx::cuda::experimental::cuda_executor,
          round_robin_pool<hpx::cuda::experimental::cuda_executor>>([](size_t gpu_id) {
              hipSetDevice(gpu_id);
              });
#if HPX_KOKKOS_CUDA_FUTURE_TYPE == 0  // cuda in the name is correct
    std::cout << "HIP with polling futures enabled!" << std::endl;
    for (size_t gpu_id = 0; gpu_id < opts().number_gpus; gpu_id++) {
      stream_pool::init_executor_pool<hpx::cuda::experimental::cuda_executor, pool_strategy>(gpu_id,
          opts().executors_per_gpu, gpu_id, true);
    }
    std::cout << "HIP with polling futures created!" << std::endl;
#else
    std::cout << "HIP with callback futures enabled!" << std::endl;
    for (size_t gpu_id = 0; gpu_id < opts().number_gpus; gpu_id++) {
      stream_pool::init_executor_pool<hpx::cuda::experimental::cuda_executor, pool_strategy>(gpu_id,
          opts().executors_per_gpu, gpu_id, false);
    }
    std::cout << "HIP with callback futures created!" << std::endl;
#endif
    octotiger::fmm::init_fmm_constants();
#endif
    std::cout << "Stencils initialized!" << std::endl;
}

void init_problem(void) {
    std::cout << "Initialize problem..." << std::endl;
    physics<NDIM>::set_n_species(opts().n_species);
    physics<NDIM>::update_n_field();
    physics<NDIM>::set_mu(opts().atomic_mass, opts().atomic_number);
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
        grid::set_fgamma(7.0 / 5.0);
        //		opts().gravity = false;
        set_problem(blast_wave);
        set_refine_test(refine_blast);
#if defined(OCTOTIGER_HAVE_BLAST_TEST)
        set_analytic(blast_wave_analytic);
#else
        std::cout << "Warning! Octotiger has been compiled without BLAST test support!" << std::endl;
        std::cout << "Blast scenario will skip the analytic part..." << std::endl;
        //exit(EXIT_FAILURE);
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
        std::cerr << "Error: No problem specified\n";
        std::terminate();
    }

    if (OCTOTIGER_MAX_NUMBER_FIELDS > physics<NDIM>::nf_) {
        std::cerr << "\nWarning! OCTOTIGER_WITH_MAX_NUMBER_FIELDS too large for this scenario!" << std::endl
                  << "This will lead to slightly reduced performance in the flux kernel!" << std::endl 
                  << "Choose -DOCTOTIGER_WITH_MAX_NUMBER_FIELDS=" << physics<NDIM>::nf_ 
                  << " to run this scenario with optimal performance" << std::endl << std::endl;
    } else if (OCTOTIGER_MAX_NUMBER_FIELDS < physics<NDIM>::nf_) {
        std::cerr << "\nERROR! OCTOTIGER_WITH_MAX_NUMBER_FIELDS too small for this scenario!" << std::endl
                  << "Recompile with -DOCTOTIGER_WITH_MAX_NUMBER_FIELDS=" <<  physics<NDIM>::nf_ 
                  <<std::endl << std::endl;
        exit(EXIT_FAILURE);
    } else {
      std::cout << "Compiled with max nf -DOCTOTIGER_WITH_MAX_NUMBER_FIELDS=" << OCTOTIGER_MAX_NUMBER_FIELDS << std::endl;
    }
    std::cout << "Problem initialized!" << std::endl;
}


void register_cppuddle_allocator_counters(void)  {
#ifdef CPPUDDLE_HAVE_COUNTERS
    // default host allocators
    hpx::register_startup_function(
        &recycler::detail::buffer_recycler::register_allocator_counters_with_hpx<
            double, boost::alignment::aligned_allocator<double, 32>>);
    hpx::register_startup_function(
        &recycler::detail::buffer_recycler::register_allocator_counters_with_hpx<
            int, boost::alignment::aligned_allocator<int, 32>>);
    hpx::register_startup_function(
        &recycler::detail::buffer_recycler::register_allocator_counters_with_hpx<
            double, std::allocator<double>>);
    hpx::register_startup_function(
        &recycler::detail::buffer_recycler::register_allocator_counters_with_hpx<
            int, std::allocator<int>>);

    // CUDA host / device allocators -- also used by KOKKOS
#if defined(OCTOTIGER_HAVE_CUDA)
    hpx::register_startup_function(
        &recycler::detail::buffer_recycler::register_allocator_counters_with_hpx<
            double, recycler::detail::cuda_pinned_allocator<double>>);
    hpx::register_startup_function(
        &recycler::detail::buffer_recycler::register_allocator_counters_with_hpx<
            int, recycler::detail::cuda_pinned_allocator<int>>);
    hpx::register_startup_function(
        &recycler::detail::buffer_recycler::register_allocator_counters_with_hpx<
            double, recycler::detail::cuda_device_allocator<double>>);
    hpx::register_startup_function(
        &recycler::detail::buffer_recycler::register_allocator_counters_with_hpx<
            int, recycler::detail::cuda_device_allocator<int>>);
#endif
    // HIP host / device allocators -- also used by KOKKOS
#if defined(OCTOTIGER_HAVE_HIP)
    hpx::register_startup_function(
        &recycler::detail::buffer_recycler::register_allocator_counters_with_hpx<
            double, recycler::detail::hip_pinned_allocator<double>>);
    hpx::register_startup_function(
        &recycler::detail::buffer_recycler::register_allocator_counters_with_hpx<
            int, recycler::detail::hip_pinned_allocator<int>>);
    hpx::register_startup_function(
        &recycler::detail::buffer_recycler::register_allocator_counters_with_hpx<
            double, recycler::detail::hip_device_allocator<double>>);
    hpx::register_startup_function(
        &recycler::detail::buffer_recycler::register_allocator_counters_with_hpx<
            int, recycler::detail::hip_device_allocator<int>>);
#endif
    // SYCL host / device allocators 
#if defined(OCTOTIGER_HAVE_KOKKOS) && defined(KOKKOS_ENABLE_SYCL)
    hpx::register_startup_function(
        &recycler::detail::buffer_recycler::register_allocator_counters_with_hpx<
            double, detail::sycl_host_default_allocator<double>>);
    hpx::register_startup_function(
        &recycler::detail::buffer_recycler::register_allocator_counters_with_hpx<
            int, detail::sycl_host_default_allocator<int>>);
    hpx::register_startup_function(
        &recycler::detail::buffer_recycler::register_allocator_counters_with_hpx<
            double, detail::sycl_device_default_allocator<double>>);
    hpx::register_startup_function(
        &recycler::detail::buffer_recycler::register_allocator_counters_with_hpx<
            int, detail::sycl_device_default_allocator<int>>);
#endif

#endif
}
