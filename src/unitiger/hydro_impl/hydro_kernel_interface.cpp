#include "octotiger/unitiger/hydro_impl/hydro_kernel_interface.hpp"
#include "octotiger/grid.hpp"
#include "octotiger/unitiger/hydro_impl/hydro_cuda_interface.hpp"

#ifdef OCTOTIGER_HAVE_KOKKOS
#include <hpx/kokkos.hpp>
#include "octotiger/common_kernel/kokkos_util.hpp"
#endif

#if defined(OCTOTIGER_HAVE_KOKKOS)
#if defined(KOKKOS_ENABLE_CUDA)
using device_executor = hpx::kokkos::cuda_executor;
using device_pool_strategy = round_robin_pool<device_executor>;
using executor_interface_t = stream_interface<device_executor, device_pool_strategy>;
#endif
//#ifdef OCTOTIGER_MONOPOLE_HOST_HPX_EXECUTOR
//using host_executor = hpx::kokkos::hpx_executor;
//#else
using host_executor = hpx::kokkos::serial_executor;
//#endif
#endif

timestep_t launch_hydro_kernels(hydro_computer<NDIM, INX, physics<NDIM>>& hydro,
    const std::vector<std::vector<safe_real>>& U, std::vector<std::vector<safe_real>>& X,
    const double omega, std::vector<hydro_state_t<std::vector<safe_real>>>& F,
    const interaction_host_kernel_type host_type, const interaction_device_kernel_type device_type,
    const size_t cuda_buffer_capacity) {
    static const cell_geometry<NDIM, INX> geo;

    // interaction_host_kernel_type host_type = opts().hydro_host_kernel_type;
    // interaction_device_kernel_type device_type = opts().hydro_device_kernel_type;

    // Timestep default value
    auto max_lambda = timestep_t{};
    bool avail = false;

    // Try accelerator implementation
    // TODO Pick correct implementation based on kernel parameter
    if (device_type != interaction_device_kernel_type::OFF) {
        if (device_type == interaction_device_kernel_type::KOKKOS_CUDA) {
#if defined(OCTOTIGER_HAVE_KOKKOS) && defined(KOKKOS_ENABLE_CUDA)
            avail = stream_pool::interface_available<device_executor, device_pool_strategy>(
                cuda_buffer_capacity);
            if (avail) {
                executor_interface_t executor;
                // TODO Device Kokkos Implementation (stub)
                // monopole_kernel<device_executor>(executor, monopoles, com_ptr, neighbors, type,
                // dx,
                //    opts().theta, is_direction_empty, grid_ptr, contains_multipole_neighbor);
            }
#else
            std::cerr << "Trying to call Hydro Kokkos device kernel in a non-kokkos build! Aborting..."
                      << std::endl;
            abort();
#endif
        }
        if (device_type == interaction_device_kernel_type::CUDA) {
#ifdef OCTOTIGER_HAVE_CUDA
            // TODO Device CUDA Implementation
            avail = stream_pool::interface_available<hpx::cuda::experimental::cuda_executor,
                pool_strategy>(cuda_buffer_capacity);
            size_t device_id = 0;
            if (avail) {
                stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy> executor;
                max_lambda = launch_hydro_cuda_kernels(hydro, U, X, omega, device_id, executor, F);
                return max_lambda;
            }
        }
#else
            std::cerr << "Trying to call Hydro CUDA device kernel in a non-CUDA build! "
                      << "Aborting..." << std::endl;
            abort();
        }
#endif
    }

    // Nothing is available or device execution is disabled - fallback to host execution
    if (!avail) {
        if (host_type == interaction_host_kernel_type::KOKKOS) {
            // TODO Kokkos Implementation (stub)
        } else if (host_type == interaction_host_kernel_type::VC) {
            // Vc implementation
            static thread_local auto f = std::vector<std::vector<std::vector<safe_real>>>(NDIM,
                std::vector<std::vector<safe_real>>(opts().n_fields, std::vector<safe_real>(H_N3)));
            // TODO Vc reconstruct?
            const auto& q = hydro.reconstruct(U, X, omega);
#if defined __x86_64__ && defined OCTOTIGER_HAVE_VC
            max_lambda = flux_cpu_kernel(q, f, X, omega, hydro.get_nf());
#else
            max_lambda = hydro.flux(U, q, f, X, omega);
#endif
            // Slightly more efficient conversion
            for (int dim = 0; dim < NDIM; dim++) {
                for (integer field = 0; field != opts().n_fields; ++field) {
#pragma GCC ivdep
                    for (integer i = 0; i <= INX; ++i) {
                        for (integer j = 0; j <= INX; ++j) {
                            for (integer k = 0; k <= INX; ++k) {
                                const auto i0 = findex(i, j, k);
                                F[dim][field][i0] =
                                    f[dim][field][hindex(i + H_BW, j + H_BW, k + H_BW)];
                                if (field == opts().n_fields - 1) {
                                    real rho_tot = 0.0;
                                    for (integer field = spc_i; field != spc_i + opts().n_species;
                                         ++field) {
                                        rho_tot += F[dim][field][i0];
                                    }
                                    F[dim][rho_i][i0] = rho_tot;
                                }
                            }
                        }
                    }
                }
            }
            return max_lambda;
        } else if (host_type == interaction_host_kernel_type::LEGACY) {
            // Legacy implementation
            static thread_local auto f = std::vector<std::vector<std::vector<safe_real>>>(NDIM,
                std::vector<std::vector<safe_real>>(opts().n_fields, std::vector<safe_real>(H_N3)));
            const auto& q = hydro.reconstruct(U, X, omega);
            max_lambda = hydro.flux(U, q, f, X, omega);
            // Use legacy conversion
            for (int dim = 0; dim < NDIM; dim++) {
                for (integer field = 0; field != opts().n_fields; ++field) {
#pragma GCC ivdep
                    for (integer i = 0; i <= INX; ++i) {
                        for (integer j = 0; j <= INX; ++j) {
                            for (integer k = 0; k <= INX; ++k) {
                                const auto i0 = findex(i, j, k);
                                F[dim][field][i0] =
                                    f[dim][field][hindex(i + H_BW, j + H_BW, k + H_BW)];
                                real rho_tot = 0.0;
                                for (integer field = spc_i; field != spc_i + opts().n_species;
                                     ++field) {
                                    rho_tot += F[dim][field][i0];
                                }
                                F[dim][rho_i][i0] = rho_tot;
                            }
                        }
                    }
                }
            }
            return max_lambda;
        } else {
            std::cerr << "No valid hydro kernel type given! " << std::endl;
            std::cerr << "Aborting..." << std::endl;
            abort();
        }
    } 
}
