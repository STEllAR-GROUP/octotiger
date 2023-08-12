#include "octotiger/common_kernel/gravity_performance_counters.hpp"
#include <hpx/include/performance_counters.hpp>

namespace octotiger {
namespace fmm {

std::atomic<std::uint64_t> p2p_kokkos_cpu_subgrids_launched(0);
std::atomic<std::uint64_t> p2p_kokkos_gpu_subgrids_launched(0);
std::atomic<std::uint64_t> p2p_cuda_gpu_subgrids_launched(0);
std::atomic<std::uint64_t> p2p_vc_cpu_subgrids_launched(0);
std::atomic<std::uint64_t> p2p_legacy_cpu_subgrids_launched(0);

std::atomic<std::uint64_t> multipole_kokkos_cpu_subgrids_launched(0);
std::atomic<std::uint64_t> multipole_kokkos_gpu_subgrids_launched(0);
std::atomic<std::uint64_t> multipole_cuda_gpu_subgrids_launched(0);
std::atomic<std::uint64_t> multipole_vc_cpu_subgrids_launched(0);
std::atomic<std::uint64_t> multipole_legacy_cpu_subgrids_launched(0);

std::uint64_t p2p_kokkos_cpu_subgrid_processed_performance_data(bool reset) {
    if (reset)
        p2p_kokkos_cpu_subgrids_launched = 0;
    return p2p_kokkos_cpu_subgrids_launched;
}
std::uint64_t p2p_kokkos_gpu_subgrid_processed_performance_data(bool reset) {
    if (reset)
        p2p_kokkos_gpu_subgrids_launched = 0;
    return p2p_kokkos_gpu_subgrids_launched;
}
std::uint64_t p2p_cuda_gpu_subgrid_processed_performance_data(bool reset) {
    if (reset)
        p2p_cuda_gpu_subgrids_launched = 0;
    return p2p_cuda_gpu_subgrids_launched;
}
std::uint64_t p2p_vc_cpu_subgrid_processed_performance_data(bool reset) {
    if (reset)
        p2p_vc_cpu_subgrids_launched = 0;
    return p2p_vc_cpu_subgrids_launched;
}
std::uint64_t p2p_legacy_cpu_subgrid_processed_performance_data(bool reset) {
    if (reset)
        p2p_legacy_cpu_subgrids_launched = 0;
    return p2p_legacy_cpu_subgrids_launched;
}

std::uint64_t multipole_kokkos_cpu_subgrid_processed_performance_data(bool reset) {
    if (reset)
        multipole_kokkos_cpu_subgrids_launched = 0;
    return multipole_kokkos_cpu_subgrids_launched;
}
std::uint64_t multipole_kokkos_gpu_subgrid_processed_performance_data(bool reset) {
    if (reset)
        multipole_kokkos_gpu_subgrids_launched = 0;
    return multipole_kokkos_gpu_subgrids_launched;
}
std::uint64_t multipole_cuda_gpu_subgrid_processed_performance_data(bool reset) {
    if (reset)
        multipole_cuda_gpu_subgrids_launched = 0;
    return multipole_cuda_gpu_subgrids_launched;
}
std::uint64_t multipole_vc_cpu_subgrid_processed_performance_data(bool reset) {
    if (reset)
        multipole_vc_cpu_subgrids_launched = 0;
    return multipole_vc_cpu_subgrids_launched;
}
std::uint64_t multipole_legacy_cpu_subgrid_processed_performance_data(bool reset) {
    if (reset)
        multipole_legacy_cpu_subgrids_launched = 0;
    return multipole_legacy_cpu_subgrids_launched;
}

void register_performance_counters(void) {
    hpx::performance_counters::install_counter_type("/octotiger/compute/gpu/p2p_cuda",
        &p2p_cuda_gpu_subgrid_processed_performance_data,
        "Number of calls to the fmm solver (p2p) with CUDA. Each call handles one sub-grid for one "
        "sub-timestep");
    hpx::performance_counters::install_counter_type("/octotiger/compute/gpu/p2p_kokkos",
        &p2p_kokkos_gpu_subgrid_processed_performance_data,
        "Number of calls to the fmm solver (p2p) with KOKKOS_GPU. Each call handles one sub-grid for one "
        "sub-timestep");
    hpx::performance_counters::install_counter_type("/octotiger/compute/cpu/p2p_kokkos",
        &p2p_kokkos_cpu_subgrid_processed_performance_data,
        "Number of calls to the fmm solver (p2p) with KOKKOS_CPU. Each call handles one sub-grid for one "
        "sub-timestep");
    hpx::performance_counters::install_counter_type("/octotiger/compute/cpu/p2p_vc",
        &p2p_vc_cpu_subgrid_processed_performance_data,
        "Number of calls to the fmm solver (p2p) with VC. Each call handles one sub-grid for one "
        "sub-timestep");
    hpx::performance_counters::install_counter_type("/octotiger/compute/cpu/p2p_legacy",
        &p2p_legacy_cpu_subgrid_processed_performance_data,
        "Number of calls to the fmm solver (p2p) with LEGACY. Each call handles one sub-grid for one "
        "sub-timestep");

    hpx::performance_counters::install_counter_type("/octotiger/compute/gpu/multipole_cuda",
        &multipole_cuda_gpu_subgrid_processed_performance_data,
        "Number of calls to the fmm solver (multipole) with CUDA. Each call handles one sub-grid for one "
        "sub-timestep");
    hpx::performance_counters::install_counter_type("/octotiger/compute/gpu/multipole_kokkos",
        &multipole_kokkos_gpu_subgrid_processed_performance_data,
        "Number of calls to the fmm solver (multipole) with KOKKOS_GPU. Each call handles one sub-grid for one "
        "sub-timestep");
    hpx::performance_counters::install_counter_type("/octotiger/compute/cpu/multipole_kokkos",
        &multipole_kokkos_cpu_subgrid_processed_performance_data,
        "Number of calls to the fmm solver (multipole) with KOKKOS_CPU. Each call handles one sub-grid for one "
        "sub-timestep");
    hpx::performance_counters::install_counter_type("/octotiger/compute/cpu/multipole_vc",
        &multipole_vc_cpu_subgrid_processed_performance_data,
        "Number of calls to the fmm solver (multipole) with VC. Each call handles one sub-grid for one "
        "sub-timestep");
    hpx::performance_counters::install_counter_type("/octotiger/compute/cpu/multipole_legacy",
        &multipole_legacy_cpu_subgrid_processed_performance_data,
        "Number of calls to the fmm solver (multipole) with LEGACY. Each call handles one sub-grid for one "
        "sub-timestep");
}

}
}
