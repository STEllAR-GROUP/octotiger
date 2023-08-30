#include "octotiger/unitiger/hydro_impl/hydro_performance_counters.hpp"
#include <hpx/include/performance_counters.hpp>

namespace octotiger {
namespace hydro {

std::atomic<std::uint64_t> hydro_cuda_gpu_subgrids_processed(0);
std::atomic<std::uint64_t> hydro_cuda_gpu_aggregated_subgrids_launches(0);
std::atomic<std::uint64_t> hydro_kokkos_gpu_subgrids_processed(0);
std::atomic<std::uint64_t> hydro_kokkos_gpu_aggregated_subgrids_launches(0);
std::atomic<std::uint64_t> hydro_kokkos_cpu_subgrids_processed(0);
std::atomic<std::uint64_t> hydro_kokkos_cpu_aggregated_subgrids_launches(0);
std::atomic<std::uint64_t> hydro_legacy_subgrids_processed(0);

std::uint64_t hydro_cuda_gpu_subgrid_processed_performance_data(bool reset) {
    if (reset)
        hydro_cuda_gpu_subgrids_processed = 0;
    return hydro_cuda_gpu_subgrids_processed;
}
std::uint64_t hydro_cuda_gpu_aggregated_subgrids_launches_performance_data(bool reset) {
    if (reset)
        hydro_cuda_gpu_aggregated_subgrids_launches = 0;
    return hydro_cuda_gpu_aggregated_subgrids_launches;
}
std::uint64_t hydro_cuda_avg_aggregation_rate(bool reset) {
    if (hydro_cuda_gpu_aggregated_subgrids_launches > 0)
        return hydro_cuda_gpu_subgrids_processed / hydro_cuda_gpu_aggregated_subgrids_launches;
    else
        return 1;
}
std::uint64_t hydro_kokkos_cpu_subgrids_processed_performance_data(bool reset) {
    if (reset)
        hydro_kokkos_cpu_subgrids_processed = 0;
    return hydro_kokkos_cpu_subgrids_processed;
}
std::uint64_t hydro_kokkos_cpu_aggregated_subgrids_launches_performance_data(bool reset) {
    if (reset)
        hydro_kokkos_cpu_aggregated_subgrids_launches = 0;
    return hydro_kokkos_cpu_aggregated_subgrids_launches;
}
std::uint64_t hydro_kokkos_cpu_avg_aggregation_rate(bool reset) {
    if (hydro_kokkos_cpu_aggregated_subgrids_launches > 0)
        return hydro_kokkos_cpu_subgrids_processed / hydro_kokkos_cpu_aggregated_subgrids_launches;
    else
        return 1;
}
std::uint64_t hydro_kokkos_gpu_subgrids_processed_performance_data(bool reset) {
    if (reset)
        hydro_kokkos_gpu_subgrids_processed = 0;
    return hydro_kokkos_gpu_subgrids_processed;
}
std::uint64_t hydro_kokkos_gpu_aggregated_subgrids_launches_performance_data(bool reset) {
    if (reset)
        hydro_kokkos_gpu_aggregated_subgrids_launches = 0;
    return hydro_kokkos_gpu_aggregated_subgrids_launches;
}
std::uint64_t hydro_kokkos_gpu_avg_aggregation_rate(bool reset) {
    if (hydro_kokkos_gpu_aggregated_subgrids_launches > 0)
        return hydro_kokkos_gpu_subgrids_processed / hydro_kokkos_gpu_aggregated_subgrids_launches;
    else
        return 1;
}
std::uint64_t hydro_legacy_subgrids_processed_performance_data(bool reset) {
    if (reset)
        hydro_legacy_subgrids_processed = 0;
    return hydro_legacy_subgrids_processed;
}

void register_performance_counters(void) {
    hpx::performance_counters::install_counter_type("/octotiger/compute/gpu/hydro_cuda",
        &hydro_cuda_gpu_subgrid_processed_performance_data,
        "Number of calls to the hydro_solver with CUDA. Each call handles one sub-grid for one "
        "sub-timestep");
    hpx::performance_counters::install_counter_type("/octotiger/compute/gpu/hydro_cuda_aggregated",
        &hydro_cuda_gpu_aggregated_subgrids_launches_performance_data,
        "Number of aggregated calls to the hydro_solver with CUDA. Each call handles one ore more "
        "sub-grids for one sub-timestep");
    hpx::performance_counters::install_counter_type(
        "/octotiger/compute/gpu/hydro_cuda_aggregation_rate", &hydro_cuda_avg_aggregation_rate,
        "Average number of hydro CUDA kernels per aggregated kernel call");
    hpx::performance_counters::install_counter_type("/octotiger/compute/cpu/hydro_kokkos",
        &hydro_kokkos_cpu_subgrids_processed_performance_data,
        "Number of calls to the hydro_solver with KOKKOS_CPU. Each call handles one sub-grid for "
        "one sub-timestep");
    hpx::performance_counters::install_counter_type(
        "/octotiger/compute/cpu/hydro_kokkos_aggregated",
        &hydro_kokkos_cpu_aggregated_subgrids_launches_performance_data,
        "Number of aggregated calls to the hydro_solver with KOKKOS_CPU. Each call handles one ore "
        "more sub-grids for one sub-timestep");
    hpx::performance_counters::install_counter_type(
        "/octotiger/compute/cpu/hydro_kokkos_aggregation_rate",
        &hydro_kokkos_cpu_avg_aggregation_rate,
        "Average number of hydro KOKKOS_CPU kernels per aggregated kernel call");
    hpx::performance_counters::install_counter_type("/octotiger/compute/gpu/hydro_kokkos",
        &hydro_kokkos_gpu_subgrids_processed_performance_data,
        "Number of calls to the hydro_solver with KOKKOS_GPU. Each call handles one sub-grid for "
        "one sub-timestep");
    hpx::performance_counters::install_counter_type(
        "/octotiger/compute/gpu/hydro_kokkos_aggregated",
        &hydro_kokkos_gpu_aggregated_subgrids_launches_performance_data,
        "Number of aggregated calls to the hydro_solver with KOKKOS_GPU. Each call handles one ore "
        "more sub-grids for one sub-timestep");
    hpx::performance_counters::install_counter_type(
        "/octotiger/compute/gpu/hydro_kokkos_aggregation_rate",
        &hydro_kokkos_gpu_avg_aggregation_rate,
        "Average number of hydro KOKKOS_GPU kernels per aggregated kernel call");
    hpx::performance_counters::install_counter_type("/octotiger/compute/cpu/hydro_legacy",
        &hydro_legacy_subgrids_processed_performance_data,
        "Number of calls to the hydro_solver with LEGACY. Each call handles one sub-grid for "
        "one sub-timestep");
}

}
}
