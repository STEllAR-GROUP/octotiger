#pragma once

#include <cstdint>
#include <atomic>

namespace octotiger {
namespace hydro {

extern std::atomic<std::uint64_t> hydro_cuda_gpu_subgrids_processed;
extern std::atomic<std::uint64_t> hydro_cuda_gpu_aggregated_subgrids_launches;
extern std::atomic<std::uint64_t> hydro_kokkos_cpu_subgrids_processed;
extern std::atomic<std::uint64_t> hydro_kokkos_cpu_aggregated_subgrids_launches;
extern std::atomic<std::uint64_t> hydro_kokkos_gpu_subgrids_processed;
extern std::atomic<std::uint64_t> hydro_kokkos_gpu_aggregated_subgrids_launches;
extern std::atomic<std::uint64_t> hydro_legacy_subgrids_processed;

std::uint64_t hydro_cuda_gpu_subgrid_processed_performance_data(bool reset);
std::uint64_t hydro_cuda_gpu_aggregated_subgrids_launches_performance_data(bool reset);
std::uint64_t hydro_cuda_avg_aggregation_rate(bool reset);
std::uint64_t hydro_kokkos_cpu_subgrids_processed_performance_data(bool reset);
std::uint64_t hydro_kokkos_cpu_aggregated_subgrids_launches_performance_data(bool reset);
std::uint64_t hydro_kokkos_cpu_avg_aggregation_rate(bool reset);
std::uint64_t hydro_kokkos_gpu_subgrids_processed_performance_data(bool reset);
std::uint64_t hydro_kokkos_gpu_aggregated_subgrids_launches_performance_data(bool reset);
std::uint64_t hydro_kokkos_gpu_avg_aggregation_rate(bool reset);
std::uint64_t hydro_legacy_subgrids_processed_performance_data(bool reset);

void register_performance_counters(void);

}
}
