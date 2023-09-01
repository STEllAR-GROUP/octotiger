
#pragma once

#include <cstdint>
#include <atomic>

namespace octotiger {
namespace fmm {

extern std::atomic<std::uint64_t> p2p_kokkos_cpu_subgrids_launched;
extern std::atomic<std::uint64_t> p2p_kokkos_gpu_subgrids_launched;
extern std::atomic<std::uint64_t> p2p_cuda_gpu_subgrids_launched;
extern std::atomic<std::uint64_t> p2p_vc_cpu_subgrids_launched;
extern std::atomic<std::uint64_t> p2p_legacy_cpu_subgrids_launched;

extern std::atomic<std::uint64_t> multipole_kokkos_cpu_subgrids_launched;
extern std::atomic<std::uint64_t> multipole_kokkos_gpu_subgrids_launched;
extern std::atomic<std::uint64_t> multipole_cuda_gpu_subgrids_launched;
extern std::atomic<std::uint64_t> multipole_vc_cpu_subgrids_launched;
extern std::atomic<std::uint64_t> multipole_legacy_cpu_subgrids_launched;

std::uint64_t p2p_kokkos_cpu_subgrid_processed_performance_data(bool reset);
std::uint64_t p2p_kokkos_gpu_subgrid_processed_performance_data(bool reset);
std::uint64_t p2p_cuda_gpu_subgrid_processed_performance_data(bool reset);
std::uint64_t p2p_vc_cpu_subgrid_processed_performance_data(bool reset);
std::uint64_t p2p_legacy_cpu_subgrid_processed_performance_data(bool reset);

std::uint64_t multipole_kokkos_cpu_subgrid_processed_performance_data(bool reset);
std::uint64_t multipole_kokkos_gpu_subgrid_processed_performance_data(bool reset);
std::uint64_t multipole_cuda_gpu_subgrid_processed_performance_data(bool reset);
std::uint64_t multipole_vc_cpu_subgrid_processed_performance_data(bool reset);
std::uint64_t multipole_legacy_cpu_subgrid_processed_performance_data(bool reset);

void register_performance_counters(void);

}
}
