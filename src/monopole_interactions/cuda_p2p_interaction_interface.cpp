#ifdef OCTOTIGER_WITH_CUDA
#include "cuda_p2p_interaction_interface.hpp"
#include "p2p_cuda_kernel.hpp"
#include "options.hpp"

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {

        cuda_p2p_interaction_interface::cuda_p2p_interaction_interface(void)
          : p2p_interaction_interface()
          , theta(opts().theta) {}

        void cuda_p2p_interaction_interface::compute_p2p_interactions(std::vector<real>& monopoles,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
            std::array<bool, geo::direction::count()>& is_direction_empty) {
            // Check where we want to run this:
            int slot = kernel_scheduler::scheduler.get_launch_slot();
            if (slot == -1) {    // Run fkernel_scheduler::allback cpu implementation
                // std::cout << "Running cpu fallback" << std::endl;
                p2p_interaction_interface::compute_p2p_interactions(
                    monopoles, neighbors, type, dx, is_direction_empty);
            } else {    // run on cuda device
                // std::cerr << "Running cuda in slot " << slot << std::endl;
                // Move data into staging arrays
                auto staging_area = kernel_scheduler::scheduler.get_staging_area(slot);
                update_input(monopoles, neighbors, type, staging_area.local_monopoles);

                // Queue moving of input data to device
                util::cuda_helper& gpu_interface =
                    kernel_scheduler::scheduler.get_launch_interface(slot);
                kernel_device_enviroment& env =
                    kernel_scheduler::scheduler.get_device_enviroment(slot);
                gpu_interface.copy_async(env.device_local_monopoles,
                    staging_area.local_monopoles.data(), local_monopoles_size,
                    cudaMemcpyHostToDevice);

                // Launch kernel and queue copying of results
                const dim3 grid_spec(3);
                const dim3 threads_per_block(8, 8, 8);
                void* args[] = {&(env.device_local_monopoles), &(env.device_blocked_monopoles),
                                &(env.device_stencil), &(env.device_four_constants),&theta, &dx};
                gpu_interface.execute(
                    &cuda_p2p_interactions_kernel, grid_spec, threads_per_block, args, 0);
                void* sum_args[] = {&(env.device_blocked_monopoles)};
                const dim3 sum_spec(1);
                const dim3 threads(512);
                gpu_interface.execute(
                    &cuda_add_pot_blocks, sum_spec, threads, sum_args, 0);
                gpu_interface.copy_async(potential_expansions_SoA.get_pod(),
                    env.device_blocked_monopoles, potential_expansions_small_size,
                    cudaMemcpyDeviceToHost);

                // Wait for stream to finish and allow thread to jump away in the meantime
                auto fut = gpu_interface.get_future();
                fut.get();

                // Copy results back into non-SoA array
                potential_expansions_SoA.add_to_non_SoA(grid_ptr->get_L());
            }
        }

    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
