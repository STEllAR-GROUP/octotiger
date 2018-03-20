#ifdef OCTOTIGER_CUDA_ENABLED
#include "cuda_multipole_interaction_interface.hpp"
#include "calculate_stencil.hpp"
#include "options.hpp"

extern options opts;
namespace octotiger {
namespace fmm {
    namespace multipole_interactions {
        // thread_local util::cuda_helper cuda_multipole_interaction_interface::gpu_interface;
        thread_local kernel_scheduler cuda_multipole_interaction_interface::scheduler;
        // Define sizes of buffers
        constexpr size_t local_monopoles_size = NUMBER_LOCAL_MONOPOLE_VALUES * sizeof(real);
        constexpr size_t local_expansions_size = NUMBER_LOCAL_EXPANSION_VALUES * sizeof(real);
        constexpr size_t center_of_masses_size = NUMBER_MASS_VALUES * sizeof(real);
        constexpr size_t potential_expansions_size = NUMBER_POT_EXPANSIONS * sizeof(real);
        constexpr size_t angular_corrections_size = NUMBER_ANG_CORRECTIONS * sizeof(real);
        constexpr size_t stencil_size = STENCIL_SIZE * sizeof(octotiger::fmm::multiindex<>);
        constexpr size_t indicator_size = STENCIL_SIZE * sizeof(real);

        kernel_scheduler::kernel_scheduler(void)
          : number_cuda_streams_managed(2)
          , slots_per_cuda_stream(1)
          , number_slots(number_cuda_streams_managed * slots_per_cuda_stream) {
            stream_interfaces = std::vector<util::cuda_helper>(number_cuda_streams_managed);

            local_expansions_slots =
                std::vector<struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>>(
                    number_slots);
            center_of_masses_slots =
                std::vector<struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>>(
                    number_slots);
            local_monopole_slots = std::vector<std::vector<real, cuda_pinned_allocator<real>>>(number_slots);
            for (std::vector<real, cuda_pinned_allocator<real>>& mons : local_monopole_slots) {
                mons = std::vector<real, cuda_pinned_allocator<real>>(ENTRIES);
            }

            kernel_device_enviroments = std::vector<kernel_device_enviroment>(number_slots);
            size_t cur_interface = 0;
            size_t cur_slot = 0;

            // Create necessary data
            const two_phase_stencil stencil = calculate_stencil();
            std::unique_ptr<real[]> indicator = std::make_unique<real[]>(STENCIL_SIZE);
            for (auto i = 0; i < STENCIL_SIZE; ++i) {
                if (stencil.stencil_phase_indicator[i])
                    indicator[i] = 1.0;
                else
                    indicator[i] = 0.0;
            }

            for (kernel_device_enviroment& env : kernel_device_enviroments) {
                // Allocate memory on device
                util::cuda_helper::cuda_error(
                    cudaMalloc((void**) &(env.device_local_monopoles), local_monopoles_size));
                util::cuda_helper::cuda_error(
                    cudaMalloc((void**) &(env.device_local_expansions), local_expansions_size));
                util::cuda_helper::cuda_error(
                    cudaMalloc((void**) &(env.device_center_of_masses), center_of_masses_size));
                util::cuda_helper::cuda_error(cudaMalloc(
                    (void**) &(env.device_potential_expansions), potential_expansions_size));
                util::cuda_helper::cuda_error(cudaMalloc(
                    (void**) &(env.device_angular_corrections), angular_corrections_size));
                util::cuda_helper::cuda_error(
                    cudaMalloc((void**) &(env.device_stencil), stencil_size));
                util::cuda_helper::cuda_error(
                    cudaMalloc((void**) &(env.device_phase_indicator), indicator_size));

                // Move data
                stream_interfaces[cur_interface].copy_async(env.device_stencil,
                    stencil.stencil_elements.data(), stencil_size, cudaMemcpyHostToDevice);
                stream_interfaces[cur_interface].copy_async(env.device_phase_indicator,
                    indicator.get(), indicator_size, cudaMemcpyHostToDevice);

                // Change stream interface if necessary
                cur_slot++;
                if (cur_slot >= slots_per_cuda_stream) {
                    util::cuda_helper::cuda_error(cudaThreadSynchronize());
                    cur_slot = 0;
                    cur_interface++;
                }
            }
            util::cuda_helper::cuda_error(cudaThreadSynchronize());
            // for (auto i = 0; i < number_slots; ++i) {
            //     slot_guards[i] = false;
            // }
        }

        kernel_scheduler::~kernel_scheduler(void) {
            // Deallocate device buffers
            for (kernel_device_enviroment& env : kernel_device_enviroments) {
                util::cuda_helper::cuda_error(cudaFree((void*) (env.device_local_monopoles)));
                util::cuda_helper::cuda_error(cudaFree((void*) (env.device_local_expansions)));
                util::cuda_helper::cuda_error(cudaFree((void*) (env.device_center_of_masses)));
                util::cuda_helper::cuda_error(cudaFree((void*) (env.device_potential_expansions)));
                util::cuda_helper::cuda_error(cudaFree((void*) (env.device_angular_corrections)));
                util::cuda_helper::cuda_error(cudaFree((void*) (env.device_stencil)));
                util::cuda_helper::cuda_error(cudaFree((void*) (env.device_phase_indicator)));
            }
        }

        int kernel_scheduler::get_launch_slot(void) {
            for (size_t slot_id = 0; slot_id < number_cuda_streams_managed; ++slot_id) {
                const cudaError_t response = stream_interfaces[slot_id].pass_through(
                    [](cudaStream_t& stream) -> cudaError_t { return cudaStreamQuery(stream); });
                if (response == cudaSuccess)    // slot is free
                    return slot_id;
            }
            // No slots available
            return -1;
        }

        kernel_staging_area kernel_scheduler::get_staging_area(size_t slot) {
            return kernel_staging_area(local_monopole_slots[slot], local_expansions_slots[slot],
                center_of_masses_slots[slot]);
        }

        kernel_device_enviroment& kernel_scheduler::get_device_enviroment(size_t slot) {
            return kernel_device_enviroments[slot];
        }

        util::cuda_helper& kernel_scheduler::get_launch_interface(size_t slot) {
            size_t interface = slot / slots_per_cuda_stream;
            return stream_interfaces[slot];
        }

        cuda_multipole_interaction_interface::cuda_multipole_interaction_interface(void)
          : multipole_interaction_interface()
          , theta(opts.theta) {}

        void cuda_multipole_interaction_interface::compute_multipole_interactions(
            std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::array<real, NDIM> xbase) {
            // Check where we want to run this:
            int slot = scheduler.get_launch_slot();
            if (slot == -1) {    // Run fallback cpu implementation
                std::cout << "Running cpu fallback" << std::endl;
                multipole_interaction_interface::compute_multipole_interactions(
                    monopoles, M_ptr, com_ptr, neighbors, type, dx, is_direction_empty, xbase);
            } else {    // run on cuda device
                std::cerr << "Running cuda in slot " << slot << std::endl;
                // Move data into SoA arrays
                auto staging_area = scheduler.get_staging_area(slot);
                update_input(monopoles, M_ptr, com_ptr, neighbors, type, dx, xbase,
                    staging_area.local_monopoles, staging_area.local_expansions_SoA,
                    staging_area.center_of_masses_SoA);

                // Queue moving of input data to device
                util::cuda_helper& gpu_interface = scheduler.get_launch_interface(slot);
                kernel_device_enviroment& env = scheduler.get_device_enviroment(slot);
                gpu_interface.copy_async(env.device_local_monopoles,
                    staging_area.local_monopoles.data(), local_monopoles_size,
                    cudaMemcpyHostToDevice);
                gpu_interface.copy_async(env.device_local_expansions,
                    staging_area.local_expansions_SoA.get_pod(), local_expansions_size,
                    cudaMemcpyHostToDevice);
                gpu_interface.copy_async(env.device_center_of_masses,
                    staging_area.center_of_masses_SoA.get_pod(), center_of_masses_size,
                    cudaMemcpyHostToDevice);

                // Launch kernel and queue copying of results
                const dim3 grid_spec(1, 1, 1);
                const dim3 threads_per_block(8, 8, 8);
                if (type == RHO) {
                    void* args[] = {&(env.device_local_monopoles), &(env.device_center_of_masses),
                        &(env.device_local_expansions), &(env.device_potential_expansions),
                        &(env.device_angular_corrections), &(env.device_stencil),
                        &(env.device_phase_indicator), &theta};
                    gpu_interface.execute(&cuda_multipole_interactions_kernel_rho, grid_spec,
                        threads_per_block, args, 0);
                    gpu_interface.copy_async(angular_corrections_SoA.get_pod(),
                        env.device_angular_corrections, angular_corrections_size,
                        cudaMemcpyDeviceToHost);

                } else {
                    void* args[] = {&(env.device_local_monopoles), &(env.device_center_of_masses),
                        &(env.device_local_expansions), &(env.device_potential_expansions),
                        &(env.device_stencil), &(env.device_phase_indicator), &theta};
                    gpu_interface.execute(&cuda_multipole_interactions_kernel_non_rho, grid_spec,
                        threads_per_block, args, 0);
                }
                gpu_interface.copy_async(potential_expansions_SoA.get_pod(),
                    env.device_potential_expansions, potential_expansions_size,
                    cudaMemcpyDeviceToHost);

                // Wait for stream to finish and allow thread to jump away in the meantime
                auto fut = gpu_interface.get_future();
                fut.get();

                // Copy results back into non-SoA array
                potential_expansions_SoA.add_to_non_SoA(grid_ptr->get_L());
                if (type == RHO)
                    angular_corrections_SoA.to_non_SoA(grid_ptr->get_L_c());
            }
        }

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
