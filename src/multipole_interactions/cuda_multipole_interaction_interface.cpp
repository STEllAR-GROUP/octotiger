#ifdef OCTOTIGER_CUDA_ENABLED
#include "cuda_multipole_interaction_interface.hpp"
#include "calculate_stencil.hpp"
#include "options.hpp"

extern options opts;
extern m2m_vector factor_half[20];
extern m2m_vector factor_sixth[20];
namespace octotiger {
namespace fmm {
    namespace multipole_interactions {
        // thread_local util::cuda_helper cuda_multipole_interaction_interface::gpu_interface;
        // thread_local kernel_scheduler cuda_multipole_interaction_interface::scheduler;
        // Define sizes of buffers
        constexpr size_t local_monopoles_size = NUMBER_LOCAL_MONOPOLE_VALUES * sizeof(real);
        constexpr size_t local_expansions_size = NUMBER_LOCAL_EXPANSION_VALUES * sizeof(real);
        constexpr size_t center_of_masses_size = NUMBER_MASS_VALUES * sizeof(real);
        constexpr size_t potential_expansions_size = NUMBER_POT_EXPANSIONS * sizeof(real);
        constexpr size_t angular_corrections_size = NUMBER_ANG_CORRECTIONS * sizeof(real);
        constexpr size_t factor_size = NUMBER_FACTORS * sizeof(real);
        constexpr size_t stencil_size = STENCIL_SIZE * sizeof(octotiger::fmm::multiindex<>);
        constexpr size_t indicator_size = STENCIL_SIZE * sizeof(real);

        kernel_scheduler::kernel_scheduler(void)
          : number_cuda_streams_managed(1)
          , slots_per_cuda_stream(2)
          , number_slots(number_cuda_streams_managed * slots_per_cuda_stream)
          , stencil(calculate_stencil()) {
            stream_interfaces = std::vector<util::cuda_helper>(number_cuda_streams_managed);

            slot_guards = std::vector<cudaEvent_t>(number_slots);
            for (cudaEvent_t& guard : slot_guards) {
                util::cuda_helper::cuda_error(cudaEventCreate(&guard, cudaEventDisableTiming));
            }

            local_expansions_slots =
                std::vector<struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>>(
                    number_slots);
            center_of_masses_slots =
                std::vector<struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>>(
                    number_slots);
            local_monopole_slots = std::vector<std::vector<real>>(number_slots);
            for (std::vector<real>& mons : local_monopole_slots) {
                mons = std::vector<real>(ENTRIES);
            }

            kernel_device_enviroments = std::vector<kernel_device_enviroment>(number_slots);
            size_t cur_interface = 0;
            size_t cur_slot = 0;
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
                util::cuda_helper::cuda_error(
                    cudaMalloc((void**) &(env.device_factor_half), factor_size));
                util::cuda_helper::cuda_error(
                    cudaMalloc((void**) &(env.device_factor_sixth), factor_size));

                // Create necessary data
                std::unique_ptr<real[]> indicator = std::make_unique<real[]>(STENCIL_SIZE);
                std::unique_ptr<real[]> factor_half_local =
                    std::make_unique<real[]>(NUMBER_FACTORS);
                std::unique_ptr<real[]> factor_sixth_local =
                    std::make_unique<real[]>(NUMBER_FACTORS);
                for (auto i = 0; i < 20; ++i) {
                    factor_half_local[i] = factor_half[i][0];
                    factor_sixth_local[i] = factor_sixth[i][0];
                }
                for (auto i = 0; i < STENCIL_SIZE; ++i) {
                    if (stencil.stencil_phase_indicator[i])
                        indicator[i] = 1.0;
                    else
                        indicator[i] = 0.0;
                }

                // Move data
                stream_interfaces[cur_interface].copy_async(env.device_stencil,
                    stencil.stencil_elements.data(), stencil_size, cudaMemcpyHostToDevice);
                stream_interfaces[cur_interface].copy_async(env.device_phase_indicator,
                    indicator.get(), indicator_size, cudaMemcpyHostToDevice);
                stream_interfaces[cur_interface].copy_async(env.device_factor_half,
                    factor_half_local.get(), factor_size, cudaMemcpyHostToDevice);
                stream_interfaces[cur_interface].copy_async(env.device_factor_sixth,
                    factor_sixth_local.get(), factor_size, cudaMemcpyHostToDevice);

                // Change stream interface if necessary
                cur_slot++;
                if (cur_slot >= slots_per_cuda_stream) {
                    cur_slot = 0;
                    cur_interface++;
                }
            }
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
                util::cuda_helper::cuda_error(cudaFree((void*) (env.device_factor_half)));
                util::cuda_helper::cuda_error(cudaFree((void*) (env.device_factor_sixth)));
            }

            // Destroy guards
            for (cudaEvent_t& guard : slot_guards) {
                util::cuda_helper::cuda_error(cudaEventDestroy(guard));
            }
        }

        int kernel_scheduler::get_launch_slot(void) {
            for (size_t slot_id = 0; slot_id < number_slots; ++slot_id) {
                const cudaError_t response = cudaEventQuery(slot_guards[slot_id]);
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
            return stream_interfaces[slot];
        }

        void kernel_scheduler::lock_slot_until_finished(size_t slot) {
            // Determine interface
            const size_t interface_id = slot / slots_per_cuda_stream;
            // Record guard event
            stream_interfaces[interface_id](
                [](cudaEvent_t& event, cudaStream_t& stream) -> cudaError_t {
                    return cudaEventRecord(event, stream);
                },
                slot_guards[slot]);
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
            slot = 0; // for debuggging
            if (slot == -1) {    // Run fallback cpu implementation
                update_input(monopoles, M_ptr, com_ptr, neighbors, type, dx, xbase,
                    local_monopoles_staging_area, local_expansions_staging_area,
                    center_of_masses_staging_area);
                compute_interactions(is_direction_empty, neighbors, local_monopoles_staging_area,
                    local_expansions_staging_area, center_of_masses_staging_area);
            } else {    // run on cuda device
                // Move data into SoA arrays
                auto staging_area = scheduler.get_staging_area(slot);

                try {
                    update_input(monopoles, M_ptr, com_ptr, neighbors, type, dx, xbase,
                        staging_area.local_monopoles, staging_area.local_expansions_SoA,
                        staging_area.center_of_masses_SoA);
                } catch (std::out_of_range& ex) {
                    std::cout << "\nOut of range exception caught.\n" << ex.what() << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << e.what() << '\n';    // or whatever
                } catch (...) {
                    std::cout << "Unknown exception" << std::endl;
                }

                // Move input data
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

                // Reset Output arrays
                gpu_interface.memset_async(
                    env.device_potential_expansions, 0, potential_expansions_size);
                gpu_interface.memset_async(
                    env.device_angular_corrections, 0, angular_corrections_size);

                // Launch kernel
                const dim3 grid_spec(1, 1, 1);
                const dim3 threads_per_block(8, 8, 8);
                if (type == RHO) {
                    void* args[] = {&(env.device_local_monopoles), &(env.device_center_of_masses),
                        &(env.device_local_expansions), &(env.device_potential_expansions),
                        &(env.device_angular_corrections), &(env.device_stencil),
                        &(env.device_phase_indicator), &(env.device_factor_half),
                        &(env.device_factor_sixth), &theta};
                    gpu_interface.execute(&cuda_multipole_interactions_kernel_rho, grid_spec,
                        threads_per_block, args, 0);
                    auto fut1 = gpu_interface.get_future();

                    fut1.get();
                    struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>
                        angular_corrections_SoA;
                    gpu_interface.copy_async(angular_corrections_SoA.get_pod(),
                        env.device_angular_corrections, angular_corrections_size,
                        cudaMemcpyDeviceToHost);
                    fut1 = gpu_interface.get_future();

                    fut1.get();
                    angular_corrections_SoA.to_non_SoA(grid_ptr->get_L_c());

                } else {
                    void* args[] = {&(env.device_local_monopoles), &(env.device_center_of_masses),
                        &(env.device_local_expansions), &(env.device_potential_expansions),
                        &(env.device_stencil), &(env.device_phase_indicator),
                        &(env.device_factor_half), &(env.device_factor_sixth), &theta};
                    gpu_interface.execute(&cuda_multipole_interactions_kernel_non_rho, grid_spec,
                        threads_per_block, args, 0);
                    auto fut1 = gpu_interface.get_future();

                    fut1.get();
                }

                // util::cuda_helper::cuda_error(cudaThreadSynchronize());
                struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>
                    potential_expansions_SoA;
                gpu_interface.copy_async(potential_expansions_SoA.get_pod(),
                    env.device_potential_expansions, potential_expansions_size, cudaMemcpyDeviceToHost);
                auto fut2 = gpu_interface.get_future();

                fut2.get();
                potential_expansions_SoA.add_to_non_SoA(grid_ptr->get_L());
            }
        }

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
