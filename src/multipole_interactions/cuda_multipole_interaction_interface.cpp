#ifdef OCTOTIGER_CUDA_ENABLED
#include "cuda_multipole_interaction_interface.hpp"
#include "options.hpp"

extern options opts;
extern m2m_vector factor_half[20];
extern m2m_vector factor_sixth[20];
namespace octotiger {
namespace fmm {
    namespace multipole_interactions {
        // thread_local util::cuda_helper cuda_multipole_interaction_interface::gpu_interface;

        cuda_multipole_interaction_interface::cuda_multipole_interaction_interface(void)
          : multipole_interaction_interface()
          , theta(opts.theta) {}

        void cuda_multipole_interaction_interface::compute_multipole_interactions(
            std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::array<real, NDIM> xbase) {
            if (true) {
            update_input(monopoles, M_ptr, com_ptr, neighbors, type, dx, xbase,
                local_monopoles_staging_area, local_expansions_staging_area,
                center_of_masses_staging_area);
            compute_interactions(is_direction_empty, neighbors, local_monopoles_staging_area,
                local_expansions_staging_area, center_of_masses_staging_area);
            } else {
                // Move data into SoA arrays
            update_input(monopoles, M_ptr, com_ptr, neighbors, type, dx, xbase,
                local_monopoles_staging_area, local_expansions_staging_area,
                center_of_masses_staging_area);
                // Check where we want to run this:
                // Define sizes of buffers
                constexpr size_t local_monopoles_size = NUMBER_LOCAL_MONOPOLE_VALUES * sizeof(real);
                constexpr size_t local_expansions_size =
                    NUMBER_LOCAL_EXPANSION_VALUES * sizeof(real);
                constexpr size_t center_of_masses_size = NUMBER_MASS_VALUES * sizeof(real);
                constexpr size_t potential_expansions_size = NUMBER_POT_EXPANSIONS * sizeof(real);
                constexpr size_t angular_corrections_size = NUMBER_ANG_CORRECTIONS * sizeof(real);
                constexpr size_t factor_size = NUMBER_FACTORS * sizeof(real);
                constexpr size_t stencil_size = STENCIL_SIZE * sizeof(octotiger::fmm::multiindex<>);
                constexpr size_t indicator_size = STENCIL_SIZE * sizeof(real);

                // Move data into easier movable data structures
                std::unique_ptr<real[]> indicator = std::make_unique<real[]>(STENCIL_SIZE);
                std::unique_ptr<real[]> factor_half_local = std::make_unique<real[]>(NUMBER_FACTORS);
                std::unique_ptr<real[]> factor_sixth_local = std::make_unique<real[]>(NUMBER_FACTORS);
                for (auto i = 0; i < 20; ++i) {
                    factor_half_local[i] = factor_half[i][0];
                    factor_sixth_local[i] = factor_sixth[i][0];
                }
                // if (STENCIL_SIZE != stencil.stencil_elements.size())
                //     std::cout << "what what" << stencil.stencil_elements.size();
                for (auto i = 0; i < STENCIL_SIZE; ++i) {
                    if (stencil.stencil_phase_indicator[i])
                        indicator[i] = 1.0;
                    else
                        indicator[i] = 0.0;
                }

                // Allocate memory on device
                util::cuda_helper::cuda_error(
                    cudaMalloc((void**) &device_local_monopoles, local_monopoles_size));
                util::cuda_helper::cuda_error(
                    cudaMalloc((void**) &device_local_expansions, local_expansions_size));
                util::cuda_helper::cuda_error(
                    cudaMalloc((void**) &device_center_of_masses, center_of_masses_size));
                util::cuda_helper::cuda_error(
                    cudaMalloc((void**) &device_potential_expansions, potential_expansions_size));
                util::cuda_helper::cuda_error(
                    cudaMalloc((void**) &device_angular_corrections, angular_corrections_size));
                util::cuda_helper::cuda_error(cudaMalloc((void**) &device_stencil, stencil_size));
                util::cuda_helper::cuda_error(
                    cudaMalloc((void**) &device_phase_indicator, indicator_size));
                util::cuda_helper::cuda_error(
                    cudaMalloc((void**) &device_factor_half, factor_size));
                util::cuda_helper::cuda_error(
                    cudaMalloc((void**) &device_factor_sixth, factor_size));
                gpu_interface.memset_async(device_local_expansions, 0, local_expansions_size);

                // Move const data
                gpu_interface.copy_async(device_stencil,
                    multipole_interaction_interface::stencil.stencil_elements.data(), stencil_size,
                    cudaMemcpyHostToDevice);
                gpu_interface.copy_async(device_phase_indicator, indicator.get(), indicator_size,
                    cudaMemcpyHostToDevice);
                gpu_interface.copy_async(
                    device_factor_half, factor_half_local.get(), factor_size, cudaMemcpyHostToDevice);
                gpu_interface.copy_async(
                    device_factor_sixth, factor_sixth_local.get(), factor_size, cudaMemcpyHostToDevice);

                // Move input data
                gpu_interface.copy_async(device_local_monopoles, local_monopoles_staging_area.data(),
                    local_monopoles_size, cudaMemcpyHostToDevice);
                gpu_interface.copy_async(device_local_expansions, local_expansions_staging_area.get_pod(),
                    local_expansions_size, cudaMemcpyHostToDevice);
                gpu_interface.copy_async(device_center_of_masses, center_of_masses_staging_area.get_pod(),
                    center_of_masses_size, cudaMemcpyHostToDevice);

                // Reset Output arrays
                gpu_interface.memset_async(
                    device_potential_expansions, 0, potential_expansions_size);
                gpu_interface.memset_async(device_angular_corrections, 0, angular_corrections_size);

                // Launch kernel
                const dim3 grid_spec(1, 1, 1);
                const dim3 threads_per_block(8, 8, 8);
                if (type == RHO) {
                    void* args[] = {&device_local_monopoles, &device_center_of_masses,
                        &device_local_expansions, &device_potential_expansions,
                        &device_angular_corrections, &device_stencil, &device_phase_indicator,
                        &device_factor_half, &device_factor_sixth, &theta};
                    gpu_interface.execute(&cuda_multipole_interactions_kernel_rho, grid_spec,
                        threads_per_block, args, 0);
                    auto fut1 = gpu_interface.get_future();

                    fut1.get();
                    struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>
                        angular_corrections_SoA;
                    gpu_interface.copy_async(angular_corrections_SoA.get_pod(),
                        device_angular_corrections, angular_corrections_size,
                        cudaMemcpyDeviceToHost);
                    fut1 = gpu_interface.get_future();

                    fut1.get();
                    angular_corrections_SoA.to_non_SoA(grid_ptr->get_L_c());

                } else {
                    void* args[] = {&device_local_monopoles, &device_center_of_masses,
                        &device_local_expansions, &device_potential_expansions, &device_stencil,
                        &device_phase_indicator, &device_factor_half, &device_factor_sixth, &theta};
                    gpu_interface.execute(&cuda_multipole_interactions_kernel_non_rho, grid_spec,
                        threads_per_block, args, 0);
                    auto fut1 = gpu_interface.get_future();

                    fut1.get();
                }

                // util::cuda_helper::cuda_error(cudaThreadSynchronize());
                struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>
                    potential_expansions_SoA;
                gpu_interface.copy_async(potential_expansions_SoA.get_pod(),
                    device_potential_expansions, potential_expansions_size, cudaMemcpyDeviceToHost);
                auto fut2 = gpu_interface.get_future();

                fut2.get();
                potential_expansions_SoA.add_to_non_SoA(grid_ptr->get_L());
            }
        }

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
