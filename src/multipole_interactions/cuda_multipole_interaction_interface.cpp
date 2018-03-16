#ifdef OCTOTIGER_CUDA_ENABLED
#include "cuda_multipole_interaction_interface.hpp"
#include "options.hpp"

extern options opts;
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
            // multipole_interaction_interface::compute_multipole_interactions(
            //     monopoles, M_ptr, com_ptr, neighbors, type, dx, is_direction_empty, xbase);

            // Move data into SoA arrays
            multipole_interaction_interface::update_input(
                monopoles, M_ptr, com_ptr, neighbors, type, dx, xbase);
            // Check where we want to run this:
            // Define sizes of buffers
            constexpr size_t local_monopoles_size = NUMBER_LOCAL_MONOPOLE_VALUES * sizeof(real);
            constexpr size_t local_expansions_size = NUMBER_LOCAL_EXPANSION_VALUES * sizeof(real);
            constexpr size_t center_of_masses_size = NUMBER_MASS_VALUES * sizeof(real);
            constexpr size_t potential_expansions_size = NUMBER_POT_EXPANSIONS * sizeof(real);
            constexpr size_t angular_corrections_size = NUMBER_ANG_CORRECTIONS * sizeof(real);
            constexpr size_t factor_size = NUMBER_FACTORS * sizeof(real);
            constexpr size_t stencil_size = STENCIL_SIZE * sizeof(octotiger::fmm::multiindex<>);
            constexpr size_t indicator_size = STENCIL_SIZE * sizeof(real);

            // Move data into easier movable data structures
            std::cout << sizeof(real) << " " << indicator_size << std::endl ;
            std::unique_ptr<real[]> indicator = std::make_unique<real[]>(STENCIL_SIZE);
            if (STENCIL_SIZE != stencil.stencil_elements.size())
              std::cout << "what what" << stencil.stencil_elements.size();
            for (auto i = 0; i < STENCIL_SIZE; ++i) {
              if(stencil.stencil_phase_indicator[i])
                indicator[i] = 1.0;
              else
                indicator[i] = 0.0;
            }
            std::unique_ptr<real[]> factor_half = std::make_unique<real[]>(NUMBER_FACTORS);
            std::unique_ptr<real[]> factor_sixth = std::make_unique<real[]>(NUMBER_FACTORS);
            for (auto i = 0; i < 20; ++i) {
                factor_half[i] = factor_half_v[i][0];
                factor_sixth[i] = factor_sixth_v[i][0];
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
            util::cuda_helper::cuda_error(cudaMalloc((void**) &device_factor_half, factor_size));
            util::cuda_helper::cuda_error(cudaMalloc((void**) &device_factor_sixth, factor_size));
            gpu_interface.memset_async(device_local_expansions, 0, local_expansions_size);

            // Move const data
            gpu_interface.copy_async(device_stencil,
                multipole_interaction_interface::stencil.stencil_elements.data(), stencil_size,
                cudaMemcpyHostToDevice);
            gpu_interface.copy_async(
                device_phase_indicator, indicator.get(), indicator_size, cudaMemcpyHostToDevice);
            gpu_interface.copy_async(
                device_factor_half, factor_half.get(), factor_size, cudaMemcpyHostToDevice);
            gpu_interface.copy_async(
                device_factor_sixth, factor_sixth.get(), factor_size, cudaMemcpyHostToDevice);

            // Move input data
            gpu_interface.copy_async(device_local_monopoles, local_monopoles.data(),
                local_monopoles_size, cudaMemcpyHostToDevice);
            gpu_interface.copy_async(device_local_expansions, local_expansions_SoA.get_pod(),
                local_expansions_size, cudaMemcpyHostToDevice);
            gpu_interface.copy_async(device_center_of_masses, center_of_masses_SoA.get_pod(),
                center_of_masses_size, cudaMemcpyHostToDevice);

            // Reset Output arrays
            gpu_interface.memset_async(device_potential_expansions, 0, potential_expansions_size);
            gpu_interface.memset_async(device_angular_corrections, 0, angular_corrections_size);

            // Launch kernel
            void* args[] = {&device_local_monopoles, &device_center_of_masses,
                &device_local_expansions, &device_potential_expansions, &device_angular_corrections,
                &device_stencil, &device_phase_indicator, &device_factor_half, &device_factor_sixth,
                &theta};
            const dim3 grid_spec(1, 1, 1);
            const dim3 threads_per_block(8, 8, 8);
            gpu_interface.execute(
                &cuda_multipole_interactions_kernel, grid_spec, threads_per_block, args, 0);
            std::cout << "Started Kernel!" << std::endl;

            struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>
                potential_expansions_SoA;
            struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>
                angular_corrections_SoA;
            util::cuda_helper::cuda_error(cudaThreadSynchronize());
            // auto fut = gpu_interface.get_future();
            // fut.get();
            // util::cuda_helper::cuda_error(cudaThreadSynchronize());
            gpu_interface.copy_async(potential_expansions_SoA.get_pod(), device_potential_expansions,
                 potential_expansions_size,
                cudaMemcpyDeviceToHost);
            util::cuda_helper::cuda_error(cudaThreadSynchronize());
            gpu_interface.copy_async(angular_corrections_SoA.get_pod(), device_angular_corrections,
                angular_corrections_size, cudaMemcpyDeviceToHost);
            util::cuda_helper::cuda_error(cudaThreadSynchronize());
            // util::cuda_helper::cuda_error(cudaThreadSynchronize());
            // auto fut2 = gpu_interface.get_future();
            // fut2.get();
            std::cout << "Kernel finished!" << std::endl;

            // std::ofstream out("gpuresults.txt");
            // potential_expansions_SoA.print(out);
            // out.close();
            // std::ofstream out2("gpuresults2.txt");
            // angular_corrections_SoA.print(out2);
            // out2.close();
            // std::cin.get();
            if (type == RHO) {
             angular_corrections_SoA.to_non_SoA(grid_ptr->get_L_c());
            }

             potential_expansions_SoA.add_to_non_SoA(grid_ptr->get_L());
        }

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
