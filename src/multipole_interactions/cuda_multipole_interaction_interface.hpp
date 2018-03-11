#pragma once
#include "multipole_interaction_interface.hpp"
#ifdef OCTOTIGER_CUDA_ENABLED
#include "../cuda_util/cuda_helper.hpp"
#include "cuda_kernel_methods.hpp"
#include <functional>
namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        class cuda_multipole_interaction_interface : public multipole_interaction_interface
        {
        public:
            cuda_multipole_interaction_interface(void)
              : multipole_interaction_interface() {}
            void update_input(std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
                std::array<real, NDIM> xbase) {
                // Move data into SoA arrays
                multipole_interaction_interface::update_input(
                    monopoles, M_ptr, com_ptr, neighbors, type, dx, xbase);
                // Check where we want to run this:
                // Allocate memory on device
                constexpr size_t local_monopoles_size = 1 * (ENTRIES) * sizeof(real);
                constexpr size_t local_expansions_size =
                    20 * (ENTRIES + SOA_PADDING) * sizeof(real);
                constexpr size_t center_of_masses_size = 3 * (ENTRIES + SOA_PADDING) * sizeof(real);
                constexpr size_t potential_expansions_size =
                    20 * (INNER_CELLS + SOA_PADDING) * sizeof(real);
                constexpr size_t angular_corrections_size =
                    3 * (INNER_CELLS + SOA_PADDING) * sizeof(real);
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
                // Queue asynchronous movement of data to device
                // Launch kernel
                // void* args[] = { &dev_c, &dev_a, &dev_b };
                void* args[] = {};
                const dim3 grid_spec(1, 1, 1);
                const dim3 threads_per_block(1, 1, 1);
                // std::function<cudaError_t(const void*,dim3,dim3,void**,size_t,cudaStream_t)> bla = std::bind(&cudaLaunchKernel);
                  // gpu_interface(&cudaLaunchKernel,(void*) &cuda_multipole_interactions_kernel,
                  //   grid_spec, threads_per_block, args);
            }

            void compute_interactions(interaction_kernel_type m2m_type,
                interaction_kernel_type m2p_type,
                std::array<bool, geo::direction::count()>& is_direction_empty,
                std::vector<neighbor_gravity_type>& all_neighbor_interaction_data) {
                multipole_interaction_interface::compute_interactions(
                    m2m_type, m2p_type, is_direction_empty, all_neighbor_interaction_data);
            }

        protected:
            void queue_multipole_kernel(void) {}

        protected:
            util::cuda_helper gpu_interface;
            real* device_local_monopoles;
            real* device_local_expansions;
            real* device_center_of_masses;
            real* device_potential_expansions;
            real* device_angular_corrections;
        };

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
