//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef OCTOTIGER_HAVE_CUDA

#include "octotiger/monopole_interactions/legacy/cuda_p2p_interaction_interface.hpp"
#include "octotiger/monopole_interactions/legacy/p2p_cuda_kernel.hpp"
#include "octotiger/monopole_interactions/util/calculate_stencil.hpp"

#include "octotiger/defs.hpp"
#include "octotiger/options.hpp"

#include <array>
#include <vector>

#include <buffer_manager.hpp>
#include <cuda_buffer_util.hpp>
#include <cuda_runtime.h>
#include <stream_manager.hpp>

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {
        cuda_p2p_interaction_interface::cuda_p2p_interaction_interface()
          : p2p_interaction_interface()
          , theta(opts().theta) {
        }

        void cuda_p2p_interaction_interface::compute_p2p_interactions(std::vector<real>& monopoles,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
            std::array<bool, geo::direction::count()>& is_direction_empty) {
            // Check where we want to run this:
            bool avail = stream_pool::interface_available<hpx::cuda::experimental::cuda_executor,
                pool_strategy>(opts().cuda_buffer_capacity);
            if (!avail || p2p_type == interaction_kernel_type::OLD) {
                // Run CPU implementation
                p2p_interaction_interface::compute_p2p_interactions(
                    monopoles, neighbors, type, dx, is_direction_empty);
            } else {
                // run on CUDA device
                cuda_launch_counter()++;

                // Pick device and stream
                size_t device_id =
                    stream_pool::get_next_device_id<hpx::cuda::experimental::cuda_executor,
                        pool_strategy>();
                stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy> executor;

                cuda_expansion_result_buffer_t potential_expansions_SoA;
                cuda_monopole_buffer_t local_monopoles(ENTRIES);
                recycler::cuda_device_buffer<double> device_local_monopoles(ENTRIES, device_id);
                recycler::cuda_device_buffer<double> erg(NUMBER_POT_EXPANSIONS_SMALL, device_id);

                // Move data into staging buffers
                update_input(monopoles, neighbors, type, local_monopoles, neighbor_empty_monopoles, grid_ptr);

                hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                    cudaMemcpyAsync, device_local_monopoles.device_side_buffer,
                    local_monopoles.data(), local_monopoles_size, cudaMemcpyHostToDevice);

                // Launch kernel and queue copying of results
                dim3 const grid_spec(1, 1, INX);
                dim3 const threads_per_block(1, INX, INX);
                void* args[] = {&(device_local_monopoles.device_side_buffer),
                    &(erg.device_side_buffer), &theta, &dx};
                // hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                //     cudalaunchkernel<decltype(cuda_p2p_interactions_kernel)>,
                //     cuda_p2p_interactions_kernel, grid_spec, threads_per_block, args, 0);
                executor.post(
                    cudaLaunchKernel<decltype(cuda_p2p_interactions_kernel)>,
                    cuda_p2p_interactions_kernel, grid_spec, threads_per_block, args, 0);
                auto fut = hpx::async(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
                    cudaMemcpyAsync, potential_expansions_SoA.get_pod(), erg.device_side_buffer,
                    potential_expansions_small_size, cudaMemcpyDeviceToHost);

                // Wait for stream to finish and allow thread to jump away in the meantime
                fut.get();

                // Copy results back into non-SoA array
                potential_expansions_SoA.add_to_non_SoA(grid_ptr->get_L());
            }
        }

    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
