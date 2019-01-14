#ifdef OCTOTIGER_WITH_CUDA
#include "cuda_scheduler.hpp"
#include "../multipole_interactions/calculate_stencil.hpp"
#include "../monopole_interactions/calculate_stencil.hpp"
#include "options.hpp"

#undef interface

namespace octotiger {
namespace fmm {
    thread_local kernel_scheduler kernel_scheduler::scheduler;
    kernel_scheduler::kernel_scheduler(void)
      : number_cuda_streams_managed(opts().cuda_streams_per_thread)
      , slots_per_cuda_stream(1)
      , number_slots(number_cuda_streams_managed * slots_per_cuda_stream) {
        stream_interfaces = std::vector<util::cuda_helper>(number_cuda_streams_managed);

        local_expansions_slots = std::vector<struct_of_array_data<expansion, real, 20, ENTRIES,
            SOA_PADDING, std::vector<real, cuda_pinned_allocator<real>>>>(number_slots);
        center_of_masses_slots = std::vector<struct_of_array_data<space_vector, real, 3, ENTRIES,
            SOA_PADDING, std::vector<real, cuda_pinned_allocator<real>>>>(number_slots);
        local_monopole_slots =
            std::vector<std::vector<real, cuda_pinned_allocator<real>>>(number_slots);
        for (std::vector<real, cuda_pinned_allocator<real>>& mons : local_monopole_slots) {
            mons = std::vector<real, cuda_pinned_allocator<real>>(ENTRIES);
        }

        kernel_device_enviroments = std::vector<kernel_device_enviroment>(number_slots);
        size_t cur_interface = 0;
        size_t cur_slot = 0;

        // Create necessary data
        const two_phase_stencil stencil = multipole_interactions::calculate_stencil();
        const std::vector<std::array<real, 4>> four_constants =
            monopole_interactions::calculate_stencil().second;
        std::unique_ptr<real[]> indicator = std::make_unique<real[]>(STENCIL_SIZE);
        std::unique_ptr<real[]> four_tmp = std::make_unique<real[]>(4 * STENCIL_SIZE);
        for (auto i = 0; i < STENCIL_SIZE; ++i) {
            if (stencil.stencil_phase_indicator[i])
                indicator[i] = 1.0;
            else
                indicator[i] = 0.0;

            four_tmp[i * 4 + 0] = four_constants[i][0];
            four_tmp[i * 4 + 1] = four_constants[i][1];
            four_tmp[i * 4 + 2] = four_constants[i][2];
            four_tmp[i * 4 + 3] = four_constants[i][3];
        }

        for (kernel_device_enviroment& env : kernel_device_enviroments) {
            // Allocate memory on device
            util::cuda_helper::cuda_error(
                cudaMalloc((void**) &(env.device_local_monopoles), local_monopoles_size));
            util::cuda_helper::cuda_error(
                cudaMalloc((void**) &(env.device_local_expansions), local_expansions_size));
            util::cuda_helper::cuda_error(
                cudaMalloc((void**) &(env.device_center_of_masses), center_of_masses_size));
            util::cuda_helper::cuda_error(
                cudaMalloc((void**) &(env.device_potential_expansions), potential_expansions_size));
            util::cuda_helper::cuda_error(
                cudaMalloc((void**) &(env.device_angular_corrections), angular_corrections_size));
            util::cuda_helper::cuda_error(cudaMalloc((void**) &(env.device_stencil), stencil_size));
            util::cuda_helper::cuda_error(
                cudaMalloc((void**) &(env.device_phase_indicator), indicator_size));
            util::cuda_helper::cuda_error(
                cudaMalloc((void**) &(env.device_four_constants), four_constants_size));

            util::cuda_helper::cuda_error(
                cudaMalloc((void**) &(env.device_blocked_monopoles), 3*potential_expansions_small_size));

            // Move data
            stream_interfaces[cur_interface].copy_async(env.device_stencil,
                stencil.stencil_elements.data(), stencil_size, cudaMemcpyHostToDevice);
            stream_interfaces[cur_interface].copy_async(env.device_phase_indicator, indicator.get(),
                indicator_size, cudaMemcpyHostToDevice);
            stream_interfaces[cur_interface].copy_async(env.device_four_constants, four_tmp.get(),
                four_constants_size, cudaMemcpyHostToDevice);

            // Change stream interface if necessary
            cur_slot++;
            if (cur_slot >= slots_per_cuda_stream) {
                util::cuda_helper::cuda_error(cudaThreadSynchronize());
                cur_slot = 0;
                cur_interface++;
            }
        }
        util::cuda_helper::cuda_error(cudaThreadSynchronize());
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
        return kernel_staging_area(
            local_monopole_slots[slot], local_expansions_slots[slot], center_of_masses_slots[slot]);
    }

    kernel_device_enviroment& kernel_scheduler::get_device_enviroment(size_t slot) {
        return kernel_device_enviroments[slot];
    }

    util::cuda_helper& kernel_scheduler::get_launch_interface(size_t slot) {
        size_t interface = slot / slots_per_cuda_stream;
        return stream_interfaces[slot];
    }
}    // namespace fmm
}    // namespace octotiger
#endif
