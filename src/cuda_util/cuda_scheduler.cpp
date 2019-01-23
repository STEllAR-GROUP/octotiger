#ifdef OCTOTIGER_HAVE_CUDA
#include "octotiger/cuda_util/cuda_scheduler.hpp"
#include "octotiger/monopole_interactions/calculate_stencil.hpp"
#include "octotiger/multipole_interactions/calculate_stencil.hpp"
#include "octotiger/monopole_interactions/p2p_cuda_kernel.hpp"
#include "octotiger/multipole_interactions/multipole_cuda_kernel.hpp"

#include "octotiger/options.hpp"
#include "octotiger/real.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#undef interface

namespace octotiger {
namespace fmm {
    thread_local kernel_scheduler kernel_scheduler::scheduler;
    kernel_scheduler::kernel_scheduler(void)
      : number_cuda_streams_managed(0)
      , slots_per_cuda_stream(1)    // Slots (queue per stream) is currently deactived
      , number_slots(number_cuda_streams_managed * slots_per_cuda_stream) {
        // Determine what the scheduler has to manage
        const size_t total_worker_count = hpx::get_os_thread_count();
        const size_t worker_id = hpx::get_worker_thread_num();
        const size_t streams_per_locality = opts().cuda_streams_per_locality;
        size_t streams_per_gpu = opts().cuda_streams_per_gpu;
        if (streams_per_gpu == 0)
          streams_per_gpu = streams_per_locality;
        if (streams_per_locality > 0) { // is cuda activated?
            size_t gpu_count = streams_per_locality / streams_per_gpu;
            // handle remaining streams by putting it on the next gpu
            if (streams_per_locality % streams_per_gpu != 0)
                gpu_count++;
            // How many streams does each worker handle?
            size_t number_of_streams_managed = streams_per_locality / total_worker_count;
            const size_t remaining_streams = streams_per_locality % total_worker_count;
            // if there are remaining streams, each worker receives one of them until there an
            // non left
            size_t offset = 0;
            if (remaining_streams != 0) {
                if (worker_id < remaining_streams)
                    // offset indicates that the current worker will get one of the remaining extra streams
                    offset = 1;
            }

            const size_t accumulated_offset =
                worker_id < remaining_streams ? worker_id : remaining_streams;
            const size_t worker_stream_id =
                worker_id * number_of_streams_managed + accumulated_offset;
            const size_t gpu_id = (worker_stream_id) / streams_per_gpu;
            // increase the number of streams by one of the remaining streams if necessary
            number_of_streams_managed += offset;
            std::cout << "Worker " << worker_id << " uses gpu " << gpu_id << " with "
                      << number_of_streams_managed << " streams "<< std::endl;

            // Number of streams the current HPX worker thread has to handle
            number_cuda_streams_managed = number_of_streams_managed;
            number_slots = number_cuda_streams_managed * slots_per_cuda_stream;

            // Get one slot per stream to handle the data on the cpu
            local_expansions_slots = std::vector<struct_of_array_data<expansion, real, 20, ENTRIES,
                SOA_PADDING, std::vector<real, cuda_pinned_allocator<real>>>>(number_slots);
            center_of_masses_slots =
                std::vector<struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING,
                    std::vector<real, cuda_pinned_allocator<real>>>>(number_slots);
            local_monopole_slots =
                std::vector<std::vector<real, cuda_pinned_allocator<real>>>(number_slots);
            for (std::vector<real, cuda_pinned_allocator<real>>& mons : local_monopole_slots) {
                mons = std::vector<real, cuda_pinned_allocator<real>>(ENTRIES);
            }

            // Get one kernel enviroment per stream to handle the data on the gpu
            kernel_device_enviroments = std::vector<kernel_device_enviroment>(number_slots);
            size_t cur_interface = 0;
            // Todo: Remove slots
            size_t cur_slot = 0;

            // Create necessary data and add padding
            const two_phase_stencil stencil = multipole_interactions::calculate_stencil();
            auto p2p_stencil_pair = monopole_interactions::calculate_stencil();

            // new p2p stencil stuff
            auto p2p_stencil_mask_pair = monopole_interactions::calculate_stencil_masks(p2p_stencil_pair.first);
            auto p2p_stencil_mask = p2p_stencil_mask_pair.first;
            auto p2p_four_constants = p2p_stencil_mask_pair.second;
            std::unique_ptr<float[]> stencil_masks = std::make_unique<float[]>(FULL_STENCIL_SIZE);
            std::unique_ptr<real[]> four_constants_tmp = std::make_unique<real[]>(4 * FULL_STENCIL_SIZE);
            for (auto i = 0; i < FULL_STENCIL_SIZE; i++) {
                four_constants_tmp[i * 4 + 0] = p2p_four_constants[i][0];
                four_constants_tmp[i * 4 + 1] = p2p_four_constants[i][1];
                four_constants_tmp[i * 4 + 2] = p2p_four_constants[i][2];
                four_constants_tmp[i * 4 + 3] = p2p_four_constants[i][3];
                if (p2p_stencil_mask[i]) {
                  stencil_masks[i] = 1.0;
                } else {
                  stencil_masks[i] = 0.0;
                }
            }

            // new multipole stencil
            auto multipole_stencil_pair =
                multipole_interactions::calculate_stencil_masks(stencil);
            auto multipole_stencil = multipole_stencil_pair.first;
            auto multipole_inner_stencil = multipole_stencil_pair.second;
            std::unique_ptr<float[]> multipole_stencil_masks = std::make_unique<float[]>(FULL_STENCIL_SIZE);
            std::unique_ptr<float[]> multipole_inner_stencil_masks = std::make_unique<float[]>(FULL_STENCIL_SIZE);
            for (auto i = 0; i < FULL_STENCIL_SIZE; ++i) {
                if (multipole_inner_stencil[i])
                    multipole_inner_stencil_masks[i] = 1.0;
                else
                    multipole_inner_stencil_masks[i] = 0.0;
                if (multipole_stencil[i])
                    multipole_stencil_masks[i] = 1.0;
                else
                    multipole_stencil_masks[i] = 0.0;
            }

            // Move data to constant memory, once per gpu
            if (worker_id == 0) {
                for (size_t gpu_id = 0; gpu_id < gpu_count; gpu_id++) {
                    util::cuda_helper::cuda_error(cudaSetDevice(gpu_id));
                    // Setting shared memory to the right (double) memory bank configuration
                    //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
/*                    monopole_interactions::copy_stencil_to_p2p_constant_memory(stencil_masks.get(),
                                                                               full_stencil_size/2);
                    monopole_interactions::copy_constants_to_p2p_constant_memory(four_constants_tmp.get(),
                                                                                 4 * full_stencil_size);
                    multipole_interactions::
                        copy_stencil_to_m2m_constant_memory(multipole_stencil_masks.get(), full_stencil_size/2);
                    multipole_interactions::
                        copy_indicator_to_m2m_constant_memory(multipole_inner_stencil_masks.get(),
                                                              full_stencil_size/2);*/
             cudaMemcpyToSymbol(multipole_interactions::device_constant_stencil_masks, multipole_stencil_masks.get(), full_stencil_size/2);
             cudaMemcpyToSymbol(multipole_interactions::device_stencil_indicator_const, multipole_inner_stencil_masks.get(), full_stencil_size/2);
             cudaMemcpyToSymbol(monopole_interactions::device_stencil_masks, multipole_stencil_masks.get(), full_stencil_size/2);
             cudaMemcpyToSymbol(monopole_interactions::device_four_constants, four_constants_tmp.get(), full_stencil_size * 4);

                }
            }

            // Allocate buffers on the gpus - once per stream
            size_t local_stream_id = 0;
            stream_interfaces.reserve(number_cuda_streams_managed);
            for (kernel_device_enviroment& env : kernel_device_enviroments) {
                const size_t worker_gpu_id = (worker_stream_id + local_stream_id) / streams_per_gpu;
                util::cuda_helper::cuda_error(cudaSetDevice(worker_gpu_id));
                stream_interfaces.emplace_back(worker_gpu_id);

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

                util::cuda_helper::cuda_error(cudaMalloc(
                    (void**) &(env.device_blocked_monopoles), 3 * potential_expansions_small_size));

                // Change stream interface if necessary
                local_stream_id++;
                cur_slot++;
                if (cur_slot >= slots_per_cuda_stream) {
                    //util::cuda_helper::cuda_error(cudaThreadSynchronize());
                    cur_slot = 0;
                    cur_interface++;
                }
            }
            // continue when all cuda things are handled
            util::cuda_helper::cuda_error(cudaThreadSynchronize());
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
