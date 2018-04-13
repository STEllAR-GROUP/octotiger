#pragma once
#ifdef OCTOTIGER_CUDA_ENABLED
#include "../common_kernel/interaction_constants.hpp"
#include "../common_kernel/multiindex.hpp"
#include "../common_kernel/struct_of_array_data.hpp"
#include "cuda_helper.hpp"
#include "taylor.hpp"
namespace octotiger {
namespace fmm {
    // Define sizes of cuda buffers
    constexpr size_t local_monopoles_size = NUMBER_LOCAL_MONOPOLE_VALUES * sizeof(real);
    constexpr size_t local_expansions_size = NUMBER_LOCAL_EXPANSION_VALUES * sizeof(real);
    constexpr size_t center_of_masses_size = NUMBER_MASS_VALUES * sizeof(real);
    constexpr size_t potential_expansions_size = NUMBER_POT_EXPANSIONS * sizeof(real);
    constexpr size_t potential_expansions_small_size = NUMBER_POT_EXPANSIONS_SMALL * sizeof(real);
    constexpr size_t angular_corrections_size = NUMBER_ANG_CORRECTIONS * sizeof(real);
    constexpr size_t stencil_size = STENCIL_SIZE * sizeof(octotiger::fmm::multiindex<>);
    constexpr size_t indicator_size = STENCIL_SIZE * sizeof(real);
    constexpr size_t four_constants_size = 4 * STENCIL_SIZE * sizeof(real);

    /// Custom allocator for host-side cuda vectors
    template <class T>
    struct cuda_pinned_allocator
    {
        typedef T value_type;
        cuda_pinned_allocator() noexcept {}
        template <class U>
        cuda_pinned_allocator(const cuda_pinned_allocator<U>&) noexcept {}
        T* allocate(std::size_t n) {
            T* data;
            util::cuda_helper::cuda_error(cudaMallocHost((void**) &data, n * sizeof(T)));
            return data;
        }
        void deallocate(T* p, std::size_t n) {
            util::cuda_helper::cuda_error(cudaFreeHost(p));
        }
    };

    template <class T, class U>
    constexpr bool operator==(
        const cuda_pinned_allocator<T>&, const cuda_pinned_allocator<U>&) noexcept {
        return true;
    }

    template <class T, class U>
    constexpr bool operator!=(
        const cuda_pinned_allocator<T>&, const cuda_pinned_allocator<U>&) noexcept {
        return false;
    }

    /// Contains references to all data needed for one FMM interaction kernel run
    class kernel_staging_area
    {
    public:
        kernel_staging_area(
            std::vector<real, cuda_pinned_allocator<real>>& local_monopoles,
            struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING,
                std::vector<real, cuda_pinned_allocator<real>>>& local_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING,
                std::vector<real, cuda_pinned_allocator<real>>>& center_of_masses_SoA)
          : local_monopoles(local_monopoles)
          , local_expansions_SoA(local_expansions_SoA)
          , center_of_masses_SoA(center_of_masses_SoA) {}
        std::vector<real, cuda_pinned_allocator<real>>& local_monopoles;
        struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING,
            std::vector<real, cuda_pinned_allocator<real>>>& local_expansions_SoA;
        struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING,
            std::vector<real, cuda_pinned_allocator<real>>>& center_of_masses_SoA;
    };

    /// Contains pointers to device buffers
    struct kernel_device_enviroment
    {
        real* device_local_monopoles;
        real* device_local_expansions;
        real* device_center_of_masses;
        real* device_potential_expansions;
        real* device_angular_corrections;

        real* device_blocked_monopoles;

        octotiger::fmm::multiindex<>* device_stencil;
        real* device_four_constants;
        real* device_phase_indicator;
    };

    /// Scheduler which decides on what device to launch kernel and what memory to use
    class kernel_scheduler
    {
    public:
        /// Get a slot on any device to run a FMM kernel. Return -1 for CPU slot, else the slot
        /// ID
        int get_launch_slot(void);
        /// Get references to SoA memory for a slot
        kernel_staging_area get_staging_area(size_t slot);
        /// Get references to SoA memory for a slot
        kernel_device_enviroment& get_device_enviroment(size_t slot);
        /// Get the cuda interface for a slot - throws exception if a CPU slot (-1)is given
        util::cuda_helper& get_launch_interface(size_t slot);
        /// Global scheduler instance for this HPX thread
        static thread_local kernel_scheduler scheduler;

    public:
        kernel_scheduler(kernel_scheduler& other) = delete;
        kernel_scheduler(const kernel_scheduler& other) = delete;
        kernel_scheduler operator=(const kernel_scheduler& other) = delete;
        /// Deallocates cuda memory
        ~kernel_scheduler(void);

    private:
        /// Constructs number of streams indicated by the options
        kernel_scheduler(void);

        /// How many cuda streams does scheduler manage
        size_t number_cuda_streams_managed;
        /// How many slots are there per stram - basically the queue length per stream
        const size_t slots_per_cuda_stream;
        /// How many slots are there
        size_t number_slots;

        /// Contains number_cuda_streams_managed cuda interfaces
        std::vector<util::cuda_helper> stream_interfaces;
        /// Pinned memory for each stream, containing room for the expansions
        std::vector<struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING,
            std::vector<real, cuda_pinned_allocator<real>>>>
            local_expansions_slots;
        /// Pinned memory for each stream, containing room for the center of masses
        std::vector<struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING,
            std::vector<real, cuda_pinned_allocator<real>>>>
            center_of_masses_slots;
        /// Pinned memory for each stream, containing room for the monopoles
        std::vector<std::vector<real, cuda_pinned_allocator<real>>> local_monopole_slots;
        /// Struct container pointers to all cuda device buffers per stream
        std::vector<kernel_device_enviroment> kernel_device_enviroments;
    };
}    // namespace fmm
}    // namespace octotiger
#endif
