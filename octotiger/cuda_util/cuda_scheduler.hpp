#pragma once
#ifdef OCTOTIGER_HAVE_CUDA

#include "octotiger/config/export_definitions.hpp"
#include "octotiger/common_kernel/interaction_constants.hpp"
#include "octotiger/common_kernel/multiindex.hpp"
#include "octotiger/common_kernel/struct_of_array_data.hpp"
#include "octotiger/cuda_util/cuda_helper.hpp"
#include "octotiger/real.hpp"
#include "octotiger/taylor.hpp"

#include <cstddef>
#include <vector>

namespace octotiger { namespace fmm {
    // Define sizes of CUDA buffers
    constexpr std::size_t local_monopoles_size =
        NUMBER_LOCAL_MONOPOLE_VALUES * sizeof(real);
    constexpr std::size_t local_expansions_size =
        NUMBER_LOCAL_EXPANSION_VALUES * sizeof(real);
    constexpr std::size_t center_of_masses_size =
        NUMBER_MASS_VALUES * sizeof(real);
    constexpr std::size_t potential_expansions_size =
        NUMBER_POT_EXPANSIONS * sizeof(real);
    constexpr std::size_t potential_expansions_small_size =
        NUMBER_POT_EXPANSIONS_SMALL * sizeof(real);
    constexpr std::size_t angular_corrections_size =
        NUMBER_ANG_CORRECTIONS * sizeof(real);
    constexpr std::size_t stencil_size =
        P2P_PADDED_STENCIL_SIZE * sizeof(octotiger::fmm::multiindex<>);
    constexpr std::size_t indicator_size = STENCIL_SIZE * sizeof(real);
    constexpr std::size_t four_constants_size =
        4 * P2P_PADDED_STENCIL_SIZE * sizeof(real);
    constexpr std::size_t full_stencil_size = FULL_STENCIL_SIZE * sizeof(real);

    // Custom allocator for host-side CUDA vectors
    template <class T>
    struct cuda_pinned_allocator
    {
        using value_type = T;
        cuda_pinned_allocator() noexcept {}
        template <class U>
        cuda_pinned_allocator(cuda_pinned_allocator<U> const&) noexcept
        {
        }
        T* allocate(std::size_t n)
        {
            T* data;
            util::cuda_helper::cuda_error(
                cudaMallocHost((void**) &data, n * sizeof(T)));
            return data;
        }
        void deallocate(T* p, std::size_t n)
        {
            util::cuda_helper::cuda_error(cudaFreeHost(p));
        }
    };

    template <class T, class U>
    constexpr bool operator==(cuda_pinned_allocator<T> const&,
        cuda_pinned_allocator<U> const&) noexcept
    {
        return true;
    }

    template <class T, class U>
    constexpr bool operator!=(cuda_pinned_allocator<T> const&,
        cuda_pinned_allocator<U> const&) noexcept
    {
        return false;
    }

    template <typename T>
    using pinned_vector = std::vector<T, cuda_pinned_allocator<T>>;

    // Contains references to all data needed for one FMM interaction kernel run
    class kernel_staging_area
    {
    public:
        kernel_staging_area(pinned_vector<real>& local_monopoles,
            struct_of_array_data<expansion,
                real,
                20,
                ENTRIES,
                SOA_PADDING,
                pinned_vector<real>>& local_expansions_SoA,
            struct_of_array_data<space_vector,
                real,
                3,
                ENTRIES,
                SOA_PADDING,
                pinned_vector<real>>& center_of_masses_SoA)
          : local_monopoles(local_monopoles)
          , local_expansions_SoA(local_expansions_SoA)
          , center_of_masses_SoA(center_of_masses_SoA)
        {
        }
        pinned_vector<real>& local_monopoles;
        struct_of_array_data<expansion,
            real,
            20,
            ENTRIES,
            SOA_PADDING,
            pinned_vector<real>>& local_expansions_SoA;
        struct_of_array_data<space_vector,
            real,
            3,
            ENTRIES,
            SOA_PADDING,
            pinned_vector<real>>& center_of_masses_SoA;
    };

    // Contains pointers to device buffers
    struct kernel_device_enviroment
    {
        real* device_local_monopoles;
        real* device_local_expansions;
        real* device_center_of_masses;
        real* device_potential_expansions;
        real* device_angular_corrections;

        real* device_blocked_monopoles;
    };

    // Scheduler which decides on what device to launch kernel and what memory to use
    class kernel_scheduler
    {
    public:
        // Get a slot on any device to run a FMM kernel.
        // Return -1 if CPU slot, otherwise the slot number
        int get_launch_slot();
        // Get references to SoA memory for a slot
        kernel_staging_area get_staging_area(std::size_t slot);
        // Get references to SoA memory for a slot
        kernel_device_enviroment& get_device_enviroment(std::size_t slot);
        // Get the CUDA interface for a slot
        // Throws if a CPU slot (-1) is given
        util::cuda_helper& get_launch_interface(std::size_t slot);
        // Global scheduler instance for this HPX thread
        static OCTOTIGER_EXPORT kernel_scheduler& scheduler();

        OCTOTIGER_EXPORT void init();
        static OCTOTIGER_EXPORT void init_constants();

    public:
        kernel_scheduler(kernel_scheduler& other) = delete;
        kernel_scheduler(kernel_scheduler const& other) = delete;
        kernel_scheduler operator=(kernel_scheduler const& other) = delete;
        // Deallocates CUDA memory
        ~kernel_scheduler();

    private:
        // Constructs number of streams indicated by the options
        kernel_scheduler();

        // How many CUDA streams does scheduler manage
        std::size_t number_cuda_streams_managed;
        // How many slots are there per stram - basically the queue length per stream
        std::size_t const slots_per_cuda_stream;
        // How many slots are there
        std::size_t number_slots;
        bool is_initialized;

        // Contains number_cuda_streams_managed CUDA interfaces
        std::vector<util::cuda_helper> stream_interfaces;
        // Pinned memory for each stream, containing room for the expansions
        std::vector<struct_of_array_data<expansion,
            real,
            20,
            ENTRIES,
            SOA_PADDING,
            pinned_vector<real>>>
            local_expansions_slots;
        // Pinned memory for each stream, containing room for the center of masses
        std::vector<struct_of_array_data<space_vector,
            real,
            3,
            ENTRIES,
            SOA_PADDING,
            pinned_vector<real>>>
            center_of_masses_slots;
        // Pinned memory for each stream, containing room for the monopoles
        std::vector<pinned_vector<real>> local_monopole_slots;
        // Struct container pointers to all CUDA device buffers per stream
        std::vector<kernel_device_enviroment> kernel_device_enviroments;
    };
}}
#endif
