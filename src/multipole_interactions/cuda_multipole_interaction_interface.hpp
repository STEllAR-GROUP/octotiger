#pragma once
#include "multipole_interaction_interface.hpp"
#ifdef OCTOTIGER_CUDA_ENABLED
#include <functional>
#include "../cuda_util/cuda_helper.hpp"
#include "cuda_kernel_methods.hpp"

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        /// Contains references to all data needed for one FMM interaction kernel run
        struct kernel_staging_area
        {
            std::vector<real>& local_monopoles;
            struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>& local_expansions_SoA;
            struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>& center_of_masses_SoA;
        };
        /// Contains pointers to device buffers containing const data on device (stencil,
        /// factors)
        struct kernel_device_enviroment
        {
            real* device_local_monopoles;
            real* device_local_expansions;
            real* device_center_of_masses;
            real* device_potential_expansions;
            real* device_angular_corrections;

            real* device_factor_half;
            real* device_factor_sixth;
            octotiger::fmm::multiindex<>* device_stencil;
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
            kernel_device_enviroment get_device_enviroment(size_t slot);
            /// Get the cuda interface for a slot - throws exception if a CPU slot (-1)is given
            util::cuda_helper get_launch_interface(size_t slot);
            /// Mark this slot as occupied until the cuda kernel finishes - this ensures the memory
            /// does not get tainted
            void lock_slot_until_finished(size_t slot);

            /// Constructs number of streams indicated by the options
            kernel_scheduler(void);
            /// Deallocates cuda memory and destroys guards
            ~kernel_scheduler(void);

        private:
            /// How many cuda streams does scheduler manage
            const size_t number_cuda_streams_managed;
            /// How many slots are there per stram - basically the queue length per stream
            const size_t slots_per_cuda_stream;
            /// How many slots are there
            const size_t number_slots;

            // Contains number_cuda_streams_managed cuda interfaces
            std::vector<util::cuda_helper> stream_interfaces;
            // Slot guards to see whether a slot is currently occupied
            std::vector<cudaEvent_t> slot_guards;
            std::vector<struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>>
                local_expansions_slots;
            std::vector<struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>>
                center_of_masses_slots;
            std::vector<std::vector<real>> local_monopole_slots;
            const two_phase_stencil stencil;

            std::vector<kernel_device_enviroment> kernel_device_enviroments;
        };

        class cuda_multipole_interaction_interface : public multipole_interaction_interface
        {
        public:
            cuda_multipole_interaction_interface(void);
            void compute_multipole_interactions(std::vector<real>& monopoles,
                std::vector<multipole>& M_ptr,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
                std::array<bool, geo::direction::count()>& is_direction_empty,
                std::array<real, NDIM> xbase);

        protected:
            void queue_multipole_kernel(void) {}

        protected:
          static thread_local kernel_scheduler scheduler;
            util::cuda_helper gpu_interface;

            real* device_local_monopoles;
            real* device_local_expansions;
            real* device_center_of_masses;
            real* device_potential_expansions;
            real* device_angular_corrections;

            real* device_factor_half;
            real* device_factor_sixth;
            octotiger::fmm::multiindex<>* device_stencil;
            real* device_phase_indicator;

        private:
            real theta;
        };

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
