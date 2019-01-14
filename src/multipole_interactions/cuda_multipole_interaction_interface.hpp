#pragma once
#include "multipole_interaction_interface.hpp" // will be used as fallback in non-cuda compilations
#ifdef OCTOTIGER_WITH_CUDA
#include <functional>
#include "../cuda_util/cuda_helper.hpp"
#include "../cuda_util/cuda_scheduler.hpp"

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {
        /** Interface to calculate multipole FMM interactions on either a cuda device or on the
         * cpu! It takes AoS data, transforms it into SoA data, moves it to the cuda device,
         * launches cuda kernel on a slot given by the scheduler, gets results and stores them in
         * the AoS arrays L and L_c. Load balancing between CPU and GPU is done by the scheduler (see
         * ../cuda_util/cuda_scheduler.hpp). If the scheduler detects that the cuda device is
         * busy it will use the normal CPU implementation of the interface as fallback!
         */
        class cuda_multipole_interaction_interface : public multipole_interaction_interface
        {
        public:
            cuda_multipole_interaction_interface(void);
            /// Takes AoS input, converts, launches kernel and writes AoS results back into L,L_c
            void compute_multipole_interactions(std::vector<real>& monopoles,
                std::vector<multipole>& M_ptr,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
                std::array<bool, geo::direction::count()>& is_direction_empty,
                std::array<real, NDIM> xbase);

        protected:
            real theta;

            /// Host-side pinned memory buffer for angular corrections results
            struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING,
                std::vector<real, cuda_pinned_allocator<real>>>
                angular_corrections_SoA;
            /// Host-side pinned memory buffer for potential expansions results
            struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING,
                std::vector<real, cuda_pinned_allocator<real>>>
                potential_expansions_SoA;
        };

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
