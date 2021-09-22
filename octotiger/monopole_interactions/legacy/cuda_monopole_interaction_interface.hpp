//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
// will be used as fallback in non-cuda compilations
#include "octotiger/monopole_interactions/legacy/monopole_interaction_interface.hpp"

#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
#include "octotiger/cuda_util/cuda_helper.hpp"
#include "octotiger/cuda_util/cuda_scheduler.hpp"

#include "octotiger/real.hpp"

#include <array>
#include <vector>

namespace octotiger {
namespace fmm {
    namespace monopole_interactions {
        /** Interface to calculate monopole monopole FMM interactions on either a cuda device or on
         * the cpu! It takes AoS data, transforms it into SoA data, moves it to the cuda device,
         * launches cuda kernel on a slot given by the scheduler, gets results and stores them in
         * the AoS arrays L and L_c. Load balancing between CPU and GPU is done by the scheduler
         * (see
         * ../cuda_util/cuda_scheduler.hpp). If the scheduler detects that the cuda device is
         * busy it will use the normal CPU implementation of the interface as fallback!
         */
        class cuda_monopole_interaction_interface : public monopole_interaction_interface
        {
        public:
            cuda_monopole_interaction_interface();
            /** Takes AoS data, converts it, calculates FMM monopole-monopole interactions,
             * stores results in L */
            void compute_interactions(std::vector<real>& monopoles,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
                std::array<bool, geo::direction::count()>& is_direction_empty,
                std::shared_ptr<grid>& grid, const bool contains_multipole_neighbor);

        protected:
            real theta;
            /// Host-side pinned memory buffer for potential expansions results
            // struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING,
            //     std::vector<real, cuda_pinned_allocator<real>>>
            //     potential_expansions_SoA;
        };

    }    // namespace monopole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
