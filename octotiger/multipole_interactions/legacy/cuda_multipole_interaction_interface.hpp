//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#include "octotiger/multipole_interactions/legacy/multipole_interaction_interface.hpp"    // will be used as fallback in non-cuda compilations

#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
#include "octotiger/cuda_util/cuda_helper.hpp"
#include "octotiger/cuda_util/cuda_scheduler.hpp"

#include "octotiger/real.hpp"

#include <array>
#include <vector>

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
            cuda_multipole_interaction_interface();
            /// Takes AoS input, converts, launches kernel and writes AoS results back into L,L_c
            void compute_multipole_interactions(std::vector<real>& monopoles,
                std::vector<multipole>& M_ptr,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
                std::array<bool, geo::direction::count()>& is_direction_empty,
                std::array<real, NDIM> xbase, const bool use_root_stencil);

        protected:
            real theta;
        };

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
