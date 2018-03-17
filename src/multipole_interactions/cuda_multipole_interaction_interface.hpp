#pragma once
#include "multipole_interaction_interface.hpp"
#ifdef OCTOTIGER_CUDA_ENABLED
#include <functional>
#include "../cuda_util/cuda_helper.hpp"
#include "cuda_kernel_methods.hpp"

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

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
