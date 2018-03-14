#pragma once
#include "multipole_interaction_interface.hpp"
#ifdef OCTOTIGER_CUDA_ENABLED
#include <functional>
#include "../cuda_util/cuda_helper.hpp"
#include "cuda_kernel_methods.hpp"

extern taylor<4, real> factor;
extern taylor<4, m2m_vector> factor_half_v;
extern taylor<4, m2m_vector> factor_sixth_v;
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
            static thread_local util::cuda_helper gpu_interface;

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
