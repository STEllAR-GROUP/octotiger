#pragma once
#ifdef OCTOTIGER_CUDA_ENABLED
#include "../cuda_util/cuda_helper.hpp"
#include "multipole_interaction_interface.hpp"
namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        class cuda_multipole_interaction_interface : public multipole_interaction_interface
        {
        public:
            cuda_multipole_interaction_interface(void)
              : multipole_interaction_interface() {}
            void update_input(std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
                std::array<real, NDIM> xbase) {
                multipole_interaction_interface::update_input(
                    monopoles, M_ptr, com_ptr, neighbors, type, dx, xbase);
            }

            void compute_interactions(interaction_kernel_type m2m_type,
                interaction_kernel_type m2p_type,
                std::array<bool, geo::direction::count()>& is_direction_empty,
                std::vector<neighbor_gravity_type>& all_neighbor_interaction_data) {
                multipole_interaction_interface::compute_interactions(
                    m2m_type, m2p_type, is_direction_empty, all_neighbor_interaction_data);
            }

        protected:
            void queue_multipole__kernel(void) {}

        protected:
            util::cuda_helper gpu_interface;
        };

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
