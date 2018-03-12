#pragma once

#include <memory>
#include <vector>

#include "../common_kernel/kernel_simd_types.hpp"

#include "geometry.hpp"
#include "grid.hpp"
// #include "node_server.hpp"
#include "interaction_types.hpp"
#include "taylor.hpp"

#include "../common_kernel/interaction_constants.hpp"
#include "../common_kernel/multiindex.hpp"
#include "m2m_kernel.hpp"
#include "m2p_kernel.hpp"

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        // for both local and multipole expansion
        // typedef taylor<4, real> expansion;

        class multipole_interaction_interface
        {
        public:
            multipole_interaction_interface(void);
            void compute_multipole_interactions(std::vector<real>& monopoles,
                std::vector<multipole>& M_ptr,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
                std::array<bool, geo::direction::count()>& is_direction_empty,
                std::array<real, NDIM> xbase);
            void set_grid_ptr(std::shared_ptr<grid> ptr) {
                grid_ptr = ptr;
            }

        protected:
            void update_input(std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
                std::array<real, NDIM> xbase);
            void compute_interactions(std::array<bool, geo::direction::count()>& is_direction_empty,
                std::vector<neighbor_gravity_type>& all_neighbor_interaction_data);

        protected:
            static thread_local std::vector<real> local_monopoles;
            static thread_local struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>
                local_expansions_SoA;
            static thread_local struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>
                center_of_masses_SoA;
            static thread_local const two_phase_stencil stencil;
            std::vector<bool> neighbor_empty_multipole;
            std::vector<bool> neighbor_empty_monopole;
            gsolve_type type;
            bool monopole_neighbors_exist;
            real dX;
            std::array<real, NDIM> xBase;
            interaction_kernel_type m2m_type;
            interaction_kernel_type m2p_type;

        private:
            std::shared_ptr<grid> grid_ptr;
            m2p_kernel mixed_interactions_kernel;

            bool z_skip[3];
            bool y_skip[3][3];
            bool x_skip[3][3][3];
        };
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
