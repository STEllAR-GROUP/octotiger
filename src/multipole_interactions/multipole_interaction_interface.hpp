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
        private:
            /*
             * logical structure of all arrays:
             * cube of 8x8x8 cells of configurable size,
             * input variables (local_expansions, center_of_masses) have
             * an additional 1-sized layer of cells around it for padding
             */

            std::shared_ptr<grid> grid_ptr;
            m2p_kernel mixed_interactions_kernel;

            bool z_skip[3];
            bool y_skip[3][3];
            bool x_skip[3][3][3];

        protected:
            std::vector<real> local_monopoles;
            struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING> local_expansions_SoA;
            struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING> center_of_masses_SoA;

            std::vector<bool> neighbor_empty_multipole;
            std::vector<bool> neighbor_empty_monopole;

            gsolve_type type;

            bool monopole_neighbors_exist;
            real dX;
            std::array<real, NDIM> xBase;

        public:
            static two_phase_stencil stencil;

            // at this point, uses the old datamembers of the grid class as input
            // and converts them to the new data structure
            multipole_interaction_interface(void);
            void update_input(std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
                std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
                std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
                std::array<real, NDIM> xbase);

            void compute_interactions(interaction_kernel_type m2m_type,
                interaction_kernel_type m2p_type,
                std::array<bool, geo::direction::count()>& is_direction_empty,
                std::vector<neighbor_gravity_type>& all_neighbor_interaction_data);

            void set_grid_ptr(std::shared_ptr<grid> ptr) {
                grid_ptr = ptr;
            }
        };
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
